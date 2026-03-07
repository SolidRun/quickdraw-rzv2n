/*
 * Quick Draw DRP-AI Server — Renesas RZ/V2N
 *
 * Unix domain socket server for DRP-AI inference.
 * Receives raw grayscale canvas from the Python GUI,
 * preprocesses (crop/pad/invert/resize/normalize) in C++,
 * runs inference on the DRP-AI3 accelerator, returns JSON predictions.
 *
 * Accuracy improvements:
 *   - Temporal smoothing: averages softmax over last N inferences
 *   - Area-based downsampling: matches PIL/training preprocessing
 *
 * Wire protocol (little-endian):
 *   Request:  [uint32 msg_len][uint16 width][uint16 height][uint8 grayscale × w*h]
 *   Response: [uint32 msg_len][utf-8 JSON bytes]
 *
 * Usage:
 *   ./app_quickdraw [--model DIR] [--labels FILE] [--size N] [--socket PATH]
 *                   [--smooth N] [--min-conf F] [--debug-dir DIR]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>

#include <sys/socket.h>
#include <sys/un.h>

#include <string>
#include <vector>
#include <fstream>

#include "define.h"
#include "drpai_inference.h"
#include "classification.h"
#include "preprocessing.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Globals
 * ═══════════════════════════════════════════════════════════════════════ */
static volatile sig_atomic_t g_running = 1;
static std::vector<std::string> g_class_names;
static std::string g_debug_dir;   /* --debug-dir: save preprocessed PGM images */
static int g_debug_counter = 0;

static void signal_handler(int) {
    g_running = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Load class names from labels.txt (one name per line)
 * ═══════════════════════════════════════════════════════════════════════ */
static bool load_labels(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: Cannot open labels file: %s\n", path.c_str());
        return false;
    }
    g_class_names.clear();
    std::string line;
    while (std::getline(f, line)) {
        /* Trim trailing whitespace/CR */
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (!line.empty())
            g_class_names.push_back(line);
    }
    if (g_class_names.empty()) {
        fprintf(stderr, "ERROR: No class names found in %s\n", path.c_str());
        return false;
    }
    return true;
}

static const char* get_class_name(int idx) {
    if (idx >= 0 && idx < (int)g_class_names.size())
        return g_class_names[idx].c_str();
    return "unknown";
}

/* Escape a string for safe embedding in JSON (handles " and \) */
static std::string json_escape(const char* s) {
    std::string out;
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') out += '\\';
        out += *s;
    }
    return out;
}

/* Wire protocol header (little-endian, packed) */
struct RequestHeader {
    uint32_t msg_len;
    uint16_t width;
    uint16_t height;
} __attribute__((packed));

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers: read/write exact N bytes
 * ═══════════════════════════════════════════════════════════════════════ */
static bool recv_exact(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    size_t left = n;
    while (left > 0 && g_running) {
        ssize_t r = read(fd, p, left);
        if (r <= 0) return false;
        p += r;
        left -= (size_t)r;
    }
    return left == 0;
}

static bool send_exact(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    size_t left = n;
    while (left > 0) {
        ssize_t w = write(fd, p, left);
        if (w <= 0) return false;
        p += w;
        left -= (size_t)w;
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Preprocessing config — set from CLI args in main()
 * ═══════════════════════════════════════════════════════════════════════ */
static PreprocessConfig g_preprocess_cfg;

/* ═══════════════════════════════════════════════════════════════════════
 * Temporal smoothing — running average of softmax probabilities
 *
 * Stabilizes predictions during live drawing by averaging the last
 * N inference results. Resets automatically on blank canvas.
 * ═══════════════════════════════════════════════════════════════════════ */
static std::vector<std::vector<float>> g_smooth_history;
static int g_smooth_window = 3;
static float g_min_confidence = 0.0f;

static void smooth_reset() {
    g_smooth_history.clear();
}

static std::vector<float> smooth_apply(const std::vector<float>& probs, int num_classes) {
    g_smooth_history.push_back(probs);
    if ((int)g_smooth_history.size() > g_smooth_window) {
        g_smooth_history.erase(g_smooth_history.begin());
    }

    /* Single sample — no smoothing needed */
    if (g_smooth_history.size() == 1) return probs;

    /* Average across history */
    std::vector<float> avg(num_classes, 0.0f);
    for (auto& p : g_smooth_history) {
        for (int i = 0; i < num_classes && i < (int)p.size(); i++) {
            avg[i] += p[i];
        }
    }
    float inv_n = 1.0f / (float)g_smooth_history.size();
    for (int i = 0; i < num_classes; i++) {
        avg[i] *= inv_n;
    }
    return avg;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Send an error JSON response
 * ═══════════════════════════════════════════════════════════════════════ */
static bool send_error(int client_fd, const char* msg) {
    std::string escaped = json_escape(msg);
    char buf[512];
    snprintf(buf, sizeof(buf), "{\"error\":\"%s\"}", escaped.c_str());
    uint32_t len = (uint32_t)strlen(buf);
    send_exact(client_fd, &len, 4);
    send_exact(client_fd, buf, len);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Handle one request: receive raw canvas, preprocess, infer, send JSON
 * ═══════════════════════════════════════════════════════════════════════ */
static bool handle_request(int client_fd, DRPAIInference& drpai) {
    /* Read request header (8 bytes: msg_len + width + height) */
    RequestHeader hdr;
    if (!recv_exact(client_fd, &hdr, sizeof(hdr))) return false;

    uint16_t width = hdr.width;
    uint16_t height = hdr.height;
    uint32_t num_pixels = (uint32_t)width * (uint32_t)height;
    uint32_t expected = 4 + num_pixels; /* header + uint8 grayscale */
    if (hdr.msg_len != expected) {
        fprintf(stderr, "Protocol error: msg_len=%u expected=%u (w=%u h=%u)\n",
                hdr.msg_len, expected, width, height);
        return false;
    }

    /* Read raw grayscale pixels — reuse buffer across requests */
    static std::vector<uint8_t> canvas;
    canvas.resize(num_pixels);
    if (!recv_exact(client_fd, canvas.data(), num_pixels)) {
        fprintf(stderr, "Failed to read %u canvas bytes\n", num_pixels);
        return false;
    }

    /* Preprocess: crop → pad → invert → resize → normalize */
    static std::vector<float> tensor;
    if (!preprocess_canvas(canvas.data(), width, height, g_preprocess_cfg, tensor)) {
        /* Canvas is blank — no ink, reset smoothing */
        smooth_reset();
        return send_error(client_fd, "blank");
    }

    /* Save debug image if --debug-dir was specified */
    if (!g_debug_dir.empty() && !g_last_debug_image.empty()) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/debug_%04d.pgm",
                 g_debug_dir.c_str(), g_debug_counter++);
        int ms = g_preprocess_cfg.model_size;
        if (save_debug_image(g_last_debug_image.data(), ms, ms, fname)) {
            fprintf(stderr, "  [DEBUG] Saved %s\n", fname);
        }
    }

    /* Run DRP-AI inference */
    if (!drpai.run(tensor.data(), (int)tensor.size())) {
        return send_error(client_fd, "DRP-AI inference failed");
    }

    int out_size = 0;
    float* logits = drpai.get_output(0, out_size);
    if (!logits || out_size == 0) {
        return send_error(client_fd, "No output from model");
    }

    /* Softmax → smooth → top-K */
    int num_classes = (int)g_class_names.size();
    if (out_size < num_classes) num_classes = out_size;

    std::vector<float> probs = softmax(logits, num_classes);
    std::vector<float> smoothed = smooth_apply(probs, num_classes);
    auto results = top_k(smoothed.data(), num_classes, TOP_K);

    /* Build JSON response */
    std::string json;
    json.reserve(512);
    json = "{\"predictions\":[";
    for (size_t i = 0; i < results.size(); i++) {
        if (i > 0) json += ",";
        int cid = results[i].class_id;
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "{\"class\":\"%s\",\"class_id\":%d,\"prob\":%.6f}",
                 json_escape(get_class_name(cid)).c_str(), cid, results[i].confidence);
        json += buf;
    }
    json += "]";

    /* Add smoothing metadata */
    {
        char meta[128];
        snprintf(meta, sizeof(meta), ",\"smooth_n\":%d,\"min_conf\":%.3f",
                 (int)g_smooth_history.size(), g_min_confidence);
        json += meta;
    }
    json += "}";

    /* Send response: [uint32 len][JSON] */
    uint32_t resp_len = (uint32_t)json.size();
    if (!send_exact(client_fd, &resp_len, 4)) return false;
    if (!send_exact(client_fd, json.data(), resp_len)) return false;

    /* Log top-1 */
    if (!results.empty()) {
        int cid = results[0].class_id;
        float conf = results[0].confidence;
        const char* tag = (conf < g_min_confidence && g_min_confidence > 0)
                          ? " [LOW]" : "";
        fprintf(stderr, "[%ux%u] %s (%.1f%% smooth=%d)%s\n",
                width, height, get_class_name(cid), conf * 100.0f,
                (int)g_smooth_history.size(), tag);
    }

    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char* argv[]) {
    std::string model_base = "model";
    std::string model_dir = DEFAULT_MODEL_DIR;
    std::string labels_path = DEFAULT_LABELS;
    std::string socket_path = DEFAULT_SOCKET;
    int input_w = DEFAULT_INPUT_W;
    int input_h = DEFAULT_INPUT_H;
    int input_c = DEFAULT_INPUT_C;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_base = argv[++i];
        else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc)
            model_dir = argv[++i];
        else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc)
            labels_path = argv[++i];
        else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            input_w = input_h = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc)
            socket_path = argv[++i];
        else if (strcmp(argv[i], "--smooth") == 0 && i + 1 < argc) {
            g_smooth_window = atoi(argv[++i]);
            if (g_smooth_window < 1) g_smooth_window = 1;
        }
        else if (strcmp(argv[i], "--min-conf") == 0 && i + 1 < argc) {
            g_min_confidence = (float)atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--debug-dir") == 0 && i + 1 < argc) {
            g_debug_dir = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                "Quick Draw DRP-AI Server — RZ/V2N\n\n"
                "Usage: %s [OPTIONS]\n\n"
                "  --model DIR      Model base directory (default: model)\n"
                "  --model-dir NAME Model subdirectory name (default: %s)\n"
                "  --labels FILE    Class names file, one per line (default: %s)\n"
                "  --size N         Model input size NxN (default: %d)\n"
                "  --socket PATH    Unix socket path (default: %s)\n"
                "  --smooth N       Temporal smoothing window (default: 3)\n"
                "  --min-conf F     Minimum confidence threshold (default: 0)\n"
                "  --debug-dir DIR  Save preprocessed PGM images to DIR\n",
                argv[0], DEFAULT_MODEL_DIR, DEFAULT_LABELS, DEFAULT_INPUT_W, DEFAULT_SOCKET);
            return 0;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* ── 0. Load class labels ── */
    if (!load_labels(labels_path)) {
        return 1;
    }
    int num_classes = (int)g_class_names.size();

    fprintf(stderr, "\n");
    fprintf(stderr, "════════════════════════════════════════════\n");
    fprintf(stderr, "  Quick Draw DRP-AI Server — RZ/V2N\n");
    fprintf(stderr, "════════════════════════════════════════════\n");
    fprintf(stderr, "Model:   %s/%s\n", model_base.c_str(), model_dir.c_str());
    fprintf(stderr, "Labels:  %s (%d classes)\n", labels_path.c_str(), num_classes);
    fprintf(stderr, "Input:   %dx%d (ch=%d)\n", input_w, input_h, input_c);
    fprintf(stderr, "Socket:  %s\n", socket_path.c_str());
    fprintf(stderr, "Smooth:  window=%d\n", g_smooth_window);
    if (g_min_confidence > 0)
        fprintf(stderr, "MinConf: %.1f%%\n", g_min_confidence * 100.0f);
    if (!g_debug_dir.empty())
        fprintf(stderr, "Debug:   %s (PGM image dumps)\n", g_debug_dir.c_str());
    fprintf(stderr, "────────────────────────────────────────────\n\n");

    /* ── 1. Setup preprocessing ── */
    g_preprocess_cfg.model_size = input_w;
    g_preprocess_cfg.ink_threshold = 245;
    g_preprocess_cfg.crop_margin = 12;
    fprintf(stderr, "Preprocess:  %dx%d, ink<%d, margin=%d\n",
            input_w, input_h, g_preprocess_cfg.ink_threshold, g_preprocess_cfg.crop_margin);

    /* ── 2. Load DRP-AI model ── */
    fprintf(stderr, "Loading DRP-AI model...\n");
    DRPAIInference drpai;
    std::string model_path = model_base + "/" + model_dir;

    if (!drpai.load(model_path, DRP_MAX_FREQ, DRPAI_FREQ)) {
        fprintf(stderr, "ERROR: Failed to load model: %s\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "Model loaded.\n");

    /* Warmup inference */
    {
        std::vector<float> dummy(input_c * input_h * input_w, 0.0f);
        drpai.run(dummy.data(), (int)dummy.size());
        int out_size;
        drpai.get_output(0, out_size);
        fprintf(stderr, "DRP-AI warmup complete (output_size=%d, labels=%d).\n", out_size, num_classes);
        if (out_size != num_classes) {
            fprintf(stderr, "WARNING: Model outputs %d values but labels.txt has %d entries!\n",
                    out_size, num_classes);
        }
    }

    /* ── 3. Create Unix domain socket ── */
    unlink(socket_path.c_str());  /* Remove stale socket */

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "ERROR: socket() failed: %s\n", strerror(errno));
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "ERROR: bind(%s) failed: %s\n",
                socket_path.c_str(), strerror(errno));
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 1) < 0) {
        fprintf(stderr, "ERROR: listen() failed: %s\n", strerror(errno));
        close(server_fd);
        unlink(socket_path.c_str());
        return 1;
    }

    fprintf(stderr, "Listening on %s\n", socket_path.c_str());
    fprintf(stderr, "Ready for connections. Press Ctrl+C to stop.\n\n");

    /* ── 4. Accept connections ── */
    while (g_running) {
        /* Use select() with a timeout so we can check g_running */
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(server_fd, &fds);
        struct timeval tv = {1, 0};  /* 1 second timeout */

        int ret = select(server_fd + 1, &fds, NULL, NULL, &tv);
        if (ret <= 0) continue;

        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) {
            if (g_running) fprintf(stderr, "accept() failed: %s\n", strerror(errno));
            continue;
        }

        /* Timeout prevents server hang if client crashes mid-transfer */
        struct timeval client_tv = {5, 0};
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &client_tv, sizeof(client_tv));

        /* Handle requests from this client until disconnect */
        while (g_running) {
            if (!handle_request(client_fd, drpai)) break;
        }

        close(client_fd);
    }

    /* ── Cleanup ── */
    close(server_fd);
    unlink(socket_path.c_str());
    fprintf(stderr, "\nServer stopped.\n");
    return 0;
}
