/*
 * Quick Draw — Combined C++ GTK3 GUI + DRP-AI Inference
 *
 * Single binary: loads DRP-AI model, then opens GTK3 fullscreen canvas.
 * No socket overhead — inference runs directly in background thread.
 *
 * All runtime parameters loaded from config.ini (DRP-AI settings)
 * and config.json (UI/comments). No rebuild needed to change settings.
 *
 * Usage:
 *   ./app_quickdraw [--config config.ini]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>

#include <gtk/gtk.h>
#include <glib-unix.h>

#include "define.h"
#include "drpai_inference.h"
#include "gui.h"

/* Ctrl+C / SIGTERM handler — calls gtk_main_quit() safely from GLib event loop */
static gboolean on_unix_signal(gpointer) {
    fprintf(stderr, "\nCaught signal — shutting down...\n");
    gtk_main_quit();
    return G_SOURCE_REMOVE;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INI file parser
 * ═══════════════════════════════════════════════════════════════════════ */

struct IniConfig {
    std::map<std::string, std::map<std::string, std::string>> sections;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line, section = "general";
        while (std::getline(f, line)) {
            /* Trim */
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            while (!line.empty() && line.front() == ' ') line.erase(line.begin());
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;
            if (line[0] == '[') {
                auto end = line.find(']');
                if (end != std::string::npos)
                    section = line.substr(1, end - 1);
                continue;
            }
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            /* Trim key/val */
            while (!key.empty() && key.back() == ' ') key.pop_back();
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());
            sections[section][key] = val;
        }
        return true;
    }

    std::string get(const std::string& sec, const std::string& key, const std::string& def = "") const {
        auto sit = sections.find(sec);
        if (sit == sections.end()) return def;
        auto kit = sit->second.find(key);
        return kit != sit->second.end() ? kit->second : def;
    }

    int get_int(const std::string& sec, const std::string& key, int def) const {
        auto v = get(sec, key, "");
        return v.empty() ? def : std::atoi(v.c_str());
    }

    float get_float(const std::string& sec, const std::string& key, float def) const {
        auto v = get(sec, key, "");
        return v.empty() ? def : (float)std::atof(v.c_str());
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * Load class names from labels.txt
 * ═══════════════════════════════════════════════════════════════════════ */
static std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: Cannot open labels: %s\n", path.c_str());
        return names;
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    return names;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char* argv[]) {
    gtk_init(&argc, &argv);

    /* Ctrl+C and kill — clean shutdown via GLib event loop */
    g_unix_signal_add(SIGINT, on_unix_signal, nullptr);
    g_unix_signal_add(SIGTERM, on_unix_signal, nullptr);

    /* Find config.ini path */
    std::string ini_path = "config.ini";
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0) && i + 1 < argc) {
            ini_path = argv[++i];
        }
    }

    /* Load config.ini */
    IniConfig ini;
    if (!ini.load(ini_path)) {
        fprintf(stderr, "WARNING: Cannot load %s — using defaults\n", ini_path.c_str());
    }

    /* Read parameters */
    std::string model_base = ini.get("model", "model_dir", "model");
    std::string model_name = ini.get("model", "model_name", DEFAULT_MODEL_DIR);
    std::string labels_path = ini.get("model", "labels", DEFAULT_LABELS);
    int input_size = ini.get_int("model", "input_size", DEFAULT_INPUT_W);
    int drp_freq = ini.get_int("drpai", "drp_max_freq", DRP_MAX_FREQ);
    int drpai_freq = ini.get_int("drpai", "drpai_freq", DRPAI_FREQ);
    std::string json_config = ini.get("ui", "config_json", "config.json");

    fprintf(stderr, "\n");
    fprintf(stderr, "================================================\n");
    fprintf(stderr, "  Quick Draw — C++ GTK3 + DRP-AI3 (RZ/V2N)\n");
    fprintf(stderr, "================================================\n");
    fprintf(stderr, "Config:  %s\n", ini_path.c_str());
    fprintf(stderr, "Model:   %s/%s\n", model_base.c_str(), model_name.c_str());
    fprintf(stderr, "Labels:  %s\n", labels_path.c_str());
    fprintf(stderr, "Input:   %dx%d\n", input_size, input_size);
    fprintf(stderr, "DRP:     freq=%d  AI-MAC=%d\n", drp_freq, drpai_freq);
    fprintf(stderr, "UI:      %s\n", json_config.c_str());
    fprintf(stderr, "------------------------------------------------\n\n");

    /* Load labels */
    auto class_names = load_labels(labels_path);
    if (class_names.empty()) {
        fprintf(stderr, "ERROR: No class names loaded\n");
        return 1;
    }
    fprintf(stderr, "Labels: %d classes\n", (int)class_names.size());

    /* Load DRP-AI model */
    fprintf(stderr, "Loading DRP-AI model...\n");
    DRPAIInference drpai;
    std::string model_path = model_base + "/" + model_name;

    if (!drpai.load(model_path, drp_freq, drpai_freq)) {
        fprintf(stderr, "ERROR: Failed to load model: %s\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "Model loaded.\n");

    /* Warmup */
    {
        int ch = DEFAULT_INPUT_C;
        std::vector<float> dummy(ch * input_size * input_size, 0.0f);
        drpai.run(dummy.data(), (int)dummy.size());
        int out_size;
        drpai.get_output(0, out_size);
        fprintf(stderr, "Warmup complete (output=%d, labels=%d)\n", out_size, (int)class_names.size());
    }

    /* Start GUI */
    QuickDrawGUI gui;
    if (!gui.init(json_config, &drpai, class_names, input_size)) {
        fprintf(stderr, "ERROR: GUI init failed\n");
        return 1;
    }

    fprintf(stderr, "GUI ready — fullscreen\n\n");
    gui.run();

    fprintf(stderr, "\nExiting.\n");
    return 0;
}
