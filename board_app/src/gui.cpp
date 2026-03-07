/*
 * Quick Draw — C++ GTK3 GUI Implementation
 *
 * Native cairo rendering for responsive drawing on ARM Cortex-A55.
 * Direct DRP-AI inference — no socket overhead.
 */

#include "gui.h"
#include "drpai_inference.h"
#include "preprocessing.h"
#include "classification.h"
#include "define.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>

static const double TWO_PI = 2.0 * M_PI;

/* Simple monotonic clock (seconds) */
static double mono_time() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

/* ═══════════════════════════════════════════════════════════════════════
 * JSON config parser (minimal — handles our config.json structure)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Tiny helper: find "key": value in a JSON string. Handles strings, numbers, arrays. */
static std::string json_file_contents;

static bool load_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    json_file_contents = ss.str();
    return true;
}

static std::string json_get_string(const std::string& json, const std::string& key, const std::string& def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return def;
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return def;
    return json.substr(pos + 1, end - pos - 1);
}

static double json_get_number(const std::string& json, const std::string& key, double def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    try {
        return std::stod(json.substr(pos, 20));
    } catch (...) {
        return def;
    }
}

static bool json_get_bool(const std::string& json, const std::string& key, bool def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    auto rest = json.substr(pos + 1, 10);
    if (rest.find("true") != std::string::npos) return true;
    if (rest.find("false") != std::string::npos) return false;
    return def;
}

static Color json_get_color(const std::string& json, const std::string& key, Color def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return def;
    auto end = json.find(']', pos);
    if (end == std::string::npos) return def;
    std::string arr = json.substr(pos + 1, end - pos - 1);
    int r = 0, g = 0, b = 0;
    if (sscanf(arr.c_str(), "%d,%d,%d", &r, &g, &b) == 3 ||
        sscanf(arr.c_str(), " %d , %d , %d", &r, &g, &b) == 3) {
        return Color(r, g, b);
    }
    return def;
}

/* Extract a JSON array of strings for a given key within a section */
static std::vector<std::string> json_get_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;

    /* Find matching ] — handle nested brackets */
    int depth = 0;
    size_t end = pos;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '[') depth++;
        else if (json[i] == ']') { depth--; if (depth == 0) { end = i; break; } }
    }

    std::string arr = json.substr(pos + 1, end - pos - 1);

    /* Extract quoted strings */
    size_t p = 0;
    while (p < arr.size()) {
        auto q1 = arr.find('"', p);
        if (q1 == std::string::npos) break;
        auto q2 = arr.find('"', q1 + 1);
        while (q2 != std::string::npos && arr[q2 - 1] == '\\') {
            q2 = arr.find('"', q2 + 1);
        }
        if (q2 == std::string::npos) break;
        result.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
        p = q2 + 1;
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Commentary
 * ═══════════════════════════════════════════════════════════════════════ */

static std::mt19937 g_rng(std::random_device{}());

void Commentary::init(const std::string& config_path) {
    if (!load_json_file(config_path)) return;
    auto& json = json_file_contents;

    /* Find comments section */
    auto cpos = json.find("\"comments\"");
    if (cpos == std::string::npos) return;
    std::string cjson = json.substr(cpos);

    enabled = json_get_bool(cjson, "enabled", false);
    confident_threshold = (float)json_get_number(cjson, "confident_threshold", 0.8);
    uncertain_threshold = (float)json_get_number(cjson, "uncertain_threshold", 0.3);
    no_repeat_buffer = (int)json_get_number(cjson, "no_repeat_buffer", 8);
    min_display_secs = (float)json_get_number(cjson, "min_display_secs", 2.5);

    confident = json_get_string_array(cjson, "confident");
    uncertain = json_get_string_array(cjson, "uncertain");
    confused = json_get_string_array(cjson, "confused");
    close_call = json_get_string_array(cjson, "close_call");
    improving = json_get_string_array(cjson, "improving");
    changing = json_get_string_array(cjson, "changing");
    long_journey = json_get_string_array(cjson, "long_journey");
    first_stroke = json_get_string_array(cjson, "first_stroke");
    streak_pool = json_get_string_array(cjson, "streak");
    wrong_feedback = json_get_string_array(cjson, "wrong_feedback");
}

std::string Commentary::pick_from(const std::vector<std::string>& pool,
                                   const std::string& name, float prob,
                                   const std::string& prev) {
    if (pool.empty()) return "";

    /* Filter out recently used */
    std::vector<const std::string*> available;
    for (auto& t : pool) {
        bool used = false;
        for (auto& r : recent) if (r == t) { used = true; break; }
        if (!used) available.push_back(&t);
    }
    if (available.empty()) {
        for (auto& t : pool) available.push_back(&t);
    }

    std::uniform_int_distribution<int> dist(0, (int)available.size() - 1);
    std::string tmpl = *available[dist(g_rng)];

    /* Add to recent */
    recent.push_back(tmpl);
    while ((int)recent.size() > no_repeat_buffer) recent.pop_front();

    /* Template substitution */
    auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };

    char prob_str[16];
    snprintf(prob_str, sizeof(prob_str), "%.1f", prob * 100.0f);

    std::string trail_str;
    for (size_t i = 0; i < guess_trail.size(); i++) {
        if (i > 0) trail_str += ", ";
        trail_str += guess_trail[i];
    }

    replace_all(tmpl, "{class}", name);
    replace_all(tmpl, "{prob}", prob_str);
    replace_all(tmpl, "{prev}", prev);
    replace_all(tmpl, "{runner}", prev);
    replace_all(tmpl, "{streak}", std::to_string(correct_streak));
    replace_all(tmpl, "{trail}", trail_str);
    replace_all(tmpl, "{first_guess}", guess_trail.empty() ? name : guess_trail[0]);
    replace_all(tmpl, "{num_guesses}", std::to_string((int)guess_trail.size()));

    /* {journey} — last 3 wrong guesses */
    std::string journey;
    for (auto& g : guess_trail) {
        if (g != name) {
            if (!journey.empty()) journey += ", ";
            journey += g;
        }
    }
    replace_all(tmpl, "{journey}", journey);

    return tmpl;
}

std::string Commentary::pick(const std::vector<PredDisplay>& preds) {
    if (!enabled || preds.empty()) return "";

    auto& top = preds[0];
    std::string name = top.class_name;
    float prob = top.prob;
    std::string runner = preds.size() >= 2 ? preds[1].class_name : "";
    float runner_prob = preds.size() >= 2 ? preds[1].prob : 0.0f;

    predict_count++;

    if (guess_trail.empty() || guess_trail.back() != name) {
        guess_trail.push_back(name);
    }

    double now = mono_time();
    bool too_soon = (now - last_comment_time) < min_display_secs;

    if (too_soon && predict_count > 1) {
        prev_prob = prob;
        prev_top = name;
        return "";
    }

    std::string result;

    if (predict_count == 1 && !first_stroke.empty()) {
        result = pick_from(first_stroke, name, prob);
    } else if ((int)guess_trail.size() >= 4 && !long_journey.empty()
               && prob >= confident_threshold) {
        /* Only show "finally see it" when AI is actually confident now */
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        if (d(g_rng) < 0.6f) result = pick_from(long_journey, name, prob);
    }

    if (result.empty() && !runner.empty() && (prob - runner_prob) < 0.10f && prob < 0.7f && !close_call.empty()) {
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        if (d(g_rng) < 0.7f) result = pick_from(close_call, name, prob, runner);
    }

    if (result.empty() && prev_top == name && prob > prev_prob + 0.05f && prob > 0.4f && predict_count > 2 && !improving.empty()) {
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        if (d(g_rng) < 0.5f) result = pick_from(improving, name, prob);
    }

    if (result.empty() && name != prev_top && !prev_top.empty() && predict_count > 1 && !changing.empty()) {
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        if (d(g_rng) < 0.6f) result = pick_from(changing, name, prob, prev_top);
    }

    if (result.empty()) {
        if (prob >= confident_threshold)
            result = pick_from(confident, name, prob);
        else if (prob >= uncertain_threshold)
            result = pick_from(uncertain, name, prob);
        else
            result = pick_from(confused, name, prob);
    }

    prev_prob = prob;
    prev_top = name;
    if (!result.empty()) {
        last_comment_time = now;
    }
    return result;
}

std::string Commentary::on_yes(const std::vector<PredDisplay>& preds) {
    correct_streak++;
    if (!streak_pool.empty() && correct_streak >= 3 && !preds.empty()) {
        return pick_from(streak_pool, preds[0].class_name, preds[0].prob);
    }
    return "";
}

std::string Commentary::on_no(const std::vector<PredDisplay>& preds) {
    correct_streak = 0;
    if (!wrong_feedback.empty() && !preds.empty()) {
        return pick_from(wrong_feedback, preds[0].class_name, preds[0].prob);
    }
    return "";
}

void Commentary::reset() {
    prev_top.clear();
    prev_prob = 0.0f;
    predict_count = 0;
    guess_trail.clear();
}

/* ═══════════════════════════════════════════════════════════════════════
 * QuickDrawGUI
 * ═══════════════════════════════════════════════════════════════════════ */

QuickDrawGUI::QuickDrawGUI() {}

QuickDrawGUI::~QuickDrawGUI() {
    if (canvas_cr) cairo_destroy(canvas_cr);
    if (canvas_surface) cairo_surface_destroy(canvas_surface);
    if (logo_surface) cairo_surface_destroy(logo_surface);
}

void QuickDrawGUI::load_colors_from_json(const std::string& config_path) {
    if (!load_json_file(config_path)) return;
    auto& j = json_file_contents;
    auto cp = j.find("\"colors\"");
    if (cp == std::string::npos) return;
    std::string cj = j.substr(cp);

    col_bg         = json_get_color(cj, "background",   Color(46, 30, 30));
    col_title_bg   = json_get_color(cj, "title_bg",     Color(78, 45, 45));
    col_text       = json_get_color(cj, "text",         Color(230, 220, 220));
    col_text_dim   = json_get_color(cj, "text_dim",     Color(160, 140, 140));
    col_accent     = json_get_color(cj, "accent",       Color(60, 200, 220));
    col_green      = json_get_color(cj, "green",        Color(100, 220, 100));
    col_red        = json_get_color(cj, "red",          Color(230, 80, 80));
    col_yellow     = json_get_color(cj, "yellow",       Color(240, 220, 60));
    col_bar_bg     = json_get_color(cj, "bar_bg",       Color(70, 50, 50));
    col_bar_fg     = json_get_color(cj, "bar_fg",       Color(60, 140, 180));
    col_panel_line = json_get_color(cj, "panel_line",   Color(120, 100, 100));
    col_btn_clear  = json_get_color(cj, "btn_clear",    Color(90, 60, 60));
    col_btn_undo   = json_get_color(cj, "btn_undo",     Color(90, 60, 60));
    col_btn_predict= json_get_color(cj, "btn_predict",  Color(40, 80, 120));
    col_btn_yes    = json_get_color(cj, "btn_yes",      Color(60, 120, 60));
    col_btn_no     = json_get_color(cj, "btn_no",       Color(140, 60, 60));
    col_comment_bg = json_get_color(cj, "comment_bg",   Color(60, 45, 50));
    col_comment_text = json_get_color(cj, "comment_text", Color(200, 180, 220));
}

bool QuickDrawGUI::init(const std::string& config_path,
                         DRPAIInference* drpai_ptr,
                         const std::vector<std::string>& names,
                         int msize)
{
    drpai = drpai_ptr;
    class_names = names;
    model_size = msize;

    /* Load config */
    if (load_json_file(config_path)) {
        auto& j = json_file_contents;

        /* Find ui section */
        auto upos = j.find("\"ui\"");
        std::string uj = (upos != std::string::npos) ? j.substr(upos) : j;

        font_scale = json_get_number(uj, "font_scale", 1.2);
        brush_radius = (int)json_get_number(uj, "brush_radius", 8);
        canvas_ratio = json_get_number(uj, "canvas_ratio", 0.72);
        canvas_padding = (int)json_get_number(uj, "canvas_padding", 6);
        auto_predict_delay_ms = (int)json_get_number(uj, "auto_predict_delay_ms", 500);
        live_predict_interval_ms = (int)json_get_number(uj, "live_predict_interval_ms", 0);
        fps = (int)json_get_number(uj, "fps", 15);
        max_history = (int)json_get_number(uj, "max_history", 8);
        title_text = json_get_string(uj, "title", "QUICK DRAW");
        subtitle_text = json_get_string(uj, "subtitle", "DRP-AI3");
        badge_text = json_get_string(uj, "badge", "RZ/V2N SR SOM");

        /* Model config */
        auto mpos = j.find("\"model\"");
        std::string mj = (mpos != std::string::npos) ? j.substr(mpos) : j;
        smooth_window = (int)json_get_number(mj, "smooth_window", 3);
        confidence_threshold = (float)json_get_number(mj, "confidence_threshold", 0.15);
    }

    load_colors_from_json(config_path);
    commentary.init(config_path);

    /* Load logo */
    std::string logo_path = config_path;
    auto slash = logo_path.rfind('/');
    if (slash != std::string::npos) logo_path = logo_path.substr(0, slash + 1);
    else logo_path = "./";
    logo_path += "solidrun_logo.png";
    logo_surface = cairo_image_surface_create_from_png(logo_path.c_str());
    if (cairo_surface_status(logo_surface) != CAIRO_STATUS_SUCCESS) {
        cairo_surface_destroy(logo_surface);
        logo_surface = nullptr;
    }

    /* Create GTK window */
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    std::string win_title = title_text + " - " + subtitle_text;
    gtk_window_set_title(GTK_WINDOW(window), win_title.c_str());
    g_signal_connect(window, "destroy", G_CALLBACK(on_destroy), this);
    g_signal_connect(window, "key-press-event", G_CALLBACK(on_key), this);

    darea = gtk_drawing_area_new();
    gtk_widget_add_events(darea,
        GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK |
        GDK_POINTER_MOTION_MASK | GDK_TOUCH_MASK);
    g_signal_connect(darea, "draw", G_CALLBACK(on_draw), this);
    g_signal_connect(darea, "button-press-event", G_CALLBACK(on_press), this);
    g_signal_connect(darea, "button-release-event", G_CALLBACK(on_release), this);
    g_signal_connect(darea, "motion-notify-event", G_CALLBACK(on_motion), this);
    g_signal_connect(darea, "touch-event", G_CALLBACK(on_touch), this);
    gtk_container_add(GTK_CONTAINER(window), darea);

    /* Fullscreen */
    gtk_window_fullscreen(GTK_WINDOW(window));
    gtk_window_set_keep_above(GTK_WINDOW(window), TRUE);
    gtk_window_set_decorated(GTK_WINDOW(window), FALSE);

    /* Tick timer */
    g_timeout_add(1000 / fps, on_tick, this);

    return true;
}

void QuickDrawGUI::run() {
    gtk_widget_show_all(window);
    gtk_main();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layout
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::compute_layout(int w, int h) {
    scr_w = w;
    scr_h = h;

    layout.title_h = 44;
    layout.btn_h = 34;
    layout.btn_bar_y = h - layout.btn_h - 2;

    int avail_h = layout.btn_bar_y - layout.title_h - 2;
    int avail_w = (int)(w * canvas_ratio) - canvas_padding * 2;
    int side = std::min(avail_w, avail_h);

    layout.canvas_x = canvas_padding;
    layout.canvas_y = layout.title_h + (avail_h - side) / 2 + 2;
    layout.canvas_w = side;
    layout.canvas_h = side;

    layout.panel_x = layout.canvas_x + side + 20;
    layout.panel_w = w - layout.panel_x - 20;
    layout.panel_max_y = layout.btn_bar_y - 10;

    /* Buttons — sized for touchscreen */
    layout.buttons.clear();
    double bx = 20.0;
    double by = layout.btn_bar_y;
    int bw = std::max(100, w / 10);
    int gap = std::max(10, w / 80);

    struct BtnDef { const char* label; const char* color; };
    BtnDef btns[] = {
        {"CLEAR", "btn_clear"}, {"UNDO", "btn_undo"},
        {"EXIT", "btn_predict"}, {"YES", "btn_yes"}, {"NO", "btn_no"}
    };
    for (auto& bd : btns) {
        int bw_cur = bw;
        std::string lbl = bd.label;
        if (lbl == "EXIT") bw_cur = bw + 20;
        else if (lbl == "YES" || lbl == "NO") bw_cur = std::max(90, bw - 10);

        Layout::Button btn;
        btn.x = bx; btn.y = by;
        btn.w = bw_cur; btn.h = layout.btn_h;
        btn.label = bd.label;
        btn.color_key = bd.color;
        layout.buttons.push_back(btn);
        bx += bw_cur + gap;
    }

    /* Recreate canvas surface if size changed */
    if (side != canvas_side) {
        if (canvas_cr) cairo_destroy(canvas_cr);
        if (canvas_surface) cairo_surface_destroy(canvas_surface);
        canvas_surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, side, side);
        canvas_cr = cairo_create(canvas_surface);
        canvas_side = side;
        clear_canvas();
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Canvas operations
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::clear_canvas() {
    cairo_set_source_rgb(canvas_cr, 1, 1, 1);
    cairo_paint(canvas_cr);
}

void QuickDrawGUI::draw_dot(double x, double y) {
    cairo_set_source_rgb(canvas_cr, 0, 0, 0);
    cairo_arc(canvas_cr, x, y, brush_radius, 0, TWO_PI);
    cairo_fill(canvas_cr);
}

void QuickDrawGUI::draw_line(double x0, double y0, double x1, double y1) {
    cairo_set_source_rgb(canvas_cr, 0, 0, 0);
    cairo_set_line_width(canvas_cr, brush_radius * 2);
    cairo_set_line_cap(canvas_cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(canvas_cr, CAIRO_LINE_JOIN_ROUND);
    cairo_move_to(canvas_cr, x0, y0);
    cairo_line_to(canvas_cr, x1, y1);
    cairo_stroke(canvas_cr);
}

void QuickDrawGUI::redraw_all_strokes() {
    clear_canvas();
    cairo_set_source_rgb(canvas_cr, 0, 0, 0);
    cairo_set_line_width(canvas_cr, brush_radius * 2);
    cairo_set_line_cap(canvas_cr, CAIRO_LINE_CAP_ROUND);
    cairo_set_line_join(canvas_cr, CAIRO_LINE_JOIN_ROUND);
    for (auto& stroke : strokes) {
        if (stroke.size() >= 2) {
            cairo_move_to(canvas_cr, stroke[0].x, stroke[0].y);
            for (size_t i = 1; i < stroke.size(); i++)
                cairo_line_to(canvas_cr, stroke[i].x, stroke[i].y);
            cairo_stroke(canvas_cr);
        } else if (stroke.size() == 1) {
            cairo_arc(canvas_cr, stroke[0].x, stroke[0].y, brush_radius, 0, TWO_PI);
            cairo_fill(canvas_cr);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Grayscale conversion
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::canvas_to_grayscale(std::vector<uint8_t>& gray, int& w, int& h) {
    cairo_surface_flush(canvas_surface);
    w = cairo_image_surface_get_width(canvas_surface);
    h = cairo_image_surface_get_height(canvas_surface);
    int stride = cairo_image_surface_get_stride(canvas_surface);
    unsigned char* data = cairo_image_surface_get_data(canvas_surface);

    int num_px = w * h;
    gray.resize(num_px);

    for (int y = 0; y < h; y++) {
        unsigned char* row = data + y * stride;
        for (int x = 0; x < w; x++) {
            /* BGRA layout */
            int b = row[x * 4 + 0];
            int g = row[x * 4 + 1];
            int r = row[x * 4 + 2];
            gray[y * w + x] = (uint8_t)((b + g + r) / 3);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Temporal smoothing
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::smooth_reset() {
    smooth_history.clear();
}

std::vector<float> QuickDrawGUI::smooth_apply(const std::vector<float>& probs, int num_classes) {
    smooth_history.push_back(probs);
    if ((int)smooth_history.size() > smooth_window)
        smooth_history.erase(smooth_history.begin());

    if (smooth_history.size() == 1) return probs;

    std::vector<float> avg(num_classes, 0.0f);
    for (auto& p : smooth_history)
        for (int i = 0; i < num_classes && i < (int)p.size(); i++)
            avg[i] += p[i];

    float inv_n = 1.0f / (float)smooth_history.size();
    for (int i = 0; i < num_classes; i++) avg[i] *= inv_n;
    return avg;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Actions
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::do_clear() {
    strokes.clear();
    current_stroke.clear();
    drawing = false;
    clear_canvas();
    {
        std::lock_guard<std::mutex> lock(mtx);
        has_predictions = false;
        predictions.clear();
        feedback_given = true;
        ai_comment.clear();
        guess_trail_display.clear();
    }
    timer_active = false;
    commentary.reset();
    smooth_reset();
    dirty = true;
    /* Force immediate redraw so CLEAR feels instant */
    gtk_widget_queue_draw(darea);
    while (gtk_events_pending()) gtk_main_iteration_do(FALSE);
}

void QuickDrawGUI::do_undo() {
    if (!strokes.empty()) {
        strokes.pop_back();
        redraw_all_strokes();
        last_stroke_time = mono_time();
        timer_active = true;
        dirty = true;
        gtk_widget_queue_draw(darea);
    }
}

void QuickDrawGUI::do_predict() {
    if (predicting.load() || shutting_down.load()) return;
    predicting.store(true);

    timer_active = false;
    dirty = true;
    gtk_widget_queue_draw(darea);

    /* Snapshot canvas to grayscale (fast in C++) */
    auto* self = this;
    std::thread([self]() {
        std::vector<uint8_t> gray;
        int w, h;
        self->canvas_to_grayscale(gray, w, h);

        /* Preprocess */
        PreprocessConfig cfg;
        cfg.model_size = self->model_size;
        cfg.ink_threshold = 245;
        cfg.crop_margin = 12;

        std::vector<float> tensor;
        bool has_ink = preprocess_canvas(gray.data(), w, h, cfg, tensor);

        if (!has_ink) {
            std::lock_guard<std::mutex> lock(self->mtx);
            self->predictions.clear();
            self->has_predictions = false;
            self->predicting.store(false);
            self->smooth_reset();
            self->dirty = true;
            if (!self->shutting_down.load()) {
                g_idle_add([](gpointer d) -> gboolean {
                    auto* s = (QuickDrawGUI*)d;
                    if (!s->shutting_down.load()) gtk_widget_queue_draw(s->darea);
                    return FALSE;
                }, self);
            }
            return;
        }

        /* DRP-AI inference */
        auto t0 = std::chrono::steady_clock::now();
        bool ok = self->drpai->run(tensor.data(), (int)tensor.size());
        auto t1 = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (!ok) {
            std::lock_guard<std::mutex> lock(self->mtx);
            self->predictions.clear();
            self->has_predictions = false;
            self->predicting.store(false);
            self->dirty = true;
            return;
        }

        int out_size = 0;
        float* logits = self->drpai->get_output(0, out_size);
        int num_classes = (int)self->class_names.size();
        if (out_size < num_classes) num_classes = out_size;

        auto probs = softmax(logits, num_classes);
        auto smoothed = self->smooth_apply(probs, num_classes);
        auto results = top_k(smoothed.data(), num_classes, TOP_K);

        /* Build predictions */
        std::vector<PredDisplay> preds;
        for (auto& r : results) {
            PredDisplay pd;
            pd.class_id = r.class_id;
            pd.prob = r.confidence;
            pd.class_name = (r.class_id >= 0 && r.class_id < (int)self->class_names.size())
                            ? self->class_names[r.class_id] : "unknown";
            preds.push_back(pd);
        }

        {
            std::lock_guard<std::mutex> lock(self->mtx);
            self->predictions = preds;
            self->has_predictions = !preds.empty();
            self->infer_ms = elapsed;
            self->feedback_given = false;

            std::string comment = self->commentary.pick(preds);
            if (!comment.empty()) self->ai_comment = comment;
            self->guess_trail_display = self->commentary.get_trail();
            self->predicting.store(false);
        }

        self->dirty = true;
        if (!self->shutting_down.load()) {
            g_idle_add([](gpointer d) -> gboolean {
                auto* s = (QuickDrawGUI*)d;
                if (!s->shutting_down.load()) gtk_widget_queue_draw(s->darea);
                return FALSE;
            }, self);
        }

        /* Log */
        if (!preds.empty()) {
            fprintf(stderr, "[%dx%d] %s (%.1f%% %.1fms)\n",
                    w, h, preds[0].class_name.c_str(),
                    preds[0].prob * 100.0f, elapsed);
        }
    }).detach();
}

void QuickDrawGUI::do_exit() {
    shutting_down.store(true);
    /* Wait briefly for any running inference to finish */
    for (int i = 0; i < 50 && predicting.load(); i++) {
        g_usleep(20000); /* 20ms, up to 1 second total */
        while (gtk_events_pending()) gtk_main_iteration_do(FALSE);
    }
    gtk_main_quit();
}

void QuickDrawGUI::do_yes() {
    std::lock_guard<std::mutex> lock(mtx);
    if (feedback_given || !has_predictions) return;
    feedback_given = true;
    score_correct++;
    if (!predictions.empty()) {
        HistoryEntry he;
        he.name = predictions[0].class_name;
        he.conf = predictions[0].prob;
        he.correct = true;
        history.push_back(he);
        while ((int)history.size() > max_history) history.pop_front();

        auto comment = commentary.on_yes(predictions);
        if (!comment.empty()) ai_comment = comment;
    }
    dirty = true;
    gtk_widget_queue_draw(darea);
}

void QuickDrawGUI::do_no() {
    std::lock_guard<std::mutex> lock(mtx);
    if (feedback_given || !has_predictions) return;
    feedback_given = true;
    score_wrong++;
    if (!predictions.empty()) {
        HistoryEntry he;
        he.name = predictions[0].class_name;
        he.conf = predictions[0].prob;
        he.correct = false;
        history.push_back(he);
        while ((int)history.size() > max_history) history.pop_front();

        auto comment = commentary.on_no(predictions);
        if (!comment.empty()) ai_comment = comment;
    }
    dirty = true;
    gtk_widget_queue_draw(darea);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cairo helpers
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::rounded_rect(cairo_t* cr, double x, double y, double w, double h, double r) {
    cairo_new_sub_path(cr);
    cairo_arc(cr, x + w - r, y + r, r, -M_PI_2, 0);
    cairo_arc(cr, x + w - r, y + h - r, r, 0, M_PI_2);
    cairo_arc(cr, x + r, y + h - r, r, M_PI_2, M_PI);
    cairo_arc(cr, x + r, y + r, r, M_PI, M_PI + M_PI_2);
    cairo_close_path(cr);
}

void QuickDrawGUI::text(cairo_t* cr, const char* txt, double x, double y, double size,
                         const Color& col, bool bold) {
    cairo_set_source_rgb(cr, col.r, col.g, col.b);
    cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL,
                            bold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    double scaled = size * font_scale;
    cairo_set_font_size(cr, scaled);
    cairo_move_to(cr, x, y + scaled);
    cairo_show_text(cr, txt);
}

double QuickDrawGUI::text_width(cairo_t* cr, const char* txt, double size) {
    cairo_set_font_size(cr, size * font_scale);
    cairo_text_extents_t ext;
    cairo_text_extents(cr, txt, &ext);
    return ext.width;
}

/* Section header with divider line */
static double section_header(QuickDrawGUI* gui, cairo_t* cr, const char* txt,
                              double x, double y, double pw, double font_scale,
                              const Color& dim, const Color& line) {
    cairo_set_source_rgb(cr, dim.r, dim.g, dim.b);
    cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    double scaled = 12 * font_scale;
    cairo_set_font_size(cr, scaled);
    cairo_move_to(cr, x, y + scaled);
    cairo_show_text(cr, txt);

    double line_y = y + 18 * font_scale;
    cairo_set_source_rgb(cr, line.r, line.g, line.b);
    cairo_move_to(cr, x, line_y);
    cairo_line_to(cr, x + pw - 20, line_y);
    cairo_set_line_width(cr, 1);
    cairo_stroke(cr);
    return line_y + 10;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Rendering
 * ═══════════════════════════════════════════════════════════════════════ */

void QuickDrawGUI::render_titlebar(cairo_t* cr) {
    cairo_set_source_rgb(cr, col_title_bg.r, col_title_bg.g, col_title_bg.b);
    cairo_rectangle(cr, 0, 0, scr_w, layout.title_h);
    cairo_fill(cr);

    text(cr, title_text.c_str(), 10, 8, 18, col_accent, true);
    text(cr, subtitle_text.c_str(), 180, 12, 12, col_text_dim);

    double right_edge = scr_w - 20;
    if (logo_surface) {
        int lw = cairo_image_surface_get_width(logo_surface);
        int lh = cairo_image_surface_get_height(logo_surface);
        double target_h = (double)layout.title_h - 8;
        double scale = target_h / lh;
        double sw = lw * scale;
        double lx = right_edge - sw;
        double ly = (layout.title_h - target_h) / 2.0;
        cairo_save(cr);
        cairo_translate(cr, lx, ly);
        cairo_scale(cr, scale, scale);
        cairo_set_source_surface(cr, logo_surface, 0, 0);
        cairo_paint(cr);
        cairo_restore(cr);
        right_edge = lx - 12;
    }

    double bw = text_width(cr, badge_text.c_str(), 11);
    text(cr, badge_text.c_str(), right_edge - bw, 12, 11, col_text_dim);
}

void QuickDrawGUI::render_canvas(cairo_t* cr) {
    auto& L = layout;
    cairo_set_source_rgb(cr, col_panel_line.r, col_panel_line.g, col_panel_line.b);
    cairo_rectangle(cr, L.canvas_x - 2, L.canvas_y - 2, L.canvas_w + 4, L.canvas_h + 4);
    cairo_set_line_width(cr, 2);
    cairo_stroke(cr);

    cairo_set_source_surface(cr, canvas_surface, L.canvas_x, L.canvas_y);
    cairo_paint(cr);
}

void QuickDrawGUI::render_panel(cairo_t* cr) {
    auto& L = layout;
    double px = L.panel_x;
    double pw = L.panel_w;
    double max_y = L.panel_max_y;
    double py = L.canvas_y + 5;

    /* Snapshot shared state */
    std::vector<PredDisplay> preds;
    bool has_preds, is_predicting;
    float inf_ms;
    std::string comment;
    std::vector<std::string> trail;
    int sc, sw;
    std::deque<HistoryEntry> hist;
    {
        std::lock_guard<std::mutex> lock(mtx);
        preds = predictions;
        has_preds = has_predictions;
        is_predicting = predicting.load();
        inf_ms = infer_ms;
        comment = ai_comment;
        trail = guess_trail_display;
        sc = score_correct;
        sw = score_wrong;
        hist = history;
    }

    /* PREDICTION section */
    py = section_header(this, cr, "PREDICTION", px, py, pw, font_scale, col_text_dim, col_panel_line);

    if (is_predicting) {
        text(cr, "Analyzing...", px, py, 18, col_yellow);
        py += 30;
    } else if (has_preds && !preds.empty()) {
        auto& top = preds[0];
        if (top.prob < confidence_threshold) {
            text(cr, "Uncertain...", px, py, 28, col_yellow, true);
            char buf[128];
            snprintf(buf, sizeof(buf), "Best: %s (%.1f%%)", top.class_name.c_str(), top.prob * 100);
            text(cr, buf, px, py + 42, 16, col_text_dim);
            py += 70;
        } else {
            text(cr, top.class_name.c_str(), px, py, 28, col_accent, true);
            char pbuf[32];
            snprintf(pbuf, sizeof(pbuf), "%.1f%%", top.prob * 100);
            text(cr, pbuf, px, py + 38, 18, col_green, true);
            if (inf_ms > 0) {
                char tbuf[64];
                snprintf(tbuf, sizeof(tbuf), "Inference: %.1f ms", inf_ms);
                text(cr, tbuf, px + 100, py + 41, 12, col_text_dim);
            }
            py += 70;
        }

        /* Comment bubble */
        if (!comment.empty()) {
            double bubble_w = pw - 20;
            double pad = 8;
            cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
            double csz = 14 * font_scale;
            cairo_set_font_size(cr, csz);

            /* Simple word wrap */
            std::vector<std::string> lines;
            std::istringstream iss(comment);
            std::string word, cur;
            while (iss >> word) {
                std::string test = cur.empty() ? word : cur + " " + word;
                cairo_text_extents_t ext;
                cairo_text_extents(cr, test.c_str(), &ext);
                if (ext.width > bubble_w - 2 * pad && !cur.empty()) {
                    lines.push_back(cur);
                    cur = word;
                } else {
                    cur = test;
                }
            }
            if (!cur.empty()) lines.push_back(cur);

            double line_h = csz * 1.35;
            double bubble_h = lines.size() * line_h + 2 * pad;
            rounded_rect(cr, px, py, bubble_w, bubble_h, 6);
            cairo_set_source_rgb(cr, col_comment_bg.r, col_comment_bg.g, col_comment_bg.b);
            cairo_fill(cr);

            for (size_t i = 0; i < lines.size(); i++) {
                text(cr, lines[i].c_str(), px + pad, py + pad + i * line_h, 14, col_comment_text);
            }
            py += bubble_h + 8;
        }
    } else {
        text(cr, "Draw something!", px, py, 18, col_text_dim);
        py += 30;
    }
    py += 10;

    /* AI THINKING trail — smart truncation to fit panel width */
    if (trail.size() >= 2 && py < max_y) {
        py = section_header(this, cr, "AI THINKING", px, py, pw, font_scale, col_text_dim, col_panel_line);

        /* Build trail items as individual segments for smart layout */
        double trail_font = 12 * font_scale;
        cairo_set_font_size(cr, trail_font);

        /* Measure separator width */
        cairo_text_extents_t sep_ext;
        cairo_text_extents(cr, " > ", &sep_ext);
        double sep_w = sep_ext.x_advance;

        /* Measure "... > " prefix width */
        cairo_text_extents_t ellip_ext;
        cairo_text_extents(cr, "... > ", &ellip_ext);
        double ellip_w = ellip_ext.x_advance;

        double avail_w = pw - 20;

        /* Calculate how many items fit from the end */
        std::vector<double> item_widths(trail.size());
        double total_w = 0;
        for (size_t i = 0; i < trail.size(); i++) {
            cairo_text_extents_t ext;
            cairo_text_extents(cr, trail[i].c_str(), &ext);
            item_widths[i] = ext.x_advance;
        }

        /* Find the starting index that fits within available width */
        int start_idx = 0;
        double w_needed = item_widths.back(); /* Last item always shown */
        for (int i = (int)trail.size() - 2; i >= 0; i--) {
            double extra = item_widths[i] + sep_w;
            double prefix = (i > 0) ? ellip_w : 0;
            if (w_needed + extra + prefix > avail_w) {
                start_idx = i + 1;
                break;
            }
            w_needed += extra;
        }

        /* Render trail with color coding */
        double tx = px;
        bool need_ellipsis = (start_idx > 0);
        if (need_ellipsis) {
            text(cr, "...", tx, py, 12, col_text_dim);
            cairo_text_extents_t ext;
            cairo_text_extents(cr, "... ", &ext);
            tx += ext.x_advance;
        }

        for (size_t i = start_idx; i < trail.size(); i++) {
            if (i > (size_t)start_idx || need_ellipsis) {
                text(cr, ">", tx, py, 12, col_panel_line);
                cairo_text_extents_t ext;
                cairo_text_extents(cr, "> ", &ext);
                tx += ext.x_advance;
            }
            /* Last item = accent color, others = dim */
            bool is_last = (i == trail.size() - 1);
            const Color& tc = is_last ? col_accent : col_text_dim;
            text(cr, trail[i].c_str(), tx, py, 12, tc, is_last);
            cairo_text_extents_t ext;
            cairo_set_font_size(cr, trail_font);
            cairo_text_extents(cr, trail[i].c_str(), &ext);
            tx += ext.x_advance + 4;
        }
        py += 22;
        py += 8;
    }

    /* TOP 5 bars */
    if (has_preds && !preds.empty() && py < max_y) {
        py = section_header(this, cr, "TOP 5", px, py, pw, font_scale, col_text_dim, col_panel_line);
        double bar_max_w = pw - 20;
        double bar_h = 24;
        for (size_t i = 0; i < preds.size() && i < 5; i++) {
            if (py + bar_h > max_y) break;
            double p = preds[i].prob;

            rounded_rect(cr, px, py, bar_max_w, bar_h, 3);
            cairo_set_source_rgb(cr, col_bar_bg.r, col_bar_bg.g, col_bar_bg.b);
            cairo_fill(cr);

            double fill_w = std::max(2.0, bar_max_w * p);
            rounded_rect(cr, px, py, fill_w, bar_h, 3);
            auto& fc = (i == 0) ? col_accent : col_bar_fg;
            cairo_set_source_rgb(cr, fc.r, fc.g, fc.b);
            cairo_fill(cr);

            char lbl[128];
            snprintf(lbl, sizeof(lbl), "%d. %s  %.1f%%", (int)i + 1, preds[i].class_name.c_str(), p * 100);
            auto& tc = (p > 0.15f) ? col_text : col_text_dim;
            text(cr, lbl, px + 6, py + 4, 12, tc);
            py += bar_h + 6;
        }
        py += 10;
    }

    /* SCORE */
    if (py < max_y) {
        py = section_header(this, cr, "SCORE", px, py, pw, font_scale, col_text_dim, col_panel_line);
        char sbuf[64];
        snprintf(sbuf, sizeof(sbuf), "Correct: %d", sc);
        text(cr, sbuf, px, py, 16, col_green);
        snprintf(sbuf, sizeof(sbuf), "Wrong: %d", sw);
        text(cr, sbuf, px + 180, py, 16, col_red);
        py += 30;
    }

    /* HISTORY */
    if (!hist.empty() && py < max_y) {
        py = section_header(this, cr, "HISTORY", px, py, pw, font_scale, col_text_dim, col_panel_line);
        for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
            if (py >= max_y) break;
            char hbuf[128];
            snprintf(hbuf, sizeof(hbuf), "[%s] %s (%.0f%%)",
                     it->correct ? "Y" : "N", it->name.c_str(), it->conf * 100);
            text(cr, hbuf, px, py, 12, it->correct ? col_green : col_red);
            py += 20;
        }
    }
}

void QuickDrawGUI::render_buttons(cairo_t* cr) {
    for (int i = 0; i < (int)layout.buttons.size(); i++) {
        auto& btn = layout.buttons[i];
        rounded_rect(cr, btn.x, btn.y, btn.w, btn.h, 6);

        /* Get color by key */
        Color bc = col_btn_clear;
        if (btn.color_key == "btn_undo") bc = col_btn_undo;
        else if (btn.color_key == "btn_predict") bc = col_btn_predict;
        else if (btn.color_key == "btn_yes") bc = col_btn_yes;
        else if (btn.color_key == "btn_no") bc = col_btn_no;

        /* Press feedback: brighten when pressed */
        if (i == pressed_btn_idx) {
            bc.r = std::min(1.0, bc.r * 1.5 + 0.1);
            bc.g = std::min(1.0, bc.g * 1.5 + 0.1);
            bc.b = std::min(1.0, bc.b * 1.5 + 0.1);
        }

        cairo_set_source_rgb(cr, bc.r, bc.g, bc.b);
        cairo_fill_preserve(cr);
        cairo_set_source_rgb(cr, col_panel_line.r, col_panel_line.g, col_panel_line.b);
        cairo_set_line_width(cr, 1);
        cairo_stroke(cr);

        cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 16 * font_scale);
        cairo_text_extents_t ext;
        cairo_text_extents(cr, btn.label.c_str(), &ext);
        double tx = btn.x + (btn.w - ext.width) / 2;
        double ty = btn.y + (btn.h + ext.height) / 2;
        cairo_set_source_rgb(cr, col_text.r, col_text.g, col_text.b);
        cairo_move_to(cr, tx, ty);
        cairo_show_text(cr, btn.label.c_str());
    }
}

void QuickDrawGUI::render_status(cairo_t* cr) {
    const char* status;
    if (predicting.load()) status = "Analyzing...";
    else if (timer_active && !strokes.empty()) status = "Auto-predict pending...";
    else status = "Ready";

    char sbuf[64];
    snprintf(sbuf, sizeof(sbuf), "Status: %s", status);

    cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 12 * font_scale);
    cairo_text_extents_t ext;
    cairo_text_extents(cr, sbuf, &ext);

    double btn_mid = layout.btn_bar_y + layout.btn_h / 2.0;
    text(cr, sbuf, scr_w - ext.width - 30, btn_mid - ext.height / 2, 12, col_text_dim);

    /* Server indicator dot (always green — direct inference) */
    cairo_set_source_rgb(cr, col_green.r, col_green.g, col_green.b);
    cairo_arc(cr, scr_w - 12, btn_mid + 2, 5, 0, TWO_PI);
    cairo_fill(cr);
}

void QuickDrawGUI::render(cairo_t* cr) {
    cairo_set_source_rgb(cr, col_bg.r, col_bg.g, col_bg.b);
    cairo_paint(cr);
    render_titlebar(cr);
    render_canvas(cr);
    render_panel(cr);
    render_buttons(cr);
    render_status(cr);
    dirty = false;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GTK Callbacks
 * ═══════════════════════════════════════════════════════════════════════ */

gboolean QuickDrawGUI::on_draw(GtkWidget* w, cairo_t* cr, gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    GtkAllocation alloc;
    gtk_widget_get_allocation(w, &alloc);
    if (alloc.width != self->scr_w || alloc.height != self->scr_h) {
        self->compute_layout(alloc.width, alloc.height);
    }
    self->render(cr);
    return FALSE;
}

/* ── Shared input handlers (used by both mouse and touch) ── */

void QuickDrawGUI::handle_press(double x, double y) {
    auto& L = layout;

    /* Hit-test buttons with expanded touch area (12px padding) */
    const double touch_pad = 12.0;
    for (int i = 0; i < (int)L.buttons.size(); i++) {
        auto& btn = L.buttons[i];
        if (x >= btn.x - touch_pad && x <= btn.x + btn.w + touch_pad &&
            y >= btn.y - touch_pad && y <= btn.y + btn.h + touch_pad) {
            /* Show press feedback immediately */
            pressed_btn_idx = i;
            dirty = true;
            gtk_widget_queue_draw(darea);
            while (gtk_events_pending()) gtk_main_iteration_do(FALSE);

            /* Execute action */
            if (btn.label == "CLEAR") do_clear();
            else if (btn.label == "UNDO") do_undo();
            else if (btn.label == "EXIT") do_exit();
            else if (btn.label == "YES") do_yes();
            else if (btn.label == "NO") do_no();

            /* Clear press feedback */
            pressed_btn_idx = -1;
            dirty = true;
            gtk_widget_queue_draw(darea);
            return;
        }
    }

    /* Canvas hit */
    if (x >= L.canvas_x && x < L.canvas_x + L.canvas_w &&
        y >= L.canvas_y && y < L.canvas_y + L.canvas_h) {
        drawing = true;
        double cx = x - L.canvas_x;
        double cy = y - L.canvas_y;
        current_stroke.clear();
        current_stroke.push_back({cx, cy});
        draw_dot(cx, cy);
        dirty = true;
        gtk_widget_queue_draw(darea);
    }
}

void QuickDrawGUI::handle_release(double x, double y) {
    (void)x; (void)y;

    /* Clear any button press feedback */
    if (pressed_btn_idx >= 0) {
        pressed_btn_idx = -1;
        dirty = true;
        gtk_widget_queue_draw(darea);
    }

    if (!drawing) return;
    drawing = false;
    if (!current_stroke.empty()) {
        strokes.push_back(current_stroke);
        current_stroke.clear();
        last_stroke_time = mono_time();
        timer_active = true;
    }
    dirty = true;
    gtk_widget_queue_draw(darea);
}

void QuickDrawGUI::handle_motion(double x, double y) {
    if (!drawing) return;

    auto& L = layout;
    if (x >= L.canvas_x && x < L.canvas_x + L.canvas_w &&
        y >= L.canvas_y && y < L.canvas_y + L.canvas_h) {
        double cx = x - L.canvas_x;
        double cy = y - L.canvas_y;
        if (!current_stroke.empty()) {
            auto& last = current_stroke.back();
            draw_line(last.x, last.y, cx, cy);
        }
        current_stroke.push_back({cx, cy});
        dirty = true;
        /* Let tick handle screen redraw — don't call queue_draw every motion */
    }
}

/* ── Mouse event callbacks (delegate to shared handlers) ── */

gboolean QuickDrawGUI::on_press(GtkWidget*, GdkEventButton* e, gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    if (e->button != 1) return FALSE;
    self->handle_press(e->x, e->y);
    return TRUE;
}

gboolean QuickDrawGUI::on_release(GtkWidget*, GdkEventButton* e, gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    if (e->button != 1) return FALSE;
    self->handle_release(e->x, e->y);
    return TRUE;
}

gboolean QuickDrawGUI::on_motion(GtkWidget*, GdkEventMotion* e, gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    self->handle_motion(e->x, e->y);
    return TRUE;
}

/* ── Touch event callback (Wayland touchscreen support) ── */

gboolean QuickDrawGUI::on_touch(GtkWidget*, GdkEventTouch* e, gpointer data) {
    auto* self = (QuickDrawGUI*)data;

    switch (e->type) {
        case GDK_TOUCH_BEGIN:
            self->handle_press(e->x, e->y);
            break;
        case GDK_TOUCH_END:
        case GDK_TOUCH_CANCEL:
            self->handle_release(e->x, e->y);
            break;
        case GDK_TOUCH_UPDATE:
            self->handle_motion(e->x, e->y);
            break;
        default:
            return FALSE;
    }
    return TRUE;
}

gboolean QuickDrawGUI::on_key(GtkWidget*, GdkEventKey* e, gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    switch (e->keyval) {
        case GDK_KEY_Escape: case GDK_KEY_q: self->do_exit(); break;
        case GDK_KEY_c: self->do_clear(); break;
        case GDK_KEY_z: self->do_undo(); break;
        case GDK_KEY_Return: case GDK_KEY_space: self->do_predict(); break;
        case GDK_KEY_y: self->do_yes(); break;
        case GDK_KEY_n: self->do_no(); break;
    }
    return TRUE;
}

gboolean QuickDrawGUI::on_tick(gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    if (self->shutting_down.load()) return FALSE; /* Stop timer */
    double now = mono_time();

    bool is_pred = self->predicting.load();

    /* Live prediction while drawing */
    if (self->live_predict_interval_ms > 0 && self->drawing && !is_pred) {
        bool has_ink = !self->strokes.empty() || self->current_stroke.size() > 3;
        if (has_ink && now - self->last_live_predict >= self->live_predict_interval_ms / 1000.0) {
            self->last_live_predict = now;
            self->do_predict();
        }
    }

    /* Auto-predict after pen release */
    double delay = self->auto_predict_delay_ms / 1000.0;
    if (self->timer_active && !is_pred &&
        self->last_stroke_time > 0 && now - self->last_stroke_time > delay) {
        self->do_predict();
    }

    /* Redraw if dirty */
    if (self->dirty) {
        gtk_widget_queue_draw(self->darea);
    }

    return TRUE;
}

void QuickDrawGUI::on_destroy(GtkWidget*, gpointer data) {
    auto* self = (QuickDrawGUI*)data;
    self->shutting_down.store(true);
    gtk_main_quit();
}
