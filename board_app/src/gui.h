#ifndef GUI_H
#define GUI_H

/*
 * Quick Draw — C++ GTK3 GUI
 *
 * Native cairo-based drawing canvas with prediction panel.
 * Replaces the Python GUI for responsive performance on ARM Cortex-A55.
 *
 * Integrates directly with DRP-AI inference (no socket overhead).
 */

#include <gtk/gtk.h>
#include <cairo.h>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>

#include "classification.h"

/* ── Color (0-1 float RGB) ── */
struct Color {
    double r, g, b;
    Color() : r(0), g(0), b(0) {}
    Color(int ri, int gi, int bi)
        : r(ri / 255.0), g(gi / 255.0), b(bi / 255.0) {}
};

/* ── Layout computed from screen size ── */
struct Layout {
    int title_h;
    int canvas_x, canvas_y, canvas_w, canvas_h;
    int panel_x, panel_w, panel_max_y;
    int btn_bar_y, btn_h;

    struct Button {
        double x, y, w, h;
        std::string label;
        std::string color_key;
    };
    std::vector<Button> buttons;
};

/* ── History entry ── */
struct HistoryEntry {
    std::string name;
    float conf;
    bool correct;
};

/* ── Prediction result for display ── */
struct PredDisplay {
    std::string class_name;
    int class_id;
    float prob;
};

/* ── Commentary engine (simplified C++ port) ── */
class Commentary {
public:
    void init(const std::string& config_path);
    std::string pick(const std::vector<PredDisplay>& preds);
    std::string on_yes(const std::vector<PredDisplay>& preds);
    std::string on_no(const std::vector<PredDisplay>& preds);
    void reset();
    std::vector<std::string> get_trail() const { return guess_trail; }

private:
    std::string pick_from(const std::vector<std::string>& pool,
                          const std::string& name, float prob,
                          const std::string& prev = "");

    /* Comment pools loaded from config.json */
    std::vector<std::string> confident, uncertain, confused;
    std::vector<std::string> close_call, improving, changing;
    std::vector<std::string> long_journey, first_stroke;
    std::vector<std::string> streak_pool, wrong_feedback;
    float confident_threshold = 0.8f;
    float uncertain_threshold = 0.3f;
    bool enabled = false;
    float min_display_secs = 2.5f;

    /* State */
    std::string prev_top;
    float prev_prob = 0.0f;
    int predict_count = 0;
    std::vector<std::string> guess_trail;
    int correct_streak = 0;
    double last_comment_time = 0.0;
    std::deque<std::string> recent;
    int no_repeat_buffer = 8;
};

/* ── Forward declaration ── */
class DRPAIInference;

/* ── Main GUI Application ── */
class QuickDrawGUI {
public:
    QuickDrawGUI();
    ~QuickDrawGUI();

    bool init(const std::string& config_path,
              DRPAIInference* drpai,
              const std::vector<std::string>& class_names,
              int model_size);
    void run();

private:
    /* GTK callbacks (static trampolines) */
    static gboolean on_draw(GtkWidget* w, cairo_t* cr, gpointer data);
    static gboolean on_press(GtkWidget* w, GdkEventButton* e, gpointer data);
    static gboolean on_release(GtkWidget* w, GdkEventButton* e, gpointer data);
    static gboolean on_motion(GtkWidget* w, GdkEventMotion* e, gpointer data);
    static gboolean on_touch(GtkWidget* w, GdkEventTouch* e, gpointer data);
    static gboolean on_key(GtkWidget* w, GdkEventKey* e, gpointer data);
    static gboolean on_tick(gpointer data);
    static void on_destroy(GtkWidget* w, gpointer data);

    /* Shared input handling (used by both mouse and touch) */
    void handle_press(double x, double y);
    void handle_release(double x, double y);
    void handle_motion(double x, double y);

    /* Drawing on canvas surface */
    void draw_dot(double x, double y);
    void draw_line(double x0, double y0, double x1, double y1);
    void clear_canvas();
    void redraw_all_strokes();

    /* Actions */
    void do_clear();
    void do_undo();
    void do_predict();
    void do_exit();
    void do_yes();
    void do_no();

    /* Rendering */
    void compute_layout(int w, int h);
    void render(cairo_t* cr);
    void render_titlebar(cairo_t* cr);
    void render_canvas(cairo_t* cr);
    void render_panel(cairo_t* cr);
    void render_buttons(cairo_t* cr);
    void render_status(cairo_t* cr);

    /* Cairo helpers */
    void rounded_rect(cairo_t* cr, double x, double y, double w, double h, double r = 6);
    void text(cairo_t* cr, const char* txt, double x, double y, double size,
              const Color& col, bool bold = false);
    double text_width(cairo_t* cr, const char* txt, double size);

    /* Grayscale conversion (canvas surface -> uint8 gray) */
    void canvas_to_grayscale(std::vector<uint8_t>& gray, int& w, int& h);

    /* Inference thread entry */
    static void* infer_thread_func(void* arg);

    /* Config */
    double font_scale = 1.2;
    int brush_radius = 8;
    double canvas_ratio = 0.72;
    int canvas_padding = 6;
    int auto_predict_delay_ms = 500;
    int live_predict_interval_ms = 0;
    int fps = 15;
    float confidence_threshold = 0.15f;
    int smooth_window = 3;
    std::string title_text = "QUICK DRAW";
    std::string subtitle_text = "DRP-AI3";
    std::string badge_text = "RZ/V2N SR SOM";

    /* Colors */
    Color col_bg, col_title_bg, col_text, col_text_dim;
    Color col_accent, col_green, col_red, col_yellow;
    Color col_bar_bg, col_bar_fg, col_panel_line;
    Color col_btn_clear, col_btn_undo, col_btn_predict, col_btn_yes, col_btn_no;
    Color col_comment_bg, col_comment_text;
    void load_colors_from_json(const std::string& config_path);

    /* GTK widgets */
    GtkWidget* window = nullptr;
    GtkWidget* darea = nullptr;

    /* Canvas */
    cairo_surface_t* canvas_surface = nullptr;
    cairo_t* canvas_cr = nullptr;
    int canvas_side = 0;

    /* Logo */
    cairo_surface_t* logo_surface = nullptr;

    /* Layout */
    Layout layout;
    int scr_w = 0, scr_h = 0;

    /* Drawing state */
    struct Point { double x, y; };
    std::vector<std::vector<Point>> strokes;
    std::vector<Point> current_stroke;
    bool drawing = false;
    bool dirty = true;

    /* Button press visual feedback */
    int pressed_btn_idx = -1;  /* Index of currently pressed button, -1 = none */

    /* Shared state (mutex-protected) */
    std::mutex mtx;
    std::vector<PredDisplay> predictions;
    bool has_predictions = false;
    std::atomic<bool> predicting{false};
    float infer_ms = 0.0f;
    bool feedback_given = true;
    std::string ai_comment;
    std::vector<std::string> guess_trail_display;
    int score_correct = 0;
    int score_wrong = 0;
    std::deque<HistoryEntry> history;
    int max_history = 8;

    /* Timing */
    double last_stroke_time = 0.0;
    bool timer_active = false;
    double last_live_predict = 0.0;

    /* Shutdown */
    std::atomic<bool> shutting_down{false};

    /* Inference */
    DRPAIInference* drpai = nullptr;
    std::vector<std::string> class_names;
    int model_size = 128;
    Commentary commentary;

    /* Temporal smoothing */
    std::vector<std::vector<float>> smooth_history;
    std::vector<float> smooth_apply(const std::vector<float>& probs, int num_classes);
    void smooth_reset();
};

#endif /* GUI_H */
