#!/usr/bin/env python3
"""
Quick Draw — Python GTK3 GUI for RZ/V2N DRP-AI

Drawing canvas with mouse/touch input. Connects to the C++ DRP-AI
inference server (app_quickdraw) via Unix socket.

All parameters loaded from config.json — edit to customize without rebuilding.

Usage:
    python3 quickdraw_gui.py [--config config.json]
"""

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

import cairo
import json
import math
import os
import sys
import struct
import socket
import random
import time
import threading
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PI = 2.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════
# Configuration helpers
# ═══════════════════════════════════════════════════════════════════════

def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: Could not load config: {path} — {e}", file=sys.stderr)
        return {}


def load_labels(path):
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        print(f"ERROR: No class names in {path}", file=sys.stderr)
        sys.exit(1)
    return names


def resolve_colors(cfg):
    """Pre-resolve all colors from config into (r, g, b) float tuples.

    Called once at init — avoids repeated dict lookups during rendering.
    """
    colors_cfg = cfg.get("colors", {})
    resolved = {}
    for name, default in [
        ("background",   (46, 30, 30)),
        ("title_bg",     (78, 45, 45)),
        ("text",         (230, 220, 220)),
        ("text_dim",     (160, 140, 140)),
        ("accent",       (60, 200, 220)),
        ("green",        (100, 220, 100)),
        ("red",          (230, 80, 80)),
        ("yellow",       (240, 220, 60)),
        ("bar_bg",       (70, 50, 50)),
        ("bar_fg",       (60, 140, 180)),
        ("panel_line",   (120, 100, 100)),
        ("btn_clear",    (90, 60, 60)),
        ("btn_undo",     (90, 60, 60)),
        ("btn_predict",  (40, 80, 120)),
        ("btn_yes",      (60, 120, 60)),
        ("btn_no",       (140, 60, 60)),
        ("comment_bg",   (60, 45, 50)),
        ("comment_text", (200, 180, 220)),
    ]:
        c = colors_cfg.get(name, default)
        if isinstance(c, list) and len(c) >= 3:
            resolved[name] = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        else:
            resolved[name] = (default[0] / 255.0, default[1] / 255.0, default[2] / 255.0)
    return resolved


# ═══════════════════════════════════════════════════════════════════════
# Persistent socket client
# ═══════════════════════════════════════════════════════════════════════

class InferenceClient:
    """Persistent Unix socket connection to the C++ DRP-AI server.

    Keeps one connection open across inference requests.
    Reconnects automatically on error. Thread-safe.
    """

    def __init__(self, socket_path):
        self.socket_path = socket_path
        self._sock = None
        self._lock = threading.Lock()

    def _connect(self):
        self._disconnect()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(self.socket_path)
        self._sock = s

    def _disconnect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _recv_exact(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        pos = 0
        while pos < n:
            nbytes = self._sock.recv_into(view[pos:], n - pos)
            if nbytes == 0:
                raise ConnectionError("server closed connection")
            pos += nbytes
        return bytes(buf)

    def infer(self, gray_bytes, width, height):
        """Send grayscale canvas, return parsed JSON response.

        Uses persistent connection; reconnects once on failure.
        """
        with self._lock:
            for attempt in range(2):
                try:
                    if self._sock is None:
                        self._connect()

                    num_px = width * height
                    msg_len = 4 + num_px
                    header = struct.pack("<IHH", msg_len, width, height)
                    self._sock.sendall(header)
                    self._sock.sendall(gray_bytes)

                    resp_len_data = self._recv_exact(4)
                    resp_len = struct.unpack("<I", resp_len_data)[0]
                    resp_data = self._recv_exact(resp_len)

                    return json.loads(resp_data.decode("utf-8"))

                except Exception as e:
                    self._disconnect()
                    if attempt == 0:
                        continue
                    return {"error": str(e)}

    def check_connected(self):
        with self._lock:
            if self._sock is not None:
                return True
            try:
                self._connect()
                return True
            except Exception:
                return False

    def close(self):
        with self._lock:
            self._disconnect()


# ═══════════════════════════════════════════════════════════════════════
# Fast grayscale conversion
# ═══════════════════════════════════════════════════════════════════════

def _grayscale_from_raw(data, w, h, stride):
    """Convert raw BGRA pixel data to grayscale bytes.

    Uses stride-4 slicing to extract B/G/R channels — runs at near C speed
    in CPython. Can be called from background thread (no cairo dependency).
    """
    if stride == w * 4:
        raw = data[:w * h * 4]
    else:
        raw = b"".join(data[y * stride:y * stride + w * 4] for y in range(h))

    num_px = w * h
    # BGRA layout: B=offset 0, G=offset 1, R=offset 2
    b_ch = raw[0::4]
    g_ch = raw[1::4]
    r_ch = raw[2::4]

    gray = bytes([(b_ch[i] + g_ch[i] + r_ch[i]) // 3 for i in range(num_px)])
    return gray, w, h


def canvas_to_grayscale(surface):
    """Convert ARGB32 cairo surface to grayscale bytes."""
    surface.flush()
    return _grayscale_from_raw(
        bytes(surface.get_data()),
        surface.get_width(),
        surface.get_height(),
        surface.get_stride(),
    )


# ═══════════════════════════════════════════════════════════════════════
# AI Commentary
# ═══════════════════════════════════════════════════════════════════════

class Commentary:
    """Context-aware comment generator inspired by Google Quick Draw and Artbitrator.

    Key design principles (from researching similar games):
    1. The AI's "thinking journey" IS the entertainment — show what it tried
    2. Comments should react to what the AI SEES, not just confidence numbers
    3. Wrong guesses are funnier than right ones — lean into misinterpretations
    4. Track confirmed history separately from live predictions

    Tracks:
    - guess_trail: ordered list of unique guesses within current drawing
    - confirmed history: drawings the user confirmed via YES/NO feedback
    """

    def __init__(self, cfg):
        self.cfg = cfg.get("comments", {})
        self.recent = deque(maxlen=self.cfg.get("no_repeat_buffer", 8))

        # Confirmed history — updated only on YES/NO feedback
        self.confirmed_counts = {}   # class_name -> times confirmed (YES or NO)
        self.confirmed_history = []  # list of {"name": str, "correct": bool}
        self.correct_streak = 0

        # Live state — within the current drawing
        self.prev_top = ""
        self.prev_prob = 0.0
        self.draw_predict_count = 0  # predictions within current drawing
        self.guess_trail = []        # ordered unique guesses this drawing
        self._last_comment_time = 0.0
        self._last_comment_prob = 0.0  # prob when comment was generated
        self._min_display_secs = cfg.get("comments", {}).get("min_display_secs", 2.5)

    def _format_trail(self):
        """Format guess trail for display: 'banana, sword, key'"""
        return ", ".join(self.guess_trail)

    def _pick(self, pool, name, prob, prev=""):
        if not pool:
            return ""
        available = [t for t in pool if t not in self.recent]
        if not available:
            available = pool
        tmpl = random.choice(available)
        self.recent.append(tmpl)
        count = self.confirmed_counts.get(name, 0)
        # Build the journey string: "first a banana, then a sword"
        journey = ""
        if len(self.guess_trail) >= 2:
            others = [g for g in self.guess_trail if g != name]
            if others:
                journey = ", ".join(others[-3:])  # last 3 wrong guesses
        first_guess = self.guess_trail[0] if self.guess_trail else name
        num_guesses = len(self.guess_trail)
        return (tmpl
                .replace("{class}", name)
                .replace("{prob}", f"{prob * 100:.1f}")
                .replace("{prev}", prev)
                .replace("{runner}", prev)
                .replace("{count}", str(count))
                .replace("{streak}", str(self.correct_streak))
                .replace("{journey}", journey)
                .replace("{first_guess}", first_guess)
                .replace("{num_guesses}", str(num_guesses))
                .replace("{trail}", self._format_trail()))

    def pick(self, predictions):
        if not self.cfg.get("enabled", False) or not predictions:
            return ""

        top = predictions[0]
        name, prob = top["class"], top["prob"]
        runner = predictions[1]["class"] if len(predictions) >= 2 else ""
        runner_prob = predictions[1]["prob"] if len(predictions) >= 2 else 0

        self.draw_predict_count += 1

        # Track guess trail — only add if different from last guess
        if not self.guess_trail or self.guess_trail[-1] != name:
            self.guess_trail.append(name)

        now = time.monotonic()

        # Don't replace the current comment until it's been visible long enough
        too_soon = (now - self._last_comment_time) < self._min_display_secs

        if too_soon and self.draw_predict_count > 1:
            # Update the percentage in the existing comment to stay in sync
            old_pct = f"{self._last_comment_prob * 100:.1f}%"
            new_pct = f"{prob * 100:.1f}%"
            self.prev_prob = prob
            self.prev_top = name
            if old_pct != new_pct:
                self._last_comment_prob = prob
                return f"__update_prob__{new_pct}__{old_pct}__"
            return ""

        result = self._pick_category(name, prob, runner, runner_prob)
        self.prev_prob = prob
        self.prev_top = name
        if result:
            self._last_comment_time = now
            self._last_comment_prob = prob
        return result

    def _pick_category(self, name, prob, runner, runner_prob):
        """Select the best comment category based on current state."""
        # First prediction of this drawing
        if self.draw_predict_count == 1:
            pool = self.cfg.get("first_stroke", [])
            if pool:
                return self._pick(pool, name, prob)

        # Long journey — AI went through many guesses before settling
        if len(self.guess_trail) >= 4:
            pool = self.cfg.get("long_journey", [])
            if pool and random.random() < 0.5:
                return self._pick(pool, name, prob)

        # Close call between top-2
        if runner and prob > 0 and (prob - runner_prob) < 0.10 and prob < 0.7:
            pool = self.cfg.get("close_call", [])
            if pool and random.random() < 0.7:
                return self._pick(pool, name, prob, runner)

        # Improving — same class but confidence is actually going up
        if (self.prev_top == name and prob > self.prev_prob + 0.05
                and prob > 0.4 and self.draw_predict_count > 2):
            pool = self.cfg.get("improving", [])
            if pool and random.random() < 0.5:
                return self._pick(pool, name, prob)

        # Repeat class — only if user CONFIRMED this class before
        confirmed_n = self.confirmed_counts.get(name, 0)
        if confirmed_n >= 2:
            pool = self.cfg.get("repeat_class", [])
            if pool and random.random() < 0.4:
                return self._pick(pool, name, prob)

        # Changing — prediction shifted within this drawing
        if name != self.prev_top and self.prev_top and self.draw_predict_count > 1:
            pool = self.cfg.get("changing", [])
            if pool and random.random() < 0.6:
                return self._pick(pool, name, prob, self.prev_top)

        # Per-class special comments
        per_class = self.cfg.get("per_class", {}).get(name, [])
        if per_class and random.random() < 0.5:
            return self._pick(per_class, name, prob, self.prev_top)

        # Confidence-based fallback
        conf_th = self.cfg.get("confident_threshold", 0.8)
        unc_th = self.cfg.get("uncertain_threshold", 0.3)
        if prob >= conf_th:
            category = "confident"
        elif prob >= unc_th:
            category = "uncertain"
        else:
            category = "confused"

        return self._pick(self.cfg.get(category, []), name, prob, self.prev_top)

    def get_guess_trail(self):
        """Return copy of guess trail for display in the panel."""
        return list(self.guess_trail)

    def on_feedback_yes(self, predictions):
        self.correct_streak += 1
        if predictions:
            top = predictions[0]
            name = top["class"]
            self.confirmed_counts[name] = self.confirmed_counts.get(name, 0) + 1
            self.confirmed_history.append({"name": name, "correct": True})
            if self.correct_streak == 10:
                pool = self.cfg.get("milestone_10", [])
            elif self.correct_streak == 5:
                pool = self.cfg.get("milestone_5", [])
            elif self.correct_streak >= 3:
                pool = self.cfg.get("streak", [])
            else:
                pool = []
            if pool:
                return self._pick(pool, top["class"], top["prob"])
        return ""

    def on_feedback_no(self, predictions):
        self.correct_streak = 0
        if predictions:
            top = predictions[0]
            name = top["class"]
            self.confirmed_counts[name] = self.confirmed_counts.get(name, 0) + 1
            self.confirmed_history.append({"name": name, "correct": False})
            pool = self.cfg.get("wrong_feedback", [])
            if pool:
                return self._pick(pool, top["class"], top["prob"])
        return ""

    def reset(self):
        """Reset per-drawing state (called on CLEAR). Confirmed history persists."""
        self.prev_top = ""
        self.prev_prob = 0.0
        self.draw_predict_count = 0
        self.guess_trail = []
        self._last_comment_prob = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════

class QuickDrawApp:

    # ── Init ──

    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.ui = self.cfg.get("ui", {})
        self.model_cfg = self.cfg.get("model", {})
        self.col = resolve_colors(self.cfg)

        # Labels
        labels_path = self.model_cfg.get("labels", "labels.txt")
        if not os.path.isabs(labels_path):
            labels_path = str(SCRIPT_DIR / labels_path)
        self.class_names = load_labels(labels_path)

        # Persistent socket client
        socket_path = self.model_cfg.get("socket", "/tmp/quickdraw.sock")
        self.client = InferenceClient(socket_path)

        # Canvas state (main thread only)
        self.strokes = []
        self.current_stroke = []
        self.drawing = False
        self.brush_radius = self.ui.get("brush_radius", 8)
        self.font_scale = self.ui.get("font_scale", 1.0)
        self.canvas_surface = None
        self._canvas_cr = None

        # Shared state (protected by _lock, accessed from inference thread)
        self._lock = threading.Lock()
        self._predictions = []
        self._has_predictions = False
        self._predicting = False
        self._server_connected = False
        self._infer_ms = 0.0
        self._feedback_given = True
        self._ai_comment = ""
        self._guess_trail = []
        self._score_correct = 0
        self._score_wrong = 0
        self._history = deque(maxlen=self.ui.get("max_history", 8))

        # Auto-predict timing
        self._last_stroke_time = 0.0
        self._timer_active = False
        self._last_live_predict = 0.0

        # Commentary engine
        self._commentary = Commentary(self.cfg)

        # Dirty flag — skip redraw when nothing changed
        self._dirty = True

        # Logo
        self._logo = None
        logo_path = str(SCRIPT_DIR / "solidrun_logo.png")
        if os.path.exists(logo_path):
            try:
                self._logo = cairo.ImageSurface.create_from_png(logo_path)
            except Exception:
                pass

        # Layout (computed on first draw)
        self._scr_w = 0
        self._scr_h = 0
        self._layout = {}

        self._build_window()

    def _build_window(self):
        title = self.ui.get("title", "QUICK DRAW")
        subtitle = self.ui.get("subtitle", "DRP-AI3")
        self._window = Gtk.Window(title=f"{title} — {subtitle}")
        self._window.connect("destroy", self._on_destroy)
        self._window.connect("key-press-event", self._on_key)

        self._darea = Gtk.DrawingArea()
        self._darea.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.TOUCH_MASK
        )
        self._darea.connect("draw", self._on_draw)
        self._darea.connect("button-press-event", self._on_press)
        self._darea.connect("button-release-event", self._on_release)
        self._darea.connect("motion-notify-event", self._on_motion)
        self._window.add(self._darea)

        win_w = self.ui.get("window_width", 0)
        win_h = self.ui.get("window_height", 0)
        if win_w > 0 and win_h > 0:
            self._window.set_default_size(win_w, win_h)
            self._window.set_decorated(True)
        else:
            self._window.fullscreen()
            self._window.set_keep_above(True)
            self._window.set_decorated(False)

        fps = self.ui.get("fps", 30)
        GLib.timeout_add(1000 // fps, self._on_tick)

        self._server_connected = self.client.check_connected()

    def _on_destroy(self, _widget):
        self.client.close()
        Gtk.main_quit()

    def run(self):
        self._window.show_all()
        Gtk.main()

    # ── Layout ──

    def _compute_layout(self, w, h):
        self._scr_w = w
        self._scr_h = h
        L = {}

        canvas_pad = self.ui.get("canvas_padding", 10)
        L["title_h"] = self.ui.get("title_height", max(48, h // 14))
        L["btn_h"] = self.ui.get("button_height", max(36, h // 18))
        L["btn_bar_y"] = h - L["btn_h"] - canvas_pad

        avail_h = L["btn_bar_y"] - L["title_h"] - canvas_pad * 2
        canvas_ratio = self.ui.get("canvas_ratio", 0.65)
        avail_w = int(w * canvas_ratio) - canvas_pad * 2
        side = min(avail_w, avail_h)
        L["canvas_x"] = canvas_pad
        L["canvas_y"] = L["title_h"] + (avail_h - side) // 2 + canvas_pad
        L["canvas_w"] = side
        L["canvas_h"] = side

        L["panel_x"] = L["canvas_x"] + side + 20
        L["panel_w"] = w - L["panel_x"] - 20
        L["panel_max_y"] = L["btn_bar_y"] - 10

        # Buttons
        bx = 30.0
        by = L["btn_bar_y"]
        bw = max(90, w // 12)
        gap = max(8, w // 100)
        L["buttons"] = []
        for label, color_key in [
            ("CLEAR", "btn_clear"), ("UNDO", "btn_undo"),
            ("EXIT", "btn_predict"), ("YES", "btn_yes"), ("NO", "btn_no"),
        ]:
            bw_cur = bw + 20 if label == "EXIT" else (
                max(80, bw - 10) if label in ("YES", "NO") else bw)
            L["buttons"].append({
                "x": bx, "y": by, "w": bw_cur, "h": L["btn_h"],
                "label": label, "color": color_key,
            })
            bx += bw_cur + gap

        self._layout = L

        # Recreate canvas surface if size changed
        if (self.canvas_surface is None or
                self.canvas_surface.get_width() != side or
                self.canvas_surface.get_height() != side):
            self.canvas_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
            self._canvas_cr = cairo.Context(self.canvas_surface)
            self._clear_canvas()

    # ── Canvas operations ──

    def _clear_canvas(self):
        cr = self._canvas_cr
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

    def _draw_dot(self, x, y):
        cr = self._canvas_cr
        cr.set_source_rgb(0, 0, 0)
        cr.arc(x, y, self.brush_radius, 0, TWO_PI)
        cr.fill()

    def _draw_line(self, x0, y0, x1, y1):
        cr = self._canvas_cr
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(self.brush_radius * 2)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.move_to(x0, y0)
        cr.line_to(x1, y1)
        cr.stroke()

    def _redraw_all_strokes(self):
        self._clear_canvas()
        cr = self._canvas_cr
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(self.brush_radius * 2)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        for stroke in self.strokes:
            if len(stroke) >= 2:
                cr.move_to(*stroke[0])
                for pt in stroke[1:]:
                    cr.line_to(*pt)
                cr.stroke()
            elif len(stroke) == 1:
                cr.arc(stroke[0][0], stroke[0][1], self.brush_radius, 0, TWO_PI)
                cr.fill()

    # ── Actions ──

    def _do_clear(self):
        self.strokes.clear()
        self.current_stroke.clear()
        self.drawing = False
        self._clear_canvas()
        with self._lock:
            self._has_predictions = False
            self._predictions.clear()
            self._feedback_given = True
            self._ai_comment = ""
            self._guess_trail = []
        self._timer_active = False
        self._commentary.reset()
        self._dirty = True
        self._darea.queue_draw()

    def _do_undo(self):
        if self.strokes:
            self.strokes.pop()
            self._redraw_all_strokes()
            self._last_stroke_time = time.monotonic()
            self._timer_active = True
            self._dirty = True
            self._darea.queue_draw()

    def _do_predict(self):
        with self._lock:
            if self._predicting:
                return
            if not self._server_connected:
                self._server_connected = self.client.check_connected()
                if not self._server_connected:
                    return
            self._predicting = True

        self._timer_active = False
        self._dirty = True
        self._darea.queue_draw()

        # Snapshot the surface data now (main thread) but do conversion in bg
        self.canvas_surface.flush()
        snap_w = self.canvas_surface.get_width()
        snap_h = self.canvas_surface.get_height()
        snap_stride = self.canvas_surface.get_stride()
        snap_data = bytes(self.canvas_surface.get_data())

        def _infer():
            gray, w, h = _grayscale_from_raw(snap_data, snap_w, snap_h, snap_stride)
            t0 = time.monotonic()
            result = self.client.infer(gray, w, h)
            elapsed = (time.monotonic() - t0) * 1000

            with self._lock:
                if "error" in result:
                    if result["error"] != "blank":
                        self._server_connected = False
                    self._predictions = []
                    self._has_predictions = False
                else:
                    preds = result.get("predictions", [])
                    self._predictions = preds
                    self._has_predictions = bool(preds)
                    self._infer_ms = elapsed
                    self._feedback_given = False
                    comment = self._commentary.pick(preds)
                    if comment and comment.startswith("__update_prob__"):
                        # Update percentage in existing comment
                        parts = comment.split("__")
                        new_pct, old_pct = parts[2], parts[3]
                        self._ai_comment = self._ai_comment.replace(old_pct, new_pct)
                    elif comment:
                        self._ai_comment = comment
                    self._guess_trail = self._commentary.get_guess_trail()
                self._predicting = False

            self._dirty = True
            GLib.idle_add(self._darea.queue_draw)

        threading.Thread(target=_infer, daemon=True).start()

    def _do_exit(self):
        self.client.close()
        Gtk.main_quit()

    def _do_yes(self):
        with self._lock:
            if self._feedback_given or not self._has_predictions:
                return
            self._feedback_given = True
            self._score_correct += 1
            if self._predictions:
                top = self._predictions[0]
                self._history.append({"name": top["class"], "conf": top["prob"], "correct": True})
                comment = self._commentary.on_feedback_yes(self._predictions)
                if comment:
                    self._ai_comment = comment
        self._dirty = True
        self._darea.queue_draw()

    def _do_no(self):
        with self._lock:
            if self._feedback_given or not self._has_predictions:
                return
            self._feedback_given = True
            self._score_wrong += 1
            if self._predictions:
                top = self._predictions[0]
                self._history.append({"name": top["class"], "conf": top["prob"], "correct": False})
                comment = self._commentary.on_feedback_no(self._predictions)
                if comment:
                    self._ai_comment = comment
        self._dirty = True
        self._darea.queue_draw()

    # ── Cairo primitives ──

    def _rounded_rect(self, cr, x, y, w, h, r=6):
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -1.5708, 0)
        cr.arc(x + w - r, y + h - r, r, 0, 1.5708)
        cr.arc(x + r, y + h - r, r, 1.5708, 3.14159)
        cr.arc(x + r, y + r, r, 3.14159, 4.71239)
        cr.close_path()

    def _text(self, cr, text, x, y, size=14, color=None, bold=False):
        cr.set_source_rgb(*(color or self.col["text"]))
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                            cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
        scaled = size * self.font_scale
        cr.set_font_size(scaled)
        cr.move_to(x, y + scaled)
        cr.show_text(text)

    def _wrap_lines(self, cr, text, max_width, size=13):
        """Word-wrap text, return list of line strings."""
        cr.set_font_size(size * self.font_scale)
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = word if not current else current + " " + word
            if cr.text_extents(test).width > max_width and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        return lines

    def _section_header(self, cr, text, x, y, panel_w):
        """Draw a labeled section divider. Returns y of content below it."""
        self._text(cr, text, x, y, 12, self.col["text_dim"], bold=True)
        line_y = y + int(18 * self.font_scale)
        cr.set_source_rgb(*self.col["panel_line"])
        cr.move_to(x, line_y)
        cr.line_to(x + panel_w - 20, line_y)
        cr.set_line_width(1)
        cr.stroke()
        return line_y + 10

    # ── Rendering: title bar ──

    def _draw_titlebar(self, cr, L):
        cr.set_source_rgb(*self.col["title_bg"])
        cr.rectangle(0, 0, self._scr_w, L["title_h"])
        cr.fill()

        title = self.ui.get("title", "QUICK DRAW")
        subtitle = self.ui.get("subtitle", "DRP-AI3")
        badge = self.ui.get("badge", "RZ/V2N SR SOM")

        self._text(cr, title, 15, 10, 22, self.col["accent"], bold=True)
        self._text(cr, subtitle, 210, 15, 15, self.col["text_dim"])

        right_edge = self._scr_w - 20
        if self._logo:
            logo_w = self._logo.get_width()
            logo_h = self._logo.get_height()
            target_h = min(L["title_h"] * 1.8, L["title_h"] - 4)
            scale = target_h / logo_h
            scaled_w = logo_w * scale
            logo_x = right_edge - scaled_w
            logo_y = (L["title_h"] - target_h) / 2
            cr.save()
            cr.translate(logo_x, logo_y)
            cr.scale(scale, scale)
            cr.set_source_surface(self._logo, 0, 0)
            cr.paint()
            cr.restore()
            right_edge = logo_x - 12

        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(13 * self.font_scale)
        ext = cr.text_extents(badge)
        self._text(cr, badge, right_edge - ext.width, 15, 13, self.col["text_dim"])

    # ── Rendering: canvas ──

    def _draw_canvas(self, cr, L):
        cr.set_source_rgb(*self.col["panel_line"])
        cr.rectangle(L["canvas_x"] - 2, L["canvas_y"] - 2,
                     L["canvas_w"] + 4, L["canvas_h"] + 4)
        cr.set_line_width(2)
        cr.stroke()

        cr.set_source_surface(self.canvas_surface, L["canvas_x"], L["canvas_y"])
        cr.paint()

    # ── Rendering: right panel (flowing y layout) ──

    def _draw_panel(self, cr, L, predictions, has_predictions, predicting,
                    infer_ms, ai_comment, guess_trail,
                    score_correct, score_wrong, history):
        px = L["panel_x"]
        pw = L["panel_w"]
        max_y = L["panel_max_y"]
        py = L["canvas_y"] + 5

        # ── Prediction section ──
        py = self._section_header(cr, "PREDICTION", px, py, pw)
        conf_thresh = self.model_cfg.get("confidence_threshold", 0.15)

        if predicting:
            self._text(cr, "Analyzing...", px, py, 18, self.col["yellow"])
            py += 30
        elif has_predictions and predictions:
            top = predictions[0]
            name, prob = top["class"], top["prob"]

            if prob < conf_thresh:
                self._text(cr, "Uncertain...", px, py, 28, self.col["yellow"], bold=True)
                self._text(cr, f"Best: {name} ({prob*100:.1f}%)",
                           px, py + 42, 16, self.col["text_dim"])
                py += 70
            else:
                self._text(cr, name, px, py, 28, self.col["accent"], bold=True)
                self._text(cr, f"{prob*100:.1f}%", px, py + 38, 18, self.col["green"], bold=True)
                if infer_ms > 0:
                    self._text(cr, f"Round-trip: {infer_ms:.1f} ms",
                               px + 100, py + 41, 12, self.col["text_dim"])
                py += 70

            # Comment bubble
            if ai_comment:
                py = self._draw_comment_bubble(cr, px, py, ai_comment, pw)
        else:
            self._text(cr, "Draw something!", px, py, 18, self.col["text_dim"])
            py += 30

        py += 10

        # ── AI Thinking — guess trail (like Google Quick Draw) ──
        if guess_trail and len(guess_trail) >= 2 and py < max_y:
            py = self._section_header(cr, "AI THINKING", px, py, pw)
            # Show the AI's journey: crossed-out wrong guesses, current guess highlighted
            trail_str = ""
            for i, guess in enumerate(guess_trail):
                if i < len(guess_trail) - 1:
                    # Previous wrong guesses — dimmed with strikethrough effect
                    trail_str += guess + "  >  "
                else:
                    trail_str += guess
            # Wrap and render the trail
            lines = self._wrap_lines(cr, trail_str, pw - 20, 12)
            for line in lines:
                if py >= max_y:
                    break
                self._text(cr, line, px, py, 12, self.col["text_dim"])
                py += 16
            py += 10

        # ── Top 5 bars ──
        if has_predictions and predictions and py < max_y:
            py = self._section_header(cr, "TOP 5", px, py, pw)
            bar_max_w = pw - 20
            bar_h = 24
            for i, pred in enumerate(predictions[:5]):
                if py + bar_h > max_y:
                    break
                p = pred["prob"]
                self._rounded_rect(cr, px, py, bar_max_w, bar_h, 3)
                cr.set_source_rgb(*self.col["bar_bg"])
                cr.fill()

                fill_w = max(2, bar_max_w * p)
                self._rounded_rect(cr, px, py, fill_w, bar_h, 3)
                cr.set_source_rgb(*(self.col["accent"] if i == 0 else self.col["bar_fg"]))
                cr.fill()

                label = f"{i+1}. {pred['class']}  {p*100:.1f}%"
                col = self.col["text"] if p > 0.15 else self.col["text_dim"]
                self._text(cr, label, px + 6, py + 4, 12, col)
                py += bar_h + 6

            py += 10

        # ── Score ──
        if py < max_y:
            py = self._section_header(cr, "SCORE", px, py, pw)
            self._text(cr, f"Correct: {score_correct}", px, py, 16, self.col["green"])
            self._text(cr, f"Wrong: {score_wrong}", px + 180, py, 16, self.col["red"])
            py += 30

        # ── History ──
        if history and py < max_y:
            py = self._section_header(cr, "HISTORY", px, py, pw)
            for entry in reversed(history):
                if py >= max_y:
                    break
                mark = "[Y]" if entry["correct"] else "[N]"
                col = self.col["green"] if entry["correct"] else self.col["red"]
                self._text(cr, f"{mark} {entry['name']} ({entry['conf']*100:.0f}%)",
                           px, py, 12, col)
                py += 20

    def _draw_comment_bubble(self, cr, x, y, text, panel_w):
        """Draw AI comment in a rounded bubble. Returns y after bubble."""
        pad = 8
        bubble_w = panel_w - 20
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        comment_sz = 14
        lines = self._wrap_lines(cr, text, bubble_w - 2 * pad, comment_sz)

        scaled_sz = comment_sz * self.font_scale
        line_h = int(scaled_sz * 1.35)
        bubble_h = len(lines) * line_h + 2 * pad
        self._rounded_rect(cr, x, y, bubble_w, bubble_h, 6)
        cr.set_source_rgb(*self.col["comment_bg"])
        cr.fill()

        ctxt = self.col["comment_text"]
        for i, line in enumerate(lines):
            self._text(cr, line, x + pad, y + pad + i * line_h, comment_sz, ctxt)

        return y + bubble_h + 8

    # ── Rendering: buttons ──

    def _draw_buttons(self, cr, L):
        for btn in L["buttons"]:
            self._rounded_rect(cr, btn["x"], btn["y"], btn["w"], btn["h"], 4)
            cr.set_source_rgb(*self.col[btn["color"]])
            cr.fill_preserve()
            cr.set_source_rgb(*self.col["panel_line"])
            cr.set_line_width(1)
            cr.stroke()

            cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            cr.set_font_size(16 * self.font_scale)
            ext = cr.text_extents(btn["label"])
            tx = btn["x"] + (btn["w"] - ext.width) / 2
            ty = btn["y"] + (btn["h"] + ext.height) / 2
            cr.set_source_rgb(*self.col["text"])
            cr.move_to(tx, ty)
            cr.show_text(btn["label"])

    # ── Rendering: status bar ──

    def _draw_status(self, cr, L, predicting, server_connected):
        if predicting:
            status = "Analyzing..."
        elif self._timer_active and self.strokes:
            status = "Auto-predict pending..."
        elif not server_connected:
            status = "Waiting for server..."
        else:
            status = "Ready"

        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12 * self.font_scale)
        stext = f"Status: {status}"
        se = cr.text_extents(stext)
        btn_mid = L["btn_bar_y"] + L["btn_h"] / 2
        self._text(cr, stext, self._scr_w - se.width - 30,
                   btn_mid - se.height / 2, 12, self.col["text_dim"])

        # Server indicator dot
        dot_col = self.col["green"] if server_connected else self.col["red"]
        cr.set_source_rgb(*dot_col)
        cr.arc(self._scr_w - 12, btn_mid + 2, 5, 0, TWO_PI)
        cr.fill()

    # ── Draw callback ──

    def _on_draw(self, widget, cr):
        alloc = widget.get_allocation()
        if alloc.width != self._scr_w or alloc.height != self._scr_h:
            self._compute_layout(alloc.width, alloc.height)
        L = self._layout
        if not L:
            return

        # Snapshot all shared state under lock
        with self._lock:
            predictions = list(self._predictions)
            has_predictions = self._has_predictions
            predicting = self._predicting
            server_connected = self._server_connected
            infer_ms = self._infer_ms
            ai_comment = self._ai_comment
            guess_trail = list(self._guess_trail)
            score_correct = self._score_correct
            score_wrong = self._score_wrong
            history = list(self._history)

        # Background
        cr.set_source_rgb(*self.col["background"])
        cr.paint()

        self._draw_titlebar(cr, L)
        self._draw_canvas(cr, L)
        self._draw_panel(cr, L, predictions, has_predictions, predicting,
                         infer_ms, ai_comment, guess_trail,
                         score_correct, score_wrong, history)
        self._draw_buttons(cr, L)
        self._draw_status(cr, L, predicting, server_connected)

        self._dirty = False

    # ── Input handlers ──

    def _in_canvas(self, x, y):
        L = self._layout
        return (L["canvas_x"] <= x < L["canvas_x"] + L["canvas_w"] and
                L["canvas_y"] <= y < L["canvas_y"] + L["canvas_h"])

    def _hit_button(self, mx, my):
        for btn in self._layout.get("buttons", []):
            if (btn["x"] <= mx <= btn["x"] + btn["w"] and
                    btn["y"] <= my <= btn["y"] + btn["h"]):
                return btn["label"]
        return None

    _BUTTON_ACTIONS = {
        "CLEAR": "_do_clear", "UNDO": "_do_undo", "EXIT": "_do_exit",
        "YES": "_do_yes", "NO": "_do_no",
    }

    def _on_press(self, widget, event):
        if event.button != 1:
            return False

        hit = self._hit_button(event.x, event.y)
        if hit:
            getattr(self, self._BUTTON_ACTIONS[hit])()
            return True

        if self._in_canvas(event.x, event.y):
            self.drawing = True
            cx = event.x - self._layout["canvas_x"]
            cy = event.y - self._layout["canvas_y"]
            self.current_stroke = [(cx, cy)]
            self._draw_dot(cx, cy)
            self._dirty = True
            self._darea.queue_draw()
        return True

    def _on_release(self, widget, event):
        if event.button != 1 or not self.drawing:
            return False
        self.drawing = False
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self._last_stroke_time = time.monotonic()
            self._timer_active = True
        self._dirty = True
        self._darea.queue_draw()
        return True

    def _on_motion(self, widget, event):
        if not self.drawing:
            return False
        if self._in_canvas(event.x, event.y):
            cx = event.x - self._layout["canvas_x"]
            cy = event.y - self._layout["canvas_y"]
            if self.current_stroke:
                lx, ly = self.current_stroke[-1]
                self._draw_line(lx, ly, cx, cy)
            self.current_stroke.append((cx, cy))
            self._dirty = True
            # Don't queue_draw on every motion — let _on_tick handle it
            # This prevents dozens of full redraws per second during drawing
        return True

    _KEY_ACTIONS = {
        Gdk.KEY_c: "_do_clear",
        Gdk.KEY_z: "_do_undo",
        Gdk.KEY_Return: "_do_predict",
        Gdk.KEY_space: "_do_predict",
        Gdk.KEY_y: "_do_yes",
        Gdk.KEY_n: "_do_no",
        Gdk.KEY_q: "_do_exit",
    }

    def _on_key(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            self.client.close()
            Gtk.main_quit()
        else:
            action = self._KEY_ACTIONS.get(event.keyval)
            if action:
                getattr(self, action)()
        return True

    # ── Tick (auto-predict, live predict, reconnect) ──

    def _on_tick(self):
        now = time.monotonic()

        with self._lock:
            is_predicting = self._predicting

        # Live prediction while drawing
        live_ms = self.ui.get("live_predict_interval_ms", 0)
        if live_ms > 0 and self.drawing and not is_predicting:
            has_ink = self.strokes or len(self.current_stroke) > 3
            if has_ink and now - self._last_live_predict >= live_ms / 1000.0:
                self._last_live_predict = now
                self._do_predict()

        # Auto-predict after pen release
        delay = self.ui.get("auto_predict_delay_ms", 500) / 1000.0
        if (self._timer_active and not is_predicting and
                self._last_stroke_time > 0 and
                now - self._last_stroke_time > delay):
            self._do_predict()

        # Periodic server reconnect check
        with self._lock:
            connected = self._server_connected
        if not connected:
            new_status = self.client.check_connected()
            with self._lock:
                if new_status != self._server_connected:
                    self._server_connected = new_status
                    self._dirty = True

        if self._dirty:
            self._darea.queue_draw()

        return True


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    config_path = str(SCRIPT_DIR / "config.json")
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config" and i < len(sys.argv) - 1:
            config_path = sys.argv[i + 1]

    print(f"Quick Draw GUI — Python GTK3", file=sys.stderr)
    print(f"Config: {config_path}", file=sys.stderr)

    QuickDrawApp(config_path).run()


if __name__ == "__main__":
    main()
