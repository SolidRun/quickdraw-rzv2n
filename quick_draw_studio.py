#!/usr/bin/env python3
"""
Quick Draw Studio — Training & Deployment GUI
==============================================

Tkinter GUI for the Quick Draw DRP-AI3 pipeline:
  1. Dataset  — Download Quick Draw data, manage categories, generate calibration
  2. Training — Train MobileNetV2 on sketches (3ch RGB, transfer learning), export ONNX
  3. Deploy   — DRP-AI TVM compilation (Docker), C++ cross-compile, package for board

Mirrors the YOLO Studio layout and workflow, adapted for classification.

Usage:
    python quick_draw_studio.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import queue
import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List


__version__ = "1.0.0"


# ═══════════════════════════════════════════════════════════════
#  Theme — GitHub-inspired dark (matches YOLO Studio)
# ═══════════════════════════════════════════════════════════════

class Theme:
    BG_DARK = "#0d1117"
    BG_MEDIUM = "#161b22"
    BG_LIGHT = "#21262d"
    TEXT_PRIMARY = "#f0f6fc"
    TEXT_SECONDARY = "#8b949e"
    TEXT_MUTED = "#6e7681"
    ACCENT_BLUE = "#58a6ff"
    ACCENT_GREEN = "#3fb950"
    ACCENT_YELLOW = "#d29922"
    ACCENT_RED = "#f85149"
    ACCENT_PURPLE = "#a371f7"
    ACCENT_CYAN = "#39c5cf"
    ACCENT_ORANGE = "#f0883e"
    BORDER = "#30363d"
    BORDER_LIGHT = "#484f58"

    FONT = "Segoe UI"
    FONT_MONO = "Consolas"

    @staticmethod
    def card_frame(parent, **kw):
        f = tk.Frame(parent, bg=Theme.BG_LIGHT, padx=12, pady=10, **kw)
        return f


# ═══════════════════════════════════════════════════════════════
#  Reusable Widgets
# ═══════════════════════════════════════════════════════════════

class Card(tk.Frame):
    """Titled card container."""

    def __init__(self, parent, title: str, **kw):
        super().__init__(parent, bg=Theme.BG_LIGHT, padx=12, pady=10, **kw)

        if title:
            tk.Label(
                self, text=title, font=(Theme.FONT, 11, "bold"),
                fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT, anchor=tk.W
            ).pack(fill=tk.X, pady=(0, 8))

        self.content = tk.Frame(self, bg=Theme.BG_LIGHT)
        self.content.pack(fill=tk.BOTH, expand=True)


class LogViewer(tk.Frame):
    """Scrollable log text area."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=Theme.BG_DARK, **kw)

        self.text = tk.Text(
            self, bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY,
            font=(Theme.FONT_MONO, 9), wrap=tk.WORD,
            insertbackground=Theme.TEXT_PRIMARY,
            selectbackground=Theme.ACCENT_BLUE,
            highlightthickness=0, padx=8, pady=8, state=tk.DISABLED
        )
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tag styles
        self.text.tag_configure("info", foreground=Theme.ACCENT_CYAN)
        self.text.tag_configure("success", foreground=Theme.ACCENT_GREEN)
        self.text.tag_configure("warning", foreground=Theme.ACCENT_YELLOW)
        self.text.tag_configure("error", foreground=Theme.ACCENT_RED)
        self.text.tag_configure("header", foreground=Theme.ACCENT_BLUE, font=(Theme.FONT_MONO, 10, "bold"))

    def append(self, text: str, tag: str = ""):
        self.text.configure(state=tk.NORMAL)
        if tag:
            self.text.insert(tk.END, text, tag)
        else:
            self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)


class ActionButton(tk.Frame):
    """Styled action button with hover effect."""

    def __init__(self, parent, text: str, command=None,
                 bg_color=None, fg_color=None, width=180, height=36, **kw):
        super().__init__(parent, bg=Theme.BG_LIGHT, **kw)

        self._bg = bg_color or Theme.ACCENT_BLUE
        self._fg = fg_color or "#ffffff"
        self._command = command
        self._enabled = True

        self.btn = tk.Label(
            self, text=text, font=(Theme.FONT, 10, "bold"),
            fg=self._fg, bg=self._bg,
            padx=20, pady=6, cursor="hand2",
            width=width // 10,  # approximate char width
        )
        self.btn.pack(fill=tk.X)

        self.btn.bind("<Enter>", self._on_enter)
        self.btn.bind("<Leave>", self._on_leave)
        self.btn.bind("<Button-1>", self._on_click)

    def _lighten(self, color):
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            r = min(255, r + 30)
            g = min(255, g + 30)
            b = min(255, b + 30)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return color

    def _on_enter(self, _e):
        if self._enabled:
            self.btn.config(bg=self._lighten(self._bg))

    def _on_leave(self, _e):
        self.btn.config(bg=self._bg if self._enabled else Theme.BG_LIGHT)

    def _on_click(self, _e):
        if self._enabled and self._command:
            self._command()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self.btn.config(
            cursor="hand2" if enabled else "",
            bg=self._bg if enabled else Theme.BG_LIGHT,
            fg=self._fg if enabled else Theme.TEXT_MUTED,
        )

    def set_text(self, text: str):
        self.btn.config(text=text)


# ═══════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════

class QuickDrawStudio:
    """Quick Draw Studio — Training & Deployment GUI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Quick Draw Studio v{__version__}")
        self.root.geometry("1500x900")
        self.root.configure(bg=Theme.BG_DARK)
        self.root.minsize(1100, 700)

        # ── Paths ──
        self.project_dir = Path(__file__).parent.resolve()
        self.train_dir = self.project_dir / "train"
        self.data_dir = self.train_dir / "data"
        self.board_app_dir = self.project_dir / "board_app"
        self.deploy_dir = self.board_app_dir / "deploy"
        self.categories_file = self.project_dir / "categories.txt"
        self.calibration_dir = self.project_dir / "calibration"
        self.model_pt = self.project_dir / "best_model.pt"
        self.model_onnx = self.project_dir / "qd_model.onnx"

        # ── State ──
        self.current_page = "dataset"
        self.pages: Dict[str, tk.Frame] = {}
        self.nav_buttons: Dict[str, tk.Label] = {}
        self.log_queue: queue.Queue = queue.Queue()
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False

        # ── Variables ──
        self._init_variables()

        # ── UI ──
        self._setup_styles()
        self._setup_layout()
        self._setup_sidebar()
        self._create_pages()
        self._setup_status_bar()

        # ── Startup ──
        self._refresh_dataset_status()
        self.root.after(100, self._process_log_queue)

    # ───────────────────────────────────────────────────────
    #  Variables
    # ───────────────────────────────────────────────────────

    def _init_variables(self):
        # Dataset
        self.max_samples_var = tk.IntVar(value=6000)

        # Training
        self.epochs_var = tk.IntVar(value=30)
        self.batch_var = tk.IntVar(value=128)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.max_per_class_var = tk.IntVar(value=5000)
        self.workers_var = tk.IntVar(value=8)
        self.input_size_var = tk.IntVar(value=128)

        # Deploy
        self.docker_container_var = tk.StringVar(value="drp-ai_tvm_v2n_container_ahmad")
        self.num_calib_var = tk.IntVar(value=345)

    # ───────────────────────────────────────────────────────
    #  Style Setup
    # ───────────────────────────────────────────────────────

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TCombobox",
                        fieldbackground=Theme.BG_DARK,
                        background=Theme.BG_MEDIUM,
                        foreground=Theme.TEXT_PRIMARY,
                        arrowcolor=Theme.TEXT_PRIMARY,
                        bordercolor=Theme.BORDER)
        style.map("TCombobox",
                  fieldbackground=[("readonly", Theme.BG_DARK)],
                  foreground=[("readonly", Theme.TEXT_PRIMARY)])

        style.configure("TScrollbar",
                        background=Theme.BG_MEDIUM,
                        troughcolor=Theme.BG_DARK,
                        arrowcolor=Theme.TEXT_PRIMARY)

        style.configure("TSpinbox",
                        fieldbackground=Theme.BG_DARK,
                        background=Theme.BG_MEDIUM,
                        foreground=Theme.TEXT_PRIMARY,
                        arrowcolor=Theme.TEXT_PRIMARY,
                        bordercolor=Theme.BORDER)

    # ───────────────────────────────────────────────────────
    #  Layout
    # ───────────────────────────────────────────────────────

    def _setup_layout(self):
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)

        self.content_frame = tk.Frame(self.root, bg=Theme.BG_DARK)
        self.content_frame.grid(row=0, column=1, sticky="nsew")

    def _setup_sidebar(self):
        sidebar = tk.Frame(self.root, bg=Theme.BG_MEDIUM, width=260)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="ns")
        sidebar.grid_propagate(False)

        # Logo
        logo_frame = tk.Frame(sidebar, bg=Theme.BG_MEDIUM)
        logo_frame.pack(fill=tk.X, pady=(15, 5))

        self.logo_image = None
        logo_path = self.project_dir / "board_app" / "solidrun_logo.png"
        if logo_path.exists():
            try:
                from PIL import Image, ImageTk
                img = Image.open(logo_path)
                # Scale to 240px wide (large logo filling sidebar)
                ratio = 240 / img.width
                new_h = int(img.height * ratio)
                img = img.resize((240, new_h), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
                tk.Label(
                    logo_frame, image=self.logo_image, bg=Theme.BG_MEDIUM
                ).pack(pady=(0, 8))
            except Exception:
                pass

        # Title
        title_frame = tk.Frame(sidebar, bg=Theme.BG_MEDIUM)
        title_frame.pack(fill=tk.X, pady=(5, 5))

        tk.Label(
            title_frame, text="Quick Draw",
            font=(Theme.FONT, 18, "bold"),
            fg=Theme.ACCENT_CYAN, bg=Theme.BG_MEDIUM
        ).pack()

        tk.Label(
            title_frame, text="Studio",
            font=(Theme.FONT, 14),
            fg=Theme.ACCENT_BLUE, bg=Theme.BG_MEDIUM
        ).pack()

        tk.Label(
            title_frame, text=f"v{__version__}  |  DRP-AI3",
            font=(Theme.FONT, 8),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM
        ).pack(pady=(2, 0))

        # Separator
        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=20, pady=15)

        # Navigation
        nav_items = [
            ("dataset", "Dataset", Theme.ACCENT_GREEN),
            ("training", "Training", Theme.ACCENT_BLUE),
            ("deploy", "Deploy", Theme.ACCENT_ORANGE),
        ]

        for key, text, color in nav_items:
            btn = self._create_nav_button(sidebar, text, key, color)
            self.nav_buttons[key] = btn

        # Bottom info
        bottom = tk.Frame(sidebar, bg=Theme.BG_MEDIUM)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        self.gpu_label = tk.Label(
            bottom, text="", font=(Theme.FONT, 9),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM
        )
        self.gpu_label.pack(pady=5)
        self._check_gpu()

        tk.Label(
            bottom, text="RZ/V2N SR SOM",
            font=(Theme.FONT, 8),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM
        ).pack()

    def _create_nav_button(self, parent, text: str, key: str, color: str) -> tk.Label:
        frame = tk.Frame(parent, bg=Theme.BG_MEDIUM)
        frame.pack(fill=tk.X, padx=10, pady=2)

        indicator = tk.Frame(frame, bg=color, width=3)
        indicator.pack(side=tk.LEFT, fill=tk.Y)
        indicator.pack_forget()

        btn = tk.Label(
            frame, text=f"  {text}",
            font=(Theme.FONT, 12),
            fg=Theme.TEXT_SECONDARY, bg=Theme.BG_MEDIUM,
            anchor=tk.W, cursor="hand2", padx=10, pady=8
        )
        btn.pack(fill=tk.X)
        btn.color = color
        btn.indicator = indicator

        btn.bind("<Enter>", lambda e, b=btn: b.config(
            bg=Theme.BG_LIGHT, fg=b.color))
        btn.bind("<Leave>", lambda e, b=btn, k=key: b.config(
            bg=Theme.BG_LIGHT if k == self.current_page else Theme.BG_MEDIUM,
            fg=b.color if k == self.current_page else Theme.TEXT_SECONDARY))
        btn.bind("<Button-1>", lambda e, k=key: self._show_page(k))

        return btn

    def _show_page(self, page_name: str):
        for p in self.pages.values():
            p.pack_forget()
        self.pages[page_name].pack(fill=tk.BOTH, expand=True)

        # Update nav styling
        for key, btn in self.nav_buttons.items():
            if key == page_name:
                btn.config(bg=Theme.BG_LIGHT, fg=btn.color)
                btn.indicator.pack(side=tk.LEFT, fill=tk.Y)
            else:
                btn.config(bg=Theme.BG_MEDIUM, fg=Theme.TEXT_SECONDARY)
                btn.indicator.pack_forget()

        self.current_page = page_name

    def _create_pages(self):
        self._create_dataset_page()
        self._create_training_page()
        self._create_deploy_page()
        self._show_page("dataset")

    def _setup_status_bar(self):
        bar = tk.Frame(self.root, bg=Theme.BG_MEDIUM, height=28)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)

        self.status_label = tk.Label(
            bar, text="Ready", font=(Theme.FONT, 9),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM, anchor=tk.W, padx=15
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_right = tk.Label(
            bar, text="", font=(Theme.FONT, 9),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM, anchor=tk.E, padx=15
        )
        self.status_right.pack(side=tk.RIGHT)

    def _set_status(self, text: str, color: str = None):
        self.status_label.config(text=text, fg=color or Theme.TEXT_MUTED)

    # ───────────────────────────────────────────────────────
    #  Hardware Check
    # ───────────────────────────────────────────────────────

    def _check_gpu(self):
        def _check():
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    self.root.after(0, lambda: self.gpu_label.config(
                        text=f"GPU: {name}", fg=Theme.ACCENT_GREEN))
                else:
                    self.root.after(0, lambda: self.gpu_label.config(
                        text="CPU only (no CUDA)", fg=Theme.ACCENT_YELLOW))
            except ImportError:
                self.root.after(0, lambda: self.gpu_label.config(
                    text="PyTorch not installed", fg=Theme.ACCENT_RED))

        threading.Thread(target=_check, daemon=True).start()

    # ═══════════════════════════════════════════════════════
    #  PAGE 1: DATASET
    # ═══════════════════════════════════════════════════════

    def _create_dataset_page(self):
        page = tk.Frame(self.content_frame, bg=Theme.BG_DARK)
        self.pages["dataset"] = page

        # Header
        header = tk.Frame(page, bg=Theme.BG_DARK)
        header.pack(fill=tk.X, padx=30, pady=(30, 10))

        tk.Label(
            header, text="Dataset",
            font=(Theme.FONT, 24, "bold"),
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_DARK
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Download & manage Quick Draw sketch data",
            font=(Theme.FONT, 10),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_DARK
        ).pack(side=tk.LEFT, padx=(15, 0), pady=(8, 0))

        # Content: left settings, right log
        content = tk.Frame(page, bg=Theme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))

        # ── Left panel ──
        left = tk.Frame(content, bg=Theme.BG_DARK, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)

        # Status card
        status_card = Card(left, title="Dataset Status")
        status_card.pack(fill=tk.X, pady=(0, 10))

        self.ds_categories_label = tk.Label(
            status_card.content, text="Categories: ...",
            font=(Theme.FONT, 10), fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT,
            anchor=tk.W)
        self.ds_categories_label.pack(fill=tk.X, pady=2)

        self.ds_downloaded_label = tk.Label(
            status_card.content, text="Downloaded: ...",
            font=(Theme.FONT, 10), fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT,
            anchor=tk.W)
        self.ds_downloaded_label.pack(fill=tk.X, pady=2)

        self.ds_calib_label = tk.Label(
            status_card.content, text="Calibration: ...",
            font=(Theme.FONT, 10), fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT,
            anchor=tk.W)
        self.ds_calib_label.pack(fill=tk.X, pady=2)

        self.ds_model_label = tk.Label(
            status_card.content, text="Model (.pt): ...",
            font=(Theme.FONT, 10), fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT,
            anchor=tk.W)
        self.ds_model_label.pack(fill=tk.X, pady=2)

        self.ds_onnx_label = tk.Label(
            status_card.content, text="Model (.onnx): ...",
            font=(Theme.FONT, 10), fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT,
            anchor=tk.W)
        self.ds_onnx_label.pack(fill=tk.X, pady=2)

        tk.Button(
            status_card.content, text="Refresh", font=(Theme.FONT, 9),
            bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY, relief=tk.FLAT,
            cursor="hand2", command=self._refresh_dataset_status
        ).pack(anchor=tk.W, pady=(5, 0))

        # Download card
        dl_card = Card(left, title="1. Download Dataset")
        dl_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            dl_card.content,
            text="Downloads Quick Draw .npy bitmaps\nfrom Google Cloud Storage.",
            font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY, bg=Theme.BG_LIGHT,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 8))

        row = tk.Frame(dl_card.content, bg=Theme.BG_LIGHT)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text="Max samples/class:", font=(Theme.FONT, 9),
                 fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1000, to=100000, increment=1000,
                     textvariable=self.max_samples_var, width=8,
                     font=(Theme.FONT, 9)).pack(side=tk.RIGHT)

        self.download_btn = ActionButton(
            dl_card.content, "Download Dataset",
            command=self._download_dataset,
            bg_color=Theme.ACCENT_GREEN, width=200
        )
        self.download_btn.pack(anchor=tk.W, pady=(8, 0))

        # Calibration card
        cal_card = Card(left, title="2. Generate Calibration Images")
        cal_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            cal_card.content,
            text="Creates 1 sample/class for INT8\nDRP-AI quantization.",
            font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY, bg=Theme.BG_LIGHT,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 8))

        self.calib_btn = ActionButton(
            cal_card.content, "Generate Calibration",
            command=self._generate_calibration,
            bg_color=Theme.ACCENT_BLUE, width=200
        )
        self.calib_btn.pack(anchor=tk.W, pady=(2, 0))

        # ── Right panel (log) ──
        right = tk.Frame(content, bg=Theme.BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right, text="Log Output",
            font=(Theme.FONT, 11, "bold"),
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_DARK
        ).pack(anchor=tk.W, pady=(0, 5))

        self.dataset_log = LogViewer(right)
        self.dataset_log.pack(fill=tk.BOTH, expand=True)

    def _refresh_dataset_status(self):
        # Categories
        cats = []
        if self.categories_file.exists():
            with open(self.categories_file) as f:
                cats = [l.strip() for l in f if l.strip()]
        n_cats = len(cats)
        self.ds_categories_label.config(
            text=f"Categories: {n_cats}" if n_cats else "Categories: not found",
            fg=Theme.ACCENT_GREEN if n_cats else Theme.ACCENT_RED)

        # Downloaded .npy files
        n_npy = 0
        if self.data_dir.exists():
            n_npy = len(list(self.data_dir.glob("*.npy")))
        color = Theme.ACCENT_GREEN if n_npy >= n_cats and n_cats > 0 else (
            Theme.ACCENT_YELLOW if n_npy > 0 else Theme.ACCENT_RED)
        self.ds_downloaded_label.config(
            text=f"Downloaded: {n_npy}/{n_cats} categories", fg=color)

        # Calibration
        n_cal = 0
        if self.calibration_dir.exists():
            n_cal = len(list(self.calibration_dir.glob("*.png")))
        color = Theme.ACCENT_GREEN if n_cal > 0 else Theme.ACCENT_RED
        self.ds_calib_label.config(
            text=f"Calibration: {n_cal} images", fg=color)

        # Model files
        if self.model_pt.exists():
            sz = self.model_pt.stat().st_size / (1024 * 1024)
            self.ds_model_label.config(
                text=f"Model (.pt): {sz:.1f} MB", fg=Theme.ACCENT_GREEN)
        else:
            self.ds_model_label.config(text="Model (.pt): not found", fg=Theme.TEXT_MUTED)

        if self.model_onnx.exists():
            sz = self.model_onnx.stat().st_size / (1024 * 1024)
            self.ds_onnx_label.config(
                text=f"Model (.onnx): {sz:.1f} MB", fg=Theme.ACCENT_GREEN)
        else:
            self.ds_onnx_label.config(text="Model (.onnx): not found", fg=Theme.TEXT_MUTED)

    def _download_dataset(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        script = self.train_dir / "download_dataset.py"
        if not script.exists():
            messagebox.showerror("Error", f"Script not found:\n{script}")
            return

        self.dataset_log.clear()
        self._log(self.dataset_log, "═" * 50 + "\n", "header")
        self._log(self.dataset_log, "  Downloading Quick Draw Dataset\n", "header")
        self._log(self.dataset_log, "═" * 50 + "\n\n", "header")

        cmd = [
            sys.executable, str(script),
            "--categories", str(self.categories_file),
            "--output", str(self.data_dir),
            "--max-samples", str(self.max_samples_var.get()),
        ]
        self._run_process(cmd, self.dataset_log, cwd=str(self.train_dir),
                          on_done=self._refresh_dataset_status)

    def _generate_calibration(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        script = self.project_dir / "generate_calibration.py"
        if not script.exists():
            messagebox.showerror("Error", f"Script not found:\n{script}")
            return

        self.dataset_log.clear()
        self._log(self.dataset_log, "═" * 50 + "\n", "header")
        self._log(self.dataset_log, "  Generating Calibration Images\n", "header")
        self._log(self.dataset_log, "═" * 50 + "\n\n", "header")

        cmd = [
            sys.executable, str(script),
            "--categories", str(self.categories_file),
            "--data", str(self.data_dir),
            "--output", str(self.calibration_dir),
        ]
        self._run_process(cmd, self.dataset_log, cwd=str(self.project_dir),
                          on_done=self._refresh_dataset_status)

    # ═══════════════════════════════════════════════════════
    #  PAGE 2: TRAINING
    # ═══════════════════════════════════════════════════════

    def _create_training_page(self):
        page = tk.Frame(self.content_frame, bg=Theme.BG_DARK)
        self.pages["training"] = page

        # Header
        header = tk.Frame(page, bg=Theme.BG_DARK)
        header.pack(fill=tk.X, padx=30, pady=(30, 10))

        tk.Label(
            header, text="Training",
            font=(Theme.FONT, 24, "bold"),
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_DARK
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Train MobileNetV2 on Quick Draw sketches",
            font=(Theme.FONT, 10),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_DARK
        ).pack(side=tk.LEFT, padx=(15, 0), pady=(8, 0))

        # Content
        content = tk.Frame(page, bg=Theme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))

        # ── Left panel (settings) ──
        left = tk.Frame(content, bg=Theme.BG_DARK, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)

        # Model info
        info_card = Card(left, title="Model")
        info_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            info_card.content,
            text="MobileNetV2 (RGB 3-channel, sketch normalization)",
            font=(Theme.FONT, 10, "bold"),
            fg=Theme.ACCENT_CYAN, bg=Theme.BG_LIGHT
        ).pack(anchor=tk.W)

        tk.Label(
            info_card.content,
            text="Input: 1x3x128x128  |  Output: N classes\n"
                 "Normalization: sketch [0,1] scaling",
            font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY, bg=Theme.BG_LIGHT,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(3, 0))

        # Hyperparameters
        hyp_card = Card(left, title="Hyperparameters")
        hyp_card.pack(fill=tk.X, pady=(0, 10))

        params = [
            ("Epochs:", self.epochs_var, 1, 500, 5),
            ("Batch Size:", self.batch_var, 8, 512, 8),
            ("Max Samples/Class:", self.max_per_class_var, 100, 50000, 500),
            ("Workers:", self.workers_var, 0, 32, 1),
        ]

        for label, var, lo, hi, inc in params:
            row = tk.Frame(hyp_card.content, bg=Theme.BG_LIGHT)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, font=(Theme.FONT, 9),
                     fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT).pack(side=tk.LEFT)
            ttk.Spinbox(row, from_=lo, to=hi, increment=inc,
                         textvariable=var, width=8,
                         font=(Theme.FONT, 9)).pack(side=tk.RIGHT)

        # Learning rate (special - DoubleVar)
        lr_row = tk.Frame(hyp_card.content, bg=Theme.BG_LIGHT)
        lr_row.pack(fill=tk.X, pady=3)
        tk.Label(lr_row, text="Learning Rate:", font=(Theme.FONT, 9),
                 fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT).pack(side=tk.LEFT)
        tk.Entry(lr_row, textvariable=self.lr_var, width=10,
                 font=(Theme.FONT, 9), bg=Theme.BG_DARK,
                 fg=Theme.TEXT_PRIMARY, insertbackground=Theme.TEXT_PRIMARY,
                 highlightbackground=Theme.BORDER,
                 highlightcolor=Theme.ACCENT_BLUE,
                 highlightthickness=1).pack(side=tk.RIGHT)

        # Actions
        action_card = Card(left, title="Actions")
        action_card.pack(fill=tk.X, pady=(0, 10))

        btn_row = tk.Frame(action_card.content, bg=Theme.BG_LIGHT)
        btn_row.pack(fill=tk.X, pady=5)

        self.train_btn = ActionButton(
            btn_row, "Train Model",
            command=self._start_training,
            bg_color=Theme.ACCENT_GREEN, width=180
        )
        self.train_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_train_btn = ActionButton(
            btn_row, "Stop",
            command=self._stop_process,
            bg_color=Theme.ACCENT_RED, width=80
        )
        self.stop_train_btn.pack(side=tk.LEFT)

        # Export button
        self.export_btn = ActionButton(
            action_card.content, "Export ONNX Only",
            command=self._export_onnx,
            bg_color=Theme.ACCENT_PURPLE, width=180
        )
        self.export_btn.pack(anchor=tk.W, pady=(5, 0))

        tk.Label(
            action_card.content,
            text="Training auto-exports ONNX when done.\n"
                 "Use 'Export ONNX Only' to re-export from\n"
                 "existing best_model.pt.",
            font=(Theme.FONT, 8), fg=Theme.TEXT_MUTED, bg=Theme.BG_LIGHT,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 0))

        # Progress
        progress_card = Card(left, title="Progress")
        progress_card.pack(fill=tk.X, pady=(0, 10))

        self.train_epoch_label = tk.Label(
            progress_card.content, text="Not started",
            font=(Theme.FONT, 10), fg=Theme.TEXT_SECONDARY, bg=Theme.BG_LIGHT
        )
        self.train_epoch_label.pack(anchor=tk.W)

        self.train_progress = ttk.Progressbar(
            progress_card.content, mode="determinate", length=350
        )
        self.train_progress.pack(fill=tk.X, pady=(5, 0))

        self.train_best_label = tk.Label(
            progress_card.content, text="",
            font=(Theme.FONT, 9), fg=Theme.ACCENT_GREEN, bg=Theme.BG_LIGHT
        )
        self.train_best_label.pack(anchor=tk.W, pady=(3, 0))

        # ── Right panel (log) ──
        right = tk.Frame(content, bg=Theme.BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right, text="Training Log",
            font=(Theme.FONT, 11, "bold"),
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_DARK
        ).pack(anchor=tk.W, pady=(0, 5))

        self.training_log = LogViewer(right)
        self.training_log.pack(fill=tk.BOTH, expand=True)

    def _start_training(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        script = self.train_dir / "train.py"
        if not script.exists():
            messagebox.showerror("Error", f"Training script not found:\n{script}")
            return

        if not self.data_dir.exists() or not list(self.data_dir.glob("*.npy")):
            messagebox.showerror("Error",
                                 "No training data found.\n"
                                 "Go to Dataset page and download first.")
            return

        self.training_log.clear()
        self._log(self.training_log, "═" * 50 + "\n", "header")
        self._log(self.training_log, "  Training MobileNetV2 on Quick Draw\n", "header")
        self._log(self.training_log, "═" * 50 + "\n\n", "header")

        self.train_progress["value"] = 0
        self.train_epoch_label.config(text="Starting...")
        self.train_best_label.config(text="")

        cmd = [
            sys.executable, str(script),
            "--data", str(self.data_dir),
            "--categories", str(self.categories_file),
            "--epochs", str(self.epochs_var.get()),
            "--batch", str(self.batch_var.get()),
            "--lr", str(self.lr_var.get()),
            "--max-per-class", str(self.max_per_class_var.get()),
            "--workers", str(self.workers_var.get()),
            "--output-pt", str(self.model_pt),
            "--output-onnx", str(self.model_onnx),
        ]

        def on_line(line):
            """Parse training output to update progress."""
            # Match: "   1 |     0.1234 |    85.23% | ..."
            parts = line.strip().split("|")
            if len(parts) >= 5:
                try:
                    epoch = int(parts[0].strip())
                    total = self.epochs_var.get()
                    pct = (epoch / total) * 100
                    self.root.after(0, lambda: self.train_progress.configure(value=pct))
                    self.root.after(0, lambda e=epoch, t=total:
                                   self.train_epoch_label.config(text=f"Epoch {e}/{t}"))

                    # Check for best marker
                    if line.rstrip().endswith("*"):
                        val_acc = parts[4].strip().rstrip("% *")
                        self.root.after(0, lambda a=val_acc:
                                       self.train_best_label.config(
                                           text=f"Best val accuracy: {a}%"))
                except (ValueError, IndexError):
                    pass

        def on_done():
            self.root.after(0, lambda: self.train_epoch_label.config(text="Complete"))
            self.root.after(0, lambda: self.train_progress.configure(value=100))
            self._refresh_dataset_status()

        self._run_process(cmd, self.training_log, cwd=str(self.train_dir),
                          on_line=on_line, on_done=on_done)

    def _export_onnx(self):
        """Export ONNX from existing best_model.pt (without retraining)."""
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        if not self.model_pt.exists():
            messagebox.showerror("Error",
                                 "best_model.pt not found.\nTrain a model first.")
            return

        self.training_log.clear()
        self._log(self.training_log, "═" * 50 + "\n", "header")
        self._log(self.training_log, "  Exporting ONNX from best_model.pt\n", "header")
        self._log(self.training_log, "═" * 50 + "\n\n", "header")

        # Create a small export-only script inline
        export_script = f"""
import sys, os
sys.path.insert(0, '{self.train_dir}')
from train import build_model, export_onnx
import torch

categories_path = '{self.categories_file}'
with open(categories_path) as f:
    categories = [l.strip() for l in f if l.strip()]
num_classes = len(categories)
print(f"Classes: {{num_classes}}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")

model = build_model(num_classes).to(device)
model.load_state_dict(torch.load('{self.model_pt}', map_location=device, weights_only=True))
print("Loaded best_model.pt")

export_onnx(model, num_classes, '{self.model_onnx}', device)
print("Done!")
"""

        cmd = [sys.executable, "-c", export_script]
        self._run_process(cmd, self.training_log, cwd=str(self.project_dir),
                          on_done=self._refresh_dataset_status)

    # ═══════════════════════════════════════════════════════
    #  PAGE 3: DEPLOY
    # ═══════════════════════════════════════════════════════

    def _create_deploy_page(self):
        page = tk.Frame(self.content_frame, bg=Theme.BG_DARK)
        self.pages["deploy"] = page

        # Header
        header = tk.Frame(page, bg=Theme.BG_DARK)
        header.pack(fill=tk.X, padx=30, pady=(30, 10))

        tk.Label(
            header, text="Deploy",
            font=(Theme.FONT, 24, "bold"),
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_DARK
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Compile, build & package for RZ/V2N SR SOM",
            font=(Theme.FONT, 10),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_DARK
        ).pack(side=tk.LEFT, padx=(15, 0), pady=(8, 0))

        # Docker status bar
        docker_bar = tk.Frame(page, bg=Theme.BG_MEDIUM, padx=12, pady=8)
        docker_bar.pack(fill=tk.X, padx=30, pady=(0, 10))

        self.docker_status_icon = tk.Label(
            docker_bar, text="\u25cf", font=(Theme.FONT, 12),
            fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM)
        self.docker_status_icon.pack(side=tk.LEFT)

        self.docker_status_label = tk.Label(
            docker_bar, text="Checking Docker...",
            font=(Theme.FONT, 9), fg=Theme.TEXT_MUTED, bg=Theme.BG_MEDIUM)
        self.docker_status_label.pack(side=tk.LEFT, padx=(6, 0))

        tk.Button(
            docker_bar, text="Refresh", font=(Theme.FONT, 8),
            bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY, relief=tk.FLAT,
            cursor="hand2", command=self._check_docker
        ).pack(side=tk.RIGHT)

        # Content
        content = tk.Frame(page, bg=Theme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))

        # ── Left panel ──
        left = tk.Frame(content, bg=Theme.BG_DARK, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)

        left_canvas = tk.Canvas(left, bg=Theme.BG_DARK, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
        left_inner = tk.Frame(left_canvas, bg=Theme.BG_DARK)
        left_inner.bind("<Configure>",
                        lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left_inner, anchor="nw", width=300)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # ── Card 1: Docker Container ──
        docker_card = Card(left_inner, title="1. Docker Container")
        docker_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(docker_card.content, text="Container Name:",
                 font=(Theme.FONT, 9), fg=Theme.TEXT_PRIMARY,
                 bg=Theme.BG_LIGHT).pack(anchor=tk.W)
        tk.Entry(docker_card.content, textvariable=self.docker_container_var,
                 font=(Theme.FONT, 9), bg=Theme.BG_DARK,
                 fg=Theme.TEXT_PRIMARY, insertbackground=Theme.TEXT_PRIMARY,
                 highlightbackground=Theme.BORDER,
                 highlightcolor=Theme.ACCENT_BLUE,
                 highlightthickness=1).pack(fill=tk.X, pady=(3, 0))

        tk.Label(docker_card.content,
                 text="DRP-AI TVM Docker with SDK + compiler",
                 font=(Theme.FONT, 8), fg=Theme.TEXT_MUTED,
                 bg=Theme.BG_LIGHT).pack(anchor=tk.W, pady=(3, 0))

        # ── Card 2: DRP-AI Compile ──
        compile_card = Card(left_inner, title="2. DRP-AI TVM Compile")
        compile_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(compile_card.content,
                 text="Compiles ONNX → INT8 DRP-AI model\nusing grayscale preprocessing.",
                 font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY,
                 bg=Theme.BG_LIGHT, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))

        row = tk.Frame(compile_card.content, bg=Theme.BG_LIGHT)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text="Calibration images:", font=(Theme.FONT, 9),
                 fg=Theme.TEXT_PRIMARY, bg=Theme.BG_LIGHT).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=10, to=1000, increment=10,
                     textvariable=self.num_calib_var, width=6,
                     font=(Theme.FONT, 9)).pack(side=tk.RIGHT)

        self.compile_btn = ActionButton(
            compile_card.content, "Compile Model",
            command=self._compile_model,
            bg_color="#E65100", width=200
        )
        self.compile_btn.pack(anchor=tk.W, pady=(8, 0))

        # ── Card 3: Cross-Compile C++ ──
        build_card = Card(left_inner, title="3. Build C++ App")
        build_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(build_card.content,
                 text="Cross-compiles the C++ DRP-AI server\nfor ARM Cortex-A55.",
                 font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY,
                 bg=Theme.BG_LIGHT, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))

        self.build_btn = ActionButton(
            build_card.content, "Build App",
            command=self._build_app,
            bg_color=Theme.ACCENT_BLUE, width=200
        )
        self.build_btn.pack(anchor=tk.W, pady=(8, 0))

        # ── Card 4: Package Deploy ──
        pkg_card = Card(left_inner, title="4. Package for Board")
        pkg_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(pkg_card.content,
                 text="Assembles deploy/ folder with binary,\n"
                      "model, libs, GUI, config, and run.sh.",
                 font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY,
                 bg=Theme.BG_LIGHT, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))

        self.package_btn = ActionButton(
            pkg_card.content, "Package Deploy",
            command=self._package_deploy,
            bg_color=Theme.ACCENT_GREEN, width=200
        )
        self.package_btn.pack(anchor=tk.W, pady=(8, 0))

        # ── Card 5: Full Pipeline ──
        full_card = Card(left_inner, title="Full Pipeline")
        full_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(full_card.content,
                 text="Run compile + build + package in sequence.",
                 font=(Theme.FONT, 9), fg=Theme.TEXT_SECONDARY,
                 bg=Theme.BG_LIGHT, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))

        self.full_pipeline_btn = ActionButton(
            full_card.content, "Run Full Pipeline",
            command=self._run_full_pipeline,
            bg_color=Theme.ACCENT_PURPLE, width=200
        )
        self.full_pipeline_btn.pack(anchor=tk.W, pady=(8, 0))

        # ── Right panel (log) ──
        right = tk.Frame(content, bg=Theme.BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right, text="Deploy Log",
            font=(Theme.FONT, 11, "bold"),
            fg=Theme.TEXT_PRIMARY, bg=Theme.BG_DARK
        ).pack(anchor=tk.W, pady=(0, 5))

        self.deploy_log = LogViewer(right)
        self.deploy_log.pack(fill=tk.BOTH, expand=True)

        # Check docker on page creation
        self.root.after(500, self._check_docker)

    # ── Docker check ──

    def _check_docker(self):
        def _check():
            container = self.docker_container_var.get().strip()
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Status}}", container],
                    capture_output=True, text=True, timeout=10)
                status = result.stdout.strip()
                if status == "running":
                    self.root.after(0, lambda: self._set_docker_status(
                        "running", f"Container '{container}' is running", Theme.ACCENT_GREEN))
                elif status:
                    self.root.after(0, lambda: self._set_docker_status(
                        "stopped", f"Container '{container}' is {status}", Theme.ACCENT_YELLOW))
                else:
                    self.root.after(0, lambda: self._set_docker_status(
                        "missing", f"Container '{container}' not found", Theme.ACCENT_RED))
            except FileNotFoundError:
                self.root.after(0, lambda: self._set_docker_status(
                    "error", "Docker not installed", Theme.ACCENT_RED))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self._set_docker_status(
                    "error", "Docker timeout", Theme.ACCENT_RED))

        threading.Thread(target=_check, daemon=True).start()

    def _set_docker_status(self, state, text, color):
        self.docker_status_icon.config(fg=color)
        self.docker_status_label.config(text=text, fg=color)

    # ── DRP-AI Compile ──

    def _compile_model(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        if not self.model_onnx.exists():
            messagebox.showerror("Error",
                                 "qd_model.onnx not found.\n"
                                 "Train and export a model first.")
            return

        if not self.calibration_dir.exists() or not list(self.calibration_dir.glob("*.png")):
            messagebox.showerror("Error",
                                 "No calibration images found.\n"
                                 "Go to Dataset page and generate calibration.")
            return

        container = self.docker_container_var.get().strip()
        if not container:
            messagebox.showerror("Error", "Docker container name is empty.")
            return

        self.deploy_log.clear()
        self._log(self.deploy_log, "═" * 50 + "\n", "header")
        self._log(self.deploy_log, "  DRP-AI TVM Model Compilation\n", "header")
        self._log(self.deploy_log, "═" * 50 + "\n\n", "header")

        # Copy ONNX + calibration into container, then run compile_model.sh
        compile_script = self.board_app_dir / "compile_model.sh"
        if not compile_script.exists():
            messagebox.showerror("Error", f"compile_model.sh not found:\n{compile_script}")
            return

        def _do_compile():
            try:
                self.root.after(0, lambda: self._set_status("Compiling model...", Theme.ACCENT_ORANGE))

                # Ensure container is running
                self._log(self.deploy_log, "Starting Docker container...\n", "info")
                subprocess.run(["docker", "start", container],
                               capture_output=True, timeout=30)

                # Create working directory in container
                subprocess.run(
                    ["docker", "exec", container, "mkdir", "-p", "/quickdraw"],
                    capture_output=True, timeout=10)

                # Copy files into container
                self._log(self.deploy_log, "Copying ONNX model to container...\n")
                subprocess.run(
                    ["docker", "cp", str(self.model_onnx), f"{container}:/quickdraw/qd_model.onnx"],
                    capture_output=True, timeout=60)

                self._log(self.deploy_log, "Copying calibration images...\n")
                subprocess.run(
                    ["docker", "cp", str(self.calibration_dir), f"{container}:/quickdraw/calibration"],
                    capture_output=True, timeout=120)

                self._log(self.deploy_log, "Copying compile script...\n")
                subprocess.run(
                    ["docker", "cp", str(compile_script), f"{container}:/quickdraw/compile_model.sh"],
                    capture_output=True, timeout=30)

                # Run compilation
                self._log(self.deploy_log, "\nStarting DRP-AI TVM compilation...\n\n", "info")

                cmd = [
                    "docker", "exec", container, "bash", "/quickdraw/compile_model.sh",
                    "/quickdraw/qd_model.onnx",
                    "/quickdraw/calibration",
                    "qd_mobilenetv2",
                    str(self.num_calib_var.get()),
                ]

                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                self.process = process
                self.is_running = True

                for line in iter(process.stdout.readline, ""):
                    self.log_queue.put(("deploy", line))
                process.wait()

                if process.returncode == 0:
                    self._log(self.deploy_log, "\nCopying compiled model from container...\n", "info")

                    # Copy compiled model back
                    drpai_out = self.project_dir / "drpai_model"
                    if drpai_out.exists():
                        shutil.rmtree(drpai_out, ignore_errors=True)

                    subprocess.run(
                        ["docker", "cp",
                         f"{container}:/drp-ai_tvm/tutorials/qd_mobilenetv2",
                         str(drpai_out)],
                        capture_output=True, timeout=120)

                    self._log(self.deploy_log, f"\nCompiled model saved to: {drpai_out}\n", "success")
                    self.root.after(0, lambda: self._set_status("Compilation complete", Theme.ACCENT_GREEN))
                else:
                    self._log(self.deploy_log, f"\nCompilation FAILED (exit code {process.returncode})\n", "error")
                    self.root.after(0, lambda: self._set_status("Compilation failed", Theme.ACCENT_RED))

            except Exception as e:
                self._log(self.deploy_log, f"\nError: {e}\n", "error")
                self.root.after(0, lambda: self._set_status(f"Error: {e}", Theme.ACCENT_RED))
            finally:
                self.is_running = False
                self.process = None

        threading.Thread(target=_do_compile, daemon=True).start()

    # ── Build C++ App ──

    def _build_app(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        container = self.docker_container_var.get().strip()
        if not container:
            messagebox.showerror("Error", "Docker container name is empty.")
            return

        self.deploy_log.clear()
        self._log(self.deploy_log, "═" * 50 + "\n", "header")
        self._log(self.deploy_log, "  Building C++ Board App\n", "header")
        self._log(self.deploy_log, "═" * 50 + "\n\n", "header")

        def _do_build():
            try:
                self.root.after(0, lambda: self._set_status("Building C++ app...", Theme.ACCENT_BLUE))

                # Ensure container is running
                subprocess.run(["docker", "start", container],
                               capture_output=True, timeout=30)

                # Copy source into container
                self._log(self.deploy_log, "Copying source files to container...\n", "info")

                # Copy src directory
                for subdir in ["src"]:
                    src_path = self.board_app_dir / subdir
                    if src_path.exists():
                        subprocess.run(
                            ["docker", "cp", str(src_path),
                             f"{container}:/tmp/board_app/{subdir}"],
                            capture_output=True, timeout=30)

                # Copy CMakeLists.txt
                subprocess.run(
                    ["docker", "cp", str(self.board_app_dir / "CMakeLists.txt"),
                     f"{container}:/tmp/board_app/CMakeLists.txt"],
                    capture_output=True, timeout=10)

                # Build inside container
                self._log(self.deploy_log, "Running cmake + make...\n\n", "info")

                build_cmd = (
                    "cd /tmp/board_app && "
                    "unset LD_LIBRARY_PATH && "
                    "SDK_ENV=$(find /opt/ -maxdepth 3 -name 'environment-setup-*poky-linux' 2>/dev/null | head -1) && "
                    "if [ -n \"$SDK_ENV\" ]; then source \"$SDK_ENV\"; fi && "
                    "export TVM_ROOT=/drp-ai_tvm && "
                    "rm -rf build && mkdir -p build && cd build && "
                    "cmake -DAPP_NAME=app_quickdraw .. && "
                    "make -j$(nproc)"
                )

                cmd = ["docker", "exec", container, "bash", "-c", build_cmd]
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                self.process = process
                self.is_running = True

                for line in iter(process.stdout.readline, ""):
                    self.log_queue.put(("deploy", line))
                process.wait()

                if process.returncode == 0:
                    # Copy binary back
                    self._log(self.deploy_log, "\nCopying binary from container...\n", "info")

                    binary_dst = self.board_app_dir / "app_quickdraw"
                    subprocess.run(
                        ["docker", "cp",
                         f"{container}:/tmp/board_app/build/app_quickdraw",
                         str(binary_dst)],
                        capture_output=True, timeout=30)

                    if binary_dst.exists():
                        sz = binary_dst.stat().st_size / (1024 * 1024)
                        self._log(self.deploy_log, f"\nBuild SUCCESS: app_quickdraw ({sz:.1f} MB)\n", "success")
                        self.root.after(0, lambda: self._set_status("Build complete", Theme.ACCENT_GREEN))
                    else:
                        self._log(self.deploy_log, "\nBuild completed but binary not found\n", "error")
                else:
                    self._log(self.deploy_log, f"\nBuild FAILED (exit code {process.returncode})\n", "error")
                    self.root.after(0, lambda: self._set_status("Build failed", Theme.ACCENT_RED))

            except Exception as e:
                self._log(self.deploy_log, f"\nError: {e}\n", "error")
                self.root.after(0, lambda: self._set_status(f"Error: {e}", Theme.ACCENT_RED))
            finally:
                self.is_running = False
                self.process = None

        threading.Thread(target=_do_build, daemon=True).start()

    # ── Package Deploy ──

    def _package_deploy(self):
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        self.deploy_log.clear()
        self._log(self.deploy_log, "═" * 50 + "\n", "header")
        self._log(self.deploy_log, "  Packaging Deploy Folder\n", "header")
        self._log(self.deploy_log, "═" * 50 + "\n\n", "header")

        try:
            deploy = self.deploy_dir
            deploy.mkdir(parents=True, exist_ok=True)

            # 1. Binary
            binary_src = self.board_app_dir / "app_quickdraw"
            if binary_src.exists():
                shutil.copy2(binary_src, deploy / "app_quickdraw")
                self._log(self.deploy_log, "  [OK] app_quickdraw\n", "success")
            else:
                self._log(self.deploy_log, "  [MISSING] app_quickdraw — build first\n", "error")

            # 2. Model
            drpai_src = self.project_dir / "drpai_model"
            model_dst = deploy / "model" / "qd_mobilenetv2"
            if drpai_src.exists():
                if model_dst.exists():
                    shutil.rmtree(model_dst)
                model_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(drpai_src, model_dst)
                self._log(self.deploy_log, "  [OK] model/qd_mobilenetv2/\n", "success")
            else:
                self._log(self.deploy_log, "  [MISSING] drpai_model/ — compile first\n", "error")

            # 3. Runtime libraries
            lib_src = self.board_app_dir / "lib"
            lib_dst = deploy / "lib"
            if lib_src.exists():
                lib_dst.mkdir(exist_ok=True)
                n = 0
                for f in lib_src.iterdir():
                    if f.suffix in (".so",) or ".so." in f.name:
                        shutil.copy2(f, lib_dst / f.name)
                        n += 1
                self._log(self.deploy_log, f"  [OK] lib/ ({n} libraries)\n", "success")
            else:
                self._log(self.deploy_log, "  [WARN] lib/ not found\n", "warning")

            # 4. Python wheels
            wheels_src = self.board_app_dir / "wheels"
            wheels_dst = deploy / "wheels"
            if wheels_src.exists():
                wheels_dst.mkdir(exist_ok=True)
                n = 0
                for f in wheels_src.glob("*.whl"):
                    shutil.copy2(f, wheels_dst / f.name)
                    n += 1
                self._log(self.deploy_log, f"  [OK] wheels/ ({n} packages)\n", "success")

            # 5. GUI + config + labels + run.sh
            for fname in ["quickdraw_gui.py", "config.json", "labels.txt",
                          "run.sh", "solidrun_logo.png"]:
                src = self.board_app_dir / fname
                if src.exists():
                    shutil.copy2(src, deploy / fname)
                    self._log(self.deploy_log, f"  [OK] {fname}\n", "success")
                else:
                    self._log(self.deploy_log, f"  [SKIP] {fname} not found\n", "warning")

            # Make run.sh executable
            run_sh = deploy / "run.sh"
            if run_sh.exists():
                run_sh.chmod(0o755)

            # Summary
            total = sum(
                f.stat().st_size for f in deploy.rglob("*") if f.is_file()
            ) / (1024 * 1024)

            self._log(self.deploy_log, f"\n{'═' * 50}\n", "header")
            self._log(self.deploy_log, f"  Deploy folder ready: {deploy}\n", "success")
            self._log(self.deploy_log, f"  Total size: {total:.1f} MB\n", "success")
            self._log(self.deploy_log, f"{'═' * 50}\n\n", "header")
            self._log(self.deploy_log, "Copy to board:\n", "info")
            self._log(self.deploy_log, f"  scp -r {deploy} root@<board-ip>:/home/root/quickdraw\n")
            self._log(self.deploy_log, "  ssh root@<board-ip> 'cd /home/root/quickdraw && ./run.sh'\n")

            self._set_status("Deploy package ready", Theme.ACCENT_GREEN)

        except Exception as e:
            self._log(self.deploy_log, f"\nError: {e}\n", "error")
            self._set_status(f"Package error: {e}", Theme.ACCENT_RED)

    # ── Full Pipeline ──

    def _run_full_pipeline(self):
        """Run compile → build → package in sequence."""
        if self.is_running:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        # Validate prerequisites
        if not self.model_onnx.exists():
            messagebox.showerror("Error",
                                 "qd_model.onnx not found.\n"
                                 "Train and export a model first.")
            return

        if not self.calibration_dir.exists() or not list(self.calibration_dir.glob("*.png")):
            messagebox.showerror("Error",
                                 "No calibration images.\n"
                                 "Go to Dataset page and generate calibration.")
            return

        container = self.docker_container_var.get().strip()
        if not container:
            messagebox.showerror("Error", "Docker container name is empty.")
            return

        self.deploy_log.clear()
        self._log(self.deploy_log, "═" * 50 + "\n", "header")
        self._log(self.deploy_log, "  Full Deploy Pipeline\n", "header")
        self._log(self.deploy_log, "  Compile → Build → Package\n", "header")
        self._log(self.deploy_log, "═" * 50 + "\n\n", "header")

        compile_script = self.board_app_dir / "compile_model.sh"

        def _pipeline():
            try:
                self.is_running = True

                # ── Step 1: Compile Model ──
                self._log(self.deploy_log, "━" * 40 + "\n", "info")
                self._log(self.deploy_log, "  STEP 1/3: Compile DRP-AI Model\n", "info")
                self._log(self.deploy_log, "━" * 40 + "\n\n", "info")
                self.root.after(0, lambda: self._set_status("Step 1/3: Compiling model...", Theme.ACCENT_ORANGE))

                subprocess.run(["docker", "start", container], capture_output=True, timeout=30)
                subprocess.run(["docker", "exec", container, "mkdir", "-p", "/quickdraw"],
                               capture_output=True, timeout=10)

                # Copy files
                subprocess.run(["docker", "cp", str(self.model_onnx),
                                f"{container}:/quickdraw/qd_model.onnx"],
                               capture_output=True, timeout=60)
                subprocess.run(["docker", "cp", str(self.calibration_dir),
                                f"{container}:/quickdraw/calibration"],
                               capture_output=True, timeout=120)
                subprocess.run(["docker", "cp", str(compile_script),
                                f"{container}:/quickdraw/compile_model.sh"],
                               capture_output=True, timeout=30)

                # Run compile
                cmd = ["docker", "exec", container, "bash",
                       "/quickdraw/compile_model.sh",
                       "/quickdraw/qd_model.onnx", "/quickdraw/calibration",
                       "qd_mobilenetv2", str(self.num_calib_var.get())]

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                self.process = process
                for line in iter(process.stdout.readline, ""):
                    self.log_queue.put(("deploy", line))
                process.wait()

                if process.returncode != 0:
                    self._log(self.deploy_log, "\nCompilation FAILED — aborting pipeline\n", "error")
                    self.root.after(0, lambda: self._set_status("Pipeline failed at compile", Theme.ACCENT_RED))
                    return

                # Copy compiled model back
                drpai_out = self.project_dir / "drpai_model"
                if drpai_out.exists():
                    shutil.rmtree(drpai_out, ignore_errors=True)
                subprocess.run(["docker", "cp",
                                f"{container}:/drp-ai_tvm/tutorials/qd_mobilenetv2",
                                str(drpai_out)], capture_output=True, timeout=120)

                self._log(self.deploy_log, "\nCompilation done.\n\n", "success")

                # ── Step 2: Build C++ App ──
                self._log(self.deploy_log, "━" * 40 + "\n", "info")
                self._log(self.deploy_log, "  STEP 2/3: Build C++ App\n", "info")
                self._log(self.deploy_log, "━" * 40 + "\n\n", "info")
                self.root.after(0, lambda: self._set_status("Step 2/3: Building C++ app...", Theme.ACCENT_BLUE))

                # Copy source files
                for subdir in ["src"]:
                    src_path = self.board_app_dir / subdir
                    if src_path.exists():
                        subprocess.run(["docker", "cp", str(src_path),
                                        f"{container}:/tmp/board_app/{subdir}"],
                                       capture_output=True, timeout=30)

                subprocess.run(["docker", "cp",
                                str(self.board_app_dir / "CMakeLists.txt"),
                                f"{container}:/tmp/board_app/CMakeLists.txt"],
                               capture_output=True, timeout=10)

                build_cmd = (
                    "cd /tmp/board_app && "
                    "unset LD_LIBRARY_PATH && "
                    "SDK_ENV=$(find /opt/ -maxdepth 3 -name 'environment-setup-*poky-linux' 2>/dev/null | head -1) && "
                    "if [ -n \"$SDK_ENV\" ]; then source \"$SDK_ENV\"; fi && "
                    "export TVM_ROOT=/drp-ai_tvm && "
                    "rm -rf build && mkdir -p build && cd build && "
                    "cmake -DAPP_NAME=app_quickdraw .. && "
                    "make -j$(nproc)"
                )

                process = subprocess.Popen(
                    ["docker", "exec", container, "bash", "-c", build_cmd],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                self.process = process
                for line in iter(process.stdout.readline, ""):
                    self.log_queue.put(("deploy", line))
                process.wait()

                if process.returncode != 0:
                    self._log(self.deploy_log, "\nBuild FAILED — aborting pipeline\n", "error")
                    self.root.after(0, lambda: self._set_status("Pipeline failed at build", Theme.ACCENT_RED))
                    return

                # Copy binary back
                binary_dst = self.board_app_dir / "app_quickdraw"
                subprocess.run(["docker", "cp",
                                f"{container}:/tmp/board_app/build/app_quickdraw",
                                str(binary_dst)], capture_output=True, timeout=30)

                self._log(self.deploy_log, "\nBuild done.\n\n", "success")

                # ── Step 3: Package ──
                self._log(self.deploy_log, "━" * 40 + "\n", "info")
                self._log(self.deploy_log, "  STEP 3/3: Package Deploy\n", "info")
                self._log(self.deploy_log, "━" * 40 + "\n\n", "info")
                self.root.after(0, lambda: self._set_status("Step 3/3: Packaging...", Theme.ACCENT_GREEN))

                # We must run package on main thread because it uses _log directly
                # So we schedule it
                self.is_running = False
                self.process = None
                self.root.after(0, self._package_deploy)

                self._log(self.deploy_log, "\n" + "═" * 50 + "\n", "header")
                self._log(self.deploy_log, "  PIPELINE COMPLETE\n", "success")
                self._log(self.deploy_log, "═" * 50 + "\n", "header")

            except Exception as e:
                self._log(self.deploy_log, f"\nPipeline error: {e}\n", "error")
                self.root.after(0, lambda: self._set_status(f"Pipeline error: {e}", Theme.ACCENT_RED))
            finally:
                self.is_running = False
                self.process = None

        threading.Thread(target=_pipeline, daemon=True).start()

    # ═══════════════════════════════════════════════════════
    #  Process Runner (shared)
    # ═══════════════════════════════════════════════════════

    def _run_process(self, cmd, log_viewer: LogViewer, cwd: str = None,
                     on_line=None, on_done=None):
        """Run a subprocess, streaming output to a log viewer."""
        self.is_running = True
        self._set_status(f"Running: {os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0])}...",
                         Theme.ACCENT_CYAN)

        def _target():
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=cwd
                )
                self.process = process

                for line in iter(process.stdout.readline, ""):
                    # Determine which log viewer to use
                    if log_viewer is self.dataset_log:
                        tag = "dataset"
                    elif log_viewer is self.training_log:
                        tag = "training"
                    else:
                        tag = "deploy"
                    self.log_queue.put((tag, line))

                    if on_line:
                        on_line(line)

                process.wait()

                if process.returncode == 0:
                    self.log_queue.put((tag, "\n[DONE] Process completed successfully.\n"))
                    self.root.after(0, lambda: self._set_status("Complete", Theme.ACCENT_GREEN))
                else:
                    self.log_queue.put((tag, f"\n[ERROR] Process exited with code {process.returncode}\n"))
                    self.root.after(0, lambda: self._set_status(
                        f"Failed (exit code {process.returncode})", Theme.ACCENT_RED))

            except Exception as e:
                self.log_queue.put((tag, f"\n[ERROR] {e}\n"))
                self.root.after(0, lambda: self._set_status(f"Error: {e}", Theme.ACCENT_RED))
            finally:
                self.is_running = False
                self.process = None
                if on_done:
                    self.root.after(0, on_done)

        threading.Thread(target=_target, daemon=True).start()

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._set_status("Process stopped", Theme.ACCENT_YELLOW)

    def _process_log_queue(self):
        """Process log queue items on main thread."""
        try:
            while True:
                tag, line = self.log_queue.get_nowait()

                if tag == "dataset":
                    viewer = self.dataset_log
                elif tag == "training":
                    viewer = self.training_log
                else:
                    viewer = self.deploy_log

                # Color-code lines
                line_lower = line.lower()
                if "[ok]" in line_lower or "[pass]" in line_lower or "success" in line_lower:
                    viewer.append(line, "success")
                elif "[error]" in line_lower or "[fail]" in line_lower or "error" in line_lower:
                    viewer.append(line, "error")
                elif "[warn]" in line_lower or "warning" in line_lower:
                    viewer.append(line, "warning")
                elif "[done]" in line_lower:
                    viewer.append(line, "success")
                elif line.startswith("═") or line.startswith("━"):
                    viewer.append(line, "header")
                else:
                    viewer.append(line)

        except queue.Empty:
            pass

        self.root.after(50, self._process_log_queue)

    def _log(self, viewer: LogViewer, text: str, tag: str = ""):
        """Thread-safe log to viewer."""
        if tag == "dataset":
            tag_key = "dataset"
        elif viewer is self.training_log:
            tag_key = "training"
        elif viewer is self.deploy_log:
            tag_key = "deploy"
        else:
            tag_key = "deploy"

        # If called from main thread, write directly
        try:
            viewer.append(text, tag)
        except RuntimeError:
            self.log_queue.put((tag_key, text))


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()

    # Set icon if available
    icon_path = Path(__file__).parent / "solidrun_logo.png"
    if icon_path.exists():
        try:
            from PIL import Image, ImageTk
            icon = Image.open(icon_path)
            photo = ImageTk.PhotoImage(icon)
            root.iconphoto(True, photo)
        except Exception:
            pass

    app = QuickDrawStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
