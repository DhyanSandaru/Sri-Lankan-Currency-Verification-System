"""
app.py
------
Desktop UI for Sri Lankan Currency Verification System.

Tab 1 — Verify Note
  User uploads front + back images, clicks Verify.
  Shows verdict, denomination, per-side error bars.
  After verification completes, a "View Pipeline" button appears
  to switch to Tab 2.

Tab 2 — Pipeline Viewer
  Automatically populated after every verification.
  Shows a side selector (Front / Back) so the user can inspect
  the preprocessing transformation of either image that was just verified.
  No separate file inputs — it purely reflects the last verification.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))

from inference import CurrencyVerifier, VERDICT_GENUINE, VERDICT_SUSPICIOUS, VERDICT_ERROR
from preprocessing import (
    _to_grayscale, _blur, _adaptive_threshold, _morphology,
    _find_note_contour, _perspective_warp, _apply_clahe_hsv,
)
from autoencoder import MODEL_H, MODEL_W

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_TITLE     = "LK Currency Verifier"
WINDOW_W      = 1200
WINDOW_H      = 830
MODELS_DIR    = "models"
DENOMINATIONS = ["100", "500", "1000"]
STEP_DELAY_MS = 500

CLR_BG         = "#1e1e2e"
CLR_SURFACE    = "#2a2a3d"
CLR_SURFACE2   = "#252538"
CLR_ACCENT     = "#7c6af7"
CLR_ACCENT2    = "#5a4fd0"
CLR_GENUINE    = "#4caf50"
CLR_SUSPICIOUS = "#f44336"
CLR_ERROR      = "#ff9800"
CLR_TEXT       = "#e0e0e0"
CLR_SUBTEXT    = "#9090a0"
CLR_BORDER     = "#3d3d55"
CLR_HIGHLIGHT  = "#3d3a6e"
CLR_STEP_DONE  = "#4caf50"
CLR_STEP_ACT   = "#7c6af7"
CLR_STEP_WAIT  = "#3d3d55"

PIPELINE_STEPS = [
    ("Original",           "Input image loaded from disk"),
    ("Grayscale",          "Convert to single-channel luminance"),
    ("Gaussian Blur",      "Suppress noise with 5×5 Gaussian kernel"),
    ("Adaptive Threshold", "Binary mask — handles uneven illumination"),
    ("Morphology",         "Close holes, dilate edges to strengthen boundary"),
    ("Contour Detection",  "Note boundary detected and highlighted"),
    ("Perspective Warp",   "Correct camera angle → flat 1024×512 template"),
    ("HSV + CLAHE",        "Normalise brightness, preserve hue for detection"),
]


# ---------------------------------------------------------------------------
# Pipeline computation
# ---------------------------------------------------------------------------

def compute_pipeline_steps(image_path: str) -> List[Tuple[str, np.ndarray, str]]:
    """
    Run the full preprocessing pipeline on one image, capturing every
    intermediate stage as a BGR numpy array.
    Returns list of (step_name, bgr_image, description).
    """
    steps = []

    def add(name, img, desc):
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        steps.append((name, bgr, desc))

    original = cv2.imread(image_path)
    if original is None:
        return []

    add("Original", original, "Input image loaded from disk")

    gray = _to_grayscale(original)
    add("Grayscale", gray, "Convert to single-channel luminance")

    blurred = _blur(gray)
    add("Gaussian Blur", blurred, "Suppress noise with 5×5 Gaussian kernel")

    thresh = _adaptive_threshold(blurred)
    add("Adaptive Threshold", thresh, "Binary mask — handles uneven illumination")

    morph = _morphology(thresh)
    add("Morphology", morph, "Close holes, dilate edges to strengthen boundary")

    pts = _find_note_contour(morph)
    contour_vis = original.copy()
    if pts is not None:
        cv2.polylines(contour_vis, [pts.astype(np.int32)], True, (0, 220, 80), 3)
        for pt in pts.astype(np.int32):
            cv2.circle(contour_vis, tuple(pt), 8, (255, 80, 80), -1)
    else:
        h, w = original.shape[:2]
        pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
        cv2.putText(contour_vis, "Full frame used (no corner detected)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
    add("Contour Detection", contour_vis, "Note boundary detected and highlighted")

    warped = _perspective_warp(original, pts)
    add("Perspective Warp", warped, "Correct camera angle → flat 1024×512 template")

    normalised = _apply_clahe_hsv(warped)
    add("HSV + CLAHE", normalised, "Normalise brightness, preserve hue for detection")

    return steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_photo(bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class CurrencyVerifierApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.resizable(True, True)
        self.configure(bg=CLR_BG)
        self.minsize(960, 720)

        self.front_path      = tk.StringVar(value="")
        self.back_path       = tk.StringVar(value="")
        self.denom_var       = tk.StringVar(value=DENOMINATIONS[0])
        self.auto_detect_var = tk.BooleanVar(value=True)
        self.verifier: Optional[CurrencyVerifier] = None
        self._refs           = {}   # PhotoImage GC guard

        # Pipeline state — populated after verification, not by user input
        self._front_steps: List[Tuple] = []
        self._back_steps:  List[Tuple] = []
        self._active_side  = tk.StringVar(value="front")
        self._current_step = -1
        self._animating    = False

        self._build_ui()
        self._load_verifier_async()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_verifier_async(self):
        def _load():
            try:
                self.verifier = CurrencyVerifier(
                    models_dir=MODELS_DIR, device_str="auto", verbose=False)
                self.after(0, lambda: self._set_status("Models loaded. Ready to verify."))
            except Exception as e:
                self.verifier = None
                self.after(0, lambda: self._set_status(f"⚠ Models not loaded: {e}", CLR_ERROR))
        threading.Thread(target=_load, daemon=True).start()

    # -----------------------------------------------------------------------
    # UI skeleton
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=CLR_SURFACE, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🇱🇰  Sri Lankan Currency Verification System",
                 font=("Segoe UI", 16, "bold"), bg=CLR_SURFACE, fg=CLR_TEXT).pack()
        tk.Label(hdr,
                 text="Convolutional Autoencoder Anomaly Detection  |  Rs. 100 · 500 · 1000",
                 font=("Segoe UI", 9), bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack()

        # Notebook style
        s = ttk.Style()
        s.theme_use("default")
        s.configure("Dark.TNotebook", background=CLR_BG, borderwidth=0, tabmargins=0)
        s.configure("Dark.TNotebook.Tab",
                    background=CLR_SURFACE, foreground=CLR_SUBTEXT,
                    font=("Segoe UI", 10, "bold"), padding=[20, 8])
        s.map("Dark.TNotebook.Tab",
              background=[("selected", CLR_ACCENT)],
              foreground=[("selected", "white")])

        self.nb = ttk.Notebook(self, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True)

        tab1 = tk.Frame(self.nb, bg=CLR_BG)
        self.nb.add(tab1, text="  🔍  Verify Note  ")
        self._build_verify_tab(tab1)

        tab2 = tk.Frame(self.nb, bg=CLR_BG)
        self.nb.add(tab2, text="  🔬  Pipeline Viewer  ")
        self._build_pipeline_tab(tab2)

        # Status bar
        sb = tk.Frame(self, bg=CLR_SURFACE2, pady=4)
        sb.pack(fill="x", side="bottom")
        self.status_lbl = tk.Label(sb, text="Starting…",
                                    font=("Segoe UI", 9),
                                    bg=CLR_SURFACE2, fg=CLR_SUBTEXT)
        self.status_lbl.pack(side="left", padx=12)

    def _set_status(self, msg: str, color: str = CLR_SUBTEXT):
        self.status_lbl.configure(text=msg, fg=color)

    # -----------------------------------------------------------------------
    # TAB 1 — Verify
    # -----------------------------------------------------------------------

    def _build_verify_tab(self, parent):
        body = tk.Frame(parent, bg=CLR_BG)
        body.pack(fill="both", expand=True, padx=18, pady=14)

        # Left — controls
        left = tk.Frame(body, bg=CLR_SURFACE, width=290, padx=16, pady=16)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)
        self._build_controls(left)

        # Right — previews + results
        right = tk.Frame(body, bg=CLR_BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_preview_row(right)
        self._build_result_panel(right)

    def _build_controls(self, parent):
        tk.Label(parent, text="DENOMINATION", font=("Segoe UI", 8, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w")

        tk.Checkbutton(parent, text="Auto-detect",
                       variable=self.auto_detect_var,
                       command=self._toggle_denom,
                       bg=CLR_SURFACE, fg=CLR_TEXT,
                       selectcolor=CLR_BG, activebackground=CLR_SURFACE,
                       font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 2))

        self.denom_frame = tk.Frame(parent, bg=CLR_SURFACE)
        self.denom_frame.pack(fill="x", pady=(0, 12))
        for d in DENOMINATIONS:
            tk.Radiobutton(self.denom_frame, text=f"Rs. {d}",
                           variable=self.denom_var, value=d,
                           bg=CLR_SURFACE, fg=CLR_TEXT,
                           selectcolor=CLR_BG, activebackground=CLR_SURFACE,
                           font=("Segoe UI", 10)).pack(anchor="w", pady=1)
        self._toggle_denom()

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(parent, text="FRONT OF NOTE", font=("Segoe UI", 8, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w")
        self._file_picker(parent, self.front_path, "front")

        tk.Label(parent, text="BACK OF NOTE", font=("Segoe UI", 8, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w", pady=(10, 0))
        self._file_picker(parent, self.back_path, "back")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=12)

        self.verify_btn = tk.Button(
            parent, text="🔍  Verify Note",
            font=("Segoe UI", 12, "bold"),
            bg=CLR_ACCENT, fg="white",
            activebackground=CLR_ACCENT2, activeforeground="white",
            relief="flat", padx=10, pady=9, cursor="hand2",
            command=self._on_verify)
        self.verify_btn.pack(fill="x", pady=(0, 6))

        # "View Pipeline" — hidden until first verification completes
        self.view_pipeline_btn = tk.Button(
            parent, text="🔬  View Pipeline →",
            font=("Segoe UI", 10),
            bg=CLR_SURFACE2, fg=CLR_SUBTEXT,
            activebackground=CLR_HIGHLIGHT, activeforeground=CLR_TEXT,
            relief="flat", padx=10, pady=6, cursor="hand2",
            command=self._go_to_pipeline)
        # Start hidden — shown after first verification
        self.view_pipeline_btn.pack(fill="x")
        self.view_pipeline_btn.pack_forget()

    def _file_picker(self, parent, var: tk.StringVar, side_key: str):
        frame = tk.Frame(parent, bg=CLR_SURFACE)
        frame.pack(fill="x", pady=(4, 10))

        lbl = tk.Label(frame, text="No file selected",
                       font=("Segoe UI", 8), bg=CLR_BG, fg=CLR_SUBTEXT,
                       anchor="w", padx=6, pady=3, wraplength=240, justify="left")
        lbl.pack(fill="x")

        def browse():
            path = filedialog.askopenfilename(
                title=f"Select {side_key} image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")])
            if path:
                var.set(path)
                lbl.configure(text=os.path.basename(path))
                self._update_verify_preview(path, side_key)

        tk.Button(frame, text=f"Browse {side_key.capitalize()} Image",
                  font=("Segoe UI", 9),
                  bg=CLR_BORDER, fg=CLR_TEXT,
                  activebackground=CLR_ACCENT, activeforeground="white",
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=browse).pack(anchor="w", pady=(4, 0))

    def _build_preview_row(self, parent):
        row = tk.Frame(parent, bg=CLR_BG)
        row.pack(fill="x", pady=(0, 10))
        for label, attr in [("Front", "front_canvas"), ("Back", "back_canvas")]:
            col = tk.Frame(row, bg=CLR_SURFACE, padx=8, pady=8)
            col.pack(side="left", fill="both", expand=True,
                     padx=(0, 8) if label == "Front" else 0)
            tk.Label(col, text=label.upper(), font=("Segoe UI", 8, "bold"),
                     bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack()
            c = tk.Canvas(col, width=380, height=185,
                          bg=CLR_BG, highlightthickness=1,
                          highlightbackground=CLR_BORDER)
            c.pack(pady=(4, 0))
            c.create_text(190, 92, text="No image loaded",
                          fill=CLR_SUBTEXT, font=("Segoe UI", 10))
            setattr(self, attr, c)

    def _update_verify_preview(self, path: str, side: str):
        try:
            img = cv2.imread(path)
            if img is None:
                return
            photo = to_photo(img, 380, 185)
            canvas = self.front_canvas if side == "front" else self.back_canvas
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            self._refs[f"prev_{side}"] = photo
        except Exception:
            pass

    def _build_result_panel(self, parent):
        rf = tk.Frame(parent, bg=CLR_SURFACE, padx=16, pady=12)
        rf.pack(fill="both", expand=True)

        tk.Label(rf, text="VERIFICATION RESULT", font=("Segoe UI", 8, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w")

        self.verdict_lbl = tk.Label(rf, text="—",
                                     font=("Segoe UI", 26, "bold"),
                                     bg=CLR_SURFACE, fg=CLR_SUBTEXT)
        self.verdict_lbl.pack(pady=(6, 2))

        self.denom_result_lbl = tk.Label(rf, text="",
                                          font=("Segoe UI", 11),
                                          bg=CLR_SURFACE, fg=CLR_SUBTEXT)
        self.denom_result_lbl.pack()

        bars = tk.Frame(rf, bg=CLR_SURFACE)
        bars.pack(fill="x", pady=10)
        self.front_bar = self._make_error_bar(bars, "Front")
        self.front_bar.pack(fill="x", pady=(0, 5))
        self.back_bar = self._make_error_bar(bars, "Back")
        self.back_bar.pack(fill="x")

        self.detail_txt = tk.Text(rf, height=4, font=("Courier New", 9),
                                   bg=CLR_BG, fg=CLR_SUBTEXT, relief="flat",
                                   state="disabled", padx=8, pady=5)
        self.detail_txt.pack(fill="x", pady=(8, 0))

    def _make_error_bar(self, parent, label):
        f = tk.Frame(parent, bg=CLR_SURFACE)
        tk.Label(f, text=f"{label}:", width=6, anchor="w",
                 font=("Segoe UI", 9), bg=CLR_SURFACE, fg=CLR_TEXT).pack(side="left")
        bg = tk.Frame(f, bg=CLR_BG, height=13, width=280)
        bg.pack(side="left", padx=(4, 8))
        bg.pack_propagate(False)
        fill = tk.Frame(bg, bg=CLR_ACCENT, height=13, width=0)
        fill.place(x=0, y=0, relheight=1, width=0)
        val = tk.Label(f, text="—", font=("Segoe UI", 9),
                       bg=CLR_SURFACE, fg=CLR_TEXT, width=28, anchor="w")
        val.pack(side="left")
        f._fill = fill
        f._val  = val
        return f

    def _update_error_bar(self, bar, error, threshold, suspicious):
        ratio = min(error / (threshold * 2 + 1e-8), 1.0)
        color = CLR_SUSPICIOUS if suspicious else CLR_GENUINE
        bar._fill.place(width=int(280 * ratio))
        bar._fill.configure(bg=color)
        flag = "⚠" if suspicious else "✓"
        bar._val.configure(text=f"{flag}  {error:.5f}  /  {threshold:.5f}", fg=color)

    def _reset_result(self):
        self.verdict_lbl.configure(text="—", fg=CLR_SUBTEXT)
        self.denom_result_lbl.configure(text="")
        for bar in [self.front_bar, self.back_bar]:
            bar._fill.place(width=0)
            bar._val.configure(text="—", fg=CLR_TEXT)
        self._set_detail("")

    def _set_detail(self, text):
        self.detail_txt.configure(state="normal")
        self.detail_txt.delete("1.0", "end")
        self.detail_txt.insert("end", text)
        self.detail_txt.configure(state="disabled")

    def _toggle_denom(self):
        st = "disabled" if self.auto_detect_var.get() else "normal"
        for w in self.denom_frame.winfo_children():
            w.configure(state=st)

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    def _on_verify(self):
        front = self.front_path.get()
        back  = self.back_path.get()

        if not front or not os.path.exists(front):
            messagebox.showwarning("Missing Image", "Please select a front image.")
            return
        if not back or not os.path.exists(back):
            messagebox.showwarning("Missing Image", "Please select a back image.")
            return
        if self.verifier is None:
            messagebox.showerror("Models Not Loaded",
                                 "Trained models not found.\n"
                                 "Run train.py and calibrate.py first.")
            return

        self._reset_result()
        self.verify_btn.configure(state="disabled")
        self.view_pipeline_btn.pack_forget()
        self._set_status("Verifying…")

        denom_override = None if self.auto_detect_var.get() else self.denom_var.get()

        def _run():
            # 1. Compute pipeline steps for both images (used by Pipeline tab)
            front_steps = compute_pipeline_steps(front)
            back_steps  = compute_pipeline_steps(back)

            # 2. Run actual verification
            result = self.verifier.verify(front, back, denomination=denom_override)

            self.after(0, lambda: self._on_verify_done(result, front_steps, back_steps))

        threading.Thread(target=_run, daemon=True).start()

    def _on_verify_done(self, result, front_steps, back_steps):
        self.verify_btn.configure(state="normal")

        # Store pipeline data for the viewer tab
        self._front_steps = front_steps
        self._back_steps  = back_steps

        if result.verdict == VERDICT_ERROR:
            self.verdict_lbl.configure(text="ERROR", fg=CLR_ERROR)
            self._set_status(f"Error: {result.error_message}", CLR_ERROR)
            self._set_detail(result.error_message or "")
            return

        # Verdict
        color = CLR_GENUINE if result.verdict == VERDICT_GENUINE else CLR_SUSPICIOUS
        icon  = "✓  GENUINE" if result.verdict == VERDICT_GENUINE else "⚠  SUSPICIOUS"
        self.verdict_lbl.configure(text=icon, fg=color)
        self.denom_result_lbl.configure(
            text=f"Rs. {result.denomination}   |   {result.processing_time_ms:.0f} ms",
            fg=CLR_SUBTEXT)

        if result.front_error is not None:
            self._update_error_bar(self.front_bar,
                                   result.front_error, result.front_threshold,
                                   result.front_suspicious)
        if result.back_error is not None:
            self._update_error_bar(self.back_bar,
                                   result.back_error, result.back_threshold,
                                   result.back_suspicious)

        lines = [
            f"Denomination  : Rs. {result.denomination}",
            f"Front error   : {result.front_error:.6f}  (threshold: {result.front_threshold:.6f})",
            f"Back  error   : {result.back_error:.6f}  (threshold: {result.back_threshold:.6f})",
        ]
        if result.denomination_scores:
            sc = "  ".join(f"{k}:{v:.3f}" for k, v in result.denomination_scores.items())
            lines.append(f"Denom scores  : {sc}")
        self._set_detail("\n".join(lines))
        self._set_status("Verification complete.")

        # Reveal the "View Pipeline" button now that data is ready
        self.view_pipeline_btn.pack(fill="x")

        # Pre-load pipeline viewer with front image, but don't switch tabs yet
        self._load_pipeline_viewer(side="front", auto_play=False)

    def _go_to_pipeline(self):
        """Switch to pipeline tab and auto-play the animation."""
        self._load_pipeline_viewer(side="front", auto_play=True)
        self.nb.select(1)

    # -----------------------------------------------------------------------
    # TAB 2 — Pipeline Viewer
    # -----------------------------------------------------------------------

    def _build_pipeline_tab(self, parent):
        # ── Top bar ─────────────────────────────────────────────────────────
        top = tk.Frame(parent, bg=CLR_SURFACE, padx=16, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="VIEWING PIPELINE FOR:",
                 font=("Segoe UI", 9, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(side="left", padx=(0, 10))

        # Side toggle — Front / Back
        for side_val, side_lbl in [("front", "Front Image"), ("back", "Back Image")]:
            tk.Radiobutton(
                top, text=side_lbl,
                variable=self._active_side, value=side_val,
                command=lambda s=side_val: self._switch_side(s),
                bg=CLR_SURFACE, fg=CLR_TEXT,
                selectcolor=CLR_ACCENT,
                activebackground=CLR_SURFACE,
                font=("Segoe UI", 10, "bold"),
                indicatoron=0,
                padx=12, pady=5,
                relief="flat", cursor="hand2",
                width=14
            ).pack(side="left", padx=3)

        # Playback controls
        self.pl_replay_btn = tk.Button(
            top, text="⟳  Replay",
            font=("Segoe UI", 9),
            bg=CLR_SURFACE2, fg=CLR_SUBTEXT,
            activebackground=CLR_HIGHLIGHT, activeforeground=CLR_TEXT,
            relief="flat", padx=10, pady=5, cursor="hand2",
            state="disabled",
            command=self._replay)
        self.pl_replay_btn.pack(side="left", padx=(16, 4))

        self.pl_back_btn = tk.Button(
            top, text="◀",
            font=("Segoe UI", 10, "bold"),
            bg=CLR_SURFACE2, fg=CLR_TEXT,
            activebackground=CLR_HIGHLIGHT,
            relief="flat", padx=10, pady=5, cursor="hand2",
            state="disabled",
            command=self._step_back)
        self.pl_back_btn.pack(side="left", padx=2)

        self.pl_fwd_btn = tk.Button(
            top, text="▶",
            font=("Segoe UI", 10, "bold"),
            bg=CLR_SURFACE2, fg=CLR_TEXT,
            activebackground=CLR_HIGHLIGHT,
            relief="flat", padx=10, pady=5, cursor="hand2",
            state="disabled",
            command=self._step_forward)
        self.pl_fwd_btn.pack(side="left", padx=2)

        # Waiting label — shown until first verification
        self.pl_waiting_lbl = tk.Label(
            top, text="Run a verification first to populate this view.",
            font=("Segoe UI", 9, "italic"),
            bg=CLR_SURFACE, fg=CLR_ERROR, padx=16)
        self.pl_waiting_lbl.pack(side="left")

        # ── Progress dots ────────────────────────────────────────────────────
        prog = tk.Frame(parent, bg=CLR_BG, pady=8)
        prog.pack(fill="x", padx=16)
        self._indicators = []
        for i, (name, _) in enumerate(PIPELINE_STEPS):
            col = tk.Frame(prog, bg=CLR_BG)
            col.pack(side="left", padx=3)
            dot = tk.Canvas(col, width=14, height=14, bg=CLR_BG, highlightthickness=0)
            dot.pack()
            dot.create_oval(2, 2, 12, 12, fill=CLR_STEP_WAIT, outline="")
            lbl = tk.Label(col, text=name, font=("Segoe UI", 7),
                           bg=CLR_BG, fg=CLR_SUBTEXT, wraplength=75, justify="center")
            lbl.pack()
            if i < len(PIPELINE_STEPS) - 1:
                tk.Label(prog, text="→", font=("Segoe UI", 10),
                         bg=CLR_BG, fg=CLR_BORDER).pack(side="left")
            self._indicators.append((dot, lbl))

        # ── Main display ─────────────────────────────────────────────────────
        display = tk.Frame(parent, bg=CLR_BG)
        display.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # Large step viewer (left)
        left_col = tk.Frame(display, bg=CLR_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.pl_step_title = tk.Label(left_col, text="",
                                       font=("Segoe UI", 15, "bold"),
                                       bg=CLR_BG, fg=CLR_TEXT)
        self.pl_step_title.pack(anchor="w")

        self.pl_step_desc = tk.Label(left_col, text="",
                                      font=("Segoe UI", 10),
                                      bg=CLR_BG, fg=CLR_SUBTEXT)
        self.pl_step_desc.pack(anchor="w", pady=(2, 8))

        self.pl_main_canvas = tk.Canvas(left_col, bg=CLR_SURFACE,
                                         highlightthickness=1,
                                         highlightbackground=CLR_BORDER)
        self.pl_main_canvas.pack(fill="both", expand=True)
        self._draw_waiting_message()

        # Thumbnail strip (right)
        right_col = tk.Frame(display, bg=CLR_SURFACE, width=210, padx=8, pady=8)
        right_col.pack(side="left", fill="y")
        right_col.pack_propagate(False)

        tk.Label(right_col, text="ALL STEPS", font=("Segoe UI", 8, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w", pady=(0, 6))

        thumb_wrap = tk.Frame(right_col, bg=CLR_SURFACE)
        thumb_wrap.pack(fill="both", expand=True)

        self.thumb_canvas = tk.Canvas(thumb_wrap, bg=CLR_SURFACE,
                                       highlightthickness=0, width=190)
        vsb = ttk.Scrollbar(thumb_wrap, orient="vertical",
                             command=self.thumb_canvas.yview)
        self.thumb_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.thumb_canvas.pack(side="left", fill="both", expand=True)

        self.thumb_inner = tk.Frame(self.thumb_canvas, bg=CLR_SURFACE)
        self.thumb_canvas.create_window(0, 0, anchor="nw", window=self.thumb_inner)
        self.thumb_inner.bind("<Configure>",
            lambda e: self.thumb_canvas.configure(
                scrollregion=self.thumb_canvas.bbox("all")))

        self._thumb_frames = []
        self._thumb_photos = []

    def _draw_waiting_message(self):
        self.pl_main_canvas.delete("all")
        self.pl_main_canvas.create_text(
            300, 200,
            text="Verify a note first.\nThe preprocessing pipeline\nwill appear here automatically.",
            fill=CLR_SUBTEXT, font=("Segoe UI", 13), justify="center")

    # -----------------------------------------------------------------------
    # Pipeline viewer — data loading & playback
    # -----------------------------------------------------------------------

    def _load_pipeline_viewer(self, side: str = "front", auto_play: bool = True):
        """
        Load pipeline steps into the viewer.
        Called automatically after verification — never by user directly.
        """
        self._animating    = False
        self._current_step = -1
        self._active_side.set(side)

        steps = self._front_steps if side == "front" else self._back_steps
        if not steps:
            return

        # Hide the waiting label once data arrives
        self.pl_waiting_lbl.pack_forget()

        # Reset progress dots
        for dot, lbl in self._indicators:
            dot.itemconfig(1, fill=CLR_STEP_WAIT)
            lbl.configure(fg=CLR_SUBTEXT, font=("Segoe UI", 7))

        # Clear titles
        self.pl_step_title.configure(text="")
        self.pl_step_desc.configure(text="")

        # Rebuild thumbnails
        self._build_thumbnails(steps)

        # Enable controls
        self.pl_replay_btn.configure(state="normal")

        if auto_play:
            self._animating = True
            self._animate_next()

    def _switch_side(self, side: str):
        """User toggled Front / Back radio button."""
        steps = self._front_steps if side == "front" else self._back_steps
        if not steps:
            return
        self._animating    = False
        self._current_step = -1
        self._build_thumbnails(steps)
        for dot, lbl in self._indicators:
            dot.itemconfig(1, fill=CLR_STEP_WAIT)
            lbl.configure(fg=CLR_SUBTEXT, font=("Segoe UI", 7))
        self.pl_step_title.configure(text="")
        self.pl_step_desc.configure(text="")
        self.pl_main_canvas.delete("all")
        self.pl_back_btn.configure(state="disabled")
        self.pl_fwd_btn.configure(state="disabled")
        # Auto-play the new side
        self._animating = True
        self._animate_next()

    def _build_thumbnails(self, steps):
        for f in self._thumb_frames:
            f.destroy()
        self._thumb_frames = []
        self._thumb_photos = []

        TW, TH = 170, 88

        for i, (name, bgr, desc) in enumerate(steps):
            photo = to_photo(bgr, TW, TH)
            self._thumb_photos.append(photo)

            frame = tk.Frame(self.thumb_inner, bg=CLR_BG,
                              highlightthickness=1,
                              highlightbackground=CLR_BORDER,
                              cursor="hand2")
            frame.pack(fill="x", pady=3, padx=2)

            badge_row = tk.Frame(frame, bg=CLR_BG)
            badge_row.pack(fill="x", padx=4, pady=(4, 2))

            badge = tk.Label(badge_row, text=str(i+1),
                              font=("Segoe UI", 7, "bold"),
                              bg=CLR_STEP_WAIT, fg="white", width=2, padx=3)
            badge.pack(side="left")
            tk.Label(badge_row, text=name,
                     font=("Segoe UI", 7, "bold"),
                     bg=CLR_BG, fg=CLR_SUBTEXT, anchor="w").pack(side="left", padx=4)

            c = tk.Canvas(frame, width=TW, height=TH, bg=CLR_SURFACE, highlightthickness=0)
            c.pack(padx=4, pady=(0, 4))
            c.create_image(0, 0, anchor="nw", image=photo)

            idx = i
            def jump(e, n=idx):
                self._animating = False
                self._show_step(n)
            frame.bind("<Button-1>", jump)
            c.bind("<Button-1>", jump)

            frame._badge = badge
            self._thumb_frames.append(frame)

        self.thumb_canvas.update_idletasks()
        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))

    def _animate_next(self):
        if not self._animating:
            return
        nxt = self._current_step + 1
        steps = self._front_steps if self._active_side.get() == "front" else self._back_steps
        if nxt >= len(steps):
            self._animating = False
            self.pl_replay_btn.configure(state="normal")
            return
        self._show_step(nxt)
        self.after(STEP_DELAY_MS, self._animate_next)

    def _show_step(self, idx: int):
        side  = self._active_side.get()
        steps = self._front_steps if side == "front" else self._back_steps
        if not steps or idx < 0 or idx >= len(steps):
            return

        self._current_step = idx
        name, bgr, desc = steps[idx]

        # Progress dots
        for i, (dot, lbl) in enumerate(self._indicators):
            if i < idx:
                dot.itemconfig(1, fill=CLR_STEP_DONE)
                lbl.configure(fg=CLR_STEP_DONE, font=("Segoe UI", 7))
            elif i == idx:
                dot.itemconfig(1, fill=CLR_STEP_ACT)
                lbl.configure(fg=CLR_TEXT, font=("Segoe UI", 7, "bold"))
            else:
                dot.itemconfig(1, fill=CLR_STEP_WAIT)
                lbl.configure(fg=CLR_SUBTEXT, font=("Segoe UI", 7))

        # Thumbnail highlights
        for i, f in enumerate(self._thumb_frames):
            if i == idx:
                f.configure(highlightbackground=CLR_ACCENT, bg=CLR_HIGHLIGHT)
                f._badge.configure(bg=CLR_ACCENT)
                # Scroll into view
                self.after(10, lambda fi=f: self.thumb_canvas.yview_moveto(
                    max(0, fi.winfo_y() / max(self.thumb_inner.winfo_height(), 1))))
            elif i < idx:
                f.configure(highlightbackground=CLR_STEP_DONE, bg=CLR_BG)
                f._badge.configure(bg=CLR_STEP_DONE)
            else:
                f.configure(highlightbackground=CLR_BORDER, bg=CLR_BG)
                f._badge.configure(bg=CLR_STEP_WAIT)

        # Titles
        self.pl_step_title.configure(
            text=f"Step {idx+1} / {len(steps)}  —  {name}")
        self.pl_step_desc.configure(text=desc)

        # Main canvas image
        self.pl_main_canvas.update_idletasks()
        cw = self.pl_main_canvas.winfo_width()
        ch = self.pl_main_canvas.winfo_height()
        if cw < 20:
            cw, ch = 640, 360

        photo = to_photo(bgr, cw, ch)
        key = f"pl_{side}_{idx}"
        self._refs[key] = photo
        self.pl_main_canvas.delete("all")
        self.pl_main_canvas.create_image(0, 0, anchor="nw", image=photo)

        # Overlay badge
        self.pl_main_canvas.create_rectangle(0, 0, 160, 26, fill="#000000", outline="")
        self.pl_main_canvas.create_text(
            8, 13, anchor="w",
            text=f"  {idx+1}/{len(steps)}  {name}  [{side.upper()}]",
            fill="white", font=("Segoe UI", 9, "bold"))

        # Nav buttons
        self.pl_back_btn.configure(state="normal" if idx > 0 else "disabled")
        self.pl_fwd_btn.configure(
            state="normal" if idx < len(steps) - 1 else "disabled")

    def _step_back(self):
        self._animating = False
        self._show_step(self._current_step - 1)

    def _step_forward(self):
        self._animating = False
        self._show_step(self._current_step + 1)

    def _replay(self):
        steps = self._front_steps if self._active_side.get() == "front" else self._back_steps
        if not steps:
            return
        self._current_step = -1
        self._animating    = True
        self.pl_replay_btn.configure(state="disabled")
        self._animate_next()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow is required: pip install Pillow")
        sys.exit(1)

    app = CurrencyVerifierApp()
    app.mainloop()