"""
app.py
------
Desktop UI for Sri Lankan Currency Verification System.
Built with Tkinter (standard library) — no extra UI dependencies.

Workflow:
  1. User selects denomination (100 / 500 / 1000)
  2. User uploads front image
  3. User uploads back image
  4. Click "Verify Note"
  5. Results shown with colour-coded verdict, per-side error bars,
     and a reconstruction diff visualisation.
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from inference import CurrencyVerifier, VERDICT_GENUINE, VERDICT_SUSPICIOUS, VERDICT_ERROR
from preprocessing import preprocess_to_tensor
from autoencoder import MODEL_H, MODEL_W

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_TITLE      = "LK Currency Verifier"
WINDOW_SIZE    = "1100x750"
PREVIEW_W      = 400
PREVIEW_H      = 200
MODELS_DIR     = "models"
DENOMINATIONS  = ["100", "500", "1000"]

# Colours
CLR_BG         = "#1e1e2e"
CLR_SURFACE    = "#2a2a3d"
CLR_ACCENT     = "#7c6af7"
CLR_GENUINE    = "#4caf50"
CLR_SUSPICIOUS = "#f44336"
CLR_ERROR      = "#ff9800"
CLR_TEXT       = "#e0e0e0"
CLR_SUBTEXT    = "#9090a0"
CLR_BORDER     = "#3d3d55"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bgr_to_photoimage(bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb   = cv2.resize(rgb, (w, h))
    pil   = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


def make_diff_image(original_path: str, verifier: CurrencyVerifier,
                    denomination: str, side: str) -> np.ndarray:
    """
    Run the autoencoder and return a colourised absolute difference map
    so the user can see exactly where anomalies were detected.
    """
    import torch
    tensor = preprocess_to_tensor(original_path, target_size=(MODEL_H, MODEL_W))
    if tensor is None:
        return np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)

    t = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(verifier.device)
    model = verifier._get_model(denomination, side)

    with torch.no_grad():
        recon = model(t)

    orig_np  = t[0].permute(1, 2, 0).cpu().numpy()
    recon_np = recon[0].permute(1, 2, 0).cpu().numpy()

    diff = np.abs(orig_np - recon_np).mean(axis=2)  # (H, W)
    diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return diff_color


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class CurrencyVerifierApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.resizable(False, False)
        self.configure(bg=CLR_BG)

        # State
        self.front_path = tk.StringVar()
        self.back_path  = tk.StringVar()
        self.denom_var  = tk.StringVar(value=DENOMINATIONS[0])
        self.auto_detect_var = tk.BooleanVar(value=True)

        self.verifier: CurrencyVerifier = None
        self._load_verifier_async()

        self._build_ui()

    # ---- Verifier loading ----

    def _load_verifier_async(self):
        def _load():
            try:
                self.verifier = CurrencyVerifier(
                    models_dir=MODELS_DIR,
                    device_str="auto",
                    verbose=False
                )
            except Exception as e:
                self.verifier = None
                self._verifier_error = str(e)
        threading.Thread(target=_load, daemon=True).start()

    # ---- UI construction ----

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=CLR_SURFACE, pady=14)
        header.pack(fill="x")
        tk.Label(
            header, text="🇱🇰  Sri Lankan Currency Verification System",
            font=("Segoe UI", 16, "bold"),
            bg=CLR_SURFACE, fg=CLR_TEXT
        ).pack()
        tk.Label(
            header, text="Powered by Convolutional Autoencoder Anomaly Detection",
            font=("Segoe UI", 9), bg=CLR_SURFACE, fg=CLR_SUBTEXT
        ).pack()

        # ── Main content area ────────────────────────────────────────────────
        content = tk.Frame(self, bg=CLR_BG)
        content.pack(fill="both", expand=True, padx=20, pady=16)

        # Left panel — controls
        left = tk.Frame(content, bg=CLR_SURFACE, width=320, padx=16, pady=16)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._build_controls(left)

        # Right panel — previews + result
        right = tk.Frame(content, bg=CLR_BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_previews(right)
        self._build_result_panel(right)

    def _build_controls(self, parent):
        tk.Label(parent, text="DENOMINATION", font=("Segoe UI", 9, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w")

        auto_chk = tk.Checkbutton(
            parent, text="Auto-detect denomination",
            variable=self.auto_detect_var,
            command=self._toggle_denom,
            bg=CLR_SURFACE, fg=CLR_TEXT,
            selectcolor=CLR_BG, activebackground=CLR_SURFACE,
            font=("Segoe UI", 10)
        )
        auto_chk.pack(anchor="w", pady=(4, 0))

        self.denom_frame = tk.Frame(parent, bg=CLR_SURFACE)
        self.denom_frame.pack(fill="x", pady=(4, 16))

        for d in DENOMINATIONS:
            rb = tk.Radiobutton(
                self.denom_frame, text=f"Rs. {d}",
                variable=self.denom_var, value=d,
                bg=CLR_SURFACE, fg=CLR_TEXT,
                selectcolor=CLR_BG, activebackground=CLR_SURFACE,
                font=("Segoe UI", 11)
            )
            rb.pack(anchor="w", pady=1)

        self._toggle_denom()

        # Separator
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        # Front image
        tk.Label(parent, text="FRONT OF NOTE", font=("Segoe UI", 9, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w")
        self._file_picker(parent, self.front_path, "Browse Front Image", "front")

        # Back image
        tk.Label(parent, text="BACK OF NOTE", font=("Segoe UI", 9, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w", pady=(10, 0))
        self._file_picker(parent, self.back_path, "Browse Back Image", "back")

        # Separator
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=14)

        # Verify button
        self.verify_btn = tk.Button(
            parent, text="🔍  Verify Note",
            font=("Segoe UI", 13, "bold"),
            bg=CLR_ACCENT, fg="white",
            activebackground="#6050d0", activeforeground="white",
            relief="flat", padx=12, pady=10,
            cursor="hand2",
            command=self._on_verify
        )
        self.verify_btn.pack(fill="x", pady=(0, 8))

        self.status_label = tk.Label(
            parent, text="Ready. Load images to begin.",
            font=("Segoe UI", 9), bg=CLR_SURFACE, fg=CLR_SUBTEXT,
            wraplength=270, justify="center"
        )
        self.status_label.pack()

    def _file_picker(self, parent, var: tk.StringVar, btn_text: str, side_key: str):
        frame = tk.Frame(parent, bg=CLR_SURFACE)
        frame.pack(fill="x", pady=(4, 12))

        path_lbl = tk.Label(frame, textvariable=var, font=("Segoe UI", 8),
                             bg=CLR_BG, fg=CLR_SUBTEXT, anchor="w",
                             padx=6, pady=4, wraplength=250, justify="left")
        path_lbl.pack(fill="x")
        var.set("No file selected")

        def browse():
            path = filedialog.askopenfilename(
                title=f"Select {side_key} image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            if path:
                var.set(path)
                self._update_preview(path, side_key)

        tk.Button(
            frame, text=btn_text,
            font=("Segoe UI", 9),
            bg=CLR_BORDER, fg=CLR_TEXT,
            activebackground=CLR_ACCENT, activeforeground="white",
            relief="flat", padx=8, pady=5,
            cursor="hand2",
            command=browse
        ).pack(anchor="w", pady=(4, 0))

    def _build_previews(self, parent):
        pframe = tk.Frame(parent, bg=CLR_BG)
        pframe.pack(fill="x", pady=(0, 10))

        for label_text, attr in [("Front", "front_canvas"), ("Back", "back_canvas")]:
            col = tk.Frame(pframe, bg=CLR_SURFACE, padx=8, pady=8)
            col.pack(side="left", fill="both", expand=True, padx=(0, 8) if label_text == "Front" else 0)

            tk.Label(col, text=label_text.upper(), font=("Segoe UI", 9, "bold"),
                     bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack()

            canvas = tk.Canvas(col, width=PREVIEW_W, height=PREVIEW_H,
                                bg=CLR_BG, highlightthickness=1,
                                highlightbackground=CLR_BORDER)
            canvas.pack(pady=(4, 0))
            canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text="No image loaded",
                                fill=CLR_SUBTEXT, font=("Segoe UI", 10))
            setattr(self, attr, canvas)

    def _update_preview(self, path: str, side: str):
        try:
            img = cv2.imread(path)
            if img is None:
                return
            photo = bgr_to_photoimage(img, PREVIEW_W, PREVIEW_H)
            canvas = self.front_canvas if side == "front" else self.back_canvas
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            # Keep reference to prevent GC
            attr = "_front_photo" if side == "front" else "_back_photo"
            setattr(self, attr, photo)
        except Exception:
            pass

    def _build_result_panel(self, parent):
        rframe = tk.Frame(parent, bg=CLR_SURFACE, padx=16, pady=12)
        rframe.pack(fill="both", expand=True)

        tk.Label(rframe, text="VERIFICATION RESULT", font=("Segoe UI", 9, "bold"),
                 bg=CLR_SURFACE, fg=CLR_SUBTEXT).pack(anchor="w")

        self.verdict_label = tk.Label(
            rframe, text="—",
            font=("Segoe UI", 28, "bold"),
            bg=CLR_SURFACE, fg=CLR_SUBTEXT
        )
        self.verdict_label.pack(pady=(8, 4))

        self.denom_result_label = tk.Label(
            rframe, text="",
            font=("Segoe UI", 12),
            bg=CLR_SURFACE, fg=CLR_SUBTEXT
        )
        self.denom_result_label.pack()

        # Error bars
        bars_frame = tk.Frame(rframe, bg=CLR_SURFACE)
        bars_frame.pack(fill="x", pady=12)

        self.front_bar = self._error_bar(bars_frame, "Front")
        self.front_bar.pack(fill="x", pady=(0, 6))
        self.back_bar = self._error_bar(bars_frame, "Back")
        self.back_bar.pack(fill="x")

        # Detail text
        self.detail_text = tk.Text(
            rframe, height=4, font=("Courier New", 9),
            bg=CLR_BG, fg=CLR_SUBTEXT, relief="flat",
            state="disabled", padx=8, pady=6
        )
        self.detail_text.pack(fill="x", pady=(10, 0))

    def _error_bar(self, parent, label: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=CLR_SURFACE)
        tk.Label(frame, text=f"{label} side :", width=10, anchor="w",
                 font=("Segoe UI", 9), bg=CLR_SURFACE, fg=CLR_TEXT).pack(side="left")

        bar_bg = tk.Frame(frame, bg=CLR_BG, height=14, width=260)
        bar_bg.pack(side="left", padx=(4, 8))
        bar_bg.pack_propagate(False)

        bar_fill = tk.Frame(bar_bg, bg=CLR_ACCENT, height=14, width=0)
        bar_fill.place(x=0, y=0, relheight=1, width=0)

        val_label = tk.Label(frame, text="—", font=("Segoe UI", 9),
                             bg=CLR_SURFACE, fg=CLR_TEXT, width=22, anchor="w")
        val_label.pack(side="left")

        frame._bar_fill  = bar_fill
        frame._bar_bg    = bar_bg
        frame._val_label = val_label
        return frame

    def _update_error_bar(self, bar_frame, error: float, threshold: float, suspicious: bool):
        ratio = min(error / (threshold * 2 + 1e-8), 1.0)
        width = int(260 * ratio)
        color = CLR_SUSPICIOUS if suspicious else CLR_GENUINE
        bar_frame._bar_fill.place(width=width)
        bar_frame._bar_fill.configure(bg=color)
        flag = "⚠" if suspicious else "✓"
        bar_frame._val_label.configure(
            text=f"{flag}  {error:.5f} / {threshold:.5f}",
            fg=color
        )

    def _reset_result_panel(self):
        self.verdict_label.configure(text="—", fg=CLR_SUBTEXT)
        self.denom_result_label.configure(text="")
        for bar in [self.front_bar, self.back_bar]:
            bar._bar_fill.place(width=0)
            bar._val_label.configure(text="—", fg=CLR_TEXT)
        self._set_detail("")

    def _set_detail(self, text: str):
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", text)
        self.detail_text.configure(state="disabled")

    def _toggle_denom(self):
        state = "disabled" if self.auto_detect_var.get() else "normal"
        for w in self.denom_frame.winfo_children():
            w.configure(state=state)

    # ---- Verification ----

    def _on_verify(self):
        front = self.front_path.get()
        back  = self.back_path.get()

        if front == "No file selected" or not os.path.exists(front):
            messagebox.showwarning("Missing Image", "Please select a front image.")
            return
        if back == "No file selected" or not os.path.exists(back):
            messagebox.showwarning("Missing Image", "Please select a back image.")
            return
        if self.verifier is None:
            messagebox.showerror(
                "Models Not Loaded",
                "Trained models could not be loaded.\n"
                "Ensure you have run train.py and calibrate.py first,\n"
                "and that the 'models/' directory exists."
            )
            return

        self._reset_result_panel()
        self.verify_btn.configure(state="disabled")
        self.status_label.configure(text="Verifying… please wait.")

        denom_override = None if self.auto_detect_var.get() else self.denom_var.get()

        def _run():
            result = self.verifier.verify(front, back, denomination=denom_override)
            self.after(0, lambda: self._display_result(result))

        threading.Thread(target=_run, daemon=True).start()

    def _display_result(self, result):
        self.verify_btn.configure(state="normal")

        if result.verdict == VERDICT_ERROR:
            self.verdict_label.configure(text="ERROR", fg=CLR_ERROR)
            self.status_label.configure(text=f"Error: {result.error_message}")
            self._set_detail(result.error_message or "")
            return

        # Verdict
        if result.verdict == VERDICT_GENUINE:
            color = CLR_GENUINE
            icon  = "✓  GENUINE"
        else:
            color = CLR_SUSPICIOUS
            icon  = "⚠  SUSPICIOUS"

        self.verdict_label.configure(text=icon, fg=color)
        self.denom_result_label.configure(
            text=f"Rs. {result.denomination} note   |   {result.processing_time_ms:.0f} ms",
            fg=CLR_SUBTEXT
        )

        # Error bars
        if result.front_error is not None:
            self._update_error_bar(
                self.front_bar,
                result.front_error, result.front_threshold, result.front_suspicious
            )
        if result.back_error is not None:
            self._update_error_bar(
                self.back_bar,
                result.back_error, result.back_threshold, result.back_suspicious
            )

        # Detail text
        lines = [
            f"Denomination     : Rs. {result.denomination}",
            f"Front error      : {result.front_error:.6f}  (threshold: {result.front_threshold:.6f})",
            f"Back  error      : {result.back_error:.6f}  (threshold: {result.back_threshold:.6f})",
        ]
        if result.denomination_scores:
            scores_str = "  ".join(f"{k}:{v:.3f}" for k, v in result.denomination_scores.items())
            lines.append(f"Denom scores     : {scores_str}")
        self._set_detail("\n".join(lines))

        self.status_label.configure(text="Verification complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow is required for the UI: pip install Pillow")
        sys.exit(1)

    app = CurrencyVerifierApp()
    app.mainloop()