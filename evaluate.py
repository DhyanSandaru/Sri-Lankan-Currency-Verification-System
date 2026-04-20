"""
evaluate.py
-----------
Evaluate trained autoencoder models against genuine note images.

Since only genuine notes are available, this script measures:
  - False Positive Rate (FPR) — % of real notes wrongly flagged as suspicious
  - Mean / std / min / max reconstruction error per denomination+side
  - Per-image error scores sorted worst-to-best
  - Error distribution histogram saved as PNG
  - Threshold visualisation — shows how much headroom exists between
    genuine note errors and the current decision threshold
  - Worst offenders list — specific filenames with highest error
    (useful for finding bad images in your dataset)

Usage:
  python evaluate.py                        # evaluate all models
  python evaluate.py --denom 500            # evaluate only 500-rupee models
  python evaluate.py --denom 1000 --side B  # evaluate only 1000_B
  python evaluate.py --save_report          # also save a text report
"""

import os
import sys
import json
import argparse
import csv
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Try importing matplotlib — it's optional (used for histogram plots)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.dirname(__file__))

from autoencoder import (
    load_model, build_weight_map,
    per_image_weighted_error,
    MODEL_H, MODEL_W
)
from preprocessing import preprocess_to_tensor
from calibrate import load_thresholds, get_threshold
from train import NoteDataset, load_image_paths

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DENOMINATIONS = ["100", "500", "1000"]
SIDES         = ["F", "B"]
DATASET_ROOT  = "Dataset"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"

# How many worst-performing images to highlight
TOP_N_WORST   = 10


# ---------------------------------------------------------------------------
# Core evaluation for one denomination + side
# ---------------------------------------------------------------------------

def evaluate_one(
    denomination: str,
    side: str,
    dataset_root: str = DATASET_ROOT,
    models_dir: str   = MODELS_DIR,
    device_str: str   = "auto",
    batch_size: int   = 8,
    verbose: bool     = True
) -> Optional[Dict]:
    """
    Run all genuine images for one denomination+side through the autoencoder
    and collect per-image reconstruction errors.

    Returns a dict with full stats, or None if model/data not found.
    """
    tag = f"{denomination}_{side}"

    # ---- Device ----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # ---- Load model ----
    model_file = os.path.join(models_dir, f"ae_{denomination}_{side}.pt")
    if not os.path.exists(model_file):
        if verbose:
            print(f"  [SKIP] Model not found: {model_file}")
        return None

    model      = load_model(denomination, side, models_dir=models_dir, device=str(device))
    weight_map = build_weight_map(denomination, side, device=str(device))

    # ---- Load threshold ----
    try:
        thresholds = load_thresholds(models_dir)
        threshold  = get_threshold(thresholds, denomination, side)
    except Exception as e:
        if verbose:
            print(f"  [WARN] Could not load threshold for {tag}: {e}")
        threshold = None

    # ---- Load images ----
    try:
        all_paths = load_image_paths(dataset_root, denomination, side)
    except FileNotFoundError as e:
        if verbose:
            print(f"  [SKIP] {e}")
        return None

    if verbose:
        print(f"\n  {'='*54}")
        print(f"  Evaluating : {tag}   ({len(all_paths)} images)   device: {device}")
        print(f"  {'='*54}")

    # ---- Run inference ----
    ds     = NoteDataset(all_paths, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_errors: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            recon  = model(batch)
            errors = per_image_weighted_error(batch, recon, weight_map)
            all_errors.extend(errors.cpu().numpy().tolist())

    errors_arr = np.array(all_errors, dtype=np.float32)

    # ---- Per-image table ----
    per_image = sorted(
        zip(all_paths, all_errors),
        key=lambda x: x[1],
        reverse=True   # worst first
    )

    # ---- Statistics ----
    mean_err = float(errors_arr.mean())
    std_err  = float(errors_arr.std())
    min_err  = float(errors_arr.min())
    max_err  = float(errors_arr.max())
    p95      = float(np.percentile(errors_arr, 95))
    p99      = float(np.percentile(errors_arr, 99))

    # False positives — images whose error exceeds the threshold
    if threshold is not None:
        fp_paths  = [p for p, e in per_image if e > threshold]
        fp_count  = len(fp_paths)
        fpr       = fp_count / len(all_errors) * 100
    else:
        fp_paths  = []
        fp_count  = 0
        fpr       = None

    # ---- Print results ----
    if verbose:
        print(f"  Images evaluated : {len(all_errors)}")
        print(f"  Error  — mean : {mean_err:.6f}  |  std : {std_err:.6f}")
        print(f"         — min  : {min_err:.6f}  |  max : {max_err:.6f}")
        print(f"         — p95  : {p95:.6f}  |  p99 : {p99:.6f}")

        if threshold is not None:
            headroom = threshold - mean_err
            print(f"  Threshold        : {threshold:.6f}")
            print(f"  Headroom (T-μ)   : {headroom:.6f}  ({headroom/std_err:.1f}σ)")
            print(f"  False positives  : {fp_count} / {len(all_errors)}  "
                  f"({fpr:.1f}%)")

            if fp_count > 0:
                print(f"\n  ⚠  Wrongly flagged genuine notes:")
                for p, e in per_image:
                    if e > threshold:
                        print(f"     {os.path.basename(p):<40}  error: {e:.6f}")

        print(f"\n  Worst {TOP_N_WORST} images (highest reconstruction error):")
        for i, (p, e) in enumerate(per_image[:TOP_N_WORST]):
            flag = " ← FLAGGED" if threshold and e > threshold else ""
            print(f"    {i+1:>2}. {os.path.basename(p):<40}  {e:.6f}{flag}")

    return {
        "denomination": denomination,
        "side":         side,
        "n_images":     len(all_errors),
        "mean":         round(mean_err, 6),
        "std":          round(std_err,  6),
        "min":          round(min_err,  6),
        "max":          round(max_err,  6),
        "p95":          round(p95,      6),
        "p99":          round(p99,      6),
        "threshold":    round(threshold, 6) if threshold else None,
        "fpr_percent":  round(fpr, 2)       if fpr is not None else None,
        "fp_count":     fp_count,
        "fp_files":     [os.path.basename(p) for p in fp_paths],
        "per_image":    [(os.path.basename(p), round(e, 6)) for p, e in per_image],
        "errors_array": errors_arr,   # kept in memory for plotting
    }


# ---------------------------------------------------------------------------
# Histogram / distribution plot
# ---------------------------------------------------------------------------

def plot_distributions(
    results: List[Dict],
    output_dir: str = REPORTS_DIR
):
    """
    For each model, plot the reconstruction error distribution of genuine
    notes alongside the decision threshold.
    One subplot per denomination+side, saved as a single PNG.
    """
    if not HAS_MATPLOTLIB:
        print("\n[INFO] matplotlib not installed — skipping plots.")
        print("       pip install matplotlib   to enable distribution plots.")
        return

    valid = [r for r in results if r is not None]
    if not valid:
        return

    n    = len(valid)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.patch.set_facecolor("#1e1e2e")

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, res in enumerate(valid):
        row = idx // cols
        col = idx % cols
        ax  = axes[row][col]

        errors    = res["errors_array"]
        threshold = res["threshold"]
        tag       = f"{res['denomination']}_{res['side']}"

        ax.set_facecolor("#2a2a3d")
        ax.tick_params(colors="#9090a0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3d3d55")

        # Histogram
        n_bins = min(40, max(10, len(errors) // 3))
        ax.hist(errors, bins=n_bins, color="#7c6af7", alpha=0.85,
                edgecolor="#3d3d55", linewidth=0.5, label="Genuine notes")

        # Mean line
        ax.axvline(res["mean"], color="#4caf50", linewidth=1.5,
                   linestyle="--", label=f"Mean ({res['mean']:.5f})")

        # Threshold line
        if threshold:
            ax.axvline(threshold, color="#f44336", linewidth=2,
                       linestyle="-", label=f"Threshold ({threshold:.5f})")

            # Shade false positive region
            ax.axvspan(threshold, errors.max() * 1.05,
                       alpha=0.15, color="#f44336", label="FP zone")

        ax.set_title(f"Rs. {res['denomination']} — {'Front' if res['side']=='F' else 'Back'}",
                     color="#e0e0e0", fontsize=11, fontweight="bold")
        ax.set_xlabel("Weighted Reconstruction Error", color="#9090a0", fontsize=9)
        ax.set_ylabel("Count", color="#9090a0", fontsize=9)

        fpr_str = f"FPR: {res['fpr_percent']:.1f}%" if res["fpr_percent"] is not None else ""
        ax.legend(fontsize=7, facecolor="#2a2a3d", labelcolor="#e0e0e0",
                  edgecolor="#3d3d55", title=fpr_str,
                  title_fontsize=8)

    # Hide unused subplots
    for idx in range(len(valid), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].set_visible(False)

    fig.suptitle("Genuine Note Reconstruction Error Distributions",
                 color="#e0e0e0", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "error_distributions.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Distribution plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Summary table + text report
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict]):
    valid = [r for r in results if r is not None]
    if not valid:
        print("No results to summarise.")
        return

    print("\n" + "=" * 78)
    print("  EVALUATION SUMMARY")
    print("=" * 78)
    print(f"  {'Model':<10} {'Images':>7} {'Mean':>10} {'Std':>10} "
          f"{'Threshold':>11} {'FPR':>8} {'FP Count':>9}")
    print("  " + "-" * 74)

    for r in valid:
        side_str  = "Front" if r["side"] == "F" else "Back"
        model_str = f"{r['denomination']} {side_str}"
        thr_str   = f"{r['threshold']:.6f}" if r["threshold"] else "N/A"
        fpr_str   = f"{r['fpr_percent']:.1f}%"  if r["fpr_percent"] is not None else "N/A"
        fp_str    = str(r["fp_count"])

        status = ""
        if r["fpr_percent"] is not None:
            if r["fpr_percent"] == 0.0:
                status = "  ✓"
            elif r["fpr_percent"] < 5.0:
                status = "  !"
            else:
                status = "  ⚠"

        print(f"  {model_str:<10} {r['n_images']:>7} {r['mean']:>10.6f} "
              f"{r['std']:>10.6f} {thr_str:>11} {fpr_str:>8} {fp_str:>9}{status}")

    print("=" * 78)
    print("  ✓ = 0% FPR (perfect)   ! = <5% FPR (acceptable)   ⚠ = ≥5% FPR (review needed)")
    print("=" * 78)


def save_report(results: List[Dict], output_dir: str = REPORTS_DIR):
    """Save a CSV and JSON report of all evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    valid = [r for r in results if r is not None]

    # Summary CSV
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["denomination", "side", "n_images", "mean", "std",
                      "min", "max", "p95", "p99", "threshold",
                      "fpr_percent", "fp_count", "fp_files"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in valid:
            row = {k: r[k] for k in fieldnames}
            row["fp_files"] = "; ".join(r["fp_files"])
            writer.writerow(row)
    print(f"  Summary CSV saved   → {csv_path}")

    # Per-image CSV for each model
    for r in valid:
        tag      = f"{r['denomination']}_{r['side']}"
        img_path = os.path.join(output_dir, f"per_image_{tag}.csv")
        with open(img_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "error", "flagged"])
            for fname, err in r["per_image"]:
                flagged = "YES" if r["threshold"] and err > r["threshold"] else "no"
                writer.writerow([fname, err, flagged])
        print(f"  Per-image CSV saved → {img_path}")

    # Full JSON
    json_path = os.path.join(output_dir, "evaluation_full.json")
    serialisable = []
    for r in valid:
        row = {k: v for k, v in r.items() if k != "errors_array"}
        serialisable.append(row)
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Full JSON saved     → {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_all(
    dataset_root: str = DATASET_ROOT,
    models_dir: str   = MODELS_DIR,
    reports_dir: str  = REPORTS_DIR,
    device_str: str   = "auto",
    save: bool        = False,
    plot: bool        = True,
    verbose: bool     = True
):
    results = []
    for denom in DENOMINATIONS:
        for side in SIDES:
            r = evaluate_one(
                denom, side,
                dataset_root=dataset_root,
                models_dir=models_dir,
                device_str=device_str,
                verbose=verbose
            )
            results.append(r)

    print_summary(results)

    if plot:
        plot_distributions([r for r in results if r], output_dir=reports_dir)

    if save:
        save_report([r for r in results if r], output_dir=reports_dir)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate currency autoencoder models")
    parser.add_argument("--dataset",     default=DATASET_ROOT)
    parser.add_argument("--models_dir",  default=MODELS_DIR)
    parser.add_argument("--reports_dir", default=REPORTS_DIR)
    parser.add_argument("--denom",       choices=DENOMINATIONS, default=None)
    parser.add_argument("--side",        choices=SIDES, default=None)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--save_report", action="store_true",
                        help="Save CSV + JSON reports to reports/")
    parser.add_argument("--no_plot",     action="store_true",
                        help="Skip distribution plots")
    args = parser.parse_args()

    if args.denom and args.side:
        r = evaluate_one(
            args.denom, args.side,
            dataset_root=args.dataset,
            models_dir=args.models_dir,
            device_str=args.device
        )
        if r:
            print_summary([r])
            if not args.no_plot:
                plot_distributions([r], output_dir=args.reports_dir)
            if args.save_report:
                save_report([r], output_dir=args.reports_dir)

    elif args.denom:
        results = []
        for side in SIDES:
            r = evaluate_one(
                args.denom, side,
                dataset_root=args.dataset,
                models_dir=args.models_dir,
                device_str=args.device
            )
            results.append(r)
        print_summary(results)
        if not args.no_plot:
            plot_distributions([r for r in results if r], output_dir=args.reports_dir)
        if args.save_report:
            save_report([r for r in results if r], output_dir=args.reports_dir)

    else:
        evaluate_all(
            dataset_root=args.dataset,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            device_str=args.device,
            save=args.save_report,
            plot=not args.no_plot
        )
