"""
calibrate.py
------------
Compute per-denomination, per-side reconstruction error thresholds.

After training the autoencoders, run this script to determine the
decision thresholds T_front_X and T_back_X that separate genuine
from suspicious notes.

Method:
  1. Run each trained autoencoder over its validation set (genuine images).
  2. Collect per-image weighted reconstruction errors.
  3. Compute threshold = mean + (k × std)
     where k is configurable (default 3.0 — covers ~99.7% of genuine notes).
  4. Save thresholds to models/thresholds.json.

Usage:
  python calibrate.py                    # all denominations and sides
  python calibrate.py --denom 500        # only 500-rupee
  python calibrate.py --denom 1000 --side F --k 2.5
"""

import os
import json
import argparse
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from autoencoder import (
    CurrencyAutoencoder,
    build_weight_map,
    per_image_weighted_error,
    load_model,
    threshold_path,
    MODEL_H, MODEL_W
)
from preprocessing import preprocess_to_tensor
from train import NoteDataset, load_image_paths, split_paths, TRAIN_SPLIT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DENOMINATIONS = ["100", "500", "1000"]
SIDES         = ["F", "B"]
DATASET_ROOT  = "Dataset"
MODELS_DIR    = "models"
DEFAULT_K     = 3.0   # Number of std devs above mean to set threshold


# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------

def calibrate_threshold(
    denomination: str,
    side: str,
    dataset_root: str = DATASET_ROOT,
    models_dir: str   = MODELS_DIR,
    k: float          = DEFAULT_K,
    batch_size: int   = 8,
    device_str: str   = "auto",
    verbose: bool     = True
) -> Dict[str, float]:
    """
    Compute the reconstruction error threshold for a single denomination + side.

    Parameters
    ----------
    denomination : "100" | "500" | "1000"
    side         : "F" | "B"
    k            : threshold = mean_error + k * std_error

    Returns
    -------
    dict with keys:
        mean, std, threshold, k,
        p95, p99  (95th and 99th percentile errors — useful reference)
    """
    tag = f"{denomination}_{side}"

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"\nCalibrating threshold: {tag}  |  device: {device}  |  k={k}")

    # ---- Load model ----
    model = load_model(denomination, side, models_dir=models_dir, device=str(device))
    weight_map = build_weight_map(denomination, side, device=str(device))

    # ---- Load validation images ----
    all_paths = load_image_paths(dataset_root, denomination, side)
    _, val_paths = split_paths(all_paths, train_ratio=TRAIN_SPLIT)

    if len(val_paths) == 0:
        raise ValueError(f"No validation images for {tag}. Increase dataset size.")

    if verbose:
        print(f"  Validation images: {len(val_paths)}")

    val_ds     = NoteDataset(val_paths, augment=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---- Collect errors ----
    all_errors: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            errors = per_image_weighted_error(batch, recon, weight_map)
            all_errors.extend(errors.cpu().numpy().tolist())

    errors_arr = np.array(all_errors, dtype=np.float32)
    mean_err   = float(errors_arr.mean())
    std_err    = float(errors_arr.std())
    threshold  = mean_err + k * std_err
    p95        = float(np.percentile(errors_arr, 95))
    p99        = float(np.percentile(errors_arr, 99))

    stats = {
        "denomination": denomination,
        "side":         side,
        "k":            k,
        "mean":         round(mean_err,  6),
        "std":          round(std_err,   6),
        "threshold":    round(threshold, 6),
        "p95":          round(p95,       6),
        "p99":          round(p99,       6),
        "n_samples":    len(all_errors)
    }

    if verbose:
        print(f"  Error stats  —  mean: {mean_err:.6f}  |  std: {std_err:.6f}")
        print(f"  p95: {p95:.6f}  |  p99: {p99:.6f}")
        print(f"  Threshold (mean + {k}σ): {threshold:.6f}")

    return stats


# ---------------------------------------------------------------------------
# Calibrate all and save JSON
# ---------------------------------------------------------------------------

def calibrate_all(
    dataset_root: str = DATASET_ROOT,
    models_dir: str   = MODELS_DIR,
    k: float          = DEFAULT_K,
    batch_size: int   = 8,
    device_str: str   = "auto",
    verbose: bool     = True
) -> dict:
    """
    Calibrate thresholds for all trained denomination+side combinations
    and save results to models/thresholds.json.

    Returns
    -------
    Nested dict: thresholds[denomination][side] = { threshold, mean, std, ... }
    """
    thresholds: Dict[str, Dict] = {}

    for denom in DENOMINATIONS:
        thresholds[denom] = {}
        for side in SIDES:
            model_file = os.path.join(models_dir, f"ae_{denom}_{side}.pt")
            if not os.path.exists(model_file):
                if verbose:
                    print(f"[SKIP] Model not found: {model_file}")
                continue
            try:
                stats = calibrate_threshold(
                    denom, side,
                    dataset_root=dataset_root,
                    models_dir=models_dir,
                    k=k,
                    batch_size=batch_size,
                    device_str=device_str,
                    verbose=verbose
                )
                thresholds[denom][side] = stats
            except Exception as e:
                if verbose:
                    print(f"[ERROR] {denom}_{side}: {e}")

    # Save to JSON
    out_path = threshold_path(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    if verbose:
        print(f"\nThresholds saved → {out_path}")
        print("\nSummary:")
        print(f"  {'Denom':>6}  {'Side':>4}  {'Threshold':>12}  {'p95':>12}  {'p99':>12}")
        print("  " + "-" * 52)
        for denom in DENOMINATIONS:
            for side in SIDES:
                if side in thresholds.get(denom, {}):
                    s = thresholds[denom][side]
                    print(f"  {denom:>6}  {side:>4}  {s['threshold']:>12.6f}  "
                          f"{s['p95']:>12.6f}  {s['p99']:>12.6f}")

    return thresholds


def load_thresholds(models_dir: str = MODELS_DIR) -> dict:
    """Load thresholds from JSON. Used by inference.py."""
    path = threshold_path(models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Thresholds file not found: {path}\n"
            "Run calibrate.py after training your models."
        )
    with open(path) as f:
        return json.load(f)


def get_threshold(
    thresholds: dict,
    denomination: str,
    side: str
) -> float:
    """
    Retrieve the threshold value for a specific denomination + side.
    Raises KeyError if not found.
    """
    try:
        return thresholds[denomination][side]["threshold"]
    except KeyError:
        raise KeyError(
            f"No threshold for {denomination}_{side}. "
            "Run calibrate.py first."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate autoencoder thresholds")
    parser.add_argument("--dataset",    default=DATASET_ROOT)
    parser.add_argument("--models_dir", default=MODELS_DIR)
    parser.add_argument("--denom",      choices=DENOMINATIONS, default=None)
    parser.add_argument("--side",       choices=SIDES, default=None)
    parser.add_argument("--k",          type=float, default=DEFAULT_K,
                        help="Threshold = mean + k*std (default 3.0)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device",     default="auto")

    args = parser.parse_args()

    if args.denom and args.side:
        # Single model calibration — update thresholds.json in place
        stats = calibrate_threshold(
            args.denom, args.side,
            dataset_root=args.dataset,
            models_dir=args.models_dir,
            k=args.k,
            batch_size=args.batch_size,
            device_str=args.device
        )
        # Load existing thresholds if present, update, re-save
        t_path = threshold_path(args.models_dir)
        if os.path.exists(t_path):
            with open(t_path) as f:
                existing = json.load(f)
        else:
            existing = {}
        existing.setdefault(args.denom, {})[args.side] = stats
        os.makedirs(args.models_dir, exist_ok=True)
        with open(t_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Updated {t_path}")

    elif args.denom:
        for side in SIDES:
            calibrate_threshold(
                args.denom, side,
                dataset_root=args.dataset,
                models_dir=args.models_dir,
                k=args.k,
                batch_size=args.batch_size,
                device_str=args.device
            )
    else:
        calibrate_all(
            dataset_root=args.dataset,
            models_dir=args.models_dir,
            k=args.k,
            batch_size=args.batch_size,
            device_str=args.device
        )
