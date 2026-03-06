"""
train.py
--------
Training pipeline for all 6 denomination+side autoencoders.
(100_F, 100_B, 500_F, 500_B, 1000_F, 1000_B)

Usage:
  # Train all models
  python train.py

  # Train a single model
  python train.py --denom 500 --side F

  # Custom dataset root
  python train.py --dataset /path/to/Dataset --epochs 100

Features:
  - Automatic train/validation split (80/20 by default)
  - Data augmentation (flip, brightness, contrast, rotation)
  - Early stopping based on validation loss
  - Saves best model checkpoint per denomination+side
  - Logs loss curves to CSV for later analysis
"""

import os
import sys
import csv
import json
import argparse
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from autoencoder import (
    CurrencyAutoencoder,
    build_weight_map,
    weighted_reconstruction_error,
    save_model,
    MODEL_H, MODEL_W
)
from preprocessing import preprocess_to_tensor

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DENOMINATIONS = ["100", "500", "1000"]
SIDES         = ["F", "B"]
DATASET_ROOT  = "Dataset"
MODELS_DIR    = "models"
LOGS_DIR      = "logs"

TRAIN_SPLIT   = 0.8
BATCH_SIZE    = 8
EPOCHS        = 80
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
PATIENCE      = 15      # Early stopping patience (epochs without improvement)
MIN_DELTA     = 1e-5    # Minimum improvement to count as progress


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoteDataset(Dataset):
    """
    Loads genuine note images for a single denomination + side.
    Applies augmentation during training.
    """

    def __init__(self, image_paths: List[str], augment: bool = True):
        self.paths   = image_paths
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        tensor = preprocess_to_tensor(path, target_size=(MODEL_H, MODEL_W))

        if tensor is None:
            # Return a blank tensor on failure (rare)
            tensor = np.zeros((MODEL_H, MODEL_W, 3), dtype=np.float32)

        if self.augment:
            tensor = self._augment(tensor)

        # (H, W, 3) → (3, H, W)
        return torch.from_numpy(tensor.transpose(2, 0, 1))

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        Realistic augmentations to simulate different capture conditions.
        All transforms preserve the note content so the autoencoder
        generalises to varied genuine notes without overfitting.
        """
        # Random horizontal flip (front and back are separately trained)
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()

        # Random brightness / contrast jitter
        alpha = random.uniform(0.85, 1.15)   # contrast
        beta  = random.uniform(-0.08, 0.08)  # brightness
        img   = np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)

        # Small random rotation (±5°) to account for slight alignment residuals
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)
            h, w  = img.shape[:2]
            M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Gaussian noise (simulate scanner grain)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img   = np.clip(img + noise, 0.0, 1.0)

        return img


def load_image_paths(dataset_root: str, denomination: str, side: str) -> List[str]:
    folder = os.path.join(dataset_root, f"{denomination}_{side}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts)
    ]
    if not paths:
        raise ValueError(f"No images found in: {folder}")
    return paths


def split_paths(paths: List[str], train_ratio: float = TRAIN_SPLIT, seed: int = 42):
    random.seed(seed)
    shuffled = paths.copy()
    random.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


# ---------------------------------------------------------------------------
# Training loop (single denomination + side)
# ---------------------------------------------------------------------------

def train_one(
    denomination: str,
    side: str,
    dataset_root: str = DATASET_ROOT,
    models_dir: str   = MODELS_DIR,
    logs_dir: str     = LOGS_DIR,
    epochs: int       = EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr: float         = LR,
    device_str: str   = "auto",
    verbose: bool     = True
) -> dict:
    """
    Train a single autoencoder and save the best checkpoint.

    Returns
    -------
    dict with keys: denomination, side, best_val_loss, stopped_epoch, model_path
    """
    tag = f"{denomination}_{side}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ---- Device ----
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training: {tag}   |   device: {device}")
        print(f"{'='*60}")

    # ---- Data ----
    all_paths   = load_image_paths(dataset_root, denomination, side)
    train_paths, val_paths = split_paths(all_paths)

    if verbose:
        print(f"  Images  : {len(all_paths)} total | {len(train_paths)} train | {len(val_paths)} val")

    if len(train_paths) < 4:
        raise ValueError(f"Too few training images for {tag}: {len(train_paths)}")

    train_ds = NoteDataset(train_paths, augment=True)
    val_ds   = NoteDataset(val_paths,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # ---- Model ----
    model      = CurrencyAutoencoder().to(device)
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    weight_map = build_weight_map(denomination, side, device=str(device))

    # ---- Training state ----
    best_val_loss  = float("inf")
    no_improve     = 0
    best_epoch     = 0
    log_rows       = []

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = weighted_reconstruction_error(batch, recon, weight_map)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # -- Validate --
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss  = weighted_reconstruction_error(batch, recon, weight_map)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        scheduler.step(val_loss)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs}  |  train: {train_loss:.6f}  |  val: {val_loss:.6f}")

        # -- Early stopping & best checkpoint --
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            path = save_model(model, denomination, side, models_dir)
            if verbose:
                print(f"  ✓ New best val loss {best_val_loss:.6f} — saved to {path}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ---- Save training log ----
    log_file = os.path.join(logs_dir, f"train_log_{tag}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    if verbose:
        print(f"  Training log saved → {log_file}")
        print(f"  Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

    return {
        "denomination": denomination,
        "side":         side,
        "best_val_loss": best_val_loss,
        "stopped_epoch": best_epoch,
        "model_path":   model_path(denomination, side, models_dir) if False else
                        os.path.join(models_dir, f"ae_{denomination}_{side}.pt")
    }


def model_path(denomination, side, models_dir):
    return os.path.join(models_dir, f"ae_{denomination}_{side}.pt")


# ---------------------------------------------------------------------------
# Train all
# ---------------------------------------------------------------------------

def train_all(
    dataset_root: str = DATASET_ROOT,
    models_dir: str   = MODELS_DIR,
    logs_dir: str     = LOGS_DIR,
    epochs: int       = EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr: float         = LR,
    device_str: str   = "auto",
    verbose: bool     = True
):
    """Train all 6 autoencoders sequentially."""
    results = []
    for denom in DENOMINATIONS:
        for side in SIDES:
            try:
                result = train_one(
                    denom, side,
                    dataset_root=dataset_root,
                    models_dir=models_dir,
                    logs_dir=logs_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device_str=device_str,
                    verbose=verbose
                )
                results.append(result)
            except Exception as e:
                print(f"\n[ERROR] Failed to train {denom}_{side}: {e}")

    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['denomination']}_{r['side']:>1}  |  "
              f"best val loss: {r['best_val_loss']:.6f}  |  "
              f"stopped epoch: {r['stopped_epoch']}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train currency note autoencoders")
    parser.add_argument("--dataset",    default=DATASET_ROOT, help="Path to Dataset root")
    parser.add_argument("--models_dir", default=MODELS_DIR)
    parser.add_argument("--logs_dir",   default=LOGS_DIR)
    parser.add_argument("--denom",      choices=DENOMINATIONS, default=None,
                        help="Train a single denomination (omit to train all)")
    parser.add_argument("--side",       choices=SIDES, default=None,
                        help="Train a single side F or B (requires --denom)")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--device",     default="auto", help="cpu | cuda | auto")

    args = parser.parse_args()

    if args.denom and args.side:
        train_one(
            args.denom, args.side,
            dataset_root=args.dataset,
            models_dir=args.models_dir,
            logs_dir=args.logs_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
        )
    elif args.denom:
        for side in SIDES:
            train_one(
                args.denom, side,
                dataset_root=args.dataset,
                models_dir=args.models_dir,
                logs_dir=args.logs_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device_str=args.device,
            )
    else:
        train_all(
            dataset_root=args.dataset,
            models_dir=args.models_dir,
            logs_dir=args.logs_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
        )