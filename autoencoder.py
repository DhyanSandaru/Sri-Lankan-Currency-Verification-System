"""
autoencoder.py
--------------
Convolutional Autoencoder (CAE) for currency note anomaly detection.

Architecture:
  Encoder: Conv2d blocks with stride-2 downsampling → bottleneck
  Decoder: ConvTranspose2d blocks → reconstructed image

  Input:  (B, 3, 256, 512)  — RGB, H=256, W=512 (half of warp size for efficiency)
  Output: (B, 3, 256, 512)  — reconstructed image

Region-weighted reconstruction error:
  Different zones of the note are given different weights when computing
  reconstruction error. Watermark and security thread zones (left strip
  and central band) are weighted higher than plain areas.

  Weight map is defined relative to note dimensions and can be customised
  per denomination via denomination_weight_maps().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Model dimensions
# ---------------------------------------------------------------------------

MODEL_H = 256
MODEL_W = 512
IN_CHANNELS = 3


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, stride=2, output_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=4, stride=stride,
                padding=1, output_padding=output_padding,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class CurrencyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for a single denomination + side.

    Encoder architecture (stride-2 downsampling at each block):
      (3, 256, 512) → (32, 128, 256) → (64, 64, 128) → (128, 32, 64)
                    → (256, 16, 32)  → bottleneck (512, 8, 16)

    Decoder mirrors the encoder with ConvTranspose2d blocks.
    """

    def __init__(self, base_channels: int = 32, bottleneck_channels: int = 512):
        super().__init__()

        c = base_channels  # 32

        # ---- Encoder ----
        self.enc1 = ConvBlock(IN_CHANNELS, c,      stride=2)   # → (c,   128, 256)
        self.enc2 = ConvBlock(c,           c * 2,  stride=2)   # → (2c,   64, 128)
        self.enc3 = ConvBlock(c * 2,       c * 4,  stride=2)   # → (4c,   32,  64)
        self.enc4 = ConvBlock(c * 4,       c * 8,  stride=2)   # → (8c,   16,  32)
        self.enc5 = ConvBlock(c * 8,       bottleneck_channels, stride=2)  # → (512, 8, 16)

        # ---- Decoder ----
        self.dec5 = DeconvBlock(bottleneck_channels, c * 8)     # → (8c,  16, 32)
        self.dec4 = DeconvBlock(c * 8,  c * 4)                  # → (4c,  32, 64)
        self.dec3 = DeconvBlock(c * 4,  c * 2)                  # → (2c,  64, 128)
        self.dec2 = DeconvBlock(c * 2,  c)                      # → (c,  128, 256)
        self.dec1 = DeconvBlock(c,      c // 2)                 # → (c/2,256, 512)

        # Final projection → 3-channel output, sigmoid to [0,1]
        self.final = nn.Sequential(
            nn.Conv2d(c // 2, IN_CHANNELS, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec5(z)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.final(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# ---------------------------------------------------------------------------
# Region-weighted reconstruction error
# ---------------------------------------------------------------------------

def build_weight_map(
    denomination: str,
    side: str,
    h: int = MODEL_H,
    w: int = MODEL_W,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Build a spatial weight map (H × W) for reconstruction error.

    Regions weighted higher (more security features):
      - Left strip  (~8% of width)  : watermark / windowed security thread
      - Right strip (~8% of width)  : serial number / colour-shift ink
      - Central horizontal band      : latent image / microprinting

    Base weight everywhere else: 1.0
    High-security zone weight:   3.0
    (These values are reasonable defaults — tune after collecting validation data.)

    Parameters
    ----------
    denomination : "100" | "500" | "1000"
    side         : "F" | "B"
    h, w         : spatial dimensions of the weight map
    device       : torch device string

    Returns
    -------
    torch.Tensor of shape (1, 1, H, W) — broadcastable with (B, C, H, W).
    """
    weight = np.ones((h, w), dtype=np.float32)

    left_w  = int(w * 0.08)
    right_w = int(w * 0.08)
    top_h   = int(h * 0.35)
    bot_h   = int(h * 0.65)

    HIGH = 3.0

    # Watermark strip (left)
    weight[:, :left_w] = HIGH

    # Security thread / serial number strip (right)
    weight[:, w - right_w:] = HIGH

    # Central horizontal band (latent image / microprinting)
    weight[top_h:bot_h, :] = np.maximum(weight[top_h:bot_h, :], HIGH)

    # Denomination-specific overrides
    # 1000-rupee note has an additional colour-shift patch top-right corner
    if denomination == "1000" and side == "F":
        patch_y = int(h * 0.05)
        patch_x = int(w * 0.78)
        weight[:patch_y, patch_x:] = HIGH * 1.5

    # Normalise so that mean weight == 1.0 (keeps loss scale stable)
    weight /= weight.mean()

    t = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return t.to(device)


def weighted_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    weight_map: torch.Tensor
) -> torch.Tensor:
    """
    Compute pixel-wise MSE weighted by the spatial weight map,
    then average over the batch.

    Parameters
    ----------
    original      : (B, C, H, W)
    reconstructed : (B, C, H, W)
    weight_map    : (1, 1, H, W) — broadcast over B and C

    Returns
    -------
    Scalar tensor — mean weighted MSE over the batch.
    """
    sq_err = (original - reconstructed) ** 2          # (B, C, H, W)
    # Average over channel dim first, then apply spatial weights
    sq_err_spatial = sq_err.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    weighted = sq_err_spatial * weight_map             # (B, 1, H, W)
    return weighted.mean()


def per_image_weighted_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    weight_map: torch.Tensor
) -> torch.Tensor:
    """
    Same as weighted_reconstruction_error but returns one score per image
    in the batch — used for threshold computation and inference.

    Returns
    -------
    Tensor of shape (B,) — one error score per image.
    """
    sq_err = (original - reconstructed) ** 2          # (B, C, H, W)
    sq_err_spatial = sq_err.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    weighted = sq_err_spatial * weight_map             # (B, 1, H, W)
    return weighted.mean(dim=(1, 2, 3))               # (B,)


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------

def model_path(denomination: str, side: str, models_dir: str = "models") -> str:
    """Return the .pt file path for a given denomination + side."""
    return f"{models_dir}/ae_{denomination}_{side}.pt"


def threshold_path(models_dir: str = "models") -> str:
    return f"{models_dir}/thresholds.json"


def load_model(
    denomination: str,
    side: str,
    models_dir: str = "models",
    device: str = "cpu"
) -> CurrencyAutoencoder:
    """Load a trained autoencoder from disk."""
    path = model_path(denomination, side, models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = CurrencyAutoencoder()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_model(
    model: CurrencyAutoencoder,
    denomination: str,
    side: str,
    models_dir: str = "models"
) -> str:
    """Save model weights to disk."""
    import os
    os.makedirs(models_dir, exist_ok=True)
    path = model_path(denomination, side, models_dir)
    torch.save(model.state_dict(), path)
    return path


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

import os

if __name__ == "__main__":
    print("=== CurrencyAutoencoder sanity check ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CurrencyAutoencoder().to(device)
    dummy = torch.randn(2, 3, MODEL_H, MODEL_W).to(device)
    out = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == dummy.shape, "Shape mismatch!"

    wmap = build_weight_map("1000", "F", device=device)
    err = per_image_weighted_error(dummy, out, wmap)
    print(f"Weighted error per image: {err.detach().cpu().numpy()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✓ All checks passed.")
