"""
inference.py
------------
Full end-to-end inference pipeline.

Given:
  - A front image and a back image of a note
  - The denomination (provided explicitly or auto-detected)

Returns:
  - Verdict: GENUINE or SUSPICIOUS
  - Per-side reconstruction errors
  - Per-side thresholds used
  - Denomination detected
  - Confidence scores

Design:
  - Denomination is detected from the front image (more colour-distinctive).
  - Both front and back are processed through their respective autoencoders.
  - If EITHER side exceeds its threshold, the note is flagged as SUSPICIOUS.
  - Side thresholds are loaded from models/thresholds.json (set by calibrate.py).
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

import cv2
import numpy as np
import torch

from preprocessing import preprocess, preprocess_to_tensor
from denomination_detector import detect, load_references
from autoencoder import (
    CurrencyAutoencoder,
    build_weight_map,
    per_image_weighted_error,
    load_model,
    MODEL_H, MODEL_W
)
from calibrate import load_thresholds, get_threshold

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

VERDICT_GENUINE    = "GENUINE"
VERDICT_SUSPICIOUS = "SUSPICIOUS"
VERDICT_ERROR      = "ERROR"


@dataclass
class VerificationResult:
    verdict: str                          # GENUINE | SUSPICIOUS | ERROR
    denomination: str                     # "100" | "500" | "1000" | "unknown"
    front_error: Optional[float] = None
    back_error:  Optional[float] = None
    front_threshold: Optional[float] = None
    back_threshold:  Optional[float] = None
    front_suspicious: Optional[bool] = None
    back_suspicious:  Optional[bool] = None
    denomination_scores: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"  Verdict      : {self.verdict}",
            f"  Denomination : {self.denomination}",
        ]
        if self.front_error is not None:
            flag = "⚠ SUSPICIOUS" if self.front_suspicious else "✓ OK"
            lines.append(f"  Front error  : {self.front_error:.6f}  (threshold: {self.front_threshold:.6f})  {flag}")
        if self.back_error is not None:
            flag = "⚠ SUSPICIOUS" if self.back_suspicious else "✓ OK"
            lines.append(f"  Back  error  : {self.back_error:.6f}  (threshold: {self.back_threshold:.6f})  {flag}")
        if self.denomination_scores:
            lines.append(f"  Denom scores : {self.denomination_scores}")
        lines.append(f"  Time         : {self.processing_time_ms:.1f} ms")
        if self.error_message:
            lines.append(f"  Error        : {self.error_message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier class
# ---------------------------------------------------------------------------

class CurrencyVerifier:
    """
    Stateful verifier — loads models and thresholds once, then reuses them
    for repeated inference. Best used in a long-running application.
    """

    def __init__(
        self,
        models_dir: str  = "models",
        device_str: str  = "auto",
        verbose: bool    = False
    ):
        self.models_dir = models_dir
        self.verbose    = verbose

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        if verbose:
            print(f"[Verifier] Loading on device: {self.device}")

        # Load denomination reference histograms
        self.denom_references = load_references(
            os.path.join(models_dir, "denomination_histograms.json")
        )

        # Load thresholds
        self.thresholds = load_thresholds(models_dir)

        # Lazy-load autoencoders on first use
        self._models: Dict[str, CurrencyAutoencoder] = {}
        self._weight_maps: Dict[str, torch.Tensor]   = {}

    def _get_model(self, denomination: str, side: str) -> CurrencyAutoencoder:
        key = f"{denomination}_{side}"
        if key not in self._models:
            if self.verbose:
                print(f"[Verifier] Loading model {key}")
            self._models[key] = load_model(
                denomination, side,
                models_dir=self.models_dir,
                device=str(self.device)
            )
        return self._models[key]

    def _get_weight_map(self, denomination: str, side: str) -> torch.Tensor:
        key = f"{denomination}_{side}"
        if key not in self._weight_maps:
            self._weight_maps[key] = build_weight_map(
                denomination, side, device=str(self.device)
            )
        return self._weight_maps[key]

    def _run_autoencoder(
        self,
        image_path: str,
        denomination: str,
        side: str
    ) -> Optional[float]:
        """
        Preprocess an image and run it through the appropriate autoencoder.
        Returns the weighted reconstruction error, or None on failure.
        """
        tensor = preprocess_to_tensor(image_path, target_size=(MODEL_H, MODEL_W))
        if tensor is None:
            return None

        # (H, W, 3) → (1, 3, H, W)
        t = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        model      = self._get_model(denomination, side)
        weight_map = self._get_weight_map(denomination, side)

        with torch.no_grad():
            recon  = model(t)
            errors = per_image_weighted_error(t, recon, weight_map)

        return float(errors[0].cpu().item())

    def verify(
        self,
        front_path: str,
        back_path: str,
        denomination: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a banknote from its front and back images.

        Parameters
        ----------
        front_path   : Path to front image.
        back_path    : Path to back image.
        denomination : If provided, skip auto-detection and use this value.
                       Must be "100", "500", or "1000".

        Returns
        -------
        VerificationResult
        """
        t_start = time.time()

        # ---- Step 1: Denomination detection ----
        denom_scores = {}

        if denomination is not None:
            detected_denom = denomination
        else:
            # Use front image for detection (more colour-distinctive)
            result = preprocess(front_path)
            if result[0] is None:
                return VerificationResult(
                    verdict="ERROR",
                    denomination="unknown",
                    error_message="Could not preprocess front image for denomination detection."
                )
            _, hsv = result
            detected_denom, denom_scores = detect(
                hsv, references=self.denom_references
            )

        if detected_denom == "unknown":
            return VerificationResult(
                verdict=VERDICT_ERROR,
                denomination="unknown",
                denomination_scores=denom_scores,
                error_message="Could not identify denomination. "
                              "Ensure the note is clearly visible.",
                processing_time_ms=(time.time() - t_start) * 1000
            )

        if self.verbose:
            print(f"[Verifier] Detected denomination: {detected_denom}  scores: {denom_scores}")

        # ---- Step 2: Front autoencoder ----
        front_error = self._run_autoencoder(front_path, detected_denom, "F")
        if front_error is None:
            return VerificationResult(
                verdict=VERDICT_ERROR,
                denomination=detected_denom,
                denomination_scores=denom_scores,
                error_message="Failed to process front image.",
                processing_time_ms=(time.time() - t_start) * 1000
            )

        # ---- Step 3: Back autoencoder ----
        back_error = self._run_autoencoder(back_path, detected_denom, "B")
        if back_error is None:
            return VerificationResult(
                verdict=VERDICT_ERROR,
                denomination=detected_denom,
                denomination_scores=denom_scores,
                error_message="Failed to process back image.",
                processing_time_ms=(time.time() - t_start) * 1000
            )

        # ---- Step 4: Threshold comparison ----
        try:
            front_threshold = get_threshold(self.thresholds, detected_denom, "F")
            back_threshold  = get_threshold(self.thresholds, detected_denom, "B")
        except KeyError as e:
            return VerificationResult(
                verdict=VERDICT_ERROR,
                denomination=detected_denom,
                error_message=str(e),
                processing_time_ms=(time.time() - t_start) * 1000
            )

        front_suspicious = front_error > front_threshold
        back_suspicious  = back_error  > back_threshold
        verdict = VERDICT_SUSPICIOUS if (front_suspicious or back_suspicious) else VERDICT_GENUINE

        elapsed_ms = (time.time() - t_start) * 1000

        return VerificationResult(
            verdict=verdict,
            denomination=detected_denom,
            front_error=round(front_error, 6),
            back_error=round(back_error,   6),
            front_threshold=round(front_threshold, 6),
            back_threshold=round(back_threshold,   6),
            front_suspicious=front_suspicious,
            back_suspicious=back_suspicious,
            denomination_scores=denom_scores,
            processing_time_ms=round(elapsed_ms, 1)
        )


# ---------------------------------------------------------------------------
# Stateless convenience function (loads everything fresh each call)
# ---------------------------------------------------------------------------

def verify_note(
    front_path: str,
    back_path: str,
    denomination: Optional[str] = None,
    models_dir: str = "models",
    device_str: str = "auto"
) -> VerificationResult:
    """
    One-shot verification. Suitable for single calls or scripts.
    For repeated inference, use CurrencyVerifier() instead.
    """
    verifier = CurrencyVerifier(
        models_dir=models_dir,
        device_str=device_str,
        verbose=False
    )
    return verifier.verify(front_path, back_path, denomination=denomination)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Verify a Sri Lankan banknote")
    parser.add_argument("front",  help="Path to front image")
    parser.add_argument("back",   help="Path to back image")
    parser.add_argument("--denom", choices=["100", "500", "1000"], default=None,
                        help="Override denomination detection")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--json",       action="store_true",
                        help="Output result as JSON")

    args = parser.parse_args()

    result = verify_note(
        args.front, args.back,
        denomination=args.denom,
        models_dir=args.models_dir,
        device_str=args.device
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"  CURRENCY VERIFICATION RESULT")
        print(f"{'='*50}")
        print(result)
        print(f"{'='*50}\n")

    sys.exit(0 if result.verdict == VERDICT_GENUINE else 1)
