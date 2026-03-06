"""
denomination_detector.py
-------------------------
Denomination detection via hue + saturation histogram comparison.

Workflow:
  1. calibrate()  — called once, reads Dataset/100_F, Dataset/500_F, Dataset/1000_F
                    (front images only — more color-distinctive than backs)
                    builds a reference histogram per denomination and saves to JSON.

  2. detect()     — given a preprocessed HSV image, loads reference histograms,
                    computes histogram similarity (Bhattacharyya distance),
                    returns the best-matching denomination.

Design notes:
  - We use the Hue channel (0–179 in OpenCV) + Saturation channel jointly
    as a 2D histogram. This is more robust than hue alone, especially after
    CLAHE normalisation.
  - Bhattacharyya distance: lower = more similar. We pick the denomination
    with the lowest distance to the input.
  - Reference histograms are averaged across all calibration images per denom.
  - If the best match confidence is below a threshold we return "unknown".
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from preprocessing import preprocess, get_hsv_from_preprocessed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DENOMINATIONS = ["100", "500", "1000"]
CALIBRATION_FILE = "models/denomination_histograms.json"

# Histogram parameters
H_BINS = 36    # Hue bins  (every 5 degrees)
S_BINS = 32    # Saturation bins
H_RANGE = [0, 180]
S_RANGE = [0, 256]

# If the best Bhattacharyya distance exceeds this, we call it "unknown"
UNKNOWN_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Histogram computation
# ---------------------------------------------------------------------------

def _compute_hs_histogram(hsv: np.ndarray) -> np.ndarray:
    """
    Compute a normalised 2D Hue-Saturation histogram from an HSV image.
    Returns a flat float32 array of length H_BINS * S_BINS.
    """
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [H_BINS, S_BINS],
        H_RANGE + S_RANGE
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute Bhattacharyya distance between two normalised histograms.
    Range [0, 1] — 0 means identical, 1 means completely different.
    """
    # cv2.compareHist expects 2D histograms; reshape back
    a = h1.reshape(H_BINS, S_BINS)
    b = h2.reshape(H_BINS, S_BINS)
    return cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    dataset_root: str = "Dataset",
    output_path: str = CALIBRATION_FILE,
    use_side: str = "F",     # "F" = front, "B" = back, "both" = average both
    verbose: bool = True
) -> Dict[str, list]:
    """
    Build reference histograms from genuine note images.

    Parameters
    ----------
    dataset_root : str
        Root folder containing subfolders like 100_F, 500_B, etc.
    output_path : str
        Where to save the JSON calibration file.
    use_side : str
        Which side(s) to use for calibration. "F", "B", or "both".
    verbose : bool

    Returns
    -------
    dict mapping denomination → list (serialised histogram)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reference_hists: Dict[str, np.ndarray] = {}

    for denom in DENOMINATIONS:
        sides = ["F", "B"] if use_side == "both" else [use_side]
        all_hists = []

        for side in sides:
            folder = os.path.join(dataset_root, f"{denom}_{side}")
            if not os.path.isdir(folder):
                if verbose:
                    print(f"  [WARN] Folder not found: {folder} — skipping")
                continue

            image_files = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            if not image_files:
                if verbose:
                    print(f"  [WARN] No images in {folder}")
                continue

            if verbose:
                print(f"  Calibrating {denom}_{side}: {len(image_files)} images")

            for fname in image_files:
                fpath = os.path.join(folder, fname)
                try:
                    result = preprocess(fpath)
                    if result[0] is None:
                        continue
                    hsv = result[1]
                    hist = _compute_hs_histogram(hsv)
                    all_hists.append(hist)
                except Exception as e:
                    if verbose:
                        print(f"    [SKIP] {fname}: {e}")

        if not all_hists:
            if verbose:
                print(f"  [ERROR] No valid images for denomination {denom}")
            continue

        # Average histogram across all images for this denomination
        mean_hist = np.mean(all_hists, axis=0).astype(np.float32)
        # Re-normalise after averaging
        mean_hist /= (mean_hist.sum() + 1e-8)
        reference_hists[denom] = mean_hist

        if verbose:
            print(f"  ✓ {denom}: reference histogram built from {len(all_hists)} images")

    # Serialise to JSON (lists of floats)
    serialised = {k: v.tolist() for k, v in reference_hists.items()}
    with open(output_path, "w") as f:
        json.dump(serialised, f)

    if verbose:
        print(f"\nCalibration saved → {output_path}")

    return serialised


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def load_references(calibration_path: str = CALIBRATION_FILE) -> Dict[str, np.ndarray]:
    """Load reference histograms from JSON."""
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}\n"
            "Run denomination_detector.calibrate() first."
        )
    with open(calibration_path) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def detect(
    hsv_image: np.ndarray,
    references: Optional[Dict[str, np.ndarray]] = None,
    calibration_path: str = CALIBRATION_FILE,
    return_scores: bool = False
) -> Tuple[str, Dict[str, float]]:
    """
    Detect the denomination of a preprocessed note image.

    Parameters
    ----------
    hsv_image : np.ndarray
        HSV image output from preprocessing.preprocess() — shape (H, W, 3).
    references : dict, optional
        Pre-loaded reference histograms. Loaded from file if None.
    calibration_path : str
        Path to calibration JSON (used if references is None).
    return_scores : bool
        If True, also return a dict of {denom: bhattacharyya_distance}.

    Returns
    -------
    (denomination, scores)
        denomination : str — "100", "500", "1000", or "unknown"
        scores       : dict — Bhattacharyya distances per denomination (lower = better match)
    """
    if references is None:
        references = load_references(calibration_path)

    input_hist = _compute_hs_histogram(hsv_image)

    scores: Dict[str, float] = {}
    for denom, ref_hist in references.items():
        dist = _bhattacharyya(input_hist, ref_hist)
        scores[denom] = round(float(dist), 4)

    best_denom = min(scores, key=scores.get)
    best_score = scores[best_denom]

    if best_score > UNKNOWN_THRESHOLD:
        return "unknown", scores

    return best_denom, scores


def detect_from_path(
    image_path: str,
    references: Optional[Dict[str, np.ndarray]] = None,
    calibration_path: str = CALIBRATION_FILE
) -> Tuple[str, Dict[str, float]]:
    """
    Convenience: run full preprocessing + denomination detection from a file path.
    """
    result = preprocess(image_path)
    if result[0] is None:
        return "unknown", {}
    _, hsv = result
    return detect(hsv, references=references, calibration_path=calibration_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python denomination_detector.py calibrate [dataset_root]")
        print("  python denomination_detector.py detect <image_path>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "calibrate":
        root = sys.argv[2] if len(sys.argv) > 2 else "Dataset"
        calibrate(dataset_root=root, verbose=True)

    elif command == "detect":
        if len(sys.argv) < 3:
            print("Provide an image path.")
            sys.exit(1)
        path = sys.argv[2]
        denom, scores = detect_from_path(path)
        print(f"\nDetected denomination : {denom}")
        print("Bhattacharyya distances (lower = better match):")
        for d, s in sorted(scores.items(), key=lambda x: x[1]):
            marker = " ← best" if d == denom else ""
            print(f"  {d:>6} : {s:.4f}{marker}")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
