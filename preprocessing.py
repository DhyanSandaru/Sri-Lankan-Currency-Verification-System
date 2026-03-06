"""
preprocessing.py
----------------
Full preprocessing pipeline for Sri Lankan currency note images.

Pipeline:
  Input Image
    → Grayscale → Gaussian Blur → Adaptive Threshold → Morphological Ops
    → Find Contours → Filter → Detect Rectangle
    → Perspective Warp (1024×512)
    → HSV conversion + CLAHE on V channel
    → Output: normalised numpy array ready for denomination detection & autoencoder
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# Fixed output size (width × height) — 2:1 ratio matching Sri Lankan note dimensions
WARP_W = 1024
WARP_H = 512


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _blur(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur to suppress noise."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def _adaptive_threshold(blurred: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding to handle uneven illumination.
    Returns a binary image.
    """
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4
    )


def _morphology(binary: np.ndarray) -> np.ndarray:
    """
    Closing to fill small holes in the note boundary,
    followed by dilation to strengthen edges.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def _find_note_contour(processed: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest quadrilateral contour — assumed to be the note boundary.
    Returns the 4-point contour or None if not found.
    """
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = processed.shape[0] * processed.shape[1]

    for cnt in contours[:5]:  # check top-5 largest
        area = cv2.contourArea(cnt)
        # Note must occupy at least 15% of frame
        if area < 0.15 * img_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _perspective_warp(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to warp the note to a fixed WARP_W × WARP_H canvas.
    Handles both landscape and portrait orientations.
    """
    rect = _order_points(pts)
    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))

    # If the note was captured in portrait, flip to landscape
    h, w = warped.shape[:2]
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        warped = cv2.resize(warped, (WARP_W, WARP_H))

    return warped


def _apply_clahe_hsv(bgr: np.ndarray) -> np.ndarray:
    """
    Convert to HSV, apply CLAHE on the V (value/brightness) channel,
    then convert back to BGR.
    This normalises illumination while preserving hue information
    needed for denomination detection.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge([h, s, v_eq])
    bgr_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return bgr_eq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    image_input,
    debug: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    image_input : str | np.ndarray
        Path to image file OR a BGR numpy array.
    debug : bool
        If True, returns intermediate visualisation alongside the result.

    Returns
    -------
    (processed_bgr, hsv_image) : Tuple
        processed_bgr — illumination-normalised BGR image at WARP_W×WARP_H.
        hsv_image     — HSV version of the same image (used by denomination detector).
        Both are None if the pipeline fails to locate a note in the frame.
    """
    # --- Load ---
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_input}")
    else:
        img = image_input.copy()

    original = img.copy()

    # --- Step 1: Grayscale ---
    gray = _to_grayscale(img)

    # --- Step 2: Gaussian Blur ---
    blurred = _blur(gray)

    # --- Step 3: Adaptive Threshold ---
    thresh = _adaptive_threshold(blurred)

    # --- Step 4: Morphological Operations ---
    morph = _morphology(thresh)

    # --- Step 5: Find & Filter Contours → Detect Rectangle ---
    pts = _find_note_contour(morph)

    if pts is None:
        # Fallback: treat full image as the note boundary
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        fallback_used = True
    else:
        fallback_used = False

    # --- Step 6: Perspective Warp ---
    warped = _perspective_warp(original, pts)

    # --- Step 7: HSV + CLAHE ---
    normalised = _apply_clahe_hsv(warped)
    hsv = cv2.cvtColor(normalised, cv2.COLOR_BGR2HSV)

    if debug:
        debug_view = _build_debug_view(gray, thresh, morph, warped, normalised, pts, original, fallback_used)
        return normalised, hsv, debug_view

    return normalised, hsv


def preprocess_to_tensor(
    image_input,
    target_size: Tuple[int, int] = (WARP_H, WARP_W)
) -> Optional[np.ndarray]:
    """
    Convenience wrapper that returns a float32 numpy array
    normalised to [0, 1] with shape (H, W, 3) — ready for PyTorch.

    Parameters
    ----------
    image_input : str | np.ndarray
    target_size : (H, W) — resize output if different from default warp size.

    Returns
    -------
    np.ndarray of shape (H, W, 3) float32, or None on failure.
    """
    result = preprocess(image_input)
    if result[0] is None:
        return None

    bgr = result[0]
    if (bgr.shape[0], bgr.shape[1]) != target_size:
        bgr = cv2.resize(bgr, (target_size[1], target_size[0]))

    # BGR → RGB, normalise
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return tensor


def get_hsv_from_preprocessed(bgr: np.ndarray) -> np.ndarray:
    """Return HSV array from an already-preprocessed BGR image."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


# ---------------------------------------------------------------------------
# Debug visualisation helper
# ---------------------------------------------------------------------------

def _build_debug_view(
    gray, thresh, morph, warped, normalised, pts, original, fallback_used
) -> np.ndarray:
    """Stitch intermediate steps into a single debug image."""
    def to_bgr_resized(img, w=320, h=160):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (w, h))

    # Draw detected contour on original
    orig_copy = original.copy()
    if not fallback_used:
        cv2.polylines(orig_copy, [pts.astype(np.int32)], True, (0, 255, 0), 3)

    row1 = np.hstack([
        to_bgr_resized(orig_copy),
        to_bgr_resized(gray),
        to_bgr_resized(thresh),
    ])
    row2 = np.hstack([
        to_bgr_resized(morph),
        to_bgr_resized(warped),
        to_bgr_resized(normalised),
    ])

    labels = ["Original + Contour", "Grayscale", "Adaptive Thresh",
              "Morphology", "Warped", "CLAHE Normalised"]
    for i, label in enumerate(labels):
        row = row1 if i < 3 else row2
        x = (i % 3) * 320 + 5
        cv2.putText(row, label, (x if i < 3 else (i - 3) * 320 + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return np.vstack([row1, row2])


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path> [--debug]")
        sys.exit(1)

    path = sys.argv[1]
    debug = "--debug" in sys.argv

    if debug:
        bgr, hsv, dbg = preprocess(path, debug=True)
        out_path = "debug_preprocessing.jpg"
        cv2.imwrite(out_path, dbg)
        print(f"Debug view saved to {out_path}")
    else:
        bgr, hsv = preprocess(path)

    if bgr is not None:
        out = "preprocessed_output.jpg"
        cv2.imwrite(out, bgr)
        print(f"Preprocessed image saved → {out}  (shape: {bgr.shape})")
    else:
        print("ERROR: Could not locate a note in the image.")
