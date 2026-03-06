# 🇱🇰 Sri Lankan Currency Verification System

Anomaly-detection pipeline for Rs. 100, 500, and 1000 banknotes using
Convolutional Autoencoders and region-weighted reconstruction error.

---

## Project Structure

```
currency_verifier/
├── preprocessing.py          # Image preprocessing pipeline
├── denomination_detector.py  # Hue+saturation histogram denomination detection
├── autoencoder.py            # CAE model definition + region-weighted error
├── train.py                  # Training loop for all 6 models
├── calibrate.py              # Threshold computation (T_front_X, T_back_X)
├── inference.py              # End-to-end verification pipeline
├── app.py                    # Desktop Tkinter UI
├── requirements.txt
└── models/                   # Created automatically after training
    ├── ae_100_F.pt
    ├── ae_100_B.pt
    ├── ae_500_F.pt
    ├── ae_500_B.pt
    ├── ae_1000_F.pt
    ├── ae_1000_B.pt
    ├── denomination_histograms.json
    └── thresholds.json
```

---

## Dataset Structure Required

```
Dataset/
├── 100_F/     ← genuine Rs.100 front images
├── 100_B/     ← genuine Rs.100 back images
├── 500_F/
├── 500_B/
├── 1000_F/
└── 1000_B/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

> **Minimum recommended images per folder:** 50  
> **More is better** — 200+ images per folder gives reliable thresholds.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Usage

### 1. Calibrate denomination detector

Builds reference hue+saturation histograms from your genuine images:

```bash
python denomination_detector.py calibrate Dataset
```

Output: `models/denomination_histograms.json`

---

### 2. Train autoencoders

Train all 6 models (100_F, 100_B, 500_F, 500_B, 1000_F, 1000_B):

```bash
python train.py --dataset Dataset --epochs 80
```

Train a single model:

```bash
python train.py --denom 500 --side F --epochs 100
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `Dataset` | Path to dataset root |
| `--epochs` | `80` | Max training epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

Training uses **early stopping** (patience=15 epochs) and saves the best
checkpoint per model. Loss curves are saved to `logs/train_log_XXX_Y.csv`.

---

### 3. Compute thresholds

After training, compute decision thresholds from validation data:

```bash
python calibrate.py --dataset Dataset
```

For tighter thresholds (flag more notes):

```bash
python calibrate.py --k 2.5
```

For looser thresholds (fewer false positives):

```bash
python calibrate.py --k 3.5
```

> **k** controls how many standard deviations above the mean genuine error
> the threshold is set. Default k=3.0 covers ~99.7% of genuine notes.

Output: `models/thresholds.json`

---

### 4. Launch the desktop app

```bash
python app.py
```

**UI workflow:**
1. Optionally select denomination (or leave on Auto-detect)
2. Click **Browse Front Image** → select front photo of note
3. Click **Browse Back Image** → select back photo of note
4. Click **🔍 Verify Note**
5. View verdict (GENUINE / SUSPICIOUS), per-side error bars, and detail scores

---

### 5. Command-line verification

```bash
python inference.py front.jpg back.jpg
```

With denomination override:

```bash
python inference.py front.jpg back.jpg --denom 1000
```

JSON output (for integration):

```bash
python inference.py front.jpg back.jpg --json
```

---

## How It Works

### Preprocessing pipeline
```
Input image
  → Grayscale → Gaussian blur → Adaptive threshold → Morphological ops
  → Find & filter contours → Detect rectangle
  → Perspective warp (1024×512 px)
  → HSV conversion + CLAHE (illumination normalisation)
```

### Denomination detection
- Computes a 2D Hue+Saturation histogram from the preprocessed front image
- Compares to per-denomination reference histograms using Bhattacharyya distance
- Returns the denomination with the lowest distance, or "unknown" if confidence is too low

### Autoencoder anomaly detection
- 6 separate Convolutional Autoencoders — one per denomination+side combination
- Trained **only on genuine notes** — learns to reconstruct the expected appearance
- Fake notes produce high reconstruction error (the model cannot reconstruct features it has never seen)

### Region-weighted reconstruction error
- The note area is divided into zones weighted by security feature density:
  - **Left strip** (watermark / windowed security thread): weight 3×
  - **Right strip** (serial number / colour-shift ink): weight 3×
  - **Central band** (latent image / microprinting): weight 3×
  - **Plain areas**: weight 1×
- Separate thresholds **T_front** and **T_back** per denomination
- If front_error > T_front **OR** back_error > T_back → **SUSPICIOUS**

---

## Threshold Tuning

After calibration, check `models/thresholds.json` and compare:
- `p95` — 95% of genuine notes fall below this
- `p99` — 99% of genuine notes fall below this
- `threshold` — the actual decision boundary (mean + k×std)

If you're seeing too many false positives (genuine flagged as suspicious),
increase k: `python calibrate.py --k 3.5`

If you want to be more aggressive: `python calibrate.py --k 2.5`

---

## Debug Mode

Visualise the preprocessing steps for any image:

```bash
python preprocessing.py my_note.jpg --debug
# Saves debug_preprocessing.jpg showing all pipeline stages
```

Test denomination detection:

```bash
python denomination_detector.py detect my_note.jpg
```

---

## Notes

- The system is trained on **genuine notes only** — you do not need fake note images for training.
- Performance improves significantly with more training data (aim for 100+ images per folder).
- Ensure consistent capture conditions (flat surface, good lighting) for best results. The CLAHE normalisation handles moderate variation but not extreme cases.
- The weight map in `autoencoder.py → build_weight_map()` can be tuned per denomination once you understand where anomalies typically appear in your target note series.
