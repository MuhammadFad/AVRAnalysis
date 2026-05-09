# Visual Regression Analyzer

**DIP Semester Project — May 2026**

A multi-pass image quality analysis pipeline for detecting and diagnosing visual degradation in compressed video frames. The system compares a high-quality baseline frame against a compressed/optimized version across three independent perceptual dimensions — structural similarity, edge preservation, and color fidelity — then applies a deterministic truth table to assign a plain-English cause hypothesis to every degraded region it finds.

---

## Table of Contents

1. [Setup](#setup)
2. [Running the Pipeline](#running-the-pipeline)
3. [Output Artifacts](#output-artifacts)
4. [Pipeline Architecture](#pipeline-architecture)
5. [The Three Passes](#the-three-passes)
   - [Pass 1 — SSIM](#pass-1--ssim-structural-similarity)
   - [Pass 2 — Edge (Canny)](#pass-2--edge-canny)
   - [Pass 3 — Color (Bhattacharyya)](#pass-3--color-bhattacharyya)
6. [Aggregation & Region Detection](#aggregation--region-detection)
7. [The Combinator Truth Table](#the-combinator-truth-table)
8. [Bounding Box Color Coding](#bounding-box-color-coding)
9. [Configuration Reference](#configuration-reference)
10. [File Structure](#file-structure)
11. [Future Work](#future-work)

---

## Setup

**Requirements:** Python 3.9+

Install dependencies:

```bash
pip install numpy scipy opencv-python matplotlib
```

Place your images in the `data/` directory:

```
data/
├── baseline.png     # high-quality reference frame
└── optimized.png    # compressed / test frame
```

Images must be:
- The same resolution (width × height must match exactly)
- RGB, 3-channel (no RGBA, no grayscale)
- Any standard format readable by OpenCV (PNG, JPG, BMP, TIFF)

---

## Running the Pipeline

**Default** — reads from `data/baseline.png` and `data/optimized.png`:

```bash
python main.py
```

**Explicit paths:**

```bash
python main.py path/to/baseline.png path/to/optimized.png
```

**Example output:**

```
===============================
 VISUAL REGRESSION ANALYZER
===============================

[Phase 1] Loading images...
  >> Images loaded: 1920 x 1080 pixels.

[Phase 2a] Computing SSIM...
  >> Score     : 0.8312
  >> Threshold : 0.9
  >> Result    : FAIL ✗

[Phase 2b] Running Canny edge pass...
  >> Edge match : 0.7841
  >> Threshold  : 0.85
  >> Result     : FAIL ✗

[Phase 2c] Running Bhattacharyya color pass...
  >> Mean distance : 0.0441
  >> Per channel   : H=0.0312  S=0.0589  V=0.0423
  >> Threshold     : 0.3
  >> Result        : PASS ✓

[Phase 3] Building heatmap and detecting degraded regions...
  >> Degraded regions detected: 4

[Phase 4] Applying combinator truth table...
  >> Region (142,88) 210×156px  [very_high]  blocking artifact
  >> Region (540,312) 88×72px   [high]       texture loss
  >> Region (902,200) 124×98px  [very_high]  blocking artifact
  >> Region (310,600) 66×54px   [high]       shader/material

[Reporting] Saving outputs...
  >> Histogram overlay : output/color/histogram_overlay.png
  >> Heatmap           : output/heatmap_composite.png
  >> JSON report       : output/regression_report.json
  >> Analysis figure   : output/analysis_figure.png

===============================
 FINAL VERDICT: ✗ FAIL
 Failed passes: SSIM, Edge
===============================
 Output saved to: output/
===============================
```

---

## Output Artifacts

All outputs are written to the `output/` directory, which is created automatically.

| File | Description |
|---|---|
| `heatmap_composite.png` | SSIM degradation heatmap with color-coded bounding boxes and cause labels burned onto each region. Self-contained — no separate legend file needed. |
| `analysis_figure.png` | Full multi-panel pipeline report. Image panels (baseline, optimized, heatmap, lost edges, SSIM components, color histograms), a scores row with all three pass results, and a horizontal region strip with per-region diagnosis. |
| `regression_report.json` | Machine-readable results. Includes global scores, verdicts, and a full entry per degraded region with cause hypothesis, confidence, and local scores. Sorted worst-first. |
| `ssim/luminance_component.png` | Local brightness similarity map (SSIM L component) |
| `ssim/contrast_component.png` | Local texture variation similarity map (SSIM C component) |
| `ssim/structure_component.png` | Local edge-pattern correlation map (SSIM S component) |
| `edge/sobel_x.png` | Horizontal Sobel gradient of the baseline |
| `edge/sobel_y.png` | Vertical Sobel gradient of the baseline |
| `edge/nms_map.png` | Gradient magnitude after non-maximum suppression |
| `edge/edge_baseline.png` | Final Canny edge map for the baseline |
| `edge/edge_optimized.png` | Final Canny edge map for the optimized image |
| `color/histogram_overlay.png` | H, S, V channel histogram overlay (baseline vs optimized) |

Intermediate files (`ssim/`, `edge/`, `color/`) are only written when `SAVE_INTERMEDIATES = True` in `config.py`.

---

## Pipeline Architecture

The pipeline runs six sequential phases:

```
Phase 1  ACQUISITION
         Load and normalize both images → float64 [0, 1] arrays

Phase 2  ANALYSIS  (three independent passes)
    2a   SSIM      → global score + per-pixel similarity map
    2b   Edge      → global score + baseline/optimized edge maps
    2c   Color     → global score + per-channel histogram data

Phase 3  AGGREGATION
         Detect degraded regions from SSIM map (morphological ops)
         Compute local scores per region:
           · SSIM local  = mean of ssim_map crop        (free — map slice)
           · Edge local  = match rate of edge map crops  (free — map slice)
           · Color local = re-run histogram on crop      (necessary)

Phase 4  COMBINATION
         Apply truth table to each region's pass attribution
         → cause hypothesis + confidence per region

Phase 5  JSON REPORTING
         Serialize all results + enriched regions → regression_report.json

Phase 6  VISUAL REPORTING
         heatmap_composite.png  — annotated heatmap
         analysis_figure.png    — full multi-panel report
```

**Key design decision — SSIM as the region anchor:**
Only SSIM produces a dense per-pixel spatial map, making it the natural source for region detection. Edge and color passes produce only global scalars. For each SSIM-detected region, the aggregator evaluates edge and color *locally* by slicing the already-computed full-image edge maps (zero extra Canny computation) and re-running the color histogram on the crop. This avoids running the full Canny pipeline once per region.

---

## The Three Passes

### Pass 1 — SSIM (Structural Similarity)

**File:** `backend/pipeline/ssim_pass.py`

SSIM measures perceptual image similarity the way a human would — not by counting different pixels, but by comparing local brightness, texture variation, and edge structure simultaneously. It produces both a global score and a per-pixel map showing *where* degradation is concentrated.

**Formula** (Wang et al., 2004):

```
SSIM(x, y) = L(x,y) × C(x,y) × S(x,y)

L = (2μ₁μ₂ + C1) / (μ₁² + μ₂² + C1)        luminance
C = (2σ₁σ₂ + C2) / (σ₁² + σ₂² + C2)        contrast
S = (σ₁₂ + C3)   / (σ₁σ₂ + C3)             structure
```

All statistics are computed locally using a Gaussian-weighted sliding window (size 11×11, σ=1.5 by default). This gives a per-pixel SSIM map rather than a single global comparison.

**Score range:** 0.0 (completely different) → 1.0 (identical)
**Default threshold:** ≥ 0.90

**Implementation note:** Hand-rolled using `scipy.ndimage.convolve` — no `skimage` dependency.

---

### Pass 2 — Edge (Canny)

**File:** `backend/pipeline/edge_pass.py`

The Canny pass asks: "Has the optimized image preserved the structural edges — outlines, boundaries, fine geometry — present in the baseline?" An image can have high SSIM (looks broadly similar) while quietly losing fine edge detail. This pass catches that.

**Algorithm** (fully hand-rolled — no `cv2.Canny`):

```
1. Grayscale conversion   ITU-R BT.601 weights (green > red > blue)
2. Gaussian blur          Suppress noise before gradient computation
3. Sobel gradients        Compute gradient magnitude and direction
4. Non-max suppression    Thin edges from blurry ridges to 1-pixel-wide lines
5. Double thresholding    Classify pixels as strong / weak / non-edge
6. Hysteresis             Promote weak pixels connected to strong ones
```

**Score:** fraction of baseline edge pixels also present in the optimized map.

```
edge_match = |edge_baseline ∩ edge_optimized| / |edge_baseline|
```

A score of 1.0 means every structural edge was preserved. A score of 0.7 means 30% of baseline edges were lost — indicative of blocking or blurring artifacts.

**Default threshold:** ≥ 0.85

---

### Pass 3 — Color (Bhattacharyya)

**File:** `backend/pipeline/color_pass.py`

The color pass detects shifts in the overall color distribution — tonemapping changes, saturation differences, or hue drift — that SSIM and edge detection might miss entirely because they don't look at color directly.

**Algorithm:**

```
1. Convert both images to HSV
   (separates hue/saturation from brightness — better than RGB for color matching)
2. Compute normalized histogram per channel (H, S, V)
   64 bins each; histograms sum to 1.0 (probability distributions)
3. Compute Bhattacharyya coefficient per channel
   BC(P,Q) = Σᵢ √(P(i) × Q(i))    — measures histogram overlap [0,1]
4. Convert to distance
   D = −ln(BC)                      — 0.0 = identical, ∞ = no overlap
5. Average across H, S, V channels → overall score
```

**Score range:** 0.0 (identical distributions) → ∞ (no overlap)
**Default threshold:** ≤ 0.30

---

## Aggregation & Region Detection

**File:** `backend/pipeline/aggregator.py`

After the three global passes, the aggregator localizes *where* the degradation is using the SSIM map:

```
1. Invert SSIM map → degradation map (high value = bad)
2. Threshold at DEGRADATION_THRESH → binary mask
3. Morphological closing (radius 15px) → fill holes, reduce noise
4. Find external contours → candidate bounding boxes
5. Filter by MIN_REGION_AREA → discard noise regions
6. For each surviving box, compute local scores:
     SSIM local  = mean(ssim_map[y:y+h, x:x+w])
     Edge local  = count_nonzero(eb_crop & eo_crop) / count_nonzero(eb_crop)
     Color local = Bhattacharyya on cropped region histograms
```

Each region is returned as a structured dict:

```json
{
  "bbox": [x, y, w, h],
  "passes_failed": ["ssim", "edge"],
  "local_scores": {
    "ssim":  0.61,
    "edge":  0.43,
    "color": 0.08
  }
}
```

---

## The Combinator Truth Table

**File:** `backend/pipeline/combinator.py`

The combinator is a pure function — no image operations, no spatial math. It takes each region's `passes_failed` list and maps it deterministically to a cause hypothesis and confidence level.

| SSIM | Edge | Color | Cause Hypothesis | Confidence |
|:----:|:----:|:-----:|---|:---:|
| ✗ | ✓ | ✓ | Texture / surface detail lost. Geometry and lighting intact. | `high` |
| ✓ | ✗ | ✓ | Likely lighting change shifting edges. Low geometry concern. | `low` |
| ✓ | ✓ | ✗ | Global lighting / tone shift only. No structural damage. | `high` |
| ✗ | ✗ | ✓ | **Compression blocking artifact.** DCT grid edges detected; perceptual blurring and ringing around blocks. | `very_high` |
| ✗ | ✓ | ✗ | Material / shader broke. Surface looks wrong, geometry fine. | `high` |
| ✓ | ✗ | ✗ | Lighting rig changed. Edge shift is shadow-driven, not geometry. | `medium` |
| ✗ | ✗ | ✗ | Total visual failure. All three dimensions degraded. | `definitive` |
| ✓ | ✓ | ✓ | No degradation detected. | `clean` |

**Why is Canny-only `low` confidence?**
A lighting change shifts edge positions without any real geometry damage — Canny fires because edges moved, not because anything was structurally lost. The moment SSIM also fires on the same region, confidence jumps to `very_high` because two independent passes independently agreeing is a much stronger signal.

**Compression blocking special case:**
When SSIM and Canny both fail on the same region (the `✗ ✗ ✓` row), the hypothesis explicitly names DCT blocking. Canny detects the sharp grid of block boundaries that DCT introduces at high compression ratios; SSIM detects the perceptual blurring and ringing around those same blocks. Their co-occurrence is a reliable signature of this specific artifact.

**Short cause tags** (shown on the heatmap bounding boxes):

| Combination | Tag |
|---|---|
| SSIM only | `texture loss` |
| Edge only | `edge shift` |
| Color only | `tone shift` |
| SSIM + Edge | `blocking artifact` |
| SSIM + Color | `shader/material` |
| Edge + Color | `lighting rig` |
| All three | `total failure` |
| None | `clean` |

---

## Bounding Box Color Coding

Both the heatmap and analysis figure color-code bounding boxes by which passes fired:

| Color | Combination | Meaning |
|---|---|---|
| 🟡 Yellow | SSIM only | Perceptual difference; structure and color intact |
| 🟠 Orange | SSIM + Edge | Structural detail also broken |
| 🔵 Blue | SSIM + Color | Color profile also shifted |
| 🔴 Red | All three | Fully degraded region |

Each box also has a short cause tag burned above it (e.g. `blocking artifact`, `texture loss`) so the heatmap is interpretable without opening any other file.

---

## Configuration Reference

All tunable parameters live in `backend/config.py`. Nothing is hardcoded elsewhere.

### SSIM

| Parameter | Default | Description |
|---|---|---|
| `SSIM_THRESHOLD` | `0.90` | Minimum acceptable SSIM score |
| `SSIM_WINDOW_SIZE` | `11` | Gaussian kernel size for local statistics |
| `SSIM_SIGMA` | `1.5` | Gaussian kernel standard deviation |

### Region Detection

| Parameter | Default | Description |
|---|---|---|
| `HEATMAP_ALPHA` | `0.55` | Heatmap / original blend ratio |
| `DEGRADATION_THRESH` | `0.30` | SSIM degradation sensitivity (lower = more regions) |
| `MORPH_RADIUS` | `15` | Morphological closing radius in pixels |
| `MIN_REGION_AREA` | `500` | Minimum region area in pixels² |

### Edge (Canny)

| Parameter | Default | Description |
|---|---|---|
| `CANNY_BLUR_KERNEL` | `5` | Pre-detection Gaussian blur kernel size (must be odd) |
| `CANNY_BLUR_SIGMA` | `1.0` | Pre-detection Gaussian blur standard deviation |
| `CANNY_LOW_THRESH` | `50` | Lower hysteresis threshold |
| `CANNY_HIGH_THRESH` | `150` | Upper hysteresis threshold |
| `CANNY_EDGE_MATCH_THRESH` | `0.85` | Minimum fraction of baseline edges to preserve |

### Color (Bhattacharyya)

| Parameter | Default | Description |
|---|---|---|
| `COLOR_HIST_BINS` | `64` | Histogram bins per HSV channel |
| `COLOR_DISTANCE_THRESH` | `0.30` | Maximum acceptable mean Bhattacharyya distance |

### Combinator / Reporting

| Parameter | Default | Description |
|---|---|---|
| `IOU_THRESHOLD` | `0.3` | IoU threshold for region co-location (future use) |
| `TOP_REGIONS_IN_REPORT` | `5` | Max regions shown in the analysis figure strip |

### Output

| Parameter | Default | Description |
|---|---|---|
| `SAVE_INTERMEDIATES` | `True` | Write per-pass diagnostic images to subdirectories |
| `OUTPUT_DIR` | `output` | Root output directory |

---

## File Structure

```
root/
├── main.py                          # Entry point — run from here
├── README.md                        # This file
│
├── data/                            # Input images (place yours here)
│   ├── baseline.png                 # High-quality reference frame
│   └── optimized.png                # Compressed / test frame
│
├── output/                          # All results written here (auto-created)
│   ├── regression_report.json       # Machine-readable results (all passes + regions)
│   ├── heatmap_composite.png        # SSIM heatmap with cause-labelled region boxes
│   ├── analysis_figure.png          # Full multi-panel pipeline report
│   ├── ssim/
│   │   ├── luminance_component.png  # SSIM L map — local brightness similarity
│   │   ├── contrast_component.png   # SSIM C map — local texture variation
│   │   └── structure_component.png  # SSIM S map — local edge-pattern correlation
│   ├── edge/
│   │   ├── sobel_x.png              # Horizontal gradient magnitude
│   │   ├── sobel_y.png              # Vertical gradient magnitude
│   │   ├── nms_map.png              # After non-maximum suppression
│   │   ├── edge_baseline.png        # Final Canny map — baseline
│   │   └── edge_optimized.png       # Final Canny map — optimized
│   └── color/
│       └── histogram_overlay.png    # HSV histogram overlay (baseline vs optimized)
│
└── backend/
    ├── config.py                    # All tunable parameters — edit here only
    │
    ├── pipeline/
    │   ├── acquisition.py           # Phase 1: image loading and normalization
    │   ├── ssim_pass.py             # Phase 2a: SSIM computation (hand-rolled)
    │   ├── edge_pass.py             # Phase 2b: Canny edge detection (hand-rolled)
    │   ├── color_pass.py            # Phase 2c: Bhattacharyya color comparison
    │   ├── aggregator.py            # Phase 3: region detection + local scoring
    │   └── combinator.py           # Phase 4: truth table → cause hypothesis
    │
    └── reporting/
        ├── json_reporter.py         # Phase 5: regression_report.json
        └── visual_reporter.py       # Phase 6: heatmap + analysis figure
```

---

## Future Work

The following extensions were scoped out of this project but are natural next steps for a Final Year Project:

**Independent spatial detection per pass**
Currently SSIM is the sole source of bounding boxes and edge/color are evaluated locally within each SSIM region. A richer approach would have the edge pass detect regions from the lost-edge map (contours on `baseline_edges ∩ ¬optimized_edges`) and the color pass use a tile-grid Bhattacharyya scan, then IoU-cluster all three region lists. The `IOU_THRESHOLD` config parameter is already in place for this.

**LLM semantic reasoning layer**
Batched region crops sent to a multimodal LLM for natural language description. The combinator currently produces rule-based hypotheses from a fixed truth table; an LLM layer could describe more nuanced artifacts (mosquito noise, banding gradients) that don't reduce cleanly to the three-pass schema.

**UE5 segmentation pass**
A custom depth/stencil render pass for per-object region tagging in game engine footage, enabling per-mesh degradation attribution rather than purely spatial bounding boxes.

**Multi-view screenshot approach**
Comparing the same scene from multiple camera angles to separate viewpoint-dependent artifacts (specular highlights, reflections) from genuine geometric or texture degradation.
