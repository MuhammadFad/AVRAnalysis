# Visual Regression Analyzer

A Python pipeline for comparing a baseline image against an optimized version across three independent quality dimensions — perceptual similarity, structural edge preservation, and color fidelity. Designed for catching visual regressions introduced by rendering optimizations, compression, or post-processing.

---

## How it works

The pipeline runs three analysis passes on the same image pair and aggregates their results into a set of reports.

**SSIM** measures perceptual similarity by comparing local brightness, texture variation, and edge patterns between the two images. It produces a per-pixel score map, detects degraded regions, and tags each region with which other passes also flagged it.

**Canny Edge** detects structural edges in both images and measures what fraction of baseline edges are still present in the optimized output. Useful for catching optimizations that silently smooth over fine geometric detail.

**Bhattacharyya Color** converts both images to HSV and computes histogram overlap per channel. Catches color grading shifts, saturation changes, and tonemapping differences that the other passes may not surface.

Each detected degraded region gets tagged by whichever passes flagged it locally, so the output tells you not just *where* something looks wrong but *why*.

---

## Project structure

```
root/
├── main.py                        # Entry point — run from here
├── data/                          # Place your input images here
│   ├── baseline.png
│   └── optimized.png
├── output/                        # All results written here (auto-created)
│   ├── regression_report.json     # Machine-readable results for all three passes
│   ├── heatmap_composite.png      # SSIM heatmap with color-coded region boxes
│   ├── analysis_figure.png        # Full multi-panel pipeline report
│   ├── ssim/                      # SSIM component maps (luminance, contrast, structure)
│   ├── edge/                      # Canny intermediates (Sobel X/Y, NMS, edge maps)
│   └── color/                     # Bhattacharyya histogram overlay
└── backend/
    ├── config.py                  # All tunable parameters live here
    ├── pipeline/
    │   ├── acquisition.py         # Image loading and normalization
    │   ├── ssim_pass.py           # SSIM computation
    │   ├── edge_pass.py           # Canny edge detection and comparison
    │   ├── color_pass.py          # Bhattacharyya color distribution comparison
    │   └── aggregator.py          # Heatmap generation and per-region tagging
    └── reporting/
        ├── json_reporter.py       # JSON report serialization
        └── visual_reporter.py     # PNG figure and heatmap generation
```

---

## Setup

Requires Python 3.8+.

```bash
pip install numpy opencv-python scipy matplotlib
```

---

## Usage

```bash
python main.py data/baseline.png data/optimized.png
```

If no arguments are provided it defaults to `data/baseline.png` and `data/optimized.png`.

---

## Output

**`regression_report.json`** — structured results for all three passes, including per-region bounding boxes with a `filters` list indicating which passes flagged each one.

**`heatmap_composite.png`** — the SSIM degradation heatmap with color-coded bounding boxes:

| Box color | Meaning |
|---|---|
| 🟡 Yellow | SSIM only — perceptual diff, edges and color OK |
| 🟠 Orange | SSIM + Edge — structural detail also lost |
| 🔵 Blue | SSIM + Color — color profile also shifted |
| 🔴 Red | All three — fully degraded region |

**`analysis_figure.png`** — a multi-panel report showing the input images, SSIM component maps, Canny edge maps, and a summary sidebar with all three pass scores and verdicts.

**`output/ssim/`**, **`output/edge/`**, **`output/color/`** — intermediate diagnostic images for each pass, written when `SAVE_INTERMEDIATES = True` in `config.py`.

---

## Configuration

All thresholds and tunable parameters are centralized in `backend/config.py`. Nothing is hardcoded elsewhere.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `SSIM_THRESHOLD` | `0.90` | Minimum acceptable SSIM score |
| `CANNY_EDGE_MATCH_THRESH` | `0.85` | Minimum fraction of baseline edges to preserve |
| `COLOR_DISTANCE_THRESH` | `0.30` | Maximum acceptable Bhattacharyya distance |
| `SAVE_INTERMEDIATES` | `True` | Whether to write per-pass diagnostic images |
| `DEGRADATION_THRESH` | `0.30` | SSIM map sensitivity for region detection |
