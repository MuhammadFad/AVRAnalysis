# AVR — Visual Regression Analyzer

A three-pass image quality analysis pipeline that compares a baseline frame against
an optimized version, detects degraded regions, and assigns a plain-English cause
hypothesis to each one. Available as a hosted web app and a local CLI tool.

**Live app → [your-app.vercel.app](https://your-app.vercel.app)**

---

## How it works

Every analysis runs three independent passes on the same image pair:

| Pass | Method | Detects |
|---|---|---|
| SSIM | Structural Similarity Index | Perceptual degradation — blur, ringing, blocking |
| Edge | Hand-rolled Canny detector | Structural loss — missing outlines, geometry decimation |
| Color | Bhattacharyya histogram distance | Color shifts — tonemapping, shader changes, hue drift |

The aggregator detects degraded regions from the SSIM map (the only pass that
produces a dense per-pixel spatial output), then evaluates each region locally
across all three passes using map slices (free) or histogram re-runs (necessary).
The combinator applies a deterministic truth table to assign a cause hypothesis
and confidence level to each region.

### Truth table

| SSIM | Edge | Color | Hypothesis | Confidence |
|:----:|:----:|:-----:|---|:---:|
| ✗ | ✓ | ✓ | Texture / surface detail lost. Geometry and lighting intact. | high |
| ✓ | ✗ | ✓ | Likely lighting change shifting edges. Low geometry concern. | low |
| ✓ | ✓ | ✗ | Global lighting / tone shift. No structural damage. | high |
| ✗ | ✗ | ✓ | Compression blocking artifact. DCT grid + perceptual blurring. | very_high |
| ✗ | ✓ | ✗ | Material / shader broke. Surface wrong, geometry fine. | high |
| ✓ | ✗ | ✗ | Lighting rig changed. Edge shift is shadow-driven. | medium |
| ✗ | ✗ | ✗ | Total visual failure. All three dimensions degraded. | definitive |
| ✓ | ✓ | ✓ | No degradation detected. | clean |

---

## File structure

```
root/
├── api.py                          # FastAPI app — /ping and /analyze
├── main.py                         # CLI entrypoint (unchanged)
├── requirements.txt                # Python deps for Render
├── render.yaml                     # Render deployment config
│
├── backend/
│   ├── config.py                   # All tunable parameters
│   ├── pipeline/
│   │   ├── acquisition.py          # Phase 1: load, align, cap resolution
│   │   ├── ssim_pass.py            # Phase 2a: SSIM
│   │   ├── edge_pass.py            # Phase 2b: Canny edge detection
│   │   ├── color_pass.py           # Phase 2c: Bhattacharyya color
│   │   ├── aggregator.py           # Phase 3: region detection + local scoring
│   │   └── combinator.py          # Phase 4: truth table → cause hypothesis
│   └── reporting/
│       ├── json_reporter.py        # Structured report (dict for API, file for CLI)
│       └── visual_reporter.py      # Heatmap + analysis figure
│
├── data/                           # Place CLI input images here
│   ├── baseline.png
│   └── optimized.png
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── App.jsx                 # Screen state machine: Loading → Upload → Results
        ├── hooks/
        │   └── usePing.js          # Polls /ping until Render warms up
        └── components/
            ├── LoadingScreen.jsx   # Cold-start screen
            ├── UploadScreen.jsx    # Drag-and-drop upload
            ├── ResultsScreen.jsx   # Tab controller
            ├── ZoomOverlay.jsx     # Full-screen zoom + hover crop card
            ├── RegionModal.jsx     # Per-region detail modal
            ├── SaveModal.jsx       # JSZip export
            └── tabs/
                ├── OverallTab.jsx  # Heatmap + region list + truth table
                ├── SSIMTab.jsx     # SSIM component maps
                ├── EdgeTab.jsx     # Edge maps grid + diff
                └── ColorTab.jsx    # HSV/RGB channels + overlay + histogram
```

---

## Local development

### Backend

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Test it's alive:
```bash
curl http://localhost:8000/ping
# {"status":"alive"}
```

Run an analysis:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "baseline=@path/to/baseline.png" \
  -F "optimized=@path/to/optimized.png"
```

### Frontend

```bash
cd frontend
cp .env.example .env.local   # already points to localhost:8000
npm install
npm run dev
# → http://localhost:5173
```

### CLI (no server needed)

```bash
python main.py data/baseline.png data/optimized.png
# outputs to output/
```

---

## Configuration

All tunable parameters live in `backend/config.py`. Nothing is hardcoded elsewhere.

| Parameter | Default | Effect |
|---|---|---|
| `SSIM_THRESHOLD` | `0.90` | Minimum acceptable SSIM score |
| `CANNY_EDGE_MATCH_THRESH` | `0.85` | Minimum fraction of baseline edges to preserve |
| `COLOR_DISTANCE_THRESH` | `0.30` | Maximum acceptable Bhattacharyya distance |
| `DEGRADATION_THRESH` | `0.30` | SSIM sensitivity for region detection (lower = more regions) |
| `MIN_REGION_AREA` | `500` | Minimum region size in px² (filters noise) |
| `MAX_IMAGE_DIM` | `1920` | Longest dimension cap after alignment (protects server RAM) |

---

## Environment variables

| Variable | Where | Value |
|---|---|---|
| `VITE_API_URL` | Vercel (frontend) | Your Render backend URL |
| *(none required)* | Render (backend) | Port is injected automatically via `$PORT` |
