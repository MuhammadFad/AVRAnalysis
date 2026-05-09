# ================================================================================
# aggregator.py — Phase 3: Degraded Region Detection & Spatial Attribution
# ================================================================================
#
# PURPOSE:
# This module processes SSIM results to:
#   1. Visualize regions of visual degradation as a heatmap
#   2. Detect and localize specific degraded areas using morphological operations
#   3. For each detected region, compute per-pass local scores using already-
#      computed maps (no redundant re-runs of Canny or SSIM)
#   4. Return structured combined regions tagged with which passes fired locally
#
# ROLE IN PIPELINE:
# This is Phase 3 of the analysis. It receives:
#   - The per-pixel SSIM map (from ssim_pass)
#   - The already-computed edge maps (from edge_pass) — sliced, not re-run
#   - The color pass params (color is re-run locally — no dense spatial map exists)
#   - Both original images (for local color crops)
# It produces:
#   - A visual heatmap (hot colormap: dark=good, yellow=bright=bad)
#   - A list of structured combined region dicts (consumed by combinator.py)
#
# DESIGN NOTE — Why SSIM is the region anchor:
#   SSIM produces a dense per-pixel similarity map, making it the natural source
#   for spatial region detection. Edge and color passes produce only global scalars.
#   For each SSIM-detected region we then evaluate edge and color *locally*:
#     - Edge local score:  slice already-computed edge_baseline / edge_optimized maps
#                          → zero extra Canny computation
#     - SSIM local score:  slice already-computed ssim_map
#                          → zero extra SSIM computation
#     - Color local score: re-run histogram on the crop
#                          → unavoidable; no dense color spatial map exists
#   Future work (FYP): add independent spatial detection to edge (lost-edge contours)
#   and color (tile-grid Bhattacharyya), then IoU-cluster all three region lists.
#
# OUTPUT SCHEMA — each combined region dict:
#   {
#     "bbox":         (x, y, w, h),       # pixel coords in original image space
#     "passes_failed": ["ssim", "edge"],  # which passes fired for this region
#     "local_scores":  {                  # per-pass local quality scores
#         "ssim":  float,                 # mean SSIM in crop (lower = worse)
#         "edge":  float | None,          # edge match score in crop (lower = worse)
#         "color": float | None,          # mean Bhattacharyya distance in crop
#     }
#   }
#
# ================================================================================

import numpy as np
import cv2
import matplotlib

from backend.pipeline.color_pass import run as run_color


def build_heatmap(img_baseline:  np.ndarray,
                  img_optimized: np.ndarray,
                  ssim_map:      np.ndarray,
                  edge_baseline: np.ndarray,
                  edge_optimized: np.ndarray,
                  params:        dict):
    """
    Build visual heatmap, detect degraded regions, and compute per-region
    local scores for all three passes.

    Local SSIM and edge scores are derived by slicing already-computed maps —
    no redundant pipeline re-runs. Color is re-run on each crop (the only pass
    without a dense spatial map).

    INPUTS:
        img_baseline:   H × W × 3 float64 [0,1] — reference image
        img_optimized:  H × W × 3 float64 [0,1] — test image
        ssim_map:       H × W float64 [0,1]      — per-pixel SSIM scores
        edge_baseline:  H × W uint8              — Canny map for baseline
        edge_optimized: H × W uint8              — Canny map for optimized image
        params: Dictionary with keys:
            - 'heatmap_alpha':           Blend factor between heatmap and image
            - 'degradation_thresh':      Threshold for SSIM-based region detection
            - 'morph_radius':            Morphological closing radius (in pixels)
            - 'min_region_area':         Minimum bounding box area (in pixels²)
            - 'ssim_threshold':          SSIM pass threshold (for local verdict)
            - 'canny_edge_match_thresh': Edge match threshold (for local verdict)
            - 'color_distance_thresh':   Bhattacharyya threshold (for local verdict)
            - (all color params forwarded to per-region color run)

    RETURNS:
        (composite, combined_regions) tuple:
        - composite:        H × W × 3 float64 image (heatmap + original blended)
        - combined_regions: List of region dicts (schema described in module docstring)
    """
    # ========================================================================
    # STEP 1: Create degradation map
    # ========================================================================
    # Invert SSIM so high degradation → high value, suitable for hot colormap.
    degradation_map = np.clip(1.0 - ssim_map, 0.0, 1.0)

    # ========================================================================
    # STEP 2: Apply 'hot' colormap for visualization
    # ========================================================================
    # 'hot' colormap: Black → Red → Yellow
    #   - 0.0 (no degradation) → Black/dark (healthy)
    #   - 0.5 (moderate)       → Red (warning)
    #   - 1.0 (severe)         → Yellow/bright (critical)
    heatmap_rgb = matplotlib.colormaps['hot'](degradation_map)[:, :, :3]

    # ========================================================================
    # STEP 3: Alpha-blend heatmap with original image
    # ========================================================================
    # Preserving some of the original provides spatial context for each region.
    alpha     = params['heatmap_alpha']
    composite = alpha * heatmap_rgb + (1.0 - alpha) * img_optimized
    composite = np.clip(composite, 0.0, 1.0)

    # ========================================================================
    # STEP 4: Create binary mask of degraded regions
    # ========================================================================
    binary_mask = (degradation_map > params['degradation_thresh']).astype(np.uint8)

    # ========================================================================
    # STEP 5: Morphological closing to reduce noise
    # ========================================================================
    # Closing fills small holes within degraded regions and removes isolated noise.
    r    = params['morph_radius']
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, disk)

    # ========================================================================
    # STEP 6: Find contours and extract bounding boxes
    # ========================================================================
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > params['min_region_area']:
            boxes.append(cv2.boundingRect(cnt))   # (x, y, w, h)

    # ========================================================================
    # STEP 7: Per-region local scoring
    # ========================================================================
    # Compute local scores for each pass within each SSIM-detected bounding box.
    # SSIM and edge use map slices (free). Color re-runs on the crop (necessary).
    combined_regions = _score_regions(
        boxes,
        ssim_map,
        edge_baseline,
        edge_optimized,
        img_baseline,
        img_optimized,
        params,
    )

    return composite, combined_regions


def _score_regions(boxes:          list,
                   ssim_map:       np.ndarray,
                   edge_baseline:  np.ndarray,
                   edge_optimized: np.ndarray,
                   img_baseline:   np.ndarray,
                   img_optimized:  np.ndarray,
                   params:         dict) -> list:
    """
    Compute per-region local scores for all three passes and return structured
    combined region dicts.

    For SSIM and edge, local scores are derived by slicing already-computed
    full-image maps — no extra pipeline computation. For color, a histogram
    comparison is re-run on the cropped region (no dense spatial map exists).

    INPUTS:
        boxes:          List of (x, y, w, h) tuples from SSIM contour detection
        ssim_map:       H × W float64 — full-image SSIM scores (from ssim_pass)
        edge_baseline:  H × W uint8   — Canny map for baseline (from edge_pass)
        edge_optimized: H × W uint8   — Canny map for optimized (from edge_pass)
        img_baseline:   H × W × 3 float64 — full baseline image (for color crop)
        img_optimized:  H × W × 3 float64 — full optimized image (for color crop)
        params:         Full params dict

    RETURNS:
        List of combined region dicts. See module docstring for schema.
    """
    # Suppress intermediate saves for per-region color sub-calls —
    # these are diagnostic crops, not primary pipeline outputs.
    crop_params = {**params, 'save_intermediates': False}

    edge_thresh  = params['canny_edge_match_thresh']
    ssim_thresh  = params['ssim_threshold']
    color_thresh = params['color_distance_thresh']

    combined_regions = []

    for (x, y, w, h) in boxes:
        passes_failed = ['ssim']  # SSIM always fires — it's why this box exists

        # ---- Local SSIM score: mean of ssim_map within bbox ----
        # Free operation — just a numpy slice and mean.
        ssim_crop   = ssim_map[y:y+h, x:x+w]
        local_ssim  = float(np.mean(ssim_crop))

        # ---- Local edge score: match rate within bbox edge map slices ----
        # Free operation — slices of the already-computed Canny maps.
        local_edge  = None
        if edge_baseline is not None and edge_optimized is not None:
            eb_crop = edge_baseline [y:y+h, x:x+w]
            eo_crop = edge_optimized[y:y+h, x:x+w]
            baseline_count = np.count_nonzero(eb_crop)
            if baseline_count > 0:
                matched_count = np.count_nonzero(eb_crop & eo_crop)
                local_edge    = float(matched_count / baseline_count)
                # Edge FAILS locally when match rate drops below threshold
                if local_edge < edge_thresh:
                    passes_failed.append('edge')
            # If no edges in this crop, edge pass is trivially satisfied — skip

        # ---- Local color score: re-run histogram on the crop ----
        # Color has no dense spatial map; histogram must be recomputed per crop.
        # Skip crops too small to yield meaningful histograms.
        local_color = None
        if w >= 3 and h >= 3:
            try:
                crop_base = img_baseline [y:y+h, x:x+w]
                crop_opt  = img_optimized[y:y+h, x:x+w]
                color_r   = run_color(crop_base, crop_opt, crop_params)
                local_color = color_r['score']
                if not color_r['passed']:
                    passes_failed.append('color')
            except Exception:
                # Pathological crops (e.g., fully uniform patches) may produce
                # degenerate histograms. Skip gracefully — SSIM tag is sufficient.
                pass

        combined_regions.append({
            "bbox":          (x, y, w, h),
            "passes_failed": passes_failed,
            "local_scores":  {
                "ssim":  local_ssim,
                "edge":  local_edge,
                "color": local_color,
            },
        })

    return combined_regions
