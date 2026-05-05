# ================================================================================
# aggregator.py — Phase 3: Degraded Region Detection & Multi-Pass Tagging
# ================================================================================
#
# PURPOSE:
# This module processes SSIM results to:
#   1. Visualize regions of visual degradation as a heatmap
#   2. Detect and localize specific degraded areas using morphological operations
#   3. For each detected region, run localized edge and color checks to determine
#      which of the three analysis passes flagged it as degraded
#   4. Return tagged bounding boxes that carry per-region filter attribution
#
# ROLE IN PIPELINE:
# This is Phase 3 of the analysis. It receives the per-pixel SSIM map, the
# full-image edge and color results, and both original images. It produces:
#   - A visual heatmap (hot colormap: dark=good, yellow=bad)
#   - Alpha-blended composite with original image for context
#   - Tagged bounding boxes: each box knows which filters flagged it
#
# TAGGING LOGIC:
# SSIM flags a region by definition — boxes only exist because SSIM detected
# degradation there. Edge and color are then evaluated *within that crop*:
#   - Edge:  compute local edge match score for the cropped region
#   - Color: compute local Bhattacharyya distance for the cropped region
# If either exceeds its threshold, that filter's tag is added to the box.
#
# BOX COLOR MEANING (rendered by visual_reporter.py):
#   Yellow — SSIM only          (perceptual diff, structure/color OK)
#   Orange — SSIM + Edge        (structure also broken)
#   Blue   — SSIM + Color       (color profile also shifted)
#   Red    — All three          (fully degraded region)
#
# ================================================================================

import numpy as np
import cv2
import matplotlib

from backend.pipeline.edge_pass  import run as run_edge
from backend.pipeline.color_pass import run as run_color


def build_heatmap(img_baseline:  np.ndarray,
                  img_optimized: np.ndarray,
                  ssim_map:      np.ndarray,
                  params:        dict):
    """
    Build visual heatmap, detect degraded regions, and tag each region by
    which analysis passes flagged it.

    INPUTS:
        img_baseline:  H × W × 3 float64 [0,1] — reference image
        img_optimized: H × W × 3 float64 [0,1] — test image
        ssim_map:      H × W float64 [0,1]      — per-pixel SSIM scores
        params: Dictionary with keys:
            - 'heatmap_alpha':           Blend factor between heatmap and image
            - 'degradation_thresh':      Threshold for SSIM-based region detection
            - 'morph_radius':            Morphological closing radius (in pixels)
            - 'min_region_area':         Minimum bounding box area (in pixels²)
            - 'canny_edge_match_thresh': Edge match threshold for per-region check
            - 'color_distance_thresh':   Bhattacharyya threshold for per-region check
            - (all other edge/color params forwarded to their respective passes)

    RETURNS:
        (composite, tagged_boxes) tuple:
        - composite:    H × W × 3 float64 image (heatmap + original blended)
        - tagged_boxes: List of (x, y, w, h, flags) tuples where flags is a
                        frozenset containing any of {'ssim', 'edge', 'color'}
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
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    # ========================================================================
    # STEP 7: Per-region multi-pass tagging
    # ========================================================================
    # For each SSIM-detected box, crop both images to that region and run
    # localized edge and color checks. This tells us not just *where* the
    # degradation is, but *which dimensions* of quality broke down there.
    tagged_boxes = _tag_boxes(boxes, img_baseline, img_optimized, params)

    return composite, tagged_boxes


def _tag_boxes(boxes:          list,
               img_baseline:   np.ndarray,
               img_optimized:  np.ndarray,
               params:         dict) -> list:
    """
    Run localized edge and color checks within each bounding box region.

    For every box detected by SSIM, we crop both images to that region and
    re-run the edge match and Bhattacharyya color distance on the crop alone.
    This gives per-region attribution: which filters agree that this specific
    area is degraded, not just the image as a whole.

    SSIM is always in the flag set — boxes only exist because SSIM flagged them.

    INPUTS:
        boxes:         List of (x, y, w, h) tuples from SSIM region detection
        img_baseline:  Full baseline image (H × W × 3, float64 [0,1])
        img_optimized: Full optimized image (H × W × 3, float64 [0,1])
        params:        Full params dict (forwarded to edge and color passes)

    RETURNS:
        List of (x, y, w, h, flags) tuples where flags is a frozenset of
        strings from {'ssim', 'edge', 'color'}
    """
    # Never save intermediates for per-region crops — they are diagnostic
    # sub-calls, not primary pipeline outputs.
    crop_params = {**params, 'save_intermediates': False}

    tagged = []
    for (x, y, w, h) in boxes:
        flags = {'ssim'}  # SSIM always present — it's why this box exists

        # Crop both images to this region for localized analysis.
        crop_base = img_baseline [y:y+h, x:x+w]
        crop_opt  = img_optimized[y:y+h, x:x+w]

        # Crops smaller than 3×3 can't support Sobel kernels or meaningful
        # histograms, so we record the SSIM tag alone and move on.
        if crop_base.shape[0] < 3 or crop_base.shape[1] < 3:
            tagged.append((x, y, w, h, frozenset(flags)))
            continue

        # ---- Edge check ----
        # Does this specific region also fail the edge match threshold?
        try:
            edge_r = run_edge(crop_base, crop_opt, crop_params)
            if not edge_r['passed']:
                flags.add('edge')
        except Exception:
            # Pathological crops (e.g., completely uniform) may produce no
            # edges at all. Skip gracefully — SSIM tag is sufficient.
            pass

        # ---- Color check ----
        # Does this specific region also fail the Bhattacharyya threshold?
        try:
            color_r = run_color(crop_base, crop_opt, crop_params)
            if not color_r['passed']:
                flags.add('color')
        except Exception:
            pass

        tagged.append((x, y, w, h, frozenset(flags)))

    return tagged
