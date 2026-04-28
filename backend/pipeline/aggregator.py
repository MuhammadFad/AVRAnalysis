# =========================================================================
#  aggregator.py — Heatmap generation and degraded region detection.
#
#  Takes the SSIM map from ssim_pass.py and produces:
#    1. A false-colour heatmap blended over the optimized image
#    2. Bounding boxes around significantly degraded regions
#
#  INPUTS:
#    img_optimized — float64 [0,1] H x W x 3  — the optimized image
#    ssim_map      — float64 H x W             — local SSIM scores from Phase 2
#    params        — dict of config values
#
#  OUTPUTS:
#    composite     — float64 [0,1] H x W x 3  — heatmap blended over image
#    boxes         — list of [x, y, w, h] bounding boxes (pixel coords)
# =========================================================================

import numpy as np
import cv2
from scipy.ndimage import binary_closing
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def build_heatmap(img_optimized: np.ndarray, ssim_map: np.ndarray, params: dict):
    """
    Build the composite heatmap image and detect degraded region bounding boxes.

    Returns:
        composite_bgr — uint8 BGR image ready for cv2.imwrite()
        boxes         — list of (x, y, w, h) tuples in pixel coordinates
    """

    # ------------------------------------------------------------------
    #  STEP 1: Build degradation map
    #
    #  SSIM map has high values (≈1) where images are similar.
    #  Invert so high values = high degradation (hot colours on heatmap).
    #
    #  degradation = 1 - SSIM_map
    #  Then clip to [0, 1] to handle any numerical edge cases.
    # ------------------------------------------------------------------
    degradation_map = np.clip(1.0 - ssim_map, 0.0, 1.0)

    # ------------------------------------------------------------------
    #  STEP 2: Apply 'hot' colormap to degradation map
    #
    #  matplotlib's 'hot' colormap:
    #    0.0 (no degradation) → black
    #    0.5 (moderate)       → red/orange
    #    1.0 (max degradation)→ white
    #
    #  cm.hot returns RGBA [0,1] — we drop the alpha channel ([:,3]).
    #  Result: heatmap_rgb is H x W x 3 float64 [0,1]
    # ------------------------------------------------------------------
    colormap    = cm.get_cmap('hot')
    heatmap_rgb = colormap(degradation_map)[:, :, :3]   # drop alpha

    # ------------------------------------------------------------------
    #  STEP 3: Alpha-blend heatmap over the optimized image
    #
    #  composite = α × heatmap + (1-α) × optimized_image
    #
    #  α = heatmap_alpha (0.55 by default) — strong enough to see the
    #  signal clearly while the scene content remains recognisable.
    # ------------------------------------------------------------------
    alpha     = params['heatmap_alpha']
    composite = alpha * heatmap_rgb + (1.0 - alpha) * img_optimized
    composite = np.clip(composite, 0.0, 1.0)

    # ------------------------------------------------------------------
    #  STEP 4: Detect degraded regions
    #
    #  4a. Threshold: pixels above degradation_thresh become 1
    #  4b. Morphological closing: merges nearby degraded clusters
    #      structure = disk of given radius
    #      binary_closing = dilation then erosion
    #  4c. findContours: finds connected blobs in the binary mask
    #  4d. boundingRect: gets [x, y, w, h] for each blob
    #  4e. Filter: discard blobs smaller than min_region_area
    # ------------------------------------------------------------------
    binary_mask = (degradation_map > params['degradation_thresh']).astype(np.uint8)

    # Build circular structuring element for morphological closing
    r    = params['morph_radius']
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, disk)

    # Find contours of connected blobs
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= params['min_region_area']:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    # ------------------------------------------------------------------
    #  STEP 5: Draw bounding boxes on the composite image
    #
    #  Convert to uint8 first (cv2.rectangle expects uint8).
    #  Draw red rectangles (BGR: 0, 0, 255) with 3px line width.
    #  Convert back to float64 for consistent downstream handling.
    # ------------------------------------------------------------------
    composite_uint8 = (composite * 255).astype(np.uint8)

    for (x, y, w, h) in boxes:
        cv2.rectangle(composite_uint8, (x, y), (x+w, y+h),
                      color=(255, 0, 0),   # RGB red
                      thickness=3)

    # Convert RGB → BGR for OpenCV saving
    composite_bgr = cv2.cvtColor(composite_uint8, cv2.COLOR_RGB2BGR)

    return composite_bgr, boxes
