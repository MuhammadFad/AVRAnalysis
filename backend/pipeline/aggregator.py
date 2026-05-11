# ================================================================================
# aggregator.py — Phase 3: Degraded Region Detection & Spatial Attribution
# ================================================================================
#
# Strictly spatial — no cause/hypothesis logic (that lives in combinator.py).
# Detects regions from the SSIM map, computes per-region local scores for all
# three passes using map slices (SSIM, edge) or crop re-runs (color).
#
# ================================================================================

import numpy as np
import cv2
import matplotlib

from backend.pipeline.color_pass import run as run_color


def build_heatmap(img_baseline:   np.ndarray,
                  img_optimized:  np.ndarray,
                  ssim_map:       np.ndarray,
                  edge_baseline:  np.ndarray,
                  edge_optimized: np.ndarray,
                  params:         dict):
    """
    Build visual heatmap composite and detect degraded regions.

    Returns (composite, combined_regions).
    composite:        H × W × 3 float64 heatmap blended with optimized image
    combined_regions: list of region dicts with bbox, passes_failed, local_scores
    """
    degradation_map = np.clip(1.0 - ssim_map, 0.0, 1.0)
    heatmap_rgb     = matplotlib.colormaps['hot'](degradation_map)[:, :, :3]

    alpha     = params['heatmap_alpha']
    composite = alpha * heatmap_rgb + (1.0 - alpha) * img_optimized
    composite = np.clip(composite, 0.0, 1.0)

    binary_mask = (degradation_map > params['degradation_thresh']).astype(np.uint8)

    r    = params['morph_radius']
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, disk)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours
             if cv2.contourArea(cnt) > params['min_region_area']]

    combined_regions = _score_regions(
        boxes, ssim_map, edge_baseline, edge_optimized,
        img_baseline, img_optimized, params,
    )

    return composite, combined_regions


def _score_regions(boxes, ssim_map, edge_baseline, edge_optimized,
                   img_baseline, img_optimized, params):
    crop_params  = {**params, 'save_intermediates': False}
    edge_thresh  = params['canny_edge_match_thresh']
    color_thresh = params['color_distance_thresh']

    combined_regions = []

    for (x, y, w, h) in boxes:
        passes_failed = ['ssim']

        ssim_crop  = ssim_map[y:y+h, x:x+w]
        local_ssim = float(np.mean(ssim_crop))

        local_edge = None
        if edge_baseline is not None and edge_optimized is not None:
            eb_crop = edge_baseline [y:y+h, x:x+w]
            eo_crop = edge_optimized[y:y+h, x:x+w]
            bc      = np.count_nonzero(eb_crop)
            if bc > 0:
                local_edge = float(np.count_nonzero(eb_crop & eo_crop) / bc)
                if local_edge < edge_thresh:
                    passes_failed.append('edge')

        local_color = None
        if w >= 3 and h >= 3:
            try:
                color_r     = run_color(img_baseline[y:y+h, x:x+w],
                                        img_optimized[y:y+h, x:x+w],
                                        crop_params)
                local_color = color_r['score']
                if not color_r['passed']:
                    passes_failed.append('color')
            except Exception:
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
