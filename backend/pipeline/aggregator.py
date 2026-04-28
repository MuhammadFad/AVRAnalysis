# =======
# aggregator.py — Heatmap generation and degraded region detection.
# =======

import numpy as np
import cv2
import matplotlib

def build_heatmap(img_optimized: np.ndarray, ssim_map: np.ndarray, params: dict):
    # -----
    # Degradation map: invert SSIM so high value = high degradation
    # -----
    degradation_map = np.clip(1.0 - ssim_map, 0.0, 1.0)

    # ----
    # Apply 'hot' colormap
    # ----
    heatmap_rgb = matplotlib.colormaps['hot'](degradation_map)[:, :, :3]

    # --
    # Alpha-blend heatmap over optimized image
    # --
    alpha = params['heatmap_alpha']
    composite = alpha * heatmap_rgb + (1.0 - alpha) * img_optimized
    composite = np.clip(composite, 0.0, 1.0)

    # -----
    # Detect degraded regions
    # -----
    binary_mask = (degradation_map > params['degradation_thresh']).astype(np.uint8)

    r = params['morph_radius']
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, disk)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > params['min_region_area']:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return composite, boxes