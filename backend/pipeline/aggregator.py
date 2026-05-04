# ================================================================================
# aggregator.py — Phase 3: Degraded Region Detection & Heatmap Generation
# ================================================================================
#
# PURPOSE:
# This module processes SSIM results to:
#   1. Visualize regions of visual degradation as a heatmap
#   2. Detect and localize specific degraded areas using morphological operations
#   3. Extract bounding boxes around significant degradation hotspots
#
# ROLE IN PIPELINE:
# This is Phase 3 of the analysis. It receives the per-pixel SSIM map from the
# SSIM module and transforms it into:
#   - A visual heatmap (hot colormap: blue=good, red=bad)
#   - Alpha-blended composite with original image for context
#   - Bounding boxes around significant degraded regions
#
# PROCESS OVERVIEW:
# 1. Invert SSIM map (high SSIM → low degradation, vice versa)
# 2. Apply 'hot' colormap for visualization
# 3. Alpha-blend with original image for visual balance
# 4. Create binary mask of "degraded" pixels (threshold-based)
# 5. Apply morphological closing to reduce noise
# 6. Find contours and extract bounding boxes
# 7. Filter boxes by minimum area to remove trivial artifacts
#
# ================================================================================

import numpy as np
import cv2
import matplotlib

def build_heatmap(img_optimized: np.ndarray, ssim_map: np.ndarray, params: dict):
    """
    Build visual heatmap and detect degraded regions from SSIM map.
    
    This function transforms the per-pixel SSIM scores into a human-interpretable
    visualization and automatically detects areas of visual degradation.
    
    INPUTS:
        img_optimized: The optimized image (H × W × 3, float64 [0,1])
        ssim_map: Per-pixel SSIM scores (H × W, float64 [0,1])
        params: Dictionary with keys:
            - 'heatmap_alpha': Blend factor between heatmap and image
            - 'degradation_thresh': Threshold for degradation detection
            - 'morph_radius': Morphological closing radius (in pixels)
            - 'min_region_area': Minimum bounding box area (in pixels²)
    
    RETURNS:
        (composite, boxes) tuple:
        - composite: H × W × 3 float64 image (heatmap + original blended)
        - boxes: List of (x, y, w, h) bounding boxes around degraded regions
    """
    # ========================================================================
    # STEP 1: Create degradation map
    # ========================================================================
    # Invert SSIM: 
    #   - SSIM = 1.0 (identical) → degradation = 0.0 (none)
    #   - SSIM = 0.5 (different) → degradation = 0.5 (moderate)
    #   - SSIM = 0.0 (very different) → degradation = 1.0 (severe)
    # Clipping to [0,1] handles numerical edge cases
    degradation_map = np.clip(1.0 - ssim_map, 0.0, 1.0)

    # ========================================================================
    # STEP 2: Apply 'hot' colormap for visualization
    # ========================================================================
    # 'hot' colormap: Black → Blue → Red → Yellow progression
    # Maps degradation value to RGB color:
    #   - 0.0 (no degradation) → Black/dark (good)
    #   - 0.5 (moderate) → Red (warning)
    #   - 1.0 (severe) → Yellow/bright (critical)
    # matplotlib.colormaps['hot'] returns RGBA, we take first 3 channels (RGB)
    heatmap_rgb = matplotlib.colormaps['hot'](degradation_map)[:, :, :3]

    # ========================================================================
    # STEP 3: Alpha-blend heatmap with original image
    # ========================================================================
    # Composite = α * heatmap_rgb + (1-α) * img_optimized
    # This shows degradation locations while preserving image context
    # α = 0.55 (default): 55% heatmap, 45% original image
    #   - Higher α: More visible heatmap, less original image context
    #   - Lower α: Less visible heatmap, more original image visible
    alpha = params['heatmap_alpha']
    composite = alpha * heatmap_rgb + (1.0 - alpha) * img_optimized
    composite = np.clip(composite, 0.0, 1.0)  # Ensure values stay in [0,1]

    # ========================================================================
    # STEP 4: Create binary mask of degraded regions
    # ========================================================================
    # Threshold degradation map: pixels with high degradation → 1 (white)
    # Default threshold: 0.30 (anything with <70% SSIM is flagged)
    # This binarization converts continuous values to foreground/background
    binary_mask = (degradation_map > params['degradation_thresh']).astype(np.uint8)

    # ========================================================================
    # STEP 5: Morphological closing to reduce noise
    # ========================================================================
    # Morphological closing = dilation followed by erosion
    # Effect: Fills small holes in degraded regions, removes tiny noise spots
    # Structuring element: Ellipse (circular, less boxy than rectangle)
    r = params['morph_radius']
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, disk)
    # Result: Connected regions without small holes or isolated pixels

    # ========================================================================
    # STEP 6: Find contours in the closed mask
    # ========================================================================
    # Contours are the boundaries of connected components (degraded regions)
    # RETR_EXTERNAL: Only get outermost contours (ignore holes)
    # CHAIN_APPROX_SIMPLE: Compress contours (e.g., store only corner points)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ========================================================================
    # STEP 7: Extract bounding boxes and filter by minimum area
    # ========================================================================
    # For each contour, find its bounding rectangle
    # Filter: only report boxes with area ≥ min_region_area
    # This removes noise/artifacts and focuses on significant degradations
    boxes = []
    for cnt in contours:
        # Check if this contour is large enough to report
        if cv2.contourArea(cnt) > params['min_region_area']:
            # boundingRect returns (x, y, width, height)
            # x, y: top-left corner coordinates
            # width, height: dimensions
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    # ========================================================================
    # Return composite heatmap and degraded region bounding boxes
    # ========================================================================
    return composite, boxes