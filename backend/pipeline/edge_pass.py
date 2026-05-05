# ================================================================================
# edge_pass.py — Phase 2b: Canny Edge Detection & Structural Edge Comparison
# ================================================================================
#
# PURPOSE:
# This module compares the structural edge content of two images using the
# Canny edge detection algorithm. Where SSIM measures broad perceptual similarity,
# this pass specifically isolates whether the optimized image has preserved the
# structural details (outlines, boundaries, fine geometry) present in the baseline.
#
# A rendering optimization might preserve overall color and brightness (high SSIM)
# while quietly losing fine edge detail — this pass catches exactly that.
#
# ROLE IN PIPELINE:
# This is Phase 2b of the analysis. It receives normalized float64 images from
# acquisition and produces:
#   - An edge match score (0.0 to 1.0) — the fraction of baseline edges that
#     are still present in the optimized image
#   - A pass/fail flag against the configured threshold
#   - Optional intermediate maps for diagnostic inspection:
#       · sobel_x, sobel_y: Raw gradient images in X and Y directions
#       · nms_map:          After non-maximum suppression (thin edges)
#       · edge_baseline:    Final Canny edge map for the baseline image
#       · edge_optimized:   Final Canny edge map for the optimized image
#
# CANNY ALGORITHM OVERVIEW:
# 1. Gaussian blur     — suppress noise before gradient computation
# 2. Sobel gradients   — compute intensity gradient magnitude and direction
# 3. Non-max suppress  — thin edges to 1-pixel-wide ridges
# 4. Double threshold  — classify pixels as strong, weak, or non-edge
# 5. Hysteresis        — connect weak edges to strong ones; discard the rest
#
# ================================================================================

import numpy as np
import cv2


# ================================================================================
# Internal helpers — Canny pipeline steps
# ================================================================================

def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert a float64 RGB image to a uint8 grayscale image.

    Canny is applied on grayscale because we care about structural intensity
    changes, not color transitions. The ITU-R BT.601 weights account for the
    human eye's different sensitivity to each channel (green > red > blue).

    INPUTS:
        img: H × W × 3 float64 array in [0.0, 1.0]

    RETURNS:
        H × W uint8 array in [0, 255]
    """
    gray_float = (0.2989 * img[:, :, 0] +
                  0.5870 * img[:, :, 1] +
                  0.1140 * img[:, :, 2])
    return (gray_float * 255).astype(np.uint8)


def _gaussian_blur(img_gray: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to suppress noise before gradient computation.

    Without this step, the Sobel filters would respond to high-frequency noise
    as strongly as they respond to real edges, flooding the output with false
    detections. The kernel size and sigma together control the trade-off between
    noise suppression and preservation of fine edge detail.

    INPUTS:
        img_gray:    H × W uint8 grayscale image
        kernel_size: Width/height of the Gaussian kernel (must be odd)
        sigma:       Standard deviation of the Gaussian

    RETURNS:
        H × W uint8 blurred image
    """
    return cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), sigma)


def _sobel_gradients(blurred: np.ndarray):
    """
    Compute gradient magnitude and direction using Sobel filters.

    The Sobel operator convolves the image with two 3×3 kernels that approximate
    the partial derivative of intensity in the X and Y directions respectively.
    The result at each pixel is a 2D gradient vector (Gx, Gy) representing both
    the strength and direction of the local intensity change.

    Gradient magnitude = sqrt(Gx² + Gy²)
        → How strong is the intensity change at this pixel?
        → Large magnitude = sharp boundary = likely edge.

    Gradient direction = arctan2(Gy, Gx)
        → Which way does the intensity change fastest?
        → Edges run perpendicular to this direction.
        → Used in NMS to know which neighbors to compare against.

    INPUTS:
        blurred: H × W uint8 blurred grayscale image

    RETURNS:
        magnitude:  H × W float64 — gradient strength at each pixel
        direction:  H × W float64 — gradient angle in degrees [0, 180)
        sobel_x:    H × W float64 — raw horizontal gradient (for visualization)
        sobel_y:    H × W float64 — raw vertical gradient (for visualization)
    """
    # Compute partial derivatives in X and Y using Sobel kernels.
    # ksize=3: standard 3×3 Sobel kernel.
    # cv2.CV_64F: output as float64 to preserve sign (negative gradients matter).
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude: length of the gradient vector at each pixel.
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Direction: angle of the gradient vector.
    # np.arctan2 handles all four quadrants correctly (unlike plain arctan).
    # We fold into [0°, 180°) because edge direction is symmetric — an edge
    # pointing "up" and one pointing "down" are the same edge.
    direction = np.degrees(np.arctan2(np.abs(sobel_y), np.abs(sobel_x)))

    return magnitude, direction, sobel_x, sobel_y


def _non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Thin edges from blurry gradient blobs to 1-pixel-wide ridges.

    After the Sobel step, each genuine edge in the image corresponds to a
    broad ridge of high-magnitude pixels. NMS keeps only the single pixel at
    the crest of each ridge, zeroing out everything else.

    For each pixel, we look at its two neighbors along the gradient direction
    (i.e., across the edge). If the pixel is not the local maximum among those
    three, it is suppressed to zero. This produces crisp, single-pixel edges.

    Because we can only step to discrete pixel neighbors, the continuous
    gradient direction is quantized into 4 sectors:
        0°  → compare left  / right   neighbors
        45° → compare ↗ / ↙ diagonal neighbors
        90° → compare up    / down    neighbors
       135° → compare ↖ / ↘ diagonal neighbors

    INPUTS:
        magnitude: H × W float64 — gradient magnitudes
        direction: H × W float64 — gradient directions in degrees [0, 180)

    RETURNS:
        H × W float64 — suppressed magnitude map (most pixels zeroed out)
    """
    H, W = magnitude.shape
    nms  = np.zeros((H, W), dtype=np.float64)

    # Quantize gradient direction to the nearest of 4 discrete neighbor axes.
    # Rounding to [0, 45, 90, 135] covers all possible edge orientations.
    angle = direction % 180

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            mag = magnitude[y, x]

            a = angle[y, x]

            # Select the two neighbors to compare against based on gradient direction.
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                # Horizontal edge → compare left and right
                n1, n2 = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= a < 67.5:
                # Diagonal ↗ edge → compare ↙ and ↗ neighbors
                n1, n2 = magnitude[y + 1, x - 1], magnitude[y - 1, x + 1]
            elif 67.5 <= a < 112.5:
                # Vertical edge → compare up and down
                n1, n2 = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                # Diagonal ↖ edge → compare ↖ and ↘ neighbors
                n1, n2 = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]

            # Keep this pixel only if it is the local maximum.
            if mag >= n1 and mag >= n2:
                nms[y, x] = mag

    return nms


def _double_threshold(nms: np.ndarray, low: float, high: float):
    """
    Classify pixels into strong edges, weak edges, and non-edges.

    After NMS we still have a continuous gradient magnitude image. Double
    thresholding converts it to three discrete classes:
        - Strong (magnitude ≥ high):  definitely a real edge
        - Weak   (low ≤ mag < high):  possibly a real edge, needs confirmation
        - None   (magnitude < low):   definitely noise, discard

    Weak pixels may still be real edges if they are connected to a strong
    one — that decision is deferred to the hysteresis step.

    INPUTS:
        nms:  H × W float64 — NMS-suppressed gradient map
        low:  Lower threshold (magnitude below this → discarded)
        high: Upper threshold (magnitude above this → strong edge)

    RETURNS:
        strong: H × W uint8 binary mask — strong edge pixels
        weak:   H × W uint8 binary mask — weak edge pixels
    """
    strong = (nms >= high).astype(np.uint8)
    weak   = ((nms >= low) & (nms < high)).astype(np.uint8)
    return strong, weak


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """
    Resolve weak edge pixels by connectivity to strong ones.

    A weak pixel is promoted to a real edge if it is 8-connected (including
    diagonals) to at least one strong pixel, either directly or through a
    chain of other weak pixels. This bridges genuine edge gaps caused by slight
    variations in gradient magnitude while discarding isolated weak responses
    that are just noise.

    Implementation uses iterative dilation: flood-fill the strong mask into
    the weak mask until no further growth is possible. This is equivalent to
    connected-component labeling but simpler to implement and fast in practice.

    INPUTS:
        strong: H × W uint8 — strong edge mask from double thresholding
        weak:   H × W uint8 — weak edge mask from double thresholding

    RETURNS:
        H × W uint8 binary edge map — the final Canny output
    """
    # 3×3 all-ones kernel for 8-connectivity dilation.
    kernel = np.ones((3, 3), dtype=np.uint8)

    edges = strong.copy()
    while True:
        # Expand the current edge mask by one pixel in all 8 directions.
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Keep only pixels that were weak — strong pixels are already in edges.
        # This restricts growth to the weak candidate set.
        promoted = cv2.bitwise_and(dilated, weak)

        # Merge the newly promoted pixels into the edge mask.
        new_edges = cv2.bitwise_or(edges, promoted)

        # Stop when no more weak pixels can be promoted.
        if np.array_equal(new_edges, edges):
            break
        edges = new_edges

    return edges


# ================================================================================
# Public interface
# ================================================================================

def run(img_baseline: np.ndarray, img_optimized: np.ndarray, params: dict) -> dict:
    """
    Run the Canny edge detection pass on both images and compare their outputs.

    Applies the full Canny pipeline independently to the baseline and optimized
    images, then computes an edge match score measuring what fraction of baseline
    edges survive in the optimized output.

    Edge match score = |edge_baseline ∩ edge_optimized| / |edge_baseline|

    A score of 1.0 means every edge in the baseline is also present in the
    optimized image. A score of 0.7 means 30% of baseline edges have been lost.

    INPUTS:
        img_baseline:  H × W × 3 float64 [0,1] — reference image
        img_optimized: H × W × 3 float64 [0,1] — test image
        params: Dictionary with keys:
            - 'canny_blur_kernel':  Gaussian blur kernel size (odd int)
            - 'canny_blur_sigma':   Gaussian blur standard deviation
            - 'canny_low_thresh':   Lower hysteresis threshold
            - 'canny_high_thresh':  Upper hysteresis threshold
            - 'canny_edge_match_thresh': Minimum acceptable edge match score
            - 'save_intermediates': Whether to include diagnostic maps in output

    RETURNS:
        Dictionary with keys:
        - 'score':          Edge match score (float, 0.0 to 1.0)
        - 'passed':         True if score ≥ canny_edge_match_thresh
        - 'edge_baseline':  Final Canny map for baseline (H × W uint8)
        - 'edge_optimized': Final Canny map for optimized image (H × W uint8)
        - 'sobel_x':        Horizontal gradient of baseline (if save_intermediates)
        - 'sobel_y':        Vertical gradient of baseline (if save_intermediates)
        - 'nms_map':        NMS output for baseline (if save_intermediates)
    """
    blur_k  = params['canny_blur_kernel']
    blur_s  = params['canny_blur_sigma']
    low_t   = params['canny_low_thresh']
    high_t  = params['canny_high_thresh']

    # ========================================================================
    # Run Canny pipeline on both images
    # ========================================================================
    def _canny_pipeline(img, save_intermediates=False):
        """Full Canny pipeline for a single image."""
        gray    = _to_gray_uint8(img)
        blurred = _gaussian_blur(gray, blur_k, blur_s)
        mag, direction, sx, sy = _sobel_gradients(blurred)
        nms     = _non_maximum_suppression(mag, direction)
        strong, weak = _double_threshold(nms, low_t, high_t)
        edges   = _hysteresis(strong, weak)

        result = {'edges': edges}
        if save_intermediates:
            result['sobel_x'] = sx
            result['sobel_y'] = sy
            result['nms']     = nms
        return result

    save = params.get('save_intermediates', False)

    baseline_result  = _canny_pipeline(img_baseline,  save_intermediates=save)
    optimized_result = _canny_pipeline(img_optimized, save_intermediates=False)

    edge_baseline  = baseline_result['edges']
    edge_optimized = optimized_result['edges']

    # ========================================================================
    # Compute edge match score
    # ========================================================================
    # Count how many baseline edge pixels are also edges in the optimized image.
    # We use the baseline as the reference — the question is "what was lost?"
    # not "what was added?" (added edges could just be compression artifacts).
    baseline_count  = np.count_nonzero(edge_baseline)
    matched_count   = np.count_nonzero(edge_baseline & edge_optimized)

    # Guard against a pathological all-black baseline (no edges at all).
    score = float(matched_count / baseline_count) if baseline_count > 0 else 1.0

    passed = score >= params['canny_edge_match_thresh']

    # ========================================================================
    # Assemble result dictionary
    # ========================================================================
    result = {
        "score":          score,
        "passed":         passed,
        "edge_baseline":  edge_baseline,
        "edge_optimized": edge_optimized,
    }

    if save:
        result["sobel_x"] = baseline_result['sobel_x']
        result["sobel_y"] = baseline_result['sobel_y']
        result["nms_map"] = baseline_result['nms']

    return result
