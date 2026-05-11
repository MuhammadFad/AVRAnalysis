# ================================================================================
# edge_pass.py — Phase 2b: Canny Edge  (unchanged algorithm, enriched schema)
# ================================================================================
#
# Added for web API:
#   - _encode_binary(): encodes a uint8 binary edge map as base64 PNG
#   - _build_edge_diff(): green = survived, red = lost
#   - 'intermediates' key when save_intermediates=True:
#       { 'baseline_edges', 'optimized_edges', 'edge_diff' }  — base64 PNGs
#
# ================================================================================

import io
import base64

import numpy as np
import cv2


# ================================================================================
# Base64 helpers
# ================================================================================

def _encode_binary(arr: np.ndarray) -> str:
    """Encode a uint8 binary (0/1 or 0/255) map as a base64 PNG."""
    u8      = (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr
    ok, buf = cv2.imencode('.png', u8)
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ""


def _encode_rgb(arr: np.ndarray) -> str:
    """Encode an H×W×3 uint8 RGB array as a base64 PNG."""
    bgr     = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ""


def _build_edge_diff(edge_baseline: np.ndarray, edge_optimized: np.ndarray) -> str:
    """
    Green pixels  = edges present in both (survived).
    Red pixels    = edges in baseline but gone in optimized (lost).
    Returns base64 PNG.
    """
    h, w  = edge_baseline.shape
    diff  = np.zeros((h, w, 3), dtype=np.uint8)

    survived = (edge_baseline > 0) & (edge_optimized > 0)
    lost     = (edge_baseline > 0) & (edge_optimized == 0)

    diff[survived] = [0,   200,  80]   # green
    diff[lost]     = [220, 40,   40]   # red

    return _encode_rgb(diff)


# ================================================================================
# Canny internals (unchanged)
# ================================================================================

def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    gray = (0.2989 * img[:, :, 0] +
            0.5870 * img[:, :, 1] +
            0.1140 * img[:, :, 2])
    return (gray * 255).astype(np.uint8)


def _gaussian_blur(img_gray, kernel_size, sigma):
    return cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), sigma)


def _sobel_gradients(blurred):
    sx  = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    dir = np.degrees(np.arctan2(np.abs(sy), np.abs(sx)))
    return mag, dir, sx, sy


def _non_maximum_suppression(magnitude, direction):
    H, W = magnitude.shape
    nms  = np.zeros((H, W), dtype=np.float64)
    angle = direction % 180

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            mag = magnitude[y, x]
            a   = angle[y, x]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[y + 1, x - 1], magnitude[y - 1, x + 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                n1, n2 = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if mag >= n1 and mag >= n2:
                nms[y, x] = mag

    return nms


def _double_threshold(nms, low, high):
    strong = (nms >= high).astype(np.uint8)
    weak   = ((nms >= low) & (nms < high)).astype(np.uint8)
    return strong, weak


def _hysteresis(strong, weak):
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges  = strong.copy()
    while True:
        dilated   = cv2.dilate(edges, kernel, iterations=1)
        promoted  = cv2.bitwise_and(dilated, weak)
        new_edges = cv2.bitwise_or(edges, promoted)
        if np.array_equal(new_edges, edges):
            break
        edges = new_edges
    return edges


# ================================================================================
# Public interface
# ================================================================================

def run(img_baseline: np.ndarray, img_optimized: np.ndarray, params: dict) -> dict:
    """
    Run Canny edge comparison.

    Returns score, passed, verdict, edge_baseline, edge_optimized,
    and optionally intermediates: { baseline_edges, optimized_edges, edge_diff }
    as base64 PNGs when save_intermediates=True.
    """
    blur_k = params['canny_blur_kernel']
    blur_s = params['canny_blur_sigma']
    low_t  = params['canny_low_thresh']
    high_t = params['canny_high_thresh']
    save   = params.get('save_intermediates', False)

    def _canny_pipeline(img, keep_intermediates=False):
        gray    = _to_gray_uint8(img)
        blurred = _gaussian_blur(gray, blur_k, blur_s)
        mag, direction, sx, sy = _sobel_gradients(blurred)
        nms     = _non_maximum_suppression(mag, direction)
        strong, weak = _double_threshold(nms, low_t, high_t)
        edges   = _hysteresis(strong, weak)
        out = {'edges': edges}
        if keep_intermediates:
            out['sobel_x'] = sx
            out['sobel_y'] = sy
            out['nms']     = nms
        return out

    br = _canny_pipeline(img_baseline,  keep_intermediates=save)
    or_ = _canny_pipeline(img_optimized, keep_intermediates=False)

    edge_baseline  = br['edges']
    edge_optimized = or_['edges']

    baseline_count = np.count_nonzero(edge_baseline)
    matched_count  = np.count_nonzero(edge_baseline & edge_optimized)
    score  = float(matched_count / baseline_count) if baseline_count > 0 else 1.0
    passed = score >= params['canny_edge_match_thresh']

    result = {
        "score":          score,
        "passed":         passed,
        "verdict":        "PASS" if passed else "FAIL",
        "edge_baseline":  edge_baseline,
        "edge_optimized": edge_optimized,
    }

    if save:
        result["sobel_x"] = br['sobel_x']
        result["sobel_y"] = br['sobel_y']
        result["nms_map"] = br['nms']
        result["intermediates"] = {
            "baseline_edges":  _encode_binary(edge_baseline),
            "optimized_edges": _encode_binary(edge_optimized),
            "edge_diff":       _build_edge_diff(edge_baseline, edge_optimized),
        }

    return result
