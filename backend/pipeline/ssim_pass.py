# ================================================================================
# ssim_pass.py — Phase 2a: SSIM  (unchanged algorithm, enriched return schema)
# ================================================================================
#
# Added for web API:
#   - _encode_map(): converts a numpy map to a base64 PNG string
#   - 'intermediates' key in result when save_intermediates=True:
#       { 'luminance', 'contrast', 'structure', 'ssim_map' }  — all base64 PNGs
#
# ================================================================================

import io
import base64

import numpy as np
import cv2
from scipy.ndimage import convolve


# ================================================================================
# Base64 helper
# ================================================================================

def _encode_map(arr: np.ndarray, colormap=cv2.COLORMAP_HOT) -> str:
    """
    Encode a float64 or uint8 2D array as a base64 PNG string.
    Applies CLAHE normalisation then the given colormap for visibility.
    """
    lo, hi = np.min(arr), np.max(arr)
    if hi > lo:
        u8 = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        u8 = np.zeros_like(arr, dtype=np.uint8)

    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    u8      = clahe.apply(u8)
    colored = cv2.applyColorMap(u8, colormap)

    ok, buf = cv2.imencode('.png', colored)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode('utf-8')


# ================================================================================
# Gaussian kernel + local mean
# ================================================================================

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d    = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g2d    = np.outer(g1d, g1d)
    return g2d / g2d.sum()


def _local_mean(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return convolve(image, kernel, mode='reflect')


# ================================================================================
# Public interface
# ================================================================================

def run(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Compute SSIM between two RGB images.

    Returns:
        score, passed, verdict, ssim_map
        l_map, c_map, s_map          — if save_intermediates
        intermediates: {luminance, contrast, structure, ssim_map}  — base64 PNGs
    """
    def to_gray(img):
        return (0.2989 * img[:, :, 0] +
                0.5870 * img[:, :, 1] +
                0.1140 * img[:, :, 2])

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    C3 = C2 / 2.0

    kernel = _gaussian_kernel(params['ssim_window_size'], params['ssim_sigma'])

    mu1 = _local_mean(gray1, kernel)
    mu2 = _local_mean(gray2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = _local_mean(gray1 * gray1, kernel) - mu1_sq
    sigma2_sq = _local_mean(gray2 * gray2, kernel) - mu2_sq
    sigma12   = _local_mean(gray1 * gray2, kernel) - mu12

    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0.0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0.0))

    l_map = (2.0 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
    c_map = (2.0 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    s_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    ssim_map = l_map * c_map * s_map
    score    = float(np.mean(ssim_map))
    passed   = score >= params['ssim_threshold']

    result = {
        "score":    score,
        "passed":   passed,
        "verdict":  "PASS" if passed else "FAIL",
        "ssim_map": ssim_map,
    }

    if params.get('save_intermediates', False):
        result["l_map"] = l_map
        result["c_map"] = c_map
        result["s_map"] = s_map
        result["intermediates"] = {
            "luminance":  _encode_map(l_map),
            "contrast":   _encode_map(c_map),
            "structure":  _encode_map(s_map),
            "ssim_map":   _encode_map(ssim_map),
        }

    return result
