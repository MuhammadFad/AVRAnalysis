# ===========
# ssim_pass.py — Structural Similarity Index Measure (SSIM)
# ===========

import numpy as np
from scipy.ndimage import convolve
from config import SSIM_WINDOW_SIZE, SSIM_SIGMA, SAVE_INTERMEDIATES

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g2d = np.outer(g1d, g1d)
    return g2d / g2d.sum()

def _local_mean(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return convolve(image, kernel, mode='reflect')

def run(img1: np.ndarray, img2: np.ndarray, save_intermediates: bool = SAVE_INTERMEDIATES) -> dict:
    # --------------
    # Grayscale conversion — ITU-R BT.601 weights
    # --------------
    def to_gray(img):
        return (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2])

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    # --------------
    # Stabilizing constants — Wang et al. (2004)
    # --------------
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    C3 = C2 / 2.0

    kernel = _gaussian_kernel(SSIM_WINDOW_SIZE, SSIM_SIGMA)

    # --------------
    # Local statistics via convolution
    # --------------
    mu1 = _local_mean(gray1, kernel)
    mu2 = _local_mean(gray2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = _local_mean(gray1 * gray1, kernel) - mu1_sq
    sigma2_sq = _local_mean(gray2 * gray2, kernel) - mu2_sq
    sigma12 = _local_mean(gray1 * gray2, kernel) - mu12

    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0.0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0.0))

    # --------------
    # SSIM components
    # --------------
    l_map = (2.0 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
    c_map = (2.0 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    s_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    ssim_map = l_map * c_map * s_map
    score = float(np.mean(ssim_map))

    result = {
        "score": score,
        "ssim_map": ssim_map,
    }

    if save_intermediates:
        result["l_map"] = l_map
        result["c_map"] = c_map
        result["s_map"] = s_map

    return result