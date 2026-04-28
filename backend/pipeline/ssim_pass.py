# =========================================================================
#  ssim_pass.py — Phase 2: Structural Similarity Index Measure (SSIM)
#
#  Computes a global SSIM score and a local SSIM difference map between
#  a baseline and optimized image.
#
#  We implement SSIM manually using scipy convolutions — identical logic
#  to our MATLAB implementation — rather than using skimage.ssim(), so
#  we have full control over the intermediate maps needed for the heatmap.
#
#  INPUTS:
#    img1 — Baseline image,  float64 [0,1], H x W x 3
#    img2 — Optimized image, float64 [0,1], H x W x 3
#
#  OUTPUTS:
#    score    — Scalar global SSIM in [-1, 1]. Closer to 1 = more similar.
#    ssim_map — 2D array H x W. Each pixel = local SSIM of its 11x11 window.
# =========================================================================

import numpy as np
from scipy.ndimage import convolve
from config import SSIM_WINDOW_SIZE, SSIM_SIGMA


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Build a 2D Gaussian kernel of given size and sigma.
    Equivalent to MATLAB's fspecial('gaussian', size, sigma).

    The kernel is constructed as the outer product of two 1D Gaussians,
    then normalised so all weights sum to 1.
    """
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)

    # 1D Gaussian: g(x) = exp(-x² / (2σ²))
    g1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))

    # 2D Gaussian = outer product of two 1D Gaussians
    g2d = np.outer(g1d, g1d)

    # Normalise so weights sum to 1
    return g2d / g2d.sum()


def _local_mean(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Compute the local (weighted) mean of the image using convolution.
    Equivalent to MATLAB's imfilter(image, kernel, 'replicate').

    scipy.ndimage.convolve with mode='reflect' mirrors pixels at borders —
    closest available equivalent to MATLAB's 'replicate' padding.
    """
    return convolve(image, kernel, mode='reflect')


def run(img1: np.ndarray, img2: np.ndarray):
    """
    Run SSIM analysis between baseline (img1) and optimized (img2).

    Returns:
        score    — float, global SSIM score
        ssim_map — 2D numpy array, local SSIM values per pixel
    """

    # ------------------------------------------------------------------
    #  STEP 1: Convert RGB → Grayscale (luminance)
    #
    #  Weights match ITU-R BT.601 — same as MATLAB's rgb2gray().
    #  We operate on luminance because SSIM models the human visual
    #  system, which is far more sensitive to luminance than colour.
    # ------------------------------------------------------------------
    def rgb_to_gray(img):
        return (0.2989 * img[:,:,0] +
                0.5870 * img[:,:,1] +
                0.1140 * img[:,:,2])

    gray1 = rgb_to_gray(img1)
    gray2 = rgb_to_gray(img2)

    # ------------------------------------------------------------------
    #  STEP 2: SSIM stabilising constants
    #
    #  Prevent division by zero when local mean or variance is near zero.
    #  Standard values from Wang et al. (2004):
    #    C1 = (K1 * L)² where K1=0.01, L=1.0 (dynamic range for [0,1])
    #    C2 = (K2 * L)² where K2=0.03
    # ------------------------------------------------------------------
    C1 = (0.01 * 1.0) ** 2    # = 0.0001
    C2 = (0.03 * 1.0) ** 2    # = 0.0009

    # ------------------------------------------------------------------
    #  STEP 3: Build 11x11 Gaussian kernel
    #
    #  Same as MATLAB: fspecial('gaussian', 11, 1.5)
    #  The kernel defines the local neighbourhood for all SSIM statistics.
    # ------------------------------------------------------------------
    kernel = _gaussian_kernel(SSIM_WINDOW_SIZE, SSIM_SIGMA)

    # ------------------------------------------------------------------
    #  STEP 4: Compute local statistics via convolution
    #
    #  All operations work on full H x W arrays simultaneously.
    #  No loops over individual pixels.
    #
    #  local mean:      mu  = kernel ⊛ image
    #  local variance:  σ²  = E[X²] - (E[X])²
    #  local covariance:σ12 = E[X·Y] - E[X]·E[Y]
    # ------------------------------------------------------------------

    # Local means
    mu1 = _local_mean(gray1, kernel)
    mu2 = _local_mean(gray2, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    # Local variances: E[X²] - (E[X])²
    sigma1_sq = _local_mean(gray1 * gray1, kernel) - mu1_sq
    sigma2_sq = _local_mean(gray2 * gray2, kernel) - mu2_sq

    # Local covariance: E[X·Y] - E[X]·E[Y]
    sigma12   = _local_mean(gray1 * gray2, kernel) - mu12

    # ------------------------------------------------------------------
    #  STEP 5: Compute the SSIM map
    #
    #  SSIM = L · CS  (luminance × contrast-structure combined term)
    #
    #  numerator   = (2·μ₁μ₂ + C1) · (2·σ₁₂ + C2)
    #  denominator = (μ₁² + μ₂² + C1) · (σ₁² + σ₂² + C2)
    #
    #  All arithmetic is element-wise across H x W arrays.
    # ------------------------------------------------------------------
    numerator   = (2.0 * mu12    + C1) * (2.0 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator    # H x W, values in [-1, 1]

    # ------------------------------------------------------------------
    #  STEP 6: Global score = mean of all local SSIM values
    # ------------------------------------------------------------------
    score = float(np.mean(ssim_map))

    return score, ssim_map
