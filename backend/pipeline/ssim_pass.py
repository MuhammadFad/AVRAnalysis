# ================================================================================
# ssim_pass.py — Phase 2a: Structural Similarity Index Measure (SSIM)
# ================================================================================
#
# PURPOSE:
# This module implements the Structural Similarity Index Measure (SSIM), which
# quantifies the perceptual similarity between two images. Unlike simple pixel
# difference metrics (MSE, MAE), SSIM captures how humans perceive image quality
# by considering:
#   1. Luminance (brightness) similarity
#   2. Contrast (texture variation) similarity
#   3. Structure (edge/pattern) similarity
#
# ROLE IN PIPELINE:
# This is Phase 2a of the analysis. It receives normalized float64 images from
# the acquisition module and produces:
#   - An overall SSIM score (0.0 to 1.0, where 1.0 = identical images)
#   - A per-pixel SSIM map showing local similarity across the image
#   - Optional component maps (L, C, S) for detailed diagnosis
#
# MATHEMATICAL FOUNDATION:
# SSIM formula (Wang et al., 2004):
#   SSIM(x, y) = (2μ_x*μ_y + C1) / (μ_x² + μ_y² + C1)    [Luminance]
#                * (2σ_x*σ_y + C2) / (σ_x² + σ_y² + C2)   [Contrast]
#                * (σ_xy + C3) / (σ_x*σ_y + C3)            [Structure]
#
# Key insight: Each component captures different perceptual aspects. A low value
# in any component indicates that component caused the degradation.
#
# ================================================================================

import numpy as np
from scipy.ndimage import convolve


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel for local windowing in SSIM computation.

    The Gaussian kernel weights the local neighborhood for SSIM calculation,
    giving more importance to pixels near the center of the window. This creates
    spatially localized SSIM values (per-pixel SSIM scores).

    INPUTS:
        size:  Width/height of kernel (must be odd; e.g., 11 for 11×11 kernel)
        sigma: Standard deviation of the Gaussian; controls spread

    RETURNS:
        2D numpy array (size × size) with Gaussian values summing to 1.0

    EXAMPLE:
        >>> kernel = _gaussian_kernel(11, 1.5)
        >>> kernel.shape
        (11, 11)
        >>> kernel.sum()
        0.9999999999999999  # ≈ 1.0
    """
    # Generate 1D Gaussian from -radius to +radius.
    # Example with size=11: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)

    # Gaussian 1D formula: exp(-(x²) / (2σ²))
    # Results in a bell curve centered at 0.
    g1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))

    # Outer product creates 2D Gaussian: G[i,j] = g1d[i] * g1d[j]
    # This ensures radial symmetry (equal weight at equal distances from center).
    g2d = np.outer(g1d, g1d)

    # Normalize so values sum to 1.0 (required for unbiased local mean computation).
    return g2d / g2d.sum()


def _local_mean(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Compute local Gaussian-weighted mean at each pixel using convolution.

    For each pixel, this computes the mean of its local neighborhood weighted
    by the Gaussian kernel — producing a smoothed map of local brightness.

    INPUTS:
        image:  Input image (H × W, float64)
        kernel: 2D Gaussian kernel (normalized to sum to 1.0)

    RETURNS:
        Local mean map (H × W), same spatial shape as input
    """
    # Convolution with 'reflect' padding mirrors edges to avoid boundary artifacts,
    # unlike zero-padding which would artificially darken image borders.
    return convolve(image, kernel, mode='reflect')


def run(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Compute Structural Similarity Index Measure (SSIM) between two RGB images.

    Main computation function that orchestrates the entire SSIM pipeline:
    1. Convert RGB images to grayscale (SSIM is computed on luminance)
    2. Build Gaussian kernel for local windowing
    3. Compute local statistics (means, variances, covariance)
    4. Calculate SSIM components (luminance, contrast, structure)
    5. Compute per-pixel SSIM map and global score

    INPUTS:
        img1, img2: RGB images (H × W × 3, float64 in [0,1])
        params: Dictionary with keys:
            - 'ssim_window_size': Size of the Gaussian kernel (odd int)
            - 'ssim_sigma':       Standard deviation of the Gaussian kernel
            - 'save_intermediates': Whether to return component maps (L, C, S)

    RETURNS:
        Dictionary with keys:
        - 'score':    Overall SSIM score (float, 0.0 to 1.0)
        - 'ssim_map': Per-pixel SSIM values (H × W array)
        - 'l_map':    Luminance component map (H × W, if save_intermediates=True)
        - 'c_map':    Contrast component map  (H × W, if save_intermediates=True)
        - 's_map':    Structure component map (H × W, if save_intermediates=True)

    INTERPRETATION:
        - score = 1.0: Images are identical
        - score = 0.90: Very similar (typical pass threshold)
        - score = 0.50: Moderately different
        - score = 0.00: Completely different
    """
    # ========================================================================
    # STEP 1: Convert RGB → Grayscale
    # ========================================================================
    # SSIM operates on luminance (brightness), not full RGB.
    # ITU-R BT.601 weights account for human eye sensitivity: green > red > blue.
    def to_gray(img):
        return (0.2989 * img[:, :, 0] +
                0.5870 * img[:, :, 1] +
                0.1140 * img[:, :, 2])

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    # ========================================================================
    # STEP 2: Define stabilizing constants
    # ========================================================================
    # Small constants prevent division-by-zero when both images have near-zero
    # variance in a local region (e.g., a uniform background patch).
    # Values from Wang et al. (2004), optimized for normalized [0,1] images.
    C1 = (0.01) ** 2   # Luminance stabilizer
    C2 = (0.03) ** 2   # Contrast stabilizer
    C3 = C2 / 2.0      # Structure stabilizer

    # Build Gaussian kernel using parameters supplied by the caller.
    kernel = _gaussian_kernel(params['ssim_window_size'], params['ssim_sigma'])

    # ========================================================================
    # STEP 3: Compute local statistics via Gaussian-weighted convolution
    # ========================================================================
    # For each pixel, compute statistics across its local Gaussian-weighted
    # neighborhood. These are the building blocks for all SSIM components.

    # Local means: μ₁ = E[gray1], μ₂ = E[gray2]
    mu1 = _local_mean(gray1, kernel)
    mu2 = _local_mean(gray2, kernel)

    # Pre-compute squared means for efficiency.
    mu1_sq = mu1 * mu1   # μ₁²
    mu2_sq = mu2 * mu2   # μ₂²
    mu12   = mu1 * mu2   # μ₁μ₂

    # Local variances: σ² = E[X²] - (E[X])²
    sigma1_sq = _local_mean(gray1 * gray1, kernel) - mu1_sq   # Var(gray1)
    sigma2_sq = _local_mean(gray2 * gray2, kernel) - mu2_sq   # Var(gray2)
    sigma12   = _local_mean(gray1 * gray2, kernel) - mu12      # Cov(gray1, gray2)

    # Standard deviations — np.maximum guards against tiny negative values
    # that can appear due to floating-point precision in the variance formula.
    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0.0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0.0))

    # ========================================================================
    # STEP 4: Calculate SSIM components
    # ========================================================================

    # Luminance: how similar are the local brightness levels?
    # l_map = (2μ₁μ₂ + C1) / (μ₁² + μ₂² + C1)
    l_map = (2.0 * mu12 + C1) / (mu1_sq + mu2_sq + C1)

    # Contrast: how similar are local texture variations?
    # c_map = (2σ₁σ₂ + C2) / (σ₁² + σ₂² + C2)
    c_map = (2.0 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)

    # Structure: how well do local edge patterns correlate?
    # s_map = (σ₁₂ + C3) / (σ₁σ₂ + C3)
    s_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # ========================================================================
    # STEP 5: Combine into per-pixel SSIM and compute global score
    # ========================================================================
    # SSIM(x,y) = L(x,y) × C(x,y) × S(x,y)
    # All three components must be high for overall high SSIM at a pixel.
    ssim_map = l_map * c_map * s_map

    # Global score: mean across all pixels. Range: 0.0 (different) → 1.0 (identical).
    score = float(np.mean(ssim_map))

    # ========================================================================
    # STEP 6: Build result dictionary
    # ========================================================================
    result = {
        "score":    score,     # Scalar: overall SSIM
        "ssim_map": ssim_map,  # H×W: per-pixel SSIM values
    }

    # Component maps are useful for diagnosing which perceptual dimension caused
    # a failure — was it brightness, texture, or structural edges?
    if params.get('save_intermediates', False):
        result["l_map"] = l_map   # H×W: luminance similarity
        result["c_map"] = c_map   # H×W: contrast similarity
        result["s_map"] = s_map   # H×W: structure similarity

    return result
