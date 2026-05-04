# ================================================================================
# ssim_pass.py — Structural Similarity Index Measure (SSIM) Computation
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
# This is Phase 2 of the analysis. It receives normalized float64 images from
# the acquisition module and produces:
#   - An overall SSIM score (0.0 to 1.0, where 1.0 = identical images)
#   - A per-pixel SSIM map showing local similarity across the image
#   - Optional component maps (L, C, S) for detailed diagnosis
#
# MATHEMATICAL FOUNDATION:
# SSIM formula (Wang et al., 2004):
#   SSIM(x, y) = (2μ_x*μ_y + C1) / (μ_x² + μ_y² + C1)    [Luminance]
#                * (2σ_x*σ_y + C2) / (σ_x² + σ_y² + C2)   [Contrast]
#                * (σ_xy + C3) / (σ_x*σ_y + C3)           [Structure]
#
# Key insight: Each component captures different perceptual aspects. A low value
# in any component indicates that component caused the degradation.
#
# ================================================================================

import numpy as np
from scipy.ndimage import convolve
from config import SSIM_WINDOW_SIZE, SSIM_SIGMA, SAVE_INTERMEDIATES

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel for local windowing in SSIM computation.
    
    The Gaussian kernel weights the local neighborhood for SSIM calculation,
    giving more importance to pixels near the center window point. This creates
    spatially localized SSIM values (per-pixel SSIM scores).
    
    INPUTS:
        size: Width/height of kernel (must be odd; e.g., 11 for 11×11 kernel)
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
    # Generate 1D Gaussian from -radius to +radius
    # Example with size=11: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    radius = size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    
    # Gaussian 1D formula: exp(-(x²) / (2σ²))
    # Results in a bell curve centered at 0
    g1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    
    # Outer product creates 2D Gaussian: G[i,j] = g1d[i] * g1d[j]
    # This ensures radial symmetry (equal weight at equal distances from center)
    g2d = np.outer(g1d, g1d)
    
    # Normalize so values sum to 1.0 (important for unbiased local mean computation)
    return g2d / g2d.sum()

def _local_mean(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Compute local Gaussian-weighted mean at each pixel using convolution.
    
    This is a core operation in SSIM: for each pixel, we compute the mean of
    its local neighborhood weighted by the Gaussian kernel. The result is a
    smoothed version of the image representing local brightness.
    
    INPUTS:
        image: Input image (H × W × 1 grayscale, float64)
        kernel: 2D Gaussian kernel (must be normalized to sum to 1.0)
    
    RETURNS:
        Local mean map (H × W), same shape as input
    
    EXAMPLE:
        >>> import numpy as np
        >>> img = np.array([[1, 2], [3, 4]], dtype=np.float64)
        >>> kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
        >>> _local_mean(img, kernel).shape
        (2, 2)
    """
    # Convolution with 'reflect' padding: extend image by mirroring edges
    # This avoids boundary artifacts compared to zero-padding
    return convolve(image, kernel, mode='reflect')

def run(img1: np.ndarray, img2: np.ndarray, save_intermediates: bool = SAVE_INTERMEDIATES) -> dict:
    """
    Compute Structural Similarity Index Measure (SSIM) between two RGB images.
    
    Main computation function that orchestrates the entire SSIM pipeline:
    1. Convert RGB images to grayscale (SSIM is typically computed on luminance)
    2. Build Gaussian kernel for local windowing
    3. Compute local statistics (means, variances, covariance)
    4. Calculate SSIM components (luminance, contrast, structure)
    5. Compute per-pixel SSIM map and global score
    
    INPUTS:
        img1, img2: RGB images (H × W × 3, float64 in [0,1])
        save_intermediates: Whether to return component maps (L, C, S)
    
    RETURNS:
        Dictionary with keys:
        - 'score': Overall SSIM score (float, 0.0 to 1.0)
        - 'ssim_map': Per-pixel SSIM values (H × W array)
        - 'l_map': Luminance component (if save_intermediates=True)
        - 'c_map': Contrast component (if save_intermediates=True)
        - 's_map': Structure component (if save_intermediates=True)
    
    INTERPRETATION:
        - score = 1.0: Images are identical
        - score = 0.90: Very similar (typical pass threshold)
        - score = 0.50: Moderately different
        - score = 0.00: Completely different
    """
    # ========================================================================
    # STEP 1: Convert RGB → Grayscale
    # ========================================================================
    # SSIM typically operates on luminance (brightness) channel, not all RGB
    # ITU-R BT.601 weights: standard for converting RGB to perceived brightness
    # These weights account for human eye sensitivity: green > red > blue
    def to_gray(img):
        # R: 0.2989, G: 0.5870, B: 0.1140 (normalized to sum to 1.0)
        # Result: single-channel grayscale image
        return (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2])

    # Convert both images to grayscale for SSIM computation
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    # ========================================================================
    # STEP 2: Define stabilizing constants
    # ========================================================================
    # These small constants prevent division-by-zero when both images have
    # near-zero variance in a local region (e.g., uniform background).
    # Values from Wang et al. (2004) SSIM paper; optimized for 8-bit images
    C1 = (0.01) ** 2   # For luminance component (L = (2μ₁μ₂ + C1) / ...)
    C2 = (0.03) ** 2   # For contrast component (C = (2σ₁σ₂ + C2) / ...)
    C3 = C2 / 2.0      # For structure component (S = (σ₁₂ + C3) / ...)

    # Build Gaussian kernel for local windowing
    # Default: 11×11 kernel with σ=1.5 (good for typical image resolutions)
    kernel = _gaussian_kernel(SSIM_WINDOW_SIZE, SSIM_SIGMA)

    # ========================================================================
    # STEP 3: Compute local statistics via Gaussian-weighted convolution
    # ========================================================================
    # For each pixel, compute statistics in its local Gaussian-weighted neighborhood
    # These are the building blocks for all SSIM components
    
    # Local means: μ₁ = E[gray1], μ₂ = E[gray2]
    mu1 = _local_mean(gray1, kernel)
    mu2 = _local_mean(gray2, kernel)

    # Pre-compute squared means for efficiency
    mu1_sq = mu1 * mu1   # μ₁²
    mu2_sq = mu2 * mu2   # μ₂²
    mu12 = mu1 * mu2     # μ₁μ₂

    # Local variances: σ₁² = E[gray1²] - (E[gray1])²
    # By definition: Var(X) = E[X²] - (E[X])²
    sigma1_sq = _local_mean(gray1 * gray1, kernel) - mu1_sq  # Variance of gray1
    sigma2_sq = _local_mean(gray2 * gray2, kernel) - mu2_sq  # Variance of gray2
    sigma12 = _local_mean(gray1 * gray2, kernel) - mu12       # Covariance (correlation)

    # Convert to standard deviations (square root of variance)
    # np.maximum(..., 0.0) prevents negative values from numerical precision errors
    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0.0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0.0))

    # ========================================================================
    # STEP 4: Calculate SSIM components
    # ========================================================================
    # Each component measures one aspect of perceptual similarity:
    
    # Luminance: How similar are the brightness levels?
    # l_map = (2μ₁μ₂ + C1) / (μ₁² + μ₂² + C1)
    # If both images bright or both dark → high l_map
    # If one bright and one dark → low l_map
    l_map = (2.0 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
    
    # Contrast: How similar are texture variations?
    # c_map = (2σ₁σ₂ + C2) / (σ₁² + σ₂² + C2)
    # If both have similar texture roughness → high c_map
    # If one smooth and one rough → low c_map
    c_map = (2.0 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # Structure: How similar are patterns/edges?
    # s_map = (σ₁₂ + C3) / (σ₁σ₂ + C3)
    # Measures correlation: how well do pixel fluctuations align?
    # High correlation (edges in same places) → high s_map
    s_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # ========================================================================
    # STEP 5: Combine components into per-pixel SSIM and compute global score
    # ========================================================================
    # Final SSIM: multiply all three components
    # SSIM(x,y) = L(x,y) × C(x,y) × S(x,y)
    # This is a geometric mean; all three must be high for overall high SSIM
    ssim_map = l_map * c_map * s_map
    
    # Global SSIM score: average of per-pixel SSIM values
    # Represents overall perceptual similarity between images
    # Range: 0.0 (completely different) to 1.0 (identical)
    score = float(np.mean(ssim_map))

    # ========================================================================
    # STEP 6: Build result dictionary
    # ========================================================================
    result = {
        "score": score,                    # Single scalar: overall SSIM
        "ssim_map": ssim_map,              # H×W map: per-pixel SSIM
    }

    # Optionally save component maps for detailed diagnosis
    # Useful for understanding: was degradation due to brightness, texture, or edges?
    if save_intermediates:
        result["l_map"] = l_map            # H×W: luminance similarity map
        result["c_map"] = c_map            # H×W: contrast similarity map
        result["s_map"] = s_map            # H×W: structure similarity map

    return result