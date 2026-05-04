# ================================================================================
# config.py — Configuration & Tunable Parameters
# ================================================================================
#
# This module centralizes ALL tunable parameters for the Visual Regression 
# Analysis pipeline. By keeping all constants in one place, we avoid hardcoding
# magic numbers throughout the codebase and make it easy to experiment with
# different thresholds without modifying core logic.
#
# IMPORTANT: Never hardcode threshold or parameter values in other files.
# Always import from this config module to maintain consistency and enable
# easy parameter tuning for different datasets or use cases.
#
# ================================================================================

# ============================
# SSIM (Structural Similarity) Parameters
# ============================
# These parameters control the SSIM computation which measures perceptual
# similarity between baseline and optimized images.

# SSIM_THRESHOLD: The minimum acceptable SSIM score (0.0 to 1.0)
#   - 0.90: Very strict; minor optimizations will trigger FAIL
#   - 0.85: Moderate; allows some visual degradation
#   - 0.80: Lenient; only large differences trigger FAIL
#   Recommendation: 0.90 for quality-critical applications
SSIM_THRESHOLD = 0.90

# SSIM_WINDOW_SIZE: Size of the Gaussian kernel for local SSIM computation
#   - Larger values (e.g., 21): Looks at broader regions, catches large-scale degradation
#   - Smaller values (e.g., 5): Sensitive to fine details, catches subtle differences
#   - Default 11: Good balance for game rendering analysis
SSIM_WINDOW_SIZE = 11

# SSIM_SIGMA: Standard deviation of the Gaussian kernel
#   - Controls the "blur" of the local window
#   - Default 1.5: Works well with window size 11
SSIM_SIGMA = 1.5

# ============================
# Heatmap & Region Detection Parameters
# ============================
# These parameters control how degraded regions are detected and visualized.

# HEATMAP_ALPHA: Blending factor between heatmap and optimized image (0.0 to 1.0)
#   - 0.55: 55% heatmap, 45% original image (default; good visual balance)
#   - Higher values: More prominent heatmap, harder to see original
#   - Lower values: More of original image visible, harder to see degradation
HEATMAP_ALPHA = 0.55

# DEGRADATION_THRESH: Sensitivity threshold for detecting degraded regions (0.0 to 1.0)
#   - 0.30: Moderate; catches noticeable quality differences
#   - 0.40: Stricter; only flags significant degradation
#   - 0.20: Looser; flags even minor differences
#   High values (0.3-0.4) avoid detecting compression artifacts as true degradation
DEGRADATION_THRESH = 0.30

# MORPH_RADIUS: Radius of morphological operations for post-processing (in pixels)
#   - Applied to binary degradation mask to close small holes and reduce noise
#   - Default 15: Effective for typical game render resolutions
#   - Larger values: More "smoothing", loses fine detail
#   - Smaller values: Preserves detail, keeps fragmented regions
MORPH_RADIUS = 15

# MIN_REGION_AREA: Minimum pixel area for a region to be reported (in pixels²)
#   - Filters out noise and trivial degradations
#   - Default 500: Ignores tiny ~22x22 pixel artifacts
#   - Higher values: Only reports significant degradations
#   - Lower values: Reports even pixel-level differences
MIN_REGION_AREA = 500

# ============================
# Debug & Intermediate Output
# ============================
# Controls verbosity and intermediate artifact generation.

# SAVE_INTERMEDIATES: Whether to save SSIM component maps (luminance, contrast, structure)
#   - True: Saves L, C, S maps to output/ directory for detailed analysis
#   - False: Only saves final summary heatmap
#   - Useful for debugging and understanding which component caused failures
SAVE_INTERMEDIATES = True

# ============================
# Output Directory
# ============================
# Where all generated reports and visualizations are saved

# OUTPUT_DIR: Directory path where analysis results are written
#   - Contains: regression_report.json, heatmap_composite.png, analysis_figure.png
#   - Directory is created automatically if it doesn't exist
OUTPUT_DIR = "output"