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
# Canny Edge Detection Parameters
# ============================
# These parameters control the edge detection pass, which compares structural
# edge content between the baseline and optimized images.

# CANNY_BLUR_KERNEL: Size of the Gaussian blur kernel applied before edge detection
#   - Must be an odd integer
#   - Larger values: More aggressive noise removal, may lose fine edges
#   - Smaller values: Less blurring, more sensitive to noise
#   - Default 5: Good balance for typical rendering resolutions
CANNY_BLUR_KERNEL = 5

# CANNY_BLUR_SIGMA: Standard deviation of the pre-detection Gaussian blur
#   - Controls how aggressively noise is smoothed before gradient computation
#   - Default 1.0: Mild smoothing, preserves most structural edges
CANNY_BLUR_SIGMA = 1.0

# CANNY_LOW_THRESH: Lower hysteresis threshold for weak edge classification
#   - Pixels with gradient magnitude below this are discarded entirely
#   - Default 50: Works well for 8-bit image content (0–255 range)
CANNY_LOW_THRESH = 50

# CANNY_HIGH_THRESH: Upper hysteresis threshold for strong edge classification
#   - Pixels above this are immediately accepted as strong edges
#   - Pixels between LOW and HIGH are kept only if connected to a strong edge
#   - Ratio to LOW is typically 2:1 or 3:1; default maintains 3:1
#   - Default 150
CANNY_HIGH_THRESH = 150

# CANNY_EDGE_MATCH_THRESH: Minimum fraction of baseline edges that must be
# present in the optimized image for the pass to succeed (0.0 to 1.0)
#   - 0.85: At least 85% of structural edges must be preserved
#   - Higher: Stricter edge preservation requirement
#   - Lower: More tolerant of lost fine detail
CANNY_EDGE_MATCH_THRESH = 0.85

# ============================
# Bhattacharyya Color Filter Parameters
# ============================
# These parameters control the color distribution comparison pass, which
# measures how closely the optimized image preserves the color profile
# of the baseline.

# COLOR_HIST_BINS: Number of bins per channel in the color histogram
#   - More bins: Finer color discrimination, but requires more pixels to be reliable
#   - Fewer bins: More robust with small regions, but less discriminating
#   - Default 64: Good balance between precision and stability
COLOR_HIST_BINS = 64

# COLOR_SPACE: Color space in which histograms are computed
#   - 'HSV': Separates hue/saturation from value (brightness); preferred for
#            color matching because it is more perceptually meaningful than RGB
#   - 'RGB': Straightforward but mixes color and brightness information
COLOR_SPACE = 'HSV'

# COLOR_DISTANCE_THRESH: Maximum acceptable Bhattacharyya distance (0.0 to ∞)
#   - 0.0: Distributions are identical
#   - Higher values: Greater dissimilarity
#   - Default 0.3: Allows minor color shifts while flagging significant changes
#   - Tune down (e.g., 0.1) for strict color fidelity requirements
COLOR_DISTANCE_THRESH = 0.3

# ============================
# Debug & Intermediate Output
# ============================
# Controls verbosity and intermediate artifact generation for each pass.

# SAVE_INTERMEDIATES: Master switch — whether any pass saves intermediate maps
#   - True: Each pass writes its diagnostic images to its output subdirectory
#   - False: Only final composite outputs are written
SAVE_INTERMEDIATES = True

# ============================
# Output Directories
# ============================
# Root output directory and per-pass subdirectories for organized results.
# All directories are created automatically by main.py at startup.

OUTPUT_DIR        = "output"
OUTPUT_DIR_SSIM   = "output/ssim"    # SSIM component maps (L, C, S)
OUTPUT_DIR_EDGE   = "output/edge"    # Canny intermediates (Sobel X/Y, NMS, final)
OUTPUT_DIR_COLOR  = "output/color"   # Bhattacharyya histogram overlay
