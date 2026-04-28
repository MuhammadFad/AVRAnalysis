# =========================================================================
#  config.py — All tunable parameters in one place.
#  Change thresholds here. Never hardcode them in other files.
# =========================================================================

# --- SSIM ---
SSIM_THRESHOLD = 0.90          # FAIL if global SSIM drops below this

# --- Gaussian blur (used inside SSIM sliding window) ---
SSIM_WINDOW_SIZE = 11          # Local window size for SSIM computation
SSIM_SIGMA       = 1.5         # Gaussian sigma for the SSIM window

# --- Heatmap ---
HEATMAP_ALPHA        = 0.55    # Blend opacity of heatmap over optimized image
DEGRADATION_THRESH   = 0.30    # Pixels above this in diff map are "degraded"
MORPH_RADIUS         = 15      # Morphological closing radius in pixels
MIN_REGION_AREA      = 500     # Minimum blob area to draw a bounding box (px²)

# --- Output ---
OUTPUT_DIR = "output"          # Folder where all results are saved
