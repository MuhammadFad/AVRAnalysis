# ================================================================================
# config.py — Configuration & Tunable Parameters
# ================================================================================

# SSIM
SSIM_THRESHOLD   = 0.90
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA       = 1.5

# Heatmap / Region Detection
HEATMAP_ALPHA      = 0.55
DEGRADATION_THRESH = 0.30
MORPH_RADIUS       = 15
MIN_REGION_AREA    = 500

# Canny Edge
CANNY_BLUR_KERNEL       = 5
CANNY_BLUR_SIGMA        = 1.0
CANNY_LOW_THRESH        = 50
CANNY_HIGH_THRESH       = 150
CANNY_EDGE_MATCH_THRESH = 0.85

# Bhattacharyya Color
COLOR_HIST_BINS      = 64
COLOR_SPACE          = 'HSV'
COLOR_DISTANCE_THRESH = 0.3

# Aggregator / Combinator
IOU_THRESHOLD        = 0.3
TOP_REGIONS_IN_REPORT = 5

# Resolution cap for API path (protects Render's 1 GB RAM limit)
MAX_IMAGE_DIM = 1920

# Debug / Intermediates
SAVE_INTERMEDIATES = True

# Output Directories (CLI path only)
OUTPUT_DIR       = "output"
OUTPUT_DIR_SSIM  = "output/ssim"
OUTPUT_DIR_EDGE  = "output/edge"
OUTPUT_DIR_COLOR = "output/color"
