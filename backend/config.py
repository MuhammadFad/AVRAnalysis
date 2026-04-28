# ===========
# config.py — All tunable parameters in one place.
# Change thresholds here. Never hardcode them in other files.
# ===========

# --- SSIM ---
SSIM_THRESHOLD = 0.90
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5

# --- Heatmap ---
HEATMAP_ALPHA = 0.55
DEGRADATION_THRESH = 0.30
MORPH_RADIUS = 15
MIN_REGION_AREA = 500

# --- Debug ---
SAVE_INTERMEDIATES = True  # if True, saves ssim component maps to output/

# --- Output ---
OUTPUT_DIR = "output"