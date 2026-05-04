# ================================================================================
# acquisition.py — Phase 1: Data Acquisition
# ================================================================================
#
# PURPOSE:
# This module handles the initial loading and validation of image pairs in the
# visual regression analysis pipeline. It is responsible for:
#   1. Reading image files from disk (PNG, JPG, etc.)
#   2. Converting color space from BGR (OpenCV default) to RGB (standard)
#   3. Validating that both images have matching dimensions and 3 channels
#   4. Normalizing pixel values from uint8 [0,255] to float64 [0,1]
#
# ROLE IN PIPELINE:
# This is the entry point for image data. All downstream processing modules
# (SSIM, aggregator, reporters) depend on receiving properly formatted images
# from this module. Any issues here cascade through the entire pipeline.
#
# OUTPUT FORMAT:
# Returns H x W x 3 float64 numpy arrays in range [0.0, 1.0], where:
#   - H = image height in pixels
#   - W = image width in pixels
#   - 3 = RGB channels (Red, Green, Blue)
#   - Values in [0, 1] enable numerically stable computations downstream
#
# ================================================================================

import cv2
import numpy as np


def load_images(baseline_path: str, optimized_path: str):
    """
    Load and normalize baseline and optimized images from disk.
    
    This function is the entry point for the image acquisition phase. It performs
    critical setup work that enables all downstream analysis:
    
    PROCESSING STEPS:
    1. Load images from disk using OpenCV (cv2.imread)
       - OpenCV reads images as BGR by default (Blue-Green-Red channel order)
    2. Convert color space from BGR → RGB
       - Necessary because most computer vision libraries expect RGB order
       - Ensures consistent results with other analysis tools
    3. Validate image compatibility
       - Both images must have identical resolution (height × width)
       - Both must have exactly 3 color channels (RGB)
    4. Convert data type from uint8 [0,255] → float64 [0.0,1.0]
       - uint8 [0,255]: Integer pixel values, standard for image files
       - float64 [0.0,1.0]: Floating-point, enables numerically stable math
       - Essential for SSIM calculations which require high precision
    
    INPUTS:
        baseline_path (str): File path to the original/high-quality reference image
        optimized_path (str): File path to the optimized/lower-quality test image
    
    RETURNS:
        tuple: (img_baseline, img_optimized)
        - img_baseline: H × W × 3 float64 numpy array with values in [0.0, 1.0]
        - img_optimized: H × W × 3 float64 numpy array with values in [0.0, 1.0]
        
    RAISES:
        FileNotFoundError: If either image file cannot be read
        ValueError: If images have different dimensions or are not RGB
    
    EXAMPLE:
        >>> baseline, optimized = load_images('./data/baseline.png', './data/optimized.png')
        >>> print(baseline.shape, baseline.dtype, baseline.min(), baseline.max())
        (1080, 1920, 3) float64 0.0 1.0
    """
    # ========================================================================
    # STEP 1: Load images from disk using OpenCV
    # ========================================================================
    # cv2.imread() reads images as BGR (Blue-Green-Red) uint8 [0,255]
    # This is OpenCV's convention, not our desired RGB format, so we'll convert.
    raw_baseline  = cv2.imread(baseline_path)
    raw_optimized = cv2.imread(optimized_path)

    # Validate files were successfully loaded
    # cv2.imread() returns None if the file cannot be read
    if raw_baseline is None:
        raise FileNotFoundError(f"Baseline image not found: {baseline_path}")
    if raw_optimized is None:
        raise FileNotFoundError(f"Optimized image not found: {optimized_path}")

    # ========================================================================
    # STEP 2: Convert color space from BGR → RGB
    # ========================================================================
    # OpenCV uses BGR by convention (legacy reason from camera libraries)
    # But most computer vision literature, PIL, matplotlib, and numpy convention is RGB
    # We must convert so all downstream modules work with consistent channel order
    raw_baseline  = cv2.cvtColor(raw_baseline,  cv2.COLOR_BGR2RGB)
    raw_optimized = cv2.cvtColor(raw_optimized, cv2.COLOR_BGR2RGB)

    # ========================================================================
    # STEP 3: Validate image compatibility
    # ========================================================================
    # Both images MUST have identical dimensions for pixel-by-pixel comparison
    # in downstream SSIM computation
    if raw_baseline.shape != raw_optimized.shape:
        raise ValueError(
            f"Image size mismatch: baseline {raw_baseline.shape} "
            f"vs optimized {raw_optimized.shape}"
        )

    # Validate that images have exactly 3 color channels (RGB)
    # Grayscale or RGBA images will cause issues in SSIM computation
    if raw_baseline.ndim != 3 or raw_baseline.shape[2] != 3:
        raise ValueError("Images must be RGB (3 channels).")

    # ========================================================================
    # STEP 4: Convert pixel values from uint8 [0,255] → float64 [0.0,1.0]
    # ========================================================================
    # Rationale:
    #   - File format uses uint8 [0,255] for compact storage
    #   - SSIM math uses float64 for numerical stability and precision
    #   - Normalization to [0,1] makes computations scale-invariant
    #   - Example: (100 uint8) / 255 = 0.39216 float64
    img_baseline  = raw_baseline.astype(np.float64)  / 255.0
    img_optimized = raw_optimized.astype(np.float64) / 255.0

    # ========================================================================
    # Report success
    # ========================================================================
    h, w, _ = img_baseline.shape
    print(f"  >> Images loaded: {w} x {h} pixels.")

    return img_baseline, img_optimized
