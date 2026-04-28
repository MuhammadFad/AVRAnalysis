# =========================================================================
#  acquisition.py — Phase 1: Data Acquisition
#  Loads baseline and optimized images, validates them, returns as
#  normalised float64 numpy arrays in range [0, 1].
# =========================================================================

import cv2
import numpy as np


def load_images(baseline_path: str, optimized_path: str):
    """
    Load baseline and optimized images from disk.

    OpenCV loads images as BGR by default — we convert to RGB immediately
    so all downstream processing works in the standard channel order.

    Images are returned as float64 arrays in [0, 1].
    Both images must be the same resolution.

    Returns:
        img_baseline  — H x W x 3 float64 numpy array [0, 1]
        img_optimized — H x W x 3 float64 numpy array [0, 1]
    """
    # Load as BGR uint8 (OpenCV default)
    raw_baseline  = cv2.imread(baseline_path)
    raw_optimized = cv2.imread(optimized_path)

    if raw_baseline is None:
        raise FileNotFoundError(f"Baseline image not found: {baseline_path}")
    if raw_optimized is None:
        raise FileNotFoundError(f"Optimized image not found: {optimized_path}")

    # Convert BGR → RGB
    raw_baseline  = cv2.cvtColor(raw_baseline,  cv2.COLOR_BGR2RGB)
    raw_optimized = cv2.cvtColor(raw_optimized, cv2.COLOR_BGR2RGB)

    # Validate matching dimensions
    if raw_baseline.shape != raw_optimized.shape:
        raise ValueError(
            f"Image size mismatch: baseline {raw_baseline.shape} "
            f"vs optimized {raw_optimized.shape}"
        )

    # Validate RGB (3 channels)
    if raw_baseline.ndim != 3 or raw_baseline.shape[2] != 3:
        raise ValueError("Images must be RGB (3 channels).")

    # Convert uint8 [0,255] → float64 [0,1]
    img_baseline  = raw_baseline.astype(np.float64)  / 255.0
    img_optimized = raw_optimized.astype(np.float64) / 255.0

    h, w, _ = img_baseline.shape
    print(f"  >> Images loaded: {w} x {h} pixels.")

    return img_baseline, img_optimized
