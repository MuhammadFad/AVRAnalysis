# ================================================================================
# acquisition.py — Phase 1: Data Acquisition
# ================================================================================
#
# Changes from CLI version:
#   - load_and_align() replaces load_images()
#   - Auto-resizes mismatched images to the smaller of the two dimensions
#   - Caps longest dimension to MAX_IMAGE_DIM (1920px) to protect server RAM
#   - Returns resolution_mismatch bool for API warning banner
#   - load_images() kept as a thin wrapper for CLI (main.py) compatibility
#
# ================================================================================

import cv2
import numpy as np

MAX_IMAGE_DIM = 1920


def load_and_align(baseline_bytes: bytes, optimized_bytes: bytes):
    """
    Decode, align, and normalize two images provided as raw bytes.

    Used by api.py — accepts bytes from UploadFile reads, decodes with OpenCV,
    aligns dimensions if they differ, caps resolution, normalizes to float64.

    RETURNS:
        (img_baseline, img_optimized, resolution_mismatch, orig_dims, analyzed_dims)
        - img_baseline:         H × W × 3 float64 [0,1]
        - img_optimized:        H × W × 3 float64 [0,1]
        - resolution_mismatch:  bool — True if the two uploads had different sizes
        - orig_dims:            {'baseline': [w,h], 'optimized': [w,h]}
        - analyzed_dims:        [w, h] after alignment + cap
    """
    b = cv2.imdecode(np.frombuffer(baseline_bytes,  np.uint8), cv2.IMREAD_COLOR)
    o = cv2.imdecode(np.frombuffer(optimized_bytes, np.uint8), cv2.IMREAD_COLOR)

    if b is None:
        raise ValueError("Could not decode baseline image.")
    if o is None:
        raise ValueError("Could not decode optimized image.")

    bh, bw = b.shape[:2]
    oh, ow = o.shape[:2]

    orig_dims = {
        'baseline':  [bw, bh],
        'optimized': [ow, oh],
    }

    mismatch = (bh, bw) != (oh, ow)

    if mismatch:
        target_h = min(bh, oh)
        target_w = min(bw, ow)
        b = cv2.resize(b, (target_w, target_h), interpolation=cv2.INTER_AREA)
        o = cv2.resize(o, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Cap longest dimension to protect server RAM on large screenshots
    h, w = b.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        scale  = MAX_IMAGE_DIM / max(h, w)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        b = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_AREA)
        o = cv2.resize(o, (new_w, new_h), interpolation=cv2.INTER_AREA)

    analyzed_dims = [b.shape[1], b.shape[0]]  # [w, h]

    # BGR → RGB, uint8 → float64 [0, 1]
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    return b, o, mismatch, orig_dims, analyzed_dims


def load_images(baseline_path: str, optimized_path: str):
    """
    CLI-compatible wrapper around load_and_align for use by main.py.
    Reads from disk paths, raises on mismatch (original behavior).
    """
    raw_b = cv2.imread(baseline_path)
    raw_o = cv2.imread(optimized_path)

    if raw_b is None:
        raise FileNotFoundError(f"Baseline image not found: {baseline_path}")
    if raw_o is None:
        raise FileNotFoundError(f"Optimized image not found: {optimized_path}")

    if raw_b.shape != raw_o.shape:
        raise ValueError(
            f"Image size mismatch: baseline {raw_b.shape} vs optimized {raw_o.shape}"
        )
    if raw_b.ndim != 3 or raw_b.shape[2] != 3:
        raise ValueError("Images must be RGB (3 channels).")

    img_b = cv2.cvtColor(raw_b, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    img_o = cv2.cvtColor(raw_o, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    h, w, _ = img_b.shape
    print(f"  >> Images loaded: {w} x {h} pixels.")

    return img_b, img_o
