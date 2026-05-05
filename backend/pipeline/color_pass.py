# ================================================================================
# color_pass.py — Phase 2c: Bhattacharyya Color Distribution Comparison
# ================================================================================
#
# PURPOSE:
# This module compares the color distributions of two images using the
# Bhattacharyya distance. Where SSIM measures perceptual similarity and the
# edge pass measures structural preservation, this pass specifically asks:
# "Has the optimization shifted the color profile of the image?"
#
# A rendering optimization might maintain structure (edges in the right places)
# while introducing color grading, tonemapping shifts, or saturation changes.
# Those would go undetected by SSIM and Canny but show up clearly here.
#
# ROLE IN PIPELINE:
# This is Phase 2c of the analysis. It receives normalized float64 images from
# acquisition and produces:
#   - A per-channel Bhattacharyya distance (H, S, V channels)
#   - A mean distance score across all channels
#   - A pass/fail flag against the configured threshold
#   - An optional histogram overlay plot for visual inspection
#
# BHATTACHARYYA OVERVIEW:
# 1. Convert both images to HSV color space
#      → Hue and Saturation carry color information independent of brightness
# 2. Compute normalized histograms for each channel
#      → Each histogram is a probability distribution over color values
# 3. Compute Bhattacharyya coefficient for each channel pair
#      BC(P, Q) = Σ sqrt(P(i) * Q(i))
#      → Measures overlap between two distributions: 0 (none) to 1 (identical)
# 4. Convert to Bhattacharyya distance
#      D = -ln(BC)
#      → 0.0 = identical, ∞ = no overlap
# 5. Average distance across channels → overall score
#
# ================================================================================

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ================================================================================
# Internal helpers
# ================================================================================

def _to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Convert a float64 RGB image to uint8 HSV for histogram computation.

    HSV is preferred over RGB for color distribution analysis because:
      - Hue (H):        Pure color (red, green, blue...) independent of brightness
      - Saturation (S): Color vividness, independent of brightness
      - Value (V):      Brightness only
    This separation means color shifts are cleanly captured in H and S, while
    exposure changes show up in V — rather than being entangled across all three
    RGB channels.

    INPUTS:
        img: H × W × 3 float64 array in [0.0, 1.0] (RGB)

    RETURNS:
        H × W × 3 uint8 array in OpenCV HSV range:
            H: [0, 179], S: [0, 255], V: [0, 255]
    """
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)


def _normalized_histogram(channel: np.ndarray, n_bins: int, value_range: tuple) -> np.ndarray:
    """
    Compute a normalized histogram (probability distribution) for a single channel.

    The histogram counts how many pixels fall into each bin, then normalizes by
    the total pixel count so the result is a proper probability distribution
    (all bins sum to 1.0). This normalization is essential for Bhattacharyya —
    we're comparing shapes of distributions, not absolute pixel counts.

    INPUTS:
        channel:     1D or 2D array of pixel values for a single channel
        n_bins:      Number of histogram bins
        value_range: (min, max) range of valid values for this channel

    RETURNS:
        1D float64 array of length n_bins, summing to 1.0
    """
    hist, _ = np.histogram(channel.ravel(), bins=n_bins, range=value_range)

    # Normalize to a probability distribution.
    # If the channel is somehow empty, return a uniform distribution.
    total = hist.sum()
    if total > 0:
        return hist.astype(np.float64) / total
    return np.ones(n_bins, dtype=np.float64) / n_bins


def _bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the Bhattacharyya distance between two probability distributions.

    The Bhattacharyya coefficient (BC) measures histogram overlap:
        BC(P, Q) = Σ_i  sqrt( P(i) * Q(i) )

    For each bin i, we multiply the two probabilities and take the square root.
    The sqrt softens the penalty: a single mismatched bin doesn't completely
    collapse the score, making the measure robust to minor histogram differences.

    The coefficient is in [0, 1]:
        BC = 1 → distributions are identical (maximum overlap)
        BC = 0 → distributions share no bins (zero overlap)

    The distance is the negative log of the coefficient:
        D = -ln(BC)
        D = 0   → identical distributions
        D → ∞   → no overlap

    INPUTS:
        p, q: 1D float64 probability distributions (same length, summing to 1.0)

    RETURNS:
        float: Bhattacharyya distance in [0.0, ∞)
    """
    # Bhattacharyya coefficient: sum of geometric means per bin.
    bc = np.sum(np.sqrt(p * q))

    # Clamp to a valid range before taking the log.
    # BC can slightly exceed 1.0 or go below 0.0 due to floating-point
    # precision in histogram normalization.
    bc = np.clip(bc, 1e-10, 1.0)

    # Convert similarity to distance via negative natural log.
    return float(-np.log(bc))


# ================================================================================
# Public interface
# ================================================================================

def run(img_baseline: np.ndarray, img_optimized: np.ndarray, params: dict) -> dict:
    """
    Run the Bhattacharyya color distribution comparison pass.

    Computes per-channel histogram distances between baseline and optimized
    images in HSV space, then averages them into a single score. A low score
    indicates the color profile has been well preserved; a high score signals
    a meaningful color shift.

    INPUTS:
        img_baseline:  H × W × 3 float64 [0,1] — reference image
        img_optimized: H × W × 3 float64 [0,1] — test image
        params: Dictionary with keys:
            - 'color_hist_bins':      Number of histogram bins per channel
            - 'color_distance_thresh': Maximum acceptable mean distance
            - 'save_intermediates':   Whether to return histogram plot data

    RETURNS:
        Dictionary with keys:
        - 'score':          Mean Bhattacharyya distance across channels (float)
        - 'passed':         True if score ≤ color_distance_thresh
        - 'distances':      Per-channel distances {'H': ..., 'S': ..., 'V': ...}
        - 'histograms':     Per-channel histogram arrays for plotting
                            (only present if save_intermediates=True)
                            Format: {'H': (p_baseline, p_optimized), ...}
    """
    n_bins = params['color_hist_bins']
    save   = params.get('save_intermediates', False)

    # ========================================================================
    # STEP 1: Convert both images to HSV
    # ========================================================================
    hsv_baseline  = _to_hsv(img_baseline)
    hsv_optimized = _to_hsv(img_optimized)

    # ========================================================================
    # STEP 2: Compute normalized histograms for each channel
    # ========================================================================
    # OpenCV HSV ranges: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]
    # Each channel gets its own range to fill the bins meaningfully.
    channel_configs = [
        ('H', 0, (0,   180)),
        ('S', 1, (0,   256)),
        ('V', 2, (0,   256)),
    ]

    distances  = {}
    histograms = {}

    for name, idx, val_range in channel_configs:
        p = _normalized_histogram(hsv_baseline[:, :, idx],  n_bins, val_range)
        q = _normalized_histogram(hsv_optimized[:, :, idx], n_bins, val_range)

        distances[name]  = _bhattacharyya_distance(p, q)

        # Store histogram pairs for downstream visualization.
        if save:
            histograms[name] = (p, q)

    # ========================================================================
    # STEP 3: Average across channels → overall score
    # ========================================================================
    # All three channels are weighted equally. If desired, you could weight
    # H more heavily than V for applications where color hue is paramount.
    score  = float(np.mean(list(distances.values())))
    passed = score <= params['color_distance_thresh']

    # ========================================================================
    # Assemble result dictionary
    # ========================================================================
    result = {
        "score":     score,
        "passed":    passed,
        "distances": distances,
    }

    if save:
        result["histograms"] = histograms

    return result


# ================================================================================
# Intermediate visualization
# ================================================================================

def save_histogram_overlay(output_dir: str, color_result: dict) -> str:
    """
    Save an overlay plot of the baseline and optimized HSV histograms.

    Renders one subplot per HSV channel, with both distributions drawn on the
    same axes. This makes it immediately visible whether the two images have
    similar color profiles or not — aligned peaks mean good color preservation,
    while shifted or differently shaped curves signal a color change.

    INPUTS:
        output_dir:   Directory where the PNG will be written
        color_result: The dictionary returned by run(), must contain 'histograms'

    RETURNS:
        str: File path where the PNG was written

    OUTPUT FILE:
        {output_dir}/histogram_overlay.png
    """
    histograms = color_result.get("histograms")
    if not histograms:
        raise ValueError(
            "color_result does not contain histogram data. "
            "Run color_pass.run() with save_intermediates=True."
        )

    distances = color_result["distances"]

    # Styling consistent with the rest of the visual reporter.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1a1a2e')
    fig.suptitle(
        "Color Distribution Comparison — HSV Histogram Overlay",
        color='white', fontsize=13, fontweight='bold', y=1.02
    )

    channel_meta = {
        'H': {'label': 'Hue',        'color_b': '#f48fb1', 'color_o': '#f06292', 'x_max': 180},
        'S': {'label': 'Saturation', 'color_b': '#80cbc4', 'color_o': '#26a69a', 'x_max': 256},
        'V': {'label': 'Value',      'color_b': '#ffe082', 'color_o': '#ffca28', 'x_max': 256},
    }

    for ax, (name, (p_base, p_opt)) in zip(axes, histograms.items()):
        meta   = channel_meta[name]
        n_bins = len(p_base)
        x      = np.linspace(0, meta['x_max'], n_bins)

        ax.set_facecolor('#16213e')

        # Baseline: filled area under curve for easy comparison.
        ax.fill_between(x, p_base, alpha=0.45, color=meta['color_b'], label='Baseline')
        ax.plot(x, p_base, color=meta['color_b'], linewidth=1.5)

        # Optimized: filled area + outline.
        ax.fill_between(x, p_opt, alpha=0.45, color=meta['color_o'], label='Optimized')
        ax.plot(x, p_opt, color=meta['color_o'], linewidth=1.5)

        ax.set_title(
            f"{meta['label']} Channel\nBhattacharyya Distance: {distances[name]:.4f}",
            color='white', fontsize=11, pad=8
        )
        ax.set_xlabel("Pixel Value", color='#aaaaaa', fontsize=9)
        ax.set_ylabel("Probability",  color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

        ax.legend(
            facecolor='#1a1a2e', edgecolor='#333355',
            labelcolor='white',  fontsize=9
        )

    # Mean score annotation at the bottom of the figure.
    fig.text(
        0.5, -0.04,
        f"Mean Bhattacharyya Distance: {color_result['score']:.4f}  |  "
        f"{'PASS ✓' if color_result['passed'] else 'FAIL ✗'}",
        ha='center', color='white', fontsize=11, fontweight='bold'
    )

    plt.tight_layout()

    path = os.path.join(output_dir, "histogram_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return path
