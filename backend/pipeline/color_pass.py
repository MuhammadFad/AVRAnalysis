# ================================================================================
# color_pass.py — Phase 2c: Bhattacharyya Color  (enriched schema for API)
# ================================================================================
#
# Added for web API when save_intermediates=True:
#   'intermediates': {
#       'baseline_hsv_channels':  base64 PNG — H/S/V channel images for baseline
#       'optimized_hsv_channels': base64 PNG — H/S/V channel images for optimized
#       'overlay_hist':           base64 PNG — histogram overlay (all 3 channels)
#   }
#
# ================================================================================

import io
import base64

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ================================================================================
# Helpers
# ================================================================================

def _to_hsv(img: np.ndarray) -> np.ndarray:
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)


def _normalized_histogram(channel: np.ndarray, n_bins: int, value_range: tuple) -> np.ndarray:
    hist, _ = np.histogram(channel.ravel(), bins=n_bins, range=value_range)
    total   = hist.sum()
    return hist.astype(np.float64) / total if total > 0 else np.ones(n_bins) / n_bins


def _bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    bc = np.clip(np.sum(np.sqrt(p * q)), 1e-10, 1.0)
    return float(-np.log(bc))


def _encode_channel_strip(hsv: np.ndarray, label: str) -> str:
    """
    Build a 1×3 strip of H, S, V channel images as a single base64 PNG.
    Each channel is rendered with an appropriate colormap.
    """
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Normalise each channel to [0,255]
    def norm(ch, max_val):
        return (ch.astype(np.float32) / max_val * 255).astype(np.uint8)

    h_img = cv2.applyColorMap(norm(h_ch, 179), cv2.COLORMAP_HSV)
    s_img = cv2.applyColorMap(norm(s_ch, 255), cv2.COLORMAP_PLASMA)
    v_img = cv2.applyColorMap(norm(v_ch, 255), cv2.COLORMAP_BONE)

    strip  = np.hstack([h_img, s_img, v_img])
    ok, buf = cv2.imencode('.png', strip)
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ""


def _encode_hist_overlay(histograms: dict, distances: dict, score: float, passed: bool) -> str:
    """Render the histogram overlay figure and return as base64 PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#1a1a2e')
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
        ax.fill_between(x, p_base, alpha=0.45, color=meta['color_b'], label='Baseline')
        ax.plot(x, p_base, color=meta['color_b'], linewidth=1.5)
        ax.fill_between(x, p_opt,  alpha=0.45, color=meta['color_o'], label='Optimized')
        ax.plot(x, p_opt,  color=meta['color_o'], linewidth=1.5)
        ax.set_title(f"{meta['label']}  Bhattacharyya: {distances[name]:.4f}",
                     color='white', fontsize=10)
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor='white', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ================================================================================
# Public interface
# ================================================================================

def run(img_baseline: np.ndarray, img_optimized: np.ndarray, params: dict) -> dict:
    """
    Run Bhattacharyya color comparison.

    Returns score, passed, verdict, distances,
    and optionally intermediates when save_intermediates=True.
    """
    n_bins = params['color_hist_bins']
    save   = params.get('save_intermediates', False)

    hsv_baseline  = _to_hsv(img_baseline)
    hsv_optimized = _to_hsv(img_optimized)

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
        histograms[name] = (p, q)

    score  = float(np.mean(list(distances.values())))
    passed = score <= params['color_distance_thresh']

    result = {
        "score":      score,
        "passed":     passed,
        "verdict":    "PASS" if passed else "FAIL",
        "distances":  distances,
        "histograms": histograms,   # always include for aggregator local crops
    }

    if save:
        result["intermediates"] = {
            "baseline_hsv_channels":  _encode_channel_strip(hsv_baseline,  "Baseline"),
            "optimized_hsv_channels": _encode_channel_strip(hsv_optimized, "Optimized"),
            "overlay_hist":           _encode_hist_overlay(histograms, distances, score, passed),
        }

    return result


# ================================================================================
# CLI helper — saves histogram PNG to disk (used by main.py)
# ================================================================================

def save_histogram_overlay(output_dir: str, color_result: dict) -> str:
    histograms = color_result.get("histograms")
    if not histograms:
        raise ValueError("color_result missing histogram data.")

    distances = color_result["distances"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1a1a2e')
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
        ax.fill_between(x, p_base, alpha=0.45, color=meta['color_b'], label='Baseline')
        ax.plot(x, p_base, color=meta['color_b'], linewidth=1.5)
        ax.fill_between(x, p_opt,  alpha=0.45, color=meta['color_o'], label='Optimized')
        ax.plot(x, p_opt,  color=meta['color_o'], linewidth=1.5)
        ax.set_title(f"{meta['label']}  Bhattacharyya: {distances[name]:.4f}",
                     color='white', fontsize=11, pad=8)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor='white', fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "histogram_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
