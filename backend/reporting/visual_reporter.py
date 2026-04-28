# ======
# visual_reporter.py — Visual report generation.
# ======

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def _normalize_cs(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.min(arr), np.max(arr)
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)

def _normalize_he(arr: np.ndarray) -> np.ndarray:
    """
    Enhanced normalization using CLAHE to make SSIM components 
    visually distinct and interesting.
    """
    # 1. Initial linear stretch to 0-255
    lo, hi = np.min(arr), np.max(arr)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    
    # Convert float64 [0,1] to uint8 [0,255] for OpenCV
    rescaled = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit: higher means more contrast; tileGridSize: local neighborhood size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(rescaled)
    
    return enhanced

def save_heatmap(output_dir: str, composite: np.ndarray, boxes: list) -> str:
    """Draw bounding boxes on the composite heatmap and save as PNG."""
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)

    for (x, y, w, h) in boxes:
        cv2.rectangle(composite_uint8, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)

    composite_bgr = cv2.cvtColor(composite_uint8, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, "heatmap_composite.png")
    cv2.imwrite(path, composite_bgr)
    return path

def save_figure(output_dir: str, img_baseline: np.ndarray, img_optimized: np.ndarray, composite: np.ndarray, results: dict) -> str:
    """Generate and save the multi-panel analysis figure."""
    ssim_score = results["ssim_score"]
    ssim_passed = results["ssim_passed"]
    boxes = results["boxes"]
    params = results["params"]
    l_map = results.get("l_map")
    c_map = results.get("c_map")
    s_map = results.get("s_map")

    has_components = all(m is not None for m in [l_map, c_map, s_map])
    n_rows = 3 if has_components else 2

    # Figure setup
    panel_bg = '#e0e0e0'
    title_color = '#16213e'
    metric_color = '#90caf9'
    pass_color = '#00c853'
    fail_color = '#ff1744'

    verdict = "PASS" if ssim_passed else "FAIL"
    verdict_color = pass_color if ssim_passed else fail_color

    fig_height = 12 if has_components else 8
    fig = plt.figure(figsize=(20, fig_height), facecolor='#1a1a2e')
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.40, wspace=0.25, left=0.04, right=0.97, top=0.92, bottom=0.06)

    fig.suptitle("Visual Regression Analyzer — SSIM Analysis Report", color='white', fontsize=15, fontweight='bold', y=0.96)

    # Row 0: baseline, optimized, heatmap
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)
    for (x, y, w, h) in boxes:
        cv2.rectangle(composite_uint8, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_baseline)
    ax1.set_title("Original (Baseline)", color='white', fontsize=10, pad=6)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_optimized)
    ax2.set_title("Optimized", color='white', fontsize=10, pad=6)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(composite_uint8)
    ax3.set_title(f"SSIM Heatmap ({len(boxes)} degraded region(s))", color='white', fontsize=10, pad=6)
    ax3.axis('off')

    # Row 1: SSIM components (optional)
    if has_components:
        components = [
            (l_map, "Luminance", "Brightness similarity"),
            (c_map, "Contrast", "Texture similarity"),
            (s_map, "Structure", "Pattern correlation")
        ]
        for col, (arr, label, sublabel) in enumerate(components):
            ax = fig.add_subplot(gs[1, col])

            # ==================================================================
            # PERFORM NORMALIZATION HERE
            # ==================================================================

            ax.imshow(_normalize_he(arr), cmap='hot')
            ax.set_title(f"{label} Component\n({sublabel})", color='white', fontsize=10, pad=6)
            ax.axis('off')

    # Summary bar (last row)
    ax_s = fig.add_subplot(gs[n_rows - 1, :])
    ax_s.set_facecolor('#16213e')
    ax_s.axis('off')

    # Verdict
    ax_s.text(0.08, 0.92, "OVERALL VERDICT", transform=ax_s.transAxes, ha='left', va='top', color='white', fontsize=11, fontweight='bold')
    ax_s.text(0.08, 0.72, verdict, transform=ax_s.transAxes, ha='left', va='top', color=verdict_color, fontsize=32, fontweight='bold', fontfamily='monospace')

    # SSIM Score
    ax_s.text(0.30, 0.92, "SSIM Score", transform=ax_s.transAxes, color=metric_color, fontsize=10, fontfamily='monospace')
    ax_s.text(0.30, 0.72, f"{ssim_score:.4f}", transform=ax_s.transAxes, color='white', fontsize=20, fontweight='bold', fontfamily='monospace')
    ax_s.text(0.30, 0.56, f"Threshold: {params['ssim_threshold']}", transform=ax_s.transAxes, color='#888888', fontsize=9, fontfamily='monospace')
    ax_s.text(0.30, 0.46, "PASS ✓" if ssim_passed else "FAIL ✗", transform=ax_s.transAxes, color=verdict_color, fontsize=11, fontweight='bold', fontfamily='monospace')

    # Degraded Regions
    ax_s.text(0.52, 0.92, "Degraded Regions", transform=ax_s.transAxes, color=metric_color, fontsize=10, fontfamily='monospace')
    ax_s.text(0.52, 0.72, str(len(boxes)), transform=ax_s.transAxes, color='white', fontsize=32, fontweight='bold', fontfamily='monospace')
    ax_s.text(0.52, 0.46, "detected" if boxes else "none found", transform=ax_s.transAxes, color='#888888', fontsize=9, fontfamily='monospace')

    # Footer
    ax_s.text(0.5, 0.08, "Visual Regression Analyzer v1.0 | SSIM Phase Report", transform=ax_s.transAxes, ha='center', color='#555577', fontsize=7, fontfamily='monospace')

    path = os.path.join(output_dir, "analysis_figure.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path