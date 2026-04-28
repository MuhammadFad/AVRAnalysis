# =========================================================================
#  json_reporting.py — Report generation
#
#  Produces:
#    1. regression_report.json  — all metrics in structured form
#    2. analysis_figure.png     — 4-panel visual: original, optimized,
#                                 heatmap overlay, metric summary
# =========================================================================

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def save_json(output_dir: str, ssim_score: float, ssim_passed: bool,
              boxes: list, params: dict) -> str:
    """
    Write the JSON report to disk.

    Returns the path to the saved file.
    """
    overall = ssim_passed

    report = {
        "overall_result":          "PASS" if overall else "FAIL",
        "ssim_score":              round(ssim_score, 4),
        "ssim_threshold":          params['ssim_threshold'],
        "ssim_passed":             ssim_passed,
        "degraded_regions_count":  len(boxes),
        "degraded_region_boxes":   [list(b) for b in boxes]
    }

    path = os.path.join(output_dir, "regression_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    return path


def save_figure(output_dir: str,
                img_baseline: np.ndarray,
                img_optimized: np.ndarray,
                composite_bgr: np.ndarray,
                ssim_score: float,
                ssim_passed: bool,
                boxes: list,
                params: dict) -> str:
    """
    Generate and save the 4-panel analysis figure.

    Panels:
      [1] Original (baseline) image
      [2] Optimized image
      [3] Heatmap overlay with bounding boxes
      [4] Metric summary text
    """

    # Convert composite BGR (uint8) → RGB float for matplotlib display
    import cv2
    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    #  Build figure with GridSpec for clean layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 8), facecolor='#1a1a2e')
    gs  = GridSpec(2, 4, figure=fig,
                   hspace=0.35, wspace=0.25,
                   left=0.04, right=0.97,
                   top=0.90, bottom=0.06)

    title_color  = '#e0e0e0'
    panel_bg     = '#16213e'
    pass_color   = '#00c853'
    fail_color   = '#ff1744'
    metric_color = '#90caf9'

    verdict       = "PASS" if ssim_passed else "FAIL"
    verdict_color = pass_color if ssim_passed else fail_color

    fig.suptitle("Visual Regression Analyzer — SSIM Analysis Report",
                 color=title_color, fontsize=15, fontweight='bold', y=0.97)

    # ------------------------------------------------------------------
    #  Panel 1: Original (baseline) image
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img_baseline)
    ax1.set_title("Original Image\n(Baseline)", color=title_color,
                  fontsize=10, pad=6)
    ax1.axis('off')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444466')

    # ------------------------------------------------------------------
    #  Panel 2: Optimized image
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(img_optimized)
    ax2.set_title("Optimized Image", color=title_color,
                  fontsize=10, pad=6)
    ax2.axis('off')

    # ------------------------------------------------------------------
    #  Panel 3: Heatmap overlay + bounding boxes
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.imshow(composite_rgb)
    ax3.set_title(f"SSIM Heatmap Overlay\n({len(boxes)} degraded region(s) detected)",
                  color=title_color, fontsize=10, pad=6)
    ax3.axis('off')

    # Redraw bounding boxes as matplotlib patches for cleaner rendering
    for (x, y, w, h) in boxes:
        rect = mpatches.Rectangle((x, y), w, h,
                                   linewidth=2,
                                   edgecolor='red',
                                   facecolor='none')
        ax3.add_patch(rect)

    # ------------------------------------------------------------------
    #  Panel 4: Metric summary
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[:, 3])
    ax4.set_facecolor(panel_bg)
    ax4.axis('off')

    # Verdict banner
    ax4.text(0.5, 0.92, "OVERALL VERDICT",
             transform=ax4.transAxes,
             ha='center', va='top',
             color=title_color, fontsize=10, fontweight='bold')

    ax4.text(0.5, 0.82, verdict,
             transform=ax4.transAxes,
             ha='center', va='top',
             color=verdict_color, fontsize=28, fontweight='bold',
             fontfamily='monospace')

    # Divider line
    ax4.axhline(y=0.73, color='#444466', linewidth=0.8,
                xmin=0.05, xmax=0.95)

    # SSIM metric row
    ssim_status_color = pass_color if ssim_passed else fail_color
    ssim_status       = "PASS ✓" if ssim_passed else "FAIL ✗"

    ax4.text(0.08, 0.67, "SSIM Score",
             transform=ax4.transAxes,
             color=metric_color, fontsize=10, fontfamily='monospace')

    ax4.text(0.08, 0.59, f"{ssim_score:.4f}",
             transform=ax4.transAxes,
             color=title_color, fontsize=16, fontweight='bold',
             fontfamily='monospace')

    ax4.text(0.08, 0.51, f"Threshold : {params['ssim_threshold']}",
             transform=ax4.transAxes,
             color='#888888', fontsize=9, fontfamily='monospace')

    ax4.text(0.08, 0.43, ssim_status,
             transform=ax4.transAxes,
             color=ssim_status_color, fontsize=11, fontweight='bold',
             fontfamily='monospace')

    # Divider
    ax4.axhline(y=0.37, color='#444466', linewidth=0.8,
                xmin=0.05, xmax=0.95)

    # Region count
    ax4.text(0.08, 0.31, "Degraded Regions",
             transform=ax4.transAxes,
             color=metric_color, fontsize=10, fontfamily='monospace')

    ax4.text(0.08, 0.22, str(len(boxes)),
             transform=ax4.transAxes,
             color=title_color, fontsize=20, fontweight='bold',
             fontfamily='monospace')

    # Divider
    ax4.axhline(y=0.15, color='#444466', linewidth=0.8,
                xmin=0.05, xmax=0.95)

    ax4.text(0.5, 0.07,
             "Visual Regression Analyzer v1.0\nSSIM Phase Report",
             transform=ax4.transAxes,
             ha='center', color='#555577', fontsize=7,
             fontfamily='monospace')

    # ------------------------------------------------------------------
    #  Save figure
    # ------------------------------------------------------------------
    path = os.path.join(output_dir, "analysis_figure.png")
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return path
