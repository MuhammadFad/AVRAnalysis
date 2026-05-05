# ================================================================================
# visual_reporter.py — Visual Report Generation (PNG Images)
# ================================================================================
#
# PURPOSE:
# This module generates publication-quality visual reports covering all three
# analysis passes (SSIM, edge, color). Outputs include:
#   1. Heatmap composite: SSIM heatmap with color-coded bounding boxes and legend
#   2. Analysis figure:   Multi-panel report — images, SSIM components, edge maps,
#                         and a summary bar with all three pass scores
#
# BOUNDING BOX COLOR CODING:
# Each detected region is drawn with a color reflecting which filters flagged it:
#   🟡 Yellow (255, 220, 0)  — SSIM only          (perceptual diff, edges/color OK)
#   🟠 Orange (255, 140, 0)  — SSIM + Edge        (structural detail also lost)
#   🔵 Blue   (30,  144, 255) — SSIM + Color       (color profile also shifted)
#   🔴 Red    (220, 30,  30)  — All three          (fully degraded region)
#
# ROLE IN PIPELINE:
# Receives the fully assembled namespaced results dict from main.py and writes:
#   output/heatmap_composite.png  — heatmap with color-coded boxes and legend
#   output/analysis_figure.png    — full multi-panel pipeline report
#   output/ssim/                  — SSIM component maps (L, C, S)
#   output/edge/                  — Edge intermediates (Sobel X/Y, NMS, edge maps)
#
# ================================================================================

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ================================================================================
# Box color scheme — shared between heatmap and figure
# ================================================================================

# Maps a frozenset of filter tags → RGB tuple (uint8 scale)
def _box_color(flags: frozenset) -> tuple:
    """
    Return the RGB draw color for a bounding box given its filter flag set.

    Color encodes which combination of passes flagged this region:
        SSIM only            → Yellow  (mildest — only perceptual score failed)
        SSIM + Edge          → Orange  (structure also broken)
        SSIM + Color         → Blue    (color profile also shifted)
        SSIM + Edge + Color  → Red     (all dimensions degraded)

    SSIM is always present in flags since boxes are SSIM-derived.
    """
    has_edge  = 'edge'  in flags
    has_color = 'color' in flags

    if has_edge and has_color:
        return (220, 30, 30)    # Red   — all three
    if has_edge:
        return (255, 140, 0)    # Orange — SSIM + Edge
    if has_color:
        return (30, 144, 255)   # Blue   — SSIM + Color
    return (255, 220, 0)        # Yellow — SSIM only


# Legend entries for the heatmap and figure
_LEGEND_ENTRIES = [
    ((255, 220, 0),  "SSIM only"),
    ((255, 140, 0),  "SSIM + Edge"),
    ((30, 144, 255), "SSIM + Color"),
    ((220, 30,  30), "SSIM + Edge + Color"),
]


# ================================================================================
# Internal normalization helpers
# ================================================================================

def _normalize_he(arr: np.ndarray) -> np.ndarray:
    """
    Enhanced normalization using CLAHE for better contrast and visibility.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) applies histogram
    equalization tile-by-tile with a clip limit, enhancing local contrast for
    visualization without over-amplifying noise.

    INPUTS:
        arr: H × W float64 array

    RETURNS:
        H × W uint8 array [0, 255]
    """
    lo, hi = np.min(arr), np.max(arr)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    rescaled = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(rescaled)


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float64 [0,1] image to uint8 [0,255]."""
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


# ================================================================================
# Intermediate savers — called from main.py when save_intermediates=True
# ================================================================================

def save_ssim_intermediates(ssim_dir: str, ssim_result: dict) -> None:
    """
    Save SSIM component maps (luminance, contrast, structure) as PNG files.

    Each component highlights a different reason for visual degradation,
    making it easier to diagnose which perceptual dimension failed.

    INPUTS:
        ssim_dir:    Path to the SSIM output subdirectory (output/ssim/)
        ssim_result: Dictionary returned by ssim_pass.run()
                     Must contain 'l_map', 'c_map', 's_map'.
    """
    for arr, name in [(ssim_result.get("l_map"), "luminance"),
                      (ssim_result.get("c_map"), "contrast"),
                      (ssim_result.get("s_map"), "structure")]:
        if arr is not None:
            cv2.imwrite(os.path.join(ssim_dir, f"{name}_component.png"),
                        _normalize_he(arr))


def save_edge_intermediates(edge_dir: str, edge_result: dict) -> None:
    """
    Save Canny edge pass intermediate images for diagnostic inspection.

    Saved files:
      - sobel_x.png:        Horizontal gradient magnitude (absolute value)
      - sobel_y.png:        Vertical gradient magnitude (absolute value)
      - nms_map.png:        Gradient after non-maximum suppression
      - edge_baseline.png:  Final Canny edge map for the baseline image
      - edge_optimized.png: Final Canny edge map for the optimized image

    INPUTS:
        edge_dir:    Path to the edge output subdirectory (output/edge/)
        edge_result: Dictionary returned by edge_pass.run()
    """
    for key, filename in [('sobel_x', 'sobel_x.png'), ('sobel_y', 'sobel_y.png')]:
        arr = edge_result.get(key)
        if arr is not None:
            cv2.imwrite(os.path.join(edge_dir, filename),
                        _normalize_he(np.abs(arr)))

    nms = edge_result.get("nms_map")
    if nms is not None:
        cv2.imwrite(os.path.join(edge_dir, "nms_map.png"), _normalize_he(nms))

    for key, filename in [('edge_baseline',  'edge_baseline.png'),
                           ('edge_optimized', 'edge_optimized.png')]:
        arr = edge_result.get(key)
        if arr is not None:
            cv2.imwrite(os.path.join(edge_dir, filename), arr * 255)


# ================================================================================
# Legend drawing helper
# ================================================================================

def _draw_legend(img_uint8: np.ndarray) -> np.ndarray:
    """
    Burn a color-coded filter legend into the bottom-left corner of an image.

    The legend maps box colors to their filter combination meaning, making
    the heatmap self-explanatory without needing a separate figure.

    INPUTS:
        img_uint8: H × W × 3 uint8 image to annotate in-place

    RETURNS:
        The same array with the legend drawn on it
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness  = 2
    pad        = 16
    swatch_w   = 26
    swatch_h   = 20
    line_h     = 32

    # Measure the widest label to size the background panel.
    max_text_w = max(
        cv2.getTextSize(label, font, font_scale, thickness)[0][0]
        for _, label in _LEGEND_ENTRIES
    )

    panel_w = swatch_w + pad + max_text_w + pad * 2
    panel_h = len(_LEGEND_ENTRIES) * line_h + pad * 2

    H, W = img_uint8.shape[:2]
    x0   = pad
    y0   = H - panel_h - pad

    # Semi-transparent dark background panel.
    overlay = img_uint8.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img_uint8, 0.35, 0, img_uint8)

    for i, (color_rgb, label) in enumerate(_LEGEND_ENTRIES):
        row_y  = y0 + pad + i * line_h
        swatch_x1 = x0 + pad
        swatch_y1 = row_y
        swatch_x2 = swatch_x1 + swatch_w
        swatch_y2 = swatch_y1 + swatch_h

        # Color swatch — cv2 uses BGR
        bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(img_uint8, (swatch_x1, swatch_y1),
                      (swatch_x2, swatch_y2), bgr, -1)

        text_x = swatch_x2 + pad // 2
        text_y = swatch_y1 + swatch_h - 2
        cv2.putText(img_uint8, label, (text_x, text_y),
                    font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA)

    return img_uint8


# ================================================================================
# Main output writers
# ================================================================================

def save_heatmap(output_dir: str, composite: np.ndarray, tagged_boxes: list) -> str:
    """
    Save the SSIM heatmap composite with color-coded bounding boxes and legend.

    Each box is drawn in a color reflecting which combination of filters (SSIM,
    edge, color) flagged that region as degraded. A legend is burned into the
    image so it is self-explanatory.

    INPUTS:
        output_dir:   Root output directory
        composite:    H × W × 3 float64 heatmap image
        tagged_boxes: List of (x, y, w, h, flags) tuples from aggregator

    RETURNS:
        str: File path where the PNG was written
    """
    img = _float_to_uint8(composite)

    for (x, y, w, h, flags) in tagged_boxes:
        color_rgb = _box_color(flags)
        # cv2 expects BGR
        bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, thickness=3)

    img = _draw_legend(img)

    path = os.path.join(output_dir, "heatmap_composite.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def save_figure(output_dir:    str,
                img_baseline:  np.ndarray,
                img_optimized: np.ndarray,
                composite:     np.ndarray,
                results:       dict) -> str:
    """
    Generate and save a comprehensive multi-panel analysis figure.

    Layout:
      Row 0:    Baseline | Optimized | SSIM heatmap (with color-coded boxes)
      Row 1:    Luminance | Contrast | Structure  (SSIM components, if available)
      Row 2:    Edge baseline | Edge optimized | Lost edges diff map
      Last row: Summary bar — all three pass scores, verdicts, and legend

    INPUTS:
        output_dir:    Root output directory
        img_baseline:  H × W × 3 float64 baseline image
        img_optimized: H × W × 3 float64 optimized image
        composite:     H × W × 3 float64 SSIM heatmap composite
        results:       Namespaced results dict from main.py:
                           results['ssim'], results['edge'], results['color'],
                           results['params']

    RETURNS:
        str: File path where the PNG was written
    """
    ssim_r  = results["ssim"]
    edge_r  = results["edge"]
    color_r = results["color"]
    params  = results["params"]

    tagged_boxes = ssim_r.get("boxes", [])

    # ---- Unpack optional maps ----
    l_map = ssim_r.get("l_map")
    c_map = ssim_r.get("c_map")
    s_map = ssim_r.get("s_map")
    has_ssim_components = all(m is not None for m in [l_map, c_map, s_map])

    edge_baseline  = edge_r.get("edge_baseline")
    edge_optimized = edge_r.get("edge_optimized")
    has_edge_maps  = edge_baseline is not None and edge_optimized is not None

    # ---- Layout ----
    # Image rows stack vertically in columns 0-2 (left 3/4).
    # The summary panel occupies column 3 (right 1/4), spanning all image rows.
    # A thin footer row sits below everything, spanning all 4 columns.
    n_img_rows = 1 + int(has_ssim_components) + int(has_edge_maps)
    n_rows     = n_img_rows + 1   # +1 for footer

    # ---- Styling ----
    metric_color = '#90caf9'
    pass_color   = '#00c853'
    fail_color   = '#ff1744'

    def vc(passed): return pass_color if passed else fail_color
    def vs(passed): return "PASS ✓" if passed else "FAIL ✗"

    all_passed = ssim_r["passed"] and edge_r["passed"] and color_r["passed"]

    # Figure is wider than tall — 3 image columns + 1 sidebar column.
    # Height scales with number of image rows; width is fixed at 24 inches.
    fig_height = max(8, 4 * n_img_rows)
    fig = plt.figure(figsize=(24, fig_height), facecolor='#1a1a2e')

    # 4-column GridSpec: columns 0-2 hold image panels, column 3 holds summary.
    # width_ratios=[1,1,1,1.05] gives the right panel just a touch more breathing room.
    # The footer row has a fixed small height_ratio.
    height_ratios = [1] * n_img_rows + [0.08]
    gs = GridSpec(
        n_rows, 4, figure=fig,
        width_ratios=[1, 1, 1, 1.05],
        height_ratios=height_ratios,
        hspace=0.35, wspace=0.18,
        left=0.03, right=0.98, top=0.93, bottom=0.04
    )

    fig.suptitle(
        "Visual Regression Analyzer — Full Pipeline Report",
        color='white', fontsize=15, fontweight='bold', y=0.97
    )

    current_row = 0

    # ========================================================================
    # ROW 0: Baseline | Optimized | SSIM heatmap  (columns 0-2)
    # ========================================================================
    heatmap_uint8 = _float_to_uint8(composite)
    for (x, y, w, h, flags) in tagged_boxes:
        color_rgb = _box_color(flags)
        bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(heatmap_uint8, (x, y), (x + w, y + h), bgr, thickness=3)

    ax = fig.add_subplot(gs[current_row, 0])
    ax.imshow(img_baseline)
    ax.set_title("Baseline Input", color='white', fontsize=9, pad=5)
    ax.axis('off')

    ax = fig.add_subplot(gs[current_row, 1])
    ax.imshow(img_optimized)
    ax.set_title("Optimized Input", color='white', fontsize=9, pad=5)
    ax.axis('off')

    ax = fig.add_subplot(gs[current_row, 2])
    ax.imshow(heatmap_uint8)
    ax.set_title(f"SSIM Degradation Heatmap  "
                 f"({len(tagged_boxes)} region(s))",
                 color='white', fontsize=9, pad=5)
    ax.axis('off')
    # Matplotlib legend for the box color scheme, pinned to this panel.
    legend_patches = [
        mpatches.Patch(color=tuple(c / 255 for c in rgb), label=label)
        for rgb, label in _LEGEND_ENTRIES
    ]
    ax.legend(handles=legend_patches, loc='lower left',
              fontsize=7, facecolor='#16213e', edgecolor='#444466',
              labelcolor='white', framealpha=0.85)

    current_row += 1

    # ========================================================================
    # ROW 1: SSIM component maps — Luminance | Contrast | Structure
    # ========================================================================
    if has_ssim_components:
        for col, (arr, label, sublabel) in enumerate([
            (l_map, "Luminance",  "local brightness similarity"),
            (c_map, "Contrast",   "local texture variation similarity"),
            (s_map, "Structure",  "local edge pattern correlation"),
        ]):
            ax = fig.add_subplot(gs[current_row, col])
            ax.imshow(_normalize_he(arr), cmap='hot')
            ax.set_title(f"{label} Component\n{sublabel}",
                         color='white', fontsize=9, pad=5)
            ax.axis('off')
        current_row += 1

    # ========================================================================
    # ROW 2: Edge maps — Baseline edges | Optimized edges | Lost edges
    # ========================================================================
    if has_edge_maps:
        ax = fig.add_subplot(gs[current_row, 0])
        ax.imshow(edge_baseline, cmap='gray')
        ax.set_title("Canny Edge Map — Baseline",
                     color='white', fontsize=9, pad=5)
        ax.axis('off')

        ax = fig.add_subplot(gs[current_row, 1])
        ax.imshow(edge_optimized, cmap='gray')
        ax.set_title(f" Edge Map — Optimized  "
                     f"(match {edge_r['score']:.2%})",
                     color='white', fontsize=9, pad=5)
        ax.axis('off')

        # Pixels present in baseline but absent in optimized — structural detail lost.
        lost_edges = edge_baseline & ~edge_optimized
        ax = fig.add_subplot(gs[current_row, 2])
        ax.imshow(lost_edges, cmap='hot')
        ax.set_title("Lost Edges\nbaseline edges absent in optimized",
                     color='white', fontsize=9, pad=5)
        ax.axis('off')
        current_row += 1

    # ========================================================================
    # SUMMARY PANEL: column 3, spans all image rows
    # ========================================================================
    ax_s = fig.add_subplot(gs[0:n_img_rows, 3])
    ax_s.set_facecolor('#16213e')
    ax_s.axis('off')

    # The summary panel uses a simple top-to-bottom text stack.
    # y positions are in axes coordinates [0,1], stepping downward.
    y = 0.97

    def label(text, dy=0.045):
        nonlocal y
        ax_s.text(0.08, y, text, transform=ax_s.transAxes,
                  color=metric_color, fontsize=8.5, fontfamily='monospace', va='top')
        y -= dy

    def value(text, color='white', size=19, dy=0.10):
        nonlocal y
        ax_s.text(0.08, y, text, transform=ax_s.transAxes,
                  color=color, fontsize=size, fontweight='bold',
                  fontfamily='monospace', va='top')
        y -= dy

    def subtext(text, dy=0.040):
        nonlocal y
        ax_s.text(0.08, y, text, transform=ax_s.transAxes,
                  color='#888888', fontsize=8, fontfamily='monospace', va='top')
        y -= dy

    def spacer(dy=0.025):
        nonlocal y
        y -= dy

    def divider(dy=0.030):
        nonlocal y
        ax_s.axhline(y=y + 0.01, xmin=0.05, xmax=0.95,
                     color='#333355', linewidth=0.8)
        y -= dy

    # ---- Overall verdict ----
    label("OVERALL VERDICT", dy=0.04)
    value("PASS" if all_passed else "FAIL", color=vc(all_passed), size=28, dy=0.12)
    divider()

    # ---- SSIM ----
    label("SSIM Score", dy=0.04)
    value(f"{ssim_r['score']:.4f}", dy=0.09)
    subtext(f"threshold  ≥ {params['ssim_threshold']}", dy=0.035)
    value(vs(ssim_r['passed']), color=vc(ssim_r['passed']), size=11, dy=0.06)
    divider()

    # ---- Edge ----
    label("Edge Match  (Canny)", dy=0.04)
    value(f"{edge_r['score']:.4f}", dy=0.09)
    subtext(f"threshold  ≥ {params['canny_edge_match_thresh']}", dy=0.035)
    value(vs(edge_r['passed']), color=vc(edge_r['passed']), size=11, dy=0.06)
    divider()

    # ---- Color ----
    label("Color Distance  (Bhattacharyya)", dy=0.04)
    value(f"{color_r['score']:.4f}", dy=0.09)
    subtext(f"H={color_r['distances']['H']:.3f}  "
            f"S={color_r['distances']['S']:.3f}  "
            f"V={color_r['distances']['V']:.3f}", dy=0.035)
    subtext(f"threshold  ≤ {params['color_distance_thresh']}", dy=0.035)
    value(vs(color_r['passed']), color=vc(color_r['passed']), size=11, dy=0.06)
    divider()

    # ========================================================================
    # FOOTER ROW: spans all 4 columns
    # ========================================================================
    ax_f = fig.add_subplot(gs[n_img_rows, :])
    ax_f.set_facecolor('#12122a')
    ax_f.axis('off')
    ax_f.text(0.5, 0.5,
              "Visual Regression Analyzer v1.0  |  SSIM · Canny Edge · Bhattacharyya Color",
              transform=ax_f.transAxes, ha='center', va='center',
              color='#555577', fontsize=8, fontfamily='monospace')

    path = os.path.join(output_dir, "analysis_figure.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
