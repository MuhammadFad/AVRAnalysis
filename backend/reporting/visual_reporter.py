# ================================================================================
# visual_reporter.py — Visual Report Generation (PNG Images)
# ================================================================================
#
# PURPOSE:
# This module generates publication-quality visual reports covering all three
# analysis passes (SSIM, edge, color). Outputs include:
#
#   1. heatmap_composite.png
#      SSIM heatmap with color-coded bounding boxes, cause labels burned onto
#      each box, and a legend. Self-explanatory at a glance — no separate figure
#      needed to understand what each region means.
#
#   2. analysis_figure.png
#      Multi-panel report with four columns:
#        Col 0-2 (image panels, stacked rows):
#          Row 0: Baseline | Optimized | SSIM heatmap (with annotated boxes)
#          Row 1: Lost edges diff map | SSIM structure component | SSIM contrast component
#          Row 2: Color histogram overlay (spanning all 3 cols)  [if available]
#        Col 3 (sidebar, spans all rows):
#          Overall verdict + per-pass score cards + top-N degraded region breakdown
#          with hypothesis in plain English
#
# DESIGN CHOICES vs the original:
#   - Removed: separate luminance component panel (least actionable of the three)
#   - Removed: side-by-side edge maps (replaced by lost-edges diff which is more
#     informative — you can immediately see *what* was lost, not just that edges exist)
#   - Added: cause label text drawn on/near each bounding box on the heatmap
#   - Added: per-region breakdown in sidebar — top N regions, worst-first, with
#     their cause hypothesis, confidence badge, and local scores
#   - Added: color histogram panel in analysis figure (was only in separate file)
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
#   output/heatmap_composite.png  — heatmap with cause-labelled boxes and legend
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

def _box_color(passes_failed: list) -> tuple:
    """
    Return the RGB draw color for a bounding box given its failed-pass list.

    Color encodes which combination of passes flagged this region:
        SSIM only            → Yellow  (mildest — only perceptual score failed)
        SSIM + Edge          → Orange  (structure also broken)
        SSIM + Color         → Blue    (color profile also shifted)
        SSIM + Edge + Color  → Red     (all dimensions degraded)

    SSIM is always present since boxes are SSIM-derived.
    """
    failed   = set(passes_failed)
    has_edge  = 'edge'  in failed
    has_color = 'color' in failed

    if has_edge and has_color:
        return (220, 30,  30)    # Red    — all three
    if has_edge:
        return (255, 140,  0)    # Orange — SSIM + Edge
    if has_color:
        return (30,  144, 255)   # Blue   — SSIM + Color
    return (255, 220,   0)       # Yellow — SSIM only


# Legend entries for the heatmap and figure (RGB, label)
_LEGEND_ENTRIES = [
    ((255, 220,   0), "SSIM only"),
    ((255, 140,   0), "SSIM + Edge"),
    ((30,  144, 255), "SSIM + Color"),
    ((220,  30,  30), "SSIM + Edge + Color"),
]

# Confidence badge colors for the sidebar region breakdown (matplotlib color strings)
_CONFIDENCE_COLORS = {
    "clean":      "#00c853",
    "low":        "#90caf9",
    "medium":     "#ffe082",
    "high":       "#ff9800",
    "very_high":  "#ff5722",
    "definitive": "#b71c1c",
}


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
# Heatmap annotation helpers
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
    font_scale = 0.65
    thickness  = 2
    pad        = 14
    swatch_w   = 22
    swatch_h   = 18
    line_h     = 28

    max_text_w = max(
        cv2.getTextSize(label, font, font_scale, thickness)[0][0]
        for _, label in _LEGEND_ENTRIES
    )

    panel_w = swatch_w + pad + max_text_w + pad * 2
    panel_h = len(_LEGEND_ENTRIES) * line_h + pad * 2

    H, W = img_uint8.shape[:2]
    x0   = pad
    y0   = H - panel_h - pad

    # Semi-transparent dark background panel
    overlay = img_uint8.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img_uint8, 0.35, 0, img_uint8)

    for i, (color_rgb, label) in enumerate(_LEGEND_ENTRIES):
        row_y     = y0 + pad + i * line_h
        swatch_x1 = x0 + pad
        swatch_y1 = row_y
        swatch_x2 = swatch_x1 + swatch_w
        swatch_y2 = swatch_y1 + swatch_h

        bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(img_uint8, (swatch_x1, swatch_y1),
                      (swatch_x2, swatch_y2), bgr, -1)

        text_x = swatch_x2 + pad // 2
        text_y = swatch_y1 + swatch_h - 2
        cv2.putText(img_uint8, label, (text_x, text_y),
                    font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA)

    return img_uint8


def _draw_boxes_with_labels(img_uint8:       np.ndarray,
                             combined_regions: list,
                             box_thickness:   int = 3) -> np.ndarray:
    """
    Draw color-coded bounding boxes and cause-tag labels onto the heatmap.

    Each box is drawn in the color corresponding to its pass combination.
    A short cause tag (e.g. "blocking artifact", "texture loss") is burned
    above or below the box so the image is interpretable without a separate
    report.

    Label placement logic:
      - Preferred: above the box (offset upward by label height + padding)
      - Fallback:  inside the box top edge (if box is too close to image top)

    INPUTS:
        img_uint8:        H × W × 3 uint8 image
        combined_regions: Enriched region dicts from combinator (must have
                          'bbox', 'passes_failed', 'cause_tag')
        box_thickness:    Rectangle stroke width in pixels

    RETURNS:
        The same array with boxes and labels drawn on it
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 2
    pad        = 5
    H, W       = img_uint8.shape[:2]

    for region in combined_regions:
        x, y, w, h   = region["bbox"]
        color_rgb     = _box_color(region["passes_failed"])
        bgr           = (color_rgb[2], color_rgb[1], color_rgb[0])
        tag           = region.get("cause_tag", "")

        # Draw the bounding box
        cv2.rectangle(img_uint8, (x, y), (x + w, y + h), bgr, box_thickness)

        if not tag:
            continue

        # Measure the label text so we can position and back-fill it cleanly
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
        label_h = th + baseline + pad * 2

        # Place label above box if there is room, otherwise place inside top edge
        if y - label_h >= 0:
            label_y0 = y - label_h
            text_y   = y - pad - baseline
        else:
            label_y0 = y
            text_y   = y + th + pad

        label_x0 = max(0, x)
        label_x1 = min(W, x + tw + pad * 2)
        label_y1 = label_y0 + label_h

        # Semi-transparent filled background behind label text
        overlay = img_uint8.copy()
        cv2.rectangle(overlay, (label_x0, label_y0), (label_x1, label_y1), bgr, -1)
        cv2.addWeighted(overlay, 0.70, img_uint8, 0.30, 0, img_uint8)

        # White label text
        cv2.putText(img_uint8, tag, (label_x0 + pad, text_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_uint8


# ================================================================================
# Main output writers
# ================================================================================

def save_heatmap(output_dir:      str,
                 composite:       np.ndarray,
                 combined_regions: list) -> str:
    """
    Save the SSIM heatmap composite with color-coded bounding boxes, cause labels,
    and a filter legend.

    Each box is drawn with a color reflecting which combination of passes (SSIM,
    edge, color) flagged that region. A short cause tag is burned above each box
    for at-a-glance interpretation. A legend is burned into the bottom-left corner.

    INPUTS:
        output_dir:       Root output directory
        composite:        H × W × 3 float64 heatmap image
        combined_regions: Enriched region dicts from combinator

    RETURNS:
        str: File path where the PNG was written
    """
    img = _float_to_uint8(composite)
    img = _draw_boxes_with_labels(img, combined_regions)
    img = _draw_legend(img)

    path = os.path.join(output_dir, "heatmap_composite.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def save_figure(output_dir:       str,
                img_baseline:     np.ndarray,
                img_optimized:    np.ndarray,
                composite:        np.ndarray,
                results:          dict) -> str:
    """
    Generate and save a comprehensive multi-panel analysis figure.

    LAYOUT:
      Col 0-2  |  Col 3 (sidebar)
      ---------+--------------------------
      Row 0:   Baseline | Optimized | SSIM heatmap (annotated)   │ Overall verdict
      Row 1:   Lost edges | SSIM structure | SSIM contrast        │ Pass score cards
      Row 2:   Color histograms (H, S, V) spanning cols 0-2      │ Top-N regions
      Footer:  Full-width footer bar

    Removed vs original:
      - Luminance component (least actionable SSIM sub-map)
      - Side-by-side raw edge maps (replaced by lost-edges diff — shows *what* was lost)
    Added vs original:
      - Cause labels on heatmap boxes
      - Color histogram panel integrated into main figure
      - Per-region breakdown in sidebar (top N, worst-first)

    INPUTS:
        output_dir:    Root output directory
        img_baseline:  H × W × 3 float64 baseline image
        img_optimized: H × W × 3 float64 optimized image
        composite:     H × W × 3 float64 SSIM heatmap composite
        results:       Namespaced results dict from main.py:
                           results['ssim']   → ssim pass result + boxes (enriched)
                           results['edge']   → edge pass result
                           results['color']  → color pass result
                           results['params'] → flat config dict

    RETURNS:
        str: File path where the PNG was written
    """
    ssim_r  = results["ssim"]
    edge_r  = results["edge"]
    color_r = results["color"]
    params  = results["params"]

    combined_regions = ssim_r.get("boxes", [])

    # ---- Unpack optional maps ----
    c_map = ssim_r.get("c_map")
    s_map = ssim_r.get("s_map")
    has_ssim_detail = c_map is not None and s_map is not None

    edge_baseline  = edge_r.get("edge_baseline")
    edge_optimized = edge_r.get("edge_optimized")
    has_edge_maps  = edge_baseline is not None and edge_optimized is not None

    has_histograms = bool(color_r.get("histograms"))

    # ---- Determine row layout ----
    # Row 0 always present (baseline / optimized / heatmap)
    # Row 1: SSIM detail + lost edges  (if edge maps + SSIM components available)
    # Row 2: Color histograms          (if histogram data available)
    n_img_rows = 1 + int(has_ssim_detail and has_edge_maps) + int(has_histograms)
    n_rows     = n_img_rows + 1    # +1 for footer

    # ---- Styling ----
    bg_dark      = '#1a1a2e'
    bg_panel     = '#16213e'
    metric_color = '#90caf9'
    pass_color   = '#00c853'
    fail_color   = '#ff1744'

    def vc(passed): return pass_color if passed else fail_color
    def vs(passed): return "PASS ✓" if passed else "FAIL ✗"

    all_passed = ssim_r["passed"] and edge_r["passed"] and color_r["passed"]

    fig_height = max(10, 4 * n_img_rows)
    fig = plt.figure(figsize=(26, fig_height), facecolor=bg_dark)

    height_ratios = [1] * n_img_rows + [0.07]
    gs = GridSpec(
        n_rows, 4, figure=fig,
        width_ratios=[1, 1, 1, 1.15],
        height_ratios=height_ratios,
        hspace=0.38, wspace=0.16,
        left=0.02, right=0.99, top=0.93, bottom=0.04,
    )

    fig.suptitle(
        "Visual Regression Analyzer — Full Pipeline Report",
        color='white', fontsize=15, fontweight='bold', y=0.97,
    )

    current_row = 0

    # ========================================================================
    # ROW 0: Baseline | Optimized | SSIM heatmap (annotated boxes + labels)
    # ========================================================================
    # Build annotated heatmap — same as heatmap_composite but without the
    # burned-in legend (the matplotlib legend handles that here).
    heatmap_uint8 = _float_to_uint8(composite)
    heatmap_uint8 = _draw_boxes_with_labels(heatmap_uint8, combined_regions)

    ax = fig.add_subplot(gs[current_row, 0])
    ax.imshow(img_baseline)
    ax.set_title("Baseline (reference)", color='white', fontsize=9, pad=5)
    ax.axis('off')

    ax = fig.add_subplot(gs[current_row, 1])
    ax.imshow(img_optimized)
    ax.set_title("Optimized (test)", color='white', fontsize=9, pad=5)
    ax.axis('off')

    ax = fig.add_subplot(gs[current_row, 2])
    ax.imshow(heatmap_uint8)
    ax.set_title(
        f"SSIM Degradation Heatmap  ({len(combined_regions)} region(s) detected)",
        color='white', fontsize=9, pad=5,
    )
    ax.axis('off')
    # Matplotlib legend for box color scheme (cleaner than burned-in legend here)
    legend_patches = [
        mpatches.Patch(color=tuple(c / 255 for c in rgb), label=label)
        for rgb, label in _LEGEND_ENTRIES
    ]
    ax.legend(handles=legend_patches, loc='lower left',
              fontsize=7, facecolor=bg_panel, edgecolor='#444466',
              labelcolor='white', framealpha=0.88)

    current_row += 1

    # ========================================================================
    # ROW 1: Lost edges diff | SSIM structure | SSIM contrast
    # ========================================================================
    # Showing *lost* edges (baseline ∩ ¬optimized) is more informative than
    # side-by-side edge maps: you immediately see what structural detail was
    # removed rather than having to mentally diff two full maps.
    if has_ssim_detail and has_edge_maps:
        lost_edges = edge_baseline & ~edge_optimized

        ax = fig.add_subplot(gs[current_row, 0])
        ax.imshow(lost_edges, cmap='hot')
        ax.set_title(
            f"Lost Edges  (match {edge_r['score']:.1%})\n"
            "baseline edges absent in optimized",
            color='white', fontsize=9, pad=5,
        )
        ax.axis('off')

        ax = fig.add_subplot(gs[current_row, 1])
        ax.imshow(_normalize_he(s_map), cmap='hot')
        ax.set_title(
            "SSIM — Structure Component\nlocal edge-pattern correlation",
            color='white', fontsize=9, pad=5,
        )
        ax.axis('off')

        ax = fig.add_subplot(gs[current_row, 2])
        ax.imshow(_normalize_he(c_map), cmap='hot')
        ax.set_title(
            "SSIM — Contrast Component\nlocal texture-variation similarity",
            color='white', fontsize=9, pad=5,
        )
        ax.axis('off')

        current_row += 1

    # ========================================================================
    # ROW 2: Color histograms (H, S, V) — spans all 3 image columns
    # ========================================================================
    # Integrating the histogram into the main figure avoids the reader having
    # to open a separate file to see the color distribution comparison.
    if has_histograms:
        histograms = color_r["histograms"]
        distances  = color_r["distances"]

        channel_meta = {
            'H': {'label': 'Hue',        'color_b': '#f48fb1', 'color_o': '#f06292', 'x_max': 180},
            'S': {'label': 'Saturation', 'color_b': '#80cbc4', 'color_o': '#26a69a', 'x_max': 256},
            'V': {'label': 'Value',      'color_b': '#ffe082', 'color_o': '#ffca28', 'x_max': 256},
        }

        for col, (name, (p_base, p_opt)) in enumerate(histograms.items()):
            ax = fig.add_subplot(gs[current_row, col])
            ax.set_facecolor(bg_panel)

            meta   = channel_meta[name]
            n_bins = len(p_base)
            x      = np.linspace(0, meta['x_max'], n_bins)

            ax.fill_between(x, p_base, alpha=0.45, color=meta['color_b'], label='Baseline')
            ax.plot(x, p_base, color=meta['color_b'], linewidth=1.5)
            ax.fill_between(x, p_opt,  alpha=0.45, color=meta['color_o'], label='Optimized')
            ax.plot(x, p_opt,  color=meta['color_o'], linewidth=1.5)

            dist_ok = distances[name] <= params['color_distance_thresh']
            ax.set_title(
                f"{meta['label']} — Bhattacharyya: {distances[name]:.4f}  "
                f"{'✓' if dist_ok else '✗'}",
                color='white', fontsize=9, pad=5,
            )
            ax.set_xlabel("Pixel Value",  color='#aaaaaa', fontsize=8)
            ax.set_ylabel("Probability",  color='#aaaaaa', fontsize=8)
            ax.tick_params(colors='#888888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            ax.legend(facecolor=bg_dark, edgecolor='#333355',
                      labelcolor='white', fontsize=8)

        current_row += 1

    # ========================================================================
    # SIDEBAR (col 3, spans all image rows): verdict + scores + region breakdown
    # ========================================================================
    ax_s = fig.add_subplot(gs[0:n_img_rows, 3])
    ax_s.set_facecolor(bg_panel)
    ax_s.axis('off')

    y = 0.98

    def _label(text, dy=0.042):
        nonlocal y
        ax_s.text(0.07, y, text, transform=ax_s.transAxes,
                  color=metric_color, fontsize=8, fontfamily='monospace', va='top')
        y -= dy

    def _value(text, color='white', size=18, dy=0.095):
        nonlocal y
        ax_s.text(0.07, y, text, transform=ax_s.transAxes,
                  color=color, fontsize=size, fontweight='bold',
                  fontfamily='monospace', va='top')
        y -= dy

    def _subtext(text, color='#888888', dy=0.036):
        nonlocal y
        ax_s.text(0.07, y, text, transform=ax_s.transAxes,
                  color=color, fontsize=7.5, fontfamily='monospace', va='top')
        y -= dy

    def _spacer(dy=0.022):
        nonlocal y
        y -= dy

    def _divider(dy=0.028):
        nonlocal y
        ax_s.axhline(y=y + 0.01, xmin=0.04, xmax=0.96,
                     color='#333355', linewidth=0.8)
        y -= dy

    # ---- Overall verdict ----
    _label("OVERALL VERDICT", dy=0.038)
    _value("PASS" if all_passed else "FAIL", color=vc(all_passed), size=26, dy=0.11)
    _divider()

    # ---- SSIM score card ----
    _label("SSIM  (perceptual similarity)", dy=0.038)
    _value(f"{ssim_r['score']:.4f}", dy=0.082)
    _subtext(f"threshold ≥ {params['ssim_threshold']}", dy=0.030)
    _value(vs(ssim_r['passed']), color=vc(ssim_r['passed']), size=10, dy=0.050)
    _divider()

    # ---- Edge score card ----
    _label("Edge Match  (Canny)", dy=0.038)
    _value(f"{edge_r['score']:.4f}", dy=0.082)
    _subtext(f"threshold ≥ {params['canny_edge_match_thresh']}", dy=0.030)
    _value(vs(edge_r['passed']), color=vc(edge_r['passed']), size=10, dy=0.050)
    _divider()

    # ---- Color score card ----
    _label("Color  (Bhattacharyya distance)", dy=0.038)
    _value(f"{color_r['score']:.4f}", dy=0.082)
    _subtext(
        f"H={color_r['distances']['H']:.3f}  "
        f"S={color_r['distances']['S']:.3f}  "
        f"V={color_r['distances']['V']:.3f}",
        dy=0.030,
    )
    _subtext(f"threshold ≤ {params['color_distance_thresh']}", dy=0.030)
    _value(vs(color_r['passed']), color=vc(color_r['passed']), size=10, dy=0.050)
    _divider()

    # ========================================================================
    # FOOTER ROW: spans all 4 columns
    # ========================================================================
    ax_f = fig.add_subplot(gs[n_img_rows, :])
    ax_f.set_facecolor('#12122a')
    ax_f.axis('off')
    ax_f.text(
        0.5, 0.5,
        "Visual Regression Analyzer  |  SSIM · Canny Edge · Bhattacharyya Color  |  "
        "Artifacts: blocking, ringing, banding, mosquito noise",
        transform=ax_f.transAxes, ha='center', va='center',
        color='#555577', fontsize=8, fontfamily='monospace',
    )

    path = os.path.join(output_dir, "analysis_figure.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path



