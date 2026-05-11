# ================================================================================
# visual_reporter.py — Visual Report Generation
# ================================================================================
#
# Dual path:
#   render_heatmap_b64()  — returns base64 PNG string (API)
#   save_heatmap()        — writes PNG to disk (CLI)
#   save_figure()         — multi-panel analysis figure (CLI)
#   save_ssim_intermediates() / save_edge_intermediates()  — CLI helpers
#
# ================================================================================

import base64
import io
import os

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ================================================================================
# Box color scheme
# ================================================================================

def _box_color(passes_failed):
    failed    = set(passes_failed)
    has_edge  = 'edge'  in failed
    has_color = 'color' in failed
    if has_edge and has_color: return (220, 30,  30)
    if has_edge:               return (255, 140,  0)
    if has_color:              return (30,  144, 255)
    return (255, 220, 0)

_LEGEND_ENTRIES = [
    ((255, 220,   0), "SSIM only"),
    ((255, 140,   0), "SSIM + Edge"),
    ((30,  144, 255), "SSIM + Color"),
    ((220,  30,  30), "SSIM + Edge + Color"),
]

_CONFIDENCE_COLORS = {
    "clean":      "#00c853",
    "low":        "#90caf9",
    "medium":     "#ffe082",
    "high":       "#ff9800",
    "very_high":  "#ff5722",
    "definitive": "#b71c1c",
}


# ================================================================================
# Normalisation helpers
# ================================================================================

def _normalize_he(arr):
    lo, hi = np.min(arr), np.max(arr)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    rescaled = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rescaled)


def _float_to_uint8(arr):
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


# ================================================================================
# Heatmap renderer (shared logic)
# ================================================================================

def _draw_heatmap(composite: np.ndarray, enriched_regions: list) -> np.ndarray:
    """Draw bounding boxes + cause labels + legend onto the composite. Returns uint8 RGB."""
    out = _float_to_uint8(composite).copy()

    for region in enriched_regions:
        x, y, w, h = region['bbox']
        color       = _box_color(region.get('passes_failed', ['ssim']))
        tag         = region.get('cause_tag', '')
        conf        = region.get('confidence', '')

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Label — cause tag above box
        label = f"{tag}  [{conf}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y - 6, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label, (x + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Legend (bottom-left)
    lx, ly = 12, out.shape[0] - 12 - len(_LEGEND_ENTRIES) * 22
    for rgb, label in _LEGEND_ENTRIES:
        cv2.rectangle(out, (lx, ly), (lx + 16, ly + 14), rgb, -1)
        cv2.putText(out, label, (lx + 22, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        ly += 22

    return out


# ================================================================================
# API path — returns base64
# ================================================================================

def render_heatmap_b64(composite: np.ndarray, enriched_regions: list) -> str:
    """Render annotated heatmap and return as base64 PNG string."""
    out     = _draw_heatmap(composite, enriched_regions)
    bgr     = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ""


# ================================================================================
# CLI path — writes to disk
# ================================================================================

def save_heatmap(output_dir: str, composite: np.ndarray, enriched_regions: list) -> str:
    out  = _draw_heatmap(composite, enriched_regions)
    bgr  = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, "heatmap_composite.png")
    cv2.imwrite(path, bgr)
    return path


def save_ssim_intermediates(ssim_dir, ssim_result):
    for arr, name in [(ssim_result.get("l_map"), "luminance"),
                      (ssim_result.get("c_map"), "contrast"),
                      (ssim_result.get("s_map"), "structure")]:
        if arr is not None:
            cv2.imwrite(os.path.join(ssim_dir, f"{name}_component.png"),
                        _normalize_he(arr))


def save_edge_intermediates(edge_dir, edge_result):
    for key, filename in [('sobel_x', 'sobel_x.png'), ('sobel_y', 'sobel_y.png')]:
        arr = edge_result.get(key)
        if arr is not None:
            cv2.imwrite(os.path.join(edge_dir, filename), _normalize_he(np.abs(arr)))
    nms = edge_result.get("nms_map")
    if nms is not None:
        cv2.imwrite(os.path.join(edge_dir, "nms_map.png"), _normalize_he(nms))
    for key, filename in [('edge_baseline', 'edge_baseline.png'),
                           ('edge_optimized', 'edge_optimized.png')]:
        arr = edge_result.get(key)
        if arr is not None:
            cv2.imwrite(os.path.join(edge_dir, filename),
                        (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr)


def save_figure(output_dir, img_baseline, img_optimized, composite, results):
    """Full multi-panel analysis figure — CLI only."""
    ssim_r  = results["ssim"]
    edge_r  = results["edge"]
    color_r = results["color"]
    params  = results["params"]

    combined_regions = ssim_r.get("boxes", [])

    has_ssim_detail = all(k in ssim_r for k in ('l_map', 'c_map', 's_map'))
    has_edge_maps   = ('edge_baseline' in edge_r and 'edge_optimized' in edge_r)
    has_histograms  = bool(color_r.get("histograms"))
    all_passed      = ssim_r["passed"] and edge_r["passed"] and color_r["passed"]

    n_img_rows = 1 + int(has_ssim_detail and has_edge_maps) + int(has_histograms)

    bg_dark  = '#0d0d1a'
    bg_panel = '#16213e'
    metric_color = '#8899bb'

    fig = plt.figure(figsize=(22, 6 * n_img_rows + 1.5), facecolor=bg_dark)
    gs  = GridSpec(n_img_rows + 1, 4, figure=fig,
                   width_ratios=[1, 1, 1, 0.55],
                   hspace=0.35, wspace=0.18)

    def vc(p): return "#00c853" if p else "#ff1744"
    def vs(p): return "PASS ✓" if p else "FAIL ✗"

    current_row = 0
    heatmap_uint8 = _draw_heatmap(composite, combined_regions)

    for col, (img, title) in enumerate([
        (img_baseline,  "Baseline"),
        (img_optimized, "Optimized"),
    ]):
        ax = fig.add_subplot(gs[current_row, col])
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=9, pad=5)
        ax.axis('off')

    ax = fig.add_subplot(gs[current_row, 2])
    ax.imshow(heatmap_uint8)
    ax.set_title(f"SSIM Heatmap  ({len(combined_regions)} region(s))",
                 color='white', fontsize=9, pad=5)
    ax.axis('off')
    legend_patches = [mpatches.Patch(color=tuple(c/255 for c in rgb), label=lbl)
                      for rgb, lbl in _LEGEND_ENTRIES]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=7,
              facecolor=bg_panel, edgecolor='#444466', labelcolor='white', framealpha=0.88)
    current_row += 1

    if has_ssim_detail and has_edge_maps:
        s_map = ssim_r['s_map']
        c_map = ssim_r['c_map']
        edge_baseline  = edge_r['edge_baseline']
        edge_optimized = edge_r['edge_optimized']
        lost_edges = edge_baseline & ~edge_optimized

        for col, (data, title, cmap) in enumerate([
            (lost_edges, f"Lost Edges  ({edge_r['score']:.1%})", 'hot'),
            (s_map,      "SSIM Structure",                       'hot'),
            (c_map,      "SSIM Contrast",                        'hot'),
        ]):
            ax = fig.add_subplot(gs[current_row, col])
            ax.imshow(_normalize_he(data) if data.dtype == np.float64 else data, cmap=cmap)
            ax.set_title(title, color='white', fontsize=9, pad=5)
            ax.axis('off')
        current_row += 1

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
            meta = channel_meta[name]
            x    = np.linspace(0, meta['x_max'], len(p_base))
            ax.fill_between(x, p_base, alpha=0.45, color=meta['color_b'], label='Baseline')
            ax.plot(x, p_base, color=meta['color_b'], linewidth=1.5)
            ax.fill_between(x, p_opt,  alpha=0.45, color=meta['color_o'], label='Optimized')
            ax.plot(x, p_opt,  color=meta['color_o'], linewidth=1.5)
            ax.set_title(f"{meta['label']} — D: {distances[name]:.4f}",
                         color='white', fontsize=9, pad=5)
            ax.tick_params(colors='#888888', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            ax.legend(facecolor=bg_dark, edgecolor='#333355', labelcolor='white', fontsize=8)
        current_row += 1

    # Sidebar
    ax_s = fig.add_subplot(gs[0:n_img_rows, 3])
    ax_s.set_facecolor(bg_panel)
    ax_s.axis('off')
    y = 0.98

    def _label(t, dy=0.042):
        nonlocal y
        ax_s.text(0.07, y, t, transform=ax_s.transAxes,
                  color=metric_color, fontsize=8, fontfamily='monospace', va='top')
        y -= dy

    def _value(t, color='white', size=18, dy=0.095):
        nonlocal y
        ax_s.text(0.07, y, t, transform=ax_s.transAxes,
                  color=color, fontsize=size, fontweight='bold',
                  fontfamily='monospace', va='top')
        y -= dy

    def _subtext(t, color='#888888', dy=0.036):
        nonlocal y
        ax_s.text(0.07, y, t, transform=ax_s.transAxes,
                  color=color, fontsize=7.5, fontfamily='monospace', va='top')
        y -= dy

    def _divider(dy=0.028):
        nonlocal y
        ax_s.axhline(y=y + 0.01, xmin=0.04, xmax=0.96, color='#333355', linewidth=0.8)
        y -= dy

    _label("OVERALL VERDICT", dy=0.038)
    _value("PASS" if all_passed else "FAIL", color=vc(all_passed), size=26, dy=0.11)
    _divider()
    _label("SSIM", dy=0.038)
    _value(f"{ssim_r['score']:.4f}", dy=0.082)
    _subtext(f"threshold ≥ {params['ssim_threshold']}", dy=0.030)
    _value(vs(ssim_r['passed']), color=vc(ssim_r['passed']), size=10, dy=0.050)
    _divider()
    _label("Edge Match", dy=0.038)
    _value(f"{edge_r['score']:.4f}", dy=0.082)
    _subtext(f"threshold ≥ {params['canny_edge_match_thresh']}", dy=0.030)
    _value(vs(edge_r['passed']), color=vc(edge_r['passed']), size=10, dy=0.050)
    _divider()
    _label("Color (Bhattacharyya)", dy=0.038)
    _value(f"{color_r['score']:.4f}", dy=0.082)
    _subtext(f"H={color_r['distances']['H']:.3f}  S={color_r['distances']['S']:.3f}  V={color_r['distances']['V']:.3f}", dy=0.030)
    _value(vs(color_r['passed']), color=vc(color_r['passed']), size=10, dy=0.050)

    # Footer
    ax_f = fig.add_subplot(gs[n_img_rows, :])
    ax_f.set_facecolor('#12122a')
    ax_f.axis('off')
    ax_f.text(0.5, 0.5, "Visual Regression Analyzer  |  SSIM · Canny Edge · Bhattacharyya Color",
              transform=ax_f.transAxes, ha='center', va='center',
              color='#555577', fontsize=8, fontfamily='monospace')

    path = os.path.join(output_dir, "analysis_figure.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
