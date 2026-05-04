# ================================================================================
# visual_reporter.py — Visual Report Generation (PNG Images)
# ================================================================================
#
# PURPOSE:
# This module generates publication-quality visual reports showing the
# complete analysis pipeline and results. Outputs include:
#   1. Heatmap composite: SSIM heatmap overlaid on optimized image with boxes
#   2. Analysis figure: Multi-panel comparison (baseline, optimized, heatmap, components)
#
# ROLE IN PIPELINE:
# This is the visual reporting component. It receives analysis results and
# creates PNG images suitable for:
#   - Developers/QA engineers (visual inspection)
#   - Reports and dashboards
#   - Build artifacts (CI/CD)
#   - Stakeholder communication
#
# OUTPUT FILES:
#   - heatmap_composite.png: 2D heatmap with bounding boxes
#   - analysis_figure.png: Multi-panel detailed analysis figure
#
# ================================================================================

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend (no GUI windows needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def _normalize_cs(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range using min-max normalization.
    
    Linear stretching: value_normalized = (value - min) / (max - min)
    This is a simple but sometimes insufficient normalization (see _normalize_he).
    
    INPUTS:
        arr: Input array (any shape)
    
    RETURNS:
        Normalized array in [0, 1] range, or all-zeros if array is constant
    """
    lo, hi = np.min(arr), np.max(arr)
    # If all values are the same (hi == lo), return zeros
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)

def _normalize_he(arr: np.ndarray) -> np.ndarray:
    """
    Enhanced normalization using CLAHE for better contrast and visibility.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization) is more effective
    than simple linear normalization for displaying subtle SSIM component maps.
    It applies histogram equalization locally to enhance contrast while avoiding
    over-saturation ("contrast limiting").
    
    INPUTS:
        arr: Input array (H × W, float64)
    
    RETURNS:
        Enhanced uint8 array [0, 255] suitable for visualization
    """
    # ====================================================================
    # STEP 1: Initial linear stretch to 0-255 range
    # ====================================================================
    lo, hi = np.min(arr), np.max(arr)
    # If all values are the same, return zeros (nothing to enhance)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    
    # Min-max normalization to [0, 255] range for OpenCV
    rescaled = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    
    # ====================================================================
    # STEP 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ====================================================================
    # Traditional histogram equalization can over-amplify noise.
    # CLAHE applies histogram equalization locally (tile-by-tile) with limits,
    # resulting in better contrast for visualization without artifacts.
    #   - clipLimit=2.0: Contrast amplification limit (0.0=no enhancement, higher=more)
    #   - tileGridSize=(8,8): Divide image into 8×8 tiles for local processing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(rescaled)
    
    return enhanced

def save_heatmap(output_dir: str, composite: np.ndarray, boxes: list) -> str:
    """
    Save heatmap composite with bounding boxes as PNG file.
    
    This creates a simple visualization showing:
    - The SSIM heatmap (colors indicate degradation severity)
    - Bounding boxes around detected degraded regions
    
    INPUTS:
        output_dir: Directory where PNG will be saved
        composite: H × W × 3 heatmap image (float64 [0,1])
        boxes: List of (x, y, w, h) bounding box tuples
    
    RETURNS:
        str: File path where PNG was written
    
    OUTPUT FILE:
        {output_dir}/heatmap_composite.png
        8-bit RGB PNG image with bounding boxes
    """
    # ========================================================================
    # Convert composite from float64 [0,1] to uint8 [0,255] for PNG
    # ========================================================================
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)

    # ========================================================================
    # Draw bounding boxes on the composite
    # ========================================================================
    # Blue boxes (255, 0, 0 in RGB = (0, 0, 255) in BGR for cv2) indicate degraded regions
    # Thickness=3: 3-pixel wide rectangle border (visible but not too intrusive)
    for (x, y, w, h) in boxes:
        # cv2.rectangle draws on image in-place
        # Coordinates: top-left (x, y) to bottom-right (x+w, y+h)
        cv2.rectangle(composite_uint8, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)

    # ========================================================================
    # Convert RGB to BGR for cv2.imwrite (OpenCV uses BGR byte order)
    # ========================================================================
    composite_bgr = cv2.cvtColor(composite_uint8, cv2.COLOR_RGB2BGR)
    
    # ========================================================================
    # Write PNG file
    # ========================================================================
    path = os.path.join(output_dir, "heatmap_composite.png")
    cv2.imwrite(path, composite_bgr)
    return path

def save_figure(output_dir: str, img_baseline: np.ndarray, img_optimized: np.ndarray, composite: np.ndarray, results: dict) -> str:
    """
    Generate and save a comprehensive multi-panel analysis figure.
    
    This creates a publication-quality figure showing:
    - Row 1: Baseline image, optimized image, SSIM heatmap composite
    - Row 2 (optional): Luminance, contrast, structure SSIM components
    - Row 3: Summary metrics (verdict, score, degraded region count)
    
    INPUTS:
        output_dir: Directory where PNG will be saved
        img_baseline: Baseline image (H × W × 3, float64 [0,1])
        img_optimized: Optimized image (H × W × 3, float64 [0,1])
        composite: SSIM heatmap composite (H × W × 3, float64 [0,1])
        results: Dictionary with keys:
            - ssim_score: Overall SSIM value
            - ssim_passed: Boolean pass/fail
            - boxes: Bounding boxes of degraded regions
            - params: Configuration parameters
            - l_map, c_map, s_map: Component maps (optional)
    
    RETURNS:
        str: File path where PNG was written
    
    OUTPUT FILE:
        {output_dir}/analysis_figure.png
        High-resolution figure (20 inches wide, 8-12 inches tall, 150 DPI)
    """
    # ========================================================================
    # Extract results and parameters
    # ========================================================================
    ssim_score = results["ssim_score"]
    ssim_passed = results["ssim_passed"]
    boxes = results["boxes"]
    params = results["params"]
    l_map = results.get("l_map")  # Luminance component (or None)
    c_map = results.get("c_map")  # Contrast component (or None)
    s_map = results.get("s_map")  # Structure component (or None)

    # Check if SSIM component maps are available
    # If not saved, figure will have 2 rows; if saved, 3 rows
    has_components = all(m is not None for m in [l_map, c_map, s_map])
    n_rows = 3 if has_components else 2

    # ========================================================================
    # Figure styling and layout setup
    # ========================================================================
    # Dark theme with accent colors for professional appearance
    panel_bg = '#e0e0e0'      # Light gray for panel backgrounds
    title_color = '#16213e'   # Dark blue for titles
    metric_color = '#90caf9'  # Light blue for metric labels
    pass_color = '#00c853'    # Green for PASS
    fail_color = '#ff1744'    # Red for FAIL

    # Determine verdict and color
    verdict = "PASS" if ssim_passed else "FAIL"
    verdict_color = pass_color if ssim_passed else fail_color

    # Adjust figure height based on whether component maps are included
    fig_height = 12 if has_components else 8
    
    # Create figure with GridSpec layout
    # Layout: n_rows × 3 columns, where:
    #   - Row 0: baseline | optimized | heatmap
    #   - Row 1: luminance | contrast | structure (if has_components)
    #   - Row n_rows-1: summary metrics (spanning all 3 columns)
    fig = plt.figure(figsize=(20, fig_height), facecolor='#1a1a2e')
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.40, wspace=0.25, left=0.04, right=0.97, top=0.92, bottom=0.06)

    # Main title
    fig.suptitle("Visual Regression Analyzer — SSIM Analysis Report", color='white', fontsize=15, fontweight='bold', y=0.96)

    # ========================================================================
    # ROW 0: Image comparison (baseline, optimized, heatmap)
    # ========================================================================
    # Prepare heatmap with bounding boxes
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)
    for (x, y, w, h) in boxes:
        # Draw blue boxes around detected degradations
        cv2.rectangle(composite_uint8, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)

    # Panel 1: Baseline image (reference/original quality)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_baseline)
    ax1.set_title("Original (Baseline)", color='white', fontsize=10, pad=6)
    ax1.axis('off')  # Hide axes (no ticks, no labels)

    # Panel 2: Optimized image (what we're testing)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_optimized)
    ax2.set_title("Optimized", color='white', fontsize=10, pad=6)
    ax2.axis('off')

    # Panel 3: SSIM heatmap composite with degradation boxes
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(composite_uint8)
    ax3.set_title(f"SSIM Heatmap ({len(boxes)} degraded region(s))", color='white', fontsize=10, pad=6)
    ax3.axis('off')

    # ========================================================================
    # ROW 1: SSIM component maps (optional, only if saved during computation)
    # ========================================================================
    # Each component shows a different aspect of perceived similarity:
    # - Luminance: Are brightness levels similar?
    # - Contrast: Are texture variations similar?
    # - Structure: Are edge patterns in the same places?
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
            # Use CLAHE normalization for better visual contrast
            normalized = _normalize_he(arr)

            ax.imshow(normalized, cmap='hot')  # Hot colormap: blue (low) → red → yellow (high)
            ax.set_title(f"{label} Component\n({sublabel})", color='white', fontsize=10, pad=6)
            ax.axis('off')

    # ========================================================================
    # SUMMARY BAR (last row): Key metrics and verdict
    # ========================================================================
    ax_s = fig.add_subplot(gs[n_rows - 1, :])  # Spans all 3 columns
    ax_s.set_facecolor('#16213e')  # Dark blue background
    ax_s.axis('off')  # No axis ticks or labels

    # --------
    # Section 1: OVERALL VERDICT
    # --------
    ax_s.text(0.08, 0.92, "OVERALL VERDICT", transform=ax_s.transAxes, ha='left', va='top', 
              color='white', fontsize=11, fontweight='bold')
    ax_s.text(0.08, 0.72, verdict, transform=ax_s.transAxes, ha='left', va='top', 
              color=verdict_color, fontsize=32, fontweight='bold', fontfamily='monospace')

    # --------
    # Section 2: SSIM SCORE DETAILS
    # --------
    ax_s.text(0.30, 0.92, "SSIM Score", transform=ax_s.transAxes, 
              color=metric_color, fontsize=10, fontfamily='monospace')
    ax_s.text(0.30, 0.72, f"{ssim_score:.4f}", transform=ax_s.transAxes, 
              color='white', fontsize=20, fontweight='bold', fontfamily='monospace')
    ax_s.text(0.30, 0.56, f"Threshold: {params['ssim_threshold']}", transform=ax_s.transAxes, 
              color='#888888', fontsize=9, fontfamily='monospace')
    ax_s.text(0.30, 0.46, "PASS ✓" if ssim_passed else "FAIL ✗", transform=ax_s.transAxes, 
              color=verdict_color, fontsize=11, fontweight='bold', fontfamily='monospace')

    # --------
    # Section 3: DEGRADED REGIONS COUNT
    # --------
    ax_s.text(0.52, 0.92, "Degraded Regions", transform=ax_s.transAxes, 
              color=metric_color, fontsize=10, fontfamily='monospace')
    ax_s.text(0.52, 0.72, str(len(boxes)), transform=ax_s.transAxes, 
              color='white', fontsize=32, fontweight='bold', fontfamily='monospace')
    ax_s.text(0.52, 0.46, "detected" if boxes else "none found", transform=ax_s.transAxes, 
              color='#888888', fontsize=9, fontfamily='monospace')

    # --------
    # Footer: Report metadata
    # --------
    ax_s.text(0.5, 0.08, "Visual Regression Analyzer v1.0 | SSIM Phase Report", 
              transform=ax_s.transAxes, ha='center', color='#555577', fontsize=7, fontfamily='monospace')

    # ========================================================================
    # Save figure to PNG file
    # ========================================================================
    path = os.path.join(output_dir, "analysis_figure.png")
    # Settings:
    #   - dpi=150: High resolution (150 dots per inch)
    #   - bbox_inches='tight': Remove whitespace around figure
    #   - facecolor: Use the figure's background color
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)  # Close figure to free memory
    return path