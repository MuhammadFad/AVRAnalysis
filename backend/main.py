#!/usr/bin/env python3
# ================================================================================
# main.py — Entry Point for the Visual Regression Analyzer Pipeline
# ================================================================================
#
# OVERVIEW:
# This module orchestrates the complete visual regression analysis pipeline.
# It coordinates five sequential phases to compare baseline and optimized images,
# detect visual degradation, and generate comprehensive reports.
#
# PIPELINE ARCHITECTURE:
# The analysis consists of 5 phases, each handled by a dedicated module:
#
#   Phase 1: ACQUISITION (acquisition.py)
#   ├─ Load baseline (high-quality reference) and optimized (test) images
#   ├─ Validate image formats and dimensions
#   └─ Return normalized float64 arrays in [0, 1]
#
#   Phase 2: SSIM ANALYSIS (ssim_pass.py)
#   ├─ Compute Structural Similarity Index Measure
#   ├─ Calculate per-pixel SSIM scores
#   ├─ Generate component maps (luminance, contrast, structure)
#   └─ Return overall SSIM score and per-pixel map
#
#   Phase 3: AGGREGATION (aggregator.py)
#   ├─ Create visual heatmap from SSIM scores
#   ├─ Detect degraded regions using morphological operations
#   ├─ Extract bounding boxes of significant degradations
#   └─ Return alpha-blended composite and region list
#
#   Phase 4: REPORTING - STRUCTURED (json_reporter.py)
#   ├─ Serialize analysis results to machine-readable JSON
#   ├─ Include SSIM score, threshold comparison, detected regions
#   └─ Enable CI/CD integration and automated testing
#
#   Phase 5: REPORTING - VISUAL (visual_reporter.py)
#   ├─ Generate publication-quality PNG visualizations
#   ├─ Create heatmap composite with bounding boxes
#   ├─ Create multi-panel figure (images + components + metrics)
#   └─ Enable human inspection and reporting
#
# USAGE:
#   Command line:
#     python main.py <baseline_image> <optimized_image>
#   Example:
#     python main.py data/baseline.png data/optimized.png
#
#   If no arguments provided, defaults to:
#     python main.py ./data/baseline.png ./data/optimized.png
#
# OUTPUT ARTIFACTS (saved to config.OUTPUT_DIR):
#   - regression_report.json: Machine-readable results (pass/fail, score, boxes)
#   - heatmap_composite.png: Simple heatmap with bounding boxes
#   - analysis_figure.png: Comprehensive multi-panel analysis report
#   - Additional intermediates (component maps) if SAVE_INTERMEDIATES=True
#
# ================================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import config
from pipeline.acquisition import load_images
from pipeline.ssim_pass import run as run_ssim
from pipeline.aggregator import build_heatmap
from reporting.json_reporter import save as save_json
from reporting.visual_reporter import save_heatmap, save_figure

def main():
    # ========================================================================
    # ARGUMENT PARSING: Get image paths from command line or use defaults
    # ========================================================================
    if len(sys.argv) == 3:
        # User provided image paths as arguments
        baseline_path = sys.argv[1]
        optimized_path = sys.argv[2]
    else:
        # Use default paths if not provided
        baseline_path = "./data/baseline.png"
        optimized_path = "./data/optimized.png"

    # ========================================================================
    # SETUP: Create output directory and gather all configuration parameters
    # ========================================================================
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Bundle all tunable parameters into a single dict for easy passing
    # This ensures consistency across all pipeline phases
    params = {
        "ssim_threshold": config.SSIM_THRESHOLD,           # Pass threshold
        "heatmap_alpha": config.HEATMAP_ALPHA,             # Heatmap opacity
        "degradation_thresh": config.DEGRADATION_THRESH,   # Region detection sensitivity
        "morph_radius": config.MORPH_RADIUS,               # Morphological closing radius
        "min_region_area": config.MIN_REGION_AREA,         # Minimum box area filter
    }

    print("===============================")
    print(" VISUAL REGRESSION ANALYZER")
    print("===============================\n")

    # --- Phase 1: Acquisition ---
    print(" [Phase 1] Loading images...")
    img_baseline, img_optimized = load_images(baseline_path, optimized_path)

    # --- Phase 2: SSIM ---
    print("\n[Phase 2] Computing SSIM...")
    ssim_result = run_ssim(img_baseline, img_optimized)
    ssim_score = ssim_result["score"]
    ssim_passed = ssim_score >= config.SSIM_THRESHOLD

    print(f" >> SSIM Score : {ssim_score:.4f}")
    print(f" >> Threshold : {config.SSIM_THRESHOLD}")
    print(f" >> Result : {'PASS ✓' if ssim_passed else 'FAIL ✗'}")

    # --- Aggregation ---
    print("\n[Aggregation] Building heatmap...")
    composite, boxes = build_heatmap(img_optimized, ssim_result["ssim_map"], params)
    print(f" >> Degraded regions: {len(boxes)}")

    # --- Assemble results dict ---
    results = {
        "ssim_score": ssim_score,
        "ssim_passed": ssim_passed,
        "boxes": boxes,
        "params": params,
        "l_map": ssim_result.get("l_map"),
        "c_map": ssim_result.get("c_map"),
        "s_map": ssim_result.get("s_map"),
    }

    # --- Reporting ---
    print("\n[Reporting] Saving outputs...")
    heatmap_path = save_heatmap(config.OUTPUT_DIR, composite, boxes)
    print(f" >> Heatmap : {heatmap_path}")

    json_path = save_json(config.OUTPUT_DIR, results)
    print(f" >> JSON report: {json_path}")

    fig_path = save_figure(config.OUTPUT_DIR, img_baseline, img_optimized, composite, results)
    print(f" >> Figure : {fig_path}")

    # --- Verdict ---
    print("\n=======================")
    if ssim_passed:
        print(" FINAL VERDICT: ✓ PASS")
        print(" Optimization is within acceptable visual bounds.")
    else:
        print(" FINAL VERDICT: ✗ FAIL")
        print(" Optimization has introduced perceptual degradation.")
    
    print("===============================")
    print(f" saved to: {config.OUTPUT_DIR}/")
    print("===============================\n")

if __name__ == "__main__":
    main()