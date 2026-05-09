#!/usr/bin/env python3
# ================================================================================
# main.py — Entry Point for the Visual Regression Analyzer Pipeline
# ================================================================================
#
# OVERVIEW:
# This module orchestrates the complete visual regression analysis pipeline.
# It coordinates six sequential phases to compare baseline and optimized images,
# detect visual degradation across three independent dimensions, assign semantic
# cause hypotheses to each degraded region, and generate comprehensive reports.
#
# PIPELINE ARCHITECTURE:
#
#   Phase 1: ACQUISITION (pipeline/acquisition.py)
#   ├─ Load baseline (high-quality reference) and optimized (test) images
#   ├─ Validate image formats and dimensions
#   └─ Return normalized float64 arrays in [0, 1]
#
#   Phase 2: ANALYSIS — three independent passes run on the same image pair
#   │
#   ├─ Phase 2a: SSIM (pipeline/ssim_pass.py)
#   │   ├─ Compute Structural Similarity Index Measure (perceptual similarity)
#   │   ├─ Generate per-pixel SSIM map and component maps (L, C, S)
#   │   └─ Returns score, verdict, and per-pixel map
#   │
#   ├─ Phase 2b: EDGE (pipeline/edge_pass.py)
#   │   ├─ Run Canny edge detection on both images
#   │   ├─ Compute edge match score (fraction of baseline edges preserved)
#   │   └─ Returns score, verdict, and edge maps (reused by aggregator)
#   │
#   └─ Phase 2c: COLOR (pipeline/color_pass.py)
#       ├─ Compute HSV color histograms for both images
#       ├─ Compute per-channel Bhattacharyya distances
#       └─ Returns score, verdict, and histogram data
#
#   Phase 3: AGGREGATION (pipeline/aggregator.py)
#   ├─ Create visual heatmap from SSIM scores
#   ├─ Detect degraded regions using morphological operations on the SSIM map
#   ├─ For each region, compute local scores using already-computed maps:
#   │     SSIM local:  slice of ssim_map (free — no re-computation)
#   │     Edge local:  slice of edge_baseline/edge_optimized (free — no re-computation)
#   │     Color local: re-run histogram on cropped region (necessary)
#   └─ Return composite image + list of structured combined region dicts
#
#   Phase 4: COMBINATION (pipeline/combinator.py)
#   ├─ Apply truth table to each region's pass attribution
#   ├─ Assign cause hypothesis and confidence per region
#   └─ Pure function — no image ops, no spatial math
#
#   Phase 5: REPORTING — STRUCTURED (reporting/json_reporter.py)
#   ├─ Serialize all pass results and enriched regions to machine-readable JSON
#   └─ Enable CI/CD integration and automated testing
#
#   Phase 6: REPORTING — VISUAL (reporting/visual_reporter.py)
#   ├─ Save per-pass intermediate images to their respective subdirectories
#   ├─ Generate SSIM heatmap composite with cause-labelled bounding boxes
#   └─ Generate multi-panel analysis figure with per-region breakdown sidebar
#
# USAGE:
#   From the project root:
#     python main.py <baseline_image> <optimized_image>
#   Example:
#     python main.py data/baseline.png data/optimized.png
#
#   If no arguments are provided, defaults to:
#     python main.py data/baseline.png data/optimized.png
#
# OUTPUT ARTIFACTS:
#   output/
#   ├─ regression_report.json     — Machine-readable results (all passes + regions)
#   ├─ heatmap_composite.png      — SSIM heatmap with cause-labelled region boxes
#   ├─ analysis_figure.png        — Full multi-panel pipeline report with sidebar
#   ├─ ssim/                      — SSIM component maps (L, C, S)
#   ├─ edge/                      — Edge intermediates (Sobel X/Y, NMS, edge maps)
#   └─ color/                     — Bhattacharyya histogram overlay
#
# ================================================================================

import sys
import os

# Ensure the project root is on the path so backend.* imports resolve correctly
# regardless of the working directory the user runs from.
sys.path.insert(0, os.path.dirname(__file__))

import backend.config as config
from backend.pipeline.acquisition   import load_images
from backend.pipeline.ssim_pass     import run as run_ssim
from backend.pipeline.edge_pass     import run as run_edge
from backend.pipeline.color_pass    import run as run_color, save_histogram_overlay
from backend.pipeline.aggregator    import build_heatmap
from backend.pipeline.combinator    import combine as combine_regions
from backend.reporting.json_reporter   import save as save_json
from backend.reporting.visual_reporter import (
    save_heatmap,
    save_figure,
    save_ssim_intermediates,
    save_edge_intermediates,
)


def main():
    # ========================================================================
    # ARGUMENT PARSING
    # ========================================================================
    if len(sys.argv) == 3:
        baseline_path  = sys.argv[1]
        optimized_path = sys.argv[2]
    else:
        baseline_path  = "./data/baseline.png"
        optimized_path = "./data/optimized.png"

    # ========================================================================
    # SETUP: Output directories and parameter bundle
    # ========================================================================
    for d in [config.OUTPUT_DIR,
              config.OUTPUT_DIR_SSIM,
              config.OUTPUT_DIR_EDGE,
              config.OUTPUT_DIR_COLOR]:
        os.makedirs(d, exist_ok=True)

    # All tunable parameters in one flat dict so every module receives them
    # uniformly — no module imports config directly.
    params = {
        # SSIM
        "ssim_threshold":       config.SSIM_THRESHOLD,
        "ssim_window_size":     config.SSIM_WINDOW_SIZE,
        "ssim_sigma":           config.SSIM_SIGMA,
        # Heatmap / aggregation
        "heatmap_alpha":        config.HEATMAP_ALPHA,
        "degradation_thresh":   config.DEGRADATION_THRESH,
        "morph_radius":         config.MORPH_RADIUS,
        "min_region_area":      config.MIN_REGION_AREA,
        # Edge
        "canny_blur_kernel":         config.CANNY_BLUR_KERNEL,
        "canny_blur_sigma":          config.CANNY_BLUR_SIGMA,
        "canny_low_thresh":          config.CANNY_LOW_THRESH,
        "canny_high_thresh":         config.CANNY_HIGH_THRESH,
        "canny_edge_match_thresh":   config.CANNY_EDGE_MATCH_THRESH,
        # Color
        "color_hist_bins":           config.COLOR_HIST_BINS,
        "color_distance_thresh":     config.COLOR_DISTANCE_THRESH,
        # Combinator / reporting
        "iou_threshold":             config.IOU_THRESHOLD,
        "top_regions_in_report":     config.TOP_REGIONS_IN_REPORT,
        # Global
        "save_intermediates":        config.SAVE_INTERMEDIATES,
    }

    print("===============================")
    print(" VISUAL REGRESSION ANALYZER")
    print("===============================\n")

    # ========================================================================
    # Phase 1: Acquisition
    # ========================================================================
    print("[Phase 1] Loading images...")
    img_baseline, img_optimized = load_images(baseline_path, optimized_path)

    # ========================================================================
    # Phase 2a: SSIM
    # ========================================================================
    print("\n[Phase 2a] Computing SSIM...")
    ssim_result = run_ssim(img_baseline, img_optimized, params)
    ssim_passed = ssim_result["score"] >= config.SSIM_THRESHOLD
    ssim_result["passed"]  = ssim_passed
    ssim_result["verdict"] = "PASS" if ssim_passed else "FAIL"

    print(f"  >> Score     : {ssim_result['score']:.4f}")
    print(f"  >> Threshold : {config.SSIM_THRESHOLD}")
    print(f"  >> Result    : {'PASS ✓' if ssim_passed else 'FAIL ✗'}")

    # ========================================================================
    # Phase 2b: Edge
    # ========================================================================
    print("\n[Phase 2b] Running Canny edge pass...")
    edge_result = run_edge(img_baseline, img_optimized, params)

    print(f"  >> Edge match : {edge_result['score']:.4f}")
    print(f"  >> Threshold  : {config.CANNY_EDGE_MATCH_THRESH}")
    print(f"  >> Result     : {'PASS ✓' if edge_result['passed'] else 'FAIL ✗'}")

    # ========================================================================
    # Phase 2c: Color
    # ========================================================================
    print("\n[Phase 2c] Running Bhattacharyya color pass...")
    color_result = run_color(img_baseline, img_optimized, params)

    print(f"  >> Mean distance : {color_result['score']:.4f}")
    print(f"  >> Per channel   : H={color_result['distances']['H']:.4f}  "
          f"S={color_result['distances']['S']:.4f}  "
          f"V={color_result['distances']['V']:.4f}")
    print(f"  >> Threshold     : {config.COLOR_DISTANCE_THRESH}")
    print(f"  >> Result        : {'PASS ✓' if color_result['passed'] else 'FAIL ✗'}")

    # ========================================================================
    # Phase 3: Aggregation
    # ========================================================================
    # Pass the already-computed edge maps into the aggregator so local edge
    # scores are derived by slicing those maps — no redundant Canny re-runs.
    print("\n[Phase 3] Building heatmap and detecting degraded regions...")
    composite, combined_regions = build_heatmap(
        img_baseline,
        img_optimized,
        ssim_result["ssim_map"],
        edge_result.get("edge_baseline"),
        edge_result.get("edge_optimized"),
        params,
    )
    print(f"  >> Degraded regions detected: {len(combined_regions)}")

    # ========================================================================
    # Phase 4: Combination — semantic cause hypothesis per region
    # ========================================================================
    print("\n[Phase 4] Applying combinator truth table...")
    enriched_regions = combine_regions(
        combined_regions,
        ssim_thresh=config.SSIM_THRESHOLD,
        edge_thresh=config.CANNY_EDGE_MATCH_THRESH,
        color_thresh=config.COLOR_DISTANCE_THRESH,
    )
    for region in enriched_regions:
        x, y, w, h = region['bbox']
        print(f"  >> Region ({x},{y}) {w}×{h}px  "
              f"[{region['confidence']}]  {region['cause_tag']}")

    # Attach enriched regions to ssim_result so reporters find them at results["ssim"]["boxes"]
    ssim_result["boxes"] = enriched_regions

    # ========================================================================
    # Assemble namespaced results dictionary
    # ========================================================================
    results = {
        "ssim":   ssim_result,
        "edge":   edge_result,
        "color":  color_result,
        "params": params,
    }

    # ========================================================================
    # Phase 5 & 6: Reporting
    # ========================================================================
    print("\n[Reporting] Saving outputs...")

    # --- Intermediates ---
    if config.SAVE_INTERMEDIATES:
        save_ssim_intermediates(config.OUTPUT_DIR_SSIM, ssim_result)
        save_edge_intermediates(config.OUTPUT_DIR_EDGE, edge_result)
        hist_path = save_histogram_overlay(config.OUTPUT_DIR_COLOR, color_result)
        print(f"  >> Histogram overlay : {hist_path}")

    # --- Main outputs ---
    heatmap_path = save_heatmap(config.OUTPUT_DIR, composite, enriched_regions)
    print(f"  >> Heatmap           : {heatmap_path}")

    json_path = save_json(config.OUTPUT_DIR, results)
    print(f"  >> JSON report       : {json_path}")

    fig_path = save_figure(config.OUTPUT_DIR, img_baseline, img_optimized, composite, results)
    print(f"  >> Analysis figure   : {fig_path}")

    # ========================================================================
    # Final verdict
    # ========================================================================
    all_passed = ssim_passed and edge_result["passed"] and color_result["passed"]

    print("\n===============================")
    if all_passed:
        print(" FINAL VERDICT: ✓ PASS")
        print(" All three passes within acceptable bounds.")
    else:
        failed = [name for name, passed in [
            ("SSIM",  ssim_passed),
            ("Edge",  edge_result["passed"]),
            ("Color", color_result["passed"]),
        ] if not passed]
        print(" FINAL VERDICT: ✗ FAIL")
        print(f" Failed passes: {', '.join(failed)}")

    print("===============================")
    print(f" Output saved to: {config.OUTPUT_DIR}/")
    print("===============================\n")


if __name__ == "__main__":
    main()
