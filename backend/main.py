#!/usr/bin/env python3
# =========================================================================
#  main.py — Entry point for the Visual Regression Analyzer
#             (Evaluation 1: SSIM Phase Only)
#
#  Usage:
#    python main.py <baseline_image> <optimized_image>
#
#  Example:
#    python main.py images/baseline.png images/optimized.png
#
#  Outputs (saved to output/ folder):
#    - heatmap_composite.png   — heatmap overlaid on optimized image
#    - analysis_figure.png     — 4-panel report figure
#    - regression_report.json  — structured JSON report
# =========================================================================

import sys
import os

# Make sure pipeline/ and reporting/ are importable regardless of
# where the script is run from
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

import config
from pipeline.acquisition import load_images
from pipeline.ssim_pass    import run as run_ssim
from pipeline.aggregator   import build_heatmap
from reporting.json_reporting import save_json, save_figure

def main():

    # ------------------------------------------------------------------
    #  Parse arguments
    # ------------------------------------------------------------------
    if len(sys.argv) == 3:
        baseline_path  = sys.argv[1]
        optimized_path = sys.argv[2]
    else:
        # Default paths for quick testing — put your images here
        baseline_path  = "./data/baseline.png"
        optimized_path = "./data/optimized.png"

    # ------------------------------------------------------------------
    #  Setup
    # ------------------------------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    params = {
        'ssim_threshold':    config.SSIM_THRESHOLD,
        'heatmap_alpha':     config.HEATMAP_ALPHA,
        'degradation_thresh':config.DEGRADATION_THRESH,
        'morph_radius':      config.MORPH_RADIUS,
        'min_region_area':   config.MIN_REGION_AREA,
    }

    print("=====================================================")
    print("  VISUAL REGRESSION ANALYZER — SSIM PHASE REPORT")
    print("=====================================================\n")

    # ------------------------------------------------------------------
    #  Phase 1: Data Acquisition
    # ------------------------------------------------------------------
    print("[Phase 1] Loading images...")
    img_baseline, img_optimized = load_images(baseline_path, optimized_path)

    # ------------------------------------------------------------------
    #  Phase 2: SSIM Analysis
    # ------------------------------------------------------------------
    print("\n[Phase 2] Computing SSIM...")
    ssim_score, ssim_map = run_ssim(img_baseline, img_optimized)
    ssim_passed = ssim_score >= params['ssim_threshold']

    print(f"  >> Global SSIM Score : {ssim_score:.4f}")
    print(f"  >> Threshold         : {params['ssim_threshold']:.2f}")
    print(f"  >> Result            : {'PASS ✓' if ssim_passed else 'FAIL ✗'}")

    # ------------------------------------------------------------------
    #  Aggregation: Build heatmap and detect degraded regions
    # ------------------------------------------------------------------
    print("\n[Aggregation] Building heatmap and detecting degraded regions...")
    composite_bgr, boxes = build_heatmap(img_optimized, ssim_map, params)
    print(f"  >> Degraded regions detected: {len(boxes)}")

    # Save the heatmap composite as a standalone PNG
    heatmap_path = os.path.join(config.OUTPUT_DIR, "heatmap_composite.png")
    cv2.imwrite(heatmap_path, composite_bgr)
    print(f"  >> Heatmap saved: {heatmap_path}")

    # ------------------------------------------------------------------
    #  Reporting: JSON + figure
    # ------------------------------------------------------------------
    print("\n[Reporting] Generating report...")

    json_path = save_json(config.OUTPUT_DIR, ssim_score, ssim_passed,
                          boxes, params)
    print(f"  >> JSON report saved : {json_path}")

    fig_path = save_figure(config.OUTPUT_DIR,
                           img_baseline, img_optimized,
                           composite_bgr,
                           ssim_score, ssim_passed,
                           boxes, params)
    print(f"  >> Figure saved      : {fig_path}")

    # ------------------------------------------------------------------
    #  Final verdict
    # ------------------------------------------------------------------
    print("\n=====================================================")
    if ssim_passed:
        print("  FINAL VERDICT:  ✓  PASS")
        print("  Optimization is within acceptable visual bounds.")
    else:
        print("  FINAL VERDICT:  ✗  FAIL")
        print("  Optimization has introduced perceptual degradation.")
    print("=====================================================")
    print(f"  Output saved to: {config.OUTPUT_DIR}/")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
