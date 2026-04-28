#!/usr/bin/env python3
# ===============
# main.py — Entry point for the Visual Regression Analyzer
# ===============

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
    if len(sys.argv) == 3:
        baseline_path = sys.argv[1]
        optimized_path = sys.argv[2]
    else:
        baseline_path = "./data/baseline.png"
        optimized_path = "./data/optimized.png"

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    params = {
        "ssim_threshold": config.SSIM_THRESHOLD,
        "heatmap_alpha": config.HEATMAP_ALPHA,
        "degradation_thresh": config.DEGRADATION_THRESH,
        "morph_radius": config.MORPH_RADIUS,
        "min_region_area": config.MIN_REGION_AREA,
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