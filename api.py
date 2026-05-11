# ================================================================================
# api.py — FastAPI Web API
# ================================================================================
#
# GET  /ping    — wake endpoint, returns {"status": "alive"}
# POST /analyze — accepts baseline + optimized image uploads, runs full pipeline,
#                 returns structured JSON report with base64 encoded images
#
# ================================================================================

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import backend.config as config
from backend.pipeline.acquisition   import load_and_align
from backend.pipeline.ssim_pass     import run as run_ssim
from backend.pipeline.edge_pass     import run as run_edge
from backend.pipeline.color_pass    import run as run_color
from backend.pipeline.aggregator    import build_heatmap
from backend.pipeline.combinator    import combine as combine_regions
from backend.reporting.json_reporter   import build_report
from backend.reporting.visual_reporter import render_heatmap_b64

app = FastAPI(title="AVR — Visual Regression Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://avr-analysis.vercel.app", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"status": "alive"}


@app.post("/analyze")
async def analyze(
    baseline:  UploadFile = File(...),
    optimized: UploadFile = File(...),
):
    # ---- Read upload bytes ----
    try:
        b_bytes = await baseline.read()
        o_bytes = await optimized.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {e}")

    # ---- Phase 1: Acquisition ----
    try:
        img_b, img_o, mismatch, orig_dims, analyzed_dims = load_and_align(b_bytes, o_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ---- Parameter bundle ----
    params = {
        "ssim_threshold":          config.SSIM_THRESHOLD,
        "ssim_window_size":        config.SSIM_WINDOW_SIZE,
        "ssim_sigma":              config.SSIM_SIGMA,
        "heatmap_alpha":           config.HEATMAP_ALPHA,
        "degradation_thresh":      config.DEGRADATION_THRESH,
        "morph_radius":            config.MORPH_RADIUS,
        "min_region_area":         config.MIN_REGION_AREA,
        "canny_blur_kernel":       config.CANNY_BLUR_KERNEL,
        "canny_blur_sigma":        config.CANNY_BLUR_SIGMA,
        "canny_low_thresh":        config.CANNY_LOW_THRESH,
        "canny_high_thresh":       config.CANNY_HIGH_THRESH,
        "canny_edge_match_thresh": config.CANNY_EDGE_MATCH_THRESH,
        "color_hist_bins":         config.COLOR_HIST_BINS,
        "color_distance_thresh":   config.COLOR_DISTANCE_THRESH,
        "iou_threshold":           config.IOU_THRESHOLD,
        "top_regions_in_report":   config.TOP_REGIONS_IN_REPORT,
        "save_intermediates":      True,   # always generate intermediates for API
    }

    # ---- Phase 2a: SSIM ----
    ssim_result = run_ssim(img_b, img_o, params)
    ssim_result["passed"]  = ssim_result["score"] >= config.SSIM_THRESHOLD
    ssim_result["verdict"] = "PASS" if ssim_result["passed"] else "FAIL"

    # ---- Phase 2b: Edge ----
    edge_result = run_edge(img_b, img_o, params)

    # ---- Phase 2c: Color ----
    color_result = run_color(img_b, img_o, params)

    # ---- Phase 3: Aggregation ----
    composite, combined_regions = build_heatmap(
        img_b, img_o,
        ssim_result["ssim_map"],
        edge_result.get("edge_baseline"),
        edge_result.get("edge_optimized"),
        params,
    )

    # ---- Phase 4: Combinator ----
    enriched_regions = combine_regions(
        combined_regions,
        ssim_thresh=config.SSIM_THRESHOLD,
        edge_thresh=config.CANNY_EDGE_MATCH_THRESH,
        color_thresh=config.COLOR_DISTANCE_THRESH,
    )
    ssim_result["boxes"] = enriched_regions

    results = {
        "ssim":   ssim_result,
        "edge":   edge_result,
        "color":  color_result,
        "params": params,
    }

    # ---- Reporting ----
    report = build_report(
        results        = results,
        resolution_mismatch = mismatch,
        orig_dims      = orig_dims,
        analyzed_dims  = analyzed_dims,
        img_baseline   = img_b,
        img_optimized  = img_o,
        composite      = composite,
    )

    return report
