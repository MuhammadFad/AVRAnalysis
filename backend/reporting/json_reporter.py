# ================================================================================
# json_reporter.py — Structured Report  (dual path: API dict / CLI file)
# ================================================================================
#
# build_report() — returns a plain Python dict (used by api.py)
# save()         — writes to disk (used by main.py CLI)
#
# ================================================================================

import json
import os
import base64
import io
import cv2
import numpy as np


def _encode_image(img: np.ndarray) -> str:
    """Encode an H×W×3 float64 [0,1] RGB image as a base64 PNG."""
    u8      = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    bgr     = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    return base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ""


def _encode_crop(img: np.ndarray, bbox: tuple) -> str:
    """Crop bbox=(x,y,w,h) from img and return as base64 PNG."""
    x, y, w, h = bbox
    crop = img[y:y+h, x:x+w]
    return _encode_image(crop)


def build_report(results:            dict,
                 resolution_mismatch: bool,
                 orig_dims:           dict,
                 analyzed_dims:       list,
                 img_baseline:        np.ndarray,
                 img_optimized:       np.ndarray,
                 composite:           np.ndarray) -> dict:
    """
    Build the full API response dict.  No disk I/O.

    Parameters
    ----------
    results             : namespaced results dict from pipeline (ssim/edge/color/params)
    resolution_mismatch : bool from acquisition.load_and_align
    orig_dims           : {'baseline': [w,h], 'optimized': [w,h]}
    analyzed_dims       : [w, h] after alignment + cap
    img_baseline        : float64 [0,1] RGB numpy array
    img_optimized       : float64 [0,1] RGB numpy array
    composite           : heatmap composite float64 [0,1] RGB numpy array
    """
    ssim_r  = results["ssim"]
    edge_r  = results["edge"]
    color_r = results["color"]
    params  = results["params"]

    all_passed = ssim_r["passed"] and edge_r["passed"] and color_r["passed"]

    # ---- Regions ----
    serialized_regions = []
    for region in ssim_r.get("boxes", []):
        x, y, w, h = region["bbox"]
        local = region["local_scores"]
        serialized_regions.append({
            "bbox":             [x, y, w, h],
            "passes_failed":    sorted(region["passes_failed"]),
            "local_scores": {
                "ssim":  round(local["ssim"],  4) if local["ssim"]  is not None else None,
                "edge":  round(local["edge"],  4) if local["edge"]  is not None else None,
                "color": round(local["color"], 4) if local["color"] is not None else None,
            },
            "cause_hypothesis": region.get("cause_hypothesis", ""),
            "confidence":       region.get("confidence", ""),
            "cause_tag":        region.get("cause_tag", ""),
            "crops": {
                "baseline":  _encode_crop(img_baseline,  (x, y, w, h)),
                "optimized": _encode_crop(img_optimized, (x, y, w, h)),
            },
        })
    serialized_regions.sort(key=lambda r: r["local_scores"]["ssim"] or 1.0)

    # ---- Pass entries ----
    ssim_entry = {
        "score":   round(ssim_r["score"], 4),
        "verdict": ssim_r["verdict"],
        "intermediates": ssim_r.get("intermediates", {}),
    }
    edge_entry = {
        "score":   round(edge_r["score"], 4),
        "verdict": edge_r["verdict"],
        "intermediates": edge_r.get("intermediates", {}),
    }
    color_entry = {
        "score":   round(color_r["score"], 4),
        "verdict": color_r["verdict"],
        "per_channel_distance": {k: round(v, 4) for k, v in color_r["distances"].items()},
        "intermediates": color_r.get("intermediates", {}),
    }

    return {
        "resolution_mismatch": resolution_mismatch,
        "original_dimensions": orig_dims,
        "analyzed_dimensions": analyzed_dims,
        "passes": {
            "ssim":  ssim_entry,
            "edge":  edge_entry,
            "color": color_entry,
        },
        "regions":         serialized_regions,
        "heatmap":         _encode_image(composite),
        "overall_verdict": "PASS" if all_passed else "FAIL",
        "source_images": {
            "baseline":  _encode_image(img_baseline),
            "optimized": _encode_image(img_optimized),
        },
    }


def save(output_dir: str, results: dict) -> str:
    """CLI path — serialise results to regression_report.json on disk."""
    ssim_r  = results["ssim"]
    edge_r  = results["edge"]
    color_r = results["color"]
    params  = results["params"]

    all_passed = ssim_r["passed"] and edge_r["passed"] and color_r["passed"]

    serialized_regions = []
    for region in ssim_r.get("boxes", []):
        x, y, w, h = region["bbox"]
        local = region["local_scores"]
        serialized_regions.append({
            "bbox":             [x, y, w, h],
            "passes_failed":    sorted(region["passes_failed"]),
            "local_scores": {
                "ssim":  round(local["ssim"],  4) if local["ssim"]  is not None else None,
                "edge":  round(local["edge"],  4) if local["edge"]  is not None else None,
                "color": round(local["color"], 4) if local["color"] is not None else None,
            },
            "cause_hypothesis": region.get("cause_hypothesis", ""),
            "confidence":       region.get("confidence", ""),
            "cause_tag":        region.get("cause_tag", ""),
        })
    serialized_regions.sort(key=lambda r: r["local_scores"]["ssim"] or 1.0)

    report = {
        "overall_result": "PASS" if all_passed else "FAIL",
        "ssim": {
            "score":                  round(ssim_r["score"], 4),
            "threshold":              params["ssim_threshold"],
            "verdict":                ssim_r["verdict"],
            "degraded_regions_count": len(serialized_regions),
        },
        "edge": {
            "score":     round(edge_r["score"], 4),
            "threshold": params["canny_edge_match_thresh"],
            "verdict":   edge_r["verdict"],
        },
        "color": {
            "mean_distance":        round(color_r["score"], 4),
            "threshold":            params["color_distance_thresh"],
            "verdict":              color_r["verdict"],
            "per_channel_distance": {k: round(v, 4) for k, v in color_r["distances"].items()},
        },
        "degraded_regions": serialized_regions,
    }

    path = os.path.join(output_dir, "regression_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path
