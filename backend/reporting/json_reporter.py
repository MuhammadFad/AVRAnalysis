# ================================================================================
# json_reporter.py — Structured JSON Report Generation
# ================================================================================
#
# PURPOSE:
# This module generates machine-readable JSON reports containing the combined
# results of all three analysis passes (SSIM, edge, color). JSON format enables:
#   - Programmatic parsing by other tools
#   - Integration with CI/CD pipelines
#   - Historical comparison and trend analysis
#   - Automated pass/fail decisions per-metric
#
# ROLE IN PIPELINE:
# Receives the assembled namespaced results dictionary from main.py and
# serializes it into a single JSON file covering all three passes.
#
# OUTPUT FORMAT:
#   - overall_result:  "PASS" (all passes passed) or "FAIL"
#   - ssim:            Score, threshold, pass/fail, tagged degraded region boxes
#   - edge:            Score, threshold, pass/fail
#   - color:           Mean distance, per-channel distances, threshold, pass/fail
#
# Each bounding box in ssim.degraded_region_boxes includes a 'filters' list
# indicating which passes flagged that specific region, e.g. ["ssim", "edge"].
#
# ================================================================================

import json
import os


def save(output_dir: str, results: dict) -> str:
    """
    Generate and save a structured JSON report covering all analysis passes.

    INPUTS:
        output_dir: Directory where the JSON file will be written
        results:    Namespaced results dictionary from main.py with keys:
                        'ssim':  { score, passed, boxes (tagged), ... }
                        'edge':  { score, passed, ... }
                        'color': { score, passed, distances, ... }
                        'params': flat configuration dictionary

    RETURNS:
        str: File path where the JSON was written

    FILE OUTPUT:
        {output_dir}/regression_report.json
    """
    ssim_r  = results["ssim"]
    edge_r  = results["edge"]
    color_r = results["color"]
    params  = results["params"]

    all_passed = ssim_r["passed"] and edge_r["passed"] and color_r["passed"]

    # Serialize tagged boxes — each entry carries its geometry and the list of
    # filters that flagged it, making the JSON independently useful for tooling
    # that wants to know not just *where* degradation is but *why*.
    serialized_boxes = []
    for box in ssim_r.get("boxes", []):
        if len(box) == 5:
            x, y, w, h, flags = box
        else:
            x, y, w, h = box
            flags = frozenset({'ssim'})

        serialized_boxes.append({
            "x": x, "y": y, "w": w, "h": h,
            # Sort for deterministic output across runs.
            "filters": sorted(flags),
        })

    report = {
        # ----------------------------------------------------------------
        # Top-level verdict
        # ----------------------------------------------------------------
        "overall_result": "PASS" if all_passed else "FAIL",

        # ----------------------------------------------------------------
        # SSIM pass
        # ----------------------------------------------------------------
        "ssim": {
            "score":                  round(ssim_r["score"], 4),
            "threshold":              params["ssim_threshold"],
            "passed":                 ssim_r["passed"],
            "degraded_regions_count": len(serialized_boxes),
            "degraded_region_boxes":  serialized_boxes,
        },

        # ----------------------------------------------------------------
        # Edge pass
        # ----------------------------------------------------------------
        "edge": {
            "score":     round(edge_r["score"], 4),
            "threshold": params["canny_edge_match_thresh"],
            "passed":    edge_r["passed"],
        },

        # ----------------------------------------------------------------
        # Color pass
        # ----------------------------------------------------------------
        "color": {
            "mean_distance":        round(color_r["score"], 4),
            "threshold":            params["color_distance_thresh"],
            "passed":               color_r["passed"],
            "per_channel_distance": {
                k: round(v, 4) for k, v in color_r["distances"].items()
            },
        },
    }

    path = os.path.join(output_dir, "regression_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    return path
