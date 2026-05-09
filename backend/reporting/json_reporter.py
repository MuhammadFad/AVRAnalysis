# ================================================================================
# json_reporter.py — Structured JSON Report Generation
# ================================================================================
#
# PURPOSE:
# This module generates machine-readable JSON reports containing the combined
# results of all three analysis passes (SSIM, edge, color) plus per-region
# cause hypotheses from the combinator. JSON format enables:
#   - Programmatic parsing by other tools
#   - Integration with CI/CD pipelines
#   - Historical comparison and trend analysis
#   - Automated pass/fail decisions per-metric
#
# ROLE IN PIPELINE:
# Receives the assembled namespaced results dictionary from main.py and
# serializes it into a single JSON file covering all three passes plus the
# enriched combined regions produced by aggregator + combinator.
#
# OUTPUT FORMAT:
#   - overall_result:    "PASS" (all passes passed) or "FAIL"
#   - ssim:              Score, threshold, verdict, degraded region count
#   - edge:              Score, threshold, verdict
#   - color:             Mean distance, per-channel distances, threshold, verdict
#   - degraded_regions:  List of enriched region entries, each containing:
#       · bbox:             [x, y, w, h] in pixel coords
#       · passes_failed:    which passes fired for this region (e.g. ["ssim","edge"])
#       · local_scores:     per-pass local quality score within the bbox
#       · cause_hypothesis: plain-English description of the likely artifact cause
#       · confidence:       how certain the combinator is (low/medium/high/
#                           very_high/definitive/clean)
#       · cause_tag:        short label (matches visual annotation on heatmap)
#
# ================================================================================

import json
import os


def save(output_dir: str, results: dict) -> str:
    """
    Generate and save a structured JSON report covering all analysis passes
    and per-region cause hypotheses.

    INPUTS:
        output_dir: Directory where the JSON file will be written
        results:    Namespaced results dictionary from main.py with keys:
                        'ssim':    { score, passed, verdict, boxes (enriched), ... }
                        'edge':    { score, passed, verdict, ... }
                        'color':   { score, passed, verdict, distances, ... }
                        'params':  flat configuration dictionary

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

    # =========================================================================
    # Serialize enriched combined regions
    # =========================================================================
    # Each region carries spatial attribution (which passes fired), local quality
    # scores (how bad it is per pass), and the combinator's cause hypothesis.
    # This makes the JSON independently useful for any downstream tooling that
    # wants to understand not just *where* degradation is, but *what caused it*.
    serialized_regions = []
    for region in ssim_r.get("boxes", []):
        x, y, w, h = region["bbox"]
        local       = region["local_scores"]

        entry = {
            "bbox":             [x, y, w, h],
            "passes_failed":    sorted(region["passes_failed"]),  # deterministic order
            "local_scores": {
                "ssim":  round(local["ssim"],  4) if local["ssim"]  is not None else None,
                "edge":  round(local["edge"],  4) if local["edge"]  is not None else None,
                "color": round(local["color"], 4) if local["color"] is not None else None,
            },
            "cause_hypothesis": region.get("cause_hypothesis", ""),
            "confidence":       region.get("confidence", ""),
            "cause_tag":        region.get("cause_tag", ""),
        }
        serialized_regions.append(entry)

    # Sort regions by severity (lowest local SSIM = most degraded first)
    # so the JSON reads from worst to best without requiring post-processing.
    serialized_regions.sort(key=lambda r: r["local_scores"]["ssim"] or 1.0)

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
            "verdict":                ssim_r["verdict"],
            "degraded_regions_count": len(serialized_regions),
        },

        # ----------------------------------------------------------------
        # Edge pass
        # ----------------------------------------------------------------
        "edge": {
            "score":     round(edge_r["score"], 4),
            "threshold": params["canny_edge_match_thresh"],
            "verdict":   edge_r["verdict"],
        },

        # ----------------------------------------------------------------
        # Color pass
        # ----------------------------------------------------------------
        "color": {
            "mean_distance":        round(color_r["score"], 4),
            "threshold":            params["color_distance_thresh"],
            "verdict":              color_r["verdict"],
            "per_channel_distance": {
                k: round(v, 4) for k, v in color_r["distances"].items()
            },
        },

        # ----------------------------------------------------------------
        # Enriched degraded regions (sorted worst-first)
        # ----------------------------------------------------------------
        "degraded_regions": serialized_regions,
    }

    path = os.path.join(output_dir, "regression_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    return path
