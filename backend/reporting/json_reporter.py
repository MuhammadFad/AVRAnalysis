# ====
# json_reporter.py — Structured JSON report generation.
# ====

import json
import os

def save(output_dir: str, results: dict) -> str:
    ssim_score = results["ssim_score"]
    ssim_passed = results["ssim_passed"]
    boxes = results["boxes"]
    params = results["params"]

    report = {
        "overall_result": "PASS" if ssim_passed else "FAIL",
        "ssim_score": round(ssim_score, 4),
        "ssim_threshold": params["ssim_threshold"],
        "ssim_passed": ssim_passed,
        "degraded_regions_count": len(boxes),
        "degraded_region_boxes": [list(b) for b in boxes],
    }

    path = os.path.join(output_dir, "regression_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    return path