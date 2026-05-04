# ================================================================================
# json_reporter.py — Structured JSON Report Generation
# ================================================================================
#
# PURPOSE:
# This module generates machine-readable JSON reports containing all analysis
# results. JSON format enables:
#   - Programmatic parsing by other tools
#   - Integration with CI/CD pipelines
#   - Historical comparison and trend analysis
#   - Automated pass/fail decisions
#
# ROLE IN PIPELINE:
# This is the structured reporting component. It receives all analysis results
# and serializes them to a JSON file for:
#   - Build system integration (fail if score < threshold)
#   - Analytics dashboards
#   - Regression tracking
#   - Debugging (bounding boxes of failed regions)
#
# OUTPUT FORMAT:
# The generated JSON contains:
#   - overall_result: "PASS" or "FAIL"
#   - ssim_score: The overall SSIM similarity score
#   - ssim_threshold: The configured threshold
#   - ssim_passed: Boolean pass/fail
#   - degraded_regions_count: Number of detected degraded areas
#   - degraded_region_boxes: List of bounding boxes [[x,y,w,h], ...]
#
# ================================================================================

import json
import os

def save(output_dir: str, results: dict) -> str:
    """
    Generate and save a structured JSON report of analysis results.
    
    INPUTS:
        output_dir: Directory where JSON file will be written
        results: Dictionary containing:
            - ssim_score: Overall SSIM value
            - ssim_passed: Boolean pass/fail
            - boxes: List of bounding boxes
            - params: Configuration parameters (including thresholds)
    
    RETURNS:
        str: File path where JSON was written
    
    FILE OUTPUT:
        {output_dir}/regression_report.json
        Contains human-readable JSON with 2-space indentation
    """
    # Extract all relevant analysis results
    ssim_score = results["ssim_score"]
    ssim_passed = results["ssim_passed"]
    boxes = results["boxes"]
    params = results["params"]

    # ========================================================================
    # Build structured report dictionary
    # ========================================================================
    report = {
        # Overall verdict: PASS or FAIL (pass if ssim_score ≥ threshold)
        "overall_result": "PASS" if ssim_passed else "FAIL",
        
        # Raw SSIM score (0.0 to 1.0, rounded to 4 decimal places)
        # This is the primary metric: higher is better
        "ssim_score": round(ssim_score, 4),
        
        # The threshold used for pass/fail decision
        # Results: if score ≥ threshold → PASS, else → FAIL
        "ssim_threshold": params["ssim_threshold"],
        
        # Boolean flag for easier scripting (no need to parse strings)
        "ssim_passed": ssim_passed,
        
        # Count of detected degradation hotspots
        # 0 means no significant degraded regions found
        "degraded_regions_count": len(boxes),
        
        # Bounding boxes of degraded regions as [x, y, width, height] tuples
        # x, y: top-left corner in image coordinates
        # width, height: dimensions in pixels
        # Empty list [] means no degradation detected
        "degraded_region_boxes": [list(b) for b in boxes],
    }

    # ========================================================================
    # Write report to JSON file
    # ========================================================================
    path = os.path.join(output_dir, "regression_report.json")
    with open(path, 'w') as f:
        # Use 2-space indentation for human readability
        json.dump(report, f, indent=2)

    return path