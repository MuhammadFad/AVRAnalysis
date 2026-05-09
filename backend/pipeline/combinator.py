# ================================================================================
# combinator.py — Semantic Cause Hypothesis from Multi-Pass Region Attribution
# ================================================================================
#
# PURPOSE:
# This module applies a deterministic truth table to the per-region pass
# attribution produced by aggregator.py, assigning a human-readable cause
# hypothesis and confidence level to each degraded region.
#
# ROLE IN PIPELINE:
# Sits between the aggregator (spatial logic) and the reporters (output).
# It is a pure function — no spatial math, no image operations, no I/O.
# Input:  list of combined region dicts from aggregator.py
# Output: same list, each dict enriched with 'cause_hypothesis' and 'confidence'
#
# TRUTH TABLE:
# The cause hypothesis is determined by which combination of passes fired on
# the region. SSIM is always present (regions exist because SSIM flagged them).
#
#  SSIM | Edge | Color | Cause hypothesis                                  | Confidence
#  -----+------+-------+---------------------------------------------------+-----------
#  FAIL | pass | pass  | Texture / surface detail lost.                    | high
#       |      |       | Geometry and lighting intact.                     |
#  -----+------+-------+---------------------------------------------------+-----------
#  pass | FAIL | pass  | Likely lighting change shifting edges.             | low
#       |      |       | Low geometry concern.                             |
#  -----+------+-------+---------------------------------------------------+-----------
#  pass | pass | FAIL  | Global lighting / tone shift only.                | high
#       |      |       | No structural damage.                             |
#  -----+------+-------+---------------------------------------------------+-----------
#  FAIL | FAIL | pass  | Geometry decimated. Detail and silhouette both    | very_high
#       |      |       | degraded. (Compression: DCT blocking artifact —   |
#       |      |       | Canny picks up grid edges, SSIM sees blurring.)   |
#  -----+------+-------+---------------------------------------------------+-----------
#  FAIL | pass | FAIL  | Material / shader broke. Surface looks wrong,     | high
#       |      |       | geometry fine.                                    |
#  -----+------+-------+---------------------------------------------------+-----------
#  pass | FAIL | FAIL  | Lighting rig changed. Edge shift is shadow-       | medium
#       |      |       | driven, not geometry.                             |
#  -----+------+-------+---------------------------------------------------+-----------
#  FAIL | FAIL | FAIL  | Total visual failure. All three dimensions        | definitive
#       |      |       | degraded.                                         |
#  -----+------+-------+---------------------------------------------------+-----------
#  pass | pass | pass  | No degradation detected.                          | clean
#
# SPECIAL CASE — Video compression blocking:
#   When SSIM and Edge both fail on the same region (with or without color),
#   the cause hypothesis is overridden to note "compression blocking artifact"
#   explicitly. DCT blocking produces a grid of sharp edges that Canny detects,
#   while the perceptual blurring/ringing around blocks drives SSIM down.
#
# NOTE on Canny-only (Edge FAIL, SSIM pass):
#   This combination carries low confidence because lighting changes shift edge
#   positions without causing real geometry damage. The moment SSIM also fires
#   on the same region, confidence jumps — both passes independently agreeing
#   is a much stronger signal.
#
# ================================================================================


# ================================================================================
# Truth table definition
# ================================================================================

# Each entry: (ssim_fails, edge_fails, color_fails) → (hypothesis, confidence)
#
# Key note on the "pass" rows: since aggregator only creates boxes where SSIM
# failed globally, the (pass, FAIL, pass) and (pass, pass, FAIL) rows apply
# when local SSIM score happens to be above threshold for a sub-region while
# edge or color still fails locally. These are edge cases but handled for
# completeness and schema consistency.

_TRUTH_TABLE = {
    # (ssim_fail, edge_fail, color_fail): (hypothesis, confidence)
    (True,  False, False): (
        "Texture / surface detail lost. Geometry and lighting intact.",
        "high",
    ),
    (False, True,  False): (
        "Likely lighting change shifting edges. Low geometry concern.",
        "low",
    ),
    (False, False, True): (
        "Global lighting / tone shift only. No structural damage.",
        "high",
    ),
    (True,  True,  False): (
        # SSIM + Edge together on compression content → blocking artifact.
        # Overridden at call time with more specific wording when appropriate.
        "Compression blocking artifact. DCT grid edges detected; "
        "perceptual blurring and ringing around blocks.",
        "very_high",
    ),
    (True,  False, True): (
        "Material / shader broke. Surface looks wrong, geometry fine.",
        "high",
    ),
    (False, True,  True): (
        "Lighting rig changed. Edge shift is shadow-driven, not geometry.",
        "medium",
    ),
    (True,  True,  True): (
        "Total visual failure. All three dimensions degraded.",
        "definitive",
    ),
    (False, False, False): (
        "No degradation detected.",
        "clean",
    ),
}

# Short display tags — used by visual_reporter to label boxes on the heatmap.
# Kept terse so they fit inside small bounding boxes without overlap.
_SHORT_TAGS = {
    (True,  False, False): "texture loss",
    (False, True,  False): "edge shift",
    (False, False, True):  "tone shift",
    (True,  True,  False): "blocking artifact",
    (True,  False, True):  "shader/material",
    (False, True,  True):  "lighting rig",
    (True,  True,  True):  "total failure",
    (False, False, False): "clean",
}


# ================================================================================
# Public interface
# ================================================================================

def combine(clustered_regions: list,
            ssim_thresh:       float,
            edge_thresh:       float,
            color_thresh:      float) -> list:
    """
    Apply truth-table logic to each combined region and enrich with cause
    hypothesis and confidence.

    This is a pure function — it performs no image operations, no spatial math,
    and no I/O. It only reads the 'passes_failed' and 'local_scores' fields of
    each region dict and writes back 'cause_hypothesis', 'confidence', and
    'cause_tag' (the short label for visual annotation).

    INPUTS:
        clustered_regions: List of combined region dicts from aggregator.py.
                           Each must contain:
                               'bbox':          (x, y, w, h)
                               'passes_failed': list of pass name strings
                               'local_scores':  {'ssim': float, 'edge': float|None,
                                                 'color': float|None}
        ssim_thresh:  SSIM pass threshold (score below this = local SSIM fail)
        edge_thresh:  Edge match threshold (score below this = local edge fail)
        color_thresh: Bhattacharyya threshold (score above this = local color fail)

    RETURNS:
        The same list with each region dict enriched with:
            'cause_hypothesis': str  — plain-English description of likely cause
            'confidence':       str  — one of: low, medium, high, very_high,
                                       definitive, clean
            'cause_tag':        str  — short label for visual annotation on heatmap
    """
    enriched = []

    for region in clustered_regions:
        local   = region['local_scores']
        failed  = set(region['passes_failed'])

        # Determine which passes failed locally.
        # 'ssim' is always in passes_failed (aggregator guarantee), but we
        # re-derive from local scores for precise threshold comparison.
        ssim_fails  = local['ssim']  < ssim_thresh   if local['ssim']  is not None else True
        edge_fails  = local['edge']  < edge_thresh   if local['edge']  is not None else False
        color_fails = local['color'] > color_thresh  if local['color'] is not None else False

        key = (ssim_fails, edge_fails, color_fails)

        hypothesis, confidence = _TRUTH_TABLE.get(key, (
            "Unclassified degradation.",
            "low",
        ))
        cause_tag = _SHORT_TAGS.get(key, "degraded")

        enriched.append({
            **region,
            "cause_hypothesis": hypothesis,
            "confidence":       confidence,
            "cause_tag":        cause_tag,
        })

    return enriched
