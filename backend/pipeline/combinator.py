# ================================================================================
# combinator.py — Phase 4: Truth Table → Cause Hypothesis
# ================================================================================
# Pure function. No image ops, no I/O. Enriches region dicts with:
#   cause_hypothesis, confidence, cause_tag
# ================================================================================

_TRUTH_TABLE = {
    (True,  False, False): ("Texture / surface detail lost. Geometry and lighting intact.", "high"),
    (False, True,  False): ("Likely lighting change shifting edges. Low geometry concern.", "low"),
    (False, False, True):  ("Global lighting / tone shift only. No structural damage.", "high"),
    (True,  True,  False): ("Compression blocking artifact. DCT grid edges detected; perceptual blurring and ringing around blocks.", "very_high"),
    (True,  False, True):  ("Material / shader broke. Surface looks wrong, geometry fine.", "high"),
    (False, True,  True):  ("Lighting rig changed. Edge shift is shadow-driven, not geometry.", "medium"),
    (True,  True,  True):  ("Total visual failure. All three dimensions degraded.", "definitive"),
    (False, False, False): ("No degradation detected.", "clean"),
}

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


def combine(clustered_regions: list,
            ssim_thresh:       float,
            edge_thresh:       float,
            color_thresh:      float) -> list:
    enriched = []
    for region in clustered_regions:
        local = region['local_scores']
        ssim_fails  = local['ssim']  < ssim_thresh   if local['ssim']  is not None else True
        edge_fails  = local['edge']  < edge_thresh   if local['edge']  is not None else False
        color_fails = local['color'] > color_thresh  if local['color'] is not None else False

        key = (ssim_fails, edge_fails, color_fails)
        hypothesis, confidence = _TRUTH_TABLE.get(key, ("Unclassified degradation.", "low"))
        cause_tag = _SHORT_TAGS.get(key, "degraded")

        enriched.append({
            **region,
            "cause_hypothesis": hypothesis,
            "confidence":       confidence,
            "cause_tag":        cause_tag,
        })
    return enriched
