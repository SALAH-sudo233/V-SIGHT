"""Per-model bounding-box coordinate-system correction.

Why this exists
---------------
`parse_bbox_output` in eval_11models_refcocog_500_run.py auto-detects ONE
normalization convention:

    if max(abs(x) for x in box) <= 1.5:      # [0,1] floats
        box = [x1*w, y1*h, x2*w, y2*h]

then clamps to the image. That covers llava's `[0.636, 0.108, ...]` but NOT the
`[0,1000]x[0,1000]` integer convention used by Qwen3-VL, Qwen3.5 and
InternVL3.5. Their `[17, 38, 355, 988]` looks like pixels, fails the <=1.5 test,
and gets clamped to the image edge -- collapsing boxes to zero area and forcing
IoU to 0. Measured on 40 images: mean IoU 0.035 raw vs 0.809 rescaled (Qwen3.5),
0.039 vs 0.566 (InternVL3.5).

Critical: rescale from `raw_numbers`, NEVER from `pred_bbox_xyxy`
-----------------------------------------------------------------
`pred_bbox_xyxy` is already clamped, so an out-of-range value like y2=988 has
been destroyed (flattened to the image height) before we see it. `raw_numbers`
carries the pre-clamp values, so it is the only correct input here.

This module does NOT modify the benchmark's parser. The existing 11-model
results stay byte-identical unless a model is explicitly declared `norm_1000`.
"""
from typing import Any, Dict, Optional, Sequence, Tuple

VALID_SYSTEMS = {"pixels", "norm_01", "norm_1000"}


def clamp_bbox(box: Sequence[float], width: int, height: int) -> list:
    """Clamp to image bounds. Mirrors the benchmark's own clamp_bbox."""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return [x1, y1, x2, y2]


def rescale_raw_numbers(raw_numbers: Optional[Sequence[float]],
                        coord_system: str,
                        image_size: Tuple[int, int]) -> Optional[list]:
    """Map pre-clamp model numbers into absolute pixels, then clamp.

    `pixels` and `norm_01` are returned unchanged: the benchmark parser already
    resolved both (norm_01 via its <=1.5 auto-detect), so touching them would
    double-scale.
    """
    if coord_system not in VALID_SYSTEMS:
        raise ValueError(f"Unknown coord_system: {coord_system!r}; "
                         f"expected one of {sorted(VALID_SYSTEMS)}")
    if raw_numbers is None or len(raw_numbers) != 4:
        return None
    if coord_system in ("pixels", "norm_01"):
        return None
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in raw_numbers]
    box = [x1 * w / 1000.0, y1 * h / 1000.0, x2 * w / 1000.0, y2 * h / 1000.0]
    if box[2] <= box[0] or box[3] <= box[1]:      # xywh fallback, as the parser does
        box = [box[0], box[1], box[0] + max(0.0, box[2]), box[1] + max(0.0, box[3])]
    return clamp_bbox(box, w, h)


def apply_coordinate_system(parsed: Dict[str, Any],
                            coord_system: str,
                            image_size: Tuple[int, int]) -> Dict[str, Any]:
    """Return a copy of a parse_bbox_output result with coordinates corrected.

    Adds `bbox_coord_system` (what was declared) and, when a correction happened,
    `bbox_rescaled` plus `pred_bbox_xyxy_preclamp` so the original parser output
    stays auditable in the records.
    """
    out = dict(parsed)
    out["bbox_coord_system"] = coord_system
    out["bbox_rescaled"] = False
    if not parsed.get("pred_found"):
        return out
    fixed = rescale_raw_numbers(parsed.get("raw_numbers"), coord_system, image_size)
    if fixed is None:
        return out
    out["pred_bbox_xyxy_preclamp"] = parsed.get("pred_bbox_xyxy")
    out["pred_bbox_xyxy"] = fixed
    out["bbox_rescaled"] = True
    return out
