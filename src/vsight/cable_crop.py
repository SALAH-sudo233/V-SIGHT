"""Optional gray-zone crop extension for CCV-CABLE.

This module is intentionally separate from the strict one-forward verifier.
It may execute at most two additional detector forwards and never invokes the
upstream MLLM.  Results therefore cannot be mixed silently with the mainline
single-pass evaluation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .ccv import CCVResult, ClaimConditionedCounterfactualVerifier, DetectorEvidence


Box = tuple[float, float, float, float]


def _expand(box: Sequence[float], scale: float, width: int, height: int) -> Box:
    x1, y1, x2, y2 = (float(value) for value in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_width, half_height = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    return (
        max(0.0, cx - half_width),
        max(0.0, cy - half_height),
        min(float(width), cx + half_width),
        min(float(height), cy + half_height),
    )


def _union(left: Sequence[float], right: Sequence[float]) -> Box:
    return (
        min(float(left[0]), float(right[0])),
        min(float(left[1]), float(right[1])),
        max(float(left[2]), float(right[2])),
        max(float(left[3]), float(right[3])),
    )


def _fit_extent(center: float, extent: float, limit: int) -> tuple[float, float]:
    extent = min(float(limit), max(1.0, float(extent)))
    start = center - extent / 2
    end = center + extent / 2
    if start < 0:
        end -= start
        start = 0.0
    if end > limit:
        start -= end - limit
        end = float(limit)
    return max(0.0, start), min(float(limit), end)


def _safe_aspect(
    box: Sequence[float],
    width: int,
    height: int,
    *,
    minimum: float,
    maximum: float,
) -> Box:
    """Expand a crop to avoid degenerate detector feature maps.

    GroundingDINO uses a fixed proposal top-k.  Extremely thin crops can yield
    fewer encoder tokens than that top-k after resize.  Expanding only the
    narrow dimension keeps the requested ROI and adds context without padding
    artificial pixels or changing coordinate transforms.
    """

    x1, y1, x2, y2 = (float(value) for value in box)
    crop_width, crop_height = x2 - x1, y2 - y1
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop must have positive extent")
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    aspect = crop_width / crop_height
    if aspect < minimum:
        crop_width = crop_height * minimum
    elif aspect > maximum:
        crop_height = crop_width / maximum
    safe_x1, safe_x2 = _fit_extent(center_x, crop_width, width)
    safe_y1, safe_y2 = _fit_extent(center_y, crop_height, height)
    return safe_x1, safe_y1, safe_x2, safe_y2


def _local(box: Sequence[float], crop: Box) -> Box:
    return (
        float(box[0]) - crop[0],
        float(box[1]) - crop[1],
        float(box[2]) - crop[0],
        float(box[3]) - crop[1],
    )


def _global(box: Sequence[float] | None, crop: Box) -> Box | None:
    if box is None:
        return None
    return (
        float(box[0]) + crop[0],
        float(box[1]) + crop[1],
        float(box[2]) + crop[0],
        float(box[3]) + crop[1],
    )


class ConditionalCropExtension:
    """Run target and target/reference-union crops only for gray-zone rows."""

    def __init__(
        self,
        *,
        target_scale: float = 1.5,
        union_scale: float = 1.25,
        minimum_crop_aspect: float = 0.75,
        maximum_crop_aspect: float = 4 / 3,
    ) -> None:
        if target_scale < 1 or union_scale < 1:
            raise ValueError("crop expansion scales must be at least one")
        if not 0 < minimum_crop_aspect <= maximum_crop_aspect:
            raise ValueError("crop aspect bounds must be positive and ordered")
        self.target_scale = float(target_scale)
        self.union_scale = float(union_scale)
        self.minimum_crop_aspect = float(minimum_crop_aspect)
        self.maximum_crop_aspect = float(maximum_crop_aspect)

    def verify(
        self,
        *,
        image,
        query: str,
        upstream_bbox: Sequence[float],
        initial_result: CCVResult,
        initial_evidence: DetectorEvidence,
        detector,
        verifier: ClaimConditionedCounterfactualVerifier,
    ) -> CCVResult:
        if initial_result.binding_status != "BINDING_UNCERTAIN":
            return initial_result
        width, height = image.size
        crops = [_expand(upstream_bbox, self.target_scale, width, height)]
        references = sorted(
            (row for row in initial_evidence.proposals if row.is_reference),
            key=lambda row: (-row.score, row.proposal_id),
        )
        if references:
            union = _union(upstream_bbox, references[0].box)
            crops.append(_expand(union, self.union_scale, width, height))
        crops = [
            _safe_aspect(
                crop,
                width,
                height,
                minimum=self.minimum_crop_aspect,
                maximum=self.maximum_crop_aspect,
            )
            for crop in crops[:2]
        ]
        selected = initial_result
        total_detector_ms = float(initial_result.latency_metadata.get("detector_ms") or 0.0)
        additional_detector_ms = 0.0
        pass_metadata: dict[str, float | int | bool] = {}
        used = 0
        for crop_index, crop in enumerate(crops, start=1):
            pixel_crop = tuple(int(round(value)) for value in crop)
            cropped_image = image.crop(pixel_crop)
            actual_crop = tuple(float(value) for value in pixel_crop)
            _, evidence = detector.infer(cropped_image, query)
            local_upstream = _local(upstream_bbox, actual_crop)
            candidate = verifier.verify(None, query, local_upstream, evidence)
            used += 1
            pass_ms = float(evidence.latency_ms or 0.0)
            additional_detector_ms += pass_ms
            total_detector_ms += pass_ms
            pass_metadata.update({
                f"conditional_crop_pass_{crop_index}_ms": pass_ms,
                f"conditional_crop_pass_{crop_index}_width": int(cropped_image.size[0]),
                f"conditional_crop_pass_{crop_index}_height": int(cropped_image.size[1]),
            })
            candidate = replace(
                candidate,
                corrected_bbox=_global(candidate.corrected_bbox, actual_crop),
                original_bbox=_global(candidate.original_bbox, actual_crop),
                alternative_bbox=_global(candidate.alternative_bbox, actual_crop),
                latency_metadata={
                    **candidate.latency_metadata,
                    "detector_ms": total_detector_ms,
                    "image_encoder_forwards": 1 + used,
                    "conditional_crop": True,
                    "conditional_crop_pass": crop_index,
                    "conditional_crop_additional_detector_ms": additional_detector_ms,
                    **pass_metadata,
                    "crop_x1": crop[0],
                    "crop_y1": crop[1],
                },
                reason=f"conditional_crop_{crop_index}:{candidate.reason}",
            )
            if candidate.binding_status != "BINDING_UNCERTAIN":
                selected = candidate
                break
        if selected is initial_result:
            selected = replace(
                initial_result,
                latency_metadata={
                    **initial_result.latency_metadata,
                    "detector_ms": total_detector_ms,
                    "image_encoder_forwards": 1 + used,
                    "conditional_crop": bool(used),
                    "conditional_crop_pass": used,
                    "conditional_crop_additional_detector_ms": additional_detector_ms,
                    **pass_metadata,
                },
                reason="conditional_crops_remain_uncertain_preserve_upstream",
            )
        return selected
