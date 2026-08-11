"""Inference-only utilities for the single-MLLM-pass postprocessor."""

from __future__ import annotations

import hashlib
import re
from typing import Mapping, Sequence

from .e2_verifier import box_iou
from .relation_supervision import (
    CATEGORY_ALIASES,
    RELATION_PATTERNS,
    category_mentions,
    extract_reference_phrase,
)


PROPOSAL_SUPPORT_FIELDS = (
    "proposal_count",
    "max_score",
    "mean_score",
    "max_candidate_iou",
    "score_weighted_candidate_iou",
    "top_score_candidate_iou",
    "best_iou_score",
    "max_score_iou_product",
    "overlap_count_010",
    "max_overlap_score_010",
    "overlap_count_030",
    "max_overlap_score_030",
    "overlap_count_050",
    "max_overlap_score_050",
)

REFERENCE_GEOMETRY_FIELDS = (
    "valid",
    "score_weighted_dx",
    "score_weighted_dy",
    "score_weighted_abs_dx",
    "score_weighted_abs_dy",
    "score_weighted_distance",
    "min_distance",
    "max_iou",
    "score_weighted_iou",
    "score_weighted_x_overlap",
    "score_weighted_y_overlap",
    "score_weighted_candidate_cover",
    "score_weighted_reference_cover",
    "candidate_left_fraction",
    "candidate_above_fraction",
    "horizontal_both_sides",
    "vertical_both_sides",
)


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def semantic_query_id(image_filename: str, query: str) -> str:
    payload = f"{image_filename}\0{normalized_text(query)}".encode("utf-8")
    return "singlepass:" + hashlib.sha256(payload).hexdigest()[:24]


def target_phrases(query: str) -> dict[str, str | None]:
    """Extract a category head and a modifier-aware target phrase.

    The function is deliberately label-free. It only uses the deployment query
    and the frozen COCO alias vocabulary used elsewhere in V-SIGHT.
    """

    text = normalized_text(query)
    mentions = category_mentions(text, tuple(CATEGORY_ALIASES))
    head = mentions[0].category if mentions else None
    matches = []
    for _, pattern in RELATION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            matches.append(match)
    end = min((match.start() for match in matches), default=len(text))
    full = text[:end].strip(" \t.,;:!?()[]{}\"")
    full = re.sub(r"^(?:the|a|an|this|that)\s+", "", full)
    full = full or text or None
    return {
        "head": head,
        "full": full,
        "reference": extract_reference_phrase(text),
    }


def proposal_support_features(
    candidate_box: Sequence[float] | None,
    proposals: Sequence[Mapping],
) -> dict[str, float]:
    """Summarize detector support without using labels or GT."""

    scores = [max(0.0, float(row.get("score") or 0.0)) for row in proposals]
    overlaps = [0.0] * len(proposals)
    if candidate_box is not None:
        for index, proposal in enumerate(proposals):
            try:
                overlaps[index] = box_iou(candidate_box, proposal["bbox_xyxy"])
            except (KeyError, TypeError, ValueError):
                overlaps[index] = 0.0
    total = sum(scores)
    weighted_overlap = (
        sum(score * overlap for score, overlap in zip(scores, overlaps, strict=True))
        / total
        if total > 0 and overlaps
        else 0.0
    )
    if scores and overlaps:
        top_score_index = max(range(len(scores)), key=scores.__getitem__)
        best_iou_index = max(range(len(overlaps)), key=overlaps.__getitem__)
        top_score_overlap = overlaps[top_score_index]
        best_iou_score = scores[best_iou_index]
        max_joint_support = max(
            score * overlap
            for score, overlap in zip(scores, overlaps, strict=True)
        )
    else:
        top_score_overlap = 0.0
        best_iou_score = 0.0
        max_joint_support = 0.0
    result = {
        "proposal_count": float(len(proposals)),
        "max_score": max(scores, default=0.0),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "max_candidate_iou": max(overlaps, default=0.0),
        "score_weighted_candidate_iou": weighted_overlap,
        "top_score_candidate_iou": top_score_overlap,
        "best_iou_score": best_iou_score,
        "max_score_iou_product": max_joint_support,
    }
    for suffix, threshold in (("010", 0.1), ("030", 0.3), ("050", 0.5)):
        supported_scores = [
            score
            for score, overlap in zip(scores, overlaps, strict=True)
            if overlap >= threshold
        ]
        result[f"overlap_count_{suffix}"] = float(len(supported_scores))
        result[f"max_overlap_score_{suffix}"] = max(supported_scores, default=0.0)
    return result


def candidate_reference_geometry_features(
    candidate_box: Sequence[float] | None,
    proposals: Sequence[Mapping],
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    """Aggregate label-free target/reference geometry for relation verification."""

    zeros = {name: 0.0 for name in REFERENCE_GEOMETRY_FIELDS}
    if candidate_box is None or image_width <= 0 or image_height <= 0:
        return zeros
    try:
        cx1, cy1, cx2, cy2 = (float(value) for value in candidate_box)
    except (TypeError, ValueError):
        return zeros
    if cx2 <= cx1 or cy2 <= cy1:
        return zeros

    candidate_width = cx2 - cx1
    candidate_height = cy2 - cy1
    candidate_area = candidate_width * candidate_height
    candidate_center_x = (cx1 + cx2) / 2
    candidate_center_y = (cy1 + cy2) / 2
    rows = []
    for proposal in proposals:
        try:
            rx1, ry1, rx2, ry2 = (
                float(value) for value in proposal["bbox_xyxy"]
            )
            score = max(0.0, float(proposal.get("score") or 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        reference_width = rx2 - rx1
        reference_height = ry2 - ry1
        reference_area = reference_width * reference_height
        dx = (candidate_center_x - (rx1 + rx2) / 2) / image_width
        dy = (candidate_center_y - (ry1 + ry2) / 2) / image_height
        intersection_width = max(0.0, min(cx2, rx2) - max(cx1, rx1))
        intersection_height = max(0.0, min(cy2, ry2) - max(cy1, ry1))
        intersection = intersection_width * intersection_height
        union = candidate_area + reference_area - intersection
        rows.append(
            {
                "score": score,
                "dx": dx,
                "dy": dy,
                "abs_dx": abs(dx),
                "abs_dy": abs(dy),
                "distance": (dx * dx + dy * dy) ** 0.5,
                "iou": intersection / union if union > 0 else 0.0,
                "x_overlap": intersection_width
                / max(min(candidate_width, reference_width), 1e-6),
                "y_overlap": intersection_height
                / max(min(candidate_height, reference_height), 1e-6),
                "candidate_cover": intersection / candidate_area,
                "reference_cover": intersection / reference_area,
            }
        )
    if not rows:
        return zeros

    total_score = sum(row["score"] for row in rows)
    weights = (
        [row["score"] / total_score for row in rows]
        if total_score > 0
        else [1.0 / len(rows)] * len(rows)
    )

    def weighted(name: str) -> float:
        return sum(
            weight * row[name] for weight, row in zip(weights, rows, strict=True)
        )

    return {
        "valid": 1.0,
        "score_weighted_dx": weighted("dx"),
        "score_weighted_dy": weighted("dy"),
        "score_weighted_abs_dx": weighted("abs_dx"),
        "score_weighted_abs_dy": weighted("abs_dy"),
        "score_weighted_distance": weighted("distance"),
        "min_distance": min(row["distance"] for row in rows),
        "max_iou": max(row["iou"] for row in rows),
        "score_weighted_iou": weighted("iou"),
        "score_weighted_x_overlap": weighted("x_overlap"),
        "score_weighted_y_overlap": weighted("y_overlap"),
        "score_weighted_candidate_cover": weighted("candidate_cover"),
        "score_weighted_reference_cover": weighted("reference_cover"),
        "candidate_left_fraction": sum(
            weight for weight, row in zip(weights, rows, strict=True) if row["dx"] < 0
        ),
        "candidate_above_fraction": sum(
            weight for weight, row in zip(weights, rows, strict=True) if row["dy"] < 0
        ),
        "horizontal_both_sides": float(
            any(row["dx"] < 0 for row in rows) and any(row["dx"] > 0 for row in rows)
        ),
        "vertical_both_sides": float(
            any(row["dy"] < 0 for row in rows) and any(row["dy"] > 0 for row in rows)
        ),
    }
