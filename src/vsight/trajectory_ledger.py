"""Dependency-light proposal trajectory summaries for TRACE-Bind.

Object-query indices are stable across GroundingDINO decoder layers.  This
module keeps that identity explicit and reduces layer tensors to auditable box,
score, rank, and hidden-state summaries.  Full hidden vectors are deliberately
not accepted by the public builder, which prevents accidental JSON artifacts
containing large model tensors.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


TRAJECTORY_LEDGER_SCHEMA_VERSION = "vsight_trajectory_ledger_v1"


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _cxcywh_box(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("every trajectory box must contain four coordinates")
    center_x, center_y, width, height = (_clip(float(value)) for value in values)
    return center_x, center_y, width, height


def cxcywh_to_xyxy(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert a normalized center-size box into clipped corner coordinates."""

    center_x, center_y, width, height = _cxcywh_box(values)
    return (
        _clip(center_x - width / 2.0),
        _clip(center_y - height / 2.0),
        _clip(center_x + width / 2.0),
        _clip(center_y + height / 2.0),
    )


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Return IoU for two ``xyxy`` boxes without external dependencies."""

    if len(left) != 4 or len(right) != 4:
        raise ValueError("IoU boxes must contain four coordinates")
    left_x1, left_y1, left_x2, left_y2 = (float(value) for value in left)
    right_x1, right_y1, right_x2, right_y2 = (float(value) for value in right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return _clip(intersection / union) if union > 1e-12 else 0.0


def _ranked_query_ids(scores: Sequence[float]) -> tuple[int, ...]:
    return tuple(
        sorted(range(len(scores)), key=lambda query_id: (-float(scores[query_id]), query_id))
    )


def select_candidate_query_ids(
    span_scores: Mapping[str, Sequence[Sequence[float]]],
    *,
    roles: Sequence[str] = ("target", "reference"),
    candidate_cap: int = 5,
) -> tuple[int, ...]:
    """Select a fixed union of final-layer Top-K object queries by role."""

    if not 1 <= candidate_cap <= 5:
        raise ValueError("TRACE candidate cap must be in [1, 5]")
    selected: list[int] = []
    for role in roles:
        layers = span_scores.get(role)
        if not layers:
            continue
        for query_id in _ranked_query_ids(layers[-1])[:candidate_cap]:
            if query_id not in selected:
                selected.append(query_id)
    return tuple(selected)


def _normalized_entropy(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    nonnegative = [max(0.0, float(value)) for value in values]
    total = sum(nonnegative)
    if total <= 1e-12:
        return 1.0
    probabilities = [value / total for value in nonnegative if value > 1e-12]
    return _clip(
        -sum(value * math.log(value) for value in probabilities) / math.log(len(values))
    )


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _validate_layers(
    layer_boxes_cxcywh: Sequence[Sequence[Sequence[float]]],
    span_scores: Mapping[str, Sequence[Sequence[float]]],
    global_scores: Sequence[Sequence[float]] | None,
    hidden_l2_norms: Sequence[Sequence[float]] | None,
    hidden_cosine_to_previous: Sequence[Sequence[float | None]] | None,
) -> tuple[int, int]:
    layer_count = len(layer_boxes_cxcywh)
    if layer_count == 0:
        raise ValueError("trajectory must contain at least one decoder layer")
    query_count = len(layer_boxes_cxcywh[0])
    if query_count == 0:
        raise ValueError("trajectory must contain at least one object query")
    if any(len(layer) != query_count for layer in layer_boxes_cxcywh):
        raise ValueError("trajectory layers must contain equal query counts")
    for layer in layer_boxes_cxcywh:
        for box in layer:
            _cxcywh_box(box)
    for role, layers in span_scores.items():
        if not role or len(layers) != layer_count:
            raise ValueError("span-score roles must cover every decoder layer")
        if any(len(layer) != query_count for layer in layers):
            raise ValueError("span-score layers must align with object queries")
        for layer in layers:
            for value in layer:
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError("span scores must be in [0, 1]")
    for name, values in (
        ("global scores", global_scores),
        ("hidden norms", hidden_l2_norms),
        ("hidden cosine summaries", hidden_cosine_to_previous),
    ):
        if values is None:
            continue
        if len(values) != layer_count or any(len(layer) != query_count for layer in values):
            raise ValueError(f"{name} must align with decoder layers and object queries")
    return layer_count, query_count


def build_trajectory_ledger(
    *,
    layer_boxes_cxcywh: Sequence[Sequence[Sequence[float]]],
    span_scores: Mapping[str, Sequence[Sequence[float]]],
    candidate_query_ids: Sequence[int] | None = None,
    candidate_roles: Sequence[str] = ("target", "reference"),
    candidate_cap: int = 5,
    global_scores: Sequence[Sequence[float]] | None = None,
    hidden_l2_norms: Sequence[Sequence[float]] | None = None,
    hidden_cosine_to_previous: Sequence[Sequence[float | None]] | None = None,
    stabilization_iou: float = 0.90,
) -> dict[str, object]:
    """Build a versioned JSON-safe proposal trajectory ledger.

    Candidate identities are detector object-query indices, so tracking never
    depends on post-hoc box matching.  Candidate selection uses only final-layer
    span scores and is then frozen for every earlier layer.
    """

    if not 1 <= candidate_cap <= 5:
        raise ValueError("TRACE candidate cap must be in [1, 5]")
    if not 0.0 <= stabilization_iou <= 1.0:
        raise ValueError("stabilization IoU must be in [0, 1]")
    layer_count, query_count = _validate_layers(
        layer_boxes_cxcywh,
        span_scores,
        global_scores,
        hidden_l2_norms,
        hidden_cosine_to_previous,
    )
    active_roles = tuple(role for role in candidate_roles if role in span_scores)
    if not active_roles:
        active_roles = tuple(span_scores)[:1]
    if candidate_query_ids is None:
        candidate_query_ids = select_candidate_query_ids(
            span_scores, roles=active_roles, candidate_cap=candidate_cap
        )
    candidate_ids = tuple(dict.fromkeys(int(value) for value in candidate_query_ids))
    if not candidate_ids:
        raise ValueError("trajectory candidate set cannot be empty")
    if any(value < 0 or value >= query_count for value in candidate_ids):
        raise ValueError("candidate query ID is outside the detector query range")

    rank_orders = {
        role: [_ranked_query_ids(layer) for layer in layers]
        for role, layers in span_scores.items()
    }
    global_ranks = {
        role: [
            {query_id: rank + 1 for rank, query_id in enumerate(order)}
            for order in orders
        ]
        for role, orders in rank_orders.items()
    }
    candidate_ranks: dict[str, list[dict[int, int]]] = {}
    for role, layers in span_scores.items():
        candidate_ranks[role] = []
        for layer in layers:
            order = sorted(candidate_ids, key=lambda query_id: (-float(layer[query_id]), query_id))
            candidate_ranks[role].append(
                {query_id: rank + 1 for rank, query_id in enumerate(order)}
            )

    trajectories = []
    for query_id in candidate_ids:
        states = []
        previous_box = None
        for layer_index in range(layer_count):
            center_box = _cxcywh_box(layer_boxes_cxcywh[layer_index][query_id])
            corner_box = cxcywh_to_xyxy(center_box)
            delta = (
                _mean([abs(value - previous) for value, previous in zip(center_box, previous_box)])
                if previous_box is not None else None
            )
            hidden_summary = None
            if hidden_l2_norms is not None or hidden_cosine_to_previous is not None:
                hidden_summary = {
                    "l2_norm": (
                        float(hidden_l2_norms[layer_index][query_id])
                        if hidden_l2_norms is not None else None
                    ),
                    "cosine_to_previous": (
                        None
                        if hidden_cosine_to_previous is None
                        else hidden_cosine_to_previous[layer_index][query_id]
                    ),
                }
            states.append(
                {
                    "layer": layer_index,
                    "object_query_id": query_id,
                    "bbox_cxcywh": list(center_box),
                    "bbox_xyxy": list(corner_box),
                    "bbox_delta_l1": delta,
                    "global_score": (
                        _clip(global_scores[layer_index][query_id])
                        if global_scores is not None else None
                    ),
                    "span_scores": {
                        role: _clip(layers[layer_index][query_id])
                        for role, layers in span_scores.items()
                    },
                    "rank_by_role": {
                        role: ranks[layer_index][query_id]
                        for role, ranks in global_ranks.items()
                    },
                    "candidate_rank_by_role": {
                        role: ranks[layer_index][query_id]
                        for role, ranks in candidate_ranks.items()
                    },
                    "top_k_by_role": {
                        role: global_ranks[role][layer_index][query_id] <= candidate_cap
                        for role in span_scores
                    },
                    "hidden_summary": hidden_summary,
                }
            )
            previous_box = center_box
        trajectories.append(
            {
                "object_query_id": query_id,
                "states": states,
                "rank_history": {
                    role: [state["rank_by_role"][role] for state in states]
                    for role in span_scores
                },
                "top_k_membership_rate": {
                    role: _mean(
                        [float(state["top_k_by_role"][role]) for state in states]
                    )
                    for role in span_scores
                },
                "mean_bbox_delta_l1": _mean(
                    [
                        float(state["bbox_delta_l1"])
                        for state in states
                        if state["bbox_delta_l1"] is not None
                    ]
                ),
            }
        )

    role_statistics: dict[str, dict[str, float | int]] = {}
    for role in active_roles:
        rank_movements = []
        if len(candidate_ids) > 1:
            for layer_index in range(1, layer_count):
                rank_movements.extend(
                    abs(
                        candidate_ranks[role][layer_index][query_id]
                        - candidate_ranks[role][layer_index - 1][query_id]
                    )
                    / (len(candidate_ids) - 1)
                    for query_id in candidate_ids
                )
        rank_stability = 1.0 - _mean(rank_movements) if rank_movements else 1.0
        top_sets = [set(order[:candidate_cap]) for order in rank_orders[role]]
        layer_agreement = _mean(
            [
                _jaccard(top_sets[index - 1], top_sets[index])
                for index in range(1, len(top_sets))
            ]
        ) if len(top_sets) > 1 else 1.0
        final_top = rank_orders[role][-1][0]
        final_box = cxcywh_to_xyxy(layer_boxes_cxcywh[-1][final_top])
        stabilization_layer = layer_count - 1
        for layer_index in range(layer_count):
            if all(
                rank_orders[role][later][0] == final_top
                and box_iou(
                    cxcywh_to_xyxy(layer_boxes_cxcywh[later][final_top]), final_box
                ) >= stabilization_iou
                for later in range(layer_index, layer_count)
            ):
                stabilization_layer = layer_index
                break
        entropy = _mean(
            [
                _normalized_entropy([layer[query_id] for query_id in candidate_ids])
                for layer in span_scores[role]
            ]
        )
        role_statistics[role] = {
            "rank_stability": _clip(rank_stability),
            "layer_agreement": _clip(layer_agreement),
            "stabilization_layer": stabilization_layer,
            "trajectory_entropy": _clip(entropy),
        }

    bbox_convergence = 1.0
    if layer_count > 1:
        bbox_convergence = _mean(
            [
                box_iou(
                    cxcywh_to_xyxy(layer_boxes_cxcywh[-2][query_id]),
                    cxcywh_to_xyxy(layer_boxes_cxcywh[-1][query_id]),
                )
                for query_id in candidate_ids
            ]
        )
    statistics = {
        "rank_stability": _mean(
            [float(row["rank_stability"]) for row in role_statistics.values()]
        ),
        "layer_agreement": _mean(
            [float(row["layer_agreement"]) for row in role_statistics.values()]
        ),
        "stabilization_layer": max(
            (int(row["stabilization_layer"]) for row in role_statistics.values()),
            default=layer_count - 1,
        ),
        "trajectory_entropy": _mean(
            [float(row["trajectory_entropy"]) for row in role_statistics.values()]
        ),
        "bbox_convergence": _clip(bbox_convergence),
    }
    return {
        "schema_version": TRAJECTORY_LEDGER_SCHEMA_VERSION,
        "tracking_key": "detector_object_query_index",
        "layer_count": layer_count,
        "detector_query_count": query_count,
        "candidate_cap_per_role": candidate_cap,
        "candidate_roles": list(active_roles),
        "candidate_query_ids": list(candidate_ids),
        "proposal_trajectories": trajectories,
        "role_statistics": role_statistics,
        "statistics": statistics,
        "hidden_tensors_serialized": False,
    }
