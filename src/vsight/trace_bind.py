"""Train-free typed assignment diagnostics for TRACE-Bind Phase 1.

The builder operates only on a fixed proposal trajectory ledger.  It emits
upstream, alternative, and role-swap witnesses but has no action-policy API and
cannot alter an upstream grounding decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .trajectory_ledger import box_iou


TRACE_BIND_SCHEMA_VERSION = "vsight_trace_bind_pilot_v1"


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


@dataclass(frozen=True)
class TraceSpan:
    """Raw query-span provenance retained by the TRACE ledger."""

    role: str
    text: str
    token_indices: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("TRACE span role cannot be empty")
        if len(self.token_indices) != len(self.token_offsets):
            raise ValueError("TRACE token indices and offsets must align")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("TRACE span confidence must be in [0, 1]")
        for start, end in self.token_offsets:
            if start < 0 or end <= start:
                raise ValueError("TRACE span offsets must be non-empty and non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "raw_span": self.text,
            "token_indices": list(self.token_indices),
            "token_offsets": [list(value) for value in self.token_offsets],
            "confidence": _clip(self.confidence),
            "source": self.source,
        }


def _normalized_upstream_box(
    box: Sequence[float] | None, image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    if len(box) != 4:
        raise ValueError("upstream box must contain four coordinates")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        _clip(x1 / image_width),
        _clip(y1 / image_height),
        _clip(x2 / image_width),
        _clip(y2 / image_height),
    )


def _trajectory_state_maps(
    trajectory_ledger: Mapping[str, object],
) -> tuple[list[dict[int, Mapping[str, object]]], list[int]]:
    layer_count = int(trajectory_ledger.get("layer_count") or 0)
    trajectories = list(trajectory_ledger.get("proposal_trajectories") or [])
    if layer_count <= 0 or not trajectories:
        return [], []
    layers: list[dict[int, Mapping[str, object]]] = [dict() for _ in range(layer_count)]
    query_ids = []
    for trajectory in trajectories:
        query_id = int(trajectory["object_query_id"])
        query_ids.append(query_id)
        states = list(trajectory.get("states") or [])
        if len(states) != layer_count:
            raise ValueError("proposal trajectory does not cover every decoder layer")
        for state in states:
            layer = int(state["layer"])
            if not 0 <= layer < layer_count or query_id in layers[layer]:
                raise ValueError("invalid or duplicate trajectory state")
            layers[layer][query_id] = state
    return layers, query_ids


def _role_score(state: Mapping[str, object], role: str) -> float:
    scores = state.get("span_scores") or {}
    return _clip(float(scores.get(role) or 0.0))


def _generic_pair_geometry(
    target_box: Sequence[float], reference_box: Sequence[float]
) -> float:
    target_x = (float(target_box[0]) + float(target_box[2])) / 2.0
    target_y = (float(target_box[1]) + float(target_box[3])) / 2.0
    reference_x = (float(reference_box[0]) + float(reference_box[2])) / 2.0
    reference_y = (float(reference_box[1]) + float(reference_box[3])) / 2.0
    center_distance = min(
        1.0, math.hypot(target_x - reference_x, target_y - reference_y) / math.sqrt(2.0)
    )
    distinctness = 1.0 - box_iou(target_box, reference_box)
    return _clip((center_distance + distinctness) / 2.0)


def _assignment(
    *,
    layer: int,
    target_query_id: int,
    reference_query_id: int | None,
    state_by_query: Mapping[int, Mapping[str, object]],
    upstream_box: Sequence[float] | None,
    atom_roles: Sequence[str],
    assignment_kind: str,
) -> dict[str, object]:
    target = state_by_query[target_query_id]
    target_box = target["bbox_xyxy"]
    target_support = _mean(
        [
            _role_score(target, role)
            for role in ("target", "full_query")
            if role in (target.get("span_scores") or {})
        ]
    )
    atom_support = _mean(
        [
            _role_score(target, role)
            for role in atom_roles
            if role in (target.get("span_scores") or {})
        ]
    ) if atom_roles else target_support
    upstream_overlap = box_iou(target_box, upstream_box) if upstream_box is not None else None
    components = [target_support, atom_support]
    reference_support = None
    geometry = upstream_overlap if upstream_overlap is not None else 0.0
    if reference_query_id is not None:
        reference = state_by_query[reference_query_id]
        reference_support = _role_score(reference, "reference")
        geometry = _generic_pair_geometry(target_box, reference["bbox_xyxy"])
        components.extend((reference_support, geometry))
    if upstream_overlap is not None:
        components.append(upstream_overlap)
    score = _clip(_mean(components))
    return {
        "layer": layer,
        "assignment_kind": assignment_kind,
        "target_object_query_id": target_query_id,
        "atom_roles": list(atom_roles),
        "reference_object_query_id": reference_query_id,
        "target_support": target_support,
        "atom_support": atom_support,
        "reference_support": reference_support,
        "edge_geometry": geometry,
        "upstream_iou": upstream_overlap,
        "score": score,
    }


def _rank_candidates(
    query_ids: Sequence[int], state_by_query: Mapping[int, Mapping[str, object]], role: str
) -> list[int]:
    return sorted(
        query_ids,
        key=lambda query_id: (-_role_score(state_by_query[query_id], role), query_id),
    )


def build_trace_bind_ledger(
    *,
    sample_id: str,
    query: str,
    upstream_box_xyxy: Sequence[float] | None,
    spans: Sequence[TraceSpan | Mapping[str, object]],
    trajectory_ledger: Mapping[str, object],
    image_width: int,
    image_height: int,
    image_encoder_forwards: int = 1,
    candidate_cap: int = 5,
    detector_only_latency_ms: float | None = None,
    trace_added_latency_ms: float | None = None,
) -> dict[str, object]:
    """Build typed, fixed-candidate TRACE assignment and swap diagnostics."""

    if image_encoder_forwards != 1:
        raise ValueError("TRACE-Bind requires exactly one detector image-encoder forward")
    if not 1 <= candidate_cap <= 5:
        raise ValueError("TRACE candidate cap must be in [1, 5]")
    serialized_spans = [
        span.as_dict() if isinstance(span, TraceSpan) else dict(span) for span in spans
    ]
    role_names = {str(span.get("role") or "") for span in serialized_spans}
    upstream_normalized = _normalized_upstream_box(
        upstream_box_xyxy, image_width, image_height
    )
    layers, candidate_query_ids = _trajectory_state_maps(trajectory_ledger)
    trajectory_statistics = dict(trajectory_ledger.get("statistics") or {})
    empty_statistics = {
        "rank_stability": float(trajectory_statistics.get("rank_stability") or 0.0),
        "layer_agreement": float(trajectory_statistics.get("layer_agreement") or 0.0),
        "stabilization_layer": int(trajectory_statistics.get("stabilization_layer") or 0),
        "swap_persistence": 0.0,
        "alternative_dominance": 0.0,
        "trajectory_entropy": float(trajectory_statistics.get("trajectory_entropy") or 0.0),
        "bbox_convergence": float(trajectory_statistics.get("bbox_convergence") or 0.0),
        "edge_uncertainty": 1.0,
    }
    base = {
        "schema_version": TRACE_BIND_SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "query": str(query),
        "upstream_box_xyxy": (
            [float(value) for value in upstream_box_xyxy]
            if upstream_box_xyxy is not None else None
        ),
        "upstream_box_normalized_xyxy": (
            list(upstream_normalized) if upstream_normalized is not None else None
        ),
        "spans": serialized_spans,
        "proposal_trajectories": list(
            trajectory_ledger.get("proposal_trajectories") or []
        ),
        "trajectory_schema_version": trajectory_ledger.get("schema_version"),
        "budget": {
            "upstream_mllm_calls": 1,
            "detector_image_encoder_forwards": image_encoder_forwards,
            "candidate_cap": candidate_cap,
        },
        "latency_ms": {
            "detector_only": detector_only_latency_ms,
            "trace_added": trace_added_latency_ms,
        },
        "action_policy_applied": False,
    }
    if not layers or upstream_normalized is None or "target" not in role_names:
        return {
            **base,
            "evidence_status": "unavailable",
            "fixed_candidates": {"target": [], "reference": []},
            "assignment_ledger": [],
            "counterfactual_witness": None,
            "statistics": empty_statistics,
        }

    final_states = layers[-1]
    target_candidates = _rank_candidates(
        candidate_query_ids, final_states, "target"
    )[:candidate_cap]
    has_reference = "reference" in role_names
    reference_candidates = (
        _rank_candidates(candidate_query_ids, final_states, "reference")[:candidate_cap]
        if has_reference else []
    )
    upstream_target = max(
        target_candidates,
        key=lambda query_id: (
            box_iou(final_states[query_id]["bbox_xyxy"], upstream_normalized),
            _role_score(final_states[query_id], "target"),
            -query_id,
        ),
    )
    atom_roles = tuple(
        role
        for role in ("modifier", "action", "modifier/action", "predicate")
        if role in role_names
    )

    def possible_assignments(
        layer_index: int,
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None]:
        states = layers[layer_index]
        if not has_reference:
            upstream = _assignment(
                layer=layer_index,
                target_query_id=upstream_target,
                reference_query_id=None,
                state_by_query=states,
                upstream_box=upstream_normalized,
                atom_roles=atom_roles,
                assignment_kind="upstream",
            )
            alternatives = [
                _assignment(
                    layer=layer_index,
                    target_query_id=query_id,
                    reference_query_id=None,
                    state_by_query=states,
                    upstream_box=upstream_normalized,
                    atom_roles=atom_roles,
                    assignment_kind="alternative_target",
                )
                for query_id in target_candidates
                if query_id != upstream_target
            ]
            return upstream, alternatives, None

        reference_edges = [
            _assignment(
                layer=layer_index,
                target_query_id=upstream_target,
                reference_query_id=reference_id,
                state_by_query=states,
                upstream_box=upstream_normalized,
                atom_roles=atom_roles,
                assignment_kind="upstream_reference_search",
            )
            for reference_id in reference_candidates
            if reference_id != upstream_target
        ]
        if not reference_edges:
            upstream = _assignment(
                layer=layer_index,
                target_query_id=upstream_target,
                reference_query_id=None,
                state_by_query=states,
                upstream_box=upstream_normalized,
                atom_roles=atom_roles,
                assignment_kind="upstream_reference_unavailable",
            )
            return upstream, [], None
        upstream = max(
            reference_edges,
            key=lambda row: (float(row["score"]), -int(row["reference_object_query_id"])),
        )
        upstream = {**upstream, "assignment_kind": "upstream"}
        upstream_reference = int(upstream["reference_object_query_id"])
        alternatives = []
        for target_id in target_candidates:
            for reference_id in reference_candidates:
                if target_id == reference_id:
                    continue
                if target_id == upstream_target and reference_id == upstream_reference:
                    continue
                alternatives.append(
                    _assignment(
                        layer=layer_index,
                        target_query_id=target_id,
                        reference_query_id=reference_id,
                        state_by_query=states,
                        upstream_box=upstream_normalized,
                        atom_roles=atom_roles,
                        assignment_kind="alternative_edge",
                    )
                )
        role_swap = None
        if upstream_reference in states and upstream_reference != upstream_target:
            role_swap = _assignment(
                layer=layer_index,
                target_query_id=upstream_reference,
                reference_query_id=upstream_target,
                state_by_query=states,
                upstream_box=upstream_normalized,
                atom_roles=atom_roles,
                assignment_kind="target_reference_role_swap",
            )
        return upstream, alternatives, role_swap

    assignment_ledger = []
    upstream_rows = []
    best_alternatives = []
    swap_rows = []
    for layer_index in range(len(layers)):
        upstream, alternatives, role_swap = possible_assignments(layer_index)
        alternatives.sort(
            key=lambda row: (
                -float(row["score"]),
                int(row["target_object_query_id"]),
                int(row.get("reference_object_query_id") or -1),
            )
        )
        best_alternative = alternatives[0] if alternatives else None
        upstream_rows.append(upstream)
        if best_alternative is not None:
            best_alternatives.append(best_alternative)
        if role_swap is not None:
            swap_rows.append(role_swap)
        assignment_ledger.append(
            {
                "layer": layer_index,
                "upstream": upstream,
                "best_alternative": best_alternative,
                "role_swap": role_swap,
                "alternative_margin": (
                    float(best_alternative["score"]) - float(upstream["score"])
                    if best_alternative is not None else None
                ),
                "swap_margin": (
                    float(role_swap["score"]) - float(upstream["score"])
                    if role_swap is not None else None
                ),
                "alternatives": alternatives[: candidate_cap * candidate_cap],
            }
        )

    alternative_dominance = _mean(
        [
            float(
                row["best_alternative"] is not None
                and float(row["best_alternative"]["score"])
                > float(row["upstream"]["score"])
            )
            for row in assignment_ledger
        ]
    )
    swap_persistence = _mean(
        [
            float(
                row["role_swap"] is not None
                and float(row["role_swap"]["score"]) > float(row["upstream"]["score"])
            )
            for row in assignment_ledger
        ]
    )
    final_upstream = upstream_rows[-1]
    final_alternative = assignment_ledger[-1]["best_alternative"]
    if final_alternative is None:
        edge_uncertainty = 1.0 - float(final_upstream["score"])
    else:
        left = float(final_upstream["score"])
        right = float(final_alternative["score"])
        edge_uncertainty = 1.0 - abs(left - right) / max(left + right, 1e-12)
    strongest_counterfactual = max(
        [*best_alternatives, *swap_rows],
        key=lambda row: float(row["score"]),
        default=None,
    )
    max_upstream_support = max(float(row["score"]) for row in upstream_rows)
    final_alternative_dominates = bool(
        final_alternative is not None
        and float(final_alternative["score"]) > float(final_upstream["score"])
    )
    if final_alternative_dominates and alternative_dominance > 0.5:
        evidence_status = "contradicted"
    elif max_upstream_support < 0.20:
        evidence_status = "unsupported"
    else:
        evidence_status = "supported"
    statistics = {
        **empty_statistics,
        "swap_persistence": _clip(swap_persistence),
        "alternative_dominance": _clip(alternative_dominance),
        "edge_uncertainty": _clip(edge_uncertainty),
    }
    return {
        **base,
        "evidence_status": evidence_status,
        "fixed_candidates": {
            "target": target_candidates,
            "reference": reference_candidates,
        },
        "upstream_target_object_query_id": upstream_target,
        "assignment_ledger": assignment_ledger,
        "counterfactual_witness": strongest_counterfactual,
        "statistics": statistics,
    }
