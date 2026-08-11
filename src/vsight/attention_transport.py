"""Ontology-free role induction and attention transport for CABLE-RAFT.

The module intentionally operates on token offsets, attention maps and boxes.
It does not import a category, attribute, action, or dataset vocabulary.  A
small universal relation grammar is used only as an optional span boundary
fallback, never for relation semantics.  A detector adapter may use it after a
single forward pass to infer a conservative target/predicate/reference ledger.
The output is diagnostic evidence; CCV's
existing action policy remains the authority for ACCEPT/REJECT/RELOCALIZE.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .ccv import DetectorEvidence, DetectorProposal, PromptProvenance
from .relation_supervision import RELATION_PATTERNS


# Function-word filtering is grammatical bookkeeping, not an object/attribute
# ontology.  It prevents determiners from becoming spurious role boundaries.
_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "his", "her", "their"})


@dataclass(frozen=True)
class RoleSpan:
    role: str
    token_indices: tuple[int, ...]
    text: str
    confidence: float
    source: str = "attention_change_point"


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalize(values: Sequence[float]) -> list[float]:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0:
        return [0.0 for _ in values]
    return [max(0.0, float(value)) / total for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return _clip(numerator / (left_norm * right_norm))


def _entropy(values: Sequence[float]) -> float:
    normalized = _normalize(values)
    nonzero = [value for value in normalized if value > 1e-12]
    if len(nonzero) <= 1:
        return 0.0
    return _clip(-sum(value * math.log(value) for value in nonzero) / math.log(len(normalized)))


def _mean_map(token_maps: Sequence[Sequence[float]], indices: Sequence[int]) -> list[float]:
    if not indices:
        return []
    width = len(token_maps[indices[0]])
    result = [0.0] * width
    for index in indices:
        row = token_maps[index]
        if len(row) != width:
            raise ValueError("attention maps must have equal widths")
        for position, value in enumerate(row):
            result[position] += max(0.0, float(value))
    return _normalize([value / len(indices) for value in result])


def _span_text(
    query: str, token_indices: Sequence[int], token_offsets: Sequence[Sequence[int]]
) -> str:
    valid = [
        (int(token_offsets[index][0]), int(token_offsets[index][1]))
        for index in token_indices
        if 0 <= index < len(token_offsets) and len(token_offsets[index]) == 2
        and int(token_offsets[index][1]) > int(token_offsets[index][0])
    ]
    if not valid:
        return ""
    start = min(item[0] for item in valid)
    end = max(item[1] for item in valid)
    prefix = query[:start]
    determiner = re.search(r"(?:^|\s)(a|an|the|this|that|his|her|their)\s*$", prefix, re.I)
    if determiner:
        start = determiner.start(1)
    return query[start:end].strip()


def _content_indices(
    query: str,
    token_offsets: Sequence[Sequence[int]],
    token_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Keep lexical tokens using offsets only, never semantic vocabularies."""

    indices = token_indices if token_indices is not None else range(len(token_offsets))
    result = []
    for index in indices:
        if index < 0 or index >= len(token_offsets) or len(token_offsets[index]) != 2:
            continue
        start, end = (int(value) for value in token_offsets[index])
        if end <= start:
            continue
        token = query[start:end]
        if re.search(r"[\w]", token, flags=re.UNICODE) and token.casefold() not in _DETERMINERS:
            result.append(index)
    return tuple(result)


def _span_score(token_maps: Sequence[Sequence[float]], groups: Sequence[Sequence[int]]) -> float:
    target_map, predicate_map, reference_map = (
        _mean_map(token_maps, group) for group in groups
    )
    if not target_map or not predicate_map or not reference_map:
        return -1.0
    separation = 1.0 - _cosine(target_map, reference_map)
    target_focus = 1.0 - _entropy(target_map)
    reference_focus = 1.0 - _entropy(reference_map)
    # A bridge must be jointly compatible with both endpoint maps.  Averaging
    # the two similarities would let a noun from one endpoint leak into the
    # predicate; the minimum is deliberately conservative.
    predicate_bridge = min(
        _cosine(predicate_map, target_map),
        _cosine(predicate_map, reference_map),
    )
    predicate_focus = 1.0 - _entropy(predicate_map)
    # Structural, rather than lexical, priors: a predicate bridge should be
    # less concentrated than either noun span and connect both sides.
    return (
        0.45 * separation
        + 0.20 * target_focus
        + 0.20 * reference_focus
        + 0.30 * predicate_bridge
        - 0.12 * predicate_focus
        - 0.03 * max(0, len(groups[1]) - 1)
    )


def infer_role_spans(
    *,
    query: str,
    token_offsets: Sequence[Sequence[int]],
    token_maps: Sequence[Sequence[float]],
    min_margin: float = 0.05,
) -> tuple[RoleSpan, ...]:
    """Infer ordered role spans from attention change points.

    All contiguous token partitions are considered.  The no-relation partition
    wins unless a three-way partition has a structural margin, which makes
    unsupported relation claims preserve upstream evidence instead of forcing
    a lexical interpretation.
    """

    if len(token_maps) != len(token_offsets):
        raise ValueError("token maps and offsets must have equal lengths")
    content = _content_indices(query, token_offsets)
    if not content:
        return (RoleSpan("target", (), "", 0.0),)
    if not any(sum(max(0.0, float(value)) for value in token_maps[index]) > 0 for index in content):
        return (
            RoleSpan("target", content, _span_text(query, content, token_offsets), 0.0),
        )
    all_map = _mean_map(token_maps, content)
    no_relation_score = 1.0 - _entropy(all_map)
    # A small, universal relation grammar supplies boundary candidates only. It
    # never assigns an object/category label and is not used for inverse or
    # contradiction semantics. This fallback prevents a noun from being
    # absorbed into a predicate when attention maps are diffuse.
    lowered = query.casefold()
    for _, pattern in RELATION_PATTERNS:
        match = re.search(pattern, lowered)
        if match is None:
            continue
        target = tuple(
            index for index in content
            if token_offsets[index][1] <= match.start()
        )
        predicate = tuple(
            index for index in content
            if token_offsets[index][0] < match.end()
            and token_offsets[index][1] > match.start()
        )
        reference = tuple(index for index in content if token_offsets[index][0] >= match.end())
        if target and predicate and reference:
            groups = (target, predicate, reference)
            structural_score = _span_score(token_maps, groups)
            confidence = _clip(structural_score)
            return (
                RoleSpan("target", target, _span_text(query, target, token_offsets), confidence, "universal_relation_boundary"),
                RoleSpan("predicate", predicate, _span_text(query, predicate, token_offsets), confidence, "universal_relation_boundary"),
                RoleSpan("reference", reference, _span_text(query, reference, token_offsets), confidence, "universal_relation_boundary"),
            )

    best: tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
    for first_end in range(1, len(content) - 1):
        for predicate_width in range(1, min(6, len(content) - first_end)):
            second_end = first_end + predicate_width
            if second_end >= len(content):
                continue
            groups = (
                content[:first_end],
                content[first_end:second_end],
                content[second_end:],
            )
            score = _span_score(token_maps, groups)
            if best is None or score > best[0]:
                best = (score, groups[0], groups[1], groups[2])
    if best is None:
        return (
            RoleSpan("target", content, _span_text(query, content, token_offsets), _clip(no_relation_score)),
        )
    score, target, predicate, reference = best
    margin = score - no_relation_score
    # Attention-only segmentation is intentionally much harder to trigger
    # than the universal-boundary path.  Without a recognized boundary it is
    # safer to keep a single target span than to hallucinate a relation edge.
    if margin < max(min_margin, 0.15) or score < 0.55:
        return (
            RoleSpan("target", content, _span_text(query, content, token_offsets), _clip(no_relation_score)),
        )
    confidence = _clip(0.5 + margin)
    return (
        RoleSpan("target", target, _span_text(query, target, token_offsets), confidence),
        RoleSpan("predicate", predicate, _span_text(query, predicate, token_offsets), confidence),
        RoleSpan("reference", reference, _span_text(query, reference, token_offsets), confidence),
    )


def _box_union(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    return (
        min(float(left[0]), float(right[0])),
        min(float(left[1]), float(right[1])),
        max(float(left[2]), float(right[2])),
        max(float(left[3]), float(right[3])),
    )


def _region_mass(
    values: Sequence[float],
    box: Sequence[float],
    coordinates: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
) -> float:
    if not values or not coordinates or len(values) != len(coordinates):
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    total = 0.0
    for value, coordinate in zip(values, coordinates, strict=True):
        if len(coordinate) != 2:
            continue
        x, y = float(coordinate[0]) * width, float(coordinate[1]) * height
        if x1 <= x <= x2 and y1 <= y <= y2:
            total += max(0.0, float(value))
    return _clip(total)


def _proposal_score(
    probabilities: Sequence[Sequence[float]], query_index: int, indices: Sequence[int]
) -> float:
    if query_index < 0 or query_index >= len(probabilities) or not indices:
        return 0.0
    values = [
        float(probabilities[query_index][index])
        for index in indices
        if 0 <= index < len(probabilities[query_index])
    ]
    return _clip(sum(values) / len(values)) if values else 0.0


def _relative_margin(top: float, second: float) -> float:
    """Return a scale-free top-1/top-2 margin.

    The former ``1 - (top - second)`` uncertainty was dominated by the
    absolute scale of attention mass.  Two equally ambiguous distributions
    should have the same margin even when one is uniformly rescaled.
    """

    top = max(0.0, float(top))
    second = max(0.0, float(second))
    if top <= 1e-12:
        return 0.0
    return _clip((top - second) / (top + second + 1e-12))


def _role_share(
    values: Sequence[float],
    indices: Sequence[int],
    content_indices: Sequence[int],
) -> float:
    """Normalize proposal-to-text attention over query content tokens."""

    denominator = sum(
        max(0.0, float(values[index]))
        for index in content_indices
        if 0 <= index < len(values)
    )
    if denominator <= 1e-12:
        return 0.0
    numerator = sum(
        max(0.0, float(values[index]))
        for index in indices
        if 0 <= index < len(values)
    )
    return _clip(numerator / denominator)


def _sample_mass_in_box(
    samples: Sequence[Sequence[float]],
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> float:
    """Measure decoder deformable-attention mass inside a proposal box."""

    if not samples:
        return 1.0
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    total = 0.0
    inside = 0.0
    for sample in samples:
        if len(sample) != 3:
            continue
        x, y, weight = (float(value) for value in sample)
        weight = max(0.0, weight)
        total += weight
        if x1 <= x * width <= x2 and y1 <= y * height <= y2:
            inside += weight
    return _clip(inside / total) if total > 1e-12 else 0.0


def _make_proposals(
    *,
    probabilities: Sequence[Sequence[float]],
    boxes: Sequence[Sequence[float]],
    global_scores: Sequence[float],
    indices: Sequence[int],
    role: str,
    query: str,
    token_offsets: Sequence[Sequence[int]],
    image_width: int,
    image_height: int,
    text_threshold: float,
    cap: int,
    is_reference: bool,
    proposal_query_indices: Sequence[int],
) -> list[DetectorProposal]:
    ranked = []
    for query_index, box in enumerate(boxes):
        score = _proposal_score(probabilities, query_index, indices)
        if score < text_threshold:
            continue
        model_query_index = (
            int(proposal_query_indices[query_index])
            if query_index < len(proposal_query_indices) else query_index
        )
        ranked.append(
            (score, float(global_scores[query_index]), query_index, model_query_index, box)
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    text = _span_text(query, indices, token_offsets)
    result = []
    for score, _, _, model_query_index, raw_box in ranked[:cap]:
        clipped = (
            max(0.0, min(float(image_width), float(raw_box[0]))),
            max(0.0, min(float(image_height), float(raw_box[1]))),
            max(0.0, min(float(image_width), float(raw_box[2]))),
            max(0.0, min(float(image_height), float(raw_box[3]))),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        result.append(
            DetectorProposal(
                proposal_id=f"raft:{role}:{model_query_index}",
                box=clipped,
                score=score,
                atom_scores={},
                full_score=None,
                is_reference=is_reference,
                label=text or None,
                segment_id=f"raft:{role}",
                prompt_provenance=PromptProvenance.CLAIM,
            )
        )
    return result


def build_raft_evidence(
    *,
    query: str,
    token_offsets: Sequence[Sequence[int]],
    token_maps: Sequence[Sequence[float]],
    token_probabilities: Sequence[Sequence[float]],
    boxes: Sequence[Sequence[float]],
    global_scores: Sequence[float],
    coordinates: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
    prompted_atom_ids: frozenset[str],
    text_threshold: float,
    proposal_cap_per_segment: int,
    proposal_query_indices: Sequence[int] | None = None,
    decoder_text_attention: Sequence[Sequence[float]] | None = None,
    decoder_visual_attention: Sequence[Sequence[Sequence[float]]] | None = None,
    causal_role_effects: Sequence[Mapping[str, object]] | None = None,
    latency_ms: float | None = None,
    attention_metadata: Mapping[str, object] | None = None,
) -> DetectorEvidence:
    """Build single-pass RAFT evidence from normalized attention maps."""

    if not (len(boxes) == len(global_scores) == len(token_probabilities)):
        raise ValueError("boxes, scores, and token probabilities must have equal lengths")
    if proposal_query_indices is None:
        proposal_query_indices = tuple(range(len(boxes)))
    if len(proposal_query_indices) != len(boxes):
        raise ValueError("proposal query indices must align with boxes")
    if decoder_text_attention is not None and len(decoder_text_attention) != len(boxes):
        raise ValueError("decoder text attention must align with boxes")
    if decoder_visual_attention is not None and len(decoder_visual_attention) != len(boxes):
        raise ValueError("decoder visual attention must align with boxes")
    if not coordinates and token_maps:
        coordinates = [
            ((index + 0.5) / len(token_maps), 0.5)
            for index in range(len(token_maps))
        ]
    roles = infer_role_spans(
        query=query,
        token_offsets=token_offsets,
        token_maps=token_maps,
    )
    by_role = {role.role: role for role in roles}
    target = by_role["target"]
    references = by_role.get("reference")
    predicate = by_role.get("predicate")
    target_proposals = _make_proposals(
        probabilities=token_probabilities,
        boxes=boxes,
        global_scores=global_scores,
        indices=target.token_indices,
        role="target",
        query=query,
        token_offsets=token_offsets,
        image_width=image_width,
        image_height=image_height,
        text_threshold=text_threshold,
        cap=proposal_cap_per_segment,
        is_reference=False,
        proposal_query_indices=proposal_query_indices,
    )
    reference_proposals = (
        _make_proposals(
            probabilities=token_probabilities,
            boxes=boxes,
            global_scores=global_scores,
            indices=references.token_indices,
            role="reference",
            query=query,
            token_offsets=token_offsets,
            image_width=image_width,
            image_height=image_height,
            text_threshold=text_threshold,
            cap=proposal_cap_per_segment,
            is_reference=True,
            proposal_query_indices=proposal_query_indices,
        )
        if references is not None else []
    )
    full_ranked = sorted(
        zip(global_scores, boxes, range(len(boxes)), strict=True),
        key=lambda item: (-float(item[0]), item[2]),
    )[:proposal_cap_per_segment]
    full_proposals = []
    full_text = query.strip()
    for score, raw_box, query_index in full_ranked:
        model_query_index = int(proposal_query_indices[query_index])
        clipped = (
            max(0.0, min(float(image_width), float(raw_box[0]))),
            max(0.0, min(float(image_height), float(raw_box[1]))),
            max(0.0, min(float(image_width), float(raw_box[2]))),
            max(0.0, min(float(image_height), float(raw_box[3]))),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        full_proposals.append(
            DetectorProposal(
                proposal_id=f"raft:full:{model_query_index}",
                box=clipped,
                score=_clip(float(score)),
                atom_scores={},
                full_score=_clip(float(score)),
                is_reference=False,
                label=full_text or None,
                segment_id="raft:full",
                prompt_provenance=PromptProvenance.CLAIM,
            )
        )

    target_map = _mean_map(token_maps, target.token_indices)
    reference_map = _mean_map(token_maps, references.token_indices) if references else []
    predicate_map = _mean_map(token_maps, predicate.token_indices) if predicate else []
    edges = []
    local_index_by_model_query = {
        int(model_query_index): local_index
        for local_index, model_query_index in enumerate(proposal_query_indices)
    }

    def proposal_local_index(row: DetectorProposal) -> int | None:
        try:
            model_query_index = int(row.proposal_id.rsplit(":", 1)[1])
        except (TypeError, ValueError):
            return None
        return local_index_by_model_query.get(model_query_index)

    for target_row in target_proposals:
        for reference_row in reference_proposals:
            target_local_index = proposal_local_index(target_row)
            reference_local_index = proposal_local_index(reference_row)
            # A single decoder query cannot witness two distinguishable role
            # endpoints.  Keeping it would turn one object proposal into a
            # spurious relation edge.
            if (
                target_local_index is None
                or reference_local_index is None
                or target_local_index == reference_local_index
            ):
                continue
            target_mass = _region_mass(
                target_map, target_row.box, coordinates, image_width, image_height
            )
            reference_mass = _region_mass(
                reference_map, reference_row.box, coordinates, image_width, image_height
            )
            union_mass = _region_mass(
                predicate_map,
                _box_union(target_row.box, reference_row.box),
                coordinates,
                image_width,
                image_height,
            )
            transport = (target_mass * reference_mass * max(union_mass, 1e-12)) ** (1.0 / 3.0)
            swapped = (
                _region_mass(reference_map, target_row.box, coordinates, image_width, image_height)
                * _region_mass(target_map, reference_row.box, coordinates, image_width, image_height)
                * max(union_mass, 1e-12)
            ) ** (1.0 / 3.0)
            edges.append(
                {
                    "target_proposal_id": target_row.proposal_id,
                    "reference_proposal_id": reference_row.proposal_id,
                    "target_attention": target_mass,
                    "reference_attention": reference_mass,
                    "predicate_bridge": union_mass,
                    "transport": _clip(transport),
                    "role_swap_transport": _clip(swapped),
                    "role_swap_margin": transport - swapped,
                }
            )
    edges.sort(key=lambda row: (-float(row["transport"]), row["target_proposal_id"], row["reference_proposal_id"]))
    top_transport = float(edges[0]["transport"]) if edges else 0.0
    second_transport = float(edges[1]["transport"]) if len(edges) > 1 else 0.0
    transport_relative_margin = _relative_margin(top_transport, second_transport)

    decoder_edges = []
    if decoder_text_attention is not None and references is not None and predicate is not None:
        content_indices = tuple(
            dict.fromkeys(
                (*target.token_indices, *predicate.token_indices, *references.token_indices)
            )
        )
        visual_rows = decoder_visual_attention or tuple(() for _ in boxes)
        for target_row in target_proposals:
            target_local_index = proposal_local_index(target_row)
            if target_local_index is None:
                continue
            target_text = decoder_text_attention[target_local_index]
            target_visual = _sample_mass_in_box(
                visual_rows[target_local_index],
                target_row.box,
                image_width,
                image_height,
            )
            for reference_row in reference_proposals:
                reference_local_index = proposal_local_index(reference_row)
                if (
                    reference_local_index is None
                    or reference_local_index == target_local_index
                ):
                    continue
                reference_text = decoder_text_attention[reference_local_index]
                reference_visual = _sample_mass_in_box(
                    visual_rows[reference_local_index],
                    reference_row.box,
                    image_width,
                    image_height,
                )
                target_share = _role_share(
                    target_text, target.token_indices, content_indices
                )
                reference_share = _role_share(
                    reference_text, references.token_indices, content_indices
                )
                predicate_target_share = _role_share(
                    target_text, predicate.token_indices, content_indices
                )
                predicate_reference_share = _role_share(
                    reference_text, predicate.token_indices, content_indices
                )
                target_binding = math.sqrt(
                    max(target_share * target_row.score, 0.0)
                )
                reference_binding = math.sqrt(
                    max(reference_share * reference_row.score, 0.0)
                )
                predicate_pair = math.sqrt(
                    max(predicate_target_share * predicate_reference_share, 0.0)
                )
                visual_pair = math.sqrt(max(target_visual * reference_visual, 0.0))
                decoder_transport = (
                    max(target_binding, 1e-12)
                    * max(reference_binding, 1e-12)
                    * max(predicate_pair, 1e-12)
                    * max(visual_pair, 1e-12)
                ) ** 0.25
                swapped_target_share = _role_share(
                    target_text, references.token_indices, content_indices
                )
                swapped_reference_share = _role_share(
                    reference_text, target.token_indices, content_indices
                )
                swapped_target = math.sqrt(
                    max(swapped_target_share * target_row.score, 0.0)
                )
                swapped_reference = math.sqrt(
                    max(swapped_reference_share * reference_row.score, 0.0)
                )
                decoder_swapped = (
                    max(swapped_target, 1e-12)
                    * max(swapped_reference, 1e-12)
                    * max(predicate_pair, 1e-12)
                    * max(visual_pair, 1e-12)
                ) ** 0.25
                decoder_edges.append(
                    {
                        "target_proposal_id": target_row.proposal_id,
                        "reference_proposal_id": reference_row.proposal_id,
                        "target_text_binding": _clip(target_binding),
                        "reference_text_binding": _clip(reference_binding),
                        "predicate_pair_attention": _clip(predicate_pair),
                        "target_visual_self_alignment": _clip(target_visual),
                        "reference_visual_self_alignment": _clip(reference_visual),
                        "transport": _clip(decoder_transport),
                        "role_swap_transport": _clip(decoder_swapped),
                        "role_swap_margin": decoder_transport - decoder_swapped,
                        "role_swap_relative_margin": (
                            (decoder_transport - decoder_swapped)
                            / (decoder_transport + decoder_swapped + 1e-12)
                        ),
                    }
                )
    decoder_edges.sort(
        key=lambda row: (
            -float(row["transport"]),
            row["target_proposal_id"],
            row["reference_proposal_id"],
        )
    )
    top_decoder_transport = (
        float(decoder_edges[0]["transport"]) if decoder_edges else 0.0
    )
    second_decoder_transport = (
        float(decoder_edges[1]["transport"]) if len(decoder_edges) > 1 else 0.0
    )
    decoder_relative_margin = _relative_margin(
        top_decoder_transport, second_decoder_transport
    )
    effect_by_query_role = {}
    for effect in causal_role_effects or ():
        try:
            key = (
                int(effect["model_query_index"]),
                str(effect["masked_role"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        effect_by_query_role[key] = dict(effect)

    def positive_effect(model_query_index: int, role: str) -> float:
        row = effect_by_query_role.get((model_query_index, role)) or {}
        return _clip(max(0.0, float(row.get("relative_drop") or 0.0)))

    causal_edges = []
    for decoder_edge in decoder_edges:
        try:
            target_query_index = int(
                str(decoder_edge["target_proposal_id"]).rsplit(":", 1)[1]
            )
            reference_query_index = int(
                str(decoder_edge["reference_proposal_id"]).rsplit(":", 1)[1]
            )
        except (KeyError, TypeError, ValueError):
            continue
        target_effect = positive_effect(target_query_index, "target")
        reference_effect = positive_effect(reference_query_index, "reference")
        predicate_target_effect = positive_effect(target_query_index, "predicate")
        predicate_reference_effect = positive_effect(reference_query_index, "predicate")
        predicate_pair_effect = math.sqrt(
            predicate_target_effect * predicate_reference_effect
        )
        causal_transport = (
            target_effect * reference_effect * predicate_pair_effect
        ) ** (1.0 / 3.0)
        swapped_target_effect = positive_effect(target_query_index, "reference")
        swapped_reference_effect = positive_effect(reference_query_index, "target")
        swapped_transport = (
            swapped_target_effect * swapped_reference_effect * predicate_pair_effect
        ) ** (1.0 / 3.0)
        causal_edges.append(
            {
                "target_proposal_id": decoder_edge["target_proposal_id"],
                "reference_proposal_id": decoder_edge["reference_proposal_id"],
                "target_relative_drop": target_effect,
                "reference_relative_drop": reference_effect,
                "predicate_pair_relative_drop": predicate_pair_effect,
                "transport": _clip(causal_transport),
                "role_swap_transport": _clip(swapped_transport),
                "role_swap_margin": causal_transport - swapped_transport,
                "role_swap_relative_margin": (
                    (causal_transport - swapped_transport)
                    / (causal_transport + swapped_transport + 1e-12)
                ),
            }
        )
    causal_edges.sort(
        key=lambda row: (
            -float(row["transport"]),
            row["target_proposal_id"],
            row["reference_proposal_id"],
        )
    )
    positive_causal_edges = [
        row for row in causal_edges if float(row["transport"]) > 1e-12
    ]
    top_causal_transport = (
        float(positive_causal_edges[0]["transport"])
        if positive_causal_edges else 0.0
    )
    second_causal_transport = (
        float(positive_causal_edges[1]["transport"])
        if len(positive_causal_edges) > 1 else 0.0
    )
    causal_relative_margin = _relative_margin(
        top_causal_transport, second_causal_transport
    )

    def causal_influence(model_query_index: int, role: str) -> float:
        row = effect_by_query_role.get((model_query_index, role)) or {}
        drift = _clip(max(0.0, float(row.get("hidden_cosine_drift") or 0.0)))
        selectivity = _clip(
            max(0.0, float(row.get("hidden_drift_selectivity") or 0.0))
        )
        return math.sqrt(drift * selectivity)

    influence_edges = []
    for decoder_edge in decoder_edges:
        try:
            target_query_index = int(
                str(decoder_edge["target_proposal_id"]).rsplit(":", 1)[1]
            )
            reference_query_index = int(
                str(decoder_edge["reference_proposal_id"]).rsplit(":", 1)[1]
            )
        except (KeyError, TypeError, ValueError):
            continue
        target_influence = causal_influence(target_query_index, "target")
        reference_influence = causal_influence(reference_query_index, "reference")
        predicate_target_influence = causal_influence(
            target_query_index, "predicate"
        )
        predicate_reference_influence = causal_influence(
            reference_query_index, "predicate"
        )
        predicate_pair_influence = math.sqrt(
            predicate_target_influence * predicate_reference_influence
        )
        influence_transport = (
            target_influence * reference_influence * predicate_pair_influence
        ) ** (1.0 / 3.0)
        swapped_target_influence = causal_influence(
            target_query_index, "reference"
        )
        swapped_reference_influence = causal_influence(
            reference_query_index, "target"
        )
        swapped_influence_transport = (
            swapped_target_influence
            * swapped_reference_influence
            * predicate_pair_influence
        ) ** (1.0 / 3.0)
        influence_edges.append(
            {
                "target_proposal_id": decoder_edge["target_proposal_id"],
                "reference_proposal_id": decoder_edge["reference_proposal_id"],
                "target_hidden_influence": target_influence,
                "reference_hidden_influence": reference_influence,
                "predicate_pair_hidden_influence": predicate_pair_influence,
                "transport": _clip(influence_transport),
                "role_swap_transport": _clip(swapped_influence_transport),
                "role_swap_margin": (
                    influence_transport - swapped_influence_transport
                ),
                "role_swap_relative_margin": (
                    (influence_transport - swapped_influence_transport)
                    / (
                        influence_transport
                        + swapped_influence_transport
                        + 1e-12
                    )
                ),
            }
        )
    influence_edges.sort(
        key=lambda row: (
            -float(row["transport"]),
            row["target_proposal_id"],
            row["reference_proposal_id"],
        )
    )
    positive_influence_edges = [
        row for row in influence_edges if float(row["transport"]) > 1e-12
    ]
    top_influence_transport = (
        float(positive_influence_edges[0]["transport"])
        if positive_influence_edges else 0.0
    )
    second_influence_transport = (
        float(positive_influence_edges[1]["transport"])
        if len(positive_influence_edges) > 1 else 0.0
    )
    influence_relative_margin = _relative_margin(
        top_influence_transport, second_influence_transport
    )
    ledger = {
        "schema_version": (
            "vsight_cable_raft_attention_v3"
            if causal_role_effects is not None
            else "vsight_cable_raft_attention_v2"
        ),
        "status": "ok" if token_maps else "attention_unavailable",
        "role_spans": [
            {
                "role": role.role,
                "token_indices": list(role.token_indices),
                "text": role.text,
                "confidence": role.confidence,
                "source": role.source,
            }
            for role in roles
        ],
        "span_entropy": {
            role.role: _entropy(_mean_map(token_maps, role.token_indices))
            for role in roles
            if role.token_indices
        },
        "edges": edges[: proposal_cap_per_segment * proposal_cap_per_segment],
        "best_edge": edges[0] if edges else None,
        "edge_uncertainty": _clip(
            1.0 - top_transport * transport_relative_margin
        ),
        "top_transport": top_transport,
        "transport_margin": top_transport - second_transport,
        "transport_relative_margin": transport_relative_margin,
        "decoder_edges": decoder_edges[: proposal_cap_per_segment * proposal_cap_per_segment],
        "best_decoder_edge": decoder_edges[0] if decoder_edges else None,
        "decoder_top_transport": top_decoder_transport,
        "decoder_transport_margin": top_decoder_transport - second_decoder_transport,
        "decoder_transport_relative_margin": decoder_relative_margin,
        "decoder_edge_uncertainty": _clip(
            1.0 - top_decoder_transport * decoder_relative_margin
        ),
        "causal_role_effects": list(causal_role_effects or ()),
        "causal_edges": causal_edges[: proposal_cap_per_segment * proposal_cap_per_segment],
        "best_causal_edge": positive_causal_edges[0] if positive_causal_edges else None,
        "causal_top_transport": top_causal_transport,
        "causal_transport_margin": top_causal_transport - second_causal_transport,
        "causal_transport_relative_margin": causal_relative_margin,
        "causal_edge_uncertainty": _clip(
            1.0 - top_causal_transport * causal_relative_margin
        ),
        "causal_influence_edges": influence_edges[
            : proposal_cap_per_segment * proposal_cap_per_segment
        ],
        "best_causal_influence_edge": (
            positive_influence_edges[0] if positive_influence_edges else None
        ),
        "causal_influence_top_transport": top_influence_transport,
        "causal_influence_transport_margin": (
            top_influence_transport - second_influence_transport
        ),
        "causal_influence_transport_relative_margin": influence_relative_margin,
        "causal_influence_edge_uncertainty": _clip(
            1.0 - top_influence_transport * influence_relative_margin
        ),
        "attention_metadata": dict(attention_metadata or {}),
    }
    return DetectorEvidence(
        proposals=tuple((*target_proposals, *reference_proposals, *full_proposals)),
        image_width=image_width,
        image_height=image_height,
        evidence_complete=True,
        prompted_atom_ids=prompted_atom_ids,
        image_encoder_forwards=1,
        latency_ms=latency_ms,
        attention_ledger=ledger,
    )
