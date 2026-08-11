"""Auditable atom/edge evidence ledger for CCV-CABLE.

The ledger deliberately contains no learned visual component.  It turns the
proposals produced by one composite detector forward into typed, inspectable
node and edge witnesses.  The verifier remains responsible for the action
policy; this module only records what can (and cannot) be inferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from .e2_verifier import box_iou


Box = tuple[float, float, float, float]

INVERSE_RELATIONS: Mapping[str, str] = {
    "to_the_left_of": "to_the_right_of",
    "to_the_right_of": "to_the_left_of",
    "above": "below",
    "below": "above",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "holding": "held_by",
    "held_by": "holding",
}
SYMMETRIC_RELATIONS = frozenset({"next_to", "beside", "with", "by"})
SUPPORT_RELATIONS = frozenset({"on", "sitting_on", "standing_on", "riding"})
ROLE_RELATIONS = frozenset({"holding", "held_by", "wearing"})


def relation_family(relation: str | None) -> str:
    value = str(relation or "")
    if value in {
        "to_the_left_of", "to_the_right_of", "above", "below",
        "in_front_of", "behind",
    }:
        return "directional"
    if value in SYMMETRIC_RELATIONS:
        return "symmetric/proximity"
    if value in SUPPORT_RELATIONS:
        return "support/contact"
    if value in ROLE_RELATIONS:
        return "role/interaction"
    if value == "between":
        return "between"
    return "unknown"


@dataclass(frozen=True)
class ObjectAttributeLedger:
    candidate_id: str
    bbox_xyxy: Box
    source: str
    object_support: float
    attribute_support: float | None
    attribute_object_overlap: float | None
    attribute_decoy_margin: float | None
    attribute_counterfactual_support: float | None
    attribute_counterfactual_margin: float | None
    full_claim_support: float | None
    candidate_ambiguity: float


@dataclass(frozen=True)
class RelationEdgeLedger:
    atom_id: str
    relation: str
    relation_family: str
    target_candidate_id: str
    reference_candidate_id: str | None
    secondary_reference_candidate_id: str | None
    edge_claim: float | None
    edge_alt_target: float | None
    edge_alt_reference: float | None
    edge_swap_margin: float | None
    counterfactual_edge: float | None
    inverse_consistency: float | None
    symmetry_consistency: float | None
    role_consistency: float | None
    full_edge_agreement: float | None
    edge_uncertainty: float
    m_contra: float | None
    interpretable: bool
    unknown_reason: str | None


@dataclass(frozen=True)
class AtomBindingLedger:
    schema_version: str
    object_attribute: tuple[ObjectAttributeLedger, ...]
    relation_edges: tuple[RelationEdgeLedger, ...]
    m_contra: float
    m_restore: float
    u_edge: float
    witness_kind: str | None
    witness_support: float
    explicit_witness: bool
    upstream_complete_edge: float
    best_complete_alternative_edge: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "object_attribute": [asdict(row) for row in self.object_attribute],
            "relation_edges": [asdict(row) for row in self.relation_edges],
            "M_contra": self.m_contra,
            "M_restore": self.m_restore,
            "U_edge": self.u_edge,
            "witness_kind": self.witness_kind,
            "witness_support": self.witness_support,
            "explicit_witness": self.explicit_witness,
            "upstream_complete_edge": self.upstream_complete_edge,
            "best_complete_alternative_edge": self.best_complete_alternative_edge,
        }


def _provenance(proposal: object) -> str:
    value = getattr(proposal, "prompt_provenance", "claim")
    return str(getattr(value, "value", value) or "claim")


def _counterfactual_for(proposal: object, atom_id: str) -> bool:
    return (
        _provenance(proposal) == "inverse"
        and str(getattr(proposal, "counterfactual_of", "") or "") == atom_id
    )


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _harmonic(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _ambiguity(scores: Sequence[float]) -> float:
    """High means that evidence is weak or top-1/top-2 are indistinguishable."""

    ranked = sorted((_bounded(value) for value in scores), reverse=True)
    if not ranked:
        return 1.0
    if len(ranked) == 1:
        return 1.0 - ranked[0]
    best, second = ranked[:2]
    return _bounded(max(1.0 - best, second / max(best, 1e-9)))


def _between_geometry(target: Box, left: Box, right: Box, width: int, height: int) -> float:
    tx = (target[0] + target[2]) / 2 / width
    ty = (target[1] + target[3]) / 2 / height
    lx = (left[0] + left[2]) / 2 / width
    ly = (left[1] + left[3]) / 2 / height
    rx = (right[0] + right[2]) / 2 / width
    ry = (right[1] + right[3]) / 2 / height
    span_x, span_y = rx - lx, ry - ly
    span2 = span_x * span_x + span_y * span_y
    if span2 <= 1e-8:
        return 0.0
    projection = ((tx - lx) * span_x + (ty - ly) * span_y) / span2
    closest_x, closest_y = lx + projection * span_x, ly + projection * span_y
    perpendicular = ((tx - closest_x) ** 2 + (ty - closest_y) ** 2) ** 0.5
    inside = max(0.0, 1.0 - abs(projection - 0.5) / 0.5)
    return _bounded(inside * max(0.0, 1.0 - perpendicular / 0.25))


def build_atom_binding_ledger(
    *,
    claim: object,
    scored_candidates: Sequence[object],
    proposals: Sequence[object],
    image_width: int,
    image_height: int,
    relation_geometry: Callable[[str, Box, Box, int, int], float],
    use_swap: bool = True,
    use_inverse: bool = True,
    use_relation_algebra: bool = True,
) -> AtomBindingLedger:
    """Construct proposal-level typed evidence without hallucination labels.

    ``claim``/candidate/proposal objects are intentionally duck typed so this
    module stays independent of the detector adapter and avoids an import
    cycle with :mod:`vsight.ccv`.
    """

    candidates = list(scored_candidates)
    references = [
        row for row in proposals
        if bool(getattr(row, "is_reference", False)) and _provenance(row) == "claim"
    ]
    inverse_proposals = [row for row in proposals if _provenance(row) == "inverse"]
    object_scores = [float(getattr(row, "object_support", 0.0)) for row in candidates]
    candidate_ambiguity = _ambiguity(object_scores) if len(candidates) > 1 else 0.0
    modifier_scores = [
        float(getattr(row, "attribute_action_support"))
        for row in candidates if getattr(row, "attribute_action_support", None) is not None
    ]
    modifier_ambiguity = _ambiguity(modifier_scores) if modifier_scores else 0.0

    modifier_atoms = [
        atom for atom in getattr(claim, "known_atoms", ())
        if str(getattr(getattr(atom, "atom_type", ""), "value", getattr(atom, "atom_type", "")))
        in {"attribute", "action"}
    ]
    nodes: list[ObjectAttributeLedger] = []
    node_complete: dict[str, float] = {}
    node_contra: list[tuple[float, str, float]] = []
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id"))
        box = tuple(getattr(candidate, "box"))
        object_support = float(getattr(candidate, "object_support", 0.0))
        attribute_support = getattr(candidate, "attribute_action_support", None)
        attribute_support = float(attribute_support) if attribute_support is not None else None
        full_support = getattr(candidate, "full_support", None)
        full_support = float(full_support) if full_support is not None else None
        other_modifier = [
            float(getattr(row, "attribute_action_support"))
            for row in candidates if row is not candidate
            and getattr(row, "attribute_action_support", None) is not None
        ]
        decoy_margin = (
            max(other_modifier) - attribute_support
            if attribute_support is not None and other_modifier else None
        )
        inverse_scores = [
            float(getattr(row, "score")) * box_iou(box, tuple(getattr(row, "box")))
            for atom in modifier_atoms for row in inverse_proposals
            if _counterfactual_for(row, str(getattr(atom, "atom_id")))
        ]
        inverse_support = max(inverse_scores, default=None)
        inverse_margin = (
            inverse_support - attribute_support
            if inverse_support is not None and attribute_support is not None else None
        )
        if inverse_margin is not None and use_inverse:
            node_contra.append((inverse_margin, "inverse_attribute", inverse_support or 0.0))
        if (
            decoy_margin is not None
            and use_swap
            and str(getattr(candidate, "source")) == "upstream"
        ):
            node_contra.append(
                (decoy_margin, "attribute_candidate_swap", max(other_modifier))
            )
        complete_values = [object_support]
        if attribute_support is not None:
            complete_values.append(attribute_support)
        if full_support is not None:
            complete_values.append(full_support)
        node_complete[candidate_id] = _harmonic(complete_values)
        nodes.append(
            ObjectAttributeLedger(
                candidate_id=candidate_id,
                bbox_xyxy=box,
                source=str(getattr(candidate, "source")),
                object_support=object_support,
                attribute_support=attribute_support,
                attribute_object_overlap=(
                    min(object_support, attribute_support)
                    if attribute_support is not None else None
                ),
                attribute_decoy_margin=decoy_margin,
                attribute_counterfactual_support=inverse_support,
                attribute_counterfactual_margin=inverse_margin,
                full_claim_support=full_support,
                candidate_ambiguity=candidate_ambiguity,
            )
        )

    relation_atoms = [
        atom for atom in getattr(claim, "known_atoms", ())
        if str(getattr(getattr(atom, "atom_type", ""), "value", getattr(atom, "atom_type", "")))
        == "relation"
    ]
    edges: list[RelationEdgeLedger] = []
    edge_complete: dict[str, list[float]] = {str(getattr(row, "candidate_id")): [] for row in candidates}
    edge_contra: list[tuple[float, str, float]] = []

    for atom in relation_atoms:
        atom_id = str(getattr(atom, "atom_id"))
        relation = str(getattr(atom, "relation") or "")
        family = relation_family(relation)
        if relation == "between" and len(references) < 2:
            for candidate in candidates:
                edges.append(
                    RelationEdgeLedger(
                        atom_id=atom_id,
                        relation=relation,
                        relation_family=family,
                        target_candidate_id=str(getattr(candidate, "candidate_id")),
                        reference_candidate_id=None,
                        secondary_reference_candidate_id=None,
                        edge_claim=None,
                        edge_alt_target=None,
                        edge_alt_reference=None,
                        edge_swap_margin=None,
                        counterfactual_edge=None,
                        inverse_consistency=None,
                        symmetry_consistency=None,
                        role_consistency=None,
                        full_edge_agreement=None,
                        edge_uncertainty=1.0,
                        m_contra=None,
                        interpretable=False,
                        unknown_reason="between_requires_two_distinguishable_references",
                    )
                )
            continue
        if not references:
            for candidate in candidates:
                edges.append(
                    RelationEdgeLedger(
                        atom_id=atom_id,
                        relation=relation,
                        relation_family=family,
                        target_candidate_id=str(getattr(candidate, "candidate_id")),
                        reference_candidate_id=None,
                        secondary_reference_candidate_id=None,
                        edge_claim=None,
                        edge_alt_target=None,
                        edge_alt_reference=None,
                        edge_swap_margin=None,
                        counterfactual_edge=None,
                        inverse_consistency=None,
                        symmetry_consistency=None,
                        role_consistency=None,
                        full_edge_agreement=None,
                        edge_uncertainty=1.0,
                        m_contra=None,
                        interpretable=False,
                        unknown_reason="reference_unobservable",
                    )
                )
            continue

        # The highest-confidence reference is the auditable claim binding;
        # other same-phrase proposals are explicit decoy slots.
        primary = max(references, key=lambda row: (float(getattr(row, "score")), str(getattr(row, "proposal_id"))))
        for candidate in candidates:
            candidate_id = str(getattr(candidate, "candidate_id"))
            target_box = tuple(getattr(candidate, "box"))
            reference_rows: list[tuple[object, object | None, float]] = []
            if relation == "between":
                for first_index, first in enumerate(references):
                    for second in references[first_index + 1 :]:
                        if box_iou(tuple(getattr(first, "box")), tuple(getattr(second, "box"))) >= 0.9:
                            continue
                        geometry = _between_geometry(
                            target_box, tuple(getattr(first, "box")), tuple(getattr(second, "box")),
                            image_width, image_height,
                        )
                        score = min(float(getattr(first, "score")), float(getattr(second, "score"))) * geometry
                        reference_rows.append((first, second, score))
                if not reference_rows:
                    edges.append(
                        RelationEdgeLedger(
                            atom_id=atom_id,
                            relation=relation,
                            relation_family=family,
                            target_candidate_id=candidate_id,
                            reference_candidate_id=None,
                            secondary_reference_candidate_id=None,
                            edge_claim=None,
                            edge_alt_target=None,
                            edge_alt_reference=None,
                            edge_swap_margin=None,
                            counterfactual_edge=None,
                            inverse_consistency=None,
                            symmetry_consistency=None,
                            role_consistency=None,
                            full_edge_agreement=None,
                            edge_uncertainty=1.0,
                            m_contra=None,
                            interpretable=False,
                            unknown_reason="between_requires_two_distinguishable_references",
                        )
                    )
                    continue
                first, second, claim_edge = max(reference_rows, key=lambda value: value[2])
                alt_reference = max((value[2] for value in reference_rows if value[:2] != (first, second)), default=None) if use_swap else None
                primary_for_edge = first
            else:
                for reference in references:
                    geometry = relation_geometry(
                        relation, target_box, tuple(getattr(reference, "box")), image_width, image_height
                    )
                    reference_rows.append((reference, None, float(getattr(reference, "score")) * geometry))
                primary_for_edge, second, claim_edge = next(
                    value for value in reference_rows if value[0] is primary
                )
                alt_reference = max(
                    (value[2] for value in reference_rows if value[0] is not primary), default=None
                ) if use_swap else None

            counterfactual_scores = []
            for proposal in inverse_proposals:
                if not use_inverse:
                    continue
                if not _counterfactual_for(proposal, atom_id):
                    continue
                inverse_relation = str(getattr(proposal, "relation", "") or INVERSE_RELATIONS.get(relation, relation))
                role_swapped = bool(getattr(proposal, "role_swapped", False))
                reference_box = tuple(getattr(primary_for_edge, "box"))
                geometry = relation_geometry(
                    inverse_relation,
                    reference_box if role_swapped else target_box,
                    target_box if role_swapped else reference_box,
                    image_width,
                    image_height,
                )
                counterfactual_scores.append(
                    float(getattr(proposal, "score"))
                    * box_iou(
                        reference_box if role_swapped else target_box,
                        tuple(getattr(proposal, "box")),
                    )
                    * geometry
                )
            counterfactual = max(counterfactual_scores, default=None)

            other_target_edges = []
            for other in candidates:
                if other is candidate:
                    continue
                other_box = tuple(getattr(other, "box"))
                if relation == "between" and second is not None:
                    geometry = _between_geometry(
                        other_box, tuple(getattr(primary_for_edge, "box")), tuple(getattr(second, "box")),
                        image_width, image_height,
                    )
                    score = min(float(getattr(primary_for_edge, "score")), float(getattr(second, "score"))) * geometry
                else:
                    score = float(getattr(primary_for_edge, "score")) * relation_geometry(
                        relation, other_box, tuple(getattr(primary_for_edge, "box")), image_width, image_height
                    )
                other_target_edges.append(score)
            alt_target = max(other_target_edges, default=None)
            competing = [claim_edge]
            if alt_reference is not None:
                competing.append(alt_reference)
            if alt_target is not None:
                competing.append(alt_target)
            if counterfactual is not None and alt_target is None and alt_reference is None:
                competing.append(counterfactual)
            uncertainty = _ambiguity(competing)
            inverse_consistency = (
                _bounded((claim_edge - counterfactual + 1.0) / 2.0)
                if counterfactual is not None else None
            )
            symmetry = None
            if relation in SYMMETRIC_RELATIONS:
                forward = relation_geometry(
                    relation, target_box, tuple(getattr(primary_for_edge, "box")), image_width, image_height
                )
                reverse = relation_geometry(
                    relation, tuple(getattr(primary_for_edge, "box")), target_box, image_width, image_height
                )
                symmetry = _bounded(1.0 - abs(forward - reverse))
            role_consistency = (
                _bounded((claim_edge - counterfactual + 1.0) / 2.0)
                if counterfactual is not None and (
                    relation in INVERSE_RELATIONS or relation in SUPPORT_RELATIONS or relation in ROLE_RELATIONS
                ) else None
            )
            full = getattr(candidate, "full_support", None)
            agreement = _harmonic([claim_edge, float(full)]) if full is not None else None
            contra_values = [value for value in (alt_reference, counterfactual) if value is not None]
            m_contra = max(contra_values) - claim_edge if contra_values else None
            # Geometry-only front/behind evidence is intentionally not a
            # negative witness. It becomes interpretable only when the inverse
            # prompt yields localized counterfactual evidence.
            interpretable = use_relation_algebra and family != "unknown" and not (
                relation in {"in_front_of", "behind"} and counterfactual is None
            )
            if m_contra is not None and interpretable:
                kind = "inverse_relation" if counterfactual is not None and counterfactual >= (alt_reference or -1) else "reference_swap"
                edge_contra.append((m_contra, kind, max(contra_values)))
            edge_complete[candidate_id].append(agreement if agreement is not None else claim_edge)
            edges.append(
                RelationEdgeLedger(
                    atom_id=atom_id,
                    relation=relation,
                    relation_family=family,
                    target_candidate_id=candidate_id,
                    reference_candidate_id=str(getattr(primary_for_edge, "proposal_id")),
                    secondary_reference_candidate_id=(
                        str(getattr(second, "proposal_id")) if second is not None else None
                    ),
                    edge_claim=claim_edge,
                    edge_alt_target=alt_target,
                    edge_alt_reference=alt_reference,
                    edge_swap_margin=(alt_reference - claim_edge if alt_reference is not None else None),
                    counterfactual_edge=counterfactual,
                    inverse_consistency=inverse_consistency,
                    symmetry_consistency=symmetry,
                    role_consistency=role_consistency,
                    full_edge_agreement=agreement,
                    edge_uncertainty=uncertainty,
                    m_contra=m_contra,
                    interpretable=interpretable,
                    unknown_reason=None if interpretable else "relation_not_reliably_interpretable",
                )
            )

    complete: dict[str, float] = {}
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id"))
        values = [node_complete.get(candidate_id, 0.0), *edge_complete.get(candidate_id, [])]
        complete[candidate_id] = _harmonic(values)
    upstream_id = next(
        (str(getattr(row, "candidate_id")) for row in candidates if str(getattr(row, "source")) == "upstream"),
        None,
    )
    upstream_complete = complete.get(upstream_id or "", 0.0)
    alternative_complete = max(
        (value for key, value in complete.items() if key != upstream_id), default=0.0
    )
    m_restore = alternative_complete - upstream_complete

    witnesses = [(value, kind, support) for value, kind, support in node_contra]
    witnesses.extend(edge_contra)
    best_witness = max(witnesses, key=lambda value: value[0], default=(0.0, None, 0.0))
    relevant_uncertainties = [
        row.edge_uncertainty for row in edges
        if row.target_candidate_id == upstream_id
    ]
    u_edge = max(
        relevant_uncertainties,
        default=modifier_ambiguity if modifier_atoms else 0.0,
    )
    return AtomBindingLedger(
        schema_version="vsight_cable_atom_binding_ledger_v1",
        object_attribute=tuple(nodes),
        relation_edges=tuple(edges),
        m_contra=float(best_witness[0]),
        m_restore=float(m_restore),
        u_edge=float(u_edge),
        witness_kind=best_witness[1],
        witness_support=float(best_witness[2]),
        explicit_witness=bool(best_witness[2] > 0 and best_witness[0] > 0),
        upstream_complete_edge=float(upstream_complete),
        best_complete_alternative_edge=float(alternative_complete),
    )
