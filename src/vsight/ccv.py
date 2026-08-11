"""Claim-Conditioned Counterfactual Visual Verifier (CCV).

The module is deliberately detector-agnostic.  A detector adapter supplies one
normalized :class:`DetectorEvidence` object per image/query; CCV builds a small
candidate set, scores typed claim atoms, and applies an auditable three-action
policy.  No training labels or benchmark metadata are accepted by this API.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .cable import build_atom_binding_ledger
from .e2_verifier import box_iou
from .relation_supervision import (
    CATEGORY_ALIASES,
    RELATION_PATTERNS,
    category_mentions,
    extract_reference_phrase,
)


Box = tuple[float, float, float, float]


class AtomType(str, Enum):
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    ACTION = "action"
    RELATION = "relation"


class CCVAction(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    RELOCALIZE = "RELOCALIZE"


class PromptProvenance(str, Enum):
    """Semantic role of a segment in the single composite detector prompt."""

    CLAIM = "claim"
    INVERSE = "inverse"
    DECOY = "decoy"
    NULL = "null"


ATTRIBUTE_WORDS = frozenset(
    {
        "red", "blue", "green", "yellow", "black", "white", "brown",
        "gray", "grey", "orange", "pink", "purple", "wooden", "metal",
        "plastic", "striped", "spotted", "small", "large", "big", "tiny",
        "young", "old", "tall", "short", "open", "closed",
    }
)

ACTION_PATTERNS = (
    ("sitting", r"\b(?:sitting|seated)\b"),
    ("standing", r"\bstanding\b"),
    ("lying", r"\b(?:lying|laying)\b"),
    ("holding", r"\b(?:holding|holds|carrying|carries)\b"),
    ("wearing", r"\b(?:wearing|wears|dressed)\b"),
    ("riding", r"\b(?:riding|rides)\b"),
    ("walking", r"\b(?:walking|walks)\b"),
    ("running", r"\b(?:running|runs)\b"),
    ("eating", r"\b(?:eating|eats)\b"),
    ("drinking", r"\b(?:drinking|drinks)\b"),
    ("looking", r"\b(?:looking|looks)\b"),
    ("playing", r"\b(?:playing|plays)\b"),
)


def _normalized(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _box(value: Sequence[float]) -> Box:
    if len(value) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("box coordinates must be finite")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box must have positive extent")
    return result  # type: ignore[return-value]


def _unit(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _open_vocabulary_target(target_span: str) -> str | None:
    """Recover a query-only target phrase when it is outside COCO aliases."""

    value = target_span.strip(" \t.,;:!?()[]{}\"")
    value = re.sub(r"^(?:the|a|an|this|that|his|her|their)\s+", "", value)
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", value)
    while len(words) > 1 and words[0] in ATTRIBUTE_WORDS:
        words.pop(0)
    return " ".join(words) or None


@dataclass(frozen=True)
class ClaimAtom:
    atom_id: str
    atom_type: AtomType
    text: str
    known: bool = True
    relation: str | None = None
    reference_text: str | None = None

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ValueError("atom_id cannot be empty")
        if self.known and not self.text:
            raise ValueError("known atom text cannot be empty")
        if self.atom_type is AtomType.RELATION and self.known:
            if not self.relation or not self.reference_text:
                raise ValueError("known relation atoms require relation and reference")


@dataclass(frozen=True)
class ParsedClaim:
    query: str
    atoms: tuple[ClaimAtom, ...]
    full_expression: str

    @property
    def known_atoms(self) -> tuple[ClaimAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.known)

    def atoms_of_type(self, atom_type: AtomType) -> tuple[ClaimAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.atom_type is atom_type)


class TypedClaimParser:
    """Conservative rule/lexicon parser that only reads the deployment query."""

    def parse(self, query: str) -> ParsedClaim:
        text = _normalized(query)
        if not text:
            raise ValueError("query cannot be empty")
        relation_match = next(
            (
                (name, match)
                for name, pattern in RELATION_PATTERNS
                if (match := re.search(pattern, text)) is not None
            ),
            None,
        )
        target_span = text[: relation_match[1].start()] if relation_match else text
        # Never promote a known reference category from the relation suffix to
        # the target role.  Open-vocabulary RefCOCOg targets such as helmet,
        # lamp, robot, or sweater are not in the COCO-80 alias table, while
        # their reference often is.  The previous whole-query lookup therefore
        # silently swapped target and reference for those expressions.
        target_mentions = category_mentions(target_span, tuple(CATEGORY_ALIASES))
        target = target_mentions[0] if target_mentions else None
        target_text = target.category if target else _open_vocabulary_target(target_span)
        atoms: list[ClaimAtom] = [
            ClaimAtom(
                "object",
                AtomType.OBJECT,
                target_text or "",
                known=target_text is not None,
            )
        ]

        words = set(re.findall(r"[a-z]+", target_span))
        for word in sorted(words & ATTRIBUTE_WORDS):
            # "orange" is an object when it is the parsed target, not a color atom.
            if target_text == word:
                continue
            atoms.append(ClaimAtom(f"attribute:{word}", AtomType.ATTRIBUTE, word))

        for name, pattern in ACTION_PATTERNS:
            if re.search(pattern, text):
                atoms.append(ClaimAtom(f"action:{name}", AtomType.ACTION, name))

        if relation_match:
            relation, _ = relation_match
            reference = extract_reference_phrase(text)
            atoms.append(
                ClaimAtom(
                    f"relation:{relation}",
                    AtomType.RELATION,
                    relation,
                    known=bool(reference),
                    relation=relation if reference else None,
                    reference_text=reference,
                )
            )

        # Preserve an explicit unknown modifier atom when lexical material cannot
        # be typed. It is reported but never treated as contradictory evidence.
        understood = set()
        if target:
            understood.update(target.alias.split())
        elif target_text:
            understood.update(target_text.split())
        understood.update(words & ATTRIBUTE_WORDS)
        for _, pattern in ACTION_PATTERNS:
            for match in re.finditer(pattern, text):
                understood.update(match.group(0).split())
        stop = {
            "a", "an", "the", "this", "that", "his", "her", "their", "of",
            "to", "in", "on", "at", "with", "by", "and", "is", "are",
        }
        residue = sorted(words - understood - stop)
        known_category_tokens = (
            set(target.alias.split()) if target else set((target_text or "").split())
        )
        residue = [word for word in residue if word not in known_category_tokens]
        if residue and not any(atom.atom_type is AtomType.ATTRIBUTE for atom in atoms):
            atoms.append(
                ClaimAtom("modifier:unknown", AtomType.ATTRIBUTE, " ".join(residue), False)
            )
        return ParsedClaim(query=query, atoms=tuple(atoms), full_expression=text)


@dataclass(frozen=True)
class DetectorProposal:
    proposal_id: str
    box: Box
    score: float
    atom_scores: Mapping[str, float] = field(default_factory=dict)
    full_score: float | None = None
    is_reference: bool = False
    label: str | None = None
    segment_id: str | None = None
    prompt_provenance: PromptProvenance | str = PromptProvenance.CLAIM
    counterfactual_of: str | None = None
    relation: str | None = None
    role_swapped: bool = False

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("proposal_id cannot be empty")
        object.__setattr__(self, "box", _box(self.box))
        object.__setattr__(self, "score", _unit(self.score, "proposal score"))
        normalized_scores = {
            str(key): _unit(value, f"atom score {key}")
            for key, value in self.atom_scores.items()
        }
        object.__setattr__(self, "atom_scores", normalized_scores)
        if self.full_score is not None:
            object.__setattr__(
                self, "full_score", _unit(self.full_score, "full expression score")
            )
        try:
            provenance = PromptProvenance(self.prompt_provenance)
        except ValueError as exc:
            raise ValueError("invalid prompt provenance") from exc
        object.__setattr__(self, "prompt_provenance", provenance)
        if provenance is PromptProvenance.INVERSE and not self.counterfactual_of:
            raise ValueError("inverse proposals require counterfactual provenance")


@dataclass(frozen=True)
class DetectorEvidence:
    proposals: tuple[DetectorProposal, ...]
    image_width: int
    image_height: int
    evidence_complete: bool
    prompted_atom_ids: frozenset[str]
    null_support: float = 0.0
    contradiction_support: float = 0.0
    image_encoder_forwards: int = 1
    latency_ms: float | None = None
    localized_attention: Mapping[str, object] | None = None
    attention_ledger: Mapping[str, object] | None = None
    trace_ledger: Mapping[str, object] | None = None
    counterfactual_atom_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_encoder_forwards != 1:
            raise ValueError("CCV composite evidence requires exactly one image-encoder forward")
        object.__setattr__(self, "null_support", _unit(self.null_support, "null support"))
        object.__setattr__(
            self,
            "contradiction_support",
            _unit(self.contradiction_support, "contradiction support"),
        )
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency cannot be negative")


@dataclass(frozen=True)
class CandidateAtomEvidence:
    candidate_id: str
    box: Box
    source: str
    atom_support: Mapping[str, float]
    object_support: float
    attribute_action_support: float | None
    relation_support: float | None
    full_support: float | None
    conjunction: float


@dataclass(frozen=True)
class CCVThresholds:
    existence_support: float = 0.25
    claim_support: float = 0.22
    atom_support: float = 0.15
    binding_margin: float = 0.12
    alternative_support: float = 0.30
    contradiction_support: float = 0.70
    duplicate_iou: float = 0.90
    proposal_cap: int = 5
    conjunction: str = "harmonic"
    attention_weight: float = 0.0
    use_atom_types: bool = True
    use_counterfactual: bool = True
    use_reference_geometry: bool = True
    relation_distance_scale: float = 0.3
    counterfactual_margin: float = 0.15
    restoration_margin: float = 0.12
    edge_uncertainty_max: float = 0.40
    witness_support: float = 0.25
    use_swap: bool = True
    use_inverse_symmetry: bool = True
    use_relation_algebra: bool = True
    require_typed_relocalization: bool = False

    def __post_init__(self) -> None:
        for name in (
            "existence_support", "claim_support", "atom_support", "binding_margin",
            "alternative_support", "contradiction_support", "duplicate_iou",
            "attention_weight",
            "counterfactual_margin", "restoration_margin",
            "edge_uncertainty_max", "witness_support",
        ):
            _unit(getattr(self, name), name)
        if not math.isfinite(float(self.relation_distance_scale)) or self.relation_distance_scale <= 0:
            raise ValueError("relation_distance_scale must be finite and positive")
        if self.proposal_cap not in {1, 3, 5}:
            raise ValueError("proposal_cap must be one of 1, 3, or 5")
        if self.conjunction not in {"minimum", "harmonic"}:
            raise ValueError("conjunction must be minimum or harmonic")


@dataclass(frozen=True)
class CCVResult:
    action: CCVAction
    corrected_bbox: Box | None
    atom_evidence: Mapping[str, object]
    existence_margin: float
    binding_margin: float
    calibrated_risk: float
    latency_metadata: Mapping[str, float | int | None]
    reason: str
    original_bbox: Box | None
    alternative_bbox: Box | None
    claim_support: float
    alternative_support: float
    evidence_sufficient: bool
    binding_status: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "corrected_bbox": list(self.corrected_bbox) if self.corrected_bbox else None,
            "atom_evidence": dict(self.atom_evidence),
            "existence_margin": self.existence_margin,
            "binding_margin": self.binding_margin,
            "calibrated_risk": self.calibrated_risk,
            "latency_metadata": dict(self.latency_metadata),
            "reason": self.reason,
            "binding_status": self.binding_status,
        }


def _aggregate(values: Sequence[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "minimum":
        return min(values)
    if any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _relation_geometry(
    relation: str,
    target: Box,
    reference: Box,
    width: int,
    height: int,
    distance_scale: float = 0.3,
) -> float:
    tx, ty = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    rx, ry = (reference[0] + reference[2]) / 2, (reference[1] + reference[3]) / 2
    dx, dy = (tx - rx) / width, (ty - ry) / height
    distance = math.hypot(dx, dy)
    overlap = box_iou(target, reference)
    # Exposed as a sweep parameter because held_by/holding boxes often do not
    # overlap even when the relation is visually true.
    distance_scale = max(float(distance_scale), 1e-6)
    directional = {
        "to_the_left_of": float(dx < -0.02),
        "to_the_right_of": float(dx > 0.02),
        "above": float(dy < -0.02),
        "below": float(dy > 0.02),
        "behind": 0.5,
        "in_front_of": 0.5,
        "next_to": max(0.0, 1.0 - distance / 0.5),
        "beside": max(0.0, 1.0 - distance / 0.5),
        "between": 0.5,
        "on": float(dy < 0.05) * max(0.0, 1.0 - abs(dx) / 0.4),
        "sitting_on": float(dy < 0.05) * max(0.0, 1.0 - abs(dx) / 0.4),
        "standing_on": float(dy < 0.05) * max(0.0, 1.0 - abs(dx) / 0.4),
        "held_by": max(overlap, max(0.0, 1.0 - distance / distance_scale)),
        "holding": max(overlap, max(0.0, 1.0 - distance / distance_scale)),
        "wearing": max(overlap, max(0.0, 1.0 - distance / 0.2)),
        "riding": max(overlap, max(0.0, 1.0 - distance / distance_scale)),
        "with": max(0.0, 1.0 - distance / 0.6),
        "by": max(0.0, 1.0 - distance / 0.6),
    }
    return min(1.0, max(0.0, directional.get(relation, 0.0)))


class ClaimConditionedCounterfactualVerifier:
    def __init__(
        self,
        thresholds: CCVThresholds | None = None,
        parser: TypedClaimParser | None = None,
        risk_calibrator=None,
    ) -> None:
        self.thresholds = thresholds or CCVThresholds()
        self.parser = parser or TypedClaimParser()
        self.risk_calibrator = risk_calibrator

    def verify(
        self,
        image: object,
        query: str,
        upstream_bbox: Sequence[float] | None,
        detector_evidence: DetectorEvidence,
        optional_localized_attention: Mapping[str, object] | None = None,
    ) -> CCVResult:
        del image  # The detector evidence is the sole visual input to this layer.
        started = time.perf_counter()
        claim = self.parser.parse(query)
        upstream = _box(upstream_bbox) if upstream_bbox is not None else None
        candidates = self._candidates(upstream, detector_evidence)
        scored = [
            self._score_candidate(
                candidate_id,
                box,
                source,
                claim,
                detector_evidence,
                optional_localized_attention or detector_evidence.localized_attention,
            )
            for candidate_id, box, source in candidates
        ]
        ledger = build_atom_binding_ledger(
            claim=claim,
            scored_candidates=scored,
            proposals=detector_evidence.proposals,
            image_width=detector_evidence.image_width,
            image_height=detector_evidence.image_height,
            relation_geometry=lambda relation, target, reference, width, height: _relation_geometry(
                relation,
                target,
                reference,
                width,
                height,
                self.thresholds.relation_distance_scale,
            ),
            use_swap=self.thresholds.use_swap,
            use_inverse=self.thresholds.use_inverse_symmetry,
            use_relation_algebra=self.thresholds.use_relation_algebra,
        )
        claim_row = next((row for row in scored if row.source == "upstream"), None)
        alternatives = [row for row in scored if row.source == "detector"]
        alternative = max(
            alternatives,
            key=lambda row: (row.conjunction, row.object_support, row.candidate_id),
            default=None,
        )
        claim_score = claim_row.conjunction if claim_row else 0.0
        alt_score = alternative.conjunction if alternative else 0.0
        # Existence is an object-level question.  Do not turn an unsupported
        # relation/attribute atom into an absence verdict: detector evidence can
        # establish that the object is present while failing to bind the typed
        # modifier.  Full-expression support still falsifies absence directly;
        # ACCEPT/RELOCALIZE continue to require the conservative conjunction.
        object_known = any(atom.atom_type is AtomType.OBJECT for atom in claim.known_atoms)
        existence_values = []
        for row in scored:
            if object_known:
                existence_values.append(row.object_support)
            if row.full_support is not None:
                existence_values.append(row.full_support)
            if not object_known:
                existence_values.append(row.conjunction)
        existence = max(existence_values, default=0.0)
        binding_margin = alt_score - claim_score
        existence_margin = existence - self.thresholds.existence_support
        known_ids = {atom.atom_id for atom in claim.known_atoms}
        coverage = known_ids <= set(detector_evidence.prompted_atom_ids)
        sufficient = detector_evidence.evidence_complete and coverage and bool(known_ids)
        contradiction = max(
            detector_evidence.null_support, detector_evidence.contradiction_support
        )

        action = CCVAction.ACCEPT
        corrected = None
        reason = "claim_supported"
        binding_status = "BOUND"
        claim_atoms_pass = bool(claim_row) and all(
            claim_row.atom_support.get(atom_id, 0.0) >= self.thresholds.atom_support
            for atom_id in known_ids
        )
        explicit_contradiction = contradiction >= self.thresholds.contradiction_support
        absent_with_complete_evidence = sufficient and existence < self.thresholds.existence_support
        typed_witness = (
            self.thresholds.use_counterfactual
            and ledger.explicit_witness
            and ledger.m_contra >= self.thresholds.counterfactual_margin
            and ledger.witness_support >= self.thresholds.witness_support
            and ledger.u_edge <= self.thresholds.edge_uncertainty_max
        )
        typed_restoration = (
            typed_witness
            and alternative is not None
            and alt_score >= self.thresholds.alternative_support
            and ledger.m_restore >= self.thresholds.restoration_margin
        )
        if explicit_contradiction or absent_with_complete_evidence:
            action = CCVAction.REJECT
            binding_status = "CONTRADICTED"
            reason = (
                "explicit_null_or_contradiction"
                if explicit_contradiction
                else "complete_evidence_lacks_target_support"
            )
        elif typed_restoration:
            action = CCVAction.RELOCALIZE
            corrected = alternative.box
            binding_status = "RESTORED"
            reason = "typed_edge_refuted_alternative_restores_claim"
        elif (
            self.thresholds.use_counterfactual
            and
            not (
                self.thresholds.require_typed_relocalization
                and any(
                    atom.atom_type in {AtomType.ATTRIBUTE, AtomType.ACTION, AtomType.RELATION}
                    for atom in claim.known_atoms
                )
            )
            and
            alternative is not None
            and alt_score >= self.thresholds.alternative_support
            and binding_margin >= self.thresholds.binding_margin
        ):
            action = CCVAction.RELOCALIZE
            corrected = alternative.box
            binding_status = "RESTORED"
            reason = "counterfactual_region_stronger"
        elif typed_witness:
            # A ROH rejection is permitted only when an inverse/swap witness is
            # explicit and the edge is not in the ambiguity gray-zone.
            action = CCVAction.REJECT
            binding_status = "CONTRADICTED"
            reason = "typed_binding_counterfactual_contradiction"
        elif not sufficient:
            binding_status = "BINDING_UNCERTAIN"
            reason = "insufficient_evidence_preserve_upstream"
        elif not claim_atoms_pass or claim_score < self.thresholds.claim_support:
            binding_status = "BINDING_UNCERTAIN"
            reason = "binding_uncertain_preserve_upstream"
        elif ledger.u_edge > self.thresholds.edge_uncertainty_max and (
            claim.atoms_of_type(AtomType.RELATION)
            or claim.atoms_of_type(AtomType.ATTRIBUTE)
            or claim.atoms_of_type(AtomType.ACTION)
        ):
            binding_status = "BINDING_UNCERTAIN"
            reason = "edge_ambiguity_preserve_upstream"
        elif binding_margin > 0:
            reason = "alternative_margin_below_relocalize_threshold"

        confidence = self._decision_confidence(
            action, claim_score, alt_score, existence, contradiction, sufficient
        )
        raw_risk = 1.0 - confidence
        risk = (
            float(self.risk_calibrator.predict(raw_risk))
            if self.risk_calibrator is not None
            else raw_risk
        )
        risk = min(1.0, max(0.0, risk))
        elapsed = (time.perf_counter() - started) * 1000
        atom_evidence = {
            "claim": self._serialize_evidence(claim_row),
            "alternative": self._serialize_evidence(alternative),
            "known_atoms": [atom.atom_id for atom in claim.known_atoms],
            "unknown_atoms": [atom.atom_id for atom in claim.atoms if not atom.known],
            "detector_coverage": sorted(detector_evidence.prompted_atom_ids),
            "null_support": detector_evidence.null_support,
            "contradiction_support": detector_evidence.contradiction_support,
            "attention_ledger": (
                dict(detector_evidence.attention_ledger)
                if detector_evidence.attention_ledger is not None else None
            ),
            "trace_ledger": (
                dict(detector_evidence.trace_ledger)
                if detector_evidence.trace_ledger is not None else None
            ),
            "ledger": ledger.as_dict(),
        }
        return CCVResult(
            action=action,
            corrected_bbox=corrected,
            atom_evidence=atom_evidence,
            existence_margin=existence_margin,
            binding_margin=binding_margin,
            calibrated_risk=risk,
            latency_metadata={
                "detector_ms": detector_evidence.latency_ms,
                "verifier_ms": elapsed,
                "image_encoder_forwards": detector_evidence.image_encoder_forwards,
                "proposal_cap": self.thresholds.proposal_cap,
                "candidate_count": len(candidates),
            },
            reason=reason,
            original_bbox=upstream,
            alternative_bbox=alternative.box if alternative else None,
            claim_support=claim_score,
            alternative_support=alt_score,
            evidence_sufficient=sufficient,
            binding_status=binding_status,
        )

    def _candidates(
        self, upstream: Box | None, evidence: DetectorEvidence
    ) -> list[tuple[str, Box, str]]:
        rows: list[tuple[str, Box, str]] = []
        if upstream is not None:
            rows.append(("upstream", upstream, "upstream"))
        ranked = sorted(
            (
                proposal for proposal in evidence.proposals
                if not proposal.is_reference
                and proposal.prompt_provenance is PromptProvenance.CLAIM
            ),
            key=lambda row: (-(row.full_score if row.full_score is not None else row.score), row.proposal_id),
        )
        for proposal in ranked:
            if len([row for row in rows if row[2] == "detector"]) >= self.thresholds.proposal_cap:
                break
            if any(box_iou(proposal.box, box) >= self.thresholds.duplicate_iou for _, box, _ in rows):
                continue
            rows.append((proposal.proposal_id, proposal.box, "detector"))
        return rows

    def _score_candidate(
        self,
        candidate_id: str,
        box: Box,
        source: str,
        claim: ParsedClaim,
        evidence: DetectorEvidence,
        attention: Mapping[str, object] | None,
    ) -> CandidateAtomEvidence:
        atom_support: dict[str, float] = {}
        for atom in claim.known_atoms:
            if atom.atom_type is AtomType.RELATION:
                references = [row for row in evidence.proposals if row.is_reference]
                scores = [
                    row.score
                    * (
                        _relation_geometry(
                            atom.relation or "", box, row.box,
                            evidence.image_width, evidence.image_height,
                            self.thresholds.relation_distance_scale,
                        )
                        if self.thresholds.use_reference_geometry else 1.0
                    )
                    for row in references
                ]
                support = max(scores, default=0.0)
            else:
                support = max(
                    (
                        (
                            row.atom_scores.get(atom.atom_id, 0.0)
                            if self.thresholds.use_atom_types else row.score
                        )
                        * box_iou(box, row.box)
                        for row in evidence.proposals
                        if not row.is_reference
                        and row.prompt_provenance is PromptProvenance.CLAIM
                    ),
                    default=0.0,
                )
            if attention and self.thresholds.attention_weight > 0:
                candidate_attention = attention.get(candidate_id)
                if isinstance(candidate_attention, Mapping):
                    raw_attention = candidate_attention.get(atom.atom_id, support)
                else:
                    raw_attention = attention.get(
                        f"{candidate_id}:{atom.atom_id}",
                        attention.get(atom.atom_id, support),
                    )
                localized = min(1.0, max(0.0, float(raw_attention)))
                weight = self.thresholds.attention_weight
                support = (1.0 - weight) * support + weight * localized
            atom_support[atom.atom_id] = support

        full_scores = [
            (row.full_score or 0.0) * box_iou(box, row.box)
            for row in evidence.proposals
            if not row.is_reference
            and row.prompt_provenance is PromptProvenance.CLAIM
            and row.full_score is not None
        ]
        full = max(full_scores, default=None)
        object_values = [
            value
            for atom_id, value in atom_support.items()
            if atom_id == "object"
        ]
        modifier_values = [
            atom_support[atom.atom_id]
            for atom in claim.known_atoms
            if atom.atom_type in {AtomType.ATTRIBUTE, AtomType.ACTION}
        ]
        relation_values = [
            atom_support[atom.atom_id]
            for atom in claim.known_atoms
            if atom.atom_type is AtomType.RELATION
        ]
        conjunction_values = list(atom_support.values())
        if full is not None:
            conjunction_values.append(full)
        return CandidateAtomEvidence(
            candidate_id=candidate_id,
            box=box,
            source=source,
            atom_support=atom_support,
            object_support=_aggregate(object_values, self.thresholds.conjunction),
            attribute_action_support=(
                _aggregate(modifier_values, self.thresholds.conjunction)
                if modifier_values else None
            ),
            relation_support=(
                _aggregate(relation_values, self.thresholds.conjunction)
                if relation_values else None
            ),
            full_support=full,
            conjunction=_aggregate(conjunction_values, self.thresholds.conjunction),
        )

    def _decision_confidence(
        self,
        action: CCVAction,
        claim: float,
        alternative: float,
        existence: float,
        contradiction: float,
        sufficient: bool,
    ) -> float:
        if not sufficient and contradiction < self.thresholds.contradiction_support:
            return 0.5
        if action is CCVAction.REJECT:
            return max(contradiction, 1.0 - existence)
        if action is CCVAction.RELOCALIZE:
            margin_scale = max(1e-6, 1.0 - self.thresholds.binding_margin)
            return min(1.0, alternative * (0.5 + 0.5 * (alternative - claim) / margin_scale))
        return min(1.0, claim * (1.0 - max(0.0, alternative - claim)))

    @staticmethod
    def _serialize_evidence(row: CandidateAtomEvidence | None) -> object:
        if row is None:
            return None
        return {
            "candidate_id": row.candidate_id,
            "bbox_xyxy": list(row.box),
            "source": row.source,
            "atom_support": dict(row.atom_support),
            "object_support": row.object_support,
            "attribute_action_support": row.attribute_action_support,
            "relation_support": row.relation_support,
            "full_support": row.full_support,
            "conjunction": row.conjunction,
        }


@dataclass(frozen=True)
class ConditionalRelocalizationRouter:
    absolute_support_threshold: float = 0.30

    def __post_init__(self) -> None:
        _unit(self.absolute_support_threshold, "absolute support threshold")

    def route(self, result: CCVResult) -> CCVResult:
        """Keep action semantics separate while suppressing unsafe box replacement."""

        if result.action is not CCVAction.RELOCALIZE:
            return result
        if (
            result.alternative_bbox is not None
            and result.alternative_support >= self.absolute_support_threshold
        ):
            return result
        return CCVResult(
            **{
                **result.__dict__,
                "corrected_bbox": None,
                "reason": "relocalize_flagged_but_router_support_too_low",
            }
        )
