"""Evidence, action, and oracle-attribution primitives for binding audits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceState(str, Enum):
    SUPPORTED_CORRECT = "SUPPORTED_CORRECT"
    WRONG_INSTANCE = "WRONG_INSTANCE"
    ABSENT_UNSUPPORTED = "ABSENT_UNSUPPORTED"
    UNOBSERVABLE_AMBIGUOUS = "UNOBSERVABLE_AMBIGUOUS"


class BindingAction(str, Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"
    RELOCALIZE = "RELOCALIZE"


class OracleAttribution(str, Enum):
    NONE = "NONE"
    PARSE_ERR = "PARSE_ERR"
    SEE_ERR = "SEE_ERR"
    TARGET_ERR = "TARGET_ERR"
    REF_ERR = "REF_ERR"
    REL_ERR = "REL_ERR"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class OracleReplacementEvidence:
    """Observable outcomes from registered, one-stage oracle replacements."""

    parser_correct: bool | None
    target_topk_covered: bool | None
    reference_required: bool
    reference_topk_covered: bool | None = None
    target_replacement_restores: bool | None = None
    reference_replacement_restores: bool | None = None
    relation_edge_supported: bool | None = None

    def __post_init__(self) -> None:
        if not self.reference_required and self.reference_replacement_restores is not None:
            raise ValueError("reference replacement is undefined for a reference-free query")
        if not self.reference_required and self.reference_topk_covered is not None:
            raise ValueError("reference coverage is undefined for a reference-free query")
        if not self.reference_required and self.relation_edge_supported is not None:
            raise ValueError("relation evidence is undefined for a reference-free query")
        if (
            self.target_replacement_restores is not None
            and self.target_topk_covered is not True
        ):
            raise ValueError("target replacement requires confirmed Top-K coverage")
        if (
            self.reference_replacement_restores is not None
            and self.reference_topk_covered is not True
        ):
            raise ValueError("reference replacement requires confirmed Top-K coverage")


def attribute_oracle_error(evidence: OracleReplacementEvidence) -> OracleAttribution:
    """Apply the preregistered parser, coverage, target, reference, relation order."""

    if evidence.parser_correct is False:
        return OracleAttribution.PARSE_ERR
    if evidence.parser_correct is None:
        return OracleAttribution.UNRESOLVED
    if evidence.target_topk_covered is False:
        return OracleAttribution.SEE_ERR
    if evidence.target_topk_covered is None:
        return OracleAttribution.UNRESOLVED
    if evidence.reference_required and evidence.reference_topk_covered is False:
        return OracleAttribution.SEE_ERR
    if evidence.reference_required and evidence.reference_topk_covered is None:
        return OracleAttribution.UNRESOLVED
    if evidence.target_replacement_restores is True:
        return OracleAttribution.TARGET_ERR
    if evidence.reference_required and evidence.reference_replacement_restores is True:
        return OracleAttribution.REF_ERR
    if evidence.reference_required and evidence.relation_edge_supported is False:
        return OracleAttribution.REL_ERR

    complete = (
        evidence.parser_correct is True
        and evidence.target_topk_covered is True
        and (
            not evidence.reference_required
            or evidence.reference_topk_covered is True
        )
        and evidence.target_replacement_restores is False
        and (
            not evidence.reference_required
            or evidence.reference_replacement_restores is False
        )
        and (
            not evidence.reference_required
            or evidence.relation_edge_supported is True
        )
    )
    return OracleAttribution.NONE if complete else OracleAttribution.UNRESOLVED


def action_for_evidence(
    state: EvidenceState,
    *,
    safe_topk_replacement: bool = False,
) -> BindingAction:
    """Derive an action without using the action itself as evidence truth."""

    if state is EvidenceState.SUPPORTED_CORRECT:
        if safe_topk_replacement:
            raise ValueError("a supported-correct row does not need a replacement")
        return BindingAction.ACCEPT
    if state is EvidenceState.WRONG_INSTANCE and safe_topk_replacement:
        return BindingAction.RELOCALIZE
    return BindingAction.ABSTAIN


def legacy_ccv_action(action: BindingAction) -> str:
    """Serialize ABSTAIN through the existing CCV three-action interface."""

    if action is BindingAction.ABSTAIN:
        return "REJECT"
    return action.value
