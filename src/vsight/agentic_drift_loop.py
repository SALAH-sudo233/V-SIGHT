"""Bounded agentic audit loop for visual-drift triage.

The loop is intentionally detector- and model-agnostic.  Its "agents" are
structured evidence roles, not additional MLLM calls: the parser, observer,
counterfactual binder, skeptic, and arbiter operate on one frozen
``DetectorEvidence`` object.  The output is a provisional audit record; it is
not a training label until a separate human/downstream adjudication promotes
it.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from .binding_taxonomy import BindingAction, EvidenceState, legacy_ccv_action
from .ccv import (
    CCVAction,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
    DetectorEvidence,
    ParsedClaim,
    TypedClaimParser,
)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _finite(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _query_fingerprint(query: str) -> str:
    normalized = " ".join(str(query or "").casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgenticDriftConfig:
    """Frozen policy for the first, train-free agentic audit.

    ``claim_support_accept`` is intentionally stricter than the legacy CCV
    claim threshold.  Until reviewed binding labels exist, the loop should
    prefer ``ABSTAIN`` over an optimistic acceptance or replacement.
    """

    schema_version: str = "vsight_agentic_drift_loop_v1"
    max_rounds: int = 3
    proposal_cap: int = 5
    claim_support_accept: float = 0.40
    alternative_support: float = 0.30
    binding_margin: float = 0.12
    counterfactual_margin: float = 0.15
    witness_support: float = 0.25
    edge_uncertainty_max: float = 0.40
    contradiction_support: float = 0.70
    existence_support: float = 0.25
    accept_risk_max: float = 0.35
    review_priority_min: float = 0.45
    feedback_priority_weight: float = 0.20
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "alternative_margin": 0.25,
            "counterfactual": 0.25,
            "edge_uncertainty": 0.20,
            "coverage_gap": 0.15,
            "claim_weakness": 0.10,
            "agent_conflict": 0.05,
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != "vsight_agentic_drift_loop_v1":
            raise ValueError("unsupported agentic drift config schema_version")
        if isinstance(self.max_rounds, bool) or self.max_rounds not in {1, 2, 3}:
            raise ValueError("max_rounds must be 1, 2, or 3")
        if isinstance(self.proposal_cap, bool) or self.proposal_cap not in {1, 3, 5}:
            raise ValueError("proposal_cap must be one of 1, 3, or 5")
        for name in (
            "claim_support_accept", "alternative_support", "binding_margin",
            "counterfactual_margin", "witness_support", "edge_uncertainty_max",
            "contradiction_support", "existence_support", "accept_risk_max",
            "review_priority_min", "feedback_priority_weight",
        ):
            value = _finite(getattr(self, name), -1.0)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if set(self.weights) != {
            "alternative_margin", "counterfactual", "edge_uncertainty",
            "coverage_gap", "claim_weakness", "agent_conflict",
        }:
            raise ValueError("weights must contain the registered risk components")
        if any(_finite(value, -1.0) < 0.0 for value in self.weights.values()):
            raise ValueError("risk weights cannot be negative")
        if not math.isclose(
            sum(float(value) for value in self.weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("risk weights must sum to 1")

    def ccv_thresholds(self) -> CCVThresholds:
        return CCVThresholds(
            proposal_cap=self.proposal_cap,
            alternative_support=self.alternative_support,
            binding_margin=self.binding_margin,
            counterfactual_margin=self.counterfactual_margin,
            witness_support=self.witness_support,
            edge_uncertainty_max=self.edge_uncertainty_max,
            contradiction_support=self.contradiction_support,
            existence_support=self.existence_support,
            require_typed_relocalization=True,
        )


@dataclass(frozen=True)
class AgentStep:
    name: str
    status: str
    findings: tuple[str, ...] = ()
    payload: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "findings": list(self.findings),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AgenticAuditResult:
    """Provisional evidence/action result returned by the bounded loop."""

    record_id: str
    query: str
    evidence_state: EvidenceState
    action: BindingAction
    drift_risk: float
    review_priority: float
    reason_codes: tuple[str, ...]
    original_bbox: tuple[float, float, float, float] | None
    alternative_bbox: tuple[float, float, float, float] | None
    claim_support: float
    alternative_support: float
    binding_margin: float
    agent_agreement: float
    memory_status: str
    trace: tuple[AgentStep, ...]
    latency_ms: float
    legacy_action: str
    audit_disposition: str
    rounds_executed: int
    stop_reason: str
    budget_ledger: Mapping[str, object]
    feedback_match_count: int = 0
    feedback_conflict: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "vsight_agentic_drift_audit_v2",
            "record_id": self.record_id,
            "query": self.query,
            "evidence_state": self.evidence_state.value,
            "action": self.action.value,
            "legacy_action": self.legacy_action,
            "audit_disposition": self.audit_disposition,
            "drift_risk_raw": self.drift_risk,
            "review_priority": self.review_priority,
            "reason_codes": list(self.reason_codes),
            "original_bbox": list(self.original_bbox) if self.original_bbox else None,
            "alternative_bbox": list(self.alternative_bbox) if self.alternative_bbox else None,
            "claim_support": self.claim_support,
            "alternative_support": self.alternative_support,
            "binding_margin": self.binding_margin,
            "agent_agreement": self.agent_agreement,
            "memory_status": self.memory_status,
            "trace": [step.as_dict() for step in self.trace],
            "latency_ms": self.latency_ms,
            "rounds_executed": self.rounds_executed,
            "stop_reason": self.stop_reason,
            "budget_ledger": dict(self.budget_ledger),
            "feedback_match_count": self.feedback_match_count,
            "feedback_conflict": self.feedback_conflict,
        }


class AgenticDriftLoop:
    """Run a deterministic parser/observer/binder/skeptic/arbiter loop.

    The three rounds reuse one frozen :class:`DetectorEvidence` object.  They
    increase verifier-side reasoning only: static K=1, typed candidate-swap
    K=3, then the full registered K=5 counterfactual ledger.  No round is
    allowed to call an image encoder or another MLLM.
    """

    _ROUND_POLICIES = (
        {
            "round": 1,
            "candidate_cap": 1,
            "evidence_mode": "static_k1",
            "use_counterfactual": False,
            "use_swap": False,
            "use_inverse_symmetry": False,
            "use_relation_algebra": False,
        },
        {
            "round": 2,
            "candidate_cap": 3,
            "evidence_mode": "typed_swap_k3",
            "use_counterfactual": True,
            "use_swap": True,
            "use_inverse_symmetry": False,
            "use_relation_algebra": False,
        },
        {
            "round": 3,
            "candidate_cap": 5,
            "evidence_mode": "registered_full_k5",
            "use_counterfactual": True,
            "use_swap": True,
            "use_inverse_symmetry": True,
            "use_relation_algebra": True,
        },
    )

    def __init__(
        self,
        config: AgenticDriftConfig | None = None,
        *,
        parser: TypedClaimParser | None = None,
        verifier: ClaimConditionedCounterfactualVerifier | None = None,
        memory: object | None = None,
    ) -> None:
        self.config = config or AgenticDriftConfig()
        self.parser = parser or TypedClaimParser()
        self.verifier = verifier or ClaimConditionedCounterfactualVerifier(
            self.config.ccv_thresholds(), parser=self.parser
        )
        # Memory is deliberately duck-typed so this module does not own the
        # persistence format.  It can only annotate the trace and review
        # priority; it is never consulted as evidence truth or action policy.
        self.memory = memory

    def audit(
        self,
        *,
        record_id: str,
        query: str,
        upstream_bbox: Sequence[float] | None,
        detector_evidence: DetectorEvidence,
        context: Mapping[str, object] | None = None,
    ) -> AgenticAuditResult:
        del context  # Context is retained by the caller/memory, never used as evidence.
        started = time.perf_counter()
        trace: list[AgentStep] = []
        original = self._valid_box(upstream_bbox)

        try:
            claim = self.parser.parse(query)
            known_ids = [atom.atom_id for atom in claim.known_atoms]
            trace.append(
                AgentStep(
                    "claim_parser",
                    "PARSE_OK" if known_ids else "PARSE_CONFLICT",
                    ("UNKNOWN_ATOM" if any(not atom.known for atom in claim.atoms) else "",),
                    {
                        "known_atom_ids": known_ids,
                        "atom_types": [atom.atom_type.value for atom in claim.known_atoms],
                    },
                )
            )
            # Empty finding strings are removed before serialization.
            trace[-1] = AgentStep(
                trace[-1].name,
                trace[-1].status,
                tuple(value for value in trace[-1].findings if value),
                trace[-1].payload,
            )
        except (TypeError, ValueError) as exc:
            reasons = ["PARSE_CONFLICT", "REVIEW_REQUIRED_UNCERTAIN"]
            trace.append(AgentStep("claim_parser", "PARSE_CONFLICT", ("PARSE_CONFLICT",), {"error": str(exc)}))
            return self._terminal(
                record_id, query, original, EvidenceState.UNOBSERVABLE_AMBIGUOUS,
                BindingAction.ABSTAIN, reasons, trace, started,
                detector_evidence=detector_evidence,
                rounds_executed=0,
                stop_reason="PARSE_CONFLICT",
            )

        if not claim.known_atoms:
            reasons = ["PARSE_CONFLICT", "REVIEW_REQUIRED_UNCERTAIN"]
            return self._terminal(
                record_id, query, original, EvidenceState.UNOBSERVABLE_AMBIGUOUS,
                BindingAction.ABSTAIN, reasons, trace, started,
                detector_evidence=detector_evidence,
                rounds_executed=0,
                stop_reason="NO_KNOWN_CLAIM_ATOMS",
            )

        rounds_executed = 0
        caps_evaluated: list[int] = []
        stop_reason = "MAX_ROUNDS_REACHED"
        ccv_result = None
        state = EvidenceState.UNOBSERVABLE_AMBIGUOUS
        action = BindingAction.ABSTAIN
        reasons: list[str] = ["REVIEW_REQUIRED_UNCERTAIN"]
        agreement = 0.0
        risk = 1.0
        action_reasons: list[str] = ["REVIEW_REQUIRED_UNCERTAIN"]

        for policy in self._ROUND_POLICIES[: self.config.max_rounds]:
            round_number = int(policy["round"])
            candidate_cap = min(int(policy["candidate_cap"]), self.config.proposal_cap)
            rounds_executed += 1
            caps_evaluated.append(candidate_cap)
            trace.append(
                AgentStep(
                    "round_controller",
                    "ROUND_STARTED",
                    (),
                    {
                        "round": round_number,
                        "candidate_cap": candidate_cap,
                        "evidence_mode": policy["evidence_mode"],
                        "frozen_detector_evidence": True,
                        "additional_image_encoder_forwards": 0,
                    },
                )
            )
            observer = self._observe(
                detector_evidence,
                claim,
                round_number=round_number,
                candidate_cap=candidate_cap,
                counterfactual_enabled=bool(policy["use_counterfactual"]),
            )
            trace.append(observer)
            try:
                verifier = self._verifier_for_round(policy, candidate_cap)
                ccv_result = verifier.verify(None, query, original, detector_evidence)
            except (TypeError, ValueError) as exc:
                reasons = ["EVIDENCE_SCHEMA_CONFLICT", "REVIEW_REQUIRED_UNCERTAIN"]
                trace.append(
                    AgentStep(
                        "counterfactual_binder",
                        "BINDER_ERROR",
                        ("EVIDENCE_SCHEMA_CONFLICT",),
                        {"round": round_number, "error": str(exc)},
                    )
                )
                return self._terminal(
                    record_id, query, original,
                    EvidenceState.UNOBSERVABLE_AMBIGUOUS,
                    BindingAction.ABSTAIN, reasons, trace, started,
                    detector_evidence=detector_evidence,
                    rounds_executed=rounds_executed,
                    stop_reason="EVIDENCE_SCHEMA_CONFLICT",
                    caps_evaluated=caps_evaluated,
                )

            binder = self._bind(ccv_result, round_number=round_number)
            trace.append(binder)
            skeptic = self._skeptic(claim, detector_evidence, ccv_result, binder)
            trace.append(skeptic)
            round_reasons = list(skeptic.findings)
            state = self._state(
                claim, detector_evidence, ccv_result, binder,
                round_reasons, original,
            )
            agreement = self._agreement((trace[0], observer, binder, skeptic))
            risk = self._risk(ccv_result, binder, skeptic, agreement)
            action, action_reasons = self._arbiter(
                state, ccv_result, binder, skeptic, original, risk
            )
            round_reasons.extend(action_reasons)
            reasons = list(dict.fromkeys(value for value in round_reasons if value))
            maybe_stop = self._round_stop_reason(
                round_number=round_number,
                claim=claim,
                evidence=detector_evidence,
                observer=observer,
                state=state,
                action=action,
            )
            is_final_budget_round = round_number >= self.config.max_rounds
            if maybe_stop is not None:
                stop_reason = maybe_stop
            elif is_final_budget_round:
                stop_reason = "MAX_ROUNDS_REACHED"
            else:
                trace.append(
                    AgentStep(
                        "round_controller",
                        "CONTINUE",
                        ("ESCALATE_UNRESOLVED",),
                        {
                            "round": round_number,
                            "next_round": round_number + 1,
                            "evidence_state": state.value,
                            "action": action.value,
                        },
                    )
                )
                continue
            trace.append(
                AgentStep(
                    "round_controller",
                    "STOP",
                    (stop_reason,),
                    {
                        "round": round_number,
                        "evidence_state": state.value,
                        "action": action.value,
                    },
                )
            )
            break

        if ccv_result is None:  # Defensive: max_rounds validation makes this unreachable.
            raise RuntimeError("agentic drift loop executed no verifier round")

        # Retrieval happens after the action is selected.  Feedback therefore
        # cannot become a hidden evidence or action signal; it only affects
        # the priority of a future human recheck.
        memory_matches = []
        if self.memory is not None and hasattr(self.memory, "retrieve"):
            query_stratum = self._query_stratum(claim)
            try:
                memory_matches = self.memory.retrieve(
                    reason_codes=reasons,
                    limit=3,
                    query_stratum=query_stratum,
                    min_overlap_ratio=0.50,
                    query_fingerprint=_query_fingerprint(query),
                )
            except TypeError:
                # Preserve compatibility with small external memory adapters
                # that implement only the original retrieve signature.
                memory_matches = self.memory.retrieve(reason_codes=reasons, limit=3)
            trace.append(
                AgentStep(
                    "memory_retriever",
                    "MATCHED" if memory_matches else "NOVEL",
                    ("MEMORY_MATCH",) if memory_matches else ("MEMORY_NOVEL",),
                    {
                        "match_count": len(memory_matches),
                        "matched_record_ids": [str(row.get("record_id")) for row in memory_matches],
                        "verified_match_count": sum(
                            str(row.get("memory_status")) == "VERIFIED_FEEDBACK"
                            for row in memory_matches
                        ),
                        "truth_used": False,
                        "action_policy_used": False,
                    },
                )
            )
        verified_matches = [
            row for row in memory_matches
            if str(row.get("memory_status")) == "VERIFIED_FEEDBACK"
        ]
        feedback_conflict = any(
            str(row.get("feedback_evidence_state") or "") not in {"", state.value}
            for row in verified_matches
        )
        priority = _clip(0.70 * risk + 0.30 * (1.0 - agreement))
        # Human feedback is a review-priority hint only.  It is intentionally
        # applied after the action has been selected and cannot alter policy.
        feedback_signal = _clip(len(verified_matches) / 3.0)
        feedback_boost = self.config.feedback_priority_weight * _clip(
            0.35 * feedback_signal + 0.65 * float(feedback_conflict)
        )
        priority = _clip(priority + feedback_boost)
        memory_status = "SELF_CONFLICT" if agreement < 0.67 else (
            "FEEDBACK_CONFLICT" if feedback_conflict else (
                "FEEDBACK_MATCHED" if verified_matches else (
                    "REVIEW_REQUIRED"
                    if action is BindingAction.ABSTAIN
                    or priority >= self.config.review_priority_min
                    else "SELF_CONSISTENT"
                )
            )
        )
        disposition = self._audit_disposition(action)
        trace.append(
            AgentStep(
                "policy_arbiter",
                action.value,
                tuple(action_reasons),
                {
                    "evidence_state": state.value,
                    "drift_risk_raw": risk,
                    "review_priority": priority,
                    "memory_status": memory_status,
                    "audit_disposition": disposition,
                    "rounds_executed": rounds_executed,
                    "stop_reason": stop_reason,
                },
            )
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return AgenticAuditResult(
            record_id=str(record_id),
            query=query,
            evidence_state=state,
            action=action,
            drift_risk=risk,
            review_priority=priority,
            reason_codes=tuple(reasons),
            original_bbox=original,
            alternative_bbox=(
                tuple(ccv_result.alternative_bbox)
                if ccv_result.alternative_bbox is not None else None
            ),
            claim_support=_clip(ccv_result.claim_support),
            alternative_support=_clip(ccv_result.alternative_support),
            binding_margin=_finite(ccv_result.binding_margin),
            agent_agreement=agreement,
            memory_status=memory_status,
            trace=tuple(trace),
            latency_ms=elapsed,
            legacy_action=legacy_ccv_action(action),
            audit_disposition=disposition,
            rounds_executed=rounds_executed,
            stop_reason=stop_reason,
            budget_ledger=self._budget_ledger(
                detector_evidence,
                rounds_executed=rounds_executed,
                caps_evaluated=caps_evaluated,
            ),
            feedback_match_count=len(verified_matches),
            feedback_conflict=feedback_conflict,
        )

    def _verifier_for_round(
        self, policy: Mapping[str, object], candidate_cap: int
    ) -> ClaimConditionedCounterfactualVerifier:
        """Clone the verifier policy while reusing the frozen evidence object."""

        thresholds = replace(
            self.verifier.thresholds,
            proposal_cap=candidate_cap,
            use_counterfactual=bool(policy["use_counterfactual"]),
            use_swap=bool(policy["use_swap"]),
            use_inverse_symmetry=bool(policy["use_inverse_symmetry"]),
            use_relation_algebra=bool(policy["use_relation_algebra"]),
        )
        return ClaimConditionedCounterfactualVerifier(
            thresholds,
            parser=self.parser,
            risk_calibrator=getattr(self.verifier, "risk_calibrator", None),
        )

    def _observe(
        self,
        evidence: DetectorEvidence,
        claim: ParsedClaim,
        *,
        round_number: int,
        candidate_cap: int,
        counterfactual_enabled: bool,
    ) -> AgentStep:
        candidates = [
            row for row in evidence.proposals
            if not row.is_reference and str(getattr(row.prompt_provenance, "value", row.prompt_provenance)) == "claim"
        ]
        references = [row for row in evidence.proposals if row.is_reference]
        inverses = [
            row for row in evidence.proposals
            if str(getattr(row.prompt_provenance, "value", row.prompt_provenance)) == "inverse"
        ]
        known_ids = {atom.atom_id for atom in claim.known_atoms}
        missing = sorted(known_ids - set(evidence.prompted_atom_ids))
        findings = []
        if not evidence.evidence_complete:
            findings.append("EVIDENCE_INCOMPLETE")
        if missing:
            findings.append("ATOM_COVERAGE_GAP")
        if not candidates:
            findings.append("NO_TARGET_CANDIDATE")
        return AgentStep(
            "evidence_observer",
            "OBSERVED" if not findings else "OBSERVATION_GAP",
            tuple(findings),
            {
                "round": round_number,
                "candidate_cap": candidate_cap,
                "available_candidate_proposals": len(candidates),
                "reference_count": len(references),
                "counterfactual_count": len(inverses),
                "counterfactual_enabled": counterfactual_enabled,
                "prompted_atom_ids": sorted(evidence.prompted_atom_ids),
                "missing_atom_ids": missing,
                "null_support": evidence.null_support,
                "contradiction_support": evidence.contradiction_support,
            },
        )

    def _bind(self, result, *, round_number: int) -> AgentStep:
        ledger = result.atom_evidence.get("ledger") or {}
        witness = bool(ledger.get("explicit_witness"))
        m_contra = _finite(ledger.get("M_contra"))
        witness_support = _finite(ledger.get("witness_support"))
        edge_uncertainty = _finite(ledger.get("U_edge"), 1.0)
        alternative = _clip(result.alternative_support)
        margin = _finite(result.binding_margin)
        strong_alternative = (
            alternative >= self.config.alternative_support
            and margin >= self.config.binding_margin
        )
        typed_witness = (
            witness
            and m_contra >= self.config.counterfactual_margin
            and witness_support >= self.config.witness_support
            and edge_uncertainty <= self.config.edge_uncertainty_max
        )
        findings = []
        if strong_alternative:
            findings.append("ALTERNATIVE_DOMINATES")
        if typed_witness:
            findings.append("TYPED_COUNTERFACTUAL_WITNESS")
        if edge_uncertainty > self.config.edge_uncertainty_max:
            findings.append("EDGE_UNCERTAIN")
        return AgentStep(
            "counterfactual_binder",
            "WITNESS_FOUND" if typed_witness else ("ALTERNATIVE_FOUND" if strong_alternative else "NO_SAFE_WITNESS"),
            tuple(findings),
            {
                "round": round_number,
                "claim_support": _clip(result.claim_support),
                "alternative_support": alternative,
                "binding_margin": margin,
                "m_contra": m_contra,
                "witness_support": witness_support,
                "edge_uncertainty": edge_uncertainty,
                "witness_kind": ledger.get("witness_kind"),
                "explicit_witness": witness,
                "alternative_bbox": list(result.alternative_bbox) if result.alternative_bbox else None,
            },
        )

    def _skeptic(self, claim, evidence, result, binder: AgentStep) -> AgentStep:
        findings = list(binder.findings)
        if not result.evidence_sufficient:
            findings.append("EVIDENCE_NOT_SUFFICIENT")
        if result.claim_support < self.config.claim_support_accept:
            findings.append("CLAIM_SUPPORT_WEAK")
        if evidence.null_support >= self.config.contradiction_support or evidence.contradiction_support >= self.config.contradiction_support:
            findings.append("EXPLICIT_NULL_OR_CONTRADICTION")
        if any(atom.atom_type.value == "relation" for atom in claim.known_atoms) and _finite(binder.payload.get("edge_uncertainty"), 1.0) > self.config.edge_uncertainty_max:
            findings.append("RELATION_EDGE_UNCERTAIN")
        return AgentStep(
            "skeptic",
            "CONFLICT" if findings else "NO_CONFLICT",
            tuple(dict.fromkeys(findings)),
            {"legacy_ccv_action": result.action.value, "finding_count": len(set(findings))},
        )

    def _state(self, claim, evidence, result, binder, reasons, original):
        if original is None:
            return EvidenceState.UNOBSERVABLE_AMBIGUOUS
        if not result.evidence_sufficient or not claim.known_atoms:
            return EvidenceState.UNOBSERVABLE_AMBIGUOUS
        typed_witness = "TYPED_COUNTERFACTUAL_WITNESS" in binder.findings
        # A score margin alone is not identity evidence.  Only an explicit,
        # typed counterfactual witness can promote a row to WRONG_INSTANCE.
        if typed_witness:
            return EvidenceState.WRONG_INSTANCE
        if evidence.null_support >= self.config.contradiction_support or evidence.contradiction_support >= self.config.contradiction_support:
            return EvidenceState.ABSENT_UNSUPPORTED
        if result.existence_margin < 0.0:
            return EvidenceState.ABSENT_UNSUPPORTED
        if result.claim_support >= self.config.claim_support_accept and not any(
            code in reasons for code in {"EVIDENCE_NOT_SUFFICIENT", "EDGE_UNCERTAIN", "RELATION_EDGE_UNCERTAIN"}
        ):
            return EvidenceState.SUPPORTED_CORRECT
        return EvidenceState.UNOBSERVABLE_AMBIGUOUS

    def _arbiter(self, state, result, binder, skeptic, original, risk):
        reasons = []
        typed = "TYPED_COUNTERFACTUAL_WITNESS" in binder.findings
        alternative = "ALTERNATIVE_DOMINATES" in binder.findings
        edge = _finite(binder.payload.get("edge_uncertainty"), 1.0)
        safe_relocalize = (
            original is not None and state is EvidenceState.WRONG_INSTANCE
            and result.alternative_bbox is not None and typed and alternative
            and edge <= self.config.edge_uncertainty_max
        )
        if safe_relocalize:
            reasons.append("SAFE_TOPK_REPLACEMENT")
            return BindingAction.RELOCALIZE, reasons
        if (
            state is EvidenceState.SUPPORTED_CORRECT
            and not skeptic.findings
            and risk <= self.config.accept_risk_max
        ):
            reasons.append("SUPPORTED_WITHOUT_SKEPTIC_CONFLICT")
            return BindingAction.ACCEPT, reasons
        if state is EvidenceState.SUPPORTED_CORRECT and risk > self.config.accept_risk_max:
            reasons.append("ACCEPT_RISK_ABOVE_GATE")
        reasons.append("REVIEW_REQUIRED_UNCERTAIN")
        return BindingAction.ABSTAIN, reasons

    def _risk(self, result, binder, skeptic, agreement):
        margin = max(0.0, _finite(result.binding_margin))
        components = {
            "alternative_margin": _clip(margin / max(1e-6, 1.0 - self.config.binding_margin)),
            "counterfactual": _clip(_finite(binder.payload.get("m_contra")) / 0.5),
            "edge_uncertainty": _clip(_finite(binder.payload.get("edge_uncertainty"), 1.0)),
            "coverage_gap": 0.0 if result.evidence_sufficient else 1.0,
            "claim_weakness": 1.0 - _clip(result.claim_support),
            "agent_conflict": 1.0 - agreement,
        }
        return _clip(sum(self.config.weights[name] * components[name] for name in components))

    @staticmethod
    def _agreement(trace: Sequence[AgentStep]) -> float:
        statuses = [step.status for step in trace if step.name in {"claim_parser", "evidence_observer", "counterfactual_binder", "skeptic"}]
        if not statuses:
            return 0.0
        conflict = sum(status in {"PARSE_CONFLICT", "OBSERVATION_GAP", "NO_SAFE_WITNESS", "CONFLICT", "BINDER_ERROR"} for status in statuses)
        return _clip(1.0 - conflict / len(statuses))

    def _round_stop_reason(
        self,
        *,
        round_number: int,
        claim: ParsedClaim,
        evidence: DetectorEvidence,
        observer: AgentStep,
        state: EvidenceState,
        action: BindingAction,
    ) -> str | None:
        """Return a conservative early-stop reason, or request escalation."""

        if "EVIDENCE_INCOMPLETE" in observer.findings or "ATOM_COVERAGE_GAP" in observer.findings:
            return "FROZEN_EVIDENCE_INSUFFICIENT"
        if state is EvidenceState.UNOBSERVABLE_AMBIGUOUS and action is BindingAction.ABSTAIN:
            if "NO_TARGET_CANDIDATE" in observer.findings:
                return "NO_TARGET_CANDIDATE"
        if (
            evidence.null_support >= self.config.contradiction_support
            or evidence.contradiction_support >= self.config.contradiction_support
        ):
            return "EXPLICIT_NULL_OR_CONTRADICTION"
        if action is BindingAction.RELOCALIZE:
            return "SAFE_TYPED_RELOCALIZATION"
        if action is BindingAction.ACCEPT:
            typed_atoms = any(
                atom.atom_type.value in {"attribute", "action", "relation"}
                for atom in claim.known_atoms
            )
            # Typed claims reach the full registered ledger before acceptance;
            # object-only claims can stop after a strong static check.
            if not typed_atoms or round_number >= 3:
                return "CONFIDENT_ACCEPT"
        return None

    @staticmethod
    def _audit_disposition(action: BindingAction) -> str:
        if action is BindingAction.ABSTAIN:
            return "REVIEW_REQUIRED"
        if action is BindingAction.RELOCALIZE:
            return "PROVISIONAL_RELOCALIZE"
        return "PROVISIONAL_ACCEPT"

    def _budget_ledger(
        self,
        evidence: DetectorEvidence,
        *,
        rounds_executed: int,
        caps_evaluated: Sequence[int],
    ) -> dict[str, object]:
        return {
            "round_budget": self.config.max_rounds,
            "rounds_executed": int(rounds_executed),
            "verifier_passes": int(rounds_executed),
            "candidate_cap_budget": self.config.proposal_cap,
            "candidate_caps_evaluated": [int(value) for value in caps_evaluated],
            "frozen_detector_evidence_reused": True,
            "input_detector_image_encoder_forwards": evidence.image_encoder_forwards,
            "additional_detector_image_encoder_forwards": 0,
            "additional_upstream_mllm_calls": 0,
            "external_teacher_calls": 0,
        }

    @staticmethod
    def _query_stratum(claim: ParsedClaim) -> str:
        atom_types = {atom.atom_type.value for atom in claim.known_atoms}
        if "relation" in atom_types:
            return "relation"
        if "action" in atom_types:
            return "action"
        if "attribute" in atom_types:
            return "attribute"
        return "object"

    @staticmethod
    def _valid_box(value: Sequence[float] | None):
        if value is None or len(value) != 4:
            return None
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in result) or result[2] <= result[0] or result[3] <= result[1]:
            return None
        return result

    def _terminal(
        self,
        record_id,
        query,
        original,
        state,
        action,
        reasons,
        trace,
        started,
        *,
        detector_evidence,
        rounds_executed,
        stop_reason,
        caps_evaluated=(),
    ):
        disposition = self._audit_disposition(action)
        trace.append(
            AgentStep(
                "policy_arbiter",
                action.value,
                tuple(dict.fromkeys(reasons)),
                {
                    "evidence_state": state.value,
                    "audit_disposition": disposition,
                    "rounds_executed": rounds_executed,
                    "stop_reason": stop_reason,
                },
            )
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return AgenticAuditResult(
            record_id=str(record_id), query=query, evidence_state=state, action=action,
            drift_risk=1.0, review_priority=1.0, reason_codes=tuple(dict.fromkeys(reasons)),
            original_bbox=original, alternative_bbox=None, claim_support=0.0,
            alternative_support=0.0, binding_margin=0.0, agent_agreement=0.0,
            memory_status="REVIEW_REQUIRED", trace=tuple(trace), latency_ms=elapsed,
            legacy_action=legacy_ccv_action(action),
            audit_disposition=disposition,
            rounds_executed=rounds_executed,
            stop_reason=stop_reason,
            budget_ledger=self._budget_ledger(
                detector_evidence,
                rounds_executed=rounds_executed,
                caps_evaluated=caps_evaluated,
            ),
        )
