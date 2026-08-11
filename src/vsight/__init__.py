"""V-SIGHT decision and experiment-integrity primitives."""

from .decision import (
    Action,
    Decision,
    DecisionPolicy,
    RegionCandidate,
    SupportScores,
)
from .verifier import CandidateEvidence, VerifierWeights, score_candidates
from .ccv import (
    AtomType,
    CCVAction,
    CCVResult,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
    ConditionalRelocalizationRouter,
    DetectorEvidence,
    DetectorProposal,
    PromptProvenance,
    TypedClaimParser,
)
from .cable import (
    AtomBindingLedger,
    ObjectAttributeLedger,
    RelationEdgeLedger,
    relation_family,
)
from .cable_crop import ConditionalCropExtension
from .attention_transport import RoleSpan, build_raft_evidence, infer_role_spans
from .trajectory_ledger import build_trajectory_ledger, select_candidate_query_ids
from .trace_bind import TraceSpan, build_trace_bind_ledger

__all__ = [
    "Action",
    "Decision",
    "DecisionPolicy",
    "RegionCandidate",
    "SupportScores",
    "CandidateEvidence",
    "VerifierWeights",
    "score_candidates",
    "AtomType",
    "CCVAction",
    "CCVResult",
    "CCVThresholds",
    "ClaimConditionedCounterfactualVerifier",
    "ConditionalRelocalizationRouter",
    "DetectorEvidence",
    "DetectorProposal",
    "PromptProvenance",
    "TypedClaimParser",
    "AtomBindingLedger",
    "ObjectAttributeLedger",
    "RelationEdgeLedger",
    "relation_family",
    "ConditionalCropExtension",
    "RoleSpan",
    "build_raft_evidence",
    "infer_role_spans",
    "TraceSpan",
    "build_trace_bind_ledger",
    "build_trajectory_ledger",
    "select_candidate_query_ids",
]
