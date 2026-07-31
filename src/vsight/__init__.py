"""V-SIGHT decision and experiment-integrity primitives."""

from .decision import (
    Action,
    Decision,
    DecisionPolicy,
    RegionCandidate,
    SupportScores,
)
from .verifier import CandidateEvidence, VerifierWeights, score_candidates

__all__ = [
    "Action",
    "Decision",
    "DecisionPolicy",
    "RegionCandidate",
    "SupportScores",
    "CandidateEvidence",
    "VerifierWeights",
    "score_candidates",
]
