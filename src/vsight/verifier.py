"""Candidate-level verifier score components.

The multimodal model is expected to predict these evidence scores. This module
keeps their interpretation and the null/difficulty separation deterministic so
the learned head cannot silently turn difficulty into a candidate preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CandidateEvidence:
    """Evidence predicted for one candidate region.

    Scores are unconstrained logits/features rather than probabilities. The
    contradiction feature is subtracted by ``VerifierWeights`` and should
    represent query atoms visibly falsified in this region.
    """

    object_support: float
    relation_support: float
    action_attribute_support: float
    localization_quality: float
    contradiction_score: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "object_support",
            "relation_support",
            "action_attribute_support",
            "localization_quality",
            "contradiction_score",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class VerifierWeights:
    object_weight: float = 1.0
    relation_weight: float = 1.0
    action_attribute_weight: float = 1.0
    localization_weight: float = 1.0
    contradiction_weight: float = 1.0
    null_bias: float = 0.0
    null_difficulty_weight: float = 1.0
    null_contradiction_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "object_weight",
            "relation_weight",
            "action_attribute_weight",
            "localization_weight",
            "contradiction_weight",
            "null_bias",
            "null_difficulty_weight",
            "null_contradiction_weight",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "object_weight",
            "relation_weight",
            "action_attribute_weight",
            "localization_weight",
            "contradiction_weight",
            "null_difficulty_weight",
            "null_contradiction_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def candidate_logit(self, evidence: CandidateEvidence) -> float:
        return (
            self.object_weight * evidence.object_support
            + self.relation_weight * evidence.relation_support
            + self.action_attribute_weight * evidence.action_attribute_support
            + self.localization_weight * evidence.localization_quality
            - self.contradiction_weight * evidence.contradiction_score
        )

    def null_logit(
        self,
        *,
        difficulty_score: float,
        best_contradiction_score: float = 0.0,
    ) -> float:
        """Score null using global difficulty, never candidate source identity."""
        return (
            self.null_bias
            + self.null_difficulty_weight * _finite(difficulty_score, "difficulty_score")
            + self.null_contradiction_weight
            * _finite(best_contradiction_score, "best_contradiction_score")
        )


def score_candidates(
    evidence: Mapping[str, CandidateEvidence],
    weights: VerifierWeights | None = None,
) -> dict[str, float]:
    """Return shared candidate logits keyed by caller-provided candidate IDs."""
    if not evidence:
        raise ValueError("at least one candidate is required")
    weights = weights or VerifierWeights()
    return {candidate_id: weights.candidate_logit(value) for candidate_id, value in evidence.items()}
