"""Joint KEEP/SWITCH/REJECT decision layer.

The learned verifier is expected to produce the component scores. This module
contains only the deterministic, auditable normalization and deployment rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite
from typing import Mapping, Sequence


Box = tuple[float, float, float, float]


class Action(str, Enum):
    KEEP = "keep"
    SWITCH = "switch"
    REJECT = "reject"


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _box(values: Sequence[float]) -> Box:
    if len(values) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    x1, y1, x2, y2 = (_finite(value, "box coordinate") for value in values)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class SupportScores:
    object_support: float
    binding_support: float
    localization_quality: float

    def __post_init__(self) -> None:
        for name in (
            "object_support",
            "binding_support",
            "localization_quality",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class RegionCandidate:
    candidate_id: str
    box: Box
    support: SupportScores
    source: str

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be empty")
        if not self.source:
            raise ValueError("source cannot be empty")
        object.__setattr__(self, "box", _box(self.box))


@dataclass(frozen=True)
class Decision:
    action: Action
    selected_candidate_id: str | None
    selected_box: Box | None
    confidence: float
    probabilities: Mapping[str, float]
    logits: Mapping[str, float]
    reason: str


@dataclass(frozen=True)
class DecisionPolicy:
    object_weight: float = 1.0
    binding_weight: float = 1.0
    localization_weight: float = 1.0
    temperature: float = 1.0
    switch_margin: float = 0.0
    reject_margin: float = 0.0
    allow_recovery_from_null: bool = False

    def __post_init__(self) -> None:
        for name in (
            "object_weight",
            "binding_weight",
            "localization_weight",
            "temperature",
            "switch_margin",
            "reject_margin",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.switch_margin < 0 or self.reject_margin < 0:
            raise ValueError("deployment margins cannot be negative")

    def candidate_logit(self, support: SupportScores) -> float:
        return (
            self.object_weight * support.object_support
            + self.binding_weight * support.binding_support
            + self.localization_weight * support.localization_quality
        )

    def decide(
        self,
        *,
        baseline: RegionCandidate | None,
        challenger: RegionCandidate | None,
        null_logit: float,
    ) -> Decision:
        null_logit = _finite(null_logit, "null_logit")
        if baseline is not None and challenger is not None:
            if baseline.candidate_id == challenger.candidate_id:
                raise ValueError("baseline and challenger IDs must differ")

        candidates: dict[str, RegionCandidate] = {}
        logits: dict[str, float] = {"null": null_logit}
        if baseline is not None:
            candidates["baseline"] = baseline
            logits["baseline"] = self.candidate_logit(baseline.support)
        if challenger is not None and (
            baseline is not None or self.allow_recovery_from_null
        ):
            candidates["challenger"] = challenger
            logits["challenger"] = self.candidate_logit(challenger.support)

        probabilities = self._softmax(logits)
        winner = max(logits, key=lambda key: (logits[key], self._tie_priority(key)))
        reason = "joint_argmax"

        if winner == "null" and baseline is not None:
            best_box_logit = max(
                logits[key] for key in logits if key != "null"
            )
            if null_logit - best_box_logit < self.reject_margin:
                winner = "baseline"
                reason = "reject_margin_not_met"

        if winner == "challenger" and baseline is not None:
            if logits["challenger"] - logits["baseline"] < self.switch_margin:
                winner = "baseline"
                reason = "switch_margin_not_met"

        if winner == "null":
            return Decision(
                action=Action.REJECT,
                selected_candidate_id=None,
                selected_box=None,
                confidence=probabilities["null"],
                probabilities=probabilities,
                logits=logits,
                reason=reason,
            )

        selected = candidates[winner]
        action = Action.KEEP if winner == "baseline" else Action.SWITCH
        return Decision(
            action=action,
            selected_candidate_id=selected.candidate_id,
            selected_box=selected.box,
            confidence=probabilities[winner],
            probabilities=probabilities,
            logits=logits,
            reason=reason,
        )

    def _softmax(self, logits: Mapping[str, float]) -> dict[str, float]:
        maximum = max(logits.values())
        unnormalized = {
            key: exp((value - maximum) / self.temperature)
            for key, value in logits.items()
        }
        denominator = sum(unnormalized.values())
        return {key: value / denominator for key, value in unnormalized.items()}

    @staticmethod
    def _tie_priority(key: str) -> int:
        # Exact ties preserve an existing box, then reject, then switch.
        return {"challenger": 0, "null": 1, "baseline": 2}[key]
