"""Transparent lightweight action head and calibration for CCV.

This is intentionally a small monotonic logistic model rather than another
visual backbone. Feature extraction remains frozen; only the two action heads
and one-dimensional isotonic calibrators are fitted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .ccv import CCVAction, CCVResult


DEFAULT_REJECT_DIRECTIONS = {
    "claim_support": -1,
    "alternative_support": -1,
    "null_support": 1,
    "contradiction_support": 1,
}

DEFAULT_BINDING_DIRECTIONS = {
    "binding_margin": 1,
    "alternative_support": 1,
    "claim_support": -1,
}

DEFAULT_ABSENCE_DIRECTIONS = {
    "object_support": -1,
    "full_claim_support": -1,
    "null_support": 1,
}

DEFAULT_CONTRADICTION_DIRECTIONS = {
    "M_contra": 1,
    "witness_support": 1,
    "object_support": -1,
    "full_claim_support": -1,
    "edge_ambiguity": -1,
}

DEFAULT_RESTORATION_DIRECTIONS = {
    "M_restore": 1,
    "alternative_support": 1,
    "edge_ambiguity": -1,
}


def action_features(result: CCVResult) -> dict[str, float]:
    """Flatten only inference-safe CCV evidence for the learned action head."""

    ledger = result.atom_evidence.get("ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    claim = result.atom_evidence.get("claim")
    claim = claim if isinstance(claim, Mapping) else {}
    return {
        "claim_support": result.claim_support,
        "alternative_support": result.alternative_support,
        "binding_margin": result.binding_margin,
        "existence_margin": result.existence_margin,
        "null_support": float(result.atom_evidence.get("null_support") or 0.0),
        "contradiction_support": float(
            result.atom_evidence.get("contradiction_support") or 0.0
        ),
        "evidence_sufficient": float(result.evidence_sufficient),
        "object_support": float(claim.get("object_support") or 0.0),
        "full_claim_support": float(claim.get("full_support") or 0.0),
        "M_contra": float(ledger.get("M_contra") or 0.0),
        "M_restore": float(ledger.get("M_restore") or 0.0),
        "edge_ambiguity": float(ledger.get("U_edge") or 0.0),
        "explicit_witness": float(bool(ledger.get("explicit_witness"))),
        "witness_support": float(ledger.get("witness_support") or 0.0),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    forward = math.exp(value)
    return forward / (1.0 + forward)


@dataclass(frozen=True)
class IsotonicCalibrator:
    boundaries: tuple[float, ...]
    values: tuple[float, ...]

    @classmethod
    def fit(
        cls, scores: Sequence[float], labels: Sequence[float]
    ) -> "IsotonicCalibrator":
        if len(scores) != len(labels) or not scores:
            raise ValueError("isotonic calibration needs equal non-empty inputs")
        pairs = sorted((float(score), float(label)) for score, label in zip(scores, labels, strict=True))
        if any(not 0 <= label <= 1 for _, label in pairs):
            raise ValueError("calibration labels must be in [0, 1]")
        blocks: list[list[float]] = []  # min_x, max_x, sum_y, count
        for score, label in pairs:
            blocks.append([score, score, label, 1.0])
            while len(blocks) >= 2:
                left, right = blocks[-2], blocks[-1]
                if left[2] / left[3] <= right[2] / right[3]:
                    break
                blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
        return cls(
            boundaries=tuple(block[1] for block in blocks),
            values=tuple(block[2] / block[3] for block in blocks),
        )

    def predict(self, score: float) -> float:
        value = float(score)
        for boundary, prediction in zip(self.boundaries, self.values, strict=True):
            if value <= boundary:
                return prediction
        return self.values[-1]

    def predict_many(self, scores: Sequence[float]) -> list[float]:
        return [self.predict(score) for score in scores]

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "IsotonicCalibrator":
        return cls(
            tuple(float(value) for value in row["boundaries"]),
            tuple(float(value) for value in row["values"]),
        )


@dataclass(frozen=True)
class MonotonicLogisticModel:
    feature_names: tuple[str, ...]
    directions: tuple[int, ...]
    weights: tuple[float, ...]
    bias: float

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, float]],
        labels: Sequence[float],
        directions: Mapping[str, int],
        *,
        learning_rate: float = 0.05,
        iterations: int = 1500,
        l2: float = 1e-3,
    ) -> "MonotonicLogisticModel":
        if len(rows) != len(labels) or not rows:
            raise ValueError("training rows and labels must be equal and non-empty")
        names = tuple(directions)
        signs = tuple(int(directions[name]) for name in names)
        if any(value not in {-1, 0, 1} for value in signs):
            raise ValueError("monotonic directions must be -1, 0, or 1")
        matrix = [[float(row[name]) for name in names] for row in rows]
        targets = [float(value) for value in labels]
        if any(not 0 <= value <= 1 for value in targets):
            raise ValueError("training labels must be in [0, 1]")
        weights = [0.0] * len(names)
        positive_rate = min(1 - 1e-6, max(1e-6, sum(targets) / len(targets)))
        bias = math.log(positive_rate / (1 - positive_rate))
        for _ in range(iterations):
            predictions = [
                _sigmoid(bias + sum(weight * value for weight, value in zip(weights, row, strict=True)))
                for row in matrix
            ]
            errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
            bias -= learning_rate * sum(errors) / len(errors)
            for index, direction in enumerate(signs):
                gradient = sum(
                    error * row[index] for error, row in zip(errors, matrix, strict=True)
                ) / len(errors) + l2 * weights[index]
                weights[index] -= learning_rate * gradient
                if direction > 0:
                    weights[index] = max(0.0, weights[index])
                elif direction < 0:
                    weights[index] = min(0.0, weights[index])
        return cls(names, signs, tuple(weights), bias)

    def predict(self, row: Mapping[str, float]) -> float:
        logit = self.bias + sum(
            weight * float(row[name])
            for name, weight in zip(self.feature_names, self.weights, strict=True)
        )
        return _sigmoid(logit)

    def audit_monotonicity(self) -> bool:
        return all(
            direction == 0
            or (direction > 0 and weight >= 0)
            or (direction < 0 and weight <= 0)
            for direction, weight in zip(self.directions, self.weights, strict=True)
        )

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "MonotonicLogisticModel":
        return cls(
            tuple(str(value) for value in row["feature_names"]),
            tuple(int(value) for value in row["directions"]),
            tuple(float(value) for value in row["weights"]),
            float(row["bias"]),
        )


@dataclass(frozen=True)
class MonotonicActionHead:
    reject_model: MonotonicLogisticModel
    binding_model: MonotonicLogisticModel
    reject_calibrator: IsotonicCalibrator | None = None
    binding_calibrator: IsotonicCalibrator | None = None

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, float]],
        actions: Sequence[str | CCVAction],
        *,
        reject_directions: Mapping[str, int],
        binding_directions: Mapping[str, int],
    ) -> "MonotonicActionHead":
        normalized = [CCVAction(value) for value in actions]
        reject = MonotonicLogisticModel.fit(
            rows,
            [float(action is CCVAction.REJECT) for action in normalized],
            reject_directions,
        )
        non_reject = [
            (row, action)
            for row, action in zip(rows, normalized, strict=True)
            if action is not CCVAction.REJECT
        ]
        if not non_reject:
            raise ValueError("binding head needs ACCEPT/RELOCALIZE training rows")
        binding = MonotonicLogisticModel.fit(
            [row for row, _ in non_reject],
            [float(action is CCVAction.RELOCALIZE) for _, action in non_reject],
            binding_directions,
        )
        return cls(reject, binding)

    def calibrate(
        self,
        rows: Sequence[Mapping[str, float]],
        actions: Sequence[str | CCVAction],
    ) -> "MonotonicActionHead":
        normalized = [CCVAction(value) for value in actions]
        reject_scores = [self.reject_model.predict(row) for row in rows]
        reject_calibrator = IsotonicCalibrator.fit(
            reject_scores,
            [float(action is CCVAction.REJECT) for action in normalized],
        )
        binding_pairs = [
            (row, action)
            for row, action in zip(rows, normalized, strict=True)
            if action is not CCVAction.REJECT
        ]
        binding_calibrator = (
            IsotonicCalibrator.fit(
                [self.binding_model.predict(row) for row, _ in binding_pairs],
                [float(action is CCVAction.RELOCALIZE) for _, action in binding_pairs],
            )
            if binding_pairs else None
        )
        return MonotonicActionHead(
            self.reject_model,
            self.binding_model,
            reject_calibrator,
            binding_calibrator,
        )

    def probabilities(self, row: Mapping[str, float]) -> dict[CCVAction, float]:
        reject_raw = self.reject_model.predict(row)
        reject = (
            self.reject_calibrator.predict(reject_raw)
            if self.reject_calibrator else reject_raw
        )
        binding_raw = self.binding_model.predict(row)
        binding = (
            self.binding_calibrator.predict(binding_raw)
            if self.binding_calibrator else binding_raw
        )
        relocalize = (1.0 - reject) * binding
        accept = (1.0 - reject) * (1.0 - binding)
        return {
            CCVAction.ACCEPT: accept,
            CCVAction.REJECT: reject,
            CCVAction.RELOCALIZE: relocalize,
        }

    def predict(self, row: Mapping[str, float]) -> tuple[CCVAction, float]:
        probabilities = self.probabilities(row)
        action = max(
            probabilities,
            key=lambda value: (probabilities[value], value is CCVAction.ACCEPT),
        )
        return action, 1.0 - probabilities[action]

    def to_dict(self) -> dict[str, object]:
        def model(value: MonotonicLogisticModel) -> dict[str, object]:
            return {
                "feature_names": value.feature_names,
                "directions": value.directions,
                "weights": value.weights,
                "bias": value.bias,
            }

        def calibrator(value: IsotonicCalibrator | None) -> object:
            return None if value is None else {
                "boundaries": value.boundaries,
                "values": value.values,
            }

        return {
            "schema_version": "vsight_ccv_monotonic_action_head_v1",
            "reject_model": model(self.reject_model),
            "binding_model": model(self.binding_model),
            "reject_calibrator": calibrator(self.reject_calibrator),
            "binding_calibrator": calibrator(self.binding_calibrator),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "MonotonicActionHead":
        reject_calibrator = row.get("reject_calibrator")
        binding_calibrator = row.get("binding_calibrator")
        return cls(
            MonotonicLogisticModel.from_dict(row["reject_model"]),
            MonotonicLogisticModel.from_dict(row["binding_model"]),
            IsotonicCalibrator.from_dict(reject_calibrator)
            if isinstance(reject_calibrator, Mapping) else None,
            IsotonicCalibrator.from_dict(binding_calibrator)
            if isinstance(binding_calibrator, Mapping) else None,
        )


@dataclass(frozen=True)
class GapBalancedActionHead:
    """Three-branch CABLE head for absence, contradiction, and restoration."""

    absence_model: MonotonicLogisticModel
    contradiction_model: MonotonicLogisticModel
    restoration_model: MonotonicLogisticModel
    absence_calibrator: IsotonicCalibrator | None = None
    contradiction_calibrator: IsotonicCalibrator | None = None
    restoration_calibrator: IsotonicCalibrator | None = None

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, float]],
        actions: Sequence[str | CCVAction],
        *,
        absence_labels: Sequence[float] | None = None,
        contradiction_labels: Sequence[float] | None = None,
        contradiction_observable: Sequence[bool] | None = None,
        absence_directions: Mapping[str, int] = DEFAULT_ABSENCE_DIRECTIONS,
        contradiction_directions: Mapping[str, int] = DEFAULT_CONTRADICTION_DIRECTIONS,
        restoration_directions: Mapping[str, int] = DEFAULT_RESTORATION_DIRECTIONS,
    ) -> "GapBalancedActionHead":
        if len(rows) != len(actions) or not rows:
            raise ValueError("CABLE head needs equal non-empty rows and actions")
        normalized = [CCVAction(value) for value in actions]
        reject = [float(action is CCVAction.REJECT) for action in normalized]
        absence = list(absence_labels) if absence_labels is not None else reject
        contradiction = (
            list(contradiction_labels) if contradiction_labels is not None else reject
        )
        if len(absence) != len(rows) or len(contradiction) != len(rows):
            raise ValueError("branch labels must align with training rows")
        observable = (
            list(contradiction_observable)
            if contradiction_observable is not None else [True] * len(rows)
        )
        if len(observable) != len(rows) or not any(observable):
            raise ValueError("contradiction branch needs observable training rows")
        contradiction_rows = [row for row, keep in zip(rows, observable, strict=True) if keep]
        contradiction_targets = [
            label for label, keep in zip(contradiction, observable, strict=True) if keep
        ]
        restoration = [float(action is CCVAction.RELOCALIZE) for action in normalized]
        return cls(
            MonotonicLogisticModel.fit(rows, absence, absence_directions),
            MonotonicLogisticModel.fit(
                contradiction_rows, contradiction_targets, contradiction_directions
            ),
            MonotonicLogisticModel.fit(rows, restoration, restoration_directions),
        )

    @staticmethod
    def _calibrated(
        model: MonotonicLogisticModel,
        calibrator: IsotonicCalibrator | None,
        row: Mapping[str, float],
    ) -> float:
        raw = model.predict(row)
        return calibrator.predict(raw) if calibrator else raw

    def calibrate(
        self,
        rows: Sequence[Mapping[str, float]],
        actions: Sequence[str | CCVAction],
        *,
        absence_labels: Sequence[float] | None = None,
        contradiction_labels: Sequence[float] | None = None,
        contradiction_observable: Sequence[bool] | None = None,
    ) -> "GapBalancedActionHead":
        if len(rows) != len(actions) or not rows:
            raise ValueError("CABLE calibration needs equal non-empty inputs")
        normalized = [CCVAction(value) for value in actions]
        reject = [float(action is CCVAction.REJECT) for action in normalized]
        absence = list(absence_labels) if absence_labels is not None else reject
        contradiction = (
            list(contradiction_labels) if contradiction_labels is not None else reject
        )
        observable = (
            list(contradiction_observable)
            if contradiction_observable is not None else [True] * len(rows)
        )
        if len(observable) != len(rows) or not any(observable):
            raise ValueError("contradiction calibration needs observable rows")
        restoration = [float(action is CCVAction.RELOCALIZE) for action in normalized]
        return GapBalancedActionHead(
            self.absence_model,
            self.contradiction_model,
            self.restoration_model,
            IsotonicCalibrator.fit(
                [self.absence_model.predict(row) for row in rows], absence
            ),
            IsotonicCalibrator.fit(
                [
                    self.contradiction_model.predict(row)
                    for row, keep in zip(rows, observable, strict=True) if keep
                ],
                [
                    label for label, keep in zip(contradiction, observable, strict=True)
                    if keep
                ],
            ),
            IsotonicCalibrator.fit(
                [self.restoration_model.predict(row) for row in rows], restoration
            ),
        )

    def branch_probabilities(self, row: Mapping[str, float]) -> dict[str, float]:
        return {
            "absence": self._calibrated(
                self.absence_model, self.absence_calibrator, row
            ),
            "contradiction": self._calibrated(
                self.contradiction_model, self.contradiction_calibrator, row
            ),
            "restoration": self._calibrated(
                self.restoration_model, self.restoration_calibrator, row
            ),
        }

    def probabilities(self, row: Mapping[str, float]) -> dict[CCVAction, float]:
        branches = self.branch_probabilities(row)
        reject = max(branches["absence"], branches["contradiction"])
        relocalize = (1.0 - reject) * branches["restoration"]
        accept = (1.0 - reject) * (1.0 - branches["restoration"])
        return {
            CCVAction.ACCEPT: accept,
            CCVAction.REJECT: reject,
            CCVAction.RELOCALIZE: relocalize,
        }

    def predict(self, row: Mapping[str, float]) -> tuple[CCVAction, float]:
        probabilities = self.probabilities(row)
        action = max(
            probabilities,
            key=lambda value: (probabilities[value], value is CCVAction.ACCEPT),
        )
        return action, 1.0 - probabilities[action]

    def to_dict(self) -> dict[str, object]:
        def model(value: MonotonicLogisticModel) -> dict[str, object]:
            return {
                "feature_names": value.feature_names,
                "directions": value.directions,
                "weights": value.weights,
                "bias": value.bias,
            }

        def calibrator(value: IsotonicCalibrator | None) -> object:
            return None if value is None else {
                "boundaries": value.boundaries,
                "values": value.values,
            }

        return {
            "schema_version": "vsight_cable_gap_balanced_action_head_v1",
            "absence_model": model(self.absence_model),
            "contradiction_model": model(self.contradiction_model),
            "restoration_model": model(self.restoration_model),
            "absence_calibrator": calibrator(self.absence_calibrator),
            "contradiction_calibrator": calibrator(self.contradiction_calibrator),
            "restoration_calibrator": calibrator(self.restoration_calibrator),
            "inference_features_exclude": ["model", "hallucination_type"],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "GapBalancedActionHead":
        def calibrator(name: str) -> IsotonicCalibrator | None:
            value = row.get(name)
            return IsotonicCalibrator.from_dict(value) if isinstance(value, Mapping) else None

        return cls(
            MonotonicLogisticModel.from_dict(row["absence_model"]),
            MonotonicLogisticModel.from_dict(row["contradiction_model"]),
            MonotonicLogisticModel.from_dict(row["restoration_model"]),
            calibrator("absence_calibrator"),
            calibrator("contradiction_calibrator"),
            calibrator("restoration_calibrator"),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def apply_gap_balanced_action_head(
    result: CCVResult, head: GapBalancedActionHead
) -> CCVResult:
    """Apply learned branches while retaining CABLE's witness safety rules."""

    features = action_features(result)
    branches = head.branch_probabilities(features)
    action, risk = head.predict(features)
    corrected = None
    reason = "learned_gap_balanced_action_head"
    if action is CCVAction.REJECT:
        contradiction_dominates = branches["contradiction"] > branches["absence"]
        if contradiction_dominates and not (
            bool(features["explicit_witness"])
            and features["witness_support"] >= 0.25
            and features["M_contra"] > 0
        ):
            action = CCVAction.ACCEPT
            reason = "learned_contradiction_without_witness_preserve_upstream"
    elif action is CCVAction.RELOCALIZE:
        if result.alternative_bbox is None or features["M_restore"] <= 0:
            action = CCVAction.ACCEPT
            reason = "learned_restoration_without_complete_alternative_preserve_upstream"
        else:
            corrected = result.alternative_bbox
    return replace(
        result,
        action=action,
        corrected_bbox=corrected,
        calibrated_risk=risk,
        reason=reason,
        binding_status=(
            "RESTORED" if action is CCVAction.RELOCALIZE
            else "CONTRADICTED" if action is CCVAction.REJECT
            else result.binding_status
        ),
    )


def apply_action_head(result: CCVResult, head: MonotonicActionHead) -> CCVResult:
    """Apply a fitted head without conflating missing alternatives with rejection."""

    action, risk = head.predict(action_features(result))
    corrected = None
    reason = "learned_action_head"
    if action is CCVAction.RELOCALIZE:
        if result.alternative_bbox is None:
            action = CCVAction.ACCEPT
            reason = "learned_binding_flag_without_alternative_preserve_upstream"
        else:
            corrected = result.alternative_bbox
    return replace(
        result,
        action=action,
        corrected_bbox=corrected,
        calibrated_risk=risk,
        reason=reason,
    )


def eligible_review_record(row: Mapping[str, object]) -> bool:
    """Apply the single-project-owner exclusions before a row can train."""

    action = str(row.get("verifier_action") or "")
    eligible = (
        action in {value.value for value in CCVAction}
        and row.get("annotation_protocol") == "single_project_owner"
        and int(row.get("reviewer_count") or 0) == 1
        and float(row.get("minimum_confidence") or 0.0) >= 0.90
        and bool(row.get("source_queue_sha256"))
        and not bool(row.get("sensitive_attribute"))
        and not bool(row.get("reviewer_disagreement"))
    )
    if not eligible:
        return False
    if action == CCVAction.RELOCALIZE.value:
        boxes = row.get("router_only_corrected_boxes_xyxy")
        return _has_single_valid_review_box(boxes)
    return True


def _has_single_valid_review_box(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 1:
        return False
    box = value[0]
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return False
    try:
        coords = tuple(float(item) for item in box)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in coords) and coords[2] > coords[0] and coords[3] > coords[1]


TRI_STATE_ATOM_LABELS = frozenset({"supported", "contradicted", "unobservable"})


def eligible_cable_review_record(row: Mapping[str, object]) -> bool:
    """Require authoritative v2 atom/box truth for a learned CABLE branch."""

    if not eligible_review_record(row):
        return False
    atoms = row.get("stage_b_authoritative_atoms")
    if atoms is None:
        atoms = row.get("stage_b_consensus_atoms")
    applicable = row.get("stage_b_applicable_atoms")
    if not isinstance(atoms, Mapping) or not isinstance(applicable, Sequence):
        return False
    applicable_ids = {str(value) for value in applicable}
    if not applicable_ids or any(atoms.get(atom) not in TRI_STATE_ATOM_LABELS for atom in applicable_ids):
        return False
    if not bool(row.get("schema_gate_passed")):
        return False
    if "relation" in applicable_ids:
        relation_state = atoms.get("relation")
        if relation_state in {"supported", "contradicted"}:
            boxes = row.get("independent_reference_boxes_xyxy")
            if not _has_single_valid_review_box(boxes):
                return False
    return True


@dataclass(frozen=True)
class GapBalancedRiskController:
    """Deployable selective thresholds keyed only by task/atom/fold.

    Model identity and hallucination type are used to audit calibration safety,
    but are never serialized as lookup features.
    """

    thresholds: Mapping[str, float]
    restoration_threshold: float = 0.5
    max_edge_ambiguity: float = 0.4
    min_witness_support: float = 0.25

    @staticmethod
    def _key(task: str, atom_type: str, calibration_fold: int) -> str:
        return f"{task}|{atom_type}|{int(calibration_fold)}"

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        candidate_thresholds: Sequence[float] | None = None,
        max_added_fnr: float = 0.03,
        max_positive_miou_loss: float = 0.005,
        max_gap_delta: float = 0.0,
    ) -> "GapBalancedRiskController":
        if not rows:
            raise ValueError("gap-balanced calibration needs rows")
        candidates = sorted(
            {float(value) for value in (candidate_thresholds or [index / 100 for index in range(101)])}
        )
        if not candidates or any(not 0 <= value <= 1 for value in candidates):
            raise ValueError("candidate thresholds must be in [0, 1]")
        buckets: dict[tuple[str, int], list[Mapping[str, object]]] = {}
        for row in rows:
            task = str(row.get("task") or "")
            fold = row.get("calibration_fold")
            if not task or fold is None:
                raise ValueError("calibration rows require task and calibration_fold")
            buckets.setdefault((task, int(fold)), []).append(row)

        selected: dict[str, float] = {}
        for (task, fold), bucket in sorted(buckets.items()):
            feasible: list[tuple[float, float]] = []
            for threshold in candidates:
                by_model: dict[str, list[Mapping[str, object]]] = {}
                for row in bucket:
                    model = str(row.get("model") or "")
                    if not model:
                        raise ValueError("calibration audit requires model identity")
                    by_model.setdefault(model, []).append(row)
                gains = []
                safe = True
                for model_rows in by_model.values():
                    positives = [row for row in model_rows if bool(row.get("label_exists"))]
                    negatives = [row for row in model_rows if not bool(row.get("label_exists"))]

                    def action_for(row: Mapping[str, object]) -> CCVAction:
                        if not bool(row.get("verifier_eligible", True)):
                            return CCVAction(str(row.get("action") or "REJECT"))
                        atom = str(row.get("atom_type") or "object")
                        risk_name = "contradiction_risk" if atom in {"attribute", "relation"} else "absence_risk"
                        witness_ok = atom not in {"attribute", "relation"} or (
                            bool(row.get("explicit_witness"))
                            and float(row.get("witness_support") or 0.0) >= 0.25
                        )
                        contradicted = witness_ok and float(row.get(risk_name) or 0.0) >= threshold
                        if (
                            contradicted
                            and atom in {"attribute", "relation"}
                            and bool(row.get("has_complete_alternative"))
                            and float(row.get("restoration_probability") or 0.0) >= 0.5
                        ):
                            return CCVAction.RELOCALIZE
                        return CCVAction.REJECT if contradicted else CCVAction.ACCEPT

                    added_fnr = sum(
                        bool(row.get("original_accept", True))
                        and action_for(row) is CCVAction.REJECT
                        for row in positives
                    ) / max(1, len(positives))
                    original_miou = sum(
                        float(row.get("original_iou") or 0.0)
                        * bool(row.get("original_accept", True))
                        for row in positives
                    ) / max(1, len(positives))
                    post_miou = sum(
                        0.0 if action_for(row) is CCVAction.REJECT
                        else float(
                            row.get("alternative_iou", row.get("corrected_iou", 0.0)) or 0.0
                        ) if action_for(row) is CCVAction.RELOCALIZE
                        else float(row.get("original_iou") or 0.0)
                        for row in positives
                    ) / max(1, len(positives))
                    miou_loss = original_miou - post_miou
                    strata = {}
                    for name in ("object", "co_occurrence", "attribute", "relation"):
                        part = [row for row in negatives if str(row.get("hallucination_type")) == name]
                        strata[name] = -sum(
                            bool(row.get("original_accept", True))
                            and action_for(row) is CCVAction.REJECT
                            for row in part
                        ) / max(1, len(part))
                    boh = (strata["object"] + strata["co_occurrence"]) / 2
                    roh = (strata["attribute"] + strata["relation"]) / 2
                    gap_delta = roh - boh
                    if (
                        added_fnr > max_added_fnr + 1e-12
                        or miou_loss > max_positive_miou_loss + 1e-12
                        or gap_delta > max_gap_delta + 1e-12
                    ):
                        safe = False
                        break
                    gains.append(-sum(strata.values()) / 4)
                mean_gain = sum(gains) / len(gains) if gains else 0.0
                if safe and gains and mean_gain > 0:
                    feasible.append((mean_gain, threshold))
            if not feasible:
                raise RuntimeError(
                    f"no deployable gap-balanced threshold for task={task}, fold={fold}"
                )
            _, threshold = max(feasible, key=lambda value: (value[0], value[1]))
            atom_types = {str(row.get("atom_type") or "object") for row in bucket}
            for atom_type in atom_types:
                selected[cls._key(task, atom_type, fold)] = threshold
        return cls(selected)

    def decide(
        self,
        *,
        task: str,
        atom_type: str,
        calibration_fold: int,
        absence_risk: float,
        contradiction_risk: float,
        restoration_probability: float,
        edge_ambiguity: float,
        explicit_witness: bool,
        has_complete_alternative: bool,
        witness_support: float = 1.0,
    ) -> CCVAction:
        key = self._key(task, atom_type, calibration_fold)
        if key not in self.thresholds:
            raise KeyError(f"no calibrated CABLE threshold for {key}")
        threshold = float(self.thresholds[key])
        typed = atom_type in {"attribute", "relation"}
        contradicted = (
            contradiction_risk >= threshold
            and explicit_witness
            and witness_support >= self.min_witness_support
            and edge_ambiguity <= self.max_edge_ambiguity
        ) if typed else absence_risk >= threshold
        if (
            typed and contradicted and has_complete_alternative
            and restoration_probability >= self.restoration_threshold
        ):
            return CCVAction.RELOCALIZE
        if contradicted:
            return CCVAction.REJECT
        return CCVAction.ACCEPT

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "vsight_cable_gap_balanced_controller_v1",
            "thresholds": dict(sorted(self.thresholds.items())),
            "restoration_threshold": self.restoration_threshold,
            "max_edge_ambiguity": self.max_edge_ambiguity,
            "min_witness_support": self.min_witness_support,
            "lookup_features": ["task", "atom_type", "calibration_fold"],
            "forbidden_inference_features": ["model", "hallucination_type"],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "GapBalancedRiskController":
        return cls(
            {str(key): float(value) for key, value in dict(row["thresholds"]).items()},
            float(row.get("restoration_threshold") or 0.5),
            float(row.get("max_edge_ambiguity") or 0.4),
            float(row.get("min_witness_support") or 0.25),
        )
