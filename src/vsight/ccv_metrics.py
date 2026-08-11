"""Evaluation metrics that keep CCV rejection and relocalization separate."""

from __future__ import annotations

from collections import Counter
import random
from typing import Mapping, Sequence

from .ccv import CCVAction


NEGATIVE_STRATA = ("object", "co_occurrence", "attribute", "relation")


def _rate(numerator: int | float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def evaluate_safety(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Evaluate accept/reject safety; corrected boxes do not alter FAR semantics."""

    positives = [row for row in rows if bool(row["label_exists"])]
    negatives = [row for row in rows if not bool(row["label_exists"])]
    if not positives or not negatives:
        raise ValueError("safety evaluation requires positive and negative rows")

    def post_accept(row: Mapping[str, object]) -> bool:
        return CCVAction(str(row["action"])) is not CCVAction.REJECT

    original_positive_accepts = sum(bool(row.get("original_accept", True)) for row in positives)
    post_positive_accepts = sum(post_accept(row) for row in positives)
    original_negative_accepts = sum(bool(row.get("original_accept", True)) for row in negatives)
    post_negative_accepts = sum(post_accept(row) for row in negatives)
    original_miou = sum(
        float(row.get("original_iou") or 0.0) * bool(row.get("original_accept", True))
        for row in positives
    ) / len(positives)
    post_miou = sum(
        float(row.get("selected_iou", row.get("original_iou")) or 0.0)
        * post_accept(row)
        for row in positives
    ) / len(positives)
    by_stratum = {}
    for name in NEGATIVE_STRATA:
        subset = [row for row in negatives if row.get("hallucination_type") == name]
        before = sum(bool(row.get("original_accept", True)) for row in subset)
        after = sum(post_accept(row) for row in subset)
        by_stratum[name] = {
            "records": len(subset),
            "original_far": _rate(before, len(subset)),
            "post_far": _rate(after, len(subset)),
            "delta_far": _rate(after - before, len(subset)),
        }

    def average(names: tuple[str, str], field: str) -> float | None:
        values = [by_stratum[name][field] for name in names]
        return sum(values) / len(values) if all(value is not None for value in values) else None

    original_boh = average(("object", "co_occurrence"), "original_far")
    post_boh = average(("object", "co_occurrence"), "post_far")
    original_roh = average(("attribute", "relation"), "original_far")
    post_roh = average(("attribute", "relation"), "post_far")
    action_counts = Counter(str(row["action"]) for row in rows)
    return {
        "records": len(rows),
        "positive_records": len(positives),
        "negative_records": len(negatives),
        "original_far": original_negative_accepts / len(negatives),
        "post_far": post_negative_accepts / len(negatives),
        "delta_far": (post_negative_accepts - original_negative_accepts) / len(negatives),
        "added_fnr": (
            (len(positives) - post_positive_accepts)
            - (len(positives) - original_positive_accepts)
        ) / len(positives),
        "original_positive_miou": original_miou,
        "post_positive_miou": post_miou,
        "positive_miou_delta": post_miou - original_miou,
        "negative_strata": by_stratum,
        "original_boh_far": original_boh,
        "post_boh_far": post_boh,
        "original_roh_far": original_roh,
        "post_roh_far": post_roh,
        "roh_minus_boh_gap_delta": (
            (post_roh - post_boh) - (original_roh - original_boh)
            if None not in {original_boh, post_boh, original_roh, post_roh}
            else None
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "reject_rate": action_counts[CCVAction.REJECT.value] / len(rows),
        "blanket_rejection": action_counts[CCVAction.REJECT.value] == len(rows),
    }


def evaluate_grouped_safety(
    rows: Sequence[Mapping[str, object]],
    group_fields: Sequence[str] = ("model", "task"),
) -> dict[str, object]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in group_fields)
        if any(not value for value in key):
            raise ValueError(f"evaluation row is missing grouping field: {group_fields}")
        grouped.setdefault(key, []).append(row)
    return {
        "group_fields": list(group_fields),
        "groups": {
            "|".join(key): evaluate_safety(group_rows)
            for key, group_rows in sorted(grouped.items())
        },
    }


def audit_safety_gates(
    grouped: Mapping[str, object],
    *,
    max_added_fnr: float = 0.03,
    max_positive_miou_loss: float = 0.005,
    max_gap_delta: float = 0.0,
) -> dict[str, object]:
    failures = []
    for name, metrics in dict(grouped.get("groups") or {}).items():
        checks = {
            "added_fnr": float(metrics["added_fnr"]) <= max_added_fnr + 1e-12,
            "positive_miou_loss": float(metrics["positive_miou_delta"])
            >= -max_positive_miou_loss - 1e-12,
            "far_decreased": float(metrics["delta_far"]) < 0,
            "not_blanket_rejection": not bool(metrics["blanket_rejection"]),
            "roh_boh_gap": (
                metrics["roh_minus_boh_gap_delta"] is not None
                and float(metrics["roh_minus_boh_gap_delta"]) <= max_gap_delta + 1e-12
            ),
        }
        if not all(checks.values()):
            failures.append({"group": name, "checks": checks})
    return {
        "passed": not failures,
        "groups_evaluated": len(dict(grouped.get("groups") or {})),
        "failures": failures,
        "thresholds": {
            "max_added_fnr": max_added_fnr,
            "max_positive_miou_loss": max_positive_miou_loss,
            "max_roh_minus_boh_gap_delta": max_gap_delta,
        },
    }


def evaluate_router(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    triggered = [row for row in rows if str(row["action"]) == CCVAction.RELOCALIZE.value]
    applied = [row for row in triggered if row.get("corrected_bbox") is not None]
    correct = sum(bool(row.get("relocalize_correct")) for row in applied)
    target = sum(bool(row.get("should_relocalize")) for row in rows)
    zero = [row for row in applied if float(row.get("original_iou") or 0.0) <= 1e-12]
    repaired_zero = sum(float(row.get("corrected_iou") or 0.0) > 1e-12 for row in zero)
    nonzero = [row for row in applied if float(row.get("original_iou") or 0.0) > 1e-12]
    regressions = sum(float(row.get("corrected_iou") or 0.0) <= 1e-12 for row in nonzero)
    corrected_miou = (
        sum(float(row.get("corrected_iou") or 0.0) for row in applied) / len(applied)
        if applied else None
    )
    captions = [row for row in rows if row.get("upstream_caption") is not None]
    caption_kept = sum(
        row.get("upstream_caption") == row.get("post_caption", row.get("upstream_caption"))
        for row in captions
    )
    return {
        "records": len(rows),
        "triggered": len(triggered),
        "applied": len(applied),
        "trigger_rate": _rate(len(triggered), len(rows)),
        "relocalize_precision": _rate(correct, len(applied)),
        "relocalize_coverage": _rate(correct, target),
        "iou_zero_repair_rate": _rate(repaired_zero, len(zero)),
        "nonzero_to_zero_regressions": regressions,
        "nonzero_to_zero_regression_rate": _rate(regressions, len(nonzero)),
        "corrected_box_miou": corrected_miou,
        "caption_preservation_rate": _rate(caption_kept, len(captions)),
    }


def latency_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    values = sorted(
        float(row["detector_latency_ms"])
        for row in rows
        if row.get("detector_latency_ms") is not None
    )
    if not values:
        return {"records": 0, "p50_ms": None, "p95_ms": None}
    at = lambda fraction: values[min(len(values) - 1, int((len(values) - 1) * fraction))]
    return {"records": len(values), "p50_ms": at(0.50), "p95_ms": at(0.95)}


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _precision_coverage(
    rows: Sequence[Mapping[str, object]], prediction: str, truth: str
) -> dict[str, float | int | None]:
    observable = [row for row in rows if row.get(truth) is not None]
    predicted = [row for row in observable if bool(row.get(prediction))]
    correct = sum(bool(row.get(truth)) for row in predicted)
    positives = sum(bool(row.get(truth)) for row in observable)
    return {
        "observable": len(observable),
        "predicted": len(predicted),
        "precision": _rate(correct, len(predicted)),
        "coverage": _rate(correct, positives),
    }


def _calibration(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    pairs = [
        (float(row["calibrated_probability"]), float(bool(row["atom_contradicted"])))
        for row in rows
        if row.get("calibrated_probability") is not None
        and row.get("atom_contradicted") is not None
    ]
    if not pairs:
        return {"records": 0, "ece": None, "brier": None}
    brier = sum((score - label) ** 2 for score, label in pairs) / len(pairs)
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        bucket = [
            pair for pair in pairs
            if lower <= pair[0] < upper or (index == 9 and pair[0] == 1.0)
        ]
        if bucket:
            confidence = sum(score for score, _ in bucket) / len(bucket)
            accuracy = sum(label for _, label in bucket) / len(bucket)
            ece += len(bucket) / len(pairs) * abs(confidence - accuracy)
    return {"records": len(pairs), "ece": ece, "brier": brier}


def _selective_curve(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float | int]]:
    pairs = sorted(
        (
            float(row["binding_risk"]),
            bool(row["atom_contradicted"]),
        )
        for row in rows
        if row.get("binding_risk") is not None and row.get("atom_contradicted") is not None
    )
    curve = []
    for coverage in (0.25, 0.50, 0.75, 1.0):
        count = max(1, round(len(pairs) * coverage)) if pairs else 0
        retained = pairs[:count]
        errors = sum((risk >= 0.5) != truth for risk, truth in retained)
        curve.append({
            "coverage": coverage,
            "records": len(retained),
            "selective_risk": errors / len(retained) if retained else 0.0,
        })
    return curve


def _bootstrap_gap_ci(
    rows: Sequence[Mapping[str, object]], *, samples: int = 1000, seed: int = 20260804
) -> dict[str, object]:
    negatives = [row for row in rows if not bool(row.get("label_exists"))]
    strata = {
        name: [row for row in negatives if row.get("hallucination_type") == name]
        for name in NEGATIVE_STRATA
    }
    if any(not value for value in strata.values()):
        return {"samples": 0, "roh_delta_far_95ci": None, "boh_delta_far_95ci": None, "gap_delta_95ci": None}
    rng = random.Random(seed)
    roh_values, boh_values, gap_values = [], [], []
    for _ in range(samples):
        deltas = {}
        for name, values in strata.items():
            draw = [values[rng.randrange(len(values))] for _ in values]
            before = sum(bool(row.get("original_accept", True)) for row in draw)
            after = sum(str(row.get("action")) != CCVAction.REJECT.value for row in draw)
            deltas[name] = (after - before) / len(draw)
        roh = (deltas["attribute"] + deltas["relation"]) / 2
        boh = (deltas["object"] + deltas["co_occurrence"]) / 2
        roh_values.append(roh)
        boh_values.append(boh)
        gap_values.append(roh - boh)

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]

    return {
        "samples": samples,
        "roh_delta_far_95ci": interval(roh_values),
        "boh_delta_far_95ci": interval(boh_values),
        "gap_delta_95ci": interval(gap_values),
    }


def evaluate_binding_metrics(
    rows: Sequence[Mapping[str, object]], *, bootstrap_samples: int = 1000
) -> dict[str, object]:
    """CABLE-specific witness, calibration, and selective-risk diagnostics."""

    contradiction = _precision_coverage(rows, "explicit_witness", "atom_contradicted")
    edge = _precision_coverage(rows, "edge_witness", "edge_witness_correct")
    swap_pairs = [
        (float(row["swap_margin"]), bool(row["atom_contradicted"]))
        for row in rows
        if row.get("swap_margin") is not None and row.get("atom_contradicted") is not None
    ]
    inverse_rows = [row for row in rows if row.get("inverse_consistent") is not None]
    inverse_accuracy = _rate(
        sum(bool(row.get("inverse_consistent")) for row in inverse_rows), len(inverse_rows)
    )
    symmetry_rows = [row for row in rows if row.get("symmetry_consistent") is not None]
    symmetry_accuracy = _rate(
        sum(bool(row.get("symmetry_consistent")) for row in symmetry_rows),
        len(symmetry_rows),
    )
    selective = {
        atom: _selective_curve([row for row in rows if row.get("atom_type") == atom])
        for atom in ("attribute", "relation")
    }
    return {
        "atom_contradiction": contradiction,
        "edge_witness": edge,
        "swap_margin_auroc": _auroc(
            [score for score, _ in swap_pairs], [label for _, label in swap_pairs]
        ),
        "inverse_consistency_accuracy": inverse_accuracy,
        "symmetry_consistency_accuracy": symmetry_accuracy,
        "selective_risk": selective,
        "calibration": _calibration(rows),
        "bootstrap": _bootstrap_gap_ci(rows, samples=bootstrap_samples),
    }


def detector_forward_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    values = sorted(int(row.get("detector_image_encoder_forwards") or 1) for row in rows)
    if not values:
        return {"records": 0, "average": None, "p95": None, "crop_trigger_rate": None}
    p95 = values[min(len(values) - 1, int((len(values) - 1) * 0.95))]
    return {
        "records": len(values),
        "average": sum(values) / len(values),
        "p95": p95,
        "crop_trigger_rate": sum(value > 1 for value in values) / len(values),
    }
