#!/usr/bin/env python3
"""Summarize a TRACE-Bind pilot without fitting thresholds or actions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        help="optional post-inference labels JSONL; no labels are used for features",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    wins = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return wins / (positive_count * negative_count)


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(labels)
    if positive_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positive_count


def _label_value(row: dict) -> bool | None:
    value = row.get("binding_label")
    if value is None:
        value = row.get("label")
    if value is None:
        value = row.get("evidence_status")
    if isinstance(value, bool):
        return value
    normalized = str(value or "").casefold().strip()
    if normalized in {"supported", "accept", "accepted", "true", "1", "positive"}:
        return True
    if normalized in {
        "contradicted", "unsupported", "reject", "rejected", "false", "0", "negative"
    }:
        return False
    return None


VARIANTS = (
    "final_confidence",
    "final_static_typed_geometry",
    "localized_attention",
    "trajectory_only",
    "trajectory_plus_swaps",
    "attention_plus_trace",
)
BASELINE_VARIANTS = ("final_confidence", "final_static_typed_geometry")


def _scores(trace: dict, attention: dict) -> dict[str, float | None]:
    trajectories = trace.get("proposal_trajectories") or []
    final_scores = []
    for trajectory in trajectories:
        states = trajectory.get("states") or []
        if not states:
            continue
        final_scores.append(float((states[-1].get("span_scores") or {}).get("target") or 0.0))
    final_confidence = max(final_scores, default=None)
    stats = trace.get("statistics") or {}
    assignment_rows = trace.get("assignment_ledger") or []
    final_static = None
    if assignment_rows and assignment_rows[-1].get("upstream") is not None:
        final_static = float(assignment_rows[-1]["upstream"].get("score") or 0.0)
    attention_score = attention.get("decoder_top_transport")
    if attention_score is None:
        attention_score = attention.get("top_transport")
    if attention_score is not None:
        attention_score = min(1.0, max(0.0, float(attention_score)))
    trajectory_only = None
    trajectory_plus_swaps = None
    if final_confidence is not None:
        trajectory_only = statistics.mean(
            (
                float(stats.get("rank_stability") or 0.0),
                float(stats.get("layer_agreement") or 0.0),
                float(stats.get("bbox_convergence") or 0.0),
                1.0 - float(stats.get("trajectory_entropy") or 0.0),
            )
        )
        trajectory_plus_swaps = statistics.mean(
            (
                trajectory_only,
                1.0 - float(stats.get("alternative_dominance") or 0.0),
                1.0 - float(stats.get("edge_uncertainty") or 0.0),
            )
        )
    attention_plus_trace = (
        statistics.mean((attention_score, trajectory_plus_swaps))
        if attention_score is not None and trajectory_plus_swaps is not None
        else None
    )
    return {
        "final_confidence": final_confidence,
        "final_static_typed_geometry": final_static,
        "localized_attention": attention_score,
        "trajectory_only": trajectory_only,
        "trajectory_plus_swaps": trajectory_plus_swaps,
        "attention_plus_trace": attention_plus_trace,
    }


def selective_curve(scores: list[float], labels: list[bool]) -> list[dict[str, float | int]]:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    curve = []
    for coverage in (0.25, 0.50, 0.75, 1.0):
        count = max(1, round(len(order) * coverage)) if order else 0
        selected = order[:count]
        correct = sum(labels[index] for index in selected)
        precision = correct / count if count else 0.0
        curve.append(
            {
                "coverage": coverage,
                "records": count,
                "precision": precision,
                "selective_risk": 1.0 - precision,
            }
        )
    return curve


def _status_metrics(rows: list[dict]) -> dict[str, object]:
    confusion = Counter()
    covered = []
    for row in rows:
        label_name = "supported" if bool(row["label"]) else "contradicted"
        status = str(row.get("trace_status") or "missing")
        confusion[(status, label_name)] += 1
        if status == "supported":
            covered.append((True, bool(row["label"])))
        elif status in {"contradicted", "unsupported"}:
            covered.append((False, bool(row["label"])))

    true_positive = sum(prediction and label for prediction, label in covered)
    false_positive = sum(prediction and not label for prediction, label in covered)
    true_negative = sum(not prediction and not label for prediction, label in covered)
    false_negative = sum(not prediction and label for prediction, label in covered)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "records": len(rows),
        "covered_records": len(covered),
        "coverage": ratio(len(covered), len(rows)),
        "covered_accuracy": ratio(true_positive + true_negative, len(covered)),
        "supported_precision": ratio(true_positive, true_positive + false_positive),
        "supported_recall": ratio(true_positive, true_positive + false_negative),
        "negative_precision": ratio(true_negative, true_negative + false_negative),
        "negative_recall": ratio(true_negative, true_negative + false_positive),
        "binary_confusion_on_covered": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "status_by_label": {
            status: {
                label: confusion[(status, label)]
                for label in ("supported", "contradicted")
            }
            for status in sorted({status for status, _ in confusion})
        },
    }


def _paired_auroc_deltas(rows: list[dict]) -> dict[str, object]:
    result = {}
    for baseline in BASELINE_VARIANTS:
        baseline_result = {}
        for variant in VARIANTS:
            if variant == baseline:
                continue
            paired = [
                row for row in rows
                if row["scores"].get(baseline) is not None
                and row["scores"].get(variant) is not None
            ]
            truth = [bool(row["label"]) for row in paired]
            baseline_auc = auroc(
                [float(row["scores"][baseline]) for row in paired], truth
            )
            variant_auc = auroc(
                [float(row["scores"][variant]) for row in paired], truth
            )
            baseline_result[variant] = {
                "records": len(paired),
                "auroc_delta": (
                    variant_auc - baseline_auc
                    if baseline_auc is not None and variant_auc is not None else None
                ),
            }
        result[baseline] = baseline_result
    return result


def _bootstrap_metrics(
    rows: list[dict], *, samples: int, seed: int
) -> dict[str, object]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("image_group_id") or row["record_id"]), []).append(row)
    if len(groups) < 2 or samples <= 0:
        return {"samples": 0, "group_count": len(groups)}
    keys = sorted(groups)
    rng = random.Random(seed)
    values = {variant: [] for variant in VARIANTS}
    deltas = {
        baseline: {
            variant: [] for variant in VARIANTS if variant != baseline
        }
        for baseline in BASELINE_VARIANTS
    }
    for _ in range(samples):
        draw = [groups[rng.choice(keys)] for _ in keys]
        flattened = [item for group in draw for item in group]
        for variant in VARIANTS:
            available = [item for item in flattened if item["scores"].get(variant) is not None]
            values[variant].append(
                auroc(
                    [float(item["scores"][variant]) for item in available],
                    [bool(item["label"]) for item in available],
                )
            )
        for baseline in BASELINE_VARIANTS:
            for variant in VARIANTS:
                if variant == baseline:
                    continue
                paired = [
                    item for item in flattened
                    if item["scores"].get(baseline) is not None
                    and item["scores"].get(variant) is not None
                ]
                truth = [bool(item["label"]) for item in paired]
                baseline_auc = auroc(
                    [float(item["scores"][baseline]) for item in paired], truth
                )
                variant_auc = auroc(
                    [float(item["scores"][variant]) for item in paired], truth
                )
                deltas[baseline][variant].append(
                    variant_auc - baseline_auc
                    if baseline_auc is not None and variant_auc is not None else None
                )

    def interval(values: list[float | None]) -> list[float] | None:
        numeric = sorted(float(value) for value in values if value is not None)
        if not numeric:
            return None
        return [numeric[int(0.025 * (len(numeric) - 1))], numeric[int(0.975 * (len(numeric) - 1))]]

    return {
        "samples": samples,
        "group_count": len(groups),
        "auroc_95ci": {
            variant: interval(variant_values)
            for variant, variant_values in values.items()
        },
        "paired_auroc_delta_95ci": {
            baseline: {
                variant: interval(variant_values)
                for variant, variant_values in baseline_values.items()
            }
            for baseline, baseline_values in deltas.items()
        },
    }


def main() -> int:
    args = arguments()
    if args.output.exists() and not args.force:
        raise FileExistsError("output exists; pass --force")
    rows = read_jsonl(args.run)
    labels = {}
    label_rows = []
    label_sha256 = None
    if args.labels is not None:
        label_rows = read_jsonl(args.labels)
        labels = {str(row["record_id"]): row for row in label_rows}
        if len(labels) != len(label_rows):
            raise ValueError("label sidecar contains duplicate record IDs")
        label_sha256 = hashlib.sha256(args.labels.read_bytes()).hexdigest()

    prepared = []
    status_counts = Counter()
    field_counts = Counter()
    trace_latencies = []
    detector_latencies = []
    policy_side_effects = 0
    for row in rows:
        detector = row.get("detector_evidence") or {}
        trace = detector.get("trace_ledger") or {}
        attention = detector.get("attention_ledger") or {}
        status_counts[str(trace.get("evidence_status") or "missing")] += 1
        if trace:
            field_counts["trace_ledger"] += 1
        if attention:
            field_counts["attention_ledger"] += 1
        stats = trace.get("statistics") or {}
        for field in (
            "rank_stability",
            "layer_agreement",
            "stabilization_layer",
            "swap_persistence",
            "alternative_dominance",
            "trajectory_entropy",
            "bbox_convergence",
            "edge_uncertainty",
        ):
            if field in stats:
                field_counts[field] += 1
        latency = (trace.get("latency_ms") or {})
        if latency.get("trace_added") is not None:
            trace_latencies.append(float(latency["trace_added"]))
        if latency.get("detector_only") is not None:
            detector_latencies.append(float(latency["detector_only"]))
        if bool(trace.get("action_policy_applied")):
            policy_side_effects += 1
        scores = _scores(trace, attention)
        label_row = labels.get(str(row.get("record_id")))
        label = _label_value(label_row) if label_row is not None else None
        if label is not None and any(value is not None for value in scores.values()):
            prepared.append(
                {
                    "record_id": str(row["record_id"]),
                    "image_group_id": row.get("image_group_id") or row.get("image_filename"),
                    "query_stratum": (
                        str(label_row.get("query_stratum") or "unknown")
                        if label_row is not None else "unknown"
                    ),
                    "label": label,
                    "trace_status": str(trace.get("evidence_status") or "missing"),
                    "scores": scores,
                }
            )

    metrics = {"records": len(prepared), "labels_joined": bool(args.labels)}
    if prepared:
        variant_metrics = {}
        for variant in VARIANTS:
            available = [row for row in prepared if row["scores"].get(variant) is not None]
            truth = [bool(row["label"]) for row in available]
            variant_scores = [float(row["scores"][variant]) for row in available]
            variant_metrics[variant] = {
                "records": len(available),
                "auroc": auroc(variant_scores, truth),
                "auprc": average_precision(variant_scores, truth),
                "precision_risk_coverage": selective_curve(variant_scores, truth),
            }
        metrics["variants"] = variant_metrics
        by_stratum = {}
        for stratum in sorted({str(row.get("query_stratum") or "unknown") for row in prepared}):
            stratum_rows = [row for row in prepared if row.get("query_stratum") == stratum]
            stratum_metrics = {}
            for variant in VARIANTS:
                available = [
                    row for row in stratum_rows if row["scores"].get(variant) is not None
                ]
                truth = [bool(row["label"]) for row in available]
                variant_scores = [float(row["scores"][variant]) for row in available]
                stratum_metrics[variant] = {
                    "records": len(available),
                    "auroc": auroc(variant_scores, truth),
                    "auprc": average_precision(variant_scores, truth),
                }
            by_stratum[stratum] = stratum_metrics
        metrics["by_query_stratum"] = by_stratum
        metrics["bootstrap_grouped"] = _bootstrap_metrics(
            prepared, samples=args.bootstrap_samples, seed=args.seed
        )
        metrics["paired_auroc_delta"] = _paired_auroc_deltas(prepared)
        metrics["trace_status"] = _status_metrics(prepared)
    else:
        metrics["variants"] = {
            variant: {
                "records": 0,
                "auroc": None,
                "auprc": None,
                "precision_risk_coverage": [],
            }
            for variant in VARIANTS
        }
        metrics["by_query_stratum"] = {}
        metrics["bootstrap_grouped"] = {"samples": 0, "group_count": 0}
        metrics["paired_auroc_delta"] = _paired_auroc_deltas([])
        metrics["trace_status"] = _status_metrics([])
    summary = {
        "schema_version": "vsight_trace_bind_pilot_diagnostics_v1",
        "records": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "field_coverage": {
            field: count / len(rows) if rows else None
            for field, count in sorted(field_counts.items())
        },
        "latency_ms": {
            "detector_only_p50": statistics.median(detector_latencies) if detector_latencies else None,
            "detector_only_p95": percentile(detector_latencies, 0.95),
            "trace_added_p50": statistics.median(trace_latencies) if trace_latencies else None,
            "trace_added_p95": percentile(trace_latencies, 0.95),
        },
        "matched_metrics": metrics,
        "labels_sha256": label_sha256,
        "label_provenance": {
            "records": len(label_rows),
            "matched_records": len(prepared),
            "semantics": sorted(
                {str(row.get("label_semantics")) for row in label_rows if row.get("label_semantics")}
            ),
            "iou_thresholds": sorted(
                {float(row["label_iou_threshold"]) for row in label_rows if row.get("label_iou_threshold") is not None}
            ),
        },
        "action_policy_applied_records": policy_side_effects,
        "diagnostic_only": True,
        "sealed_heldout_accessed": False,
        "source_run": str(args.run),
        "source_run_sha256": hashlib.sha256(args.run.read_bytes()).hexdigest(),
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary["payload_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
