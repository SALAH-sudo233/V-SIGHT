#!/usr/bin/env python3
"""Merge CABLE crop-pilot shards and report quality/cost diagnostics."""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv import CCVAction  # noqa: E402
from vsight.ccv_metrics import (  # noqa: E402
    audit_safety_gates,
    evaluate_grouped_safety,
    evaluate_router,
    evaluate_safety,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=660)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def percentile(values: list[float | int], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def read_rows(pattern: str) -> tuple[list[dict], list[dict]]:
    paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no crop pilot shards match {pattern}")
    rows = []
    shard_summaries = []
    seen = set()
    for path in paths:
        summary_path = path.with_suffix(path.suffix + ".summary.json")
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary.get("output_sha256"):
            raise ValueError(f"crop pilot shard hash mismatch: {path}")
        shard_summaries.append(summary)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["model"]), str(row["task"]), str(row["sample_id"]))
                if key in seen:
                    raise ValueError(f"duplicate crop pilot row: {key}")
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (row["model"], row["task"], row["sample_id"]))
    return rows, shard_summaries


def flatten(rows: list[dict], policy: str) -> list[dict]:
    output = []
    for row in rows:
        decision = row["policies"][policy]
        output.append(
            {
                "model": row["model"],
                "task": row["task"],
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "label_exists": row["label_exists"],
                "hallucination_type": row["hallucination_type"],
                "query_atom_type": row["query_atom_type"],
                "original_accept": row["original_accept"],
                "original_iou": row["original_iou"],
                "crop_triggered": row["crop_triggered"],
                "crop_passes": row["crop_passes"],
                "action": decision["action"],
                "selected_iou": decision["selected_iou"],
                "corrected_bbox": decision["corrected_bbox"],
                "corrected_iou": decision["corrected_iou"],
                "relocalize_correct": decision["relocalize_correct"],
                "should_relocalize": decision["should_relocalize"],
                "detector_image_encoder_forwards": row["cost"][f"{policy}_forwards"],
                "detector_latency_ms": row["cost"][f"{policy}_detector_ms"],
            }
        )
    return output


def contradiction_metrics(rows: list[dict]) -> dict[str, dict[str, float | int | None]]:
    output = {}
    for name in ("all", "attribute", "relation", "object"):
        subset = [
            row
            for row in rows
            if row["crop_triggered"]
            and (name == "all" or row["query_atom_type"] == name)
        ]
        predicted = [row for row in subset if row["action"] == CCVAction.REJECT.value]
        correct = sum(not bool(row["label_exists"]) for row in predicted)
        negatives = sum(not bool(row["label_exists"]) for row in subset)
        false_positive = sum(bool(row["label_exists"]) for row in predicted)
        output[name] = {
            "triggered_records": len(subset),
            "predicted_contradictions": len(predicted),
            "correct_negative_contradictions": correct,
            "false_positive_contradictions": false_positive,
            "precision": correct / len(predicted) if predicted else None,
            "precision_95ci_wilson": wilson_interval(correct, len(predicted)),
            "negative_coverage": correct / negatives if negatives else None,
            "negative_coverage_95ci_wilson": wilson_interval(correct, negatives),
        }
    return output


def cost_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    forwards = [int(row["detector_image_encoder_forwards"]) for row in rows]
    latency = [float(row["detector_latency_ms"]) for row in rows]
    return {
        "records": len(rows),
        "crop_trigger_rate": sum(bool(row["crop_triggered"]) for row in rows) / len(rows),
        "average_detector_forwards": statistics.mean(forwards),
        "p95_detector_forwards": percentile(forwards, 0.95),
        "detector_latency_ms_p50": percentile(latency, 0.50),
        "detector_latency_ms_p95": percentile(latency, 0.95),
    }


def positive_regression_metrics(rows: list[dict]) -> dict[str, float | int | bool | None]:
    eligible = [
        row
        for row in rows
        if row["label_exists"]
        and row["original_accept"]
        and float(row["original_iou"]) > 0
    ]
    regressions = [row for row in eligible if float(row["selected_iou"]) <= 0]
    rate = len(regressions) / len(eligible) if eligible else None
    return {
        "positive_original_nonzero": len(eligible),
        "nonzero_to_zero_regressions": len(regressions),
        "nonzero_to_zero_regression_rate": rate,
        "nonzero_to_zero_rate_95ci_wilson": wilson_interval(len(regressions), len(eligible)),
        "passes_one_percent_budget": rate is not None and rate <= 0.01,
    }


def policy_summary(rows: list[dict]) -> dict[str, object]:
    grouped = evaluate_grouped_safety(rows)
    return {
        "overall": evaluate_safety(rows),
        "heldout_model_task": grouped,
        "pilot_safety_gate": audit_safety_gates(grouped),
        "contradiction": contradiction_metrics(rows),
        "router": evaluate_router(rows),
        "positive_regression": positive_regression_metrics(rows),
        "cost": cost_metrics(rows),
    }


def main() -> int:
    args = arguments()
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "predictions.jsonl.gz"
    if not args.force and (summary_path.exists() or predictions_path.exists()):
        raise FileExistsError("crop pilot merged output exists; pass --force")
    rows, shard_summaries = read_rows(args.input_glob)
    if len(rows) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} pilot records, found {len(rows)}")
    target = flatten(rows, "target_only")
    sequential = flatten(rows, "target_then_union")
    target_by_key = {
        (row["model"], row["task"], row["sample_id"]): row for row in target
    }
    incremental = Counter()
    for row in sequential:
        key = (row["model"], row["task"], row["sample_id"])
        prior = target_by_key[key]
        if row["action"] != prior["action"]:
            incremental[
                f"{prior['action']}->{row['action']}|"
                f"{'positive' if row['label_exists'] else row['hallucination_type']}"
            ] += 1
    sample_counts = Counter(
        (str(row["model"]), str(row["task"]), str(row["hallucination_type"]))
        for row in rows
    )
    crop_passes = Counter(int(row["crop_passes"]) for row in rows if row["crop_triggered"])
    result_reasons = Counter(
        str(row["crop_result_reason"])
        for row in rows
        if row.get("crop_result_reason") is not None
    )
    summary = {
        "schema_version": "vsight_cable_crop_pilot_summary_v1",
        "records": len(rows),
        "unique_rows": len(
            {(row["model"], row["task"], row["sample_id"]) for row in rows}
        ),
        "sample_bucket_count": len(sample_counts),
        "sample_bucket_sizes": dict(Counter(sample_counts.values())),
        "sample_strata": dict(Counter(str(row["hallucination_type"]) for row in rows)),
        "query_atom_types": dict(Counter(str(row["query_atom_type"]) for row in rows)),
        "initial_result_mismatches": sum(
            int(value.get("initial_result_mismatches") or 0) for value in shard_summaries
        ),
        "triggered_records": sum(bool(row["crop_triggered"]) for row in rows),
        "crop_pass_counts": {str(key): value for key, value in sorted(crop_passes.items())},
        "resolved_records": sum(bool(row["crop_resolved"]) for row in rows),
        "result_reason_counts": dict(result_reasons.most_common()),
        "policies": {
            "target_only": policy_summary(target),
            "target_then_union": policy_summary(sequential),
        },
        "union_incremental_action_changes": dict(incremental),
        "gpu_peak_memory_mb": max(
            (
                float(value["gpu_peak_memory_mb"])
                for value in shard_summaries
                if value.get("gpu_peak_memory_mb") is not None
            ),
            default=None,
        ),
        "maximum_additional_detector_forwards": 2,
        "additional_upstream_mllm_calls": 0,
        "dataset_role": "development_pilot_not_final_test",
        "sealed_heldout_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with deterministic_gzip(predictions_path) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(rows),
                "triggered_records": summary["triggered_records"],
                "target_only": summary["policies"]["target_only"]["overall"],
                "target_then_union": summary["policies"]["target_then_union"]["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
