#!/usr/bin/env python3
"""Join pairwise VLM scores to E2 supervision and evaluate the probe."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_e2_verifier as evaluator  # noqa: E402
from vsight.clip_verifier import read_selector_rows  # noqa: E402
from vsight.e2_verifier import selector_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e1/p1/selector/e1_p1_selector.summary.json",
    )
    parser.add_argument(
        "--output-pattern",
        default=str(
            ROOT / "data/e1/p1/vlm_probe/outputs/e2_vlm_probe.shard-*.jsonl"
        ),
    )
    parser.add_argument("--benchmark", type=Path, default=evaluator.DEFAULT_BENCHMARK)
    parser.add_argument(
        "--canonical-records", type=Path, default=evaluator.DEFAULT_RECORDS
    )
    parser.add_argument("--candidate-pattern", default=evaluator.DEFAULT_CANDIDATES)
    parser.add_argument("--image-root", type=Path, default=evaluator.DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/e2_vlm_probe/evaluation.json",
    )
    parser.add_argument("--t2-regression-budget", type=int, default=8)
    parser.add_argument("--t4-regression-budget", type=int, default=11)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--training-supervision-used", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latest = {}
    attempts = 0
    for value in sorted(glob.glob(args.output_pattern)):
        for row in evaluator.read_jsonl(Path(value)):
            attempts += 1
            latest[str(row["query_id"])] = row
    if not latest:
        raise FileNotFoundError(args.output_pattern)

    selector = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    calibration = read_selector_rows(
        evaluator.resolve_path(selector["outputs"]["calibration"]["path"])
    )
    dev = evaluator.load_dev_rows(args)
    suites = {
        "calibration": calibration,
        "t2": dev["t2_vqa_grounding"],
        "t4": dev["t4_caption_grounding"],
    }
    expected = {
        str(row["query_id"])
        for rows in suites.values()
        for row in rows
        if row.get("selector_eligible")
    }
    successful = {
        query_id: row
        for query_id, row in latest.items()
        if not row.get("error") and row.get("candidate_1_minus_0_score") is not None
    }
    missing = sorted(expected - set(successful))
    if args.require_complete and missing:
        raise ValueError(f"VLM probe is incomplete: {len(missing)} missing")
    scores = {
        name: {
            str(row["query_id"]): float(successful[str(row["query_id"])]["candidate_1_minus_0_score"])
            for row in rows
            if str(row["query_id"]) in successful
        }
        for name, rows in suites.items()
    }
    raw = {
        name: selector_metrics(rows, scores[name], 0.0)
        for name, rows in suites.items()
    }
    calibration_budget = (
        raw["calibration"]["unconditional_nonzero_to_zero_regressions"] // 2
    )
    threshold, tuned = evaluator.choose_joint_dev_threshold(
        calibration,
        scores["calibration"],
        dev,
        {
            "t2_vqa_grounding": scores["t2"],
            "t4_caption_grounding": scores["t4"],
        },
        calibration_budget,
        args.t2_regression_budget,
        args.t4_regression_budget,
    )
    latencies = sorted(
        float(row["latency_sec"]) for row in successful.values()
    )

    def percentile(probability: float) -> float:
        if len(latencies) == 1:
            return latencies[0]
        position = (len(latencies) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        fraction = position - lower
        return latencies[lower] * (1 - fraction) + latencies[upper] * fraction

    result = {
        "schema_version": "vsight_e2_vlm_probe_evaluation_v1",
        "status": "complete" if not missing else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "expected": len(expected),
            "attempted_unique": len(latest),
            "successful": len(successful),
            "missing": len(missing),
            "missing_examples": missing[:20],
            "attempts": attempts,
            "parse_valid": sum(bool(row.get("parse_valid")) for row in successful.values()),
        },
        "efficiency": {
            "latency_mean_sec": statistics.fmean(latencies),
            "latency_p50_sec": percentile(0.5),
            "latency_p95_sec": percentile(0.95),
            "additional_vlm_calls_per_eligible_query": 1,
        },
        "raw_zero_threshold_results": raw,
        "joint_dev_threshold": threshold,
        "joint_dev_results": tuned,
        "threshold_budgets": {
            "calibration": calibration_budget,
            "t2": args.t2_regression_budget,
            "t4": args.t4_regression_budget,
        },
        "training_supervision_used": args.training_supervision_used,
        "sealed_heldout_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = args.output.with_suffix(".md")
    lines = [
        "# E2 pairwise VLM verifier probe",
        "",
        f"- Coverage: {len(successful)}/{len(expected)}",
        f"- Added latency p50/p95: {percentile(0.5):.3f}/{percentile(0.95):.3f} sec",
        f"- Joint development threshold: {threshold:.6f}",
    ]
    for name, value in tuned.items():
        lines.append(
            f"- {name}: fixed {value['strongest_fixed_miou']:.6f}; selector "
            f"{value['selector_miou']:.6f}; capture from fixed "
            f"{value['oracle_gap_capture_from_strongest_fixed']:.3%}; "
            f"nonzero-to-zero {value['nonzero_to_zero_regressions']}"
        )
    lines.extend(("", "Sealed repaired-1996 was not accessed.", ""))
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
