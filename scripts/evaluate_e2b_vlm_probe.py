#!/usr/bin/env python3
"""Evaluate local E2b Qwen pair scores on the calibration relation pairs."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e2b_relation_verifier import choose_joint_threshold  # noqa: E402
from vsight.e1_data import sha256  # noqa: E402

DEFAULT_SELECTOR = ROOT / "data/e2b/selector/e2b_selector.summary.json"
DEFAULT_PATTERN = str(ROOT / "data/e2b/vlm_probe/outputs/e2_vlm_probe.shard-*.jsonl")
DEFAULT_OUTPUT = ROOT / "outputs/e2b_vlm_lora/evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-summary", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--missing-policy",
        choices=("challenger", "baseline"),
        default="challenger",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    selector = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    item = selector["outputs"]["calibration"]
    calibration_path = resolve_path(item["path"])
    if sha256(calibration_path) != item["sha256"]:
        raise ValueError("E2b calibration hash mismatch")
    rows = read_gzip(calibration_path)
    expected = {
        str(row["query_id"])
        for row in rows
        if row.get("relation_selector_eligible")
    }
    latest = {}
    for value in sorted(glob.glob(args.output_pattern)):
        with Path(value).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[str(row["query_id"])] = row
    successful = {
        query_id: row
        for query_id, row in latest.items()
        if not row.get("error") and row.get("candidate_1_minus_0_score") is not None
    }
    abstained = {
        query_id
        for query_id, row in latest.items()
        if query_id in expected and not row.get("error") and row.get("abstain")
    }
    missing = expected - set(successful) - abstained
    if args.require_complete and missing:
        raise ValueError(f"E2b VLM probe incomplete: {len(missing)} missing")
    scores = {
        query_id: float(row["candidate_1_minus_0_score"])
        for query_id, row in successful.items()
        if query_id in expected
    }
    baseline_ids = abstained | missing if args.missing_policy == "baseline" else set()
    if baseline_ids:
        floor = min(scores.values(), default=0.0) - 1.0
        scores.update({query_id: floor for query_id in baseline_ids})
    threshold, metrics, objective = choose_joint_threshold(rows, scores)
    result = {
        "schema_version": "vsight_e2b_vlm_probe_evaluation_v1",
        "status": "complete" if not missing else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "expected": len(expected),
            "scored": len(expected & set(successful)),
            "abstained": len(abstained),
            "missing": len(missing),
            "missing_examples": sorted(missing)[:20],
        },
        "threshold": threshold,
        "objective": objective,
        "metrics": metrics,
        "training_supervision_used": True,
        "missing_policy": args.missing_policy,
        "sealed_heldout_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
