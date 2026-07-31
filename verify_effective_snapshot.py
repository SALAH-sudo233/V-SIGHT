#!/usr/bin/env python3
"""Verify the frozen, state-preserving candidate replacement result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data" / "qwen2.5-vl-7b.canonical_records.jsonl"
RESULT = ROOT / "data" / "roh_vcd_state_preserving_records.jsonl"

EXPECTED = {
    "baseline_sha256": "c3a5a7b82bc104896329d2c313ed091133bfcd3351736f4be3d2b1018142f63e",
    "result_sha256": "183661867e70278d3ede732d9bda4fbdc76270afd5234fb2c23fba7c5432cead",
    "t2_baseline_zero": 0.242,
    "t2_result_zero": 0.220,
    "t4_baseline_zero": 0.238,
    "t4_result_zero": 0.216,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def key(row: dict) -> tuple:
    return (
        row.get("task"),
        row.get("query_role"),
        row.get("base_sample_id"),
        row.get("pair_id"),
    )


def positive_iou_stats(rows: list[dict], task: str) -> tuple[float, float, int]:
    values = [
        float(row["iou"]) if row.get("iou") is not None else 0.0
        for row in rows
        if row.get("task") == task and row.get("query_role") == "positive"
    ]
    if len(values) != 500:
        raise AssertionError(f"{task}: expected 500 positives, found {len(values)}")
    return sum(values) / len(values), sum(value == 0.0 for value in values) / len(values), len(values)


def main() -> None:
    baseline = read_jsonl(BASELINE)
    result = read_jsonl(RESULT)
    if len(baseline) != 7500 or len(result) != 7500:
        raise AssertionError("both record files must contain 7,500 rows")

    baseline_by_key = {key(row): row for row in baseline}
    result_by_key = {key(row): row for row in result}
    if baseline_by_key.keys() != result_by_key.keys():
        raise AssertionError("baseline/result record keys differ")
    if any(
        bool(baseline_by_key[row_key].get("pred_found")) != bool(result_by_key[row_key].get("pred_found"))
        for row_key in baseline_by_key
    ):
        raise AssertionError("state-preserving policy changed a pred_found decision")

    print("SHA-256")
    print(f"  baseline: {sha256(BASELINE)}")
    print(f"  result:   {sha256(RESULT)}")
    assert sha256(BASELINE) == EXPECTED["baseline_sha256"]
    assert sha256(RESULT) == EXPECTED["result_sha256"]

    for task, baseline_expected, result_expected in (
        ("t2_vqa_grounding", EXPECTED["t2_baseline_zero"], EXPECTED["t2_result_zero"]),
        ("t4_caption_grounding", EXPECTED["t4_baseline_zero"], EXPECTED["t4_result_zero"]),
    ):
        baseline_miou, baseline_zero, _ = positive_iou_stats(baseline, task)
        result_miou, result_zero, _ = positive_iou_stats(result, task)
        print(
            f"{task}: mIoU {baseline_miou:.4f} -> {result_miou:.4f}; "
            f"IoU=0 {baseline_zero:.1%} -> {result_zero:.1%}"
        )
        assert abs(baseline_zero - baseline_expected) < 1e-12
        assert abs(result_zero - result_expected) < 1e-12

    replacements = sum(
        row.get("roh_vcd_candidate_source") == "binding_aware" for row in result
    )
    print(f"State-preserving binding-aware replacements: {replacements}")
    print("Verification passed.")


if __name__ == "__main__":
    main()
