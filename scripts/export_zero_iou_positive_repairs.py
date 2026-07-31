#!/usr/bin/env python3
"""Export audited keep/rewrite repairs as a human-reviewable candidate set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "data/audits/zero_iou_positive_repairs.qwen3.7-max-2026-05-17.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/audits/zero_iou_positive_repairs.accepted.jsonl"
DEFAULT_REVIEW_CSV = ROOT / "data/audits/zero_iou_positive_repairs.review.csv"
DEFAULT_SUMMARY = ROOT / "data/audits/zero_iou_positive_repairs.summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_successes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "ok":
            latest[str(row["base_sample_id"])] = row
    return latest


def compact_record(row: dict[str, Any], source_hash: str) -> dict[str, Any]:
    repair = row["repair"]
    request = row.get("request") or {}
    return {
        "schema_version": "vsight_zero_iou_positive_candidate_v1",
        "review_status": "pending_human_confirmation",
        "eligible_for_training": False,
        "base_sample_id": row["base_sample_id"],
        "image_filename": row["image_filename"],
        "source_expression": row["source_expression"],
        "repaired_expression": repair["repaired_expression"],
        "repair_decision": repair["decision"],
        "source_expression_truth": repair["source_expression_truth"],
        "head_object": repair["head_object"],
        "added_atoms": repair["added_atoms"],
        "removed_or_replaced_atoms": repair["removed_or_replaced_atoms"],
        "evidence_citations": repair["evidence_citations"],
        "confidence": repair["confidence"],
        "reason": repair["reason"],
        "target_category_hint": request.get("target_category_hint"),
        "expression_structure": request.get("expression_structure"),
        "same_category_distractors_hint": request.get("same_category_distractors_hint"),
        "human_review_present": request.get("human_review") is not None,
        "source_repair_jsonl_sha256": source_hash,
        "repair_prompt_sha256": row.get("repair_prompt_sha256"),
    }


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "base_sample_id",
        "image_filename",
        "source_expression",
        "repaired_expression",
        "repair_decision",
        "source_expression_truth",
        "confidence",
        "added_atoms",
        "removed_or_replaced_atoms",
        "evidence_citations",
        "reason",
        "review_status",
        "eligible_for_training",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                field: json.dumps(record[field], ensure_ascii=False)
                if isinstance(record[field], (list, dict))
                else record[field]
                for field in fields
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_hash = sha256(args.input)
    latest = latest_successes(read_jsonl(args.input))
    records = [
        compact_record(row, source_hash)
        for row in latest.values()
        if row.get("repair", {}).get("decision") in {"keep", "rewrite"}
    ]
    records.sort(key=lambda row: row["base_sample_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_review_csv(args.review_csv, records)
    summary = {
        "input": str(args.input),
        "input_sha256": source_hash,
        "raw_unique_successes": len(latest),
        "accepted_candidate_count": len(records),
        "rewrite_count": sum(row["repair_decision"] == "rewrite" for row in records),
        "keep_count": sum(row["repair_decision"] == "keep" for row in records),
        "raw_decision_counts": {
            decision: sum(
                row.get("repair", {}).get("decision") == decision for row in latest.values()
            )
            for decision in ("keep", "rewrite", "reject", "needs_human")
        },
        "raw_truth_counts": {
            truth: sum(
                row.get("repair", {}).get("source_expression_truth") == truth
                for row in latest.values()
            )
            for truth in ("supported", "ambiguous", "contradicted", "uncertain")
        },
        "review_status": "pending_human_confirmation",
        "eligible_for_training": False,
        "output": str(args.output),
        "review_csv": str(args.review_csv),
        "summary": str(args.summary),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
