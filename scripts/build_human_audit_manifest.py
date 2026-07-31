#!/usr/bin/env python3
"""Build the pending human-review manifest from frozen IoU transition rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def parse_box(value: str):
    return json.loads(value) if value else None


def relevant(row: dict[str, str]) -> bool:
    return (
        row["baseline_state"] == "valid_zero"
        or row["transition"] == "nonzero_regressed_to_zero"
    )


def build(rows: list[dict[str, str]]) -> list[dict]:
    relevant_groups = {
        row["base_sample_id"] for row in rows if relevant(row)
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["base_sample_id"] in relevant_groups:
            grouped[row["base_sample_id"]].append(row)

    manifest = []
    for group_id in sorted(grouped):
        group_rows = sorted(grouped[group_id], key=lambda row: row["task"])
        first = group_rows[0]
        cases = []
        for row in group_rows:
            cases.append(
                {
                    "task": row["task"],
                    "transition": row["transition"],
                    "baseline_box": parse_box(row["baseline_box"]),
                    "challenger_box": parse_box(row["result_box"]),
                    "gt_box": parse_box(row["gt_box"]),
                    "baseline_iou": float(row["baseline_iou"]),
                    "challenger_iou": float(row["result_iou"]),
                    "automatic_baseline_class": row["baseline_zero_box_class"] or None,
                    "automatic_challenger_class": row["result_zero_box_class"] or None,
                }
            )
        manifest.append(
            {
                "base_sample_id": group_id,
                "image_filename": first["image_filename"],
                "query": first["query"],
                "expression_structure": first["expression_structure"],
                "target_category": first["target_category"],
                "same_category_distractors": int(first["same_category_distractors"]),
                "cases": cases,
                "review": {
                    "status": "pending",
                    "failure_mode": None,
                    "preferred_action": None,
                    "binding_evidence": [],
                    "ambiguity": None,
                    "notes": "",
                },
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    with args.samples.open(newline="", encoding="utf-8") as handle:
        manifest = build(list(csv.DictReader(handle)))
    if args.expected_count is not None and len(manifest) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} groups, generated {len(manifest)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(manifest)} audit groups to {args.output}")


if __name__ == "__main__":
    main()
