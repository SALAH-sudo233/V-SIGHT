#!/usr/bin/env python3
"""Validate P1 candidate shard coverage, parsing, latency, and provenance."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import bbox_iou, sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e1/p1/e1_p1_queries.summary.json",
    )
    parser.add_argument(
        "--candidate-pattern",
        default=str(ROOT / "data/e1/p1/candidates/e1_p1_candidates.shard-*.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/e1/p1/candidates/e1_p1_candidates.summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/e1/p1/candidates/E1_P1_CANDIDATE_REPORT.md",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print coverage without rewriting frozen manifests.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_queue(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    args = parse_args()
    queue_summary = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    expected: dict[str, dict] = {}
    for split, item in queue_summary["outputs"].items():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"queue hash mismatch: {path}")
        for row in read_queue(path):
            query_id = str(row["query_id"])
            if query_id in expected:
                raise ValueError(f"duplicate queue query: {query_id}")
            expected[query_id] = row

    paths = [Path(value) for value in sorted(glob.glob(args.candidate_pattern))]
    if not paths:
        raise FileNotFoundError(f"no candidate shards match {args.candidate_pattern}")
    latest: dict[str, dict] = {}
    attempts = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row["query_id"])
                if query_id not in expected:
                    raise ValueError(f"{path}:{line_number}: query not in frozen queue")
                if row["data_split"] != expected[query_id]["data_split"]:
                    raise ValueError(f"{path}:{line_number}: split mismatch")
                forbidden = {
                    "ann_id",
                    "gt_bbox_xyxy",
                    "gt_bbox_xywh",
                    "hallucination_type",
                    "audit_decision",
                } & set(row)
                if forbidden:
                    raise ValueError(f"candidate output leaks fields: {sorted(forbidden)}")
                attempts[query_id] += 1
                latest[query_id] = row

    missing = sorted(set(expected) - set(latest))
    extra_attempts = sum(value - 1 for value in attempts.values())
    generator_specs = {
        str(row.get("generator_spec_sha256"))
        for row in latest.values()
        if not row.get("error")
    }
    if len(generator_specs) > 1:
        raise ValueError(f"multiple generator specifications: {sorted(generator_specs)}")

    by_split = {}
    for split in sorted(queue_summary["outputs"]):
        expected_ids = {
            query_id
            for query_id, row in expected.items()
            if row["data_split"] == split
        }
        rows = [latest[query_id] for query_id in expected_ids if query_id in latest]
        successful = [row for row in rows if not row.get("error")]
        baseline_valid = [
            row for row in successful if (row.get("baseline") or {}).get("parse_valid")
        ]
        challenger_valid = [
            row for row in successful if (row.get("challenger") or {}).get("parse_valid")
        ]
        both_boxes = [
            row
            for row in successful
            if (row.get("baseline") or {}).get("pred_bbox_xyxy") is not None
            and (row.get("challenger") or {}).get("selected_bbox_xyxy") is not None
        ]
        materially_distinct = sum(
            bbox_iou(
                row["baseline"]["pred_bbox_xyxy"],
                row["challenger"]["selected_bbox_xyxy"],
            )
            < 0.95
            for row in both_boxes
        )
        latencies = [float(row["latency_sec"]) for row in successful]
        by_split[split] = {
            "expected": len(expected_ids),
            "attempted": len(rows),
            "missing": len(expected_ids) - len(rows),
            "errors": len(rows) - len(successful),
            "baseline_parse_valid": len(baseline_valid),
            "baseline_found": sum(
                bool((row.get("baseline") or {}).get("pred_found"))
                for row in successful
            ),
            "challenger_parse_valid": len(challenger_valid),
            "both_boxes": len(both_boxes),
            "materially_distinct_two_boxes_iou_lt_0_95": materially_distinct,
            "latency_p50_sec": percentile(latencies, 0.5),
            "latency_p95_sec": percentile(latencies, 0.95),
        }

    complete = not missing and all(
        item["errors"] == 0 for item in by_split.values()
    )
    if args.require_complete and not complete:
        raise ValueError(
            f"candidate generation incomplete: missing={len(missing)} "
            f"errors={sum(item['errors'] for item in by_split.values())}"
        )
    summary = {
        "schema_version": "vsight_e1_p1_candidate_manifest_v1",
        "status": "complete" if complete else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queue_manifest": {
            "path": manifest_path(args.queue_summary),
            "sha256": sha256(args.queue_summary),
        },
        "generator_spec_sha256": next(iter(generator_specs), None),
        "candidate_shards": [
            {
                "path": manifest_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "coverage": {
            "expected": len(expected),
            "attempted": len(latest),
            "missing": len(missing),
            "missing_examples": missing[:20],
            "extra_attempts": extra_attempts,
        },
        "splits": by_split,
    }
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "coverage": summary["coverage"],
                    "splits": summary["splits"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        "\n".join(
            [
                "# E1 P1 Candidate Generation",
                "",
                f"**Status:** {summary['status']}",
                "",
                f"- Expected queries: {len(expected):,}",
                f"- Attempted queries: {len(latest):,}",
                f"- Missing queries: {len(missing):,}",
                f"- Inference errors: {sum(item['errors'] for item in by_split.values()):,}",
                f"- Generator specs: {len(generator_specs)}",
                "",
                *[
                    f"- {split}: baseline parse {item['baseline_parse_valid']:,}/{item['attempted']:,}; "
                    f"challenger parse {item['challenger_parse_valid']:,}/{item['attempted']:,}; "
                    f"distinct pairs {item['materially_distinct_two_boxes_iou_lt_0_95']:,}"
                    for split, item in by_split.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
