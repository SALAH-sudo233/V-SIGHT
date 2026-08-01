#!/usr/bin/env python3
"""Validate E2b T4 coverage and freeze shard hashes."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e2b/queue/e2b_refcocog.summary.json",
    )
    parser.add_argument(
        "--pattern", default=str(ROOT / "data/e2b/t4_candidates/e2b_t4.shard-*.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/e2b/t4_candidates/e2b_t4.summary.json",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    queue = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    expected = {}
    for item in queue["outputs"].values():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"queue hash mismatch: {path}")
        expected.update((str(row["query_id"]), row) for row in read_gzip(path))
    paths = [Path(value) for value in sorted(glob.glob(args.pattern))]
    latest = {}
    attempts = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row["query_id"])
                if query_id not in expected:
                    raise ValueError(f"T4 output not in queue: {query_id}")
                attempts[query_id] += 1
                latest[query_id] = row
    missing = sorted(set(expected) - set(latest))
    errors = sum(bool(row.get("error")) for row in latest.values())
    valid = sum(
        not row.get("error") and bool((row.get("t4") or {}).get("parse_valid"))
        for row in latest.values()
    )
    latencies = [
        float(row["latency_sec"]) for row in latest.values() if not row.get("error")
    ]
    complete = not missing and errors == 0
    if args.require_complete and not complete:
        raise ValueError(f"E2b T4 incomplete: missing={len(missing)} errors={errors}")
    summary = {
        "schema_version": "vsight_e2b_t4_manifest_v1",
        "status": "complete" if complete else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(args.queue_summary), "sha256": sha256(args.queue_summary)},
        "coverage": {
            "expected": len(expected),
            "attempted": len(latest),
            "missing": len(missing),
            "errors": errors,
            "parse_valid": valid,
            "parse_invalid": len(latest) - errors - valid,
            "extra_attempts": sum(value - 1 for value in attempts.values()),
        },
        "latency": {
            "mean_sec": statistics.fmean(latencies) if latencies else None,
            "median_sec": statistics.median(latencies) if latencies else None,
        },
        "shards": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in paths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
