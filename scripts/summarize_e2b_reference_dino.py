#!/usr/bin/env python3
"""Freeze Grounding DINO reference-proposal coverage and shard hashes."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e2b/reference_queue/e2b_reference_queue.summary.json",
    )
    parser.add_argument(
        "--pattern",
        default=str(ROOT / "data/e2b/reference_proposals/e2b_reference_dino.shard-*.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/e2b/reference_proposals/e2b_reference_dino.summary.json",
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
    expected = set()
    for item in queue["outputs"].values():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"reference queue hash mismatch: {path}")
        expected.update(str(row["query_id"]) for row in read_gzip(path))
    paths = [Path(value) for value in sorted(glob.glob(args.pattern))]
    latest = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[str(row["query_id"])] = row
    missing = sorted(expected - set(latest))
    errors = sum(bool(row.get("error")) for row in latest.values())
    if args.require_complete and (missing or errors):
        raise ValueError(f"reference proposals incomplete: missing={len(missing)} errors={errors}")
    successful = [row for row in latest.values() if not row.get("error")]
    summary = {
        "schema_version": "vsight_e2b_reference_dino_manifest_v1",
        "status": "complete" if not missing and not errors else "partial",
        "queue": {"path": str(args.queue_summary), "sha256": sha256(args.queue_summary)},
        "coverage": {
            "expected": len(expected),
            "attempted": len(latest),
            "missing": len(missing),
            "errors": errors,
            "with_proposals": sum(bool(row["proposals"]) for row in successful),
            "without_proposals": sum(not row["proposals"] for row in successful),
        },
        "latency_median_sec": statistics.median(
            float(row["latency_sec"]) for row in successful
        ),
        "max_proposals": max((len(row["proposals"]) for row in successful), default=0),
        "shards": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "external_api_required": False,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
