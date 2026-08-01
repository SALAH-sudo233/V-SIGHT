#!/usr/bin/env python3
"""Reparse saved E2b T4 raw text after parser-only compatibility fixes."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.candidate_generation import parse_t4_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern", default=str(ROOT / "data/e2b/t4_candidates/e2b_t4.shard-*.jsonl")
    )
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e2b/queue/e2b_refcocog.summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    image_sizes = {}
    for item in queue["outputs"].values():
        path = ROOT / item["path"]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    image_sizes[str(row["query_id"])] = (
                        int(row["image_width"]),
                        int(row["image_height"]),
                    )
    totals = {"records": 0, "valid": 0, "invalid": 0}
    for value in sorted(glob.glob(args.pattern)):
        path = Path(value)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with path.open(encoding="utf-8") as source, temporary.open(
                "w", encoding="utf-8"
            ) as destination:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prior = row.get("t4") or {}
                    reparsed = parse_t4_output(
                        str(prior.get("raw_output_text") or ""),
                        image_sizes[str(row["query_id"])],
                    )
                    reparsed["raw_output_text"] = prior.get("raw_output_text", "")
                    row["t4"] = reparsed
                    destination.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    totals["records"] += 1
                    totals["valid" if reparsed["parse_valid"] else "invalid"] += 1
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
