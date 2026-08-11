#!/usr/bin/env python3
"""Deduplicate the legacy three-view queue into CCV composite query inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/e3/singlepass/target_queue/e3_singlepass_target_queue.development.jsonl.gz"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/e3/ccv/development_queue.jsonl.gz")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = arguments()
    if args.output.exists() and not args.force:
        raise FileExistsError("output exists; pass --force")
    deduplicated = {}
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            semantic_id = str(source["semantic_query_id"])
            row = {
                "schema_version": "vsight_ccv_development_input_v1",
                "query_id": semantic_id,
                "data_split": "development",
                "image_filename": str(source["image_filename"]),
                "query": str(source["query"]),
            }
            if semantic_id in deduplicated and deduplicated[semantic_id] != row:
                raise ValueError(f"inconsistent semantic query: {semantic_id}")
            deduplicated[semantic_id] = row
    if len(deduplicated) != 2500:
        raise ValueError(f"expected 2,500 semantic queries, found {len(deduplicated)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with deterministic_gzip(args.output) as handle:
        for row in sorted(deduplicated.values(), key=lambda value: value["query_id"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "vsight_ccv_development_queue_v1",
        "records": len(deduplicated),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "supervision_fields": [],
        "sealed_heldout_accessed": False,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
