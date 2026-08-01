#!/usr/bin/env python3
"""Freeze the RefCOCOg slice used for task-matched E2b generation."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p1-summary",
        type=Path,
        default=ROOT / "data/e1/p1/e1_p1_queries.summary.json",
    )
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=ROOT / "data/e1/p1/candidates/e1_p1_candidates.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e2b/queue")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    p1 = json.loads(args.p1_summary.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    if candidates.get("status") != "complete":
        raise ValueError("P1 candidate generation is incomplete")
    if candidates["queue_manifest"]["sha256"] != sha256(args.p1_summary):
        raise ValueError("candidate and P1 queue manifests disagree")

    outputs = {
        split: args.output_dir / f"e2b_refcocog.{split}.jsonl.gz"
        for split in ("train", "calibration")
    }
    summary_path = args.output_dir / "e2b_refcocog.summary.json"
    existing = [path for path in (*outputs.values(), summary_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError("E2b queue exists; pass --force to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = {}
    for split, item in p1["outputs"].items():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"P1 queue hash mismatch: {path}")
        rows = [row for row in read_gzip(path) if row["source_dataset"] == "RefCOCOg"]
        if len({row["query_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate E2b query IDs")
        selected[split] = sorted(rows, key=lambda row: int(row["queue_rank"]))
        write_rows(outputs[split], selected[split])

    if {row["image_id"] for row in selected["train"]} & {
        row["image_id"] for row in selected["calibration"]
    }:
        raise ValueError("E2b train/calibration images overlap")
    summary = {
        "schema_version": "vsight_e2b_refcocog_queue_v1",
        "status": "t2_and_challenger_reused_t4_pending",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "p1_queue": {"path": str(args.p1_summary), "sha256": sha256(args.p1_summary)},
        "p1_candidates": {
            "path": str(args.candidate_summary),
            "sha256": sha256(args.candidate_summary),
        },
        "policy": {
            "dataset": "RefCOCOg",
            "t2_baseline": "reuse exact P1 output",
            "challenger": "reuse exact P1 binding-aware output",
            "t4_baseline": "generate canonical repaired-500 Qwen T4 prompt",
            "external_api_required": False,
        },
        "outputs": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "records": len(selected[split]),
                "unique_images": len({row["image_id"] for row in selected[split]}),
            }
            for split, path in outputs.items()
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
