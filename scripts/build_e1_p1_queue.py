#!/usr/bin/env python3
"""Freeze a compute-bounded P1 query subset for local candidate inference."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import (  # noqa: E402
    inference_queue_record,
    select_compute_subset,
    sha256,
)


FORBIDDEN_QUEUE_FIELDS = {
    "ann_id",
    "gt_bbox_xywh",
    "gt_bbox_xyxy",
    "category_id",
    "category_name",
    "same_category_distractor_count",
    "hallucination_type",
    "candidate_source_id",
    "audit_decision",
    "human_review_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=ROOT / "data/e1/source/e1_source.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e1/p1")
    parser.add_argument("--train-queries", type=int, default=12000)
    parser.add_argument("--calibration-queries", type=int, default=2000)
    parser.add_argument("--seed", default="vsight-e1-p1-query-subset-v1")
    parser.add_argument("--force", action="store_true")
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


def quotas(total: int, sources: list[str]) -> dict[str, int]:
    if total < 0:
        raise ValueError("query totals must be non-negative")
    base, remainder = divmod(total, len(sources))
    return {
        source: base + (index < remainder)
        for index, source in enumerate(sorted(sources))
    }


def read_shard(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with deterministic_gzip_writer(temporary) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def selection_stats(rows: list[dict], base: dict) -> dict:
    categories = Counter(str(row["category_name"]) for row in rows)
    datasets = Counter(str(row["source_dataset"]) for row in rows)
    targets = {(int(row["image_id"]), int(row["ann_id"])) for row in rows}
    images = {int(row["image_id"]) for row in rows}
    return {
        **base,
        "unique_images": len(images),
        "unique_targets": len(targets),
        "dataset_queries": dict(sorted(datasets.items())),
        "category_queries": dict(sorted(categories.items())),
        "person_query_rate": categories["person"] / len(rows),
    }


def main() -> int:
    args = parse_args()
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if not source_summary.get("isolation", {}).get("all_disjoint"):
        raise ValueError("E1 source has not passed image isolation")
    sources = sorted(source_summary["sources"])
    requested = {
        "train": args.train_queries,
        "calibration": args.calibration_queries,
    }
    split_quotas = {split: quotas(total, sources) for split, total in requested.items()}

    records = {split: {} for split in requested}
    for source in sources:
        for split in requested:
            shard = source_summary["sources"][source]["query_shards"][split]
            path = resolve_path(shard["path"])
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"source shard hash mismatch: {path}")
            rows = read_shard(path)
            if any(row["data_split"] != split for row in rows):
                raise ValueError(f"source shard contains the wrong split: {path}")
            records[split][source] = rows

    selected = {}
    stats = {}
    for split in requested:
        selected[split], base_stats = select_compute_subset(
            records[split], split_quotas[split], f"{args.seed}:{split}"
        )
        stats[split] = selection_stats(selected[split], base_stats)

    train_images = {int(row["image_id"]) for row in selected["train"]}
    calibration_images = {
        int(row["image_id"]) for row in selected["calibration"]
    }
    if train_images & calibration_images:
        raise ValueError("P1 train and calibration images overlap")

    queue_rows = {
        split: [
            inference_queue_record(row, rank)
            for rank, row in enumerate(selected[split])
        ]
        for split in selected
    }
    leaks = {
        field
        for rows in queue_rows.values()
        for row in rows
        for field in row
        if field in FORBIDDEN_QUEUE_FIELDS
    }
    if leaks:
        raise ValueError(f"forbidden inference queue fields: {sorted(leaks)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: args.output_dir / f"e1_p1_queries.{split}.jsonl.gz"
        for split in requested
    }
    summary_path = args.output_dir / "e1_p1_queries.summary.json"
    report_path = args.output_dir / "E1_P1_QUEUE_REPORT.md"
    outputs = [*paths.values(), summary_path, report_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "outputs exist; pass --force to replace generated files: "
            + ", ".join(existing)
        )
    for split, path in paths.items():
        write_rows(path, queue_rows[split])

    summary = {
        "schema_version": "vsight_e1_p1_query_manifest_v1",
        "status": "candidate_inference_queue_frozen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": manifest_path(args.source_summary),
            "sha256": sha256(args.source_summary),
        },
        "selection": {
            "seed": args.seed,
            "policy": (
                "equal source quotas; deterministic hash order; exact target/query "
                "deduplication; maximize unique images before allowing reuse"
            ),
            "quotas": split_quotas,
        },
        "statistics": stats,
        "outputs": {
            split: {
                "path": manifest_path(path),
                "sha256": sha256(path),
                "records": len(queue_rows[split]),
                "bytes": path.stat().st_size,
            }
            for split, path in paths.items()
        },
        "inference_input": {
            "forbidden_fields": sorted(FORBIDDEN_QUEUE_FIELDS),
            "forbidden_fields_present": [],
            "requires_external_api": False,
            "requires_local_grounding_checkpoint": True,
        },
    }
    write_json(summary_path, summary)
    report_path.write_text(
        "\n".join(
            [
                "# E1 P1 Candidate-Inference Queue",
                "",
                "**Status:** frozen; local candidate inference pending",
                "",
                f"- Train queries: {len(queue_rows['train']):,}",
                f"- Train images: {stats['train']['unique_images']:,}",
                f"- Calibration queries: {len(queue_rows['calibration']):,}",
                f"- Calibration images: {stats['calibration']['unique_images']:,}",
                "- Exact target/query duplicates: excluded",
                "- GT, ann_id, category, audit labels, and candidate source: absent",
                "- External API: not required for baseline/challenger inference",
                "",
                "Typed null and target/reference swap generation is a separate visual",
                "annotation stage and is not included in this queue.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
