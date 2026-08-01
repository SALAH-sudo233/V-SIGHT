#!/usr/bin/env python3
"""Audit conservative COCO reference-box coverage for the E2b queue."""

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

from vsight.e1_data import load_coco_index, sha256, xywh_to_clipped_xyxy  # noqa: E402
from vsight.relation_supervision import extract_reference_phrase, parse_relation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e2b/queue/e2b_refcocog.summary.json",
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=ROOT / "data/e1/source/e1_source.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e2b/reference")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = parse_args()
    queue = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    coco_path = resolve_path(source["coco_instances"]["path"])
    if sha256(coco_path) != source["coco_instances"]["sha256"]:
        raise ValueError("COCO instances hash mismatch")
    coco = load_coco_index(coco_path)
    category_ids = {name: category_id for category_id, name in coco.categories.items()}
    wanted = {
        split: {
            str(row["query_id"]): row
            for row in read_gzip(resolve_path(item["path"]))
        }
        for split, item in queue["outputs"].items()
    }
    source_rows = {}
    for item in source["sources"]["refcocog_umd"]["query_shards"].values():
        for row in read_gzip(resolve_path(item["path"])):
            query_id = str(row["query_id"])
            if any(query_id in values for values in wanted.values()):
                source_rows[query_id] = row
    expected = set().union(*(set(values) for values in wanted.values()))
    if set(source_rows) != expected:
        raise ValueError(f"missing E2b source rows: {len(expected - set(source_rows))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: args.output_dir / f"e2b_reference_candidates.{split}.jsonl.gz"
        for split in wanted
    }
    summary_path = args.output_dir / "e2b_reference_candidates.summary.json"
    if not args.force and any(path.exists() for path in (*outputs.values(), summary_path)):
        raise FileExistsError("reference audit exists; pass --force to replace")
    stats = {split: Counter() for split in wanted}
    temporary = {split: path.with_name(path.name + ".tmp") for split, path in outputs.items()}
    try:
        contexts = {split: deterministic_gzip(path) for split, path in temporary.items()}
        handles = {split: context.__enter__() for split, context in contexts.items()}
        try:
            for split, rows in wanted.items():
                for query_id in sorted(rows):
                    row = source_rows[query_id]
                    parsed = parse_relation(
                        str(row["query"]),
                        str(row["category_name"]),
                        tuple(category_ids),
                    )
                    reference_candidates = []
                    status = "no_relation_pattern"
                    if parsed.relation is not None:
                        status = "no_reference_category"
                    if len(parsed.reference_categories) > 1:
                        status = "multiple_reference_categories"
                    elif len(parsed.reference_categories) == 1:
                        reference_category = parsed.reference_categories[0]
                        reference_category_id = category_ids[reference_category]
                        ann_ids = coco.category_annotations.get(
                            (int(row["image_id"]), reference_category_id), ()
                        )
                        for ann_id in ann_ids:
                            if int(ann_id) == int(row["ann_id"]):
                                continue
                            annotation = coco.annotations[int(ann_id)]
                            reference_candidates.append(
                                {
                                    "ann_id": int(ann_id),
                                    "category_id": reference_category_id,
                                    "category_name": reference_category,
                                    "bbox_xyxy": xywh_to_clipped_xyxy(
                                        annotation["bbox"],
                                        int(row["image_width"]),
                                        int(row["image_height"]),
                                    ),
                                }
                            )
                        if not reference_candidates:
                            status = "reference_category_no_annotation"
                        elif len(reference_candidates) == 1:
                            status = "unique_reference_annotation"
                        else:
                            status = "multiple_reference_annotations"
                    stats[split][status] += 1
                    if parsed.relation:
                        stats[split][f"relation:{parsed.relation}"] += 1
                    output = {
                        "schema_version": "vsight_e2b_reference_candidate_v1",
                        "record_type": "training_supervision_audit",
                        "query_id": query_id,
                        "data_split": split,
                        "image_id": row["image_id"],
                        "image_filename": row["image_filename"],
                        "query": row["query"],
                        "target_ann_id": row["ann_id"],
                        "target_category_id": row["category_id"],
                        "target_category_name": row["category_name"],
                        "target_bbox_xyxy": row["gt_bbox_xyxy"],
                        "relation": parsed.relation,
                        "reference_phrase": (
                            parsed.reference_categories[0]
                            if len(parsed.reference_categories) == 1
                            else extract_reference_phrase(str(row["query"]))
                        ),
                        "reference_categories": list(parsed.reference_categories),
                        "reference_candidates": reference_candidates,
                        "reference_status": status,
                        "verifier_training_eligible": status == "unique_reference_annotation",
                    }
                    handles[split].write(
                        json.dumps(
                            output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
        finally:
            for split, context in contexts.items():
                context.__exit__(None, None, None)
        for split in temporary:
            temporary[split].replace(outputs[split])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    summary = {
        "schema_version": "vsight_e2b_reference_coverage_v1",
        "status": "automatic_coco_coverage_audited",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(args.queue_summary), "sha256": sha256(args.queue_summary)},
        "sources": {"path": str(args.source_summary), "sha256": sha256(args.source_summary)},
        "statistics": {split: dict(sorted(values.items())) for split, values in stats.items()},
        "outputs": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for split, path in outputs.items()
        },
        "api_required_for_this_stage": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
