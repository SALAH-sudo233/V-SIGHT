#!/usr/bin/env python3
"""Build source-disjoint annotation pairs for E2 ranking auxiliary losses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402


RELATION_TERMS = frozenset(
    "left right above below behind front next near between under over beside "
    "holding wearing sitting standing riding looking facing with on in by"
    .split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=ROOT / "data/e1/source/e1_source.summary.json",
    )
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=ROOT / "data/e1/supervision/e1_annotation_candidates.summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/e1/p1/selector/e2_annotation_auxiliary.train.jsonl.gz",
    )
    parser.add_argument("--same-class-per-source", type=int, default=4000)
    parser.add_argument("--localization-per-source", type=int, default=2000)
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


def relation_score(query: str) -> int:
    words = {word.strip(".,;:!?()[]{}\"").casefold() for word in query.split()}
    return len(words & RELATION_TERMS)


def stable_rank(row: dict, auxiliary_type: str) -> tuple:
    digest = hashlib.sha256(
        f"vsight-e2-aux-v1:{auxiliary_type}:{row['query_id']}".encode("utf-8")
    ).digest()
    return (-relation_score(str(row["query"])), digest, str(row["query_id"]))


def pair_record(source: dict, bank: dict, auxiliary_type: str) -> dict:
    if auxiliary_type == "same_category_instance":
        candidate = max(
            bank["same_category_candidates"],
            key=lambda value: float(value["annotation_hardness"]),
        )
    elif auxiliary_type == "localization_quality":
        candidate = min(
            bank["localization_candidates"],
            key=lambda value: float(value["iou_to_gt"]),
        )
    else:
        raise ValueError(auxiliary_type)
    return {
        "schema_version": "vsight_e2_annotation_auxiliary_pair_v1",
        "supervision_kind": "annotation_auxiliary",
        "auxiliary_type": auxiliary_type,
        "query_id": f"aux:{auxiliary_type}:{source['query_id']}",
        "source_query_id": source["query_id"],
        "group_id": source["group_id"],
        "data_split": "train",
        "source_dataset": source["source_dataset"],
        "image_id": source["image_id"],
        "image_filename": source["image_filename"],
        "query": source["query"],
        "baseline_bbox_xyxy": bank["gt_bbox_xyxy"],
        "challenger_bbox_xyxy": candidate["bbox_xyxy"],
        "baseline_iou": 1.0,
        "challenger_iou": float(candidate["iou_to_gt"]),
        "selector_action": "keep",
        "selector_eligible": True,
        "training_only": True,
    }


def main() -> int:
    args = parse_args()
    if args.same_class_per_source < 0 or args.localization_per_source < 0:
        raise ValueError("pair quotas must be non-negative")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace: {args.output}")
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    candidate_summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    bank_path = resolve_path(candidate_summary["outputs"]["train"]["path"])
    if sha256(bank_path) != candidate_summary["outputs"]["train"]["sha256"]:
        raise ValueError("annotation candidate bank hash mismatch")
    banks = {int(row["target_ann_id"]): row for row in read_gzip(bank_path)}

    # Keep one expression per target so repeated annotations cannot dominate.
    best_by_target: dict[int, dict] = {}
    for source in source_summary["sources"].values():
        item = source["query_shards"]["train"]
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source shard hash mismatch: {path}")
        for row in read_gzip(path):
            ann_id = int(row["ann_id"])
            current = best_by_target.get(ann_id)
            if current is None or stable_rank(row, "expression") < stable_rank(
                current, "expression"
            ):
                best_by_target[ann_id] = row

    available = defaultdict(list)
    for ann_id, source in best_by_target.items():
        bank = banks.get(ann_id)
        if bank is None:
            continue
        dataset = str(source["source_dataset"])
        if bank.get("same_category_candidates"):
            available[(dataset, "same_category_instance")].append((source, bank))
        if bank.get("localization_candidates"):
            available[(dataset, "localization_quality")].append((source, bank))

    selected = []
    datasets = sorted({key[0] for key in available})
    for dataset in datasets:
        for auxiliary_type, quota in (
            ("same_category_instance", args.same_class_per_source),
            ("localization_quality", args.localization_per_source),
        ):
            candidates = sorted(
                available[(dataset, auxiliary_type)],
                key=lambda item: stable_rank(item[0], auxiliary_type),
            )
            if len(candidates) < quota:
                raise ValueError(
                    f"insufficient {dataset} {auxiliary_type} rows: {len(candidates)} < {quota}"
                )
            selected.extend(
                pair_record(source, bank, auxiliary_type)
                for source, bank in candidates[:quota]
            )
    selected.sort(key=lambda row: str(row["query_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in selected:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)

    counts = Counter(
        (row["source_dataset"], row["auxiliary_type"]) for row in selected
    )
    summary = {
        "schema_version": "vsight_e2_annotation_auxiliary_manifest_v1",
        "status": "training_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(selected),
        "counts": {
            f"{dataset}:{kind}": count
            for (dataset, kind), count in sorted(counts.items())
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "inference_features": [
            "image",
            "complete_query",
            "candidate_views",
            "source_agnostic_geometry",
        ],
        "forbidden_model_features": [
            "gt_identity",
            "candidate_source_id",
            "source_dataset",
            "auxiliary_type",
        ],
    }
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
