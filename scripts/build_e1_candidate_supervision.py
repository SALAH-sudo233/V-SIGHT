#!/usr/bin/env python3
"""Build annotation-derived hard-instance and localization candidates for E1."""

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
    annotation_candidate_record,
    load_coco_index,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=ROOT / "data/e1/source/e1_source.summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e1/supervision"
    )
    parser.add_argument("--max-same-category", type=int, default=5)
    parser.add_argument("--max-same-category-iou", type=float, default=0.9)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def iter_query_shard(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_row(handle: TextIO, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def main() -> int:
    args = parse_args()
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if not source.get("isolation", {}).get("all_disjoint"):
        raise ValueError("source manifest has not passed image isolation")
    instances_path = resolve_manifest_path(source["coco_instances"]["path"])
    if sha256(instances_path) != source["coco_instances"]["sha256"]:
        raise ValueError("COCO instances hash differs from the frozen source manifest")

    targets: dict[tuple[int, int], str] = {}
    query_counts = Counter()
    for source_info in source["sources"].values():
        for split, shard in source_info["query_shards"].items():
            path = resolve_manifest_path(shard["path"])
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"query shard hash mismatch: {path}")
            for row in iter_query_shard(path):
                if row["data_split"] != split:
                    raise ValueError(f"query shard contains the wrong split: {path}")
                key = (int(row["image_id"]), int(row["ann_id"]))
                previous = targets.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"target {key} crosses train/calibration")
                query_counts[split] += 1

    coco = load_coco_index(instances_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_paths = {
        split: args.output_dir / f"e1_annotation_candidates.{split}.jsonl.gz"
        for split in ("train", "calibration")
    }
    summary_path = args.output_dir / "e1_annotation_candidates.summary.json"
    report_path = args.output_dir / "E1_CANDIDATE_SUPERVISION_REPORT.md"
    outputs = [*bank_paths.values(), summary_path, report_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "outputs already exist; pass --force to replace generated files: "
            + ", ".join(existing)
        )

    stats = {
        split: {
            "targets": 0,
            "same_category_candidates": 0,
            "targets_with_same_category_candidates": 0,
            "targets_without_same_category_candidates": 0,
            "overlap_candidates_excluded": 0,
            "localization_candidates": 0,
            "same_category_available": Counter(),
            "localization_types": Counter(),
        }
        for split in ("train", "calibration")
    }
    temporaries = {
        split: path.with_name(path.name + ".tmp")
        for split, path in bank_paths.items()
    }
    try:
        handles = {
            split: deterministic_gzip_writer(temporary)
            for split, temporary in temporaries.items()
        }
        with handles["train"] as train_handle, handles[
            "calibration"
        ] as calibration_handle:
            output_handles = {
                "train": train_handle,
                "calibration": calibration_handle,
            }
            for (image_id, ann_id), split in sorted(targets.items()):
                annotation = coco.annotations.get(ann_id)
                if annotation is None or int(annotation["image_id"]) != image_id:
                    raise ValueError(f"unresolved source target: {(image_id, ann_id)}")
                row = annotation_candidate_record(
                    ann_id,
                    split,
                    coco,
                    args.max_same_category,
                    args.max_same_category_iou,
                )
                write_row(output_handles[split], row)
                current = stats[split]
                current["targets"] += 1
                availability_key = (
                    "targets_with_same_category_candidates"
                    if row["same_category_candidates"]
                    else "targets_without_same_category_candidates"
                )
                current[availability_key] += 1
                current["overlap_candidates_excluded"] += row[
                    "same_category_overlap_excluded"
                ]
                current["same_category_candidates"] += len(
                    row["same_category_candidates"]
                )
                current["localization_candidates"] += len(
                    row["localization_candidates"]
                )
                current["same_category_available"][
                    str(row["same_category_available"])
                ] += 1
                current["localization_types"].update(
                    item["candidate_type"] for item in row["localization_candidates"]
                )
        for split, temporary in temporaries.items():
            temporary.replace(bank_paths[split])
    except BaseException:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        raise

    rendered_stats = {}
    for split, current in stats.items():
        rendered_stats[split] = {
            "queries_linked": query_counts[split],
            "unique_targets": current["targets"],
            "same_category_candidates": current["same_category_candidates"],
            "targets_with_same_category_candidates": current[
                "targets_with_same_category_candidates"
            ],
            "targets_without_same_category_candidates": current[
                "targets_without_same_category_candidates"
            ],
            "overlap_candidates_excluded": current[
                "overlap_candidates_excluded"
            ],
            "localization_candidates": current["localization_candidates"],
            "same_category_available_distribution": dict(
                sorted(
                    current["same_category_available"].items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "localization_type_counts": dict(
                sorted(current["localization_types"].items())
            ),
        }

    summary = {
        "schema_version": "vsight_e1_candidate_supervision_manifest_v1",
        "status": "annotation_candidates_ready_semantic_swaps_pending",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": manifest_path(args.source_summary),
            "sha256": sha256(args.source_summary),
        },
        "max_same_category_per_target": args.max_same_category,
        "max_same_category_iou_exclusive": args.max_same_category_iou,
        "selection": (
            "top annotation_hardness: 0.6 size similarity + "
            "0.4 inverse normalized center distance"
        ),
        "outputs": {
            split: {
                "path": manifest_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for split, path in bank_paths.items()
        },
        "statistics": rendered_stats,
        "training_eligibility": {
            "same_category_ranking": True,
            "localization_quality": True,
            "candidate_listwise_action": False,
            "counterfactual_binding": False,
            "reason": "model candidates and query-level semantic swaps/nulls are pending",
        },
    }
    write_json(summary_path, summary)
    report_path.write_text(
        "\n".join(
            [
                "# E1 Candidate Supervision",
                "",
                "**Status:** annotation candidates ready; semantic candidates pending",
                "",
                f"- Train targets: {rendered_stats['train']['unique_targets']:,}",
                f"- Calibration targets: {rendered_stats['calibration']['unique_targets']:,}",
                f"- Train same-class candidates: {rendered_stats['train']['same_category_candidates']:,}",
                f"- Train targets without a safe (< {args.max_same_category_iou:g} IoU) same-class box: "
                f"{rendered_stats['train']['targets_without_same_category_candidates']:,}",
                f"- Near-duplicate train annotations excluded: "
                f"{rendered_stats['train']['overlap_candidates_excluded']:,}",
                f"- Train localization candidates: {rendered_stats['train']['localization_candidates']:,}",
                "- Target/reference swaps: pending query-level semantic annotation",
                "- Typed object/attribute/relation nulls: pending independent validation",
                "- Baseline/binding-aware candidates: pending frozen-checkpoint inference",
                "",
                "These candidates supervise shared support and localization heads. They do",
                "not define KEEP/SWITCH/REJECT labels and are never candidate-source input",
                "features at inference time.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
