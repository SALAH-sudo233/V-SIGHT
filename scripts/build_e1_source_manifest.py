#!/usr/bin/env python3
"""Build the protected-image-disjoint E1 positive-query source manifest."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.data_isolation import audit_splits  # noqa: E402
from vsight.e1_data import (  # noqa: E402
    SourceSpec,
    assign_image_splits,
    image_split_records,
    iter_query_records,
    load_coco_index,
    load_ref_records,
    protected_image_ids,
    sha256,
    source_image_ids,
)


DEFAULT_LENS_DATA = Path("/home/u2025141034/models/LENS/data")
DEFAULT_INSTANCES = Path(
    "/home/u2025141034/CHAIR/coco/annotations/instances_train2014.json"
)
DEFAULT_DEV = ROOT / "legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json"
DEFAULT_HELDOUT = Path(
    "/home/u2025141034/benchmark/repaired/refcocog_1996_heldout.manual_v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build query-level E1 sources without touching protected images."
    )
    parser.add_argument("--lens-data", type=Path, default=DEFAULT_LENS_DATA)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="COCO train2014 JPEG directory (defaults under --lens-data)",
    )
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e1/source")
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--seed", default="vsight-e1-image-split-v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_specs(lens_data: Path) -> list[SourceSpec]:
    return [
        SourceSpec(
            name="refcoco_unc",
            dataset="RefCOCO",
            split_by="unc",
            path=lens_data / "refcoco/refs(unc).p",
        ),
        SourceSpec(
            name="refcoco_plus_unc",
            dataset="RefCOCO+",
            split_by="unc",
            path=lens_data / "refcoco+/refs(unc).p",
        ),
        SourceSpec(
            name="refcocog_umd",
            dataset="RefCOCOg",
            split_by="umd",
            path=lens_data / "refcocog/refs(umd).p",
        ),
    ]


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    """Write gzip with a fixed header timestamp so reruns hash identically."""

    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def write_jsonl_row(handle: TextIO, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_image_index(path: Path, rows: Iterator[dict]) -> int:
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            write_jsonl_row(handle, row)
            count += 1
    temporary.replace(path)
    return count


def _input_stats(refs: list[dict], protected: frozenset[int]) -> dict:
    train_refs = [row for row in refs if row.get("split") == "train"]
    excluded = [row for row in train_refs if int(row["image_id"]) in protected]
    retained = [row for row in train_refs if int(row["image_id"]) not in protected]
    return {
        "original_train_refs": len(train_refs),
        "original_train_queries": sum(len(row["sentences"]) for row in train_refs),
        "original_train_images": len({int(row["image_id"]) for row in train_refs}),
        "protected_overlap_images": len(
            {int(row["image_id"]) for row in excluded}
        ),
        "excluded_refs": len(excluded),
        "excluded_queries": sum(len(row["sentences"]) for row in excluded),
        "retained_refs": len(retained),
        "retained_queries": sum(len(row["sentences"]) for row in retained),
        "retained_images": len({int(row["image_id"]) for row in retained}),
    }


def _new_emission_stats() -> dict:
    return {
        "queries": Counter(),
        "refs": {"train": set(), "calibration": set()},
        "images": {"train": set(), "calibration": set()},
        "categories": {"train": Counter(), "calibration": Counter()},
        "same_category_hard_negative_queries": Counter(),
    }


def _finalize_emission_stats(stats: dict) -> dict:
    return {
        split: {
            "queries": stats["queries"][split],
            "refs": len(stats["refs"][split]),
            "images": len(stats["images"][split]),
            "categories": len(stats["categories"][split]),
            "same_category_hard_negative_queries": stats[
                "same_category_hard_negative_queries"
            ][split],
            "same_category_hard_negative_rate": (
                stats["same_category_hard_negative_queries"][split]
                / stats["queries"][split]
                if stats["queries"][split]
                else 0.0
            ),
            "category_query_counts": dict(
                sorted(stats["categories"][split].items())
            ),
        }
        for split in ("train", "calibration")
    }


def _render_report(summary: dict) -> str:
    rows = []
    for name, source in summary["sources"].items():
        train = source["emitted"]["train"]
        calibration = source["emitted"]["calibration"]
        rows.append(
            f"| {name} | {source['input']['retained_images']:,} | "
            f"{train['queries']:,} | {calibration['queries']:,} | "
            f"{train['same_category_hard_negative_rate']:.1%} |"
        )
    split = summary["split_assignment"]
    return "\n".join(
        [
            "# V-SIGHT E1 Source Corpus",
            "",
            "**Status:** positive source frozen; candidate inference and typed nulls pending",
            "",
            "This corpus contains standard RefCOCO-family training queries and GT",
            "supervision only. It does not contain the 114 IoU=0 audit groups, repaired",
            "positive candidates, model candidates, action labels, or inferred nulls.",
            "",
            "## Image boundary",
            "",
            f"- Protected repaired-500 images: {summary['protected']['dev_images']:,}",
            f"- Protected repaired-1996 images: {summary['protected']['heldout_images']:,}",
            f"- Protected union: {summary['protected']['union_images']:,}",
            f"- Retained source images: {split['total_images']:,}",
            f"- Retained JPEGs verified: {summary['images']['verified_jpegs']:,}",
            f"- Train images: {split['train_images']:,}",
            f"- Calibration images: {split['calibration_images']:,}",
            f"- Isolation check: {'PASS' if summary['isolation']['all_disjoint'] else 'FAIL'}",
            "",
            "## Query volume",
            "",
            "| Source | Retained images | Train queries | Calibration queries | Train queries with same-class distractor |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"Exact cross-source duplicate target/query rows retained with provenance: "
            f"{summary['deduplication']['exact_duplicate_queries']:,}",
            "",
            "## Interpretation",
            "",
            "The volume is sufficient to construct E1 candidate supervision. Same-class",
            "availability is an annotation-derived capacity statistic, not an inference",
            "feature. The next build stage must generate one baseline and one challenger",
            "per query, deterministic localization proposals, and independently validated",
            "typed nulls. Train-time balancing must be sampler-side; calibration remains at",
            "its natural image-group distribution.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    image_root = args.image_root or args.lens_data / "refcoco/train2014"
    specs = source_specs(args.lens_data)
    required = [args.instances, args.dev, args.heldout, *(spec.path for spec in specs)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(missing))
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_paths = {
        (spec.name, split): args.output_dir / f"e1_source.{spec.name}.{split}.jsonl.gz"
        for spec in specs
        for split in ("train", "calibration")
    }
    train_index = args.output_dir / "e1_images.train.jsonl"
    calibration_index = args.output_dir / "e1_images.calibration.jsonl"
    summary_path = args.output_dir / "e1_source.summary.json"
    isolation_path = args.output_dir / "e1_source.isolation.json"
    report_path = args.output_dir / "E1_SOURCE_REPORT.md"
    outputs = [
        *query_paths.values(),
        train_index,
        calibration_index,
        summary_path,
        isolation_path,
        report_path,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "outputs already exist; pass --force to replace generated files: "
            + ", ".join(existing)
        )

    protected_dev = protected_image_ids([args.dev])
    protected_heldout = protected_image_ids([args.heldout])
    if protected_dev & protected_heldout:
        raise ValueError("dev and held-out protected image sets overlap")
    protected = protected_dev | protected_heldout

    coco = load_coco_index(args.instances)
    refs_by_source = {spec.name: load_ref_records(spec.path) for spec in specs}
    all_images: set[int] = set()
    for spec in specs:
        all_images.update(source_image_ids(refs_by_source[spec.name], protected))
    unresolved_images = sorted(all_images - set(coco.images))
    if unresolved_images:
        raise ValueError(f"source images absent from COCO metadata: {unresolved_images[:10]}")
    missing_jpegs = [
        image_id
        for image_id in sorted(all_images)
        if not (image_root / str(coco.images[image_id]["file_name"])).is_file()
    ]
    if missing_jpegs:
        raise FileNotFoundError(f"source JPEGs are missing: {missing_jpegs[:10]}")
    train_images, calibration_images = assign_image_splits(
        all_images, args.calibration_fraction, args.seed
    )

    write_image_index(train_index, image_split_records(train_images, "train", coco))
    write_image_index(
        calibration_index,
        image_split_records(calibration_images, "calibration", coco),
    )

    source_summaries = {}
    seen_query_ids: set[str] = set()
    seen_semantic_rows: set[tuple[int, int, str]] = set()
    exact_duplicates = 0
    for spec in specs:
        paths = {
            split: query_paths[(spec.name, split)] for split in ("train", "calibration")
        }
        temporaries = {
            split: path.with_name(path.name + ".tmp") for split, path in paths.items()
        }
        stats = _new_emission_stats()
        try:
            with ExitStack() as stack:
                handles = {
                    split: stack.enter_context(deterministic_gzip_writer(temporary))
                    for split, temporary in temporaries.items()
                }
                records = iter_query_records(
                    refs_by_source[spec.name],
                    spec,
                    coco,
                    protected,
                    train_images,
                    calibration_images,
                )
                for record in records:
                    query_id = record["query_id"]
                    if query_id in seen_query_ids:
                        raise ValueError(f"duplicate query_id: {query_id}")
                    seen_query_ids.add(query_id)
                    semantic_key = (
                        record["image_id"],
                        record["ann_id"],
                        record["query"].casefold(),
                    )
                    if semantic_key in seen_semantic_rows:
                        exact_duplicates += 1
                    else:
                        seen_semantic_rows.add(semantic_key)

                    split = record["data_split"]
                    write_jsonl_row(handles[split], record)
                    stats["queries"][split] += 1
                    stats["refs"][split].add(record["ref_id"])
                    stats["images"][split].add(record["image_id"])
                    stats["categories"][split][record["category_name"]] += 1
                    if record["same_category_distractor_count"] > 0:
                        stats["same_category_hard_negative_queries"][split] += 1
            for split, temporary in temporaries.items():
                temporary.replace(paths[split])
        except BaseException:
            for temporary in temporaries.values():
                temporary.unlink(missing_ok=True)
            raise

        source_summaries[spec.name] = {
            "dataset": spec.dataset,
            "split_by": spec.split_by,
            "path": str(spec.path),
            "sha256": sha256(spec.path),
            "input": _input_stats(refs_by_source[spec.name], protected),
            "emitted": _finalize_emission_stats(stats),
            "query_shards": {
                split: {
                    "path": manifest_path(paths[split]),
                    "sha256": sha256(paths[split]),
                    "bytes": paths[split].stat().st_size,
                }
                for split in ("train", "calibration")
            },
        }

    isolation = audit_splits(
        {
            "train": train_index,
            "calibration": calibration_index,
            "dev": args.dev,
            "heldout": args.heldout,
        }
    )
    if not isolation["all_disjoint"]:
        raise ValueError("E1 image isolation failed")
    for split in isolation["splits"].values():
        split["path"] = manifest_path(Path(split["path"]))
    atomic_json(isolation_path, isolation)

    summary = {
        "schema_version": "vsight_e1_source_manifest_v1",
        "status": "positive_source_frozen_candidates_and_nulls_pending",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build": {
            "script": manifest_path(Path(__file__)),
            "calibration_fraction": args.calibration_fraction,
            "image_split_seed": args.seed,
            "canonical_source_versions": [
                "RefCOCO unc", "RefCOCO+ unc", "RefCOCOg umd"
            ],
        },
        "coco_instances": {
            "path": str(args.instances),
            "sha256": sha256(args.instances),
        },
        "images": {
            "root": str(image_root.resolve()),
            "verified_jpegs": len(all_images),
        },
        "protected": {
            "dev_path": manifest_path(args.dev),
            "dev_sha256": sha256(args.dev),
            "dev_images": len(protected_dev),
            "heldout_path": str(args.heldout),
            "heldout_sha256": sha256(args.heldout),
            "heldout_images": len(protected_heldout),
            "union_images": len(protected),
        },
        "split_assignment": {
            "total_images": len(all_images),
            "train_images": len(train_images),
            "calibration_images": len(calibration_images),
            "train_index": manifest_path(train_index),
            "train_index_sha256": sha256(train_index),
            "calibration_index": manifest_path(calibration_index),
            "calibration_index_sha256": sha256(calibration_index),
        },
        "sources": source_summaries,
        "deduplication": {
            "policy": "retain_exact_cross_source_duplicates_with_provenance",
            "unique_query_ids": len(seen_query_ids),
            "unique_target_query_rows": len(seen_semantic_rows),
            "exact_duplicate_queries": exact_duplicates,
        },
        "isolation": {
            "all_disjoint": isolation["all_disjoint"],
            "report_path": manifest_path(isolation_path),
            "comparisons": isolation["comparisons"],
        },
        "training_eligibility": {
            "positive_source": True,
            "candidate_action_training": False,
            "reason": "baseline, challenger, action labels, and typed nulls are not built",
        },
    }
    atomic_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
