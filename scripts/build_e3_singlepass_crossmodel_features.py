#!/usr/bin/env python3
"""Build model-agnostic single-pass gate features for 11-model development CV."""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import math
import re
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402
from vsight.relation_supervision import RELATION_PATTERNS  # noqa: E402
from vsight.single_pass import (  # noqa: E402
    PROPOSAL_SUPPORT_FIELDS,
    REFERENCE_GEOMETRY_FIELDS,
    candidate_reference_geometry_features,
    normalized_text,
    proposal_support_features,
    semantic_query_id,
    target_phrases,
)


DEFAULT_BENCHMARK = (
    ROOT / "legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json"
)
DEFAULT_RECORD_ROOT = Path(
    "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/"
    "run_500_semantic_strict"
)
DEFAULT_PROPOSALS = (
    ROOT / "data/e3/singlepass/target_proposals/e3_singlepass_target_dino.summary.json"
)
TASKS = ("t2_vqa_grounding", "t4_caption_grounding")
INFERENCE_FEATURE_FIELDS = (
    "original_found",
    "candidate_valid",
    "candidate_area",
    "candidate_center_x",
    "candidate_center_y",
    "candidate_aspect_log",
    "candidate_edge_distance",
    "head_parsed",
    "full_distinct",
    "reference_parsed",
    *(f"head_{name}" for name in PROPOSAL_SUPPORT_FIELDS),
    *(f"full_{name}" for name in PROPOSAL_SUPPORT_FIELDS),
    *(f"reference_{name}" for name in PROPOSAL_SUPPORT_FIELDS),
    "full_minus_head_score",
    "full_minus_head_iou",
    "full_minus_head_joint_support",
    "full_minus_head_overlap_score_030",
    *(f"reference_geometry_{name}" for name in REFERENCE_GEOMETRY_FIELDS),
    *(f"relation_{name}" for name, _ in RELATION_PATTERNS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--proposal-summary", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument(
        "--image-root", type=Path, default=Path("/home/u2025141034/benchmark/benchmark_images")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e3/singlepass/crossmodel"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def base_id(row: dict) -> str:
    value = str(row.get("base_sample_id") or row.get("sample_id") or "")
    return value.split("__", 1)[0]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def read_proposals(summary_path: Path) -> dict[str, dict[str, list[dict]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("target proposal inference is incomplete")
    output: dict[str, dict[str, list[dict]]] = {}
    for item in summary["shards"]:
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"proposal shard hash mismatch: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                semantic_id, view = str(row["query_id"]).rsplit(":", 1)
                if view in output.setdefault(semantic_id, {}):
                    raise ValueError(f"duplicate proposal view: {semantic_id}:{view}")
                output[semantic_id][view] = list(row.get("proposals") or [])
    return output


def build_metadata(benchmark_path: Path) -> dict[str, dict]:
    rows = json.loads(benchmark_path.read_text(encoding="utf-8"))
    metadata = {}
    for source in rows:
        group_id = base_id(source)
        image = str(source["image_filename"])
        positive_query = str(source.get("chosen") or source.get("positive_text") or "").strip()
        negative_query = str(source.get("rejected") or source.get("negative_text") or "").strip()
        entries = (
            (
                group_id,
                positive_query,
                True,
                "positive",
                source.get("gt_bbox_xyxy") or source.get("positive_bbox"),
            ),
            (
                str(source["sample_id"]),
                negative_query,
                False,
                str(source["hallucination_type"]),
                None,
            ),
        )
        for sample_id, query, label, kind, gt in entries:
            row = {
                "sample_id": sample_id,
                "group_id": group_id,
                "image_filename": image,
                "query": query,
                "semantic_query_id": semantic_query_id(image, query),
                "label_exists": label,
                "hallucination_type": kind,
                "gt_bbox_xyxy": gt,
            }
            if sample_id in metadata:
                if metadata[sample_id] != row:
                    raise ValueError(f"inconsistent benchmark metadata: {sample_id}")
            else:
                metadata[sample_id] = row
    if len(metadata) != 2500:
        raise ValueError(f"expected 2,500 benchmark samples, found {len(metadata)}")
    return metadata


def relation_name(query: str) -> str | None:
    text = normalized_text(query)
    return next(
        (name for name, pattern in RELATION_PATTERNS if re.search(pattern, text)), None
    )


def candidate_geometry(box, width: int, height: int) -> dict[str, float]:
    zeros = {
        "candidate_valid": 0.0,
        "candidate_area": 0.0,
        "candidate_center_x": 0.0,
        "candidate_center_y": 0.0,
        "candidate_aspect_log": 0.0,
        "candidate_edge_distance": 0.0,
    }
    if not isinstance(box, list) or len(box) != 4 or width <= 0 or height <= 0:
        return zeros
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return zeros
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return zeros
    nx1, nx2 = x1 / width, x2 / width
    ny1, ny2 = y1 / height, y2 / height
    box_width, box_height = nx2 - nx1, ny2 - ny1
    return {
        "candidate_valid": 1.0,
        "candidate_area": box_width * box_height,
        "candidate_center_x": (nx1 + nx2) / 2,
        "candidate_center_y": (ny1 + ny2) / 2,
        "candidate_aspect_log": math.log((box_width + 1e-6) / (box_height + 1e-6)),
        "candidate_edge_distance": min(nx1, ny1, 1 - nx2, 1 - ny2),
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir / "e3_singlepass_crossmodel.development.jsonl.gz"
    summary_path = args.output_dir / "e3_singlepass_crossmodel.summary.json"
    if not args.force and (output.exists() or summary_path.exists()):
        raise FileExistsError("cross-model feature manifest exists; pass --force")

    proposals = read_proposals(args.proposal_summary)
    metadata = build_metadata(args.benchmark)
    from PIL import Image

    dimensions = {}
    models = []
    records = []
    stats = Counter()
    for path_value in sorted(glob.glob(str(args.record_root / "*/records.jsonl"))):
        path = Path(path_value)
        model = path.parent.name
        models.append(model)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                task = str(source.get("task"))
                if task not in TASKS:
                    continue
                sample_id = str(source.get("sample_id") or "")
                meta = metadata.get(sample_id)
                if meta is None:
                    raise ValueError(f"model record has unknown sample: {model}:{sample_id}")
                if normalized_text(source.get("query")) != normalized_text(meta["query"]):
                    raise ValueError(f"query mismatch: {model}:{task}:{sample_id}")
                filename = meta["image_filename"]
                if filename not in dimensions:
                    with Image.open(args.image_root / Path(filename).name) as image:
                        dimensions[filename] = (image.width, image.height)
                width, height = dimensions[filename]
                box = source.get("pred_bbox_xyxy")
                found = bool(source.get("pred_found", source.get("pred_exists", False)))
                geometry = candidate_geometry(box, width, height)
                candidate = box if found and geometry["candidate_valid"] else None
                evidence = proposals.get(meta["semantic_query_id"], {})
                head_rows = evidence.get("head") or evidence.get("full") or []
                full_rows = evidence.get("full") or head_rows
                reference_rows = evidence.get("reference") or []
                head = proposal_support_features(candidate, head_rows)
                full = proposal_support_features(candidate, full_rows)
                reference = proposal_support_features(candidate, reference_rows)
                phrases = target_phrases(meta["query"])
                relation = relation_name(meta["query"])
                reference_geometry = candidate_reference_geometry_features(
                    candidate, reference_rows, width, height
                )
                features = {
                    "original_found": float(found),
                    **geometry,
                    "head_parsed": float(bool(phrases["head"])),
                    "full_distinct": float(
                        normalized_text(phrases["head"])
                        != normalized_text(phrases["full"])
                    ),
                    "reference_parsed": float(bool(phrases["reference"])),
                    **{f"head_{key}": value for key, value in head.items()},
                    **{f"full_{key}": value for key, value in full.items()},
                    **{f"reference_{key}": value for key, value in reference.items()},
                    "full_minus_head_score": full["max_score"] - head["max_score"],
                    "full_minus_head_iou": (
                        full["max_candidate_iou"] - head["max_candidate_iou"]
                    ),
                    "full_minus_head_joint_support": (
                        full["max_score_iou_product"]
                        - head["max_score_iou_product"]
                    ),
                    "full_minus_head_overlap_score_030": (
                        full["max_overlap_score_030"]
                        - head["max_overlap_score_030"]
                    ),
                    **{
                        f"reference_geometry_{key}": value
                        for key, value in reference_geometry.items()
                    },
                    **{
                        f"relation_{name}": float(relation == name)
                        for name, _ in RELATION_PATTERNS
                    },
                }
                if set(features) != set(INFERENCE_FEATURE_FIELDS):
                    raise ValueError("single-pass feature allowlist mismatch")
                record = {
                    "schema_version": "vsight_e3_singlepass_crossmodel_row_v1",
                    "model": model,
                    "task": task,
                    **meta,
                    "original_iou": float(source.get("iou") or 0.0),
                    "features": features,
                }
                records.append(record)
                stats["rows"] += 1
                stats[f"task:{task}"] += 1
                stats[f"model:{model}"] += 1
                stats["original_found"] += found

    expected = len(models) * len(TASKS) * len(metadata)
    if len(records) != expected:
        raise ValueError(f"expected {expected} cross-model rows, found {len(records)}")
    records.sort(key=lambda row: (row["model"], row["task"], row["sample_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in records:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": "vsight_e3_singlepass_crossmodel_manifest_v1",
        "status": "cross_validation_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "development_not_final_test",
        "inputs": {
            "benchmark": {"path": str(args.benchmark), "sha256": sha256(args.benchmark)},
            "proposal_summary": {
                "path": str(args.proposal_summary),
                "sha256": sha256(args.proposal_summary),
            },
            "record_root": str(args.record_root),
            "record_files": [
                {"path": str(path), "sha256": sha256(path)}
                for path in sorted(args.record_root.glob("*/records.jsonl"))
            ],
        },
        "models": models,
        "inference_feature_fields": list(INFERENCE_FEATURE_FIELDS),
        "forbidden_inference_fields": [
            "model",
            "label_exists",
            "hallucination_type",
            "gt_bbox_xyxy",
            "original_iou",
        ],
        "statistics": dict(sorted(stats.items())),
        "outputs": {
            "development": {
                "path": str(output.relative_to(ROOT)),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
        },
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
