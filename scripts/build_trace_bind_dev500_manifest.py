#!/usr/bin/env python3
"""Freeze the 500-group relation development manifest and label sidecar."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import bbox_iou, sha256  # noqa: E402


DEFAULT_BENCHMARK = (
    ROOT / "legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json"
)
DEFAULT_RECORDS = Path(
    "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/"
    "run_500_semantic_strict/LENS/records.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/e3/trace_bind_dev500"
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/benchmark/benchmark_images")
FORBIDDEN_QUEUE_FIELDS = {
    "binding_label",
    "gt_bbox_xyxy",
    "hallucination_type",
    "iou",
    "label",
    "model",
    "model_id",
    "source_model",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--upstream-model", default="LENS")
    parser.add_argument("--upstream-task", default="t2_vqa_grounding")
    parser.add_argument("--label-iou-threshold", type=float, default=0.5)
    parser.add_argument("--expected-records", type=int, default=500)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def base_sample_id(value: object) -> str:
    return str(value or "").split("__", 1)[0]


def valid_box(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in box) and box[2] > box[0] and box[3] > box[1]


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with deterministic_gzip(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    args = arguments()
    if args.expected_records <= 0:
        raise ValueError("--expected-records must be positive")
    if not 0.0 <= args.label_iou_threshold <= 1.0:
        raise ValueError("--label-iou-threshold must be in [0, 1]")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    queue_path = output_dir / "trace_bind_dev500_queue.jsonl.gz"
    labels_path = output_dir / "trace_bind_dev500_labels.jsonl.gz"
    summary_path = queue_path.with_suffix(queue_path.suffix + ".summary.json")
    if not args.force and any(path.exists() for path in (queue_path, labels_path, summary_path)):
        raise FileExistsError("dev500 artifact exists; pass --force")

    benchmark_rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark_by_group: dict[str, dict] = {}
    for row in benchmark_rows:
        group_id = base_sample_id(row.get("base_sample_id") or row.get("sample_id"))
        if not group_id:
            raise ValueError("benchmark row has no base sample ID")
        existing = benchmark_by_group.setdefault(group_id, row)
        if (
            normalized(existing.get("positive_text")) != normalized(row.get("positive_text"))
            or str(existing.get("image_filename")) != str(row.get("image_filename"))
        ):
            raise ValueError(f"inconsistent benchmark group: {group_id}")

    inference_rows = []
    label_sources = []
    seen_groups = set()
    with args.records.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("task")) != args.upstream_task:
                continue
            if str(row.get("query_role") or "positive") != "positive":
                continue
            if str(row.get("model")) != args.upstream_model:
                raise ValueError("upstream record stream contains an unexpected model")
            predicted_box = row.get("pred_bbox_xyxy")
            if row.get("pred_found") and not valid_box(predicted_box):
                raise ValueError("selected upstream row marks an invalid box as found")
            if not row.get("pred_found") and predicted_box is not None:
                raise ValueError("selected upstream row has a box despite pred_found=false")
            group_id = base_sample_id(row.get("base_sample_id") or row.get("sample_id"))
            if group_id in seen_groups:
                raise ValueError(f"duplicate upstream development group: {group_id}")
            benchmark = benchmark_by_group.get(group_id)
            if benchmark is None:
                raise ValueError(f"upstream group is missing from benchmark: {group_id}")
            if normalized(row.get("query")) != normalized(benchmark.get("positive_text")):
                raise ValueError(f"query mismatch for development group: {group_id}")
            if not valid_box(benchmark.get("gt_bbox_xyxy")):
                raise ValueError(f"benchmark group has no valid GT box: {group_id}")
            seen_groups.add(group_id)
            record_id = f"trace-dev500:{group_id}"
            inference = {
                "schema_version": "vsight_trace_bind_dev500_input_v1",
                "record_id": record_id,
                "image_group_id": group_id,
                "image_filename": str(benchmark["image_filename"]),
                "query": str(row["query"]),
                "upstream_box_xyxy": (
                    [float(value) for value in predicted_box]
                    if predicted_box is not None else None
                ),
            }
            leaked = FORBIDDEN_QUEUE_FIELDS & set(inference)
            if leaked:
                raise ValueError(f"inference queue leaks supervision: {sorted(leaked)}")
            inference_rows.append(inference)
            label_sources.append((record_id, group_id, row, benchmark))

    if len(inference_rows) != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} development rows, found {len(inference_rows)}"
        )
    if set(benchmark_by_group) != seen_groups:
        raise ValueError("upstream stream does not cover the complete development benchmark")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(queue_path, inference_rows)
    queue_sha256 = sha256(queue_path)

    label_rows = []
    label_counts = Counter()
    for record_id, group_id, upstream, benchmark in label_sources:
        overlap = (
            bbox_iou(upstream["pred_bbox_xyxy"], benchmark["gt_bbox_xyxy"])
            if upstream.get("pred_bbox_xyxy") is not None else 0.0
        )
        label = "supported" if overlap >= args.label_iou_threshold else "contradicted"
        label_counts[label] += 1
        label_rows.append(
            {
                "schema_version": "vsight_trace_bind_dev500_proxy_label_v1",
                "record_id": record_id,
                "image_group_id": group_id,
                "query_stratum": "relation",
                "binding_label": label,
                "upstream_iou": overlap,
                "label_semantics": "full_query_grounding_iou_proxy",
                "label_iou_threshold": args.label_iou_threshold,
                "source_queue_sha256": queue_sha256,
            }
        )
    write_jsonl_gz(labels_path, label_rows)

    image_root = args.image_root
    missing_images = [
        row["image_filename"] for row in inference_rows if not (image_root / row["image_filename"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"development image root is missing {len(missing_images)} images")
    summary = {
        "schema_version": "vsight_trace_bind_dev500_manifest_v1",
        "dataset_role": "development_relation_proxy_not_final_test",
        "records": len(inference_rows),
        "image_groups": len(seen_groups),
        "query_strata": {"relation": len(inference_rows)},
        "source_benchmark": str(args.benchmark),
        "source_benchmark_sha256": sha256(args.benchmark),
        "source_records": str(args.records),
        "source_records_sha256": sha256(args.records),
        "upstream_model": args.upstream_model,
        "upstream_task": args.upstream_task,
        "selection_rule": "all_positive_task_rows_in_source_order",
        "upstream_box_records": sum(
            row["upstream_box_xyxy"] is not None for row in inference_rows
        ),
        "upstream_null_records": sum(
            row["upstream_box_xyxy"] is None for row in inference_rows
        ),
        "queue": str(queue_path),
        "queue_sha256": queue_sha256,
        "labels": str(labels_path),
        "labels_sha256": sha256(labels_path),
        "label_semantics": "full_query_grounding_iou_proxy_not_double_reviewed_binding",
        "label_iou_threshold": args.label_iou_threshold,
        "label_counts": dict(sorted(label_counts.items())),
        "labels_in_queue": False,
        "gt_in_queue": False,
        "model_id_in_queue": False,
        "image_root": str(image_root),
        "missing_images": 0,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
