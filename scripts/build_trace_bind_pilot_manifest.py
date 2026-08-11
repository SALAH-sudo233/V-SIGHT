#!/usr/bin/env python3
"""Freeze the matched 100-query TRACE-Bind development manifest.

The query/image row set is copied in its existing order from the completed
CABLE-RAFT VLM2 pilot.  One already-generated upstream grounding box is joined
per query using a deterministic model order.  Labels, GT, and source model IDs
are not written to inference rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "outputs/cable_raft_attention_vlm2_pilot/evidence.jsonl"
DEFAULT_BENCHMARK = ROOT / "legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json"
DEFAULT_RECORD_ROOT = Path(
    "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/"
    "run_500_semantic_strict"
)
DEFAULT_OUTPUT = ROOT / "data/e3/trace_bind_pilot/trace_bind_pilot_queue.jsonl.gz"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--upstream-task", default="t2_vqa_grounding")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def base_sample_id(value: object) -> str:
    return str(value or "").split("__", 1)[0]


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
    evidence_rows = read_jsonl(args.evidence)
    if len(evidence_rows) != 100:
        raise ValueError(
            f"expected the frozen 100-query evidence row set, found {len(evidence_rows)}"
        )
    evidence_keys = []
    for row in evidence_rows:
        key = (str(row["image_filename"]), normalized(row["query"]))
        if key in evidence_keys:
            raise ValueError(f"duplicate image/query row in frozen evidence: {key}")
        evidence_keys.append(key)

    benchmark_rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    image_by_sample = {
        base_sample_id(row.get("base_sample_id") or row.get("sample_id")): str(
            row["image_filename"]
        )
        for row in benchmark_rows
    }
    boxes: dict[tuple[str, str], tuple[list[float], str]] = {}
    source_hashes = {}
    for record_path in sorted(args.record_root.glob("*/records.jsonl")):
        model = record_path.parent.name
        source_hashes[model] = hashlib.sha256(record_path.read_bytes()).hexdigest()
        with record_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("task")) != args.upstream_task or not row.get("pred_found"):
                    continue
                raw_box = row.get("pred_bbox_xyxy")
                image = image_by_sample.get(base_sample_id(row.get("sample_id")))
                if image is None or not isinstance(raw_box, list) or len(raw_box) != 4:
                    continue
                key = (image, normalized(row.get("query")))
                boxes.setdefault(key, ([float(value) for value in raw_box], model))
    missing = [key for key in evidence_keys if key not in boxes]
    if missing:
        raise ValueError(f"upstream record streams are missing {len(missing)} frozen rows")

    output_rows = []
    source_counts = Counter()
    for row, key in zip(evidence_rows, evidence_keys, strict=True):
        upstream_box, source_model = boxes[key]
        source_counts[source_model] += 1
        output_rows.append(
            {
                "schema_version": "vsight_trace_bind_pilot_input_v1",
                "record_id": str(row["record_id"]),
                "image_group_id": str(row["image_filename"]),
                "image_filename": str(row["image_filename"]),
                "query": str(row["query"]),
                "upstream_box_xyxy": upstream_box,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with deterministic_gzip(args.output) as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "vsight_trace_bind_pilot_manifest_v1",
        "records": len(output_rows),
        "image_groups": len({row["image_filename"] for row in output_rows}),
        "source_evidence": str(args.evidence),
        "source_evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "benchmark_metadata_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "upstream_task": args.upstream_task,
        "upstream_selection": "first_valid_box_in_sorted_model_order",
        "upstream_source_counts": dict(sorted(source_counts.items())),
        "upstream_record_stream_sha256": source_hashes,
        "labels_in_queue": False,
        "gt_in_queue": False,
        "sealed_heldout_accessed": False,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
