#!/usr/bin/env python3
"""Build a supervision-free queue for the pairwise VLM verifier probe."""

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
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_e2_verifier as evaluator  # noqa: E402
from vsight.clip_verifier import read_selector_rows  # noqa: E402
from vsight.e1_data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e1/p1/selector/e1_p1_selector.summary.json",
    )
    parser.add_argument("--benchmark", type=Path, default=evaluator.DEFAULT_BENCHMARK)
    parser.add_argument(
        "--canonical-records", type=Path, default=evaluator.DEFAULT_RECORDS
    )
    parser.add_argument("--candidate-pattern", default=evaluator.DEFAULT_CANDIDATES)
    parser.add_argument("--image-root", type=Path, default=evaluator.DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/e1/p1/vlm_probe/e2_vlm_probe_queue.jsonl.gz",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def inference_row(row: dict, suite: str, image_root: Path) -> dict:
    return {
        "schema_version": "vsight_e2_vlm_probe_input_v1",
        "query_id": str(row["query_id"]),
        "suite": suite,
        "image_root": str(image_root.resolve()),
        "image_filename": str(row["image_filename"]),
        "query": str(row["query"]),
        "boxes_xyxy": [
            [float(value) for value in row["baseline_bbox_xyxy"]],
            [float(value) for value in row["challenger_bbox_xyxy"]],
        ],
    }


def main() -> int:
    args = parse_args()
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    if (args.output.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError("probe queue exists; pass --force to replace")
    selector = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    calibration_path = evaluator.resolve_path(
        selector["outputs"]["calibration"]["path"]
    )
    calibration = read_selector_rows(calibration_path)

    candidate_summary = json.loads(
        evaluator.resolve_path(selector["candidate_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    queue_summary = json.loads(
        evaluator.resolve_path(candidate_summary["queue_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source_summary = json.loads(
        evaluator.resolve_path(queue_summary["source_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    calibration_image_root = Path(source_summary["images"]["root"])
    dev = evaluator.load_dev_rows(args)

    rows = [
        inference_row(row, "calibration", calibration_image_root)
        for row in calibration
        if row.get("selector_eligible")
    ]
    for task, task_rows in dev.items():
        suite = "t2" if task == "t2_vqa_grounding" else "t4"
        rows.extend(
            inference_row(row, suite, args.image_root)
            for row in task_rows
            if row.get("selector_eligible")
        )
    rows.sort(key=lambda row: str(row["query_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    counts = {
        suite: sum(row["suite"] == suite for row in rows)
        for suite in ("calibration", "t2", "t4")
    }
    summary = {
        "schema_version": "vsight_e2_vlm_probe_queue_manifest_v1",
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(rows),
        "counts": counts,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "forbidden_fields_checked": [
            "gt_bbox_xyxy",
            "baseline_iou",
            "challenger_iou",
            "selector_action",
            "candidate_source",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
