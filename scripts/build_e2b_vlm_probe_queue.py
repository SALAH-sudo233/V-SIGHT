#!/usr/bin/env python3
"""Build a supervision-free E2b calibration queue for the local Qwen verifier."""

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

DEFAULT_SELECTOR = ROOT / "data/e2b/selector/e2b_selector.summary.json"
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")
DEFAULT_OUTPUT = ROOT / "data/e2b/vlm_probe/e2b_vlm_probe_queue.jsonl.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-summary", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def main() -> int:
    args = parse_args()
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    if not args.force and (args.output.exists() or summary_path.exists()):
        raise FileExistsError("E2b VLM queue exists; pass --force to replace")
    selector = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    calibration_item = selector["outputs"]["calibration"]
    calibration_path = resolve_path(calibration_item["path"])
    if sha256(calibration_path) != calibration_item["sha256"]:
        raise ValueError("E2b calibration hash mismatch")
    rows = []
    for source in read_gzip(calibration_path):
        if not source.get("relation_selector_eligible"):
            continue
        row = {
            "schema_version": "vsight_e2b_vlm_probe_input_v1",
            "query_id": str(source["query_id"]),
            "suite": str(source["task"]),
            "image_root": str(args.image_root.resolve()),
            "image_filename": str(source["image_filename"]),
            "query": str(source["query"]),
            "relation": str(source["relation"]),
            "reference_phrase": str(source["reference_phrase"]),
            "reference_proposals": source["reference_proposals"],
            "boxes_xyxy": [
                [float(value) for value in source["baseline_bbox_xyxy"]],
                [float(value) for value in source["challenger_bbox_xyxy"]],
            ],
        }
        forbidden = {
            "gt_bbox_xyxy",
            "baseline_iou",
            "challenger_iou",
            "selector_action",
            "reference_best_index",
        }
        if forbidden & set(row):
            raise ValueError("E2b VLM queue leaks supervision")
        rows.append(row)
    rows.sort(key=lambda item: str(item["query_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {
        "schema_version": "vsight_e2b_vlm_probe_queue_manifest_v1",
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selector": {"path": str(args.selector_summary), "sha256": sha256(args.selector_summary)},
        "records": len(rows),
        "counts": {task: sum(row["suite"] == task for row in rows) for task in ("t2", "t4")},
        "output": {
            "path": str(args.output.relative_to(ROOT)),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
