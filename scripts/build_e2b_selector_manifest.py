#!/usr/bin/env python3
"""Join task-matched E2b candidates, relation proposals, and posterior labels."""

from __future__ import annotations

import argparse
import glob
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

from vsight.e1_data import bbox_iou, sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary", type=Path, default=ROOT / "data/e2b/queue/e2b_refcocog.summary.json"
    )
    parser.add_argument(
        "--source-summary", type=Path, default=ROOT / "data/e1/source/e1_source.summary.json"
    )
    parser.add_argument(
        "--p1-candidates",
        type=Path,
        default=ROOT / "data/e1/p1/candidates/e1_p1_candidates.summary.json",
    )
    parser.add_argument(
        "--t4-summary", type=Path, default=ROOT / "data/e2b/t4_candidates/e2b_t4.summary.json"
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=ROOT / "data/e2b/reference/e2b_reference_candidates.summary.json",
    )
    parser.add_argument(
        "--dino-summary",
        type=Path,
        default=ROOT / "data/e2b/reference_proposals/e2b_reference_dino.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e2b/selector")
    parser.add_argument("--switch-iou-margin", type=float, default=0.05)
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


def read_jsonl_shards(items: list[dict]) -> dict[str, dict]:
    latest = {}
    for item in items:
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[str(row["query_id"])] = row
    return latest


def main() -> int:
    args = parse_args()
    queue = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    p1 = json.loads(args.p1_candidates.read_text(encoding="utf-8"))
    t4_manifest = json.loads(args.t4_summary.read_text(encoding="utf-8"))
    reference_manifest = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    dino_manifest = json.loads(args.dino_summary.read_text(encoding="utf-8"))
    if t4_manifest["status"] != "complete" or dino_manifest["status"] != "complete":
        raise ValueError("E2b generated inputs are incomplete")

    wanted = {}
    for split, item in queue["outputs"].items():
        wanted[split] = {
            str(row["query_id"]): row for row in read_gzip(resolve_path(item["path"]))
        }
    query_ids = set().union(*(set(rows) for rows in wanted.values()))
    source_rows = {}
    for item in source["sources"]["refcocog_umd"]["query_shards"].values():
        for row in read_gzip(resolve_path(item["path"])):
            if str(row["query_id"]) in query_ids:
                source_rows[str(row["query_id"])] = row
    if set(source_rows) != query_ids:
        raise ValueError("E2b source join is incomplete")
    p1_rows = read_jsonl_shards(p1["candidate_shards"])
    t4_rows = read_jsonl_shards(t4_manifest["shards"])
    dino_rows = read_jsonl_shards(dino_manifest["shards"])
    reference_rows = {}
    for item in reference_manifest["outputs"].values():
        for row in read_gzip(resolve_path(item["path"])):
            reference_rows[str(row["query_id"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: args.output_dir / f"e2b_selector.{split}.jsonl.gz" for split in wanted
    }
    summary_path = args.output_dir / "e2b_selector.summary.json"
    if not args.force and any(path.exists() for path in (*outputs.values(), summary_path)):
        raise FileExistsError("E2b selector exists; pass --force to replace")
    stats = {split: Counter() for split in wanted}
    temporaries = {split: path.with_name(path.name + ".tmp") for split, path in outputs.items()}
    contexts = {split: deterministic_gzip(path) for split, path in temporaries.items()}
    handles = {split: context.__enter__() for split, context in contexts.items()}
    try:
        for split, rows in wanted.items():
            for query_id in sorted(rows):
                source_row = source_rows[query_id]
                gt = [float(value) for value in source_row["gt_bbox_xyxy"]]
                challenger = (p1_rows[query_id].get("challenger") or {}).get(
                    "selected_bbox_xyxy"
                )
                reference = reference_rows[query_id]
                dino = dino_rows.get(query_id) or {}
                proposals = list(dino.get("proposals") or [])
                reference_gt = (
                    reference["reference_candidates"][0]["bbox_xyxy"]
                    if reference["reference_status"] == "unique_reference_annotation"
                    else None
                )
                best_reference_index = None
                if reference_gt and proposals:
                    proposal_ious = [
                        bbox_iou(proposal["bbox_xyxy"], reference_gt) for proposal in proposals
                    ]
                    best = max(range(len(proposal_ious)), key=proposal_ious.__getitem__)
                    if proposal_ious[best] >= 0.5:
                        best_reference_index = best
                for task in ("t2", "t4"):
                    if task == "t2":
                        baseline = p1_rows[query_id].get("baseline") or {}
                        baseline_box = baseline.get("pred_bbox_xyxy")
                        parse_valid = bool(baseline.get("parse_valid"))
                        pred_found = bool(baseline.get("pred_found"))
                    else:
                        baseline = t4_rows[query_id].get("t4") or {}
                        baseline_box = baseline.get("pred_bbox_xyxy")
                        parse_valid = bool(baseline.get("parse_valid"))
                        pred_found = bool(baseline.get("pred_found"))
                    baseline_iou = bbox_iou(baseline_box, gt) if baseline_box else 0.0
                    challenger_iou = bbox_iou(challenger, gt) if challenger else 0.0
                    exclusion = None
                    if not parse_valid:
                        exclusion = "baseline_parse_invalid"
                    elif not pred_found or baseline_box is None:
                        exclusion = "baseline_null_locked_stage1"
                    elif challenger is None:
                        exclusion = "challenger_parse_invalid"
                    action = None
                    if exclusion is None:
                        action = (
                            "switch"
                            if challenger_iou - baseline_iou >= args.switch_iou_margin
                            else "keep"
                        )
                    relation_eligible = (
                        action is not None
                        and bool(reference.get("relation"))
                        and bool(reference.get("reference_phrase"))
                        and bool(proposals)
                    )
                    output = {
                        "schema_version": "vsight_e2b_relation_selector_v1",
                        "query_id": f"{task}:{query_id}",
                        "source_query_id": query_id,
                        "group_id": source_row["group_id"],
                        "data_split": split,
                        "task": task,
                        "image_id": source_row["image_id"],
                        "image_filename": source_row["image_filename"],
                        "image_width": source_row["image_width"],
                        "image_height": source_row["image_height"],
                        "query": source_row["query"],
                        "gt_bbox_xyxy": gt,
                        "baseline_bbox_xyxy": baseline_box,
                        "challenger_bbox_xyxy": challenger,
                        "baseline_iou": baseline_iou,
                        "challenger_iou": challenger_iou,
                        "selector_action": action,
                        "selector_eligible": action is not None,
                        "selector_exclusion": exclusion,
                        "relation": reference.get("relation"),
                        "reference_phrase": reference.get("reference_phrase"),
                        "reference_proposals": proposals,
                        "reference_best_index": best_reference_index,
                        "relation_selector_eligible": relation_eligible,
                    }
                    handles[split].write(
                        json.dumps(
                            output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
                    stats[split][f"{task}:{action or 'excluded'}"] += 1
                    stats[split][f"{task}:relation_eligible"] += relation_eligible
                    stats[split][f"{task}:reference_supervised"] += (
                        relation_eligible and best_reference_index is not None
                    )
    finally:
        for context in contexts.values():
            context.__exit__(None, None, None)
    for split, temporary in temporaries.items():
        temporary.replace(outputs[split])
    summary = {
        "schema_version": "vsight_e2b_relation_selector_manifest_v1",
        "status": "training_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statistics": {split: dict(sorted(values.items())) for split, values in stats.items()},
        "outputs": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for split, path in outputs.items()
        },
        "model_forbidden_inputs": [
            "task",
            "gt_bbox_xyxy",
            "baseline_iou",
            "challenger_iou",
            "selector_action",
            "reference_best_index",
        ],
        "external_api_required": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
