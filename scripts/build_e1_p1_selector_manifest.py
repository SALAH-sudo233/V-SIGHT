#!/usr/bin/env python3
"""Join frozen P1 boxes to GT and build positive KEEP/SWITCH supervision."""

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
        "--candidate-summary",
        type=Path,
        default=ROOT / "data/e1/p1/candidates/e1_p1_candidates.summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e1/p1/selector"
    )
    parser.add_argument("--switch-iou-margin", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_gzip(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@contextmanager
def deterministic_gzip_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def write_row(handle: TextIO, row: dict) -> None:
    handle.write(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def main() -> int:
    args = parse_args()
    if args.switch_iou_margin < 0:
        raise ValueError("switch-iou-margin must be non-negative")
    candidate_summary = json.loads(
        args.candidate_summary.read_text(encoding="utf-8")
    )
    if candidate_summary.get("status") != "complete":
        raise ValueError("candidate generation summary is not complete")
    queue_summary_path = resolve_path(candidate_summary["queue_manifest"]["path"])
    if sha256(queue_summary_path) != candidate_summary["queue_manifest"]["sha256"]:
        raise ValueError("P1 queue manifest hash mismatch")
    queue_summary = json.loads(queue_summary_path.read_text(encoding="utf-8"))
    source_summary_path = resolve_path(queue_summary["source_manifest"]["path"])
    if sha256(source_summary_path) != queue_summary["source_manifest"]["sha256"]:
        raise ValueError("E1 source manifest hash mismatch")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))

    wanted: set[str] = set()
    for item in queue_summary["outputs"].values():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"P1 queue shard hash mismatch: {path}")
        wanted.update(str(row["query_id"]) for row in read_gzip(path))

    source_rows = {}
    for source in source_summary["sources"].values():
        for item in source["query_shards"].values():
            path = resolve_path(item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"E1 source shard hash mismatch: {path}")
            for row in read_gzip(path):
                query_id = str(row["query_id"])
                if query_id in wanted:
                    source_rows[query_id] = row
    if set(source_rows) != wanted:
        raise ValueError(f"P1 queries missing source supervision: {len(wanted-set(source_rows))}")

    latest = {}
    for item in candidate_summary["candidate_shards"]:
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"candidate shard hash mismatch: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[str(row["query_id"])] = row
    if set(latest) != wanted:
        raise ValueError("candidate outputs do not exactly cover the P1 queue")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: args.output_dir / f"e1_p1_selector.{split}.jsonl.gz"
        for split in ("train", "calibration")
    }
    summary_path = args.output_dir / "e1_p1_selector.summary.json"
    report_path = args.output_dir / "E1_P1_SELECTOR_REPORT.md"
    outputs = [*output_paths.values(), summary_path, report_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "outputs exist; pass --force to replace: " + ", ".join(existing)
        )

    stats = {
        split: {
            "records": 0,
            "actions": Counter(),
            "exclusions": Counter(),
            "baseline_iou_sum": 0.0,
            "challenger_iou_sum": 0.0,
            "oracle_iou_sum": 0.0,
            "baseline_zero": 0,
            "challenger_recovers_baseline_zero": 0,
        }
        for split in output_paths
    }
    temporaries = {
        split: path.with_name(path.name + ".tmp")
        for split, path in output_paths.items()
    }
    try:
        with deterministic_gzip_writer(temporaries["train"]) as train_handle, deterministic_gzip_writer(
            temporaries["calibration"]
        ) as calibration_handle:
            handles = {"train": train_handle, "calibration": calibration_handle}
            for query_id in sorted(wanted):
                source = source_rows[query_id]
                candidate = latest[query_id]
                split = str(source["data_split"])
                baseline = candidate.get("baseline") or {}
                challenger = candidate.get("challenger") or {}
                baseline_box = baseline.get("pred_bbox_xyxy")
                challenger_box = challenger.get("selected_bbox_xyxy")
                gt = [float(value) for value in source["gt_bbox_xyxy"]]
                baseline_iou = bbox_iou(baseline_box, gt) if baseline_box else 0.0
                challenger_iou = bbox_iou(challenger_box, gt) if challenger_box else 0.0

                action = None
                exclusion = None
                if candidate.get("error"):
                    exclusion = "inference_error"
                elif not baseline.get("parse_valid"):
                    exclusion = "baseline_parse_invalid"
                elif not baseline.get("pred_found"):
                    exclusion = "baseline_null_locked_stage1"
                elif not challenger.get("parse_valid") or challenger_box is None:
                    exclusion = "challenger_parse_invalid"
                elif challenger_iou - baseline_iou >= args.switch_iou_margin:
                    action = "switch"
                else:
                    action = "keep"
                stage1_oracle_iou = (
                    max(baseline_iou, challenger_iou)
                    if action is not None
                    else baseline_iou
                )

                record = {
                    "schema_version": "vsight_e1_p1_positive_selector_v1",
                    "query_id": query_id,
                    "group_id": source["group_id"],
                    "data_split": split,
                    "source_dataset": source["source_dataset"],
                    "image_id": source["image_id"],
                    "image_filename": source["image_filename"],
                    "image_width": source["image_width"],
                    "image_height": source["image_height"],
                    "query": source["query"],
                    "ann_id": source["ann_id"],
                    "category_id": source["category_id"],
                    "category_name": source["category_name"],
                    "gt_bbox_xyxy": gt,
                    "baseline_bbox_xyxy": baseline_box,
                    "challenger_bbox_xyxy": challenger_box,
                    "baseline_iou": baseline_iou,
                    "challenger_iou": challenger_iou,
                    "raw_two_box_oracle_iou": max(baseline_iou, challenger_iou),
                    "two_box_oracle_iou": stage1_oracle_iou,
                    "selector_action": action,
                    "selector_eligible": action is not None,
                    "selector_exclusion": exclusion,
                    "switch_iou_margin": args.switch_iou_margin,
                    "generator_spec_sha256": candidate["generator_spec_sha256"],
                }
                write_row(handles[split], record)
                current = stats[split]
                current["records"] += 1
                current["actions"][action or "excluded"] += 1
                if exclusion:
                    current["exclusions"][exclusion] += 1
                current["baseline_iou_sum"] += baseline_iou
                current["challenger_iou_sum"] += challenger_iou
                current["oracle_iou_sum"] += stage1_oracle_iou
                current["baseline_zero"] += baseline_iou == 0.0
                current["challenger_recovers_baseline_zero"] += (
                    baseline_iou == 0.0 and challenger_iou > 0.0
                )
        for split, temporary in temporaries.items():
            temporary.replace(output_paths[split])
    except BaseException:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
        raise

    rendered_stats = {}
    for split, current in stats.items():
        count = current["records"]
        rendered_stats[split] = {
            "records": count,
            "actions": dict(sorted(current["actions"].items())),
            "exclusions": dict(sorted(current["exclusions"].items())),
            "baseline_miou": current["baseline_iou_sum"] / count,
            "challenger_miou": current["challenger_iou_sum"] / count,
            "two_box_oracle_miou": current["oracle_iou_sum"] / count,
            "oracle_gain_over_baseline": (
                current["oracle_iou_sum"] - current["baseline_iou_sum"]
            )
            / count,
            "baseline_zero": current["baseline_zero"],
            "challenger_recovers_baseline_zero": current[
                "challenger_recovers_baseline_zero"
            ],
        }
    summary = {
        "schema_version": "vsight_e1_p1_selector_manifest_v1",
        "status": "positive_selector_supervision_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": {
            "path": manifest_path(args.candidate_summary),
            "sha256": sha256(args.candidate_summary),
        },
        "label_policy": {
            "switch_iou_margin": args.switch_iou_margin,
            "near_ties": "keep",
            "baseline_null": "excluded because stage 1 disallows recovery",
            "parse_invalid": "excluded",
        },
        "statistics": rendered_stats,
        "outputs": {
            split: {
                "path": manifest_path(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for split, path in output_paths.items()
        },
        "training_eligibility": {
            "e2_positive_selector": True,
            "e3_joint_null_verifier": False,
            "reason": "typed semantic null supervision is not built",
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                "# E1 P1 Positive Selector",
                "",
                "**Status:** positive KEEP/SWITCH supervision ready",
                "",
                *[
                    f"- {split}: baseline mIoU {item['baseline_miou']:.4f}; "
                    f"two-box oracle {item['two_box_oracle_miou']:.4f}; "
                    f"actions {item['actions']}"
                    for split, item in rendered_stats.items()
                ],
                "",
                "This manifest is valid for E2 only. E3 remains blocked on typed nulls.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
