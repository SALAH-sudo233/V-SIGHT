#!/usr/bin/env python3
"""Build auditable 7,500-record ROH-VCD results from frozen Qwen outputs.

The policy is deliberately state preserving: for every query it first keeps the
base Qwen existence decision, then replaces an emitted T2/T4 box (and only an
emitted box) with the binding-aware candidate generated for that exact query.
Candidate generation and selection are used symmetrically for positives and
all four negative types; no label, GT box, or evaluation metric is read by the
decision rule.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL_KEY = "ROH-VCD"
MODEL_NAME = "ROH-VCD (Qwen2.5-VL-7B + state-preserving binding-aware selection)"
BBOX_TASKS = {"t2_vqa_grounding", "t4_caption_grounding"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_many(pattern: str) -> Iterable[dict[str, Any]]:
    for name in sorted(glob.glob(pattern)):
        yield from read_jsonl(Path(name))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def load_candidates(pattern: str, role: str) -> dict[str, dict[str, Any]]:
    """Load exactly one valid binding-aware candidate per positive/group or negative/pair."""
    key_name = "base_sample_id" if role == "positive" else "pair_id"
    selected: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in read_many(pattern):
        if row.get("query_role") != role or row.get("candidate_source") != "binding_aware":
            continue
        if row.get("error"):
            continue
        key = str(row.get(key_name) or "")
        box = valid_box((row.get("candidate_boxes") or [None])[0])
        if not key or box is None:
            continue
        item = {"box": box, "query": str(row.get("query") or ""), "source_file": row.get("source_file")}
        if key in selected and selected[key]["box"] != item["box"]:
            duplicates.append(key)
        else:
            selected[key] = item
    if duplicates:
        raise ValueError(f"conflicting binding-aware candidates: {sorted(set(duplicates))[:5]}")
    return selected


def box_iou(box: list[float], gt: list[float]) -> float:
    x1 = max(box[0], gt[0])
    y1 = max(box[1], gt[1])
    x2 = min(box[2], gt[2])
    y2 = min(box[3], gt[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    gt_area = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    union = box_area + gt_area - intersection
    return intersection / union if union else 0.0


def build_records(
    baseline: list[dict[str, Any]],
    positive_candidates: dict[str, dict[str, Any]],
    negative_candidates: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    replacements: Counter[str] = Counter()
    for original in baseline:
        row = dict(original)
        row["model"] = MODEL_KEY
        row["inference_reused"] = True
        row["reuse_source"] = str(original.get("reuse_source") or "frozen_qwen_canonical_record")
        row["roh_vcd_policy"] = "state_preserving_binding_aware_v1"

        task = str(row.get("task") or "")
        role = str(row.get("query_role") or "")
        # Preserve T1 and every reject/missing-box state.  An emitted T2/T4
        # box is refined only with the exact-query binding-aware candidate.
        if task in BBOX_TASKS and bool(row.get("pred_found")):
            key = str(row.get("base_sample_id") or "") if role == "positive" else str(row.get("pair_id") or "")
            candidate = positive_candidates.get(key) if role == "positive" else negative_candidates.get(key)
            if candidate is not None:
                if candidate["query"] != str(row.get("query") or ""):
                    raise ValueError(f"query mismatch for {task}/{role}/{key}")
                row["pred_bbox_xyxy"] = candidate["box"]
                row["parse_valid"] = True
                if task == "t2_vqa_grounding":
                    row["parse_method"] = "roh_vcd_binding_aware"
                else:
                    row["bbox_parse_valid"] = True
                    row["bbox_parse_method"] = "roh_vcd_binding_aware"
                row["roh_vcd_candidate_source"] = "binding_aware"
                row["roh_vcd_candidate_query_role"] = role
                if role == "positive":
                    row["iou"] = box_iou(candidate["box"], [float(value) for value in row["gt_bbox_xyxy"]])
                else:
                    row["iou"] = None
                replacements[f"{task}:{role}"] += 1
            else:
                replacements[f"fallback_no_candidate:{task}:{role}"] += 1
        output.append(row)
    return output, replacements


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    fields = {"task"}
    for task, values in metrics.items():
        row: dict[str, Any] = {"task": task}
        for key, value in values.items():
            if isinstance(value, dict) and "value" in value:
                row[key] = value["value"]
                if "ci_lower" in value:
                    row[f"{key}_ci_lower"] = value["ci_lower"]
                    row[f"{key}_ci_upper"] = value["ci_upper"]
            elif not isinstance(value, (dict, list)):
                row[key] = value
        fields.update(row)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", *sorted(fields - {"task"})])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--positive-candidate-glob", required=True)
    parser.add_argument("--negative-candidate-glob", required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline_records)
    if len(baseline) != 7500:
        raise ValueError(f"expected 7500 canonical baseline records, got {len(baseline)}")
    if any("error" in row for row in baseline):
        raise ValueError("baseline contains inference error rows")

    positives = load_candidates(args.positive_candidate_glob, "positive")
    negatives = load_candidates(args.negative_candidate_glob, "negative")
    if len(positives) != 500 or len(negatives) != 2000:
        raise ValueError(f"candidate coverage must be 500/2000, got {len(positives)}/{len(negatives)}")

    records, replacements = build_records(baseline, positives, negatives)
    if len(records) != 7500:
        raise AssertionError("record count changed during policy application")

    if args.dry_run:
        print(json.dumps({"records": len(records), "positive_candidates": len(positives), "negative_candidates": len(negatives), "replacements": replacements}, indent=2))
        return

    sys.path.insert(0, str(args.evaluator_dir))
    evaluator = importlib.import_module("eval_11models_refcocog_500_run")
    metrics = evaluator.compute_metrics(records, args.iou_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.jsonl", records)
    write_metrics_csv(args.output_dir / "metrics.csv", metrics)
    summary = {
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "num_samples": 2000,
        "num_positive_groups": 500,
        "num_records": len(records),
        "metrics": metrics,
        "record_count_by_task": dict(Counter(str(row["task"]) for row in records)),
        "inference_reuse": {"reused": len(records), "generated": 0},
        "policy": {
            "name": "state_preserving_binding_aware_v1",
            "base_exists_decision": "frozen qwen canonical T1/T2/T4 decision",
            "bbox_action": "replace only emitted T2/T4 bbox with exact-query binding_aware candidate",
            "label_or_gt_used_at_decision_time": False,
        },
        "provenance": {
            "baseline_records": str(args.baseline_records),
            "baseline_sha256": sha256(args.baseline_records),
            "positive_candidate_glob": args.positive_candidate_glob,
            "negative_candidate_glob": args.negative_candidate_glob,
            "positive_candidate_count": len(positives),
            "negative_candidate_count": len(negatives),
            "replacements": dict(replacements),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "provenance.json", summary["provenance"])
    print(json.dumps({"output_dir": str(args.output_dir), "records": len(records), "replacements": replacements, "t1": metrics["t1_discriminative_vqa"], "t2": metrics["t2_vqa_grounding"], "t4": metrics["t4_caption_grounding"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
