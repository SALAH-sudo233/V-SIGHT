#!/usr/bin/env python3
"""Analyze positive IoU=0 failures before and after candidate replacement."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def xywh_to_xyxy(box: list[float]) -> list[float]:
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def record_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task"]), str(row["base_sample_id"])


def normalized_iou(row: dict[str, Any]) -> float:
    value = row.get("iou")
    return float(value) if value is not None else 0.0


def state(row: dict[str, Any]) -> str:
    if not bool(row.get("pred_found")):
        return "false_rejection"
    return "valid_zero" if normalized_iou(row) == 0.0 else "overlap"


def transition(baseline: dict[str, Any], result: dict[str, Any]) -> str:
    before, after = state(baseline), state(result)
    if before == "false_rejection":
        return "false_rejection_unchanged"
    if before == "valid_zero" and after == "overlap":
        return "valid_zero_recovered"
    if before == "valid_zero" and after == "valid_zero":
        return "valid_zero_unresolved"
    if before == "overlap" and after == "valid_zero":
        return "nonzero_regressed_to_zero"
    if before == "overlap" and after == "overlap":
        return "nonzero_remained_nonzero"
    return f"{before}_to_{after}"


def image_id_from_name(name: str) -> int:
    match = re.search(r"(\d{12})", name)
    if not match:
        raise ValueError(f"cannot parse COCO image id from {name!r}")
    return int(match.group(1))


def expression_structure(group: dict[str, Any]) -> str:
    annotation = group.get("chair_annotation") or {}
    units = annotation.get("target_eval_units") or {}
    if units.get("positive_relations"):
        return "relation"
    if units.get("positive_attributes"):
        return "attribute"
    return "object_only"


def best_annotation(box: list[float] | None, annotations: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if not box:
        return None, 0.0
    best, best_iou = None, 0.0
    for annotation in annotations:
        value = iou(box, annotation["box_xyxy"])
        if value > best_iou:
            best, best_iou = annotation, value
    return best, best_iou


def zero_box_class(
    pred_box: list[float] | None,
    annotations: list[dict[str, Any]],
    target_annotation: dict[str, Any] | None,
    target_category_id: int | None,
) -> tuple[str, str, float]:
    match, match_iou = best_annotation(pred_box, annotations)
    if match is None or match_iou < 0.1:
        return "background_or_unannotated", "", match_iou
    if match_iou < 0.5:
        return "partial_or_oversized_region", str(match.get("category_name", "")), match_iou
    if target_annotation is not None and match["id"] == target_annotation["id"]:
        return "target_annotation_coordinate_mismatch", str(match.get("category_name", "")), match_iou
    if target_category_id is not None and match["category_id"] == target_category_id:
        return "wrong_same_category_instance", str(match.get("category_name", "")), match_iou
    return "other_category_or_reference", str(match.get("category_name", "")), match_iou


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task in ("t2_vqa_grounding", "t4_caption_grounding"):
        task_rows = [row for row in rows if row["task"] == task]
        before = [float(row["baseline_iou"]) for row in task_rows]
        after = [float(row["result_iou"]) for row in task_rows]
        top2_oracle = [max(row["baseline_iou"], row["result_iou"]) for row in task_rows]
        baseline_valid_zero = [row for row in task_rows if row["baseline_state"] == "valid_zero"]
        transition_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            transition_groups[row["transition"]].append(row)
        class_recovery: dict[str, dict[str, Any]] = {}
        for class_name in sorted({row["baseline_zero_box_class"] for row in baseline_valid_zero}):
            class_rows = [row for row in baseline_valid_zero if row["baseline_zero_box_class"] == class_name]
            recovered = sum(row["transition"] == "valid_zero_recovered" for row in class_rows)
            class_recovery[class_name] = {
                "n": len(class_rows),
                "recovered": recovered,
                "recovery_rate": recovered / len(class_rows),
            }
        output[task] = {
            "n": len(task_rows),
            "baseline_miou": sum(before) / len(before),
            "result_miou": sum(after) / len(after),
            "baseline_vs_candidate_top2_oracle": {
                "miou": sum(top2_oracle) / len(top2_oracle),
                "iou_zero": sum(value == 0.0 for value in top2_oracle),
                "acc_at_0_5": sum(value >= 0.5 for value in top2_oracle) / len(top2_oracle),
                "baseline_better": sum(row["iou_delta"] < -1e-9 for row in task_rows),
                "candidate_better": sum(row["iou_delta"] > 1e-9 for row in task_rows),
                "tie": sum(abs(row["iou_delta"]) <= 1e-9 for row in task_rows),
            },
            "baseline_iou_zero": sum(value == 0.0 for value in before),
            "result_iou_zero": sum(value == 0.0 for value in after),
            "baseline_false_rejection": sum(row["baseline_state"] == "false_rejection" for row in task_rows),
            "baseline_valid_box_iou_zero": len(baseline_valid_zero),
            "transitions": dict(Counter(row["transition"] for row in task_rows)),
            "baseline_valid_zero_coco_classes": dict(
                Counter(row["baseline_zero_box_class"] for row in baseline_valid_zero)
            ),
            "baseline_valid_zero_expression_structure": dict(
                Counter(row["expression_structure"] for row in baseline_valid_zero)
            ),
            "baseline_valid_zero_recovery_by_coco_class": class_recovery,
            "result_valid_zero_coco_classes": dict(
                Counter(
                    row["result_zero_box_class"]
                    for row in task_rows
                    if row["result_state"] == "valid_zero"
                )
            ),
            "recovered_to_iou_0_5": sum(
                row["baseline_iou"] == 0.0 and row["result_iou"] >= 0.5 for row in task_rows
            ),
            "regressed_from_iou_0_5": sum(
                row["baseline_iou"] >= 0.5 and row["result_iou"] < 0.5 for row in task_rows
            ),
            "mean_iou_delta": sum(row["iou_delta"] for row in task_rows) / len(task_rows),
            "iou_delta_direction": {
                "improved": sum(row["iou_delta"] > 1e-9 for row in task_rows),
                "unchanged": sum(abs(row["iou_delta"]) <= 1e-9 for row in task_rows),
                "degraded": sum(row["iou_delta"] < -1e-9 for row in task_rows),
            },
            "iou_0_5_transition": {
                "below_to_at_least_0_5": sum(
                    row["baseline_iou"] < 0.5 <= row["result_iou"] for row in task_rows
                ),
                "at_least_0_5_to_below": sum(
                    row["result_iou"] < 0.5 <= row["baseline_iou"] for row in task_rows
                ),
            },
            "transition_context": {
                name: {
                    "n": len(group_rows),
                    "median_gt_area_ratio": statistics.median(row["gt_area_ratio"] for row in group_rows),
                    "mean_same_category_distractors": sum(
                        row["same_category_distractors"] for row in group_rows
                    ) / len(group_rows),
                    "mean_iou_delta": sum(row["iou_delta"] for row in group_rows) / len(group_rows),
                }
                for name, group_rows in transition_groups.items()
            },
        }
    by_task = {
        task: {row["base_sample_id"]: row for row in rows if row["task"] == task}
        for task in ("t2_vqa_grounding", "t4_caption_grounding")
    }
    t2, t4 = by_task["t2_vqa_grounding"], by_task["t4_caption_grounding"]
    output["cross_task"] = {
        "baseline_iou_zero_intersection": sum(
            t2[group_id]["baseline_iou"] == 0.0 and t4[group_id]["baseline_iou"] == 0.0
            for group_id in t2
        ),
        "baseline_valid_zero_intersection": sum(
            t2[group_id]["baseline_state"] == "valid_zero"
            and t4[group_id]["baseline_state"] == "valid_zero"
            for group_id in t2
        ),
        "result_iou_zero_intersection": sum(
            t2[group_id]["result_iou"] == 0.0 and t4[group_id]["result_iou"] == 0.0
            for group_id in t2
        ),
        "recovered_in_both": sum(
            t2[group_id]["transition"] == "valid_zero_recovered"
            and t4[group_id]["transition"] == "valid_zero_recovered"
            for group_id in t2
        ),
        "same_final_box_when_both_found": sum(
            t2[group_id]["result_state"] != "false_rejection"
            and t4[group_id]["result_state"] != "false_rejection"
            and t2[group_id]["result_box"] == t4[group_id]["result_box"]
            for group_id in t2
        ),
        "both_found": sum(
            t2[group_id]["result_state"] != "false_rejection"
            and t4[group_id]["result_state"] != "false_rejection"
            for group_id in t2
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/qwen2.5-vl-7b.canonical_records.jsonl")
    parser.add_argument("--result", type=Path, default=ROOT / "data/roh_vcd_state_preserving_records.jsonl")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/refcocog_500_dev.semantic_strict.json")
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        default=Path("/home/u2025141034/CHAIR/coco/annotations/instances_train2014.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/zero_iou")
    args = parser.parse_args()

    benchmark_rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    for row in benchmark_rows:
        groups.setdefault(str(row["base_sample_id"]), row)
    image_ids = {image_id_from_name(str(group["image_filename"])) for group in groups.values()}

    coco = json.loads(args.coco_annotations.read_text(encoding="utf-8"))
    category_names = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    image_shapes = {
        int(item["id"]): (int(item["width"]), int(item["height"]))
        for item in coco["images"]
        if int(item["id"]) in image_ids
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in coco["annotations"]:
        image_id = int(item["image_id"])
        if image_id not in image_ids or item.get("iscrowd"):
            continue
        annotations_by_image[image_id].append(
            {
                "id": int(item["id"]),
                "category_id": int(item["category_id"]),
                "category_name": category_names[int(item["category_id"])],
                "box_xyxy": xywh_to_xyxy([float(value) for value in item["bbox"]]),
            }
        )

    baseline = {
        record_key(row): row
        for row in read_jsonl(args.baseline)
        if row.get("query_role") == "positive" and row.get("task") in {"t2_vqa_grounding", "t4_caption_grounding"}
    }
    result = {
        record_key(row): row
        for row in read_jsonl(args.result)
        if row.get("query_role") == "positive" and row.get("task") in {"t2_vqa_grounding", "t4_caption_grounding"}
    }
    if baseline.keys() != result.keys() or len(baseline) != 1000:
        raise AssertionError("expected 1,000 aligned T2/T4 positive records")

    rows: list[dict[str, Any]] = []
    for record_id, baseline_row in sorted(baseline.items()):
        result_row = result[record_id]
        group_id = str(baseline_row["base_sample_id"])
        group = groups[group_id]
        image_name = str(group["image_filename"])
        image_id = image_id_from_name(image_name)
        annotations = annotations_by_image[image_id]
        gt = [float(value) for value in baseline_row["gt_bbox_xyxy"]]
        target_annotation, target_match_iou = best_annotation(gt, annotations)
        target_category_id = target_annotation["category_id"] if target_annotation else None
        same_category_distractors = sum(
            item["category_id"] == target_category_id and item is not target_annotation
            for item in annotations
        )

        baseline_class, baseline_match_category, baseline_match_iou = ("", "", 0.0)
        if state(baseline_row) == "valid_zero":
            baseline_class, baseline_match_category, baseline_match_iou = zero_box_class(
                baseline_row.get("pred_bbox_xyxy"), annotations, target_annotation, target_category_id
            )
        result_class, result_match_category, result_match_iou = ("", "", 0.0)
        if state(result_row) == "valid_zero":
            result_class, result_match_category, result_match_iou = zero_box_class(
                result_row.get("pred_bbox_xyxy"), annotations, target_annotation, target_category_id
            )

        width, height = image_shapes[image_id]
        gt_area_ratio = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (width * height)
        before_iou, after_iou = normalized_iou(baseline_row), normalized_iou(result_row)
        rows.append(
            {
                "task": record_id[0],
                "base_sample_id": group_id,
                "image_filename": image_name,
                "query": baseline_row["query"],
                "expression_structure": expression_structure(group),
                "target_category": target_annotation["category_name"] if target_annotation else "",
                "target_coco_match_iou": target_match_iou,
                "same_category_distractors": same_category_distractors,
                "gt_area_ratio": gt_area_ratio,
                "baseline_state": state(baseline_row),
                "result_state": state(result_row),
                "transition": transition(baseline_row, result_row),
                "baseline_iou": before_iou,
                "result_iou": after_iou,
                "iou_delta": after_iou - before_iou,
                "baseline_box": json.dumps(baseline_row.get("pred_bbox_xyxy"), separators=(",", ":")),
                "result_box": json.dumps(result_row.get("pred_bbox_xyxy"), separators=(",", ":")),
                "gt_box": json.dumps(gt, separators=(",", ":")),
                "candidate_applied": result_row.get("roh_vcd_candidate_source") == "binding_aware",
                "baseline_zero_box_class": baseline_class,
                "baseline_zero_match_category": baseline_match_category,
                "baseline_zero_match_iou": baseline_match_iou,
                "result_zero_box_class": result_class,
                "result_zero_match_category": result_match_category,
                "result_zero_match_iou": result_match_iou,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
