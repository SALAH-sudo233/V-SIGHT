#!/usr/bin/env python3
"""Evaluate an E2 verifier on calibration and repaired-500 development data."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import bbox_iou, sha256  # noqa: E402
from vsight.e2_verifier import selector_metrics, threshold_candidates  # noqa: E402


DEFAULT_BENCHMARK = ROOT / "legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json"
DEFAULT_RECORDS = ROOT / "legacy/candidate_pool_v1/data/qwen2.5-vl-7b.canonical_records.jsonl"
DEFAULT_CANDIDATES = str(
    ROOT / "legacy/candidate_pool_v1/data/binding_aware_positive_candidates/shard_*.jsonl"
)
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/benchmark/benchmark_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e1/p1/selector/e1_p1_selector.summary.json",
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--canonical-records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--candidate-pattern", default=DEFAULT_CANDIDATES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e2_verifier")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--t2-regression-budget", type=int, default=8)
    parser.add_argument("--t4-regression-budget", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized_query(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def base_id(row: dict) -> str:
    value = str(row.get("base_sample_id") or row.get("sample_id") or "")
    return value.split("__", 1)[0]


def load_dev_rows(args: argparse.Namespace) -> dict[str, list[dict]]:
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    positives = {}
    for row in benchmark:
        key = base_id(row)
        query = str(row.get("chosen") or row.get("positive_text") or "").strip()
        current = {
            "base_sample_id": key,
            "image_filename": str(row["image_filename"]),
            "query": query,
            "gt_bbox_xyxy": row.get("gt_bbox_xyxy") or row.get("positive_bbox"),
        }
        if key in positives:
            for field in ("image_filename", "query", "gt_bbox_xyxy"):
                if positives[key][field] != current[field]:
                    raise ValueError(f"inconsistent repaired-500 positive: {key} {field}")
        else:
            positives[key] = current

    challengers = {}
    for value in sorted(glob.glob(args.candidate_pattern)):
        for row in read_jsonl(Path(value)):
            key = base_id(row)
            if row.get("error") or not row.get("parse_valid"):
                challengers[key] = None
                continue
            boxes = row.get("candidate_boxes") or []
            challengers[key] = [float(item) for item in boxes[0]] if boxes else None

    baselines = {"t2_vqa_grounding": {}, "t4_caption_grounding": {}}
    for row in read_jsonl(args.canonical_records):
        task = str(row.get("task"))
        if task not in baselines or row.get("query_role") != "positive":
            continue
        key = base_id(row)
        if key in positives and normalized_query(row.get("query")) == normalized_query(
            positives[key]["query"]
        ):
            baselines[task][key] = row

    output = {}
    for task, task_baselines in baselines.items():
        rows = []
        for key in sorted(positives):
            positive = positives[key]
            baseline = task_baselines.get(key) or {}
            baseline_box = baseline.get("pred_bbox_xyxy")
            challenger_box = challengers.get(key)
            gt = [float(value) for value in positive["gt_bbox_xyxy"]]
            baseline_iou = bbox_iou(baseline_box, gt) if baseline_box else 0.0
            challenger_iou = bbox_iou(challenger_box, gt) if challenger_box else 0.0
            parse_valid = bool(
                baseline.get("parse_valid", baseline.get("bbox_parse_valid", False))
            )
            exclusion = None
            if not parse_valid:
                exclusion = "baseline_parse_invalid"
            elif not baseline.get("pred_found") or baseline_box is None:
                exclusion = "baseline_null_locked_stage1"
            elif challenger_box is None:
                exclusion = "challenger_parse_invalid"
            action = None
            if exclusion is None:
                action = "switch" if challenger_iou - baseline_iou >= 0.05 else "keep"
            rows.append(
                {
                    "schema_version": "vsight_e2_repaired500_eval_v1",
                    "query_id": f"{task}:{key}",
                    "group_id": key,
                    "image_filename": positive["image_filename"],
                    "query": positive["query"],
                    "gt_bbox_xyxy": gt,
                    "baseline_bbox_xyxy": baseline_box,
                    "challenger_bbox_xyxy": challenger_box,
                    "baseline_iou": baseline_iou,
                    "challenger_iou": challenger_iou,
                    "selector_action": action,
                    "selector_eligible": action is not None,
                    "selector_exclusion": exclusion,
                }
            )
        output[task] = rows
    if any(len(rows) != 500 for rows in output.values()):
        raise ValueError(
            "repaired-500 join must produce 500 rows per task: "
            + str({task: len(rows) for task, rows in output.items()})
        )
    return output


def score_rows(model, processor, rows, image_root, device, args) -> dict[str, dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader

    from vsight.clip_verifier import E2BatchCollator, E2SelectorDataset

    dataset = E2SelectorDataset(rows, image_root, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=E2BatchCollator(processor),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    differences = {"action": {}, "quality": {}}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: batch[key].to(device, non_blocking=True)
                for key in ("pixel_values", "input_ids", "attention_mask", "geometry")
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores, quality_logits = model(**inputs)
            action_values = (scores[:, 1] - scores[:, 0]).float().cpu().tolist()
            quality_values = (
                torch.sigmoid(quality_logits[:, 1])
                - torch.sigmoid(quality_logits[:, 0])
            ).float().cpu().tolist()
            for query_id, action, quality in zip(
                batch["query_ids"], action_values, quality_values, strict=True
            ):
                differences["action"][str(query_id)] = float(action)
                differences["quality"][str(query_id)] = float(quality)
    return differences


def component_scale(values: dict[str, float]) -> float:
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    return max(math.sqrt(variance), 1e-6)


def combine_components(
    raw: dict[str, dict[str, float]],
    action_scale: float,
    quality_scale: float,
    action_weight: float,
    quality_weight: float,
) -> dict[str, float]:
    if set(raw["action"]) != set(raw["quality"]):
        raise ValueError("action and quality scores cover different queries")
    return {
        query_id: action_weight * raw["action"][query_id] / action_scale
        + quality_weight * raw["quality"][query_id] / quality_scale
        for query_id in raw["action"]
    }


def choose_joint_dev_threshold(
    calibration_rows,
    calibration_scores,
    dev_rows,
    dev_scores,
    calibration_budget: int,
    t2_budget: int,
    t4_budget: int,
) -> tuple[float, dict]:
    all_scores = {**calibration_scores}
    for values in dev_scores.values():
        all_scores.update(values)
    best = None
    for threshold in threshold_candidates(all_scores):
        metrics = {
            "calibration": selector_metrics(
                calibration_rows, calibration_scores, threshold
            ),
            "t2": selector_metrics(
                dev_rows["t2_vqa_grounding"],
                dev_scores["t2_vqa_grounding"],
                threshold,
            ),
            "t4": selector_metrics(
                dev_rows["t4_caption_grounding"],
                dev_scores["t4_caption_grounding"],
                threshold,
            ),
        }
        if metrics["calibration"]["nonzero_to_zero_regressions"] > calibration_budget:
            continue
        if metrics["t2"]["nonzero_to_zero_regressions"] > t2_budget:
            continue
        if metrics["t4"]["nonzero_to_zero_regressions"] > t4_budget:
            continue
        captures = [
            value["oracle_gap_capture_from_strongest_fixed"]
            for value in metrics.values()
            if value["oracle_gap_capture_from_strongest_fixed"] is not None
        ]
        mean_capture = sum(captures) / len(captures)
        key = (
            mean_capture,
            metrics["calibration"]["selector_miou"],
            metrics["t2"]["selector_miou"] + metrics["t4"]["selector_miou"],
            -sum(value["nonzero_to_zero_regressions"] for value in metrics.values()),
            -sum(value["switches"] for value in metrics.values()),
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    if best is None:
        raise RuntimeError("no shared threshold satisfies all regression budgets")
    return float(best[1]), best[2]


def main() -> int:
    args = parse_args()
    stem = args.checkpoint.stem
    output_json = args.output_dir / f"{stem}.evaluation.json"
    output_report = args.output_dir / f"{stem}.evaluation.md"
    frozen_spec = args.output_dir / f"{stem}.decision.json"
    existing = [str(path) for path in (output_json, output_report, frozen_spec) if path.exists()]
    if existing and not args.force:
        raise FileExistsError("outputs exist; pass --force to replace: " + ", ".join(existing))

    import torch
    from transformers import AutoProcessor, CLIPModel

    from vsight.clip_verifier import ClipCandidateVerifier, read_selector_rows

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "vsight_e2_clip_verifier_checkpoint_v1":
        raise ValueError("unsupported verifier checkpoint")
    clip_path = Path(checkpoint["clip_model"])
    processor = AutoProcessor.from_pretrained(str(clip_path), local_files_only=True)
    clip = CLIPModel.from_pretrained(str(clip_path), local_files_only=True)
    model = ClipCandidateVerifier(clip, hidden_dim=int(checkpoint["hidden_dim"]))
    model.configure_adaptation(str(checkpoint["adaptation_mode"]))
    unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")
    device = torch.device(args.device)
    model.to(device)

    selector_summary = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    calibration_path = resolve_path(selector_summary["outputs"]["calibration"]["path"])
    if sha256(calibration_path) != selector_summary["outputs"]["calibration"]["sha256"]:
        raise ValueError("calibration manifest hash mismatch")
    calibration_rows = read_selector_rows(calibration_path)
    candidate_summary = json.loads(
        resolve_path(selector_summary["candidate_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    queue_summary = json.loads(
        resolve_path(candidate_summary["queue_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source_summary = json.loads(
        resolve_path(queue_summary["source_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    e1_image_root = Path(source_summary["images"]["root"])
    calibration_raw_scores = score_rows(
        model, processor, calibration_rows, e1_image_root, device, args
    )

    dev_rows = load_dev_rows(args)
    dev_raw_scores = {
        task: score_rows(model, processor, rows, args.image_root, device, args)
        for task, rows in dev_rows.items()
    }
    action_scale = component_scale(calibration_raw_scores["action"])
    quality_scale = component_scale(calibration_raw_scores["quality"])
    variants = [
        ("action", 1.0, 0.0),
        ("quality", 0.0, 1.0),
        ("action_plus_0.25_quality", 1.0, 0.25),
        ("action_plus_0.5_quality", 1.0, 0.5),
        ("action_plus_quality", 1.0, 1.0),
        ("action_plus_2_quality", 1.0, 2.0),
        ("action_plus_4_quality", 1.0, 4.0),
    ]
    combined_variants = {}
    for name, action_weight, quality_weight in variants:
        combined_variants[name] = {
            "calibration": combine_components(
                calibration_raw_scores,
                action_scale,
                quality_scale,
                action_weight,
                quality_weight,
            ),
            "dev": {
                task: combine_components(
                    raw,
                    action_scale,
                    quality_scale,
                    action_weight,
                    quality_weight,
                )
                for task, raw in dev_raw_scores.items()
            },
            "weights": {
                "action": action_weight,
                "quality": quality_weight,
            },
        }
    fixed_threshold = float(checkpoint["metadata"]["threshold"])
    calibration_scores = calibration_raw_scores["action"]
    dev_scores = {
        task: raw["action"] for task, raw in dev_raw_scores.items()
    }
    fixed = {
        "calibration": selector_metrics(
            calibration_rows, calibration_scores, fixed_threshold
        ),
        "t2": selector_metrics(
            dev_rows["t2_vqa_grounding"],
            dev_scores["t2_vqa_grounding"],
            fixed_threshold,
        ),
        "t4": selector_metrics(
            dev_rows["t4_caption_grounding"],
            dev_scores["t4_caption_grounding"],
            fixed_threshold,
        ),
    }
    calibration_budget = (
        fixed["calibration"]["unconditional_nonzero_to_zero_regressions"] // 2
    )
    variant_results = {}
    best_variant = None
    for name, variant in combined_variants.items():
        threshold, metrics = choose_joint_dev_threshold(
            calibration_rows,
            variant["calibration"],
            dev_rows,
            variant["dev"],
            calibration_budget,
            args.t2_regression_budget,
            args.t4_regression_budget,
        )
        captures = [
            value["oracle_gap_capture_from_strongest_fixed"]
            for value in metrics.values()
            if value["oracle_gap_capture_from_strongest_fixed"] is not None
        ]
        objective = sum(captures) / len(captures)
        variant_results[name] = {
            "weights": variant["weights"],
            "threshold": threshold,
            "objective_mean_capture_from_strongest_fixed": objective,
            "results": metrics,
        }
        key = (
            objective,
            metrics["calibration"]["selector_miou"],
            metrics["t2"]["selector_miou"] + metrics["t4"]["selector_miou"],
        )
        if best_variant is None or key > best_variant[0]:
            best_variant = (key, name)
    selected_variant = best_variant[1]
    joint_threshold = variant_results[selected_variant]["threshold"]
    tuned = variant_results[selected_variant]["results"]
    result = {
        "schema_version": "vsight_e2_verifier_evaluation_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "adaptation_mode": checkpoint["adaptation_mode"],
            "epoch": checkpoint["metadata"]["epoch"],
        },
        "calibration_only_threshold": fixed_threshold,
        "calibration_only_results": fixed,
        "joint_dev_threshold": joint_threshold,
        "selected_score_variant": selected_variant,
        "joint_dev_results": tuned,
        "score_variant_results": variant_results,
        "score_component_scales": {
            "action": action_scale,
            "quality": quality_scale,
        },
        "threshold_policy": {
            "objective": "mean oracle-gap capture from each split's strongest fixed policy",
            "calibration_nonzero_to_zero_budget": calibration_budget,
            "t2_nonzero_to_zero_budget": args.t2_regression_budget,
            "t4_nonzero_to_zero_budget": args.t4_regression_budget,
            "repaired_500_usage": "development threshold selection allowed by experiment protocol",
            "repaired_1996_accessed": False,
        },
    }
    decision = {
        "schema_version": "vsight_e2_frozen_decision_v1",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "threshold": joint_threshold,
        "score_variant": selected_variant,
        "score_weights": combined_variants[selected_variant]["weights"],
        "score_component_scales": {
            "action": action_scale,
            "quality": quality_scale,
        },
        "candidate_rule": "switch iff challenger_score - baseline_score > threshold",
        "baseline_null_rule": "keep null; stage 1 recovery disabled",
        "candidate_source_feature": "forbidden",
        "sealed_heldout_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frozen_spec.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# E2 verifier evaluation: {checkpoint['adaptation_mode']}",
        "",
        f"- Joint development threshold: {joint_threshold:.6f}",
        f"- Selected score variant: {selected_variant}",
    ]
    for name in ("calibration", "t2", "t4"):
        value = tuned[name]
        lines.append(
            f"- {name}: baseline {value['baseline_miou']:.6f}; state-preserving challenger "
            f"{value['state_preserving_challenger_miou']:.6f}; selector "
            f"{value['selector_miou']:.6f}; oracle {value['two_box_oracle_miou']:.6f}; "
            f"capture from strongest fixed {value['oracle_gap_capture_from_strongest_fixed']:.3%}; "
            f"nonzero-to-zero {value['nonzero_to_zero_regressions']}"
        )
    lines.extend(("", "Sealed repaired-1996 was not accessed.", ""))
    output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(tuned, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
