#!/usr/bin/env python3
"""Nested image/model cross-validation for the single-pass existence gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402
from vsight.relation_supervision import RELATION_PATTERNS  # noqa: E402


TASKS = ("t2_vqa_grounding", "t4_caption_grounding")
FAMILIES = {
    "base_or_supervised": ("qwen2.5-vl-7b", "Qwen3-VL-8B", "LENS"),
    "rl_or_reasoning": (
        "visual-rft",
        "Seg-R1",
        "Seg-zero",
        "VisionReasoner",
        "TreeVGR",
        "Orsta-7B",
        "Vision-R1",
        "UniVG-R1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-summary",
        type=Path,
        default=ROOT / "data/e3/singlepass/crossmodel/e3_singlepass_crossmodel.summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/e3_singlepass_crossmodel"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-added-fnr", type=float, default=0.03)
    parser.add_argument("--max-miou-loss", type=float, default=0.005)
    parser.add_argument("--heldout-max-added-fnr", type=float, default=0.03)
    parser.add_argument("--heldout-max-miou-loss", type=float, default=0.005)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument(
        "--classifier", choices=("histgb", "extra_trees"), default="histgb"
    )
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "geometry",
            "object_support",
            "modifier_relation_support",
            "legacy_modifier_relation_support",
        ),
        default=None,
    )
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


def fold_for(group_id: str, count: int) -> int:
    digest = hashlib.sha256(f"vsight-e3-singlepass-cv-v1:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def feature_sets(all_fields: list[str]) -> dict[str, list[str]]:
    geometry = [
        name
        for name in all_fields
        if name.startswith("candidate_") or name == "original_found"
    ]
    head = geometry + [
        name
        for name in all_fields
        if name == "head_parsed" or name.startswith("head_")
    ]
    legacy = [
        "original_found",
        "candidate_valid",
        "candidate_area",
        "candidate_center_x",
        "candidate_center_y",
        "candidate_aspect_log",
        "candidate_edge_distance",
        "head_parsed",
        "full_distinct",
        "head_proposal_count",
        "head_max_score",
        "head_mean_score",
        "head_max_candidate_iou",
        "head_score_weighted_candidate_iou",
        "full_proposal_count",
        "full_max_score",
        "full_mean_score",
        "full_max_candidate_iou",
        "full_score_weighted_candidate_iou",
        "full_minus_head_score",
        "full_minus_head_iou",
        *(f"relation_{name}" for name, _ in RELATION_PATTERNS),
    ]
    return {
        "geometry": sorted(set(geometry), key=all_fields.index),
        "object_support": sorted(set(head), key=all_fields.index),
        "modifier_relation_support": list(all_fields),
        "legacy_modifier_relation_support": [
            field for field in legacy if field in all_fields
        ],
    }


def matrix(rows: list[dict], fields: list[str]):
    import numpy as np

    return np.asarray(
        [[float(row["features"][field]) for field in fields] for row in rows],
        dtype="float32",
    )


def choose_threshold(
    rows: list[dict], probabilities, max_added_fnr: float, max_miou_loss: float
) -> tuple[float, dict]:
    import numpy as np

    if len(rows) != len(probabilities):
        raise ValueError("calibration rows/probabilities differ")
    models = sorted({str(row["model"]) for row in rows})
    candidates = np.unique(np.asarray(probabilities, dtype="float64"))
    candidates = np.concatenate(([-math.inf], candidates, [math.inf]))
    added_fnr = []
    miou_loss = []
    negative_rejections = []
    model_audit = {}
    for model in models:
        indices = [index for index, row in enumerate(rows) if row["model"] == model]
        model_rows = [rows[index] for index in indices]
        model_probabilities = np.asarray([probabilities[index] for index in indices])
        order = np.argsort(model_probabilities, kind="stable")
        sorted_probabilities = model_probabilities[order]
        sorted_rows = [model_rows[index] for index in order]
        positive = np.asarray([float(row["label_exists"]) for row in sorted_rows])
        negative = 1.0 - positive
        positive_iou = np.asarray(
            [float(row["original_iou"]) if row["label_exists"] else 0.0 for row in sorted_rows]
        )
        positive_prefix = np.concatenate(([0.0], np.cumsum(positive)))
        negative_prefix = np.concatenate(([0.0], np.cumsum(negative)))
        iou_prefix = np.concatenate(([0.0], np.cumsum(positive_iou)))
        rejected = np.searchsorted(sorted_probabilities, candidates, side="left")
        positive_total = sum(bool(row["label_exists"]) for row in rows if row["model"] == model)
        negative_total = sum(not bool(row["label_exists"]) for row in rows if row["model"] == model)
        model_added_fnr = positive_prefix[rejected] / max(positive_total, 1)
        model_miou_loss = iou_prefix[rejected] / max(positive_total, 1)
        model_negative_rejections = negative_prefix[rejected] / max(negative_total, 1)
        added_fnr.append(model_added_fnr)
        miou_loss.append(model_miou_loss)
        negative_rejections.append(model_negative_rejections)
        model_audit[model] = {
            "positive_total": positive_total,
            "negative_total": negative_total,
        }
    added_fnr = np.stack(added_fnr)
    miou_loss = np.stack(miou_loss)
    negative_rejections = np.stack(negative_rejections)
    feasible = np.all(added_fnr <= max_added_fnr + 1e-12, axis=0) & np.all(
        miou_loss <= max_miou_loss + 1e-12, axis=0
    )
    indices = np.flatnonzero(feasible)
    if not len(indices):
        raise RuntimeError("no threshold satisfies calibration safety budgets")
    objective = negative_rejections.mean(axis=0)
    mean_iou_loss = miou_loss.mean(axis=0)
    mean_fnr = added_fnr.mean(axis=0)
    best = max(
        indices,
        key=lambda index: (
            float(objective[index]),
            -float(mean_iou_loss[index]),
            -float(mean_fnr[index]),
            -float(candidates[index]),
        ),
    )
    threshold = float(candidates[best])
    return threshold, {
        "threshold": threshold,
        "mean_negative_rejection_gain": float(objective[best]),
        "mean_added_fnr": float(mean_fnr[best]),
        "max_added_fnr": float(added_fnr[:, best].max()),
        "mean_miou_loss": float(mean_iou_loss[best]),
        "max_miou_loss": float(miou_loss[:, best].max()),
        "models": model_audit,
    }


def evaluate_rows(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["label_exists"]]
    negative = [row for row in rows if not row["label_exists"]]
    if not positive or not negative:
        raise ValueError("evaluation needs positive and negative rows")

    def accepted(row: dict, field: str) -> bool:
        return bool(row[field])

    result = {}
    for name, field in (("original", "original_accept"), ("postprocessor", "post_accept")):
        pos_accept = sum(accepted(row, field) for row in positive)
        neg_accept = sum(accepted(row, field) for row in negative)
        miou = sum(
            float(row["original_iou"]) if accepted(row, field) else 0.0
            for row in positive
        ) / len(positive)
        by_type = {}
        for kind in ("object", "co_occurrence", "attribute", "relation"):
            subset = [row for row in negative if row["hallucination_type"] == kind]
            by_type[kind] = {
                "records": len(subset),
                "false_accepts": sum(accepted(row, field) for row in subset),
                "rate": (
                    sum(accepted(row, field) for row in subset) / len(subset)
                    if subset
                    else None
                ),
            }
        boh = (by_type["object"]["rate"] + by_type["co_occurrence"]["rate"]) / 2
        roh = (by_type["attribute"]["rate"] + by_type["relation"]["rate"]) / 2
        result[name] = {
            "positive_records": len(positive),
            "negative_records": len(negative),
            "positive_miou": miou,
            "positive_fnr": 1 - pos_accept / len(positive),
            "negative_false_accept_rate": neg_accept / len(negative),
            "balanced_accuracy": (
                pos_accept / len(positive) + 1 - neg_accept / len(negative)
            )
            / 2,
            "false_accept_by_type": by_type,
            "boh_false_accept_rate": boh,
            "roh_false_accept_rate": roh,
            "roh_minus_boh_gap": roh - boh,
        }
    result["delta"] = {
        "positive_miou": result["postprocessor"]["positive_miou"]
        - result["original"]["positive_miou"],
        "positive_fnr": result["postprocessor"]["positive_fnr"]
        - result["original"]["positive_fnr"],
        "negative_false_accept_rate": result["postprocessor"][
            "negative_false_accept_rate"
        ]
        - result["original"]["negative_false_accept_rate"],
        "balanced_accuracy": result["postprocessor"]["balanced_accuracy"]
        - result["original"]["balanced_accuracy"],
        "roh_minus_boh_gap": result["postprocessor"]["roh_minus_boh_gap"]
        - result["original"]["roh_minus_boh_gap"],
    }
    return result


def main() -> int:
    args = parse_args()
    args.feature_summary = resolve_path(str(args.feature_summary))
    args.output_dir = resolve_path(str(args.output_dir))
    if args.folds < 3:
        raise ValueError("at least three folds are required")
    budgets = (
        args.max_added_fnr,
        args.max_miou_loss,
        args.heldout_max_added_fnr,
        args.heldout_max_miou_loss,
    )
    if any(not 0 <= value <= 1 for value in budgets):
        raise ValueError("safety budgets must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "evaluation.json"
    decisions_path = args.output_dir / "decisions.jsonl.gz"
    if not args.force and (summary_path.exists() or decisions_path.exists()):
        raise FileExistsError("single-pass evaluation exists; pass --force")

    manifest = json.loads(args.feature_summary.read_text(encoding="utf-8"))
    item = manifest["outputs"]["development"]
    feature_path = resolve_path(item["path"])
    if sha256(feature_path) != item["sha256"]:
        raise ValueError("cross-model feature manifest hash mismatch")
    rows = read_gzip(feature_path)
    models = list(manifest["models"])
    feature_variants = feature_sets(list(manifest["inference_feature_fields"]))
    if args.variants:
        feature_variants = {
            name: feature_variants[name]
            for name in args.variants
        }

    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    all_results = {}
    all_decisions = []
    for variant_name, fields in feature_variants.items():
        variant_decisions = []
        fold_audits = []
        for task in TASKS:
            task_rows = [row for row in rows if row["task"] == task]
            for heldout_model in models:
                for outer_fold in range(args.folds):
                    calibration_fold = (outer_fold + 1) % args.folds
                    train_rows = [
                        row
                        for row in task_rows
                        if row["model"] != heldout_model
                        and fold_for(row["group_id"], args.folds)
                        not in {outer_fold, calibration_fold}
                        and row["features"]["original_found"] > 0.5
                    ]
                    calibration_rows = [
                        row
                        for row in task_rows
                        if row["model"] != heldout_model
                        and fold_for(row["group_id"], args.folds) == calibration_fold
                        and row["features"]["original_found"] > 0.5
                    ]
                    test_rows = [
                        row
                        for row in task_rows
                        if row["model"] == heldout_model
                        and fold_for(row["group_id"], args.folds) == outer_fold
                    ]
                    if args.classifier == "histgb":
                        classifier = HistGradientBoostingClassifier(
                            learning_rate=0.08,
                            max_iter=args.max_iter,
                            max_leaf_nodes=15,
                            min_samples_leaf=30,
                            l2_regularization=1.0,
                            class_weight="balanced",
                            random_state=20260803,
                        )
                    else:
                        classifier = ExtraTreesClassifier(
                            n_estimators=args.n_estimators,
                            max_features=0.7,
                            min_samples_leaf=20,
                            class_weight="balanced",
                            n_jobs=args.n_jobs,
                            random_state=20260803,
                        )
                    classifier.fit(
                        matrix(train_rows, fields),
                        np.asarray([int(row["label_exists"]) for row in train_rows]),
                    )
                    calibration_probabilities = classifier.predict_proba(
                        matrix(calibration_rows, fields)
                    )[:, 1]
                    threshold, threshold_audit = choose_threshold(
                        calibration_rows,
                        calibration_probabilities,
                        args.max_added_fnr,
                        args.max_miou_loss,
                    )
                    found_test = [
                        row for row in test_rows if row["features"]["original_found"] > 0.5
                    ]
                    probabilities = (
                        classifier.predict_proba(matrix(found_test, fields))[:, 1]
                        if found_test
                        else np.asarray([])
                    )
                    probability_by_sample = {
                        str(row["sample_id"]): float(probability)
                        for row, probability in zip(found_test, probabilities, strict=True)
                    }
                    for row in test_rows:
                        original_accept = bool(row["features"]["original_found"] > 0.5)
                        probability = probability_by_sample.get(str(row["sample_id"]))
                        post_accept = bool(
                            original_accept
                            and probability is not None
                            and probability >= threshold
                        )
                        decision = {
                            **row,
                            "feature_variant": variant_name,
                            "outer_fold": outer_fold,
                            "heldout_model": heldout_model,
                            "support_probability": probability,
                            "threshold": threshold,
                            "original_accept": original_accept,
                            "post_accept": post_accept,
                        }
                        variant_decisions.append(decision)
                    fold_audits.append(
                        {
                            "task": task,
                            "heldout_model": heldout_model,
                            "outer_fold": outer_fold,
                            "calibration_fold": calibration_fold,
                            "train_records": len(train_rows),
                            "calibration_records": len(calibration_rows),
                            "test_records": len(test_rows),
                            **threshold_audit,
                        }
                    )
        expected = len(models) * len(TASKS) * 2500
        if len(variant_decisions) != expected:
            raise ValueError(
                f"variant {variant_name} expected {expected} decisions, got {len(variant_decisions)}"
            )
        per_model = {
            model: {
                task: evaluate_rows(
                    [
                        row
                        for row in variant_decisions
                        if row["model"] == model and row["task"] == task
                    ]
                )
                for task in TASKS
            }
            for model in models
        }
        per_family = {
            family: {
                task: evaluate_rows(
                    [
                        row
                        for row in variant_decisions
                        if row["model"] in family_models and row["task"] == task
                    ]
                )
                for task in TASKS
            }
            for family, family_models in FAMILIES.items()
        }
        test_safety = {}
        for model in models:
            test_safety[model] = {}
            for task in TASKS:
                delta = per_model[model][task]["delta"]
                added_fnr = float(delta["positive_fnr"])
                miou_loss = -float(delta["positive_miou"])
                test_safety[model][task] = {
                    "added_fnr": added_fnr,
                    "positive_miou_loss": miou_loss,
                    "passes_added_fnr_budget": (
                        added_fnr <= args.heldout_max_added_fnr + 1e-12
                    ),
                    "passes_positive_miou_budget": (
                        miou_loss <= args.heldout_max_miou_loss + 1e-12
                    ),
                }
        all_results[variant_name] = {
            "feature_fields": fields,
            "per_model": per_model,
            "per_family": per_family,
            "heldout_model_safety": test_safety,
            "fold_audits": fold_audits,
        }
        all_decisions.extend(variant_decisions)

    temporary = decisions_path.with_name(decisions_path.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in all_decisions:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        temporary.replace(decisions_path)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": "vsight_e3_singlepass_crossmodel_evaluation_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "nested_cross_validation_on_development_not_final_test",
        "protocol": {
            "outer": "simultaneously held-out model and image fold",
            "calibration": "different image fold from outer test",
            "threshold_constraints_apply_to_every_calibration_model": True,
            "folds": args.folds,
            "max_added_fnr": args.max_added_fnr,
            "max_positive_miou_loss": args.max_miou_loss,
            "heldout_max_added_fnr": args.heldout_max_added_fnr,
            "heldout_max_positive_miou_loss": args.heldout_max_miou_loss,
            "classifier": (
                "HistGradientBoostingClassifier"
                if args.classifier == "histgb"
                else "ExtraTreesClassifier"
            ),
            "max_iter": args.max_iter,
            "classifier_name": (
                "HistGradientBoostingClassifier"
                if args.classifier == "histgb"
                else "ExtraTreesClassifier"
            ),
            "n_estimators": args.n_estimators,
            "n_jobs": args.n_jobs,
        },
        "feature_manifest": {
            "path": str(args.feature_summary),
            "sha256": sha256(args.feature_summary),
        },
        "models": models,
        "families": {key: list(value) for key, value in FAMILIES.items()},
        "results": all_results,
        "decisions": {
            "path": str(decisions_path.relative_to(ROOT)),
            "sha256": sha256(decisions_path),
            "bytes": decisions_path.stat().st_size,
        },
        "upstream_autoregressive_calls_per_query": 1,
        "local_detector_required": True,
        "caption_rewritten": False,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dataset_role": summary["dataset_role"],
                "results": {
                    variant: result["per_family"]
                    for variant, result in all_results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
