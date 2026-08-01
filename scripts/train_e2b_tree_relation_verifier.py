#!/usr/bin/env python3
"""Fit antisymmetric tree regressors on explicit E2b relation features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e2b_relation_verifier import choose_joint_threshold  # noqa: E402
from vsight.e1_data import sha256  # noqa: E402
from vsight.relation_supervision import (  # noqa: E402
    RELATION_PATTERNS,
    normalized_box_features,
    relation_context_features,
)


RELATIONS = tuple(name for name, _ in RELATION_PATTERNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e2b/selector/e2b_selector.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e2b_tree_relation")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def utility_features(row: dict, box) -> list[float]:
    width, height = int(row["image_width"]), int(row["image_height"])
    relation_context = relation_context_features(
        box,
        row["reference_proposals"],
        str(row["relation"]),
        width,
        height,
    )
    # Relation one-hot is included once at the pair level, not repeated in
    # each candidate utility vector.
    return [
        *normalized_box_features(box, width, height),
        *relation_context[len(RELATIONS) :],
    ]


def comparison_features(row: dict, reverse: bool = False) -> list[float]:
    first = utility_features(row, row["baseline_bbox_xyxy"])
    second = utility_features(row, row["challenger_bbox_xyxy"])
    if reverse:
        first, second = second, first
    relation = [float(name == row["relation"]) for name in RELATIONS]
    return [*relation, *(right - left for left, right in zip(first, second, strict=True))]


def antisymmetric_predict(model, rows) -> dict[str, float]:
    import numpy as np

    forward = np.asarray([comparison_features(row) for row in rows], dtype="float32")
    reverse = np.asarray(
        [comparison_features(row, reverse=True) for row in rows], dtype="float32"
    )
    values = (model.predict(forward) - model.predict(reverse)) / 2.0
    return {str(row["query_id"]): float(value) for row, value in zip(rows, values, strict=True)}


def main() -> int:
    args = parse_args()
    summary_path = args.output_dir / "training.json"
    model_path = args.output_dir / "e2b_tree_relation.joblib"
    if not args.force and (summary_path.exists() or model_path.exists()):
        raise FileExistsError("tree verifier output exists; pass --force to replace")

    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

    from vsight.relation_verifier import read_e2b_rows

    manifest = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    paths = {
        split: resolve_path(item["path"]) for split, item in manifest["outputs"].items()
    }
    for split, path in paths.items():
        if sha256(path) != manifest["outputs"][split]["sha256"]:
            raise ValueError(f"selector hash mismatch: {split}")
    train_all = read_e2b_rows(paths["train"])
    calibration_all = read_e2b_rows(paths["calibration"])
    train = [row for row in train_all if row["relation_selector_eligible"]]
    calibration = [row for row in calibration_all if row["relation_selector_eligible"]]
    forward = np.asarray([comparison_features(row) for row in train], dtype="float32")
    reverse = np.asarray(
        [comparison_features(row, reverse=True) for row in train], dtype="float32"
    )
    target = np.asarray(
        [float(row["challenger_iou"]) - float(row["baseline_iou"]) for row in train],
        dtype="float32",
    )
    features = np.concatenate((forward, reverse), axis=0)
    targets = np.concatenate((target, -target), axis=0)
    variants = {
        "extra_trees_leaf_2": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.seed,
        ),
        "extra_trees_leaf_5": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.seed,
        ),
        "extra_trees_leaf_10": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=-1,
            random_state=args.seed,
        ),
        "hist_gradient": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=args.seed,
        ),
    }
    started = time.perf_counter()
    results = {}
    best = None
    for name, model in variants.items():
        model.fit(features, targets)
        scores = antisymmetric_predict(model, calibration)
        threshold, metrics, objective = choose_joint_threshold(calibration_all, scores)
        result = {
            "threshold": threshold,
            "objective": objective,
            "calibration": metrics,
        }
        results[name] = result
        key = (
            objective,
            metrics["t2"]["selector_miou"] + metrics["t4"]["selector_miou"],
        )
        print(
            f"variant={name} objective={objective:.3f} "
            f"t2={metrics['t2']['selector_miou']:.5f} "
            f"t4={metrics['t4']['selector_miou']:.5f}",
            flush=True,
        )
        if best is None or key > best[0]:
            best = (key, name, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[2], model_path)
    summary = {
        "schema_version": "vsight_e2b_tree_relation_training_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_records": len(train),
        "calibration_records": len(calibration),
        "feature_dim": int(features.shape[1]),
        "antisymmetric_augmentation": True,
        "best_variant": best[1],
        "best": results[best[1]],
        "variants": results,
        "duration_sec": time.perf_counter() - started,
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256(model_path),
            "bytes": model_path.stat().st_size,
        },
        "external_api_used": False,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
