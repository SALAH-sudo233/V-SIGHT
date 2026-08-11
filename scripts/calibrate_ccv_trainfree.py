#!/usr/bin/env python3
"""Nested calibration of the train-free CCV absence threshold."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/ccv_trainfree_development/main.predictions.jsonl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ccv_trainfree_calibrated")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-added-fnr", type=float, default=0.03)
    parser.add_argument("--max-miou-loss", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold_for(group_id: str, count: int) -> int:
    digest = hashlib.sha256(f"vsight-ccv-calibration-v1:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def policy(row, threshold):
    action = str(row["action"])
    if (
        action == "REJECT"
        and row.get("verifier_eligible")
        and row.get("existence_margin") is not None
        and float(row["existence_margin"]) >= threshold
    ):
        return "ACCEPT", float(row["original_iou"] or 0.0)
    return action, float(row.get("selected_iou", row.get("original_iou")) or 0.0)


def safety(rows, threshold):
    positives = [r for r in rows if r["label_exists"]]
    negatives = [r for r in rows if not r["label_exists"]]
    pos_orig = sum(bool(r["original_accept"]) for r in positives)
    pos_post = sum(policy(r, threshold)[0] != "REJECT" for r in positives)
    orig_miou = sum(float(r["original_iou"] or 0.0) * bool(r["original_accept"]) for r in positives) / max(1, len(positives))
    post_miou = sum(policy(r, threshold)[1] * (policy(r, threshold)[0] != "REJECT") for r in positives) / max(1, len(positives))
    neg_orig = sum(bool(r["original_accept"]) for r in negatives)
    neg_post = sum(policy(r, threshold)[0] != "REJECT" for r in negatives)
    return {
        "added_fnr": ((len(positives) - pos_post) - (len(positives) - pos_orig)) / max(1, len(positives)),
        "miou_loss": orig_miou - post_miou,
        "negative_rejection_gain": (neg_orig - neg_post) / max(1, len(negatives)),
    }


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = arguments()
    if args.folds < 3:
        raise ValueError("folds must be >= 3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "main.calibrated.predictions.jsonl.gz"
    summary_path = args.output_dir / "main.calibrated.summary.json"
    if not args.force and (output.exists() or summary_path.exists()):
        raise FileExistsError("calibrated output exists; pass --force")
    rows = read_rows(args.predictions)
    models = sorted({str(r["model"]) for r in rows})
    tasks = sorted({str(r["task"]) for r in rows})
    candidates = [round(-0.25 + 0.005 * index, 4) for index in range(201)]
    decisions = []
    audits = []
    for task in tasks:
        for heldout_model in models:
            for outer_fold in range(args.folds):
                calibration_fold = (outer_fold + 1) % args.folds
                calibration = [
                    r for r in rows
                    if r["task"] == task and r["model"] != heldout_model
                    and fold_for(str(r["group_id"]), args.folds) == calibration_fold
                ]
                feasible = []
                for threshold in candidates:
                    by_model = defaultdict(list)
                    for row in calibration:
                        by_model[str(row["model"])].append(row)
                    metrics = [safety(part, threshold) for part in by_model.values()]
                    if metrics and all(
                        metric["added_fnr"] <= args.max_added_fnr
                        and metric["miou_loss"] <= args.max_miou_loss
                        for metric in metrics
                    ):
                        feasible.append((sum(metric["negative_rejection_gain"] for metric in metrics) / len(metrics), threshold))
                if not feasible:
                    raise RuntimeError(f"no safe threshold for {task}/{heldout_model}/fold{outer_fold}")
                _, threshold = max(feasible, key=lambda value: (value[0], -value[1]))
                test = [
                    r for r in rows
                    if r["task"] == task and r["model"] == heldout_model
                    and fold_for(str(r["group_id"]), args.folds) == outer_fold
                ]
                for row in test:
                    action, selected_iou = policy(row, threshold)
                    decisions.append({
                        **row,
                        "calibrated_threshold": threshold,
                        "uncalibrated_action": row["action"],
                        "action": action,
                        "selected_iou": selected_iou,
                    })
                audits.append({
                    "task": task,
                    "heldout_model": heldout_model,
                    "outer_fold": outer_fold,
                    "calibration_fold": calibration_fold,
                    "threshold": threshold,
                    "calibration_records": len(calibration),
                })
    decisions.sort(key=lambda r: (str(r["model"]), str(r["task"]), str(r["sample_id"])))
    with deterministic_gzip(output) as handle:
        for row in decisions:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "vsight_ccv_trainfree_calibrated_v1",
        "records": len(decisions),
        "folds": args.folds,
        "safety_budget": {"max_added_fnr": args.max_added_fnr, "max_miou_loss": args.max_miou_loss},
        "audits": audits,
        "output": str(output),
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
