#!/usr/bin/env python3
"""Train and calibrate the lightweight E2b relation-set verifier."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402
from vsight.e2_verifier import selector_metrics, threshold_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e2b/selector/e2b_selector.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e2b_relation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--switch-sampling-weight", type=float, default=2.0)
    parser.add_argument("--reference-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def move(batch, device):
    import torch

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def predict(model, loader, device) -> dict[str, float]:
    import torch

    model.eval()
    differences = {}
    with torch.inference_mode():
        for original in loader:
            batch = move(original, device)
            scores, _, _ = model(
                batch["relation_index"],
                batch["candidate_features"],
                batch["pair_features"],
                batch["reference_mask"],
            )
            values = (scores[:, 1] - scores[:, 0]).cpu().tolist()
            for query_id, value in zip(batch["query_id"], values, strict=True):
                differences[str(query_id)] = float(value)
    return differences


def choose_joint_threshold(rows, scores) -> tuple[float, dict, float]:
    by_task = {
        task: [row for row in rows if row["task"] == task] for task in ("t2", "t4")
    }
    budgets = {
        task: selector_metrics(
            values, {}, math.inf, unscored_policy="challenger"
        )["unconditional_nonzero_to_zero_regressions"]
        // 2
        for task, values in by_task.items()
    }
    best = None
    for threshold in threshold_candidates(scores):
        metrics = {
            task: selector_metrics(
                values,
                scores,
                threshold,
                unscored_policy="challenger",
            )
            for task, values in by_task.items()
        }
        if any(
            metrics[task]["nonzero_to_zero_regressions"] > budgets[task]
            for task in by_task
        ):
            continue
        captures = [
            metrics[task]["oracle_gap_capture_from_strongest_fixed"]
            for task in by_task
            if metrics[task]["oracle_gap_capture_from_strongest_fixed"] is not None
        ]
        objective = sum(captures) / len(captures)
        key = (
            objective,
            sum(metrics[task]["selector_miou"] for task in by_task),
            -sum(metrics[task]["nonzero_to_zero_regressions"] for task in by_task),
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics, objective, budgets)
    if best is None:
        raise RuntimeError("no E2b threshold satisfies both task budgets")
    return float(best[1]), {**best[2], "budgets": best[4]}, float(best[3])


def main() -> int:
    args = parse_args()
    checkpoint_path = args.output_dir / "e2b_relation.best.pt"
    summary_path = args.output_dir / "training.json"
    if not args.force and (checkpoint_path.exists() or summary_path.exists()):
        raise FileExistsError("E2b training output exists; pass --force to replace")
    random.seed(args.seed)

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from vsight.relation_verifier import (
        E2bRelationDataset,
        RelationSetVerifier,
        read_e2b_rows,
        relation_verifier_loss,
    )

    torch.manual_seed(args.seed)
    manifest = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    paths = {
        split: resolve_path(item["path"]) for split, item in manifest["outputs"].items()
    }
    for split, path in paths.items():
        if sha256(path) != manifest["outputs"][split]["sha256"]:
            raise ValueError(f"E2b selector hash mismatch: {split}")
    train_rows = read_e2b_rows(paths["train"])
    calibration_rows = read_e2b_rows(paths["calibration"])
    train_dataset = E2bRelationDataset(train_rows, training=True)
    calibration_dataset = E2bRelationDataset(calibration_rows, training=False)
    weights = [
        args.switch_sampling_weight if row["selector_action"] == "switch" else 1.0
        for row in train_dataset.rows
    ]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    calibration_loader = DataLoader(
        calibration_dataset, batch_size=args.batch_size * 2, shuffle=False
    )
    device = torch.device(args.device)
    model = RelationSetVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = None
    best_epoch = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        count = 0
        for original in train_loader:
            batch = move(original, device)
            scores, quality, reference_logits = model(
                batch["relation_index"],
                batch["candidate_features"],
                batch["pair_features"],
                batch["reference_mask"],
            )
            loss, pieces = relation_verifier_loss(
                scores,
                quality,
                reference_logits,
                batch["label"],
                batch["ious"],
                batch["reference_best_index"],
                reference_weight=args.reference_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = int(batch["label"].numel())
            count += batch_size
            totals["loss"] += float(loss.detach()) * batch_size
            for name, value in pieces.items():
                totals[name] += float(value) * batch_size
        scores = predict(model, calibration_loader, device)
        threshold, metrics, objective = choose_joint_threshold(calibration_rows, scores)
        entry = {
            "epoch": epoch,
            "train": {name: value / count for name, value in totals.items()},
            "threshold": threshold,
            "objective": objective,
            "calibration": metrics,
        }
        history.append(entry)
        key = (
            objective,
            metrics["t2"]["selector_miou"] + metrics["t4"]["selector_miou"],
        )
        print(
            f"epoch={epoch} loss={entry['train']['loss']:.4f} objective={objective:.3f} "
            f"t2={metrics['t2']['selector_miou']:.5f} "
            f"t4={metrics['t4']['selector_miou']:.5f} threshold={threshold:.4f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema_version": "vsight_e2b_relation_checkpoint_v1",
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "threshold": threshold,
                    "calibration": metrics,
                    "selector_manifest_sha256": sha256(args.selector_summary),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    result = {
        "schema_version": "vsight_e2b_relation_training_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_records": len(train_dataset),
        "calibration_records": len(calibration_dataset),
        "train_actions": dict(
            sorted(Counter(row["selector_action"] for row in train_dataset.rows).items())
        ),
        "best_epoch": best_epoch,
        "best": history[best_epoch - 1],
        "history": history,
        "duration_sec": time.perf_counter() - started,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "external_api_used": False,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
