#!/usr/bin/env python3
"""Train the E2b CLIP candidate scorer with explicit reference geometry."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e2b_relation_verifier import choose_joint_threshold  # noqa: E402
from train_e2_verifier import DEFAULT_CLIP  # noqa: E402
from vsight.e1_data import sha256  # noqa: E402


DEFAULT_INITIALIZATION = ROOT / "outputs/e2_verifier/e2_frozen.best.pt"
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e2b/selector/e2b_selector.summary.json",
    )
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--initialize", type=Path, default=DEFAULT_INITIALIZATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e2b_clip_relation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--switch-sampling-weight", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def move(batch, device):
    import torch

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_inputs(batch):
    return {
        key: batch[key]
        for key in ("pixel_values", "input_ids", "attention_mask", "geometry")
    }


def predict(model, loader, device) -> dict[str, float]:
    import torch

    model.eval()
    scores_by_query = {}
    with torch.inference_mode():
        for original in loader:
            batch = move(original, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores, _ = model(**model_inputs(batch))
            values = (scores[:, 1] - scores[:, 0]).float().cpu().tolist()
            for query_id, value in zip(batch["query_ids"], values, strict=True):
                scores_by_query[str(query_id)] = float(value)
    return scores_by_query


def main() -> int:
    args = parse_args()
    checkpoint_path = args.output_dir / "e2b_clip_relation.best.pt"
    summary_path = args.output_dir / "training.json"
    if not args.force and (checkpoint_path.exists() or summary_path.exists()):
        raise FileExistsError("E2b CLIP output exists; pass --force to replace")
    random.seed(args.seed)

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoProcessor, CLIPModel

    from vsight.clip_verifier import (
        ClipCandidateVerifier,
        E2BatchCollator,
        read_selector_rows,
        verifier_loss,
    )
    from vsight.e2b_clip_verifier import E2B_GEOMETRY_DIM, E2bClipRelationDataset

    torch.manual_seed(args.seed)
    manifest = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    paths = {
        split: resolve_path(item["path"]) for split, item in manifest["outputs"].items()
    }
    for split, path in paths.items():
        if sha256(path) != manifest["outputs"][split]["sha256"]:
            raise ValueError(f"E2b selector hash mismatch: {split}")
    train_rows = read_selector_rows(paths["train"])
    calibration_rows = read_selector_rows(paths["calibration"])
    processor = AutoProcessor.from_pretrained(str(args.clip_model), local_files_only=True)
    model = ClipCandidateVerifier(
        CLIPModel.from_pretrained(str(args.clip_model), local_files_only=True),
        geometry_dim=E2B_GEOMETRY_DIM,
    )
    model.configure_adaptation("frozen")
    initialization = None
    if args.initialize.is_file():
        initial = torch.load(args.initialize, map_location="cpu", weights_only=False)
        compatible = {
            name: value
            for name, value in initial["state_dict"].items()
            if name.startswith("candidate_encoder.")
        }
        model.load_state_dict(compatible, strict=False)
        initialization = {"path": str(args.initialize.resolve()), "sha256": sha256(args.initialize)}
    device = torch.device(args.device)
    model.to(device)

    train_dataset = E2bClipRelationDataset(train_rows, args.image_root, training=True)
    calibration_dataset = E2bClipRelationDataset(
        calibration_rows, args.image_root, training=False
    )
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
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": E2BatchCollator(processor),
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    calibration_loader = DataLoader(
        calibration_dataset, shuffle=False, **options
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
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
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores, quality = model(**model_inputs(batch))
                loss, pieces = verifier_loss(
                    scores,
                    quality,
                    batch["labels"],
                    batch["ious"],
                    quality_weight=0.5,
                    delta_weight=0.25,
                    safe_weight=1.0,
                    safe_margin=0.75,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["labels"].numel())
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
            f"t4={metrics['t4']['selector_miou']:.5f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            state = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
                if not name.startswith("clip.")
            }
            torch.save(
                {
                    "schema_version": "vsight_e2b_clip_relation_checkpoint_v1",
                    "state_dict": state,
                    "clip_model": str(args.clip_model.resolve()),
                    "geometry_dim": E2B_GEOMETRY_DIM,
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
        "schema_version": "vsight_e2b_clip_relation_training_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_records": len(train_dataset),
        "calibration_records": len(calibration_dataset),
        "initialization": initialization,
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
