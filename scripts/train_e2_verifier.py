#!/usr/bin/env python3
"""Train and calibrate the source-agnostic E2 CLIP candidate verifier."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402
from vsight.e2_verifier import choose_safe_threshold  # noqa: E402


DEFAULT_CLIP = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--openai--clip-vit-base-patch32/snapshots/"
    "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e1/p1/selector/e1_p1_selector.summary.json",
    )
    parser.add_argument("--clip-model", type=Path, default=DEFAULT_CLIP)
    parser.add_argument(
        "--initialize-checkpoint",
        type=Path,
        help="Optional compatible checkpoint used to initialize shared heads.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/e2_verifier")
    parser.add_argument("--mode", choices=("frozen", "last_block"), default="frozen")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--clip-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--switch-sampling-weight", type=float, default=3.0)
    parser.add_argument("--refcocog-sampling-weight", type=float, default=1.0)
    parser.add_argument("--auxiliary-manifest", type=Path)
    parser.add_argument("--auxiliary-sampling-weight", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--safe-weight", type=float, default=1.0)
    parser.add_argument("--safe-margin", type=float, default=0.75)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-calibration", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device) -> dict:
    import torch

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_inputs(batch: dict) -> dict:
    return {
        key: batch[key]
        for key in ("pixel_values", "input_ids", "attention_mask", "geometry")
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    args: argparse.Namespace,
    epoch: int,
) -> dict:
    import torch

    from vsight.clip_verifier import verifier_loss

    model.train()
    totals = defaultdict(float)
    examples = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step, original in enumerate(loader, start=1):
        batch = move_batch(original, device)
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
                quality_weight=args.quality_weight,
                delta_weight=args.delta_weight,
                safe_weight=args.safe_weight,
                safe_margin=args.safe_margin,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        batch_size = int(batch["labels"].shape[0])
        examples += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in pieces.items():
            totals[name] += float(value) * batch_size
        if step == 1 or step % 100 == 0 or step == len(loader):
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={totals['loss']/examples:.4f} examples={examples}",
                flush=True,
            )
    return {
        **{name: value / examples for name, value in totals.items()},
        "examples": examples,
        "duration_sec": time.perf_counter() - started,
    }


def predict_score_differences(model, loader, device) -> tuple[dict[str, float], float]:
    import torch

    model.eval()
    differences: dict[str, float] = {}
    correct = total = 0
    with torch.inference_mode():
        for original in loader:
            batch = move_batch(original, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores, _ = model(**model_inputs(batch))
            values = (scores[:, 1] - scores[:, 0]).float().cpu().tolist()
            for query_id, value in zip(batch["query_ids"], values, strict=True):
                differences[str(query_id)] = float(value)
            correct += int((scores.argmax(dim=1) == batch["labels"]).sum())
            total += int(batch["labels"].numel())
    return differences, correct / total


def trainable_state_dict(model) -> dict:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def save_checkpoint(path: Path, model, args: argparse.Namespace, metadata: dict) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "vsight_e2_clip_verifier_checkpoint_v1",
            "adaptation_mode": args.mode,
            "clip_model": str(args.clip_model.resolve()),
            "hidden_dim": 256,
            "state_dict": trainable_state_dict(model),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if (
        args.switch_sampling_weight <= 0
        or args.auxiliary_sampling_weight <= 0
        or args.refcocog_sampling_weight <= 0
    ):
        raise ValueError("sampling weights must be positive")
    outputs = {
        "checkpoint": args.output_dir / f"e2_{args.mode}.best.pt",
        "summary": args.output_dir / f"e2_{args.mode}.training.json",
        "report": args.output_dir / f"E2_{args.mode.upper()}_TRAINING.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError("outputs exist; pass --force to replace: " + ", ".join(existing))
    if not args.clip_model.is_dir():
        raise FileNotFoundError(args.clip_model)
    seed_everything(args.seed)

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoProcessor, CLIPModel

    from vsight.clip_verifier import (
        ClipCandidateVerifier,
        E2BatchCollator,
        E2SelectorDataset,
        read_selector_rows,
    )

    summary = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "positive_selector_supervision_ready":
        raise ValueError("selector supervision is not ready")
    paths = {
        split: resolve_path(summary["outputs"][split]["path"])
        for split in ("train", "calibration")
    }
    for split, path in paths.items():
        if sha256(path) != summary["outputs"][split]["sha256"]:
            raise ValueError(f"selector {split} hash mismatch")
    train_rows = read_selector_rows(paths["train"])
    calibration_rows = read_selector_rows(paths["calibration"])
    eligible_train = [row for row in train_rows if row.get("selector_eligible")]
    eligible_calibration = [
        row for row in calibration_rows if row.get("selector_eligible")
    ]
    if args.max_train is not None:
        eligible_train = eligible_train[: args.max_train]
    if args.max_calibration is not None:
        eligible_calibration = eligible_calibration[: args.max_calibration]

    source_summary_path = resolve_path(
        json.loads(resolve_path(summary["candidate_manifest"]["path"]).read_text(encoding="utf-8"))[
            "queue_manifest"
        ]["path"]
    )
    queue_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    e1_source = json.loads(
        resolve_path(queue_summary["source_manifest"]["path"]).read_text(encoding="utf-8")
    )
    image_root = Path(e1_source["images"]["root"])

    processor = AutoProcessor.from_pretrained(str(args.clip_model), local_files_only=True)
    clip = CLIPModel.from_pretrained(str(args.clip_model), local_files_only=True)
    model = ClipCandidateVerifier(clip)
    model.configure_adaptation(args.mode)
    initialization = None
    if args.initialize_checkpoint is not None:
        initial = torch.load(
            args.initialize_checkpoint, map_location="cpu", weights_only=False
        )
        if initial.get("schema_version") != "vsight_e2_clip_verifier_checkpoint_v1":
            raise ValueError("unsupported initialization checkpoint")
        if Path(initial["clip_model"]).resolve() != args.clip_model.resolve():
            raise ValueError("initialization checkpoint uses a different CLIP model")
        unexpected = model.load_state_dict(initial["state_dict"], strict=False).unexpected_keys
        if unexpected:
            raise ValueError(f"unexpected initialization keys: {unexpected}")
        initialization = {
            "path": str(args.initialize_checkpoint.resolve()),
            "sha256": sha256(args.initialize_checkpoint),
        }
    device = torch.device(args.device)
    model.to(device)

    primary_train_count = len(eligible_train)
    auxiliary_rows = []
    auxiliary_provenance = None
    if args.auxiliary_manifest is not None:
        auxiliary_rows = read_selector_rows(args.auxiliary_manifest)
        if not all(row.get("training_only") for row in auxiliary_rows):
            raise ValueError("auxiliary manifest contains non-training rows")
        auxiliary_provenance = {
            "path": str(args.auxiliary_manifest.resolve()),
            "sha256": sha256(args.auxiliary_manifest),
            "records": len(auxiliary_rows),
        }
    train_dataset = E2SelectorDataset(
        [*eligible_train, *auxiliary_rows], image_root, training=True
    )
    calibration_dataset = E2SelectorDataset(
        eligible_calibration, image_root, training=False
    )
    collator = E2BatchCollator(processor)
    weights = []
    for row in train_dataset.rows:
        if row.get("supervision_kind") == "annotation_auxiliary":
            weight = args.auxiliary_sampling_weight
        else:
            weight = (
                args.switch_sampling_weight
                if str(row["selector_action"]) == "switch"
                else 1.0
            )
            if str(row.get("source_dataset")) == "RefCOCOg":
                weight *= args.refcocog_sampling_weight
        weights.append(weight)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=primary_train_count,
        replacement=True,
        generator=generator,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    calibration_loader = DataLoader(
        calibration_dataset, shuffle=False, **loader_options
    )

    clip_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("clip.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("clip.") and parameter.requires_grad
    ]
    groups = [{"params": head_parameters, "lr": args.head_lr}]
    if clip_parameters:
        groups.append({"params": clip_parameters, "lr": args.clip_lr})
    optimizer = torch.optim.AdamW(
        groups, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.head_lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    actions = Counter(str(row["selector_action"]) for row in eligible_train)
    calibration_unconditional_regressions = sum(
        float(row["baseline_iou"]) > 1e-12
        and float(row["challenger_iou"]) <= 1e-12
        for row in calibration_rows
        if row.get("selector_eligible")
    )
    regression_budget = calibration_unconditional_regressions // 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"mode={args.mode} train_primary={primary_train_count} aux={len(auxiliary_rows)} "
        f"calibration={len(calibration_dataset)} "
        f"actions={dict(actions)} regression_budget={regression_budget}",
        flush=True,
    )

    history = []
    best_key = None
    best_epoch = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        training = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args, epoch
        )
        score_differences, raw_accuracy = predict_score_differences(
            model, calibration_loader, device
        )
        threshold, metrics = choose_safe_threshold(
            calibration_rows,
            score_differences,
            max_nonzero_to_zero=regression_budget,
        )
        entry = {
            "epoch": epoch,
            "training": training,
            "calibration_raw_argmax_accuracy": raw_accuracy,
            "calibration": metrics,
        }
        history.append(entry)
        key = (
            float(metrics["selector_miou"]),
            -int(metrics["nonzero_to_zero_regressions"]),
            float(metrics["action_accuracy"] or 0.0),
        )
        print(
            f"epoch={epoch} calibration_miou={metrics['selector_miou']:.5f} "
            f"capture={metrics['oracle_gap_capture_fraction']:.3f} "
            f"regressions={metrics['nonzero_to_zero_regressions']} "
            f"switches={metrics['switches']} threshold={threshold:.5f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                outputs["checkpoint"],
                model,
                args,
                {
                    "epoch": epoch,
                    "threshold": threshold,
                    "calibration": metrics,
                    "selector_manifest_sha256": sha256(args.selector_summary),
                },
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    checkpoint_sha = sha256(outputs["checkpoint"])
    result = {
        "schema_version": "vsight_e2_verifier_training_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adaptation_mode": args.mode,
        "seed": args.seed,
        "data": {
            "selector_summary": str(args.selector_summary.resolve()),
            "selector_summary_sha256": sha256(args.selector_summary),
            "train_eligible": primary_train_count,
            "auxiliary": auxiliary_provenance,
            "calibration_eligible": len(calibration_dataset),
            "train_actions": dict(sorted(actions.items())),
            "image_root": str(image_root),
        },
        "optimization": {
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "clip_lr": args.clip_lr if args.mode == "last_block" else None,
            "switch_sampling_weight": args.switch_sampling_weight,
            "refcocog_sampling_weight": args.refcocog_sampling_weight,
            "auxiliary_sampling_weight": (
                args.auxiliary_sampling_weight if auxiliary_rows else None
            ),
            "quality_weight": args.quality_weight,
            "delta_weight": args.delta_weight,
            "safe_weight": args.safe_weight,
            "safe_margin": args.safe_margin,
            "regression_budget": regression_budget,
            "initialization_checkpoint": initialization,
        },
        "best_epoch": best_epoch,
        "best_calibration": history[best_epoch - 1]["calibration"],
        "checkpoint": {
            "path": str(outputs["checkpoint"].resolve()),
            "sha256": checkpoint_sha,
            "bytes": outputs["checkpoint"].stat().st_size,
        },
        "history": history,
    }
    outputs["summary"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    best = result["best_calibration"]
    outputs["report"].write_text(
        "\n".join(
            [
                f"# E2 {args.mode} verifier training",
                "",
                f"- Best epoch: {best_epoch}",
                f"- Calibration selector mIoU: {best['selector_miou']:.6f}",
                f"- Two-box oracle mIoU: {best['two_box_oracle_miou']:.6f}",
                f"- Oracle-gap capture: {best['oracle_gap_capture_fraction']:.3%}",
                f"- Nonzero-to-zero regressions: {best['nonzero_to_zero_regressions']}",
                f"- Switch rate: {best['switch_rate']:.3%}",
                f"- Checkpoint SHA-256: `{checkpoint_sha}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result["best_calibration"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
