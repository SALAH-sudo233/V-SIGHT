#!/usr/bin/env python3
"""LoRA-adapt Qwen2.5-VL to compare two randomly labeled grounding boxes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_e2_vlm_probe_inference import (  # noqa: E402
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    USER_PROMPT,
    render_pair,
)
from vsight.e1_data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-summary",
        type=Path,
        default=ROOT / "data/e1/p1/selector/e1_p1_selector.summary.json",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/e2_vlm_lora"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_assignment(query_id: str, epoch: int) -> tuple[int, int]:
    value = hashlib.sha256(
        f"vsight-vlm-lora-label-v1:{epoch}:{query_id}".encode("utf-8")
    ).digest()[0] & 1
    return (1, 0) if value else (0, 1)


class PairwiseSftDataset:
    def __init__(self, rows, image_root: Path, processor) -> None:
        self.rows = list(rows)
        self.image_root = image_root
        self.processor = processor
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        row = self.rows[index]
        boxes = [row["baseline_bbox_xyxy"], row["challenger_bbox_xyxy"]]
        assignment = training_assignment(str(row["query_id"]), self.epoch)
        target_candidate = 1 if str(row["selector_action"]) == "switch" else 0
        target_label = "A" if assignment[0] == target_candidate else "B"
        image_path = self.image_root / Path(row["image_filename"]).name
        with Image.open(image_path) as opened:
            marked = render_pair(opened, boxes, assignment)
        user_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": marked},
                    {
                        "type": "text",
                        "text": USER_PROMPT.format(query=str(row["query"])),
                    },
                ],
            },
        ]
        full_messages = [
            *user_messages,
            {"role": "assistant", "content": target_label},
        ]
        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        images, videos = process_vision_info(user_messages)
        prompt = self.processor(
            text=[prompt_text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        full = self.processor(
            text=[full_text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100
        full["labels"] = labels
        return dict(full)


def initialize_distributed():
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return world_size, local_rank, torch.device(f"cuda:{local_rank}")


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs and gradient accumulation must be positive")
    world_size, local_rank, device = initialize_distributed()
    rank = int(os.environ.get("RANK", "0"))
    adapter_dir = args.output_dir / "adapter"
    summary_path = args.output_dir / "training.json"
    if rank == 0 and (adapter_dir.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError("LoRA output exists; pass --force to replace")

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )

    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    selector = json.loads(args.selector_summary.read_text(encoding="utf-8"))
    train_path = resolve_path(selector["outputs"]["train"]["path"])
    if sha256(train_path) != selector["outputs"]["train"]["sha256"]:
        raise ValueError("training selector hash mismatch")
    rows = [row for row in read_gzip(train_path) if row.get("selector_eligible")]
    if args.max_records is not None:
        rows = rows[: args.max_records]

    candidate_summary = json.loads(
        resolve_path(selector["candidate_manifest"]["path"]).read_text(encoding="utf-8")
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
    image_root = Path(source_summary["images"]["root"])

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if rank == 0:
        model.print_trainable_parameters()
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if world_size > 1
        else model
    )
    dataset = PairwiseSftDataset(rows, image_root, processor)
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=lambda samples: samples[0],
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = max(
        1, (len(loader) + args.gradient_accumulation - 1) // args.gradient_accumulation
    )
    total_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps,
    )
    optimizer.zero_grad(set_to_none=True)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        loss_sum = 0.0
        example_count = 0
        optimizer_steps = 0
        for step, original in enumerate(loader, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in original.items()
            }
            output = wrapped(**batch)
            loss = output.loss / args.gradient_accumulation
            loss.backward()
            loss_sum += float(output.loss.detach())
            example_count += 1
            boundary = step % args.gradient_accumulation == 0 or step == len(loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if rank == 0 and (
                    optimizer_steps == 1
                    or optimizer_steps % args.log_every == 0
                    or step == len(loader)
                ):
                    print(
                        f"epoch={epoch+1} optimizer_step={optimizer_steps}/"
                        f"{optimizer_steps_per_epoch} loss={loss_sum/example_count:.4f}",
                        flush=True,
                    )
        if world_size > 1:
            totals = torch.tensor(
                [loss_sum, example_count], dtype=torch.float64, device=device
            )
            dist.all_reduce(totals)
            epoch_loss = float(totals[0] / totals[1])
        else:
            epoch_loss = loss_sum / example_count
        history.append({"epoch": epoch + 1, "loss": epoch_loss})

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = wrapped.module if world_size > 1 else wrapped
        unwrapped.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        adapter_model = adapter_dir / "adapter_model.safetensors"
        summary = {
            "schema_version": "vsight_e2_vlm_lora_training_v1",
            "status": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_model": str(args.model.resolve()),
            "selector_manifest": {
                "path": str(args.selector_summary.resolve()),
                "sha256": sha256(args.selector_summary),
            },
            "train_records": len(rows),
            "world_size": world_size,
            "epochs": args.epochs,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": world_size * args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "lora": {
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "history": history,
            "duration_sec": time.perf_counter() - started,
            "adapter": {
                "path": str(adapter_dir.resolve()),
                "model_sha256": sha256(adapter_model),
                "bytes": adapter_model.stat().st_size,
            },
            "sealed_heldout_accessed": False,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
