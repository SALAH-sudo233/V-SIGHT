#!/usr/bin/env python3
"""Generate one local baseline and one binding-aware challenger per P1 query."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.candidate_generation import (  # noqa: E402
    BASELINE_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    CHALLENGER_PROMPT,
    CHALLENGER_SYSTEM_PROMPT,
    generation_spec,
    parse_baseline_output,
    parse_challenger_output,
)
from vsight.e1_data import sha256  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e1/p1/e1_p1_queries.summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e1/p1/candidates"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "calibration"), default=("train", "calibration")
    )
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_queue(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def completed_queries(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error"):
                completed.add(str(row["query_id"]))
    return completed


def assigned_to_shard(query_id: str, shard_index: int, num_shards: int) -> bool:
    value = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:8], "big")
    return value % num_shards == shard_index


def load_model(model_path: Path, gpu: int):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.cuda.set_device(gpu)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu}"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    return model, processor


def generate_one(
    model,
    processor,
    image_path: Path,
    system_prompt: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    continuation = generated[0, inputs.input_ids.shape[1] :]
    return processor.decode(continuation, skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    queue_summary = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    source_summary_path = resolve_path(queue_summary["source_manifest"]["path"])
    if sha256(source_summary_path) != queue_summary["source_manifest"]["sha256"]:
        raise ValueError("source manifest hash differs from the P1 queue freeze")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    image_root = Path(source_summary["images"]["root"])
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    rows = []
    for split in args.splits:
        item = queue_summary["outputs"][split]
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"queue shard hash mismatch: {path}")
        rows.extend(read_queue(path))
    rows = [
        row
        for row in rows
        if assigned_to_shard(str(row["query_id"]), args.shard_index, args.num_shards)
    ]
    if args.max_queries is not None:
        rows = rows[: args.max_queries]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"e1_p1_candidates.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    )
    completed = completed_queries(output)
    pending = [row for row in rows if str(row["query_id"]) not in completed]
    print(
        f"shard={args.shard_index}/{args.num_shards} assigned={len(rows)} "
        f"completed={len(completed)} pending={len(pending)} gpu={args.gpu}",
        flush=True,
    )
    if not pending:
        return 0

    model, processor = load_model(args.model, args.gpu)
    spec = generation_spec()
    spec["model_key"] = "qwen2.5-vl-7b-instruct"
    spec["checkpoint_revision"] = args.model.resolve().name
    spec["max_new_tokens"] = args.max_new_tokens
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    for index, row in enumerate(pending, start=1):
        started = time.perf_counter()
        image_path = image_root / Path(row["image_filename"]).name
        record = {
            "schema_version": "vsight_e1_p1_candidate_output_v1",
            "query_id": row["query_id"],
            "group_id": row["group_id"],
            "data_split": row["data_split"],
            "image_id": row["image_id"],
            "image_filename": row["image_filename"],
            "query": row["query"],
            "generator_spec_sha256": spec_hash,
            "generator_spec": spec,
            "baseline": None,
            "challenger": None,
            "error": None,
        }
        try:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            baseline_started = time.perf_counter()
            baseline_raw = generate_one(
                model,
                processor,
                image_path,
                BASELINE_SYSTEM_PROMPT,
                BASELINE_PROMPT.format(expr=row["query"]),
                args.max_new_tokens,
            )
            baseline = parse_baseline_output(
                baseline_raw, (int(row["image_width"]), int(row["image_height"]))
            )
            baseline.update(
                {
                    "raw_output_text": baseline_raw,
                    "latency_sec": time.perf_counter() - baseline_started,
                }
            )
            record["baseline"] = baseline

            challenger_started = time.perf_counter()
            challenger_raw = generate_one(
                model,
                processor,
                image_path,
                CHALLENGER_SYSTEM_PROMPT,
                CHALLENGER_PROMPT.format(expr=row["query"]),
                args.max_new_tokens,
            )
            challenger = parse_challenger_output(
                challenger_raw, (int(row["image_width"]), int(row["image_height"]))
            )
            challenger.update(
                {
                    "raw_output_text": challenger_raw,
                    "latency_sec": time.perf_counter() - challenger_started,
                }
            )
            record["challenger"] = challenger
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_sec"] = time.perf_counter() - started
        append_jsonl(output, record)
        if index == 1 or index % args.log_every == 0 or index == len(pending):
            baseline_valid = bool((record["baseline"] or {}).get("parse_valid"))
            challenger_valid = bool((record["challenger"] or {}).get("parse_valid"))
            print(
                f"[{index}/{len(pending)}] query={row['query_id']} "
                f"baseline={baseline_valid} challenger={challenger_valid} "
                f"error={record['error']} latency={record['latency_sec']:.2f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
