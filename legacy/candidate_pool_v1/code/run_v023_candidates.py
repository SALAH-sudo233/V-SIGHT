#!/usr/bin/env python3
"""Generate recall-oriented v0.23 candidate boxes with a frozen Qwen VLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from hallucination_defense.core.candidate_recovery import (
    CANDIDATE_PROMPTS,
    CandidatePrompt,
    parse_candidate_boxes,
)


DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
DEFAULT_BENCHMARK = Path(
    "/home/u2025141034/benchmark/repaired/refcocog_500_dev.semantic_strict.json"
)
DEFAULT_IMAGE_DIR = Path("/home/u2025141034/benchmark/benchmark_images")

SYSTEM_PROMPT = (
    "You are a careful visual grounding candidate generator. Use only visible image "
    "evidence and follow the requested JSON format exactly."
)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def load_groups(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("benchmark must be a JSON list")
    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        group_id = str(row.get("base_sample_id") or row.get("sample_id") or "")
        if not group_id:
            raise ValueError("benchmark row is missing base_sample_id/sample_id")
        candidate = {
            "base_sample_id": group_id,
            "sample_id": str(row.get("sample_id") or group_id),
            "image_filename": str(row.get("image_filename") or ""),
            "query": str(row.get("chosen") or row.get("positive_text") or "").strip(),
            "gt_bbox_xyxy": row.get("gt_bbox_xyxy") or row.get("positive_bbox"),
            "source": row.get("source"),
        }
        if not candidate["image_filename"] or not candidate["query"]:
            raise ValueError(f"group {group_id} has no image/query")
        if group_id in groups:
            prior = groups[group_id]
            for key in ("image_filename", "query", "gt_bbox_xyxy"):
                if prior[key] != candidate[key]:
                    raise ValueError(f"group {group_id} has inconsistent {key}")
        else:
            groups[group_id] = candidate
    return list(groups.values())


def load_completed(path: Path) -> set[Tuple[str, str]]:
    completed: set[Tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("error"):
                completed.add(
                    (str(record.get("base_sample_id")), str(record.get("candidate_source")))
                )
    return completed


def select_prompts(names: Iterable[str]) -> List[CandidatePrompt]:
    available = {prompt.name: prompt for prompt in CANDIDATE_PROMPTS}
    selected: List[CandidatePrompt] = []
    for name in names:
        if name not in available:
            raise ValueError(f"unknown variant {name!r}; choose from {sorted(available)}")
        selected.append(available[name])
    return selected


def load_model(model_path: Path, gpu: int):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(model_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu}"},
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


def generate_one(model, processor, image_path: Any, prompt: str, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    image_value = str(image_path) if isinstance(image_path, (str, Path)) else image_path
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_value},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[chat], images=images, videos=videos, padding=True, return_tensors="pt"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[prompt.name for prompt in CANDIDATE_PROMPTS],
    )
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if not args.model.exists():
        parser.error(f"model does not exist: {args.model}")

    prompts = select_prompts(args.variants)
    groups = load_groups(args.benchmark)
    groups = [
        group for index, group in enumerate(groups) if index % args.num_shards == args.shard_index
    ]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    completed = load_completed(args.output)

    pending = [
        (group, prompt)
        for group in groups
        for prompt in prompts
        if (group["base_sample_id"], prompt.name) not in completed
    ]
    print(
        f"shard={args.shard_index}/{args.num_shards} groups={len(groups)} "
        f"pending_generations={len(pending)} gpu={args.gpu}",
        flush=True,
    )
    if not pending:
        return

    model, processor = load_model(args.model, args.gpu)
    from PIL import Image

    for index, (group, prompt_spec) in enumerate(pending, start=1):
        image_path = args.image_dir / Path(group["image_filename"]).name
        started = time.perf_counter()
        record = {
            **group,
            "query_role": "positive",
            "candidate_source": prompt_spec.name,
            "prompt": prompt_spec.render(group["query"]),
            "candidate_boxes": [],
            "raw_output_text": "",
            "parse_valid": False,
            "error": None,
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                image_size = image.size
            raw = generate_one(
                model,
                processor,
                image_path,
                record["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
            boxes = parse_candidate_boxes(
                raw, image_size=image_size, max_candidates=prompt_spec.max_candidates
            )
            record["raw_output_text"] = raw
            record["candidate_boxes"] = [list(box) for box in boxes]
            record["parse_valid"] = bool(boxes)
            record["image_size"] = list(image_size)
        except Exception as exc:  # preserve exact-record resume information
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_sec"] = time.perf_counter() - started
        append_jsonl(args.output, record)
        print(
            f"[{index}/{len(pending)}] {group['base_sample_id']} {prompt_spec.name} "
            f"boxes={len(record['candidate_boxes'])} error={record['error']}",
            flush=True,
        )


if __name__ == "__main__":
    # Avoid tokenizer parallelism noise when launching one process per GPU.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
