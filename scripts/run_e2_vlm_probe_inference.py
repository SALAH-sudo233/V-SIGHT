#!/usr/bin/env python3
"""Run a short-output Qwen pairwise verifier without exposing supervision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
SYSTEM_PROMPT = "You are a strict visual grounding verifier. Use only visible evidence."
USER_PROMPT = (
    'Referring expression: "{query}"\n'
    "The image contains two marked candidate target boxes: A is red and B is blue. "
    "Choose which box better localizes the exact target of the complete expression. "
    "Distinguish the target object from attribute or relation reference objects. "
    "Answer exactly A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "data/e1/p1/vlm_probe/e2_vlm_probe_queue.jsonl.gz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e1/p1/vlm_probe/outputs"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def read_queue(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assigned(query_id: str, shard_index: int, num_shards: int) -> bool:
    value = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:8], "big")
    return value % num_shards == shard_index


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["query_id"])
        for row in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
        if not row.get("error")
    }


def label_assignment(query_id: str) -> tuple[int, int]:
    swap = hashlib.sha256(f"vsight-vlm-label-v1:{query_id}".encode("utf-8")).digest()[0] & 1
    return (1, 0) if swap else (0, 1)


def render_pair(image, boxes, assignment):
    from PIL import ImageDraw, ImageFont

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    width = max(4, min(marked.size) // 100)
    label_size = max(28, min(marked.size) // 14)
    colors = ((230, 32, 32), (32, 100, 230))
    labels = ("A", "B")
    font = ImageFont.load_default(size=max(16, label_size // 2))
    for label_index, candidate_index in enumerate(assignment):
        x1, y1, x2, y2 = (int(round(value)) for value in boxes[candidate_index])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(marked.width - 1, x2), min(marked.height - 1, y2)
        color = colors[label_index]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        draw.rectangle((x1, y1, min(x2, x1 + label_size), min(y2, y1 + label_size)), fill=color)
        draw.text((x1 + 6, y1 + 3), labels[label_index], fill="white", font=font)
    return marked


def load_model(path: Path, gpu: int, adapter: Path | None = None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.cuda.set_device(gpu)
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu}"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, processor


def answer_token_ids(tokenizer, value: str) -> list[int]:
    ids = set()
    for text in (value, " " + value):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"no single-token encoding for {value}")
    return sorted(ids)


def logsumexp(values) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def infer_one(model, processor, image, query: str) -> tuple[str, float]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT.format(query=query)},
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
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    continuation = generated.sequences[0, inputs.input_ids.shape[1] :]
    answer = processor.decode(continuation, skip_special_tokens=True).strip()
    first_logits = generated.scores[0][0].float()
    a_score = logsumexp(
        [float(first_logits[index]) for index in answer_token_ids(processor.tokenizer, "A")]
    )
    b_score = logsumexp(
        [float(first_logits[index]) for index in answer_token_ids(processor.tokenizer, "B")]
    )
    return answer, b_score - a_score


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    rows = [
        row
        for row in read_queue(args.queue)
        if assigned(str(row["query_id"]), args.shard_index, args.num_shards)
    ]
    if args.max_records is not None:
        rows = rows[: args.max_records]
    output = args.output_dir / f"e2_vlm_probe.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    done = completed(output)
    pending = [row for row in rows if str(row["query_id"]) not in done]
    print(
        f"shard={args.shard_index}/{args.num_shards} assigned={len(rows)} "
        f"pending={len(pending)} gpu={args.gpu}",
        flush=True,
    )
    if not pending:
        return 0
    model, processor = load_model(args.model, args.gpu, args.adapter)
    from PIL import Image

    for index, row in enumerate(pending, start=1):
        started = time.perf_counter()
        assignment = label_assignment(str(row["query_id"]))
        record = {
            "schema_version": "vsight_e2_vlm_probe_output_v1",
            "query_id": row["query_id"],
            "suite": row["suite"],
            "label_assignment_candidate_indices": list(assignment),
            "candidate_1_minus_0_score": None,
            "raw_answer": None,
            "parse_valid": False,
            "error": None,
        }
        try:
            image_path = Path(row["image_root"]) / Path(row["image_filename"]).name
            with Image.open(image_path) as opened:
                marked = render_pair(opened, row["boxes_xyxy"], assignment)
            answer, b_minus_a = infer_one(model, processor, marked, str(row["query"]))
            chosen = re.search(r"(?<![A-Za-z])([AB])(?![A-Za-z])", answer.upper())
            record["raw_answer"] = answer
            record["parse_valid"] = chosen is not None
            record["candidate_1_minus_0_score"] = (
                b_minus_a if assignment == (0, 1) else -b_minus_a
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_sec"] = time.perf_counter() - started
        append_jsonl(output, record)
        if index == 1 or index % args.log_every == 0 or index == len(pending):
            print(
                f"[{index}/{len(pending)}] query={row['query_id']} "
                f"answer={record['raw_answer']} error={record['error']} "
                f"latency={record['latency_sec']:.2f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
