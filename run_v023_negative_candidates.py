#!/usr/bin/env python3
"""Generate binding-aware candidates independently for all repaired negatives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping

from hallucination_defense.core.candidate_recovery import CANDIDATE_PROMPTS, parse_candidate_boxes
from run_v023_candidates import (
    DEFAULT_BENCHMARK,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MODEL,
    generate_one,
    load_model,
)
from run_v023_oracle import read_jsonl


BINDING_PROMPT = next(prompt for prompt in CANDIDATE_PROMPTS if prompt.name == "binding_aware")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def load_negative_examples(path: Path) -> list[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    seen = set()
    for row in rows:
        pair_id = str(row.get("pair_id") or row.get("sample_id") or "")
        if not pair_id or pair_id in seen:
            raise ValueError(f"missing or duplicate pair_id: {pair_id!r}")
        seen.add(pair_id)
        examples.append(
            {
                "pair_id": pair_id,
                "base_sample_id": str(row.get("base_sample_id") or row.get("sample_id")),
                "sample_id": row.get("sample_id"),
                "image_filename": row.get("image_filename"),
                "query": str(row.get("rejected") or row.get("negative_text") or "").strip(),
                "query_role": "negative",
                "hallucination_type": row.get("hallucination_type"),
            }
        )
    return examples


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record.get("pair_id"))
        for record in read_jsonl(path)
        if not record.get("error")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    examples = load_negative_examples(args.benchmark)
    examples = [
        example for index, example in enumerate(examples)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    completed = load_completed(args.output)
    examples = [example for example in examples if example["pair_id"] not in completed]
    print(
        f"shard={args.shard_index}/{args.num_shards} pending={len(examples)} gpu={args.gpu}",
        flush=True,
    )
    if not examples:
        return

    model, processor = load_model(args.model, args.gpu)
    from PIL import Image

    for index, example in enumerate(examples, start=1):
        started = time.perf_counter()
        result = {
            **example,
            "candidate_source": "binding_aware",
            "candidate_boxes": [],
            "raw_output_text": "",
            "parse_valid": False,
            "error": None,
        }
        try:
            image_path = args.image_dir / Path(example["image_filename"]).name
            with Image.open(image_path) as image:
                image_size = image.size
            raw = generate_one(
                model,
                processor,
                image_path,
                BINDING_PROMPT.render(example["query"]),
                max_new_tokens=128,
            )
            boxes = parse_candidate_boxes(raw, image_size=image_size, max_candidates=1)
            result["raw_output_text"] = raw
            result["candidate_boxes"] = [list(box) for box in boxes]
            result["parse_valid"] = bool(boxes)
            result["image_size"] = list(image_size)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["latency_sec"] = time.perf_counter() - started
        append_jsonl(args.output, result)
        print(
            f"[{index}/{len(examples)}] {example['pair_id']} "
            f"boxes={len(result['candidate_boxes'])} error={result['error']}",
            flush=True,
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
