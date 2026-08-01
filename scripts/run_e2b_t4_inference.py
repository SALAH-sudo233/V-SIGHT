#!/usr/bin/env python3
"""Generate the canonical T4 baseline for the frozen E2b RefCOCOg queue."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_e1_p1_candidate_inference import (  # noqa: E402
    DEFAULT_MODEL,
    append_jsonl,
    completed_queries,
    generate_one,
    load_model,
)
from vsight.candidate_generation import (  # noqa: E402
    BASELINE_SYSTEM_PROMPT,
    T4_PROMPT,
    parse_t4_output,
    t4_generation_spec,
)
from vsight.e1_data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e2b/queue/e2b_refcocog.summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e2b/t4_candidates"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assigned(query_id: str, shard_index: int, num_shards: int) -> bool:
    value = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:8], "big")
    return value % num_shards == shard_index


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    queue = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    rows = []
    for item in queue["outputs"].values():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"E2b queue hash mismatch: {path}")
        rows.extend(read_gzip(path))
    rows = [
        row
        for row in rows
        if assigned(str(row["query_id"]), args.shard_index, args.num_shards)
    ]
    if args.max_records is not None:
        rows = rows[: args.max_records]

    p1_summary = json.loads(resolve_path(queue["p1_queue"]["path"]).read_text(encoding="utf-8"))
    source_summary = json.loads(
        resolve_path(p1_summary["source_manifest"]["path"]).read_text(encoding="utf-8")
    )
    image_root = Path(source_summary["images"]["root"])
    output = args.output_dir / (
        f"e2b_t4.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    )
    done = completed_queries(output)
    pending = [row for row in rows if str(row["query_id"]) not in done]
    print(
        f"shard={args.shard_index}/{args.num_shards} assigned={len(rows)} "
        f"pending={len(pending)} gpu={args.gpu}",
        flush=True,
    )
    if not pending:
        return 0
    model, processor = load_model(args.model, args.gpu)
    spec = t4_generation_spec()
    spec.update(
        {
            "model_key": "qwen2.5-vl-7b-instruct",
            "checkpoint_revision": args.model.resolve().name,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for index, row in enumerate(pending, start=1):
        started = time.perf_counter()
        record = {
            "schema_version": "vsight_e2b_t4_output_v1",
            "query_id": row["query_id"],
            "data_split": row["data_split"],
            "image_id": row["image_id"],
            "image_filename": row["image_filename"],
            "query": row["query"],
            "generator_spec_sha256": spec_hash,
            "generator_spec": spec,
            "t4": None,
            "error": None,
        }
        try:
            image_path = image_root / Path(row["image_filename"]).name
            raw = generate_one(
                model,
                processor,
                image_path,
                BASELINE_SYSTEM_PROMPT,
                T4_PROMPT.format(expr=row["query"]),
                args.max_new_tokens,
            )
            parsed = parse_t4_output(
                raw, (int(row["image_width"]), int(row["image_height"]))
            )
            parsed["raw_output_text"] = raw
            record["t4"] = parsed
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_sec"] = time.perf_counter() - started
        append_jsonl(output, record)
        if index == 1 or index % args.log_every == 0 or index == len(pending):
            print(
                f"[{index}/{len(pending)}] query={row['query_id']} "
                f"valid={bool((record['t4'] or {}).get('parse_valid'))} "
                f"error={record['error']} latency={record['latency_sec']:.2f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
