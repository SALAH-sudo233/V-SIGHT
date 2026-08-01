#!/usr/bin/env python3
"""Generate at most five open-vocabulary reference proposals per E2b query."""

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

from vsight.e1_data import sha256  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--IDEA-Research--grounding-dino-base/snapshots/"
    "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=ROOT / "data/e2b/reference_queue/e2b_reference_queue.summary.json",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e2b/reference_proposals"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--max-proposals", type=int, default=5)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assigned(query_id: str, index: int, count: int) -> bool:
    value = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:8], "big")
    return value % count == index


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["query_id"])
        for row in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
        if not row.get("error")
    }


def clip_box(values, width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = (float(value) for value in values)
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    queue = json.loads(args.queue_summary.read_text(encoding="utf-8"))
    rows = []
    for item in queue["outputs"].values():
        path = resolve_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"reference queue hash mismatch: {path}")
        rows.extend(read_gzip(path))
    rows = [
        row
        for row in rows
        if assigned(str(row["query_id"]), args.shard_index, args.num_shards)
    ]
    if args.max_records is not None:
        rows = rows[: args.max_records]
    output = args.output_dir / (
        f"e2b_reference_dino.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    )
    done = completed(output)
    pending = [row for row in rows if str(row["query_id"]) not in done]
    print(
        f"shard={args.shard_index}/{args.num_shards} assigned={len(rows)} "
        f"pending={len(pending)} gpu={args.gpu}",
        flush=True,
    )
    if not pending:
        return 0

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(args.model), local_files_only=True
    ).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, start=1):
            started = time.perf_counter()
            record = {
                "schema_version": "vsight_e2b_reference_dino_output_v1",
                "query_id": row["query_id"],
                "data_split": row["data_split"],
                "relation": row["relation"],
                "reference_phrase": row["reference_phrase"],
                "proposals": [],
                "error": None,
            }
            try:
                with Image.open(
                    args.image_root / Path(row["image_filename"]).name
                ) as opened:
                    image = opened.convert("RGB")
                inputs = processor(
                    images=image,
                    text=str(row["reference_phrase"]).strip() + ".",
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    prediction = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    prediction,
                    inputs.input_ids,
                    threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    target_sizes=[image.size[::-1]],
                )[0]
                labels = list(result.get("text_labels") or [])
                proposals = []
                for proposal_index, (score, box) in enumerate(
                    zip(result["scores"], result["boxes"], strict=True)
                ):
                    clipped = clip_box(box.tolist(), image.width, image.height)
                    if clipped is None:
                        continue
                    proposals.append(
                        {
                            "score": float(score),
                            "bbox_xyxy": clipped,
                            "label": (
                                str(labels[proposal_index])
                                if proposal_index < len(labels)
                                else str(row["reference_phrase"])
                            ),
                        }
                    )
                proposals.sort(key=lambda value: -value["score"])
                record["proposals"] = proposals[: args.max_proposals]
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_sec"] = time.perf_counter() - started
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            if index == 1 or index % args.log_every == 0 or index == len(pending):
                print(
                    f"[{index}/{len(pending)}] query={row['query_id']} "
                    f"proposals={len(record['proposals'])} error={record['error']} "
                    f"latency={record['latency_sec']:.2f}s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
