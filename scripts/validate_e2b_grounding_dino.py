#!/usr/bin/env python3
"""Validate local Grounding DINO against unique COCO reference boxes."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e2_verifier import box_iou  # noqa: E402


DEFAULT_MODEL = Path(
    "/home/u2025141034/.cache/huggingface/hub/"
    "models--IDEA-Research--grounding-dino-base/snapshots/"
    "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data/e2b/reference/e2b_reference_candidates.train.jsonl.gz",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "data/e2b/reference/e2b_reference_candidates.calibration.jsonl.gz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/e2b/reference/e2b_grounding_dino_validation.jsonl",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clipped_box(values, width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = (float(value) for value in values)
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def main() -> int:
    args = parse_args()
    rows = []
    for split, path in (("train", args.train), ("calibration", args.calibration)):
        rows.extend(
            {**row, "validation_split": split}
            for row in read_rows(path)
            if row["reference_status"] == "unique_reference_annotation"
        )
    rows.sort(key=lambda row: str(row["query_id"]))
    if args.max_records is not None:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("no unique reference annotations to validate")

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(args.model), local_files_only=True
    ).to(args.device).eval()
    outputs = []
    batch_latencies = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        images = []
        for row in batch:
            with Image.open(args.image_root / Path(row["image_filename"]).name) as image:
                images.append(image.convert("RGB"))
        texts = [str(row["reference_phrase"]).strip() + "." for row in batch]
        inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(
            args.device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            predictions = model(**inputs)
        torch.cuda.synchronize()
        batch_latencies.append((time.perf_counter() - started) / len(batch))
        results = processor.post_process_grounded_object_detection(
            predictions,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1] for image in images],
        )
        for row, image, result in zip(batch, images, results, strict=True):
            gt = row["reference_candidates"][0]["bbox_xyxy"]
            proposals = []
            text_labels = list(result.get("text_labels") or [])
            for proposal_index, (score, box) in enumerate(
                zip(result["scores"], result["boxes"], strict=True)
            ):
                clipped = clipped_box(box.tolist(), image.width, image.height)
                if clipped is None:
                    continue
                label = (
                    str(text_labels[proposal_index])
                    if proposal_index < len(text_labels)
                    else str(row["reference_phrase"])
                )
                proposals.append(
                    {
                        "score": float(score),
                        "bbox_xyxy": clipped,
                        "label": label,
                        "iou_to_reference": box_iou(clipped, gt),
                    }
                )
            proposals.sort(key=lambda value: -value["score"])
            outputs.append(
                {
                    "schema_version": "vsight_e2b_dino_reference_validation_v1",
                    "query_id": row["query_id"],
                    "validation_split": row["validation_split"],
                    "reference_phrase": row["reference_phrase"],
                    "reference_bbox_xyxy": gt,
                    "proposals": proposals,
                    "top_iou": proposals[0]["iou_to_reference"] if proposals else 0.0,
                    "best_iou": max(
                        (value["iou_to_reference"] for value in proposals), default=0.0
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "vsight_e2b_dino_reference_validation_summary_v1",
        "records": len(outputs),
        "proposal_coverage": sum(bool(row["proposals"]) for row in outputs) / len(outputs),
        "top_iou_at_0_5": sum(row["top_iou"] >= 0.5 for row in outputs) / len(outputs),
        "best_iou_at_0_5": sum(row["best_iou"] >= 0.5 for row in outputs) / len(outputs),
        "mean_top_iou": statistics.fmean(row["top_iou"] for row in outputs),
        "mean_best_iou": statistics.fmean(row["best_iou"] for row in outputs),
        "warm_latency_per_query_median_sec": statistics.median(batch_latencies[1:] or batch_latencies),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "external_api_required": False,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
