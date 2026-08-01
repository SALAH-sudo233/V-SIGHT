#!/usr/bin/env python3
"""Strip E2b reference supervision into a Grounding DINO inference queue."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402


FORBIDDEN_FIELDS = {
    "target_ann_id",
    "target_bbox_xyxy",
    "reference_candidates",
    "reference_status",
    "verifier_training_eligible",
    "target_category_id",
    "target_category_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=ROOT / "data/e2b/reference/e2b_reference_candidates.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e2b/reference_queue")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_gzip(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    outputs = {
        split: args.output_dir / f"e2b_reference_queue.{split}.jsonl.gz"
        for split in reference["outputs"]
    }
    summary_path = args.output_dir / "e2b_reference_queue.summary.json"
    if not args.force and any(path.exists() for path in (*outputs.values(), summary_path)):
        raise FileExistsError("reference queue exists; pass --force to replace")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, item in reference["outputs"].items():
        source_path = resolve_path(item["path"])
        if sha256(source_path) != item["sha256"]:
            raise ValueError(f"reference manifest hash mismatch: {source_path}")
        rows = []
        relation_counts = Counter()
        for source in read_gzip(source_path):
            if not source.get("relation") or not source.get("reference_phrase"):
                continue
            row = {
                "schema_version": "vsight_e2b_reference_inference_input_v1",
                "query_id": source["query_id"],
                "data_split": split,
                "image_id": source["image_id"],
                "image_filename": source["image_filename"],
                "query": source["query"],
                "relation": source["relation"],
                "reference_phrase": source["reference_phrase"],
            }
            if FORBIDDEN_FIELDS & set(row):
                raise ValueError("reference queue leaks supervision")
            rows.append(row)
            relation_counts[str(row["relation"])] += 1
        rows.sort(key=lambda row: str(row["query_id"]))
        temporary = outputs[split].with_name(outputs[split].name + ".tmp")
        try:
            with deterministic_gzip(temporary) as handle:
                for row in rows:
                    handle.write(
                        json.dumps(
                            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
            temporary.replace(outputs[split])
        finally:
            temporary.unlink(missing_ok=True)
        stats[split] = {
            "records": len(rows),
            "unique_images": len({row["image_id"] for row in rows}),
            "relations": dict(sorted(relation_counts.items())),
        }
    summary = {
        "schema_version": "vsight_e2b_reference_queue_manifest_v1",
        "status": "grounding_dino_inference_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.reference_summary), "sha256": sha256(args.reference_summary)},
        "statistics": stats,
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "outputs": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for split, path in outputs.items()
        },
        "external_api_required": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
