#!/usr/bin/env python3
"""Build label-free head/full target evidence inputs for repaired-500."""

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
from vsight.single_pass import semantic_query_id, target_phrases  # noqa: E402


DEFAULT_BENCHMARK = (
    ROOT / "legacy/candidate_pool_v1/data/refcocog_500_dev.semantic_strict.json"
)
FORBIDDEN_FIELDS = {
    "query_role",
    "hallucination_type",
    "label_exists",
    "gt_bbox_xyxy",
    "positive_bbox",
    "chosen_bbox_xyxy",
    "selector_action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/e3/singlepass/target_queue",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def base_id(row: dict) -> str:
    value = str(row.get("base_sample_id") or row.get("sample_id") or "")
    return value.split("__", 1)[0]


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = parse_args()
    output = args.output_dir / "e3_singlepass_target_queue.development.jsonl.gz"
    summary_path = args.output_dir / "e3_singlepass_target_queue.summary.json"
    if not args.force and (output.exists() or summary_path.exists()):
        raise FileExistsError("single-pass target queue exists; pass --force")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    semantic = {}
    for row in benchmark:
        image = str(row["image_filename"])
        positive = str(row.get("chosen") or row.get("positive_text") or "").strip()
        negative = str(row.get("rejected") or row.get("negative_text") or "").strip()
        for query in (positive, negative):
            query_id = semantic_query_id(image, query)
            current = {"image_filename": image, "query": query}
            if query_id in semantic and semantic[query_id] != current:
                raise ValueError(f"semantic query hash collision: {query_id}")
            semantic[query_id] = current
    if len(semantic) != 2500:
        raise ValueError(f"expected 2,500 semantic queries, found {len(semantic)}")

    rows = []
    view_counts = Counter()
    missing_head = 0
    for semantic_id, source in sorted(semantic.items()):
        phrases = target_phrases(source["query"])
        if not phrases["head"]:
            missing_head += 1
        emitted = set()
        for view in ("head", "full", "reference"):
            phrase = phrases[view]
            normalized = str(phrase or "").strip().casefold()
            if not normalized or normalized in emitted:
                continue
            emitted.add(normalized)
            row = {
                "schema_version": "vsight_e3_singlepass_target_input_v1",
                "query_id": f"{semantic_id}:{view}",
                "semantic_query_id": semantic_id,
                "evidence_view": view,
                "data_split": "development",
                "image_id": base_id({"sample_id": source["image_filename"]}),
                "image_filename": source["image_filename"],
                "query": source["query"],
                "relation": f"target_{view}",
                "reference_phrase": phrase,
            }
            leaked = FORBIDDEN_FIELDS & set(row)
            if leaked:
                raise ValueError(f"target queue leaks supervision: {sorted(leaked)}")
            rows.append(row)
            view_counts[view] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": "vsight_e3_singlepass_target_queue_v1",
        "status": "grounding_dino_inference_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "development_not_final_test",
        "benchmark": {"path": str(args.benchmark), "sha256": sha256(args.benchmark)},
        "statistics": {
            "source_rows": len(benchmark),
            "semantic_queries": len(semantic),
            "inference_rows": len(rows),
            "missing_coco_head": missing_head,
            "views": dict(sorted(view_counts.items())),
        },
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "outputs": {
            "development": {
                "path": str(output.relative_to(ROOT)),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
        },
        "external_api_required": False,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
