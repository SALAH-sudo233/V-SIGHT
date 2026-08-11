#!/usr/bin/env python3
"""Build a train-only, supervision-free binding audit queue."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import sha256  # noqa: E402
from vsight.cable import relation_family  # noqa: E402
from vsight.ccv import AtomType, TypedClaimParser  # noqa: E402
from vsight.relation_supervision import (  # noqa: E402
    RELATION_PATTERNS,
    extract_reference_phrase,
)


DEFAULT_INPUT = ROOT / "data/e1/p1/selector/e1_p1_selector.train.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "data/e3/binding_annotation_queue"
ATTRIBUTE_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray",
    "grey", "orange", "pink", "purple", "wooden", "metal", "plastic",
    "striped", "small", "large", "big", "young", "old",
}
SENSITIVE_ATTRIBUTE_WORDS = {
    "young", "old", "male", "female", "man", "men", "woman", "women",
    "boy", "boys", "girl", "girls", "lady", "child", "children",
}
FORBIDDEN_FIELDS = {
    "ann_id",
    "baseline_iou",
    "challenger_bbox_xyxy",
    "challenger_iou",
    "gt_bbox_xyxy",
    "raw_two_box_oracle_iou",
    "selector_action",
    "selector_eligible",
    "switch_iou_margin",
    "two_box_oracle_iou",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--relation-count", type=int, default=200)
    parser.add_argument("--attribute-count", type=int, default=150)
    parser.add_argument("--object-count", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def deterministic_key(group_id: str) -> str:
    return hashlib.sha256(f"vsight-e3-binding-queue-v1:{group_id}".encode()).hexdigest()


def stratum(query: str) -> str:
    text = query.casefold()
    if any(re.search(pattern, text) for _, pattern in RELATION_PATTERNS):
        return "relation"
    if any(word in text.split() for word in ATTRIBUTE_WORDS):
        return "attribute"
    return "object"


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = parse_args()
    if min(args.limit, args.relation_count, args.attribute_count, args.object_count) < 0:
        raise ValueError("queue counts must be non-negative")
    requested = args.relation_count + args.attribute_count + args.object_count
    if requested != args.limit:
        raise ValueError("stratum counts must sum to --limit")
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_path = output_dir / "e3_binding_annotation_queue.train.jsonl.gz"
    summary_path = output_dir / "e3_binding_annotation_queue.summary.json"
    if not args.force and (output_path.exists() or summary_path.exists()):
        raise FileExistsError("binding annotation queue exists; pass --force")

    source_rows = read_gzip(input_path)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    seen_groups: set[str] = set()
    for source in source_rows:
        if source.get("data_split") != "train":
            raise ValueError("binding queue source must be train-only")
        group_id = str(source.get("group_id") or "")
        query = str(source.get("query") or "").strip()
        box = source.get("baseline_bbox_xyxy")
        if not group_id or not query or not isinstance(box, list) or len(box) != 4:
            continue
        if group_id in seen_groups:
            raise ValueError(f"duplicate image group in source: {group_id}")
        seen_groups.add(group_id)
        by_stratum[stratum(query)].append(source)

    counts = {
        "relation": args.relation_count,
        "attribute": args.attribute_count,
        "object": args.object_count,
    }
    selected: list[dict] = []
    for kind in ("relation", "attribute", "object"):
        candidates = sorted(
            by_stratum[kind], key=lambda row: deterministic_key(str(row["group_id"]))
        )
        if len(candidates) < counts[kind]:
            raise ValueError(f"not enough {kind} rows: {len(candidates)}")
        selected.extend(candidates[: counts[kind]])
    selected.sort(key=lambda row: str(row["group_id"]))

    rows = []
    parser = TypedClaimParser()
    for index, source in enumerate(selected):
        query = str(source["query"])
        relation = next(
            (name for name, pattern in RELATION_PATTERNS if re.search(pattern, query.casefold())),
            None,
        )
        reference = extract_reference_phrase(query)
        parsed = parser.parse(query)
        applicable_atoms = ["identity"]
        if parsed.atoms_of_type(AtomType.ATTRIBUTE):
            applicable_atoms.append("attribute")
        if parsed.atoms_of_type(AtomType.ACTION):
            applicable_atoms.append("action")
        if parsed.atoms_of_type(AtomType.RELATION):
            applicable_atoms.append("relation")
        row = {
            "schema_version": "vsight_e3_binding_annotation_input_v2",
            "annotation_id": f"e3-binding-{index:04d}",
            "data_split": "train",
            "group_id": str(source["group_id"]),
            "image_id": int(source["image_id"]),
            "image_filename": str(source["image_filename"]),
            "query": query,
            "original_bbox_xyxy": [float(value) for value in source["baseline_bbox_xyxy"]],
            "query_stratum": stratum(query),
            "parsed_relation": relation,
            "relation_family": relation_family(relation),
            "reference_phrase": reference,
            "sensitive_attribute": bool(
                set(re.findall(r"[a-z]+", query.casefold())) & SENSITIVE_ATTRIBUTE_WORDS
            ),
            "stage_a_target_status": None,
            "stage_b_applicable_atoms": applicable_atoms,
            "stage_b_atoms": {
                "identity": None,
                "attribute": None,
                "action": None,
                "relation": None,
            },
            "stage_b_note": None,
            "stage_c_action": None,
            "corrected_bbox_xyxy": None,
            "reference_bbox_xyxy": None,
            "reference_visibility": None,
            "confidence": None,
            "reviewer_ids": [],
            "review_status": "pending",
        }
        leaked = FORBIDDEN_FIELDS & set(row)
        if leaked:
            raise ValueError(f"annotation input leaks supervision: {sorted(leaked)}")
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": "vsight_e3_binding_annotation_queue_v2",
        "status": "human_review_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "train_annotation_queue",
        "source": {"path": str(input_path), "sha256": sha256(input_path)},
        "statistics": {
            "source_rows": len(source_rows),
            "source_unique_groups": len(seen_groups),
            "queued_rows": len(rows),
            "queued_unique_images": len({row["image_filename"] for row in rows}),
            "strata": dict(sorted(Counter(row["query_stratum"] for row in rows).items())),
        },
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "outputs": {
            "train": {
                "path": str(output_path.relative_to(ROOT)),
                "sha256": sha256(output_path),
                "bytes": output_path.stat().st_size,
            }
        },
        "requires_external_api": False,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
