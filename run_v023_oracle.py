#!/usr/bin/env python3
"""Measure the candidate-pool upper bound before training-free reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hallucination_defense.core.candidate_recovery import evaluate_record_oracle


DEFAULT_RECORDS = Path(
    "/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/"
    "run_500_semantic_strict/qwen2.5-vl-7b/records.jsonl"
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def _positive_key(record: Dict[str, Any]) -> tuple[str, str]:
    """Return a stable join key for external records and repaired canonical rows."""
    sample_id = str(record.get("base_sample_id") or record.get("sample_id") or "")
    # v0.23 candidate jobs use ``<base>__obj``/``__attr`` suffixes while the
    # repaired canonical rows store the unsuffixed base id.
    if "__" in sample_id:
        sample_id = sample_id.split("__", 1)[0]
    return (
        sample_id,
        " ".join(str(record.get("query") or record.get("positive_text") or "").lower().split()),
    )


def enrich_external_candidates(
    records: Iterable[Dict[str, Any]], canonical_positive: Dict[tuple[str, str], Dict[str, Any]]
) -> Iterable[Dict[str, Any]]:
    """Attach GT metadata to legacy model outputs before oracle analysis.

    Historical model records intentionally contain predictions only and omit
    benchmark annotations.  They are still useful as frozen candidate sources,
    but must be joined to the repaired canonical positive by exact sample/query
    key.  Rows without a valid box are skipped instead of creating empty groups.
    """

    for original in records:
        if original.get("query_role") != "positive":
            continue
        if original.get("task") and original.get("task") not in {
            "t2_vqa_grounding", "t4_caption_grounding"
        }:
            continue
        has_box = bool(original.get("pred_bbox_xyxy")) or bool(original.get("candidate_boxes"))
        if not has_box:
            continue
        key = _positive_key(original)
        canonical = canonical_positive.get(key)
        if canonical is None:
            continue
        record = dict(original)
        record["base_sample_id"] = canonical.get("base_sample_id") or canonical.get("sample_id")
        record["gt_bbox_xyxy"] = canonical.get("gt_bbox_xyxy")
        record["candidate_source"] = (
            str(original.get("candidate_source") or original.get("model") or "external")
        )
        yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["t2_vqa_grounding", "t4_caption_grounding"],
        help="Canonical task names to use as candidate sources.",
    )
    parser.add_argument("--baseline", default="t2_vqa_grounding")
    parser.add_argument(
        "--candidate-records",
        type=Path,
        nargs="*",
        default=[],
        help="Optional v0.23 JSONL files containing candidate_source/candidate_boxes.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    canonical_rows = list(read_jsonl(args.records))
    canonical_positive = {
        _positive_key(record): record
        for record in canonical_rows
        if record.get("task") in args.tasks and record.get("query_role") == "positive"
    }
    for record in canonical_rows:
        if record.get("task") in args.tasks and record.get("query_role") == "positive":
            records.append(record)

    extra_sources: List[str] = []
    for path in args.candidate_records:
        for record in enrich_external_candidates(read_jsonl(path), canonical_positive):
            source = str(record.get("candidate_source") or record.get("task") or "")
            if source and source not in extra_sources:
                extra_sources.append(source)
            records.append(record)

    sources = list(args.tasks) + [source for source in extra_sources if source not in args.tasks]
    summary = evaluate_record_oracle(records, sources, args.baseline)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
