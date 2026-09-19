#!/usr/bin/env python3
"""Apply reviewed, deterministic fixes to two RefCOCOg-500 relation pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIXES = {
    "hallu_000008_COCO_train2014_000000500224::relation": {
        "negative_text": "the catcher behind the blue uniform",
        "positive_relations": [
            {"subject": "catcher", "relation": "in", "object": "uniform"},
        ],
    },
    "hallu_000720_COCO_train2014_000000265114::attribute": {
        "negative_text": "the person holding the plastic object",
        "positive_attributes": [
            {"object": "object", "attribute": "metal"},
        ],
    },
    "hallu_000617_COCO_train2014_000000559647::co_occurrence": {
        "negative_text": "the person sliding into the base while holding a bat",
    },
    "hallu_000717_COCO_train2014_000000352073::relation": {
        "negative_text": "the person in the yellow bikini next to the girl with the phone",
    },
    "hallu_000717_COCO_train2014_000000352073::co_occurrence": {
        "negative_text": "the person in the yellow bikini wearing a life jacket",
    },
    "hallu_000836_COCO_train2014_000000411738::relation": {
        "negative_text": "the bed with the dog under it",
        "negative_relations": [
            {"subject": "dog", "relation": "under", "object": "bed"},
        ],
        "hallucination_units": [
            {
                "type": "relation",
                "object": "",
                "attribute": "",
                "subject": "dog",
                "relation": "under",
                "target_object": "bed",
                "count_as_hallucination_if_mentioned": True,
                "note": "The dog is visibly on top of the bed, not underneath it.",
            }
        ],
    },
    "hallu_000588_COCO_train2014_000000025353::relation": {
        "negative_text": "the person standing while holding the snowboard",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    applied = set()
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        fix = FIXES.get(pair_id)
        if not fix:
            continue
        row["negative_text"] = fix["negative_text"]
        row["rejected"] = fix["negative_text"]
        target = (row.get("chair_annotation") or {}).get("target_eval_units") or {}
        if "positive_attributes" in fix:
            target["positive_attributes"] = fix["positive_attributes"]
        if "positive_relations" in fix:
            target["positive_relations"] = fix["positive_relations"]
        if "negative_relations" in fix:
            target["negative_relations"] = fix["negative_relations"]
        if "hallucination_units" in fix:
            target["hallucination_units"] = fix["hallucination_units"]
        row.setdefault("pair_metadata", {})["semantic_edge_fix"] = "human_reviewed_v1"
        applied.add(pair_id)

    missing = set(FIXES) - applied
    if missing:
        raise RuntimeError(f"Fix targets not found: {sorted(missing)}")
    temporary = args.dataset.with_suffix(args.dataset.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.dataset)
    print(json.dumps({"fixed_pair_ids": sorted(applied)}, indent=2))


if __name__ == "__main__":
    main()
