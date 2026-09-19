"""Rebuild the auto_repair read-only pair index from verified remote source copies.

The source JSON files are never modified. This script only derives the compact
index required by auto_repair/runner.py and writes provenance with SHA256s.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(r"C:/Users/30796/AppData/Local/Temp/grpo")
INDEX = OUT / "pairs_dump.json"
ROOT = Path(r"C:/Users/30796/Desktop/V-SIGHT-assets/data_quality_audit")
SRC_1996 = OUT / "refcocog_1996_heldout.manual_v2.remote.json"
SRC_500 = OUT / "refcocog_500_dev.remote.json"
MANIFEST = ROOT / "pairs_dump_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise TypeError(f"source must be a JSON list of objects: {path}")
    return data


def get(row: dict, *names: str):
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    raise KeyError(f"none of {names} found in row {row.get('sample_id')}")


def normalize(rows: list[dict], set_name: str) -> list[dict]:
    return [
        {
            "set": set_name,
            "sid": str(get(row, "sample_id")),
            "ht": str(get(row, "hallucination_type")),
            "img": str(get(row, "image_filename")),
            "bbox": list(get(row, "positive_bbox", "gt_bbox_xyxy", "chosen_bbox_xyxy")),
            "pos": str(get(row, "positive_text", "chosen")),
            "neg": str(get(row, "negative_text", "rejected")),
        }
        for row in rows
    ]


def main() -> None:
    sources = [("1996", SRC_1996), ("500dev", SRC_500)]
    source_meta = []
    normalized = []
    for set_name, path in sources:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = load(path)
        source_meta.append(
            {
                "set": set_name,
                "path": str(path),
                "source_origin": "retrieved_directly_from_vlm1_via_scp",
                "sha256": sha256(path),
                "source_rows": len(rows),
            }
        )
        normalized.extend(normalize(rows, set_name))

    if len(normalized) != 9984:
        raise AssertionError(f"total pair count {len(normalized)} != 9984")
    set_counts = Counter(row["set"] for row in normalized)
    if set_counts != Counter({"1996": 7984, "500dev": 2000}):
        raise AssertionError(f"set counts mismatch: {set_counts}")
    pair_keys = [(row["set"], row["sid"]) for row in normalized]
    if len(set(pair_keys)) != 9984:
        dupes = [k for k, n in Counter(pair_keys).items() if n > 1]
        raise AssertionError(f"duplicate (set,sid) keys: {dupes[:10]}")
    for row in normalized:
        if len(row["bbox"]) != 4 or not all(isinstance(x, (int, float)) for x in row["bbox"]):
            raise AssertionError(f"bad bbox: {row['set']} {row['sid']}")
        if not row["img"] or not row["pos"] or not row["neg"]:
            raise AssertionError(f"empty required field: {row['set']} {row['sid']}")

    OUT.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema": "vsight_auto_repair_pairs_dump_manifest_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_script": str(Path(__file__).resolve()),
        "source_files": source_meta,
        "conversion": {
            "input_counts": {m["set"]: m["source_rows"] for m in source_meta},
            "output_count": len(normalized),
            "output_counts": dict(set_counts),
            "unique_set_sid": len(set(pair_keys)),
            "duplicate_count": len(normalized) - len(set(pair_keys)),
            "fields": ["set", "sid", "ht", "img", "bbox", "pos", "neg"],
        },
        "output": {
            "path": str(INDEX),
            "sha256": sha256(INDEX),
            "bytes": INDEX.stat().st_size,
        },
        "original_sources_modified": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
