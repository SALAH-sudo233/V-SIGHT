"""Dataset identity extraction and split-leakage checks."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class DatasetIdentity:
    records: int
    groups: frozenset[str]
    images: frozenset[str]
    pairs: frozenset[str]


def read_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"{path}: expected a JSON list or a records/data/samples list")


def identity(records: Iterable[Mapping]) -> DatasetIdentity:
    rows = list(records)
    groups: set[str] = set()
    images: set[str] = set()
    pairs: set[str] = set()
    for index, row in enumerate(rows):
        group = _value(row, "base_sample_id", "group_id")
        image = _value(row, "image_filename", "image_id", "image")
        pair = _value(row, "pair_id")
        if group is None:
            sample = _value(row, "sample_id", "id")
            if sample is not None:
                group = sample.split("__", 1)[0]
        if group is None or image is None:
            raise ValueError(f"record {index} lacks group or image identity")
        groups.add(str(group))
        images.add(_image_identity(image))
        if pair is not None:
            pairs.add(str(pair))
    return DatasetIdentity(
        records=len(rows),
        groups=frozenset(groups),
        images=frozenset(images),
        pairs=frozenset(pairs),
    )


def compare(left: DatasetIdentity, right: DatasetIdentity) -> dict:
    group_overlap = sorted(left.groups & right.groups)
    image_overlap = sorted(left.images & right.images)
    pair_overlap = sorted(left.pairs & right.pairs)
    return {
        "is_disjoint": not (group_overlap or image_overlap or pair_overlap),
        "group_overlap_count": len(group_overlap),
        "image_overlap_count": len(image_overlap),
        "pair_overlap_count": len(pair_overlap),
        "group_overlap_examples": group_overlap[:10],
        "image_overlap_examples": image_overlap[:10],
        "pair_overlap_examples": pair_overlap[:10],
    }


def audit_splits(paths: Mapping[str, Path]) -> dict:
    identities = {
        name: identity(read_records(path)) for name, path in paths.items()
    }
    report = {
        "splits": {
            name: {
                "path": str(paths[name]),
                "sha256": _sha256(paths[name]),
                "records": item.records,
                "groups": len(item.groups),
                "images": len(item.images),
                "pairs": len(item.pairs),
            }
            for name, item in identities.items()
        },
        "comparisons": {},
    }
    names = list(paths)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            report["comparisons"][f"{left}__{right}"] = compare(
                identities[left], identities[right]
            )
    report["all_disjoint"] = all(
        item["is_disjoint"] for item in report["comparisons"].values()
    )
    return report


def _value(row: Mapping, *keys: str):
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _image_identity(value) -> str:
    text = Path(str(value)).name
    match = re.search(r"(?<!\d)(\d{12})(?!\d)", text)
    if match:
        return f"coco:{match.group(1)}"
    if text.isdigit():
        return f"coco:{int(text):012d}"
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
