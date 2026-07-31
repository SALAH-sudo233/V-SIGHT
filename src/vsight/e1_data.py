"""Deterministic source-data primitives for the E1 training corpus."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .data_isolation import read_records


SOURCE_SCHEMA = "vsight_e1_positive_query_v1"
IMAGE_SPLIT_SCHEMA = "vsight_e1_image_split_v1"
CANDIDATE_BANK_SCHEMA = "vsight_e1_annotation_candidate_bank_v1"


@dataclass(frozen=True)
class SourceSpec:
    """One canonical RefCOCO-family split definition."""

    name: str
    dataset: str
    split_by: str
    path: Path


@dataclass(frozen=True)
class CocoIndex:
    """COCO metadata needed to resolve RefCOCO annotations."""

    images: Mapping[int, Mapping]
    annotations: Mapping[int, Mapping]
    categories: Mapping[int, str]
    category_annotations: Mapping[tuple[int, int], tuple[int, ...]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_coco_image_id(value: object) -> int:
    """Extract a COCO image ID from an integer or canonical file name."""

    if isinstance(value, bool):
        raise ValueError(f"invalid COCO image identity: {value!r}")
    if isinstance(value, int):
        return value
    text = Path(str(value)).name
    if text.isdigit():
        return int(text)
    match = re.search(r"(?<!\d)(\d{12})(?!\d)", text)
    if not match:
        raise ValueError(f"cannot extract COCO image ID from {value!r}")
    return int(match.group(1))


def protected_image_ids(paths: Iterable[Path]) -> frozenset[int]:
    """Read only image identities from protected JSON/JSONL datasets."""

    image_ids: set[int] = set()
    for path in paths:
        for index, row in enumerate(read_records(path)):
            value = row.get("image_id")
            if value in (None, ""):
                value = row.get("image_filename", row.get("image"))
            if value in (None, ""):
                raise ValueError(f"{path}: record {index} has no image identity")
            image_ids.add(extract_coco_image_id(value))
    return frozenset(image_ids)


def load_ref_records(path: Path) -> list[dict]:
    """Load a trusted local REFER pickle."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted benchmark artifact
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path}: expected a list of dictionaries")
    return payload


def load_coco_index(path: Path) -> CocoIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload.get("images", [])}
    annotations = {int(row["id"]): row for row in payload.get("annotations", [])}
    categories = {
        int(row["id"]): str(row["name"]) for row in payload.get("categories", [])
    }
    if not images or not annotations or not categories:
        raise ValueError(f"{path}: incomplete COCO instances payload")

    grouped: dict[tuple[int, int], list[int]] = {}
    for annotation in annotations.values():
        if not _valid_coco_bbox(annotation.get("bbox")):
            continue
        key = (int(annotation["image_id"]), int(annotation["category_id"]))
        grouped.setdefault(key, []).append(int(annotation["id"]))
    return CocoIndex(
        images=images,
        annotations=annotations,
        categories=categories,
        category_annotations={
            key: tuple(sorted(annotation_ids))
            for key, annotation_ids in grouped.items()
        },
    )


def assign_image_splits(
    image_ids: Iterable[int], calibration_fraction: float, seed: str
) -> tuple[frozenset[int], frozenset[int]]:
    """Assign an exact, deterministic calibration fraction by COCO image ID."""

    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")
    unique = sorted(set(image_ids))
    if not unique:
        raise ValueError("cannot split an empty image set")
    calibration_count = round(len(unique) * calibration_fraction)
    if calibration_fraction > 0 and len(unique) > 1:
        calibration_count = min(max(calibration_count, 1), len(unique) - 1)

    ranked = sorted(
        unique,
        key=lambda image_id: (
            hashlib.sha256(f"{seed}:{image_id}".encode("ascii")).digest(),
            image_id,
        ),
    )
    calibration = frozenset(ranked[:calibration_count])
    train = frozenset(set(unique) - calibration)
    return train, calibration


def xywh_to_clipped_xyxy(
    bbox: Sequence[object], image_width: int, image_height: int
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"expected xywh bbox, got {bbox!r}")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"bbox contains a non-finite coordinate: {bbox!r}")
    x, y, width, height = values
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError(f"bbox or image has non-positive extent: {bbox!r}")
    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox is empty after clipping: {bbox!r}")
    return [x1, y1, x2, y2]


def source_image_ids(
    refs: Iterable[Mapping], protected: frozenset[int]
) -> frozenset[int]:
    return frozenset(
        int(row["image_id"])
        for row in refs
        if row.get("split") == "train" and int(row["image_id"]) not in protected
    )


def iter_query_records(
    refs: Iterable[Mapping],
    spec: SourceSpec,
    coco: CocoIndex,
    protected: frozenset[int],
    train_images: frozenset[int],
    calibration_images: frozenset[int],
) -> Iterator[dict]:
    """Resolve REFER sentences into query-level positive-supervision rows."""

    for ref in refs:
        if ref.get("split") != "train":
            continue
        image_id = int(ref["image_id"])
        if image_id in protected:
            continue
        if image_id not in train_images and image_id not in calibration_images:
            raise ValueError(f"image {image_id} lacks an E1 split assignment")
        data_split = "train" if image_id in train_images else "calibration"

        ref_id = int(ref["ref_id"])
        ann_id = int(ref["ann_id"])
        category_id = int(ref["category_id"])
        image = coco.images.get(image_id)
        annotation = coco.annotations.get(ann_id)
        if image is None or annotation is None:
            raise ValueError(f"{spec.name}: unresolved image {image_id} or ann {ann_id}")
        if int(annotation["image_id"]) != image_id:
            raise ValueError(f"{spec.name}: ann {ann_id} points to a different image")
        if int(annotation["category_id"]) != category_id:
            raise ValueError(f"{spec.name}: ann {ann_id} has a different category")
        if category_id not in coco.categories:
            raise ValueError(f"{spec.name}: unknown COCO category {category_id}")

        width, height = int(image["width"]), int(image["height"])
        gt_xyxy = xywh_to_clipped_xyxy(annotation["bbox"], width, height)
        gt_xywh = [
            gt_xyxy[0],
            gt_xyxy[1],
            gt_xyxy[2] - gt_xyxy[0],
            gt_xyxy[3] - gt_xyxy[1],
        ]
        same_category = coco.category_annotations.get((image_id, category_id), ())
        distractor_count = sum(candidate != ann_id for candidate in same_category)

        sentences = ref.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"{spec.name}: ref {ref_id} has no sentences")
        for sentence in sentences:
            sent_id = int(sentence["sent_id"])
            query = str(sentence.get("sent") or sentence.get("raw") or "").strip()
            query_raw = str(sentence.get("raw") or query).strip()
            if not query:
                raise ValueError(f"{spec.name}: sentence {sent_id} is empty")
            yield {
                "schema_version": SOURCE_SCHEMA,
                "record_type": "positive_query",
                "query_id": f"{spec.name}:sent:{sent_id}",
                "group_id": f"{spec.name}:ref:{ref_id}",
                "data_split": data_split,
                "source_dataset": spec.dataset,
                "source_split_by": spec.split_by,
                "source_original_split": "train",
                "ref_id": ref_id,
                "sent_id": sent_id,
                "ann_id": ann_id,
                "image_id": image_id,
                "image_filename": str(image["file_name"]),
                "image_width": width,
                "image_height": height,
                "query": query,
                "query_raw": query_raw,
                "category_id": category_id,
                "category_name": coco.categories[category_id],
                "gt_bbox_xywh": gt_xywh,
                "gt_bbox_xyxy": gt_xyxy,
                "same_category_distractor_count": distractor_count,
            }


def image_split_records(
    image_ids: Iterable[int], split: str, coco: CocoIndex
) -> Iterator[dict]:
    for image_id in sorted(image_ids):
        image = coco.images[image_id]
        yield {
            "schema_version": IMAGE_SPLIT_SCHEMA,
            "group_id": f"coco:{image_id:012d}",
            "image_id": image_id,
            "image_filename": str(image["file_name"]),
            "data_split": split,
        }


def bbox_iou(left: Sequence[object], right: Sequence[object]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("IoU requires two xyxy boxes")
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    if lx2 <= lx1 or ly2 <= ly1 or rx2 <= rx1 or ry2 <= ry1:
        raise ValueError("IoU requires positive-area xyxy boxes")
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union


def annotation_candidate_record(
    ann_id: int,
    data_split: str,
    coco: CocoIndex,
    max_same_category: int = 5,
    max_same_category_iou: float = 0.9,
) -> dict:
    """Build query-independent instance and localization supervision."""

    if max_same_category < 1:
        raise ValueError("max_same_category must be positive")
    if not 0.0 < max_same_category_iou <= 1.0:
        raise ValueError("max_same_category_iou must be in (0, 1]")
    target = coco.annotations[ann_id]
    image_id = int(target["image_id"])
    category_id = int(target["category_id"])
    image = coco.images[image_id]
    width, height = int(image["width"]), int(image["height"])
    gt = xywh_to_clipped_xyxy(target["bbox"], width, height)
    diagonal = math.hypot(width, height)
    gt_area = _box_area(gt)
    gt_center = ((gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0)

    distractors = []
    raw_distractor_count = 0
    overlap_excluded = 0
    for candidate_ann_id in coco.category_annotations.get(
        (image_id, category_id), ()
    ):
        if candidate_ann_id == ann_id:
            continue
        candidate_ann = coco.annotations[candidate_ann_id]
        if int(candidate_ann.get("iscrowd", 0)) != 0:
            continue
        raw_distractor_count += 1
        box = xywh_to_clipped_xyxy(candidate_ann["bbox"], width, height)
        overlap = bbox_iou(gt, box)
        if overlap >= max_same_category_iou:
            overlap_excluded += 1
            continue
        candidate_area = _box_area(box)
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        center_distance = math.hypot(
            center[0] - gt_center[0], center[1] - gt_center[1]
        ) / diagonal
        size_similarity = math.exp(-abs(math.log(candidate_area / gt_area)))
        hardness = 0.6 * size_similarity + 0.4 * max(0.0, 1.0 - center_distance)
        distractors.append(
            {
                "candidate_id": f"same_category_ann:{candidate_ann_id}",
                "candidate_type": "same_category_instance",
                "ann_id": candidate_ann_id,
                "bbox_xyxy": box,
                "iou_to_gt": overlap,
                "center_distance_normalized": center_distance,
                "area_ratio_to_gt": candidate_area / gt_area,
                "annotation_hardness": hardness,
            }
        )
    distractors.sort(
        key=lambda row: (-row["annotation_hardness"], row["ann_id"])
    )

    return {
        "schema_version": CANDIDATE_BANK_SCHEMA,
        "record_type": "annotation_candidate_supervision",
        "candidate_bank_id": f"coco:{image_id:012d}:ann:{ann_id}",
        "group_id": f"coco:{image_id:012d}",
        "data_split": data_split,
        "image_id": image_id,
        "image_filename": str(image["file_name"]),
        "image_width": width,
        "image_height": height,
        "target_ann_id": ann_id,
        "category_id": category_id,
        "category_name": coco.categories[category_id],
        "gt_bbox_xyxy": gt,
        "same_category_raw": raw_distractor_count,
        "same_category_available": len(distractors),
        "same_category_overlap_excluded": overlap_excluded,
        "same_category_max_iou": max_same_category_iou,
        "same_category_candidates": distractors[:max_same_category],
        "localization_candidates": localization_candidates(gt, width, height),
    }


def localization_candidates(
    gt_bbox_xyxy: Sequence[object], image_width: int, image_height: int
) -> list[dict]:
    """Create deterministic lower-IoU boxes without using model outputs."""

    gt = [float(value) for value in gt_bbox_xyxy]
    x1, y1, x2, y2 = gt
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("localization candidates require a positive-area GT box")
    raw = [
        ("partial_horizontal", [x1, y1, x1 + width * 0.55, y2]),
        ("partial_vertical", [x1, y1, x2, y1 + height * 0.55]),
        (
            "oversized_context",
            [
                max(0.0, x1 - width * 0.4),
                max(0.0, y1 - height * 0.4),
                min(float(image_width), x2 + width * 0.4),
                min(float(image_height), y2 + height * 0.4),
            ],
        ),
        (
            "jitter_horizontal",
            _shift_box(gt, width * 0.35, 0.0, image_width, image_height),
        ),
        (
            "jitter_vertical",
            _shift_box(gt, 0.0, height * 0.35, image_width, image_height),
        ),
    ]
    candidates = []
    seen = {tuple(gt)}
    for candidate_type, box in raw:
        key = tuple(round(value, 10) for value in box)
        if key in seen:
            continue
        seen.add(key)
        iou = bbox_iou(gt, box)
        if iou >= 0.95:
            continue
        candidates.append(
            {
                "candidate_id": f"localization:{candidate_type}",
                "candidate_type": candidate_type,
                "bbox_xyxy": box,
                "iou_to_gt": iou,
            }
        )
    return candidates


def _shift_box(
    box: Sequence[float],
    dx: float,
    dy: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    shifted_x1 = x1 + dx
    shifted_y1 = y1 + dy
    if shifted_x1 + width > image_width:
        shifted_x1 = x1 - dx
    if shifted_y1 + height > image_height:
        shifted_y1 = y1 - dy
    shifted_x1 = min(max(shifted_x1, 0.0), float(image_width) - width)
    shifted_y1 = min(max(shifted_y1, 0.0), float(image_height) - height)
    return [
        shifted_x1,
        shifted_y1,
        shifted_x1 + width,
        shifted_y1 + height,
    ]


def _box_area(box: Sequence[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _valid_coco_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        coordinates = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in coordinates) and coordinates[2] > 0 and coordinates[3] > 0
