"""Frozen P1 prompts and conservative grounding-output parsers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


BASELINE_SYSTEM_PROMPT = (
    "You are a careful visual grounding assistant. Use only visible image "
    "evidence. Follow the requested output format exactly."
)
BASELINE_PROMPT = (
    "Task: Decide whether the referring expression strictly matches a visible "
    "target in the image, then localize it if it exists.\n"
    'Referring expression: "{expr}"\n'
    "If the target exists, return exactly one bounding box in absolute image "
    "pixels as [x1, y1, x2, y2].\n"
    "If the target does not exist, return exactly: not found.\n"
    "Do not output explanations."
)
CHALLENGER_SYSTEM_PROMPT = (
    "You are a careful visual grounding candidate generator. Use only visible "
    "image evidence and follow the requested JSON format exactly."
)
CHALLENGER_PROMPT = (
    "Task: propose diverse target boxes for the referring expression below.\n"
    'Expression: "{expr}"\n'
    "First distinguish the target object from any object used only as an "
    "attribute or spatial-relation reference. Then return up to 3 target-object "
    "boxes, ordered by how well the complete expression matches. Keep alternative "
    "target instances when uncertain and never answer not found.\n"
    'Return JSON only: {{"boxes":[[x1,y1,x2,y2], ...]}}. Coordinates must be '
    "absolute image pixels."
)


def prompt_hash(system_prompt: str, prompt_template: str) -> str:
    return hashlib.sha256(
        (system_prompt + "\n\0\n" + prompt_template).encode("utf-8")
    ).hexdigest()


def generation_spec() -> dict:
    return {
        "baseline_prompt_sha256": prompt_hash(
            BASELINE_SYSTEM_PROMPT, BASELINE_PROMPT
        ),
        "challenger_prompt_sha256": prompt_hash(
            CHALLENGER_SYSTEM_PROMPT, CHALLENGER_PROMPT
        ),
        "decoding": {
            "do_sample": False,
            "use_cache": True,
        },
        "challenger_selection": "first_box_from_ordered_binding_aware_output",
    }


def parse_baseline_output(raw_text: str, image_size: tuple[int, int]) -> dict:
    answer = _answer_content(raw_text)
    lowered = answer.strip().casefold()
    if "not found" in lowered or lowered in {"no", "none", "null", "[]", "{}"}:
        return {
            "pred_found": False,
            "pred_bbox_xyxy": None,
            "parse_valid": True,
            "parse_method": "not_found",
        }
    boxes = parse_candidate_boxes(answer, image_size, max_candidates=1)
    if boxes:
        return {
            "pred_found": True,
            "pred_bbox_xyxy": boxes[0],
            "parse_valid": True,
            "parse_method": "bbox",
        }
    return {
        "pred_found": False,
        "pred_bbox_xyxy": None,
        "parse_valid": False,
        "parse_method": "no_bbox",
    }


def parse_challenger_output(raw_text: str, image_size: tuple[int, int]) -> dict:
    boxes = parse_candidate_boxes(raw_text, image_size, max_candidates=3)
    return {
        "candidate_boxes_xyxy": boxes,
        "selected_bbox_xyxy": boxes[0] if boxes else None,
        "parse_valid": bool(boxes),
        "parse_method": "ordered_first_box" if boxes else "no_bbox",
    }


def parse_candidate_boxes(
    raw_text: str,
    image_size: tuple[int, int],
    max_candidates: int,
) -> list[list[float]]:
    sequences: list[Sequence[Any]] = []
    raw = str(raw_text or "").strip()
    candidates = [raw]
    candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.I))
    object_match = re.search(r"\{[\s\S]*\}", raw)
    array_match = re.search(r"\[[\s\S]*\]", raw)
    if object_match:
        candidates.append(object_match.group(0))
    if array_match:
        candidates.append(array_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        _collect_boxes(parsed, sequences)
        if sequences:
            break
    if not sequences:
        number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        pattern = re.compile(
            rf"[\[\(]\s*({number})\s*,\s*({number})\s*,\s*"
            rf"({number})\s*,\s*({number})\s*[\]\)]"
        )
        sequences.extend(match.groups() for match in pattern.finditer(raw))

    boxes = []
    for sequence in sequences:
        box = _normalize_box(sequence, image_size)
        if box is not None and box not in boxes:
            boxes.append(box)
        if len(boxes) >= max_candidates:
            break
    return boxes


def _answer_content(raw_text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", str(raw_text), flags=re.S | re.I)
    return match.group(1).strip() if match else str(raw_text).strip()


def _collect_boxes(value: Any, output: list[Sequence[Any]]) -> None:
    if isinstance(value, Mapping):
        preferred = ("boxes", "bboxes", "bbox", "bbox_2d", "box", "bbox_xyxy")
        matched = False
        for key in preferred:
            if key in value:
                _collect_boxes(value[key], output)
                matched = True
        if not matched:
            for child in value.values():
                _collect_boxes(child, output)
    elif isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            output.append(value)
        else:
            for child in value:
                _collect_boxes(child, output)


def _normalize_box(
    value: Sequence[Any], image_size: tuple[int, int]
) -> list[float] | None:
    if len(value) != 4:
        return None
    try:
        coordinates = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in coordinates):
        return None
    width, height = image_size
    if max(abs(item) for item in coordinates) <= 1.5:
        coordinates = [
            coordinates[0] * width,
            coordinates[1] * height,
            coordinates[2] * width,
            coordinates[3] * height,
        ]
    x1, y1, x2, y2 = coordinates
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]
