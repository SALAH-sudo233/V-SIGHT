#!/usr/bin/env python3
"""Audit GT-target attributes for valid-box IoU=0 groups with Bailian."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/audits/zero_iou_127.template.jsonl"
DEFAULT_HUMAN_REVIEWS = ROOT / "data/audits/zero_iou_127.reviews.jsonl"
DEFAULT_IMAGES = Path("/home/u2025141034/benchmark/benchmark_images")
DEFAULT_OUTPUT = (
    ROOT
    / "data/audits/zero_iou_attributes.qwen3.7-max-2026-05-17.jsonl"
)
DEFAULT_MODEL = "qwen3.7-max-2026-05-17"
DEFAULT_VISION_MODEL = "qwen3-vl-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CONFIDENCE = {"high", "medium", "low"}
VERDICTS = {"supported", "contradicted", "not_visible", "not_applicable"}
CHECK_TYPES = {
    "object",
    "gender",
    "color",
    "material",
    "attribute",
    "action_state",
    "count",
    "relation",
}
CHECK_TYPE_ALIASES = {
    "category": "object",
    "object_identity": "object",
    "action": "action_state",
    "state": "action_state",
    "spatial": "relation",
    "spatial_relation": "relation",
    "property": "attribute",
    "material_attribute": "material",
    "number": "count",
}
GENDER_VALUES = {
    "male_presenting",
    "female_presenting",
    "unclear",
    "not_applicable",
}
CAUSES = {
    "same_category_instance_confusion",
    "target_reference_role_swap",
    "wrong_category",
    "localization_or_box_quality",
    "visually_ambiguous_reference",
    "annotation_or_gt_issue",
    "other",
}

VISION_SYSTEM_PROMPT = r"""You are the visual evidence extractor in a
two-stage visual-grounding audit. You receive two views of one image:
1. a full image where the GREEN rectangle is the exact ground-truth target;
   RED is the T2 baseline box and ORANGE is the T4 baseline box when present;
2. an unmarked crop centered on the green ground-truth target.

Inspect the exact green target and record only directly visible evidence. Do
not infer hidden properties. For a person, apparent_gender describes only
visible gender presentation; use "unclear" when evidence is insufficient. For
non-person targets use "not_applicable". Split colors and materials by object
part. Check every visually testable atom in the referring expression. Also
describe what each colored zero-IoU baseline box selected, because this is
needed to distinguish instance confusion, target-reference role swaps, wrong
categories, and poor boxes. The category_hint may be wrong; pixels and the
green box take priority.

Return pure JSON with exactly this structure:
{
  "target_visible": true,
  "target_identity_observation": {
    "category": "visible target category or unknown",
    "apparent_gender": "male_presenting|female_presenting|unclear|not_applicable",
    "gender_evidence": "short visible evidence or empty"
  },
  "visible_attributes": {
    "colors": [{"part":"", "value":"", "confidence":"high|medium|low", "evidence":""}],
    "materials": [{"part":"", "value":"", "confidence":"high|medium|low", "evidence":""}],
    "other": [{"attribute":"", "confidence":"high|medium|low", "evidence":""}],
    "actions_or_states": [{"value":"", "confidence":"high|medium|low", "evidence":""}]
  },
  "spatial_and_relational_facts": [
    {"fact":"", "confidence":"high|medium|low", "evidence":""}
  ],
  "query_visual_checks": [
    {"atom":"", "type":"object|gender|color|material|attribute|action_state|count|relation", "verdict":"supported|contradicted|not_visible|not_applicable", "evidence":""}
  ],
  "gt_disambiguating_cues": ["short visible cue"],
  "zero_iou_baseline_observations": [
    {"task":"t2_vqa_grounding|t4_caption_grounding", "predicted_region_description":"", "same_object_as_gt":"yes|no|uncertain", "confused_with":"same_category_other_instance|relation_anchor|different_category|background_or_bad_box|uncertain", "evidence":""}
  ],
  "limitations": ["visibility or resolution limitation"]
}

Use empty arrays when evidence is unavailable. Never return markdown or facts
inferred only from stereotypes, names, or world knowledge."""


ADJUDICATION_SYSTEM_PROMPT = r"""You are a conservative visual-grounding auditor.

You are the text-only adjudication stage. You receive request_context plus
verified_vision_evidence produced by a separate vision model. You do not
receive images. Treat the supplied visual evidence as the only authority for
visible facts, retain its uncertainty, and do not invent or strengthen facts.

Conservatively verify gender presentation, color by part, material, other
attributes, actions/states, and spatial relations for the exact GT target.
Check every visually testable atom in the referring expression, including
object identity, gender/person term, color, material, action/state, count, and
target-reference relation. The category_hint is metadata and may be wrong.

Use baseline observations only to diagnose why grounding drifted, never as
evidence about GT attributes. Return pure JSON with exactly this structure:
{
  "target_visible": true,
  "query_binds_gt_target": "yes|no|uncertain",
  "target_identity": {
    "category": "visible target category or unknown",
    "apparent_gender": "male_presenting|female_presenting|unclear|not_applicable",
    "gender_evidence": "short visible evidence or empty"
  },
  "attributes": {
    "colors": [{"part":"", "value":"", "confidence":"high|medium|low", "evidence":""}],
    "materials": [{"part":"", "value":"", "confidence":"high|medium|low", "evidence":""}],
    "other": [{"attribute":"", "confidence":"high|medium|low", "evidence":""}],
    "actions_or_states": [{"value":"", "confidence":"high|medium|low", "evidence":""}]
  },
  "spatial_and_relational_facts": [
    {"fact":"", "confidence":"high|medium|low", "evidence":""}
  ],
  "query_attribute_checks": [
    {"atom":"", "type":"object|gender|color|material|attribute|action_state|count|relation", "verdict":"supported|contradicted|not_visible|not_applicable", "evidence":""}
  ],
  "disambiguating_cues": ["short cue that distinguishes the green target"],
  "instance_confusion_risk": "low|medium|high",
  "likely_zero_iou_cause": "same_category_instance_confusion|target_reference_role_swap|wrong_category|localization_or_box_quality|visually_ambiguous_reference|annotation_or_gt_issue|other",
  "reason": "short image-grounded conclusion"
}

Use empty arrays when an attribute family cannot be verified. Never return
markdown or facts inferred only from stereotypes or world knowledge."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256() -> str:
    prompts = json.dumps(
        {
            "vision": VISION_SYSTEM_PROMPT,
            "adjudication": ADJUDICATION_SYSTEM_PROMPT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(prompts.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()


def select_groups(manifest: Path) -> list[dict[str, Any]]:
    selected = []
    for group in read_jsonl(manifest):
        zero_cases = [
            case
            for case in group.get("cases") or []
            if case.get("baseline_box") is not None
            and float(case.get("baseline_iou", -1)) == 0.0
        ]
        if not zero_cases:
            continue
        copied = dict(group)
        copied["zero_iou_cases"] = zero_cases
        selected.append(copied)
    return selected


def completed_human_review_ids(path: Path) -> set[str]:
    """Return samples whose latest record for any reviewer is completed."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("base_sample_id") or "")
        reviewer_id = str(row.get("reviewer_id") or "")
        if sample_id and reviewer_id:
            latest[(sample_id, reviewer_id)] = row
    return {
        sample_id
        for (sample_id, _), row in latest.items()
        if row.get("status") == "completed"
    }


def _clamped_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(1, min(width, round(x2))),
        max(1, min(height, round(y2))),
    )


def _jpeg_data_url(image: Any, max_side: int = 1600) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_views(group: Mapping[str, Any], image_dir: Path) -> list[tuple[str, str]]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required for attribute-audit image views") from exc

    image_path = image_dir / Path(str(group["image_filename"])).name
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    marked = original.copy()
    draw = ImageDraw.Draw(marked)
    width = max(3, round(min(marked.size) / 110))
    font = ImageFont.load_default()

    gt_box = _clamped_box(group["zero_iou_cases"][0]["gt_box"], *marked.size)
    draw.rectangle(gt_box, outline=(20, 180, 70), width=width)
    draw.text((gt_box[0] + 4, max(0, gt_box[1] + 4)), "GT TARGET", fill=(20, 180, 70), font=font)
    colors = {
        "t2_vqa_grounding": ((220, 35, 35), "T2 BASE"),
        "t4_caption_grounding": ((235, 130, 20), "T4 BASE"),
    }
    for case in group["zero_iou_cases"]:
        color, label = colors[case["task"]]
        box = _clamped_box(case["baseline_box"], *marked.size)
        draw.rectangle(box, outline=color, width=width)
        draw.text((box[0] + 4, max(0, box[1] + 4)), label, fill=color, font=font)

    x1, y1, x2, y2 = gt_box
    pad_x = max(12, round((x2 - x1) * 0.18))
    pad_y = max(12, round((y2 - y1) * 0.18))
    crop = original.crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(original.width, x2 + pad_x),
            min(original.height, y2 + pad_y),
        )
    )
    return [
        ("full image with GT and zero-IoU baseline boxes", _jpeg_data_url(marked)),
        ("unmarked GT target crop", _jpeg_data_url(crop, max_side=1000)),
    ]


def request_context(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_sample_id": group["base_sample_id"],
        "image_filename": group["image_filename"],
        "referring_expression": group["query"],
        "category_hint": group.get("target_category") or "",
        "expression_structure_hint": group.get("expression_structure") or "",
        "same_category_distractor_count_hint": group.get("same_category_distractors"),
        "gt_box_xyxy": group["zero_iou_cases"][0]["gt_box"],
        "zero_iou_baselines": [
            {
                "task": case["task"],
                "baseline_box_xyxy": case["baseline_box"],
                "automatic_match_class_hint": case.get("automatic_baseline_class"),
            }
            for case in group["zero_iou_cases"]
        ],
    }


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def normalize_audit_types(value: dict[str, Any]) -> list[str]:
    """Normalize a small, explicit set of model aliases before validation."""
    corrections: list[str] = []
    checks = value.get("query_attribute_checks")
    if not isinstance(checks, list):
        return corrections
    for index, item in enumerate(checks):
        if not isinstance(item, dict):
            continue
        original = item.get("type")
        normalized = CHECK_TYPE_ALIASES.get(str(original))
        if normalized:
            item["type"] = normalized
            corrections.append(f"query_attribute_checks[{index}].type:{original}->{normalized}")
    return corrections


def _validate_attribute_items(items: Any, fields: tuple[str, ...], label: str) -> None:
    if not isinstance(items, list):
        raise ValueError(f"attributes.{label} must be a list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"attributes.{label}[{index}] must be an object")
        for field in fields:
            if not str(item.get(field) or "").strip():
                raise ValueError(f"attributes.{label}[{index}].{field} is empty")
        if item.get("confidence") not in CONFIDENCE:
            raise ValueError(f"attributes.{label}[{index}].confidence is invalid")


def validate_audit(value: Mapping[str, Any]) -> None:
    if not isinstance(value.get("target_visible"), bool):
        raise ValueError("target_visible must be boolean")
    if value.get("query_binds_gt_target") not in {"yes", "no", "uncertain"}:
        raise ValueError("query_binds_gt_target is invalid")
    identity = value.get("target_identity")
    if not isinstance(identity, dict) or not str(identity.get("category") or "").strip():
        raise ValueError("target_identity.category is required")
    if identity.get("apparent_gender") not in GENDER_VALUES:
        raise ValueError("target_identity.apparent_gender is invalid")
    if (
        identity.get("apparent_gender") in {"male_presenting", "female_presenting"}
        and not str(identity.get("gender_evidence") or "").strip()
    ):
        raise ValueError("target_identity.gender_evidence is required")
    attributes = value.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be an object")
    _validate_attribute_items(attributes.get("colors"), ("part", "value", "evidence"), "colors")
    _validate_attribute_items(attributes.get("materials"), ("part", "value", "evidence"), "materials")
    _validate_attribute_items(attributes.get("other"), ("attribute", "evidence"), "other")
    _validate_attribute_items(
        attributes.get("actions_or_states"), ("value", "evidence"), "actions_or_states"
    )
    facts = value.get("spatial_and_relational_facts")
    if not isinstance(facts, list):
        raise ValueError("spatial_and_relational_facts must be a list")
    for index, item in enumerate(facts):
        if not isinstance(item, dict) or not str(item.get("fact") or "").strip():
            raise ValueError(f"spatial_and_relational_facts[{index}] is malformed")
        if item.get("confidence") not in CONFIDENCE:
            raise ValueError(f"spatial_and_relational_facts[{index}].confidence is invalid")
    checks = value.get("query_attribute_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("query_attribute_checks must be a non-empty list")
    for index, item in enumerate(checks):
        if not isinstance(item, dict) or not str(item.get("atom") or "").strip():
            raise ValueError(f"query_attribute_checks[{index}] is malformed")
        if item.get("verdict") not in VERDICTS:
            raise ValueError(f"query_attribute_checks[{index}].verdict is invalid")
        if item.get("type") not in CHECK_TYPES:
            raise ValueError(f"query_attribute_checks[{index}].type is invalid")
    cues = value.get("disambiguating_cues")
    if not isinstance(cues, list) or any(not isinstance(item, str) for item in cues):
        raise ValueError("disambiguating_cues must be a string list")
    if value.get("instance_confusion_risk") not in {"low", "medium", "high"}:
        raise ValueError("instance_confusion_risk is invalid")
    if value.get("likely_zero_iou_cause") not in CAUSES:
        raise ValueError("likely_zero_iou_cause is invalid")
    if not str(value.get("reason") or "").strip():
        raise ValueError("reason is empty")


def validate_vision_evidence(value: Mapping[str, Any]) -> None:
    if not isinstance(value.get("target_visible"), bool):
        raise ValueError("vision target_visible must be boolean")
    identity = value.get("target_identity_observation")
    if not isinstance(identity, dict) or not str(identity.get("category") or "").strip():
        raise ValueError("vision target_identity_observation.category is required")
    if identity.get("apparent_gender") not in GENDER_VALUES:
        raise ValueError("vision apparent_gender is invalid")
    if (
        identity.get("apparent_gender") in {"male_presenting", "female_presenting"}
        and not str(identity.get("gender_evidence") or "").strip()
    ):
        raise ValueError("vision gender_evidence is required")
    attributes = value.get("visible_attributes")
    if not isinstance(attributes, dict):
        raise ValueError("vision visible_attributes must be an object")
    _validate_attribute_items(attributes.get("colors"), ("part", "value", "evidence"), "colors")
    _validate_attribute_items(
        attributes.get("materials"), ("part", "value", "evidence"), "materials"
    )
    _validate_attribute_items(attributes.get("other"), ("attribute", "evidence"), "other")
    _validate_attribute_items(
        attributes.get("actions_or_states"), ("value", "evidence"), "actions_or_states"
    )
    facts = value.get("spatial_and_relational_facts")
    if not isinstance(facts, list):
        raise ValueError("vision spatial_and_relational_facts must be a list")
    for index, item in enumerate(facts):
        if not isinstance(item, dict) or not str(item.get("fact") or "").strip():
            raise ValueError(f"vision spatial_and_relational_facts[{index}] is malformed")
        if item.get("confidence") not in CONFIDENCE:
            raise ValueError(f"vision spatial_and_relational_facts[{index}].confidence is invalid")
    checks = value.get("query_visual_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("vision query_visual_checks must be a non-empty list")
    for index, item in enumerate(checks):
        if not isinstance(item, dict) or not str(item.get("atom") or "").strip():
            raise ValueError(f"vision query_visual_checks[{index}] is malformed")
        if item.get("verdict") not in VERDICTS | {"uncertain"}:
            raise ValueError(f"vision query_visual_checks[{index}].verdict is invalid")
        if item.get("type") not in CHECK_TYPES | {"category"}:
            raise ValueError(f"vision query_visual_checks[{index}].type is invalid")
    cues = value.get("gt_disambiguating_cues")
    if not isinstance(cues, list) or any(not isinstance(item, str) for item in cues):
        raise ValueError("vision gt_disambiguating_cues must be a string list")
    observations = value.get("zero_iou_baseline_observations")
    if not isinstance(observations, list):
        raise ValueError("vision zero_iou_baseline_observations must be a list")
    for index, item in enumerate(observations):
        if not isinstance(item, dict):
            raise ValueError(f"vision zero_iou_baseline_observations[{index}] is malformed")
        if item.get("task") not in {"t2_vqa_grounding", "t4_caption_grounding"}:
            raise ValueError(f"vision zero_iou_baseline_observations[{index}].task is invalid")
        if item.get("same_object_as_gt") not in {"yes", "no", "uncertain"}:
            raise ValueError(
                f"vision zero_iou_baseline_observations[{index}].same_object_as_gt is invalid"
            )
        if item.get("confused_with") not in {
            "same_category_other_instance",
            "relation_anchor",
            "different_category",
            "background_or_bad_box",
            "uncertain",
        }:
            raise ValueError(
                f"vision zero_iou_baseline_observations[{index}].confused_with is invalid"
            )
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ValueError("vision limitations must be a string list")


class ClientPool:
    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.local = threading.local()

    def get(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("the openai package is required") from exc
        if not hasattr(self.local, "client"):
            self.local.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self.local.client


def _usage(response: Any) -> dict[str, Any]:
    usage = response.usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_vision_model(
    pool: ClientPool,
    args: argparse.Namespace,
    group: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(request_context(group), ensure_ascii=False),
        }
    ]
    for label, data_url in render_views(group, args.image_dir):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    kwargs: dict[str, Any] = {
        "model": args.vision_model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": args.vision_max_tokens,
    }
    response = pool.get().chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    evidence = parse_json_response(raw)
    validate_vision_evidence(evidence)
    return evidence, _usage(response)


def call_adjudication_model(
    pool: ClientPool,
    args: argparse.Namespace,
    group: Mapping[str, Any],
    vision_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    payload = {
        "request_context": request_context(group),
        "verified_vision_evidence": vision_evidence,
    }
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    kwargs["extra_body"] = {"enable_thinking": not args.disable_thinking}
    response = pool.get().chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    audit = parse_json_response(raw)
    normalizations = normalize_audit_types(audit)
    validate_audit(audit)
    return audit, _usage(response), normalizations


def audit_group(
    pool: ClientPool,
    args: argparse.Namespace,
    group: Mapping[str, Any],
    manifest_hash: str,
) -> dict[str, Any]:
    cached_evidence = group.get("_cached_vision_evidence")
    vision_evidence = dict(cached_evidence) if isinstance(cached_evidence, dict) else None
    cached_usage = group.get("_cached_vision_usage")
    vision_usage = dict(cached_usage) if isinstance(cached_usage, dict) else None
    vision_evidence_source = "resumed_output" if vision_evidence is not None else "api"
    last_error = ""
    vision_attempt = 0
    if vision_evidence is None:
        for vision_attempt in range(1, args.retries + 1):
            try:
                vision_evidence, vision_usage = call_vision_model(pool, args, group)
                break
            except Exception as exc:  # API and schema errors are resumable state.
                last_error = f"{type(exc).__name__}: {exc}"
                if vision_attempt < args.retries:
                    time.sleep(min(20.0, args.retry_delay * (2 ** (vision_attempt - 1))))
    if vision_evidence is None:
        return {
            "schema_version": "vsight_zero_iou_attribute_audit_v2",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "error",
            "error_stage": "vision_evidence",
            "base_sample_id": group["base_sample_id"],
            "image_filename": group["image_filename"],
            "query": group["query"],
            "model": args.model,
            "vision_model": args.vision_model,
            "source_manifest_sha256": manifest_hash,
            "prompt_sha256": prompt_sha256(),
            "vision_attempt": vision_attempt,
            "error": last_error,
        }

    adjudication_attempt = 0
    for adjudication_attempt in range(1, args.retries + 1):
        try:
            audit, adjudication_usage, audit_normalizations = call_adjudication_model(
                pool, args, group, vision_evidence
            )
            return {
                "schema_version": "vsight_zero_iou_attribute_audit_v2",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "ok",
                "base_sample_id": group["base_sample_id"],
                "image_filename": group["image_filename"],
                "query": group["query"],
                "model": args.model,
                "vision_model": args.vision_model,
                "source_manifest_sha256": manifest_hash,
                "prompt_sha256": prompt_sha256(),
                "vision_attempt": vision_attempt,
                "vision_evidence_source": vision_evidence_source,
                "adjudication_attempt": adjudication_attempt,
                "audited_tasks": [case["task"] for case in group["zero_iou_cases"]],
                "human_review_completed_at_selection": bool(
                    group.get("human_review_completed_at_selection")
                ),
                "request_context": request_context(group),
                "vision_evidence": vision_evidence,
                "audit": audit,
                "audit_normalizations": audit_normalizations,
                "usage": {
                    "vision_evidence": vision_usage,
                    "adjudication": adjudication_usage,
                },
            }
        except Exception as exc:  # API and schema errors are resumable state.
            last_error = f"{type(exc).__name__}: {exc}"
            if adjudication_attempt < args.retries:
                time.sleep(
                    min(20.0, args.retry_delay * (2 ** (adjudication_attempt - 1)))
                )
    return {
        "schema_version": "vsight_zero_iou_attribute_audit_v2",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "error",
        "error_stage": "adjudication",
        "base_sample_id": group["base_sample_id"],
        "image_filename": group["image_filename"],
        "query": group["query"],
        "model": args.model,
        "vision_model": args.vision_model,
        "source_manifest_sha256": manifest_hash,
        "prompt_sha256": prompt_sha256(),
        "vision_attempt": vision_attempt,
        "vision_evidence_source": vision_evidence_source,
        "adjudication_attempt": adjudication_attempt,
        "vision_evidence": vision_evidence,
        "usage": {"vision_evidence": vision_usage},
        "error": last_error,
    }


def latest_successes(
    output: Path, model: str, vision_model: str, manifest_hash: str
) -> set[str]:
    return {
        str(row.get("base_sample_id"))
        for row in read_jsonl(output)
        if row.get("status") == "ok"
        and row.get("model") == model
        and row.get("vision_model") == vision_model
        and row.get("source_manifest_sha256") == manifest_hash
        and row.get("prompt_sha256") == prompt_sha256()
    }


def latest_cached_vision(
    output: Path, model: str, vision_model: str, manifest_hash: str
) -> dict[str, tuple[dict[str, Any], dict[str, Any] | None]]:
    cached: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for row in read_jsonl(output):
        evidence = row.get("vision_evidence")
        if (
            row.get("model") != model
            or row.get("vision_model") != vision_model
            or row.get("source_manifest_sha256") != manifest_hash
            or row.get("prompt_sha256") != prompt_sha256()
            or not isinstance(evidence, dict)
        ):
            continue
        try:
            validate_vision_evidence(evidence)
        except ValueError:
            continue
        usage = row.get("usage")
        vision_usage = usage.get("vision_evidence") if isinstance(usage, dict) else None
        cached[str(row.get("base_sample_id"))] = (
            evidence,
            vision_usage if isinstance(vision_usage, dict) else None,
        )
    return cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--human-reviews", type=Path, default=DEFAULT_HUMAN_REVIEWS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--vision-max-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--include-human-reviewed",
        action="store_true",
        help="also model-audit samples whose latest human review is completed",
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = select_groups(args.manifest)
    human_completed = completed_human_review_ids(args.human_reviews)
    for group in groups:
        group["human_review_completed_at_selection"] = (
            str(group["base_sample_id"]) in human_completed
        )
    eligible_groups = [
        group
        for group in groups
        if args.include_human_reviewed
        or str(group["base_sample_id"]) not in human_completed
    ]
    missing_images = [
        group["image_filename"]
        for group in groups
        if not (args.image_dir / group["image_filename"]).is_file()
    ]
    manifest_hash = sha256(args.manifest)
    if args.check:
        model_completed = latest_successes(
            args.output, args.model, args.vision_model, manifest_hash
        )
        cached_vision = latest_cached_vision(
            args.output, args.model, args.vision_model, manifest_hash
        )
        eligible_ids = {str(group["base_sample_id"]) for group in eligible_groups}
        print(
            json.dumps(
                {
                    "groups": len(groups),
                    "zero_iou_task_cases": sum(len(group["zero_iou_cases"]) for group in groups),
                    "human_review_completed_groups": len(
                        {str(group["base_sample_id"]) for group in groups} & human_completed
                    ),
                    "model_eligible_groups": len(eligible_groups),
                    "model_completed_groups": len(eligible_ids & model_completed),
                    "model_pending_groups": len(eligible_ids - model_completed),
                    "cached_vision_groups": len(eligible_ids & set(cached_vision)),
                    "include_human_reviewed": args.include_human_reviewed,
                    "missing_images": missing_images,
                    "model": args.model,
                    "vision_model": args.vision_model,
                    "manifest_sha256": manifest_hash,
                    "prompt_sha256": prompt_sha256(),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return
    if missing_images:
        raise SystemExit(f"missing {len(missing_images)} images")
    if not args.api_key:
        raise SystemExit("DASHSCOPE_API_KEY or --api-key is required")
    pool = ClientPool(args.api_key, args.base_url, args.timeout)
    if args.probe:
        if not eligible_groups:
            raise SystemExit("no groups are eligible for model audit")
        record = audit_group(pool, args, eligible_groups[0], manifest_hash)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        if record["status"] != "ok":
            raise SystemExit(1)
        return

    completed = latest_successes(
        args.output, args.model, args.vision_model, manifest_hash
    )
    cached_vision = latest_cached_vision(
        args.output, args.model, args.vision_model, manifest_hash
    )
    pending = [
        group
        for group in eligible_groups
        if group["base_sample_id"] not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    for group in pending:
        cached = cached_vision.get(str(group["base_sample_id"]))
        if cached:
            group["_cached_vision_evidence"], group["_cached_vision_usage"] = cached
    print(
        f"Selected {len(groups)} groups; skipped {len(groups) - len(eligible_groups)} "
        f"human-reviewed; {len(completed)} model-audited; running {len(pending)} "
        f"with {args.workers} workers ({sum(group['base_sample_id'] in cached_vision for group in pending)} "
        f"reuse cached vision evidence).",
        flush=True,
    )
    lock = threading.Lock()
    ok_count = 0
    error_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_group, pool, args, group, manifest_hash): group
            for group in pending
        }
        for finished, future in enumerate(as_completed(futures), 1):
            record = future.result()
            append_jsonl(args.output, record, lock)
            if record["status"] == "ok":
                ok_count += 1
            else:
                error_count += 1
            print(
                f"[{finished}/{len(pending)}] {record['base_sample_id']} "
                f"{record['status']} (ok={ok_count}, error={error_count})",
                flush=True,
            )
    if error_count:
        raise SystemExit(f"completed with {error_count} errors; rerun to resume")


if __name__ == "__main__":
    main()
