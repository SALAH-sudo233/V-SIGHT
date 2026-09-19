#!/usr/bin/env python3
"""Build strictly paired ROH/BOH grounding benchmark expansions.

Each benchmark group has exactly one immutable positive expression and bbox.
The four negative rows are counterfactual modifications of that same positive:
object, co_occurrence, attribute, and relation. Source files are never
overwritten. Generation is resumable through a JSONL state file.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from openai import OpenAI


HALLUCINATION_TYPES = ("object", "co_occurrence", "attribute", "relation")
TYPE_SUFFIX = {
    "object": "obj",
    "co_occurrence": "cooc",
    "attribute": "attr",
    "relation": "rel",
}
SUFFIX_RE = re.compile(r"_(cooc|attr|rel)$")

SYSTEM_PROMPT = """You construct publication-quality visual grounding counterfactuals.

The immutable positive referring expression identifies the object inside the supplied bbox. Generate only the requested negative types. Every negative must be a natural referring expression derived from that exact positive target, not a description of a different visible target.

Definitions:
- object: replace the positive target head object with an implausible/absent object. Preserve the positive context as much as grammar permits. The replacement object must not exist at that referred location.
- co_occurrence: replace or extend the target with a contextually plausible but absent object that commonly co-occurs with the visible scene/anchor. Preserve shared context.
- attribute: keep the same target identity and relations; change or add exactly one visually false target attribute.
- relation: keep the same target identity; change or add exactly one false relation to a visible anchor object.

Return pure JSON with this schema:
{
  "base_positive_text": "exactly copied input expression",
  "positive_parse": {
    "head_object": "target head noun",
    "related_objects": ["object"],
    "attributes": [{"object": "object", "attribute": "attribute"}],
    "relations": [{"subject": "object", "relation": "relation", "object": "object"}]
  },
  "negatives": {
    "TYPE": {
      "negative_text": "counterfactual expression",
      "negative_parse": {
        "head_object": "head noun",
        "related_objects": ["object"],
        "attributes": [{"object": "object", "attribute": "attribute"}],
        "relations": [{"subject": "object", "relation": "relation", "object": "object"}]
      },
      "changed_positive_span": "text changed or empty for insertion",
      "replacement_span": "new text",
      "hallucination_object": "object or empty",
      "wrong_attribute": "attribute or empty",
      "correct_attribute": "attribute or empty",
      "subject": "relation subject or empty",
      "wrong_relation": "relation or empty",
      "correct_relation": "relation or empty",
      "target_object": "relation object/anchor or empty",
      "shared_objects": ["objects shared with the positive"],
      "rationale": "why the modified claim is false in this image",
      "positive_target_visible": true,
      "negative_claim_false": true,
      "target_identity_preserved": true,
      "only_intended_factor_changed": true
    }
  }
}

For object/co_occurrence, target_identity_preserved must be false because the target head changes. For attribute/relation it must be true. All other boolean checks must be true. Do not return markdown."""

COMPACT_SYSTEM_PROMPT = """Create minimal false referring expressions from one true target in the image.
object: replace its head with an absent object; keep context. co_occurrence: use a plausible absent co-occurring object; keep context. attribute: keep identity/relation, change or add one false visual attribute. relation: keep identity, change or add one false relation to a visible anchor.
Pure JSON only, using requested types:
{"p":"exact input positive","n":{"TYPE":{"t":"negative text","o":"absent object","a":"false attribute","s":"relation subject","r":"false relation","g":"relation anchor","w":"short reason false"}}}
Use o for object/co_occurrence, a for attribute, and s/r/g for relation. Omit irrelevant short keys. Claims must be false in the image."""


def normalize_type(value: Any) -> str:
    value = str(value or "").strip().lower().replace("-", "_")
    if value in {"cooccurrence", "co_occur", "cooccur"}:
        return "co_occurrence"
    return value


def normalize_text(value: Any) -> str:
    return re.sub(r"\W+", " ", str(value or "").lower(), flags=re.UNICODE).strip()


def positive_text(row: dict[str, Any]) -> str:
    return str(row.get("positive_text") or row.get("chosen") or "").strip()


def negative_text(row: dict[str, Any]) -> str:
    return str(row.get("negative_text") or row.get("rejected") or "").strip()


def get_base_id(row: dict[str, Any]) -> str:
    explicit = row.get("base_sample_id")
    if explicit:
        return str(explicit)
    sample_id = str(row.get("sample_id") or "")
    if row.get("expansion_method") == "original_reclassified":
        return sample_id
    return SUFFIX_RE.sub("", sample_id)


def canonicalize_bbox(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    result = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return result if result[2] > result[0] and result[3] > result[1] else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return rows


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


def find_bases(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bases = {
        str(row.get("sample_id")): row
        for row in rows
        if row.get("expansion_method") == "original_reclassified"
    }
    if not bases:
        # Also accept a repaired flat file, where all rows explicitly name a base.
        for row in rows:
            group_id = get_base_id(row)
            if group_id and group_id not in bases:
                bases[group_id] = row
    return bases


def select_existing_pairs(
    rows: list[dict[str, Any]], bases: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[get_base_id(row)].append(row)

    selected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group_id, base in bases.items():
        base_positive = normalize_text(positive_text(base))
        candidates = sorted(
            grouped.get(group_id, []),
            key=lambda row: row.get("expansion_method") != "original_reclassified",
        )
        for row in candidates:
            hallu_type = normalize_type(row.get("hallucination_type"))
            if hallu_type not in HALLUCINATION_TYPES or hallu_type in selected[group_id]:
                continue
            if not negative_text(row):
                continue
            if normalize_text(positive_text(row)) != base_positive:
                continue
            selected[group_id][hallu_type] = row
    return selected


def generation_manifest(
    bases: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, dict[str, Any]]],
    regenerate_all: bool,
) -> list[dict[str, Any]]:
    manifest = []
    for group_id, base in bases.items():
        bbox = canonicalize_bbox(
            base.get("gt_bbox_xyxy") or base.get("positive_bbox") or base.get("chosen_bbox_xyxy")
        )
        available = set() if regenerate_all else set(selected.get(group_id, {}))
        missing = [kind for kind in HALLUCINATION_TYPES if kind not in available]
        manifest.append({
            "base_sample_id": group_id,
            "image_filename": base.get("image_filename"),
            "positive_text": positive_text(base),
            "gt_bbox_xyxy": bbox,
            "existing_strict_types": sorted(available),
            "missing_types": missing,
            "generation_required": bool(missing),
            "input_valid": bool(positive_text(base) and bbox and base.get("image_filename")),
        })
    return manifest


def load_generation_state(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid state JSONL at line {line_number}: {exc}") from exc
            if record.get("status") == "ok":
                records[str(record["base_sample_id"])] = record
    return records


def state_covers(record: Optional[dict[str, Any]], required_types: Iterable[str]) -> bool:
    if not record or record.get("status") != "ok":
        return False
    negatives = record.get("output", {}).get("negatives", {})
    return all(hallu_type in negatives for hallu_type in required_types)


def append_state(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_parse(value: Any, label: str) -> list[str]:
    errors = []
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    if not str(value.get("head_object") or "").strip():
        errors.append(f"{label}.head_object is empty")
    for key in ("related_objects", "attributes", "relations"):
        if not isinstance(value.get(key), list):
            errors.append(f"{label}.{key} must be a list")
    return errors


def validate_generation(
    output: dict[str, Any], base: dict[str, Any], requested_types: Iterable[str], compact: bool = False
) -> list[str]:
    errors = []
    base_positive = positive_text(base)
    if normalize_text(output.get("base_positive_text")) != normalize_text(base_positive):
        errors.append("base_positive_text does not exactly match the immutable positive")
    if not compact:
        errors.extend(validate_parse(output.get("positive_parse"), "positive_parse"))
    negatives = output.get("negatives")
    if not isinstance(negatives, dict):
        return errors + ["negatives must be an object"]

    seen = set()
    for hallu_type in requested_types:
        item = negatives.get(hallu_type)
        if not isinstance(item, dict):
            errors.append(f"missing negatives.{hallu_type}")
            continue
        text = str(item.get("negative_text") or "").strip()
        normalized = normalize_text(text)
        if not text or normalized == normalize_text(base_positive):
            errors.append(f"{hallu_type}.negative_text is empty or unchanged")
        if normalized in seen:
            errors.append(f"{hallu_type}.negative_text duplicates another type")
        seen.add(normalized)
        if not compact:
            errors.extend(validate_parse(item.get("negative_parse"), f"{hallu_type}.negative_parse"))
        flags = ("negative_claim_false", "only_intended_factor_changed")
        if not compact:
            flags = ("positive_target_visible",) + flags
        for flag in flags:
            if item.get(flag) is not True:
                errors.append(f"{hallu_type}.{flag} must be true")
        identity_expected = hallu_type in {"attribute", "relation"}
        if not compact and item.get("target_identity_preserved") is not identity_expected:
            errors.append(
                f"{hallu_type}.target_identity_preserved must be {str(identity_expected).lower()}"
            )
        if hallu_type == "attribute" and not str(item.get("wrong_attribute") or "").strip():
            errors.append("attribute.wrong_attribute is empty")
        if hallu_type in {"object", "co_occurrence"} and not str(item.get("hallucination_object") or "").strip():
            errors.append(f"{hallu_type}.hallucination_object is empty")
        if hallu_type == "relation":
            for key in ("subject", "wrong_relation", "target_object"):
                if not str(item.get(key) or "").strip():
                    errors.append(f"relation.{key} is empty")
    return errors


def normalize_compact_generation(output: dict[str, Any], base_positive: str) -> dict[str, Any]:
    if "base_positive_text" in output and "negatives" in output:
        output["base_positive_text"] = base_positive
        return output
    negatives = {}
    for hallu_type, item in (output.get("n") or {}).items():
        if not isinstance(item, dict):
            continue
        negatives[hallu_type] = {
            "negative_text": item.get("t", ""),
            "hallucination_object": item.get("o", ""),
            "wrong_attribute": item.get("a", ""),
            "subject": item.get("s", ""),
            "wrong_relation": item.get("r", ""),
            "target_object": item.get("g", ""),
            "shared_objects": [],
            "rationale": item.get("w", ""),
            "negative_claim_false": True,
            "only_intended_factor_changed": True,
        }
    return {
        "base_positive_text": base_positive,
        "negatives": negatives,
    }


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate_for_base(
    base: dict[str, Any], requested_types: list[str], model_name: str, args: argparse.Namespace
) -> dict[str, Any]:
    image_path = args.image_dir / str(base.get("image_filename"))
    if not image_path.exists():
        return {
            "status": "error",
            "base_sample_id": get_base_id(base),
            "requested_types": requested_types,
            "error": f"missing image: {image_path}",
        }
    bbox = canonicalize_bbox(
        base.get("gt_bbox_xyxy") or base.get("positive_bbox") or base.get("chosen_bbox_xyxy")
    )
    if not bbox or not positive_text(base):
        return {
            "status": "error",
            "base_sample_id": get_base_id(base),
            "requested_types": requested_types,
            "error": "base positive text or bbox is invalid",
        }

    media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    image_url = f"data:{media_type};base64,{encode_image(image_path)}"
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    previous_errors: list[str] = []
    for attempt in range(1, args.retries + 1):
        correction = ""
        if previous_errors:
            correction = "\nPrevious response failed validation. Fix these errors:\n- " + "\n- ".join(previous_errors)
        if args.compact_output:
            request = (
                f"Positive: {positive_text(base)}\nBBox xyxy: {bbox}\n"
                f"Types: {','.join(requested_types)}" + correction
            )
        else:
            request = (
                f"Immutable positive expression: {positive_text(base)}\n"
                f"Positive target bbox in image pixel xyxy coordinates: {bbox}\n"
                f"Generate these types only: {json.dumps(requested_types)}\n"
                "Inspect the image, preserve the referred target as required, and return the exact JSON schema."
                + correction
            )
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": COMPACT_SYSTEM_PROMPT if args.compact_output else SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": request},
                    ]},
                ],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if not args.enable_thinking:
                request_kwargs["extra_body"] = {"enable_thinking": False}
            response = client.chat.completions.create(**request_kwargs)
            raw = response.choices[0].message.content or ""
            output = extract_json(raw)
            if args.compact_output:
                output = normalize_compact_generation(output, positive_text(base))
            errors = validate_generation(output, base, requested_types, compact=args.compact_output)
            if not errors:
                usage = getattr(response, "usage", None)
                return {
                    "status": "ok",
                    "base_sample_id": get_base_id(base),
                    "requested_types": requested_types,
                    "model": model_name,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    },
                    "output": output,
                }
            previous_errors = errors
        except Exception as exc:  # API and malformed response errors are retryable.
            previous_errors = [f"{type(exc).__name__}: {exc}"]
        if attempt < args.retries:
            time.sleep(args.retry_delay * attempt)
    return {
        "status": "error",
        "base_sample_id": get_base_id(base),
        "requested_types": requested_types,
        "model": model_name,
        "error": "; ".join(previous_errors),
    }


def parse_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_target_eval(
    hallu_type: str,
    positive_parse: dict[str, Any],
    item: dict[str, Any],
    fallback_target_eval: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    fallback_target_eval = fallback_target_eval or {}
    negative_parse = item.get("negative_parse") or {}
    positive_head = positive_parse.get("head_object") or fallback_target_eval.get("positive_head_object", "")
    positive_related = parse_list(positive_parse.get("related_objects")) or parse_list(
        fallback_target_eval.get("positive_related_objects")
    )
    hallucination_unit = {
        "type": hallu_type,
        "object": item.get("hallucination_object", ""),
        "attribute": item.get("wrong_attribute", ""),
        "subject": item.get("subject", ""),
        "relation": item.get("wrong_relation", ""),
        "target_object": item.get("target_object", ""),
        "count_as_hallucination_if_mentioned": True,
        "note": item.get("rationale", ""),
    }
    return {
        "eval_type": hallu_type,
        "positive_head_object": positive_head,
        "positive_related_objects": positive_related,
        "positive_attributes": parse_list(positive_parse.get("attributes")) or parse_list(fallback_target_eval.get("positive_attributes")),
        "positive_relations": parse_list(positive_parse.get("relations")) or parse_list(fallback_target_eval.get("positive_relations")),
        "negative_head_object": negative_parse.get("head_object", item.get("hallucination_object", positive_head)),
        "negative_related_objects": parse_list(negative_parse.get("related_objects")),
        "negative_attributes": parse_list(negative_parse.get("attributes")),
        "negative_relations": parse_list(negative_parse.get("relations")),
        "shared_objects": parse_list(item.get("shared_objects")),
        "target_coverage_units": [positive_head] if positive_head else [],
        "hallucination_units": [hallucination_unit],
    }


def make_pair_row(
    base_id: str,
    base: dict[str, Any],
    hallu_type: str,
    source_row: Optional[dict[str, Any]] = None,
    generated: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    bbox = canonicalize_bbox(
        base.get("gt_bbox_xyxy") or base.get("positive_bbox") or base.get("chosen_bbox_xyxy")
    )
    positive = positive_text(base)
    row = copy.deepcopy(base)
    if generated is None:
        if source_row is None:
            raise ValueError("source_row or generated is required")
        negative = negative_text(source_row)
        annotation = copy.deepcopy(source_row.get("chair_annotation") or base.get("chair_annotation") or {})
        origin = "existing_pair_aligned"
        source_sample_id = source_row.get("sample_id")
    else:
        item = generated["output"]["negatives"][hallu_type]
        negative = str(item["negative_text"]).strip()
        annotation = copy.deepcopy(base.get("chair_annotation") or {})
        annotation["version"] = "chair_v2_strict_pair"
        annotation["annotation_source"] = "vlm_image_counterfactual"
        fallback_target_eval = annotation.get("target_eval_units") or {}
        annotation["target_eval_units"] = build_target_eval(
            hallu_type,
            generated["output"].get("positive_parse") or {},
            item,
            fallback_target_eval,
        )
        annotation["quality_flags"] = {
            "needs_human_review": True,
            "reason": "VLM-generated strict counterfactual; semantic image verification required.",
        }
        annotation["annotator_model"] = generated.get("model")
        annotation["annotated_at"] = generated.get("generated_at")
        annotation["used_image"] = True
        origin = "generated_strict_counterfactual"
        source_sample_id = None

    row.update({
        "sample_id": f"{base_id}__{TYPE_SUFFIX[hallu_type]}",
        "base_sample_id": base_id,
        "pair_id": f"{base_id}::{hallu_type}",
        "chosen": positive,
        "positive_text": positive,
        "rejected": negative,
        "negative_text": negative,
        "hallucination_type": hallu_type,
        "difficulty_tier": hallu_type,
        "positive_bbox": bbox,
        "positive_bbox_format": "xyxy",
        "gt_bbox_xyxy": bbox,
        "chosen_bbox_xyxy": bbox,
        "expansion_method": "strict_paired_v2",
        "chair_annotation": annotation,
        "pair_metadata": {
            "schema_version": "roh_vcd_strict_pair_v2",
            "base_sample_id": base_id,
            "positive_is_shared_with_group": True,
            "positive_bbox_is_shared_with_group": True,
            "pair_alignment_validated": True,
            "semantic_status": "needs_human_review",
            "negative_origin": origin,
            "source_negative_sample_id": source_sample_id,
        },
    })
    return row


def export_repaired(
    bases: dict[str, dict[str, Any]],
    existing: dict[str, dict[str, dict[str, Any]]],
    generated: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    rows = []
    incomplete = []
    origin_counts = Counter()
    for base_id, base in bases.items():
        group_rows = []
        generated_record = generated.get(base_id)
        for hallu_type in HALLUCINATION_TYPES:
            generated_item = (generated_record or {}).get("output", {}).get("negatives", {}).get(hallu_type)
            if generated_item is not None:
                row = make_pair_row(base_id, base, hallu_type, generated=generated_record)
            elif hallu_type in existing.get(base_id, {}):
                row = make_pair_row(
                    base_id, base, hallu_type, source_row=existing[base_id][hallu_type]
                )
            else:
                continue
            group_rows.append(row)
            origin_counts[row["pair_metadata"]["negative_origin"]] += 1
        if len(group_rows) == len(HALLUCINATION_TYPES):
            rows.extend(group_rows)
        else:
            incomplete.append({
                "base_sample_id": base_id,
                "present_types": [row["hallucination_type"] for row in group_rows],
                "missing_types": [
                    kind for kind in HALLUCINATION_TYPES
                    if kind not in {row["hallucination_type"] for row in group_rows}
                ],
            })

    dump_json(output_path, rows)
    summary = {
        "schema_version": "roh_vcd_repaired_export_v2",
        "source_base_count": len(bases),
        "complete_group_count": len(rows) // len(HALLUCINATION_TYPES),
        "exported_pair_count": len(rows),
        "incomplete_group_count": len(incomplete),
        "negative_origin_counts": dict(origin_counts),
        "output_file": str(output_path),
        "incomplete_groups": incomplete,
    }
    dump_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=script_dir / "refcocog_500_expanded.json")
    parser.add_argument("--output", type=Path, default=script_dir / "repaired" / "refcocog_500_expanded.strict.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--image-dir", type=Path, default=script_dir / "benchmark_images")
    parser.add_argument("--model", default=os.environ.get("ROH_VCD_MODEL", "qwen3.7-plus"))
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model names to rotate deterministically across pending base groups",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "ROH_VCD_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROH_VCD_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Generate at most this many base groups in this run")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Use the quota-efficient atomic annotation schema",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model reasoning tokens; disabled by default to conserve annotation quota",
    )
    parser.add_argument("--regenerate-all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.models:
        args.models = [args.model]
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.dry_run and not args.api_key:
        parser.error(
            "ROH_VCD_API_KEY, DASHSCOPE_API_KEY, or --api-key is required unless --dry-run is used"
        )
    if args.manifest is None:
        args.manifest = args.output.with_suffix(".generation_manifest.json")
    if args.state is None:
        args.state = args.output.with_suffix(".generation_state.jsonl")
    return args


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    bases = find_bases(rows)
    existing = select_existing_pairs(rows, bases)
    manifest = generation_manifest(bases, existing, args.regenerate_all)
    dump_json(args.manifest, {
        "schema_version": "roh_vcd_generation_manifest_v2",
        "source_file": str(args.input),
        "base_count": len(bases),
        "required_group_count": sum(item["generation_required"] for item in manifest),
        "missing_slot_count": sum(len(item["missing_types"]) for item in manifest),
        "items": manifest,
    })

    state = load_generation_state(args.state)
    pending = [
        item for item in manifest
        if item["generation_required"]
        and item["input_valid"]
        and not state_covers(state.get(item["base_sample_id"]), item["missing_types"])
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"bases={len(bases)} existing_strict={sum(len(v) for v in existing.values())} "
        f"missing_slots={sum(len(item['missing_types']) for item in manifest)} "
        f"resumed={len(state)} pending_this_run={len(pending)}"
    )
    if args.dry_run:
        print(f"Dry run: wrote manifest to {args.manifest}")
        return

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_for_base,
                bases[item["base_sample_id"]],
                item["missing_types"],
                args.models[index % len(args.models)],
                args,
            ): item
            for index, item in enumerate(pending)
        }
        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "status": "error",
                    "base_sample_id": item["base_sample_id"],
                    "requested_types": item["missing_types"],
                    "error": f"worker failure: {type(exc).__name__}: {exc}",
                }
            append_state(args.state, record)
            completed += 1
            if record["status"] == "ok":
                state[record["base_sample_id"]] = record
            else:
                failures.append(record)
            print(
                f"[{completed}/{len(pending)}] {record['status']}: "
                f"{record['base_sample_id']} ({','.join(record['requested_types'])})"
            )

    summary = export_repaired(bases, existing, state, args.output)
    dump_json(args.output.with_suffix(".failures.json"), failures)
    print(
        f"exported_groups={summary['complete_group_count']} "
        f"exported_pairs={summary['exported_pair_count']} "
        f"incomplete_groups={summary['incomplete_group_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
