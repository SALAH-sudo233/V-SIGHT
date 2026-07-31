#!/usr/bin/env python3
"""Adjudicate minimal positive-expression repairs from audited evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/audits/zero_iou_127.template.jsonl"
DEFAULT_AUDIT = ROOT / "data/audits/zero_iou_attributes.qwen3.7-max-2026-05-17.jsonl"
DEFAULT_HUMAN = ROOT / "data/audits/zero_iou_127.reviews.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "data/audits/zero_iou_positive_repairs.qwen3.7-max-2026-05-17.jsonl"
)
DEFAULT_MODEL = "qwen3.7-max-2026-05-17"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DECISIONS = {"keep", "rewrite", "reject", "needs_human"}
TRUTHS = {"supported", "ambiguous", "contradicted", "uncertain"}
CONFIDENCE = {"high", "medium", "low"}
ATOM_TYPES = {"object", "color", "material", "attribute", "action_state", "relation"}
FORBIDDEN_METADATA = (
    "green box",
    "red box",
    "orange box",
    "ground truth",
    "gt target",
    "baseline",
    "panel",
    "pixel",
    "coordinate",
    "bbox",
)

REPAIR_SYSTEM_PROMPT = r"""You are a conservative text-only referring-expression
repair adjudicator. You receive the original expression, structured visual
evidence for the exact annotated target, and optional independent human review.
You do not see the image. Never invent a visual fact outside the supplied
evidence and never mention annotations, boxes, panels, coordinates, or model
outputs in a repaired expression.

Decide one action:
- keep: the source expression truthfully and sufficiently identifies the exact
  target. repaired_expression must be copied exactly.
- rewrite: the source refers to the target but is ambiguous or lacks a needed
  distinguishing cue. Make the smallest natural rewrite, adding only one or
  two cues copied or faithfully paraphrased from disambiguating_cues,
  visible_attributes, spatial_and_relational_facts, or supported query checks.
- reject: the source cannot truthfully refer to the annotated target, or the
  target/query appears to have an annotation problem. Do not fabricate a
  positive expression.
- needs_human: evidence is insufficient or conflicting for a safe decision.

Prefer reject/needs_human over a speculative rewrite. A same-category
confusion with query_binds_gt_target=yes is usually rewrite only when a
supported cue makes the target unique. A relation atom contradicted by visual
evidence must not be copied into a repaired expression. Preserve the target
head category and all true source atoms unless the evidence explicitly shows
that an atom is wrong. Do not use gender stereotypes; apparent_gender is only
an optional visible presentation cue and should normally not be added.

Return pure JSON exactly:
{
  "decision":"keep|rewrite|reject|needs_human",
  "repaired_expression":"",
  "head_object":"",
  "source_expression_truth":"supported|ambiguous|contradicted|uncertain",
  "added_atoms":[{"text":"", "type":"object|color|material|attribute|action_state|relation", "evidence_cue":""}],
  "removed_or_replaced_atoms":[{"source_text":"", "replacement_text":"", "reason":""}],
  "evidence_citations":["exact or faithful short cue used"],
  "confidence":"high|medium|low",
  "reason":"short evidence-grounded explanation",
  "rejection_reason":"empty unless decision is reject or needs_human"
}
For keep, use empty added/replaced arrays and copy the source exactly. For
reject/needs_human, repaired_expression and head_object must be empty. Do not
return markdown."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256() -> str:
    return hashlib.sha256(REPAIR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("repair response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("repair response must be a JSON object")
    return value


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def validate_repair(value: Mapping[str, Any], source_expression: str) -> None:
    decision = value.get("decision")
    if decision not in DECISIONS:
        raise ValueError("decision is invalid")
    truth = value.get("source_expression_truth")
    if truth not in TRUTHS:
        raise ValueError("source_expression_truth is invalid")
    if value.get("confidence") not in CONFIDENCE:
        raise ValueError("confidence is invalid")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason is empty")
    citations = value.get("evidence_citations")
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise ValueError("evidence_citations must be a string list")
    for field in ("added_atoms", "removed_or_replaced_atoms"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"{field} must be a list")
    for index, item in enumerate(value["added_atoms"]):
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            raise ValueError(f"added_atoms[{index}] is malformed")
        if item.get("type") not in ATOM_TYPES:
            raise ValueError(f"added_atoms[{index}].type is invalid")
        if not str(item.get("evidence_cue") or "").strip():
            raise ValueError(f"added_atoms[{index}].evidence_cue is empty")
    for index, item in enumerate(value["removed_or_replaced_atoms"]):
        if not isinstance(item, dict) or not str(item.get("source_text") or "").strip():
            raise ValueError(f"removed_or_replaced_atoms[{index}] is malformed")
        if not str(item.get("reason") or "").strip():
            raise ValueError(f"removed_or_replaced_atoms[{index}].reason is empty")
    repaired = str(value.get("repaired_expression") or "").strip()
    head = str(value.get("head_object") or "").strip()
    if decision == "keep":
        if repaired != source_expression:
            raise ValueError("keep must copy source expression exactly")
        if not head:
            raise ValueError("keep head_object is empty")
        if value["added_atoms"] or value["removed_or_replaced_atoms"]:
            raise ValueError("keep cannot contain edits")
    elif decision == "rewrite":
        if not repaired or repaired == source_expression:
            raise ValueError("rewrite must change source expression")
        if not head:
            raise ValueError("rewrite head_object is empty")
        if not value["added_atoms"] and not value["removed_or_replaced_atoms"]:
            raise ValueError("rewrite must describe an edit")
    else:
        if repaired or head:
            raise ValueError("reject/needs_human must not create a positive")
        if not str(value.get("rejection_reason") or "").strip():
            raise ValueError("reject/needs_human requires rejection_reason")
    lowered = normalize_text(repaired)
    if any(term in lowered for term in FORBIDDEN_METADATA):
        raise ValueError("repaired expression contains forbidden metadata")
    if decision in {"reject", "needs_human"} and value["evidence_citations"]:
        # Citations are useful for the reason but cannot masquerade as a repair.
        pass


def latest_audits(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "ok":
            latest[str(row["base_sample_id"])] = row
    return latest


def latest_human(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[(str(row.get("base_sample_id")), str(row.get("reviewer_id")))] = row
    completed: dict[str, dict[str, Any]] = {}
    for (sample_id, _), row in latest.items():
        if row.get("status") == "completed":
            completed[sample_id] = row
    return completed


def build_request(
    group: Mapping[str, Any], audit_row: Mapping[str, Any], human_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "base_sample_id": group["base_sample_id"],
        "source_expression": group["query"],
        "target_category_hint": group.get("target_category") or "",
        "expression_structure": group.get("expression_structure") or "",
        "same_category_distractors_hint": group.get("same_category_distractors"),
        "audited_tasks": audit_row.get("audited_tasks") or [],
        "verified_vision_evidence": audit_row["vision_evidence"],
        "max_adjudication": audit_row["audit"],
        "human_review": human_row.get("case_reviews") if human_row else None,
        "human_query_support": human_row.get("query_support") if human_row else None,
    }


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
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self.local.client


def call_repair(
    pool: ClientPool, args: argparse.Namespace, request: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = str(request["source_expression"])
    response = pool.get().chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=args.max_tokens,
        extra_body={"enable_thinking": not args.disable_thinking},
    )
    raw = response.choices[0].message.content or ""
    value = parse_json_response(raw)
    validate_repair(value, source)
    usage = response.usage
    return value, {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def repair_group(
    pool: ClientPool,
    args: argparse.Namespace,
    group: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    human_row: Mapping[str, Any] | None,
    manifest_hash: str,
    audit_hash: str,
    human_hash: str,
) -> dict[str, Any]:
    request = build_request(group, audit_row, human_row)
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            repair, usage = call_repair(pool, args, request)
            return {
                "schema_version": "vsight_zero_iou_positive_repair_v1",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "ok",
                "base_sample_id": group["base_sample_id"],
                "image_filename": group["image_filename"],
                "source_expression": group["query"],
                "model": args.model,
                "source_manifest_sha256": manifest_hash,
                "source_attribute_audit_sha256": audit_hash,
                "source_human_review_sha256": human_hash,
                "repair_prompt_sha256": prompt_sha256(),
                "attempt": attempt,
                "repair": repair,
                "request": request,
                "usage": usage,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(min(20.0, args.retry_delay * (2 ** (attempt - 1))))
    return {
        "schema_version": "vsight_zero_iou_positive_repair_v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "error",
        "base_sample_id": group["base_sample_id"],
        "image_filename": group["image_filename"],
        "source_expression": group["query"],
        "model": args.model,
        "source_manifest_sha256": manifest_hash,
        "source_attribute_audit_sha256": audit_hash,
        "source_human_review_sha256": human_hash,
        "repair_prompt_sha256": prompt_sha256(),
        "attempt": args.retries,
        "error": last_error,
    }


def latest_successes(path: Path, model: str, manifest_hash: str, audit_hash: str) -> set[str]:
    return {
        str(row.get("base_sample_id"))
        for row in read_jsonl(path)
        if row.get("status") == "ok"
        and row.get("model") == model
        and row.get("source_manifest_sha256") == manifest_hash
        and row.get("source_attribute_audit_sha256") == audit_hash
        and row.get("repair_prompt_sha256") == prompt_sha256()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attribute-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--human-reviews", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = {
        str(row["base_sample_id"]): row
        for row in read_jsonl(args.manifest)
        if any(
            case.get("baseline_box") is not None
            and float(case.get("baseline_iou", -1)) == 0.0
            for case in row.get("cases") or []
        )
    }
    audits = latest_audits(args.attribute_audit)
    human = latest_human(args.human_reviews)
    manifest_hash = sha256(args.manifest)
    audit_hash = sha256(args.attribute_audit)
    human_hash = sha256(args.human_reviews)
    eligible = sorted(set(groups) & set(audits))
    completed = latest_successes(args.output, args.model, manifest_hash, audit_hash)
    pending = [sample_id for sample_id in eligible if sample_id not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    if args.check:
        print(
            json.dumps(
                {
                    "eligible_groups_with_attribute_audit": len(eligible),
                    "completed_repairs": len(set(eligible) & completed),
                    "pending_repairs": len(set(pending)),
                    "attribute_audit_rows": len(audits),
                    "model": args.model,
                    "repair_prompt_sha256": prompt_sha256(),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return
    if not args.api_key:
        raise SystemExit("DASHSCOPE_API_KEY or --api-key is required")
    if not pending:
        print("No pending positive repairs.")
        return
    pool = ClientPool(args.api_key, args.base_url, args.timeout)
    print(
        f"Eligible {len(eligible)} groups; {len(completed)} repairs complete; "
        f"running {len(pending)} with {args.workers} workers.",
        flush=True,
    )
    lock = threading.Lock()
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                repair_group,
                pool,
                args,
                groups[sample_id],
                audits[sample_id],
                human.get(sample_id),
                manifest_hash,
                audit_hash,
                human_hash,
            ): sample_id
            for sample_id in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
            if record["status"] != "ok":
                errors += 1
            print(
                f"[{index}/{len(pending)}] {record['base_sample_id']} "
                f"{record['status']} (errors={errors})",
                flush=True,
            )
    if errors:
        raise SystemExit(f"completed with {errors} errors; rerun to resume")


if __name__ == "__main__":
    main()
