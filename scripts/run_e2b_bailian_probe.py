#!/usr/bin/env python3
"""Run a bounded randomized A/B Bailian probe for E2b relation binding.

The probe is deliberately calibration-only. It does not write the API key,
images, or hidden IoUs to its output and it never accesses repaired-1996.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SELECTOR = ROOT / "data/e2b/selector/e2b_selector.calibration.jsonl.gz"
DEFAULT_IMAGE_ROOT = Path("/home/u2025141034/models/LENS/data/refcoco/train2014")
DEFAULT_OUTPUT = ROOT / "data/e2b/teacher_probe/e2b_bailian_probe.jsonl"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max-preview"
DEFAULT_VISION_MODEL = "qwen3-vl-plus"

VISION_SYSTEM_PROMPT = r"""You are the visual evidence stage of a grounding
verifier. The user query names one target object. Candidate A and candidate B
are two proposed target boxes. Green boxes R1... are independent proposals for
the relation reference object; they are context only and are not target
candidates.

Inspect the pixels and report evidence separately for A and B. Do not choose a
winner. Do not use hidden annotations or assume that either candidate is
correct. Return pure JSON with exactly:
{
  "candidate_A":{"visible_object":"", "query_support":[""], "query_contradictions":[""]},
  "candidate_B":{"visible_object":"", "query_support":[""], "query_contradictions":[""]},
  "reference_evidence":[{"proposal":"R1", "visible_object":"", "relation_to_A":"", "relation_to_B":""}],
  "scene_evidence":[""],
  "uncertainties":[""]
}
Use empty arrays when evidence is unavailable and never return markdown.
"""

ADJUDICATION_SYSTEM_PROMPT = r"""You are the final decision stage of a visual
grounding verifier. You receive the query, candidate coordinates, and
structured visual evidence from a separate vision model. The evidence is the
only authority for visible facts. Choose the candidate that best satisfies the
complete query, including category, attributes, action, and target-reference
relation. Do not prefer a candidate by A/B label, position, or box size. Return
uncertain when the evidence does not support a reliable distinction.

Return pure JSON with exactly:
{"choice":"A|B|uncertain", "confidence":0.0, "reason":"short evidence-grounded reason"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(1, min(width, round(x2))),
        max(1, min(height, round(y2))),
    )


def image_data_url(row: Mapping[str, Any], image_root: Path, order: list[int]) -> str:
    path = image_root / Path(str(row["image_filename"])).name
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    marked = image.copy()
    draw = ImageDraw.Draw(marked, "RGBA")
    colors = [(220, 35, 35, 255), (35, 85, 220, 255)]
    labels = ["A", "B"]
    candidates = [row["baseline_bbox_xyxy"], row["challenger_bbox_xyxy"]]
    for display_index, original_index in enumerate(order):
        box = clamp_box(candidates[original_index], width, height)
        color = colors[display_index]
        line_width = max(3, min(width, height) // 100)
        draw.rectangle(box, outline=color, width=line_width)
        x1, y1, _, _ = box
        draw.rectangle((x1, max(0, y1 - 28), x1 + 30, y1), fill=color)
        draw.text((x1 + 8, max(0, y1 - 25)), labels[display_index], fill=(255, 255, 255, 255))
    for index, proposal in enumerate(row.get("reference_proposals") or [], 1):
        box = clamp_box(proposal["bbox_xyxy"], width, height)
        draw.rectangle(box, outline=(35, 170, 65, 210), width=max(2, line_width // 2))
        x1, y1, _, _ = box
        draw.text((x1, y1), f"R{index}", fill=(20, 120, 40, 255))
    marked.thumbnail((1600, 1600))
    buffer = BytesIO()
    marked.save(buffer, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_response(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain JSON")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    choice = str(value.get("choice") or "").lower()
    if choice not in {"a", "b", "uncertain"}:
        raise ValueError(f"invalid choice: {choice!r}")
    confidence = float(value.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return {"choice": choice, "confidence": confidence, "reason": str(value.get("reason") or "")[:500]}


def usage(response: Any) -> dict[str, Any]:
    value = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(value, "prompt_tokens", None),
        "completion_tokens": getattr(value, "completion_tokens", None),
        "total_tokens": getattr(value, "total_tokens", None),
    }


def call_json(client: Any, kwargs: Mapping[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], int]:
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            response = client.chat.completions.create(**dict(kwargs))
            return parse_json_object(response.choices[0].message.content or ""), usage(response), attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(min(30.0, args.retry_delay * (2 ** (attempt - 1))))
    raise RuntimeError(last_error)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain JSON")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    return value


def call_one(client: Any, row: Mapping[str, Any], image_root: Path, order: list[int], args: argparse.Namespace) -> dict[str, Any]:
    image_url = image_data_url(row, image_root, order)
    candidates = [row["baseline_bbox_xyxy"], row["challenger_bbox_xyxy"]]
    displayed = [candidates[index] for index in order]
    context = {
        "query": row["query"],
        "relation": row.get("relation"),
        "reference_phrase": row.get("reference_phrase"),
        "reference_proposals_xyxy": [item["bbox_xyxy"] for item in row.get("reference_proposals") or []],
        "candidate_A_xyxy": displayed[0],
        "candidate_B_xyxy": displayed[1],
    }
    vision_kwargs = {
        "model": args.vision_model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 1400,
    }
    try:
        evidence, vision_usage, vision_attempts = call_json(client, vision_kwargs, args)
    except Exception as exc:
        return {"status": "error", "error_stage": "vision", "order": order, "error": str(exc)}
    decision_kwargs = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"context": context, "visual_evidence": evidence}, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "extra_body": {"enable_thinking": True},
    }
    try:
        decision, decision_usage, decision_attempts = call_json(client, decision_kwargs, args)
        parsed = parse_response(json.dumps(decision))
    except Exception as exc:
        return {
            "status": "error",
            "error_stage": "adjudication",
            "order": order,
            "visual_evidence": evidence,
            "usage": {"vision": vision_usage},
            "error": str(exc),
        }
    return {
        "status": "ok",
        "choice": parsed["choice"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "order": order,
        "visual_evidence": evidence,
        "usage": {"vision": vision_usage, "adjudication": decision_usage},
        "attempts": {"vision": vision_attempts, "adjudication": decision_attempts},
    }


def metrics(rows: list[dict[str, Any]], results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from vsight.e2_verifier import selector_metrics

    scores: dict[str, float] = {}
    for row in rows:
        result = results.get(str(row["query_id"]))
        if not result or result.get("status") != "ok":
            continue
        choice = result["choice"]
        order = list(result["order"])
        selected_original = order[0] if choice == "a" else order[1] if choice == "b" else 0
        scores[str(row["query_id"])] = 1.0 if selected_original == 1 else 0.0
    return selector_metrics(rows, scores, 0.5)


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("DASHSCOPE_API_KEY or --api-key is required")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.force:
        args.output.unlink(missing_ok=True)
        args.output.with_suffix(".summary.json").unlink(missing_ok=True)
    rows = [row for row in read_rows(args.selector) if row.get("relation_selector_eligible")]
    by_task = {task: [row for row in rows if row.get("task") == task] for task in ("t2", "t4")}
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    per_task = min(args.limit // 2, len(by_task["t2"]), len(by_task["t4"]))
    selected.extend(rng.sample(by_task["t2"], per_task))
    selected.extend(rng.sample(by_task["t4"], per_task))
    selected.sort(key=lambda row: str(row["query_id"]))
    if len(selected) < args.limit:
        remaining = [row for row in rows if row not in selected]
        selected.extend(rng.sample(remaining, min(args.limit - len(selected), len(remaining))))
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("the openai package is required") from exc
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    existing = {
        str(item.get("query_id")): item
        for item in read_jsonl(args.output)
        if item.get("status") == "ok"
    }
    pending = [row for row in selected if str(row["query_id"]) not in existing]
    output_rows: list[dict[str, Any]] = list(existing.values())
    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for row in pending:
            order = [0, 1]
            if rng.random() < 0.5:
                order.reverse()
            futures[pool.submit(call_one, client, row, args.image_root, order, args)] = row
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            record = {
                "schema_version": "vsight_e2b_bailian_probe_row_v1",
                "query_id": str(row["query_id"]),
                "task": row["task"],
                "status": result.get("status"),
                "choice": result.get("choice"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "order": result.get("order"),
                "usage": result.get("usage"),
                "attempts": result.get("attempts"),
                "error_stage": result.get("error_stage"),
                "error": result.get("error"),
                "visual_evidence": result.get("visual_evidence"),
            }
            with lock:
                output_rows.append(record)
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                completed = len(output_rows)
            if completed % 10 == 0 or completed == len(selected):
                print(f"completed={completed}/{len(selected)}", flush=True)
    output_rows.sort(key=lambda item: item["query_id"])
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output_rows), encoding="utf-8")
    result_map = {item["query_id"]: item for item in output_rows}
    summary = {
        "schema_version": "vsight_e2b_bailian_probe_v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selector_sha256": sha256(args.selector),
        "model": args.model,
        "vision_model": args.vision_model,
        "queries": len(selected),
        "successful": sum(item.get("status") == "ok" for item in output_rows),
        "external_api_used": True,
        "sealed_repaired_1996_accessed": False,
        "metrics": {
            task: metrics([row for row in selected if row["task"] == task], result_map)
            for task in ("t2", "t4")
        },
        "output": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
