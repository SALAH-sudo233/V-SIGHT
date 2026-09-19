"""Scoped repair for the verified 2,000-image training set.

This module is deliberately separate from the historical mixed runner:
- 500dev is excluded completely.
- The 2,000-image set is the preserved 1,996 source plus the four verified
  expansion images.
- A failed first validator result gets one fine-grained secondary audit.
- Only a secondary pass is published; uncertain results remain isolated.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw
from dotenv import dotenv_values
import yaml

ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = ROOT.parent
DATA = Path(r"C:/Users/30796/AppData/Local/Temp/grpo")
EXPANSION = AUDIT_ROOT / "expansion_2000"
MODEL = "gpt-6-astra"
RUN = AUDIT_ROOT / "auto_repair" / "runs" / "2000held_finegrained_v2"
SOURCE = RUN / "source_2000held.json"
EVENTS = RUN / "events.jsonl"
CORRECTED = RUN / "corrected.jsonl"
FIRST_RUN = AUDIT_ROOT / "auto_repair" / "runs" / "20260911T083146Z"
FIRST_EVENTS = FIRST_RUN / "events.jsonl"
EXCLUDED_SETS = {"500dev"}
ALLOWED_SETS = {"1996", "added4"}
TYPES = {"object", "co_occurrence", "attribute", "relation"}
SECONDARY_CHECKS = {
    "bbox_valid",
    "target_preserved",
    "grammar_ok",
    "no_internal_conflict",
    "natural_language",
    "single_factor_type_correct",
    "reference_clear",
    "absence_visually_certain",
    "pass",
}
RULES = """Preserve the original physical target and original bbox; never guess or move the box.
If target identity, box, or relation reference is uncertain, return needs_human.
The positive must uniquely identify the boxed target in the entire image.
The negative must have no matching target anywhere in the entire image, with visibly supported absence.
Do not accept a merely self-contradictory phrase as a valid negative.
Four types: object changes only the target object noun; attribute changes only one attribute;
relation retains target and a visible unique reference and changes only the relation;
co_occurrence adds only one absent plausible companion entity.
Avoid hidden properties, profession guesses, synonyms that do not create a true counterfactual,
and simultaneous multi-factor edits. Relation direction is target relative to reference.
The red box is fallible evidence, not a truth label."""
SECONDARY_PROMPT = """You are a second, fine-grained visual adjudicator. A previous validator did not accept
this candidate, but you must ignore its reason and make a fresh judgment from the two attached images.
The first image is the original scene; the second has the unchanged red candidate bbox.
Audit the candidate expressions in randomized A/B order. Return ONLY JSON with:
A_matches_box,B_matches_box,A_has_match_anywhere,B_has_match_anywhere,A_unique,B_unique
(each exactly yes|no|uncertain), bbox_valid,target_preserved,grammar_ok,no_internal_conflict,
natural_language,single_factor_type_correct,reference_clear,absence_visually_certain,pass
(each boolean), and a specific reason string.
The positive role is not disclosed. The original positive below is only a target-identity anchor,
not a truth label. Accept only if every required check is certain and pass=true."""


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canon(value).encode("utf-8")).hexdigest()


def content_id(row):
    return digest({key: row[key] for key in ("set", "sid", "ht", "img", "bbox", "pos", "neg")})


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def final_validation_failed(events):
    """Return the latest event for each content_id when it ended validation_failed."""
    latest = {}
    for event in events:
        content = event.get("content_id")
        if content:
            latest[content] = event
    return [event for event in latest.values() if event.get("status") == "validation_failed"]


def load_generation_candidate(event):
    """Load the original candidate; the secondary pass must not invent a new repair."""
    evidence_dir = Path(event["evidence_dir"])
    generation_path = evidence_dir / "generation.json"
    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    raw = payload.get("raw_response", {})
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError("generation_response_missing_choices")
    text = choices[0].get("message", {}).get("content", "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    candidate = json.loads(text)
    if not isinstance(candidate, dict):
        raise ValueError("generation_candidate_not_object")
    return candidate


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canon(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def scope_rows(rows):
    return [row for row in rows if row.get("set") in ALLOWED_SETS and row.get("set") not in EXCLUDED_SETS]


def normalize_source(row, set_name):
    return {
        "set": set_name,
        "sid": str(row["sample_id"]),
        "ht": str(row["hallucination_type"]),
        "img": str(row["image_filename"]),
        "bbox": list(row.get("positive_bbox", row.get("gt_bbox_xyxy", row["chosen_bbox_xyxy"]))),
        "pos": str(row.get("positive_text", row["chosen"])),
        "neg": str(row.get("negative_text", row["rejected"])),
    }


def build_source():
    original = json.loads((EXPANSION / "original_train1996.json").read_text(encoding="utf-8"))
    merged = json.loads((EXPANSION / "refcocog_train2000.expanded_v1.json").read_text(encoding="utf-8"))
    if merged[: len(original)] != original:
        raise AssertionError("2000 source does not preserve original 1996 prefix")
    rows = [normalize_source(row, "1996") for row in original]
    rows.extend(normalize_source(row, "added4") for row in merged[len(original) :])
    if len(rows) != 8000:
        raise AssertionError(len(rows))
    if len({(r["set"], r["sid"]) for r in rows}) != 8000:
        raise AssertionError("duplicate scoped source key")
    if any(r["set"] == "500dev" for r in rows):
        raise AssertionError("500dev leaked into 2000held source")
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(SOURCE, rows)
    return rows


def source_image(row, folder):
    path = DATA / "all_auditimgs" / Path(row["img"]).name
    if row["set"] == "added4":
        path = EXPANSION / "images" / Path(row["img"]).name
    if not path.exists():
        raise FileNotFoundError("missing_source_image")
    raw = path.read_bytes()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    box = row["bbox"]
    if len(box) != 4 or not all(isinstance(x, (int, float)) for x in box):
        raise ValueError("invalid_original_bbox")
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("invalid_original_bbox")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=93)
    boxed = buffer.getvalue()
    (folder / "original.jpg").write_bytes(raw)
    (folder / "boxed.jpg").write_bytes(boxed)
    return [raw, boxed], [width, height]


def call(prompt, images, output_path):
    home = Path(r"C:/Users/30796/AppData/Local/hermes")
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]
    secret = dotenv_values(home / ".env")[config["key_env"]]
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}} for image in images] + [{"type": "text", "text": prompt}]}],
    }
    provenance = {"model_requested": MODEL, "input_hash": digest(payload), "prompt": prompt, "image_hashes": [hashlib.sha256(image).hexdigest() for image in images], "started": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(output_path.with_name(output_path.stem + "_input.json"), provenance)
    request = urllib.request.Request(config["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=210) as response:
        raw = json.load(response)
    write_json(output_path, {**provenance, "raw_response": raw, "usage": raw.get("usage"), "model": raw.get("model")})
    text = raw["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("schema_not_object")
    return value


def accept_secondary(verdict, positive_label):
    negative_label = "B" if positive_label == "A" else "A"
    enum_fields = [f"{role}_{suffix}" for role in ("A", "B") for suffix in ("matches_box", "has_match_anywhere", "unique")]
    if not all(verdict.get(key) in {"yes", "no"} for key in enum_fields):
        return False
    if not all(verdict.get(key) is True for key in SECONDARY_CHECKS):
        return False
    return verdict.get(f"{positive_label}_matches_box") == "yes" and verdict.get(f"{positive_label}_has_match_anywhere") == "yes" and verdict.get(f"{positive_label}_unique") == "yes" and verdict.get(f"{negative_label}_matches_box") == "no" and verdict.get(f"{negative_label}_has_match_anywhere") == "no" and isinstance(verdict.get("reason"), str) and bool(verdict["reason"].strip())


def evidence_page(folder, row, candidate, verdict, status):
    esc = html.escape
    body = '<!doctype html><meta charset="utf-8"><title>V-SIGHT fine-grained repair</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:48%}td,th{padding:12px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>'
    body += f"<h1>{esc(status)}</h1><p>2000held only; 500dev excluded. Secondary fine-grained validation; same model, independent context, not human gold.</p><p>{esc(str((row['set'], row['sid'], row['ht'])))}</p><img src=\"original.jpg\"><img src=\"boxed.jpg\"><table><tr><th></th><th>Before</th><th>After</th></tr>"
    for label in ("pos", "neg"):
        body += f"<tr><th>{label}</th><td>{esc(row[label])}</td><td>{esc(candidate.get(label, ''))}</td></tr>"
    body += "</table><h2>Candidate</h2><pre>" + esc(json.dumps(candidate, ensure_ascii=False, indent=2)) + "</pre><h2>Secondary validation</h2><pre>" + esc(json.dumps(verdict, ensure_ascii=False, indent=2)) + "</pre>"
    (folder / "before_after.html").write_text(body, encoding="utf-8")


def run_one(row, first_event, attempt):
    cid = content_id(row)
    folder = RUN / "evidence" / cid / str(attempt)
    folder.mkdir(parents=True, exist_ok=True)
    result = {"content_id": cid, "key": [row["set"], row["sid"], row["ht"]], "attempt": attempt, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "evidence_dir": str(folder), "secondary_validation": True}
    write_json(folder / "source.json", {"source": row, "first_event": first_event})
    try:
        images, size = source_image(row, folder)
        positive_label = "B" if int(digest([cid, "secondary-v1"]), 16) % 2 else "A"
        negative_label = "B" if positive_label == "A" else "A"
        candidate = load_generation_candidate(first_event)
        source_generation_dir = Path(first_event["evidence_dir"])
        for filename in ("generation.json", "generation_input.json"):
            source_file = source_generation_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, folder / ("source_" + filename))
        write_json(folder / "candidate_source.json", {
            "source_evidence_dir": str(source_generation_dir),
            "candidate": candidate,
            "mode": "reuse_existing_generation_for_fine_grained_secondary_validation",
        })
        if candidate.get("status") == "needs_human":
            result.update(status="needs_human", reason=candidate.get("reason")); evidence_page(folder, row, candidate, {}, result["status"]); return result
        if candidate.get("status") != "proposed" or any(not isinstance(candidate.get(k), str) or not candidate[k].strip() for k in ("pos", "neg", "reason")):
            raise ValueError("invalid_generation_schema")
        if candidate["pos"] == row["pos"] and candidate["neg"] == row["neg"]:
            result.update(status="needs_human", reason="secondary_generator_returned_unchanged_pair"); evidence_page(folder, row, candidate, {}, result["status"]); return result
        expressions = {positive_label: candidate["pos"], negative_label: candidate["neg"]}
        validation_prompt = SECONDARY_PROMPT + "\nTarget type: " + row["ht"] + "\nOriginal target anchor: " + row["pos"] + "\nOriginal bbox: " + canon(row["bbox"]) + "\nUnlabeled randomized expressions: " + canon(expressions)
        verdict = call(validation_prompt, images, folder / "validation.json")
        write_json(folder / "mapping.json", {"positive_label": positive_label, "expressions": expressions, "method": "sha256 deterministic secondary randomization"})
        status = "accepted" if accept_secondary(verdict, positive_label) else "validation_failed"
        evidence_page(folder, row, candidate, verdict, status)
        result.update(status=status, reason=verdict.get("reason"), positive_label=positive_label)
        if status == "accepted":
            corrected = {**row, "pos": candidate["pos"], "neg": candidate["neg"], "content_id": cid, "original_pos": row["pos"], "original_neg": row["neg"], "repair_version": RUN.name, "evidence_dir": str(folder), "validation": verdict, "validation_positive_label": positive_label, "verification_kind": "same_model_fine_grained_secondary_context_not_human_gold"}
            write_json(folder / "accepted.json", corrected)
            existing = {item["content_id"] for item in read_jsonl(CORRECTED)}
            if cid not in existing:
                append_jsonl(CORRECTED, corrected)
        return result
    except FileNotFoundError as exc:
        result.update(status="needs_human", reason=str(exc), error_type=type(exc).__name__); return result
    except Exception as exc:
        result.update(status="error", reason=str(exc)[:300], error_type=type(exc).__name__); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=0)
    args = parser.parse_args()
    rows = build_source()
    source_map = {content_id(row): row for row in rows}
    first_events = read_jsonl(FIRST_EVENTS)
    eligible = {}
    for event in final_validation_failed(first_events):
        cid = event.get("content_id")
        row = source_map.get(cid)
        if row is not None:
            eligible[cid] = event
    historical = {
        e.get("content_id")
        for e in read_jsonl(EVENTS)
        if e.get("status") in {"accepted", "needs_human", "validation_failed"}
    }
    todo = [(source_map[cid], event) for cid, event in eligible.items() if cid not in historical]
    RUN.mkdir(parents=True, exist_ok=True)
    count = 0
    for row, event in todo:
        result = run_one(row, event, 1)
        append_jsonl(EVENTS, result)
        count += 1
        if args.max_items and count >= args.max_items:
            break
    print(json.dumps({"scope_rows": len(rows), "eligible_final_validation_failed": len(eligible), "processed": count, "run": str(RUN)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
