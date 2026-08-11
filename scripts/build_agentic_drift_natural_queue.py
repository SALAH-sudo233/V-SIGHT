#!/usr/bin/env python3
"""Build a correctness-blind natural Agentic Drift review cohort."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.cable import relation_family  # noqa: E402
from vsight.ccv import AtomType, TypedClaimParser  # noqa: E402

DEFAULT_AUDIT = ROOT / "outputs/agentic_drift_shadow_v2_nomemory/audit.jsonl.gz"
DEFAULT_IMAGES = Path("/home/u2025141034/benchmark/benchmark_images")
DEFAULT_OUTPUT = ROOT / "data/e3/agentic_drift_natural_review_v1"
STRATUM_TARGETS = {"relation": 180, "attribute": 40, "action": 40, "object": 40}
MODEL_ORDER = (
    "LENS", "Orsta-7B", "Qwen3-VL-8B", "Seg-R1", "Seg-zero", "TreeVGR",
    "UniVG-R1", "Vision-R1", "VisionReasoner", "qwen2.5-vl-7b", "visual-rft",
)
FORBIDDEN_QUEUE_FIELDS = frozenset(
    {
        "model", "task", "sample_id", "source_record_id", "evidence_state",
        "action", "drift_risk_raw", "review_priority", "reason_codes",
        "alternative_bbox", "claim_support", "alternative_support",
        "binding_margin", "memory_status", "trace", "gt_bbox_xyxy", "iou",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audit(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deterministic_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")


def query_metadata(query: str) -> tuple[list[str], str, str | None, str | None]:
    claim = TypedClaimParser().parse(query)
    atom_types = {atom.atom_type for atom in claim.known_atoms}
    applicable = ["identity"]
    if AtomType.ATTRIBUTE in atom_types:
        applicable.append("attribute")
    if AtomType.ACTION in atom_types:
        applicable.append("action")
    relation_atom = next((atom for atom in claim.known_atoms if atom.atom_type is AtomType.RELATION), None)
    if relation_atom is not None:
        applicable.append("relation")
        stratum = "relation"
    elif AtomType.ACTION in atom_types:
        stratum = "action"
    elif AtomType.ATTRIBUTE in atom_types:
        stratum = "attribute"
    else:
        stratum = "object"
    return (
        applicable,
        stratum,
        relation_family(relation_atom.relation) if relation_atom else None,
        relation_atom.reference_text if relation_atom else None,
    )


def _key(row: dict[str, Any]) -> tuple[str, str]:
    context = row.get("context") or {}
    return str(context.get("group_id") or ""), str(context.get("task") or "")


def _stable_rank(seed: str, key: tuple[str, str]) -> str:
    return hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).hexdigest()


def select_natural_rows(
    rows: list[dict[str, Any]],
    *,
    seed: str = "agentic-natural-v1",
    stratum_targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Select unique image-group/task rows without reading agent outcomes."""

    targets = dict(stratum_targets or STRATUM_TARGETS)
    if any(value < 0 for value in targets.values()):
        raise ValueError("stratum targets cannot be negative")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in rows:
        key = _key(row)
        if not all(key):
            continue
        context = row.get("context") or {}
        model = str(context.get("model") or "")
        if model:
            grouped[key][model] = row

    candidates: list[dict[str, Any]] = []
    models = [model for model in MODEL_ORDER if any(model in values for values in grouped.values())]
    if not models:
        models = sorted({model for values in grouped.values() for model in values})
    for key, by_model in grouped.items():
        _, stratum, _, _ = query_metadata(str(next(iter(by_model.values())).get("query") or ""))
        if not stratum:
            continue
        index = int(hashlib.sha256(f"{seed}|model|{key[0]}|{key[1]}".encode("utf-8")).hexdigest()[:8], 16)
        chosen_model = models[index % len(models)]
        row = by_model.get(chosen_model) or by_model[sorted(by_model)[0]]
        candidates.append({"row": row, "stratum": stratum, "key": key, "rank": _stable_rank(seed, key)})

    candidates.sort(key=lambda item: (item["stratum"], item["rank"]))
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    for stratum, target in targets.items():
        for item in (candidate for candidate in candidates if candidate["stratum"] == stratum):
            if len([x for x in selected if x["stratum"] == stratum]) >= target:
                break
            selected.append(item)
            selected_keys.add(item["key"])
    total_target = sum(targets.values())
    if len(selected) < total_target:
        for item in sorted(candidates, key=lambda value: value["rank"]):
            if item["key"] in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(item["key"])
            if len(selected) >= total_target:
                break
    return [item["row"] | {"_natural_stratum": item["stratum"]} for item in selected[:total_target]]


def build_rows(rows: list[dict[str, Any]], *, source_audit_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queue: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    for row in rows:
        context = row.get("context") or {}
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError("natural row missing record_id")
        annotation_id = "agentic-natural-review:" + hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:20]
        applicable, stratum, family, reference = query_metadata(str(row.get("query") or ""))
        queue_row = {
            "schema_version": "vsight_agentic_drift_natural_review_input_v1",
            "annotation_id": annotation_id,
            "data_role": "development_agentic_natural_audit_not_training",
            "image_filename": context.get("image_filename"),
            "image_group_id": context.get("group_id"),
            "original_bbox_xyxy": row.get("original_bbox"),
            "query": row.get("query"),
            "query_stratum": stratum,
            "reference_phrase": reference,
            "relation_family": family,
            "stage_b_applicable_atoms": applicable,
        }
        leaked = FORBIDDEN_QUEUE_FIELDS & set(queue_row)
        if leaked:
            raise ValueError(f"natural review queue leaks private fields: {sorted(leaked)}")
        queue.append(queue_row)
        sidecar.append(
            {
                "schema_version": "vsight_agentic_drift_natural_hypothesis_sidecar_v1",
                "annotation_id": annotation_id,
                "source_record_id": record_id,
                "source_audit_sha256": source_audit_sha256,
                "selection_rule": "hash_stratified_unique_image_group_task_model_rotation",
                "selection_stratum": str(row.get("_natural_stratum") or stratum),
                "source_model": context.get("model"),
                "source_task": context.get("task"),
                "agent_evidence_state": row.get("evidence_state"),
                "agent_action": row.get("action"),
                "agent_reason_codes": list(row.get("reason_codes") or []),
                "agent_review_priority": row.get("review_priority"),
                "served_to_reviewer": False,
            }
        )
    if len({row["annotation_id"] for row in queue}) != len(queue):
        raise ValueError("natural queue annotation_id collision")
    return queue, sidecar


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default="agentic-natural-v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = arguments()
    if not options.audit.exists():
        raise FileNotFoundError(options.audit)
    output_dir = options.output_dir
    queue_path = output_dir / "agentic_drift_natural_review_queue.jsonl.gz"
    sidecar_path = output_dir / "agentic_drift_natural_hypotheses.private.jsonl.gz"
    summary_path = output_dir / "agentic_drift_natural_review_queue.summary.json"
    if not options.force and any(path.exists() for path in (queue_path, sidecar_path, summary_path)):
        raise FileExistsError("natural review artifacts exist; pass --force")
    rows = read_audit(options.audit)
    selected = select_natural_rows(rows, seed=options.seed)
    if len(selected) != sum(STRATUM_TARGETS.values()):
        raise ValueError(f"only selected {len(selected)} natural rows")
    queue, sidecar = build_rows(selected, source_audit_sha256=sha256(options.audit))
    output_dir.mkdir(parents=True, exist_ok=True)
    deterministic_gzip(queue_path, queue)
    deterministic_gzip(sidecar_path, sidecar)
    queue_hash = sha256(queue_path)
    missing = sorted({str(row["image_filename"]) for row in queue if not (options.images / str(row["image_filename"])).exists()})
    if missing:
        raise FileNotFoundError(f"missing review images: {missing[:5]}")
    summary = {
        "schema_version": "vsight_agentic_drift_natural_review_queue_summary_v1",
        "dataset_role": "development_agentic_natural_audit_not_training",
        "records": len(queue),
        "image_groups": len({row["image_group_id"] for row in queue}),
        "query_strata": dict(sorted(collections.Counter(row["query_stratum"] for row in queue).items())),
        "requested_query_strata": dict(sorted(STRATUM_TARGETS.items())),
        "source_audit": str(options.audit),
        "source_audit_sha256": sha256(options.audit),
        "queue": str(queue_path),
        "queue_sha256": queue_hash,
        "private_hypothesis_sidecar": str(sidecar_path),
        "model_identity_in_queue": False,
        "agent_hypotheses_in_queue": False,
        "gt_in_queue": False,
        "iou_in_queue": False,
        "selection_rule": "unique image_group x task; hash-stratified query type; deterministic model rotation; no agent outcome fields used",
        "minimum_reviewers": 1,
        "review_authority": "项目负责人",
        "single_review_authorized": True,
        "training_eligible": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
