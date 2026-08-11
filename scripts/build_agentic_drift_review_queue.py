#!/usr/bin/env python3
"""Build a blinded human-review queue from agentic drift audit output."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.cable import relation_family  # noqa: E402
from vsight.ccv import AtomType, TypedClaimParser  # noqa: E402


DEFAULT_AUDIT = ROOT / "outputs/agentic_drift_audit_v2_nomemory/audit.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "data/e3/agentic_drift_review"
DEFAULT_IMAGES = Path("/home/u2025141034/benchmark/benchmark_images")
QUEUE_FORBIDDEN_FIELDS = frozenset({
    "model", "task", "sample_id", "source_record_id", "evidence_state",
    "action", "drift_risk_raw", "review_priority", "reason_codes",
    "alternative_bbox", "claim_support", "alternative_support",
    "binding_margin", "memory_status", "trace", "gt_bbox_xyxy", "iou",
})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--ambiguous-high-priority", type=int, default=20)
    parser.add_argument("--accept-controls", type=int, default=3)
    parser.add_argument("--low-risk-controls", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def read_audit(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ranked(rows: list[dict], *, reverse: bool = True) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("review_priority") or 0.0) if reverse else float(row.get("review_priority") or 0.0),
            str(row.get("record_id") or ""),
        ),
    )


def select_rows(
    rows: list[dict],
    *,
    ambiguous_high_priority: int,
    accept_controls: int,
    low_risk_controls: int,
) -> list[tuple[str, dict]]:
    for value in (ambiguous_high_priority, accept_controls, low_risk_controls):
        if value < 0:
            raise ValueError("selection counts cannot be negative")
    eligible = [row for row in rows if row.get("original_bbox") is not None]
    selected: list[tuple[str, dict]] = []
    seen = set()

    def add(band: str, candidates: list[dict], cap: int | None = None) -> None:
        added = 0
        for row in candidates:
            key = str(row.get("record_id") or "")
            if not key or key in seen:
                continue
            selected.append((band, row))
            seen.add(key)
            added += 1
            if cap is not None and added >= cap:
                break

    add(
        "provisional_wrong_instance",
        _ranked([row for row in eligible if row.get("evidence_state") == "WRONG_INSTANCE"]),
    )
    add(
        "ambiguous_high_priority",
        _ranked([row for row in eligible if row.get("evidence_state") == "UNOBSERVABLE_AMBIGUOUS"]),
        ambiguous_high_priority,
    )
    add(
        "accept_control",
        _ranked([row for row in eligible if row.get("action") == "ACCEPT"]),
        accept_controls,
    )
    low_risk = sorted(
        (
            row for row in eligible
            if row.get("evidence_state") == "UNOBSERVABLE_AMBIGUOUS"
            and str(row.get("record_id") or "") not in seen
        ),
        key=lambda row: (
            float(row.get("review_priority") or 0.0),
            str(row.get("record_id") or ""),
        ),
    )
    add("low_risk_control", low_risk, low_risk_controls)
    return selected


def _claim_metadata(query: str) -> tuple[list[str], str, str | None, str | None]:
    claim = TypedClaimParser().parse(query)
    atom_types = {atom.atom_type for atom in claim.known_atoms}
    applicable = ["identity"]
    if AtomType.ATTRIBUTE in atom_types:
        applicable.append("attribute")
    if AtomType.ACTION in atom_types:
        applicable.append("action")
    relation_atom = next(
        (atom for atom in claim.known_atoms if atom.atom_type is AtomType.RELATION),
        None,
    )
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


def build_rows(selected: list[tuple[str, dict]]) -> tuple[list[dict], list[dict]]:
    queue = []
    hypotheses = []
    for band, row in selected:
        context = dict(row.get("context") or {})
        record_id = str(row["record_id"])
        annotation_id = "agentic-review:" + hashlib.sha256(record_id.encode()).hexdigest()[:20]
        applicable, stratum, family, reference = _claim_metadata(str(row["query"]))
        queue_row = {
            "schema_version": "vsight_agentic_drift_review_input_v1",
            "annotation_id": annotation_id,
            "data_role": "development_agentic_audit_not_training",
            "image_group_id": str(context.get("group_id") or annotation_id),
            "image_filename": str(context["image_filename"]),
            "query": str(row["query"]),
            "query_stratum": stratum,
            "relation_family": family,
            "reference_phrase": reference,
            "original_bbox_xyxy": [float(value) for value in row["original_bbox"]],
            "stage_b_applicable_atoms": applicable,
        }
        leaked = QUEUE_FORBIDDEN_FIELDS & set(queue_row)
        if leaked:
            raise ValueError(f"review queue leaks agent/supervision fields: {sorted(leaked)}")
        queue.append(queue_row)
        hypotheses.append({
            "schema_version": "vsight_agentic_drift_hypothesis_sidecar_v1",
            "annotation_id": annotation_id,
            "source_record_id": record_id,
            "selection_band": band,
            "agent_evidence_state": row.get("evidence_state"),
            "agent_action": row.get("action"),
            "drift_risk_raw": row.get("drift_risk_raw"),
            "review_priority": row.get("review_priority"),
            "reason_codes": list(row.get("reason_codes") or []),
            "alternative_bbox": row.get("alternative_bbox"),
            "claim_support": row.get("claim_support"),
            "alternative_support": row.get("alternative_support"),
            "binding_margin": row.get("binding_margin"),
            "memory_status": row.get("memory_status"),
        })
    queue.sort(key=lambda row: row["annotation_id"])
    order = {row["annotation_id"]: index for index, row in enumerate(queue)}
    hypotheses.sort(key=lambda row: order[row["annotation_id"]])
    if len({row["annotation_id"] for row in queue}) != len(queue):
        raise ValueError("duplicate blinded annotation ID")
    return queue, hypotheses


def write_rows(path: Path, rows: list[dict]) -> None:
    with deterministic_gzip(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    args = arguments()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    queue_path = output_dir / "agentic_drift_review_queue.jsonl.gz"
    sidecar_path = output_dir / "agentic_drift_hypotheses.private.jsonl.gz"
    summary_path = output_dir / "agentic_drift_review_queue.summary.json"
    if not args.force and any(path.exists() for path in (queue_path, sidecar_path, summary_path)):
        raise FileExistsError("agentic review artifacts exist; pass --force")
    audit = read_audit(args.audit)
    selected = select_rows(
        audit,
        ambiguous_high_priority=args.ambiguous_high_priority,
        accept_controls=args.accept_controls,
        low_risk_controls=args.low_risk_controls,
    )
    queue, hypotheses = build_rows(selected)
    missing = [
        row["image_filename"] for row in queue
        if not (args.images / Path(row["image_filename"]).name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"review image root is missing {len(missing)} images")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(queue_path, queue)
    queue_hash = sha256(queue_path)
    for row in hypotheses:
        row["source_queue_sha256"] = queue_hash
        row["served_to_reviewer"] = False
    write_rows(sidecar_path, hypotheses)
    counts = Counter(row["selection_band"] for row in hypotheses)
    strata = Counter(row["query_stratum"] for row in queue)
    summary = {
        "schema_version": "vsight_agentic_drift_review_queue_summary_v1",
        "dataset_role": "development_agentic_audit_not_training",
        "records": len(queue),
        "image_groups": len({row["image_group_id"] for row in queue}),
        "selection_bands": dict(sorted(counts.items())),
        "query_strata": dict(sorted(strata.items())),
        "audit": str(args.audit),
        "audit_sha256": sha256(args.audit),
        "queue": str(queue_path),
        "queue_sha256": queue_hash,
        "private_hypothesis_sidecar": str(sidecar_path),
        "private_hypothesis_sidecar_sha256": sha256(sidecar_path),
        "agent_hypotheses_in_queue": False,
        "gt_in_queue": False,
        "iou_in_queue": False,
        "model_identity_in_queue": False,
        "served_sidecar_to_reviewer": False,
        "minimum_reviewers": 1,
        "review_authority": "项目负责人",
        "single_review_authorized": True,
        "training_eligible": False,
        "image_root": str(args.images),
        "missing_images": 0,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
