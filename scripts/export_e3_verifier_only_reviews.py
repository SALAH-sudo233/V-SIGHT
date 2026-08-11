#!/usr/bin/env python3
"""Export conservatively adjudicated E3 verifier action training rows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = (
    ROOT
    / "data/e3/binding_annotation_queue/e3_binding_annotation_queue.train.jsonl.gz"
)
DEFAULT_REVIEWS = ROOT / "data/e3/binding_annotation_queue/e3_binding_reviews.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/e3/binding_annotation_queue"
ACTION_STAGE_A = {
    "ACCEPT": "supported",
    "REJECT": "contradicted",
    "RELOCALIZE": "supported",
}
ATOM_STATES = {"supported", "contradicted", "unobservable"}
ANNOTATION_PROTOCOL = "single_project_owner"
DEFAULT_AUTHORITATIVE_REVIEWER = "project_owner"


def box_iou(left, right):
    try:
        lx1, ly1, lx2, ly2 = (float(value) for value in left)
        rx1, ry1, rx2, ry2 = (float(value) for value in right)
    except (TypeError, ValueError):
        return 0.0
    values = (lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2)
    if not all(math.isfinite(value) for value in values):
        return 0.0
    if lx2 <= lx1 or ly2 <= ly1 or rx2 <= rx1 or ry2 <= ry1:
        return 0.0
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-reviewers", type=int, default=1)
    parser.add_argument(
        "--reviewer-id",
        default=DEFAULT_AUTHORITATIVE_REVIEWER,
        help="authoritative project-owner reviewer id",
    )
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--min-corrected-box-iou", type=float, default=0.50)
    parser.add_argument("--min-reference-box-iou", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queue(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row.get("annotation_id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("queue has missing or duplicate annotation_id")
    if any(row.get("data_split") != "train" for row in rows):
        raise ValueError("verifier training export must be train-only")
    return rows


def read_reviews(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_by_reviewer(rows: list[dict]) -> dict[tuple[str, str], dict]:
    latest = {}
    for row in rows:
        annotation_id = str(row.get("annotation_id") or "")
        reviewer_id = str(row.get("reviewer_id") or "")
        if annotation_id and reviewer_id:
            latest[(annotation_id, reviewer_id)] = row
    return latest


def adjudicate(
    queue_rows: list[dict],
    review_rows: list[dict],
    queue_hash: str,
    min_reviewers: int = 1,
    min_confidence: float = 0.90,
    min_corrected_box_iou: float = 0.50,
    min_reference_box_iou: float = 0.50,
    authoritative_reviewer_id: str = DEFAULT_AUTHORITATIVE_REVIEWER,
) -> tuple[list[dict], dict[str, int]]:
    if min_reviewers != 1:
        raise ValueError("single_project_owner requires min_reviewers=1")
    if not authoritative_reviewer_id.strip():
        raise ValueError("authoritative_reviewer_id cannot be empty")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if not 0 <= min_corrected_box_iou <= 1:
        raise ValueError("min_corrected_box_iou must be in [0, 1]")
    if not 0 <= min_reference_box_iou <= 1:
        raise ValueError("min_reference_box_iou must be in [0, 1]")

    latest = latest_by_reviewer(review_rows)
    exported = []
    audit: Counter[str] = Counter()
    for source in queue_rows:
        annotation_id = str(source["annotation_id"])
        if bool(source.get("sensitive_attribute")):
            audit["excluded_sensitive_attribute"] += 1
            continue
        reviews = [
            row
            for (row_id, reviewer_id), row in latest.items()
            if row_id == annotation_id
            and reviewer_id == authoritative_reviewer_id
            and row.get("status") == "completed"
            and row.get("source_queue_sha256") == queue_hash
            and float(row.get("confidence") or 0.0) >= min_confidence
        ]
        if len(reviews) < min_reviewers:
            audit["insufficient_completed_reviewers"] += 1
            continue
        decisions = {
            (
                str(row.get("stage_a_target_status") or ""),
                str(row.get("stage_c_action") or ""),
            )
            for row in reviews
        }
        if len(decisions) != 1:
            audit["reviewer_disagreement"] += 1
            continue
        stage_a, action = next(iter(decisions))
        is_v2 = str(source.get("schema_version") or "").endswith("_v2")
        if action == "UNCERTAIN":
            audit[f"reserved_{action.casefold()}"] += 1
            continue
        if ACTION_STAGE_A.get(action) != stage_a:
            audit["invalid_action_existence_pair"] += 1
            continue

        corrected_boxes = []
        if action == "RELOCALIZE":
            corrected_boxes = [row.get("corrected_bbox_xyxy") for row in reviews]
            if any(not isinstance(box, list) or len(box) != 4 for box in corrected_boxes):
                audit["invalid_relocalize_box"] += 1
                continue
            if len(corrected_boxes) > 1 and any(
                box_iou(corrected_boxes[left], corrected_boxes[right])
                < min_corrected_box_iou
                for left in range(len(corrected_boxes))
                for right in range(left + 1, len(corrected_boxes))
            ):
                audit["corrected_box_disagreement"] += 1
                continue

        reference_boxes = []
        reference_visibility = None
        if is_v2:
            applicable = [str(value) for value in source.get("stage_b_applicable_atoms") or []]
            atom_maps = [row.get("stage_b_atoms") for row in reviews]
            if any(not isinstance(value, dict) for value in atom_maps):
                audit["invalid_stage_b_schema"] += 1
                continue
            signatures = {
                tuple((atom, value.get(atom)) for atom in applicable)
                for value in atom_maps
            }
            if len(signatures) != 1:
                audit["stage_b_reviewer_disagreement"] += 1
                continue
            consensus_atoms = dict(next(iter(signatures)))
            if any(value not in ATOM_STATES for value in consensus_atoms.values()):
                audit["incomplete_stage_b_atoms"] += 1
                continue
            states = list(consensus_atoms.values())
            if (
                action == "ACCEPT" and any(value != "supported" for value in states)
            ) or (
                action in {"REJECT", "RELOCALIZE"} and "contradicted" not in states
            ):
                audit["invalid_action_atom_pair"] += 1
                continue
            if (
                "relation" in applicable
                and consensus_atoms.get("relation") in {"supported", "contradicted"}
            ):
                reference_boxes = [row.get("reference_bbox_xyxy") for row in reviews]
                if any(not isinstance(box, list) or len(box) != 4 for box in reference_boxes):
                    audit["invalid_reference_box"] += 1
                    continue
                if len(reference_boxes) > 1 and any(
                    box_iou(reference_boxes[left], reference_boxes[right])
                    < min_reference_box_iou
                    for left in range(len(reference_boxes))
                    for right in range(left + 1, len(reference_boxes))
                ):
                    audit["reference_box_disagreement"] += 1
                    continue
                visibilities = {str(row.get("reference_visibility") or "") for row in reviews}
                if len(visibilities) != 1 or not visibilities <= {"visible", "partially_visible"}:
                    audit["reference_visibility_disagreement"] += 1
                    continue
                reference_visibility = next(iter(visibilities))
        else:
            atom_sets = [set(row.get("stage_b_atoms") or []) for row in reviews]
            consensus_atoms = sorted(set.intersection(*atom_sets)) if atom_sets else []
        confidences = [float(row["confidence"]) for row in reviews]
        reviewer_ids = sorted(str(row["reviewer_id"]) for row in reviews)
        exported.append(
            {
                "schema_version": (
                    "vsight_e3_verifier_only_training_v2"
                    if is_v2 else "vsight_e3_verifier_only_training_v1"
                ),
                "annotation_id": annotation_id,
                "data_split": "train",
                "group_id": str(source["group_id"]),
                "image_id": int(source["image_id"]),
                "image_filename": str(source["image_filename"]),
                "query": str(source["query"]),
                "original_bbox_xyxy": [
                    float(value) for value in source["original_bbox_xyxy"]
                ],
                "query_stratum": str(source["query_stratum"]),
                "relation_family": source.get("relation_family"),
                "stage_a_target_status": stage_a,
                "stage_b_applicable_atoms": (
                    list(source.get("stage_b_applicable_atoms") or []) if is_v2 else []
                ),
                "stage_b_consensus_atoms": consensus_atoms,
                "stage_b_authoritative_atoms": consensus_atoms,
                "verifier_action": action,
                "reviewer_ids": reviewer_ids,
                "reviewer_count": len(reviewer_ids),
                "authoritative_reviewer_id": authoritative_reviewer_id,
                "annotation_protocol": ANNOTATION_PROTOCOL,
                "inter_reviewer_agreement_computed": False,
                "minimum_confidence": min(confidences),
                "mean_confidence": sum(confidences) / len(confidences),
                "source_queue_sha256": queue_hash,
                "router_only_corrected_boxes_xyxy": corrected_boxes,
                "independent_reference_boxes_xyxy": reference_boxes,
                "reference_visibility": reference_visibility,
                "sensitive_attribute": bool(source.get("sensitive_attribute")),
                "reviewer_disagreement": False,
                "schema_gate_passed": is_v2,
            }
        )
        audit[f"exported_{action.casefold()}"] += 1
    audit["exported_total"] = len(exported)
    return exported, dict(sorted(audit.items()))


@contextmanager
def deterministic_gzip(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def main() -> int:
    args = parse_args()
    queue_path = resolve_path(args.queue)
    reviews_path = resolve_path(args.reviews)
    output_dir = resolve_path(args.output_dir)
    output_path = output_dir / "e3_verifier_only_adjudicated.train.jsonl.gz"
    summary_path = output_dir / "e3_verifier_only_adjudicated.summary.json"
    if not args.force and (output_path.exists() or summary_path.exists()):
        raise FileExistsError("verifier-only export exists; pass --force")

    queue_hash = sha256(queue_path)
    queue_rows = read_queue(queue_path)
    review_rows = read_reviews(reviews_path)
    exported, audit = adjudicate(
        queue_rows,
        review_rows,
        queue_hash,
        min_reviewers=args.min_reviewers,
        min_confidence=args.min_confidence,
        min_corrected_box_iou=args.min_corrected_box_iou,
        min_reference_box_iou=args.min_reference_box_iou,
        authoritative_reviewer_id=args.reviewer_id,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with deterministic_gzip(temporary) as handle:
            for row in exported:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema_version": "vsight_e3_verifier_only_export_v2",
        "status": "ready" if exported else "awaiting_reviews",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queue": {
            "path": str(queue_path),
            "sha256": queue_hash,
            "rows": len(queue_rows),
        },
        "reviews": {
            "path": str(reviews_path),
            "sha256": sha256(reviews_path) if reviews_path.exists() else None,
            "records": len(review_rows),
        },
        "policy": {
            "annotation_protocol": ANNOTATION_PROTOCOL,
            "authoritative_reviewer_id": args.reviewer_id,
            "min_reviewers": args.min_reviewers,
            "min_confidence": args.min_confidence,
            "inter_reviewer_agreement_computed": False,
            "pairwise_corrected_box_iou_applied": False,
            "pairwise_reference_box_iou_applied": False,
            "actions": ["ACCEPT", "REJECT", "RELOCALIZE"],
            "relocalize_preserved": True,
            "uncertain_excluded": True,
            "stage_b_atom_states": sorted(ATOM_STATES),
            "single_corrected_box_required_for_relocalize": True,
            "single_reference_box_required_for_observable_relation": True,
            "unobservable_excluded_from_contradiction_training": True,
        },
        "audit": audit,
        "output": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "rows": len(exported),
        },
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
