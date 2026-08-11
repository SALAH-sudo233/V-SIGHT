#!/usr/bin/env python3
"""Join reviewed CCV actions with composite evidence into isolated train/cal rows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv import ClaimConditionedCounterfactualVerifier  # noqa: E402
from vsight.ccv_learning import (  # noqa: E402
    action_features,
    eligible_cable_review_record,
    eligible_review_record,
)
from vsight.composite_detector import evidence_from_dict  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for(group_id: str, calibration_fraction: float) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"vsight-ccv-binding-v1:{group_id}".encode()).digest()[:8],
        "big",
    ) / 2**64
    return "calibration" if value < calibration_fraction else "train"


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def build_rows(review_rows: list[dict], evidence_rows: list[dict], fraction: float) -> dict[str, list[dict]]:
    if not 0 < fraction < 1:
        raise ValueError("calibration fraction must be in (0, 1)")
    evidence = {str(row["record_id"]): row for row in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("duplicate composite evidence record_id")
    verifier = ClaimConditionedCounterfactualVerifier()
    output = {"train": [], "calibration": []}
    seen_groups = {}
    for review in review_rows:
        is_v2 = str(review.get("schema_version") or "").endswith("_v2")
        if not (
            eligible_cable_review_record(review)
            if is_v2 else eligible_review_record(review)
        ):
            continue
        annotation_id = str(review["annotation_id"])
        if annotation_id not in evidence:
            raise ValueError(f"missing composite evidence: {annotation_id}")
        group_id = str(review["group_id"])
        split = split_for(group_id, fraction)
        if group_id in seen_groups and seen_groups[group_id] != split:
            raise ValueError(f"image group crosses splits: {group_id}")
        seen_groups[group_id] = split
        detector = evidence_from_dict(evidence[annotation_id]["detector_evidence"])
        result = verifier.verify(
            None,
            str(review["query"]),
            review["original_bbox_xyxy"],
            detector,
        )
        output[split].append(
            {
                "schema_version": "vsight_cable_reviewed_training_v2" if is_v2 else "vsight_ccv_reviewed_training_v1",
                "annotation_id": annotation_id,
                "data_split": split,
                "group_id": group_id,
                "image_id": review["image_id"],
                "image_filename": review["image_filename"],
                "query": review["query"],
                "original_bbox_xyxy": review["original_bbox_xyxy"],
                "atom_evidence": result.atom_evidence,
                "ccv_features": action_features(result),
                "verifier_action": review["verifier_action"],
                "query_stratum": review.get("query_stratum"),
                "stage_a_target_status": review.get("stage_a_target_status"),
                "stage_b_applicable_atoms": review.get("stage_b_applicable_atoms", []),
                "stage_b_consensus_atoms": review.get("stage_b_consensus_atoms", {}),
                "stage_b_authoritative_atoms": review.get(
                    "stage_b_authoritative_atoms",
                    review.get("stage_b_consensus_atoms", {}),
                ),
                "independent_reference_boxes_xyxy": review.get(
                    "independent_reference_boxes_xyxy", []
                ),
                "reference_visibility": review.get("reference_visibility"),
                "schema_gate_passed": bool(review.get("schema_gate_passed")),
                "independent_corrected_boxes_xyxy": review.get(
                    "router_only_corrected_boxes_xyxy", []
                ),
                "router_only_corrected_boxes_xyxy": review.get(
                    "router_only_corrected_boxes_xyxy", []
                ),
                "reviewer_ids": review["reviewer_ids"],
                "reviewer_count": review["reviewer_count"],
                "authoritative_reviewer_id": review.get(
                    "authoritative_reviewer_id"
                ),
                "annotation_protocol": review.get("annotation_protocol"),
                "minimum_confidence": review["minimum_confidence"],
                "mean_confidence": review["mean_confidence"],
                "inter_reviewer_agreement_computed": False,
                "reviewer_disagreement": False,
                "sensitive_attribute": False,
                "source_queue_sha256": review["source_queue_sha256"],
                "composite_evidence_record_id": annotation_id,
            }
        )
    for rows in output.values():
        rows.sort(key=lambda row: row["annotation_id"])
    return output


def main() -> int:
    args = arguments()
    outputs = {
        split: args.output_dir / f"ccv_reviewed.{split}.jsonl.gz"
        for split in ("train", "calibration")
    }
    summary_path = args.output_dir / "ccv_reviewed.summary.json"
    if not args.force and (summary_path.exists() or any(path.exists() for path in outputs.values())):
        raise FileExistsError("training manifest exists; pass --force")
    rows = build_rows(
        read_rows(args.reviews), read_rows(args.evidence), args.calibration_fraction
    )
    if not rows["train"] or not rows["calibration"]:
        raise ValueError("reviewed rows must populate both train and calibration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in outputs.items():
        with deterministic_gzip(path) as handle:
            for row in rows[split]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "vsight_cable_reviewed_manifest_v2",
        "reviews": {"path": str(args.reviews), "sha256": sha256(args.reviews)},
        "composite_evidence": {"path": str(args.evidence), "sha256": sha256(args.evidence)},
        "calibration_fraction": args.calibration_fraction,
        "outputs": {
            split: {"path": str(path), "sha256": sha256(path), "rows": len(rows[split])}
            for split, path in outputs.items()
        },
        "image_group_overlap": 0,
        "annotation_protocol": "single_project_owner",
        "minimum_reviewers": 1,
        "inter_reviewer_agreement_reported": False,
        "sealed_heldout_accessed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
