#!/usr/bin/env python3
"""Fit the transparent three-branch CABLE head on reviewed image groups."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv_learning import (  # noqa: E402
    DEFAULT_ABSENCE_DIRECTIONS,
    DEFAULT_CONTRADICTION_DIRECTIONS,
    DEFAULT_RESTORATION_DIRECTIONS,
    GapBalancedActionHead,
    eligible_cable_review_record,
)


FORBIDDEN_FEATURES = {
    "model", "model_name", "upstream_model", "gt", "gt_bbox", "gt_bbox_xyxy",
    "iou", "original_iou", "hallucination_type", "benchmark_label", "query_role",
    "verifier_action", "reviewer_id", "reviewer_ids",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def validated(rows: list[dict], role: str):
    features, actions, groups = [], [], set()
    absence_labels, contradiction_labels = [], []
    contradiction_observable = []
    for row in rows:
        if not eligible_cable_review_record(row):
            continue
        group = str(row.get("group_id") or "")
        if not group:
            raise ValueError(f"{role} row has no image group")
        raw = row.get("ccv_features")
        if not isinstance(raw, dict):
            raise ValueError(f"{role} row has no ccv_features mapping")
        leaked = FORBIDDEN_FEATURES & set(raw)
        if leaked:
            raise ValueError(f"{role} inference features leak supervision: {sorted(leaked)}")
        features.append({str(key): float(value) for key, value in raw.items()})
        actions.append(str(row["verifier_action"]))
        atoms = (
            row.get("stage_b_authoritative_atoms")
            or row.get("stage_b_consensus_atoms")
            or {}
        )
        stratum = str(row.get("query_stratum") or "object")
        absence_labels.append(float(
            row["verifier_action"] == "REJECT"
            and row.get("stage_a_target_status") == "contradicted"
            and stratum == "object"
        ))
        contradiction_labels.append(float(
            row["verifier_action"] == "REJECT"
            and stratum in {"attribute", "relation"}
            and any(value == "contradicted" for value in atoms.values())
        ))
        relevant_atom = "relation" if stratum == "relation" else "attribute"
        contradiction_observable.append(
            stratum in {"attribute", "relation"}
            and atoms.get(relevant_atom) in {"supported", "contradicted"}
        )
        groups.add(group)
    if not features:
        raise ValueError(
            f"{role} contains no eligible single-project-owner reviewed rows"
        )
    return (
        features, actions, groups, absence_labels, contradiction_labels,
        contradiction_observable,
    )


def main() -> int:
    args = arguments()
    if args.output.exists() and not args.force:
        raise FileExistsError("output exists; pass --force")
    train_features, train_actions, train_groups, train_absence, train_contradiction, train_observable = validated(read_rows(args.train), "train")
    cal_features, cal_actions, cal_groups, cal_absence, cal_contradiction, cal_observable = validated(
        read_rows(args.calibration), "calibration"
    )
    overlap = train_groups & cal_groups
    if overlap:
        raise ValueError(f"train/calibration image-group overlap: {sorted(overlap)[:5]}")
    required = (
        set(DEFAULT_ABSENCE_DIRECTIONS)
        | set(DEFAULT_CONTRADICTION_DIRECTIONS)
        | set(DEFAULT_RESTORATION_DIRECTIONS)
    )
    for role, rows in (("train", train_features), ("calibration", cal_features)):
        missing = [index for index, row in enumerate(rows) if not required <= set(row)]
        if missing:
            raise ValueError(f"{role} rows missing default features: {missing[:5]}")
    head = GapBalancedActionHead.fit(
        train_features,
        train_actions,
        absence_labels=train_absence,
        contradiction_labels=train_contradiction,
        contradiction_observable=train_observable,
    ).calibrate(
        cal_features,
        cal_actions,
        absence_labels=cal_absence,
        contradiction_labels=cal_contradiction,
        contradiction_observable=cal_observable,
    )
    artifact = head.to_dict()
    artifact["training_protocol"] = {
        "train_path": str(args.train),
        "train_sha256": sha256(args.train),
        "train_rows": len(train_features),
        "train_image_groups": len(train_groups),
        "calibration_path": str(args.calibration),
        "calibration_sha256": sha256(args.calibration),
        "calibration_rows": len(cal_features),
        "calibration_image_groups": len(cal_groups),
        "image_group_overlap": 0,
        "minimum_reviewers": 1,
        "annotation_protocol": "single_project_owner",
        "inter_reviewer_agreement_reported": False,
        "minimum_confidence": 0.90,
        "uncertain_excluded": True,
        "sensitive_attributes_excluded": True,
        "visual_backbone_frozen": True,
        "stage_b_schema_gate": "tri_state_v2",
        "branches": ["absence", "contradiction", "restoration"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact["training_protocol"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
