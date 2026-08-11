#!/usr/bin/env python3
"""Evaluate paired legacy-label and token-span CABLE evidence on a pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import statistics
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.ccv import (  # noqa: E402
    CCVAction,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
)
from vsight.ccv_metrics import (  # noqa: E402
    audit_safety_gates,
    evaluate_grouped_safety,
    evaluate_safety,
)
from vsight.composite_detector import evidence_from_dict  # noqa: E402


DEFAULT_FEATURES = (
    ROOT / "data/e3/singlepass/crossmodel/e3_singlepass_crossmodel.development.jsonl.gz"
)
DEFAULT_BASELINE = (
    ROOT / "outputs/cable_trainfree_v2_t010/cable_trainfree.predictions.jsonl.gz"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--baseline-predictions", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--token-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_map(path: Path) -> dict[str, dict]:
    output = {}
    for row in read_jsonl(path):
        record_id = str(row["record_id"])
        if record_id in output:
            raise ValueError(f"duplicate evidence record: {record_id}")
        output[record_id] = row
    return output


@contextmanager
def deterministic_gzip(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def compact_source(feature: dict, baseline: dict) -> dict:
    claim = (baseline.get("atom_evidence") or {}).get("claim") or {}
    return {
        "model": str(feature["model"]),
        "task": str(feature["task"]),
        "sample_id": str(feature["sample_id"]),
        "group_id": str(feature["group_id"]),
        "semantic_query_id": str(feature["semantic_query_id"]),
        "query": str(feature["query"]),
        "label_exists": bool(feature["label_exists"]),
        "hallucination_type": str(feature["hallucination_type"]),
        "original_iou": float(feature.get("original_iou") or 0.0),
        "gt_bbox_xyxy": feature.get("gt_bbox_xyxy"),
        "original_accept": bool(baseline.get("original_accept")),
        "verifier_eligible": bool(baseline.get("verifier_eligible")),
        "upstream_bbox_xyxy": claim.get("bbox_xyxy"),
    }


def selected_sources(features: Path, baseline: Path, semantic_ids: set[str]) -> tuple[list[dict], list[dict]]:
    feature_rows = read_jsonl(features)
    baseline_rows = read_jsonl(baseline)
    if len(feature_rows) != len(baseline_rows):
        raise ValueError("feature and baseline manifests have different lengths")
    selected = []
    frozen = []
    for feature, prediction in zip(feature_rows, baseline_rows, strict=True):
        identity = (str(feature["model"]), str(feature["task"]), str(feature["sample_id"]))
        prediction_identity = (
            str(prediction["model"]), str(prediction["task"]), str(prediction["sample_id"])
        )
        if identity != prediction_identity:
            raise ValueError(f"feature/baseline identity mismatch: {identity}")
        if str(feature["semantic_query_id"]) not in semantic_ids:
            continue
        selected.append(compact_source(feature, prediction))
        frozen.append(prediction)
    return selected, frozen


def run_variant(sources: list[dict], evidence: dict[str, dict]) -> list[dict]:
    verifier = ClaimConditionedCounterfactualVerifier(
        CCVThresholds(require_typed_relocalization=True)
    )
    predictions = []
    for row in sources:
        detector = evidence_from_dict(
            evidence[str(row["semantic_query_id"])]["detector_evidence"]
        )
        if row["verifier_eligible"]:
            upstream = row.get("upstream_bbox_xyxy")
            if not isinstance(upstream, list) or len(upstream) != 4:
                raise ValueError("eligible alignment row has no upstream bbox")
            result = verifier.verify(None, row["query"], upstream, detector)
            action = result.action
            reason = result.reason
            binding_status = result.binding_status
            ledger = (result.atom_evidence.get("ledger") or {})
        else:
            action = CCVAction.ACCEPT if row["original_accept"] else CCVAction.REJECT
            reason = "upstream_passthrough"
            binding_status = None
            ledger = {}
        predictions.append(
            {
                **row,
                "action": action.value,
                "selected_iou": (
                    0.0 if action is CCVAction.REJECT else float(row["original_iou"])
                ),
                "reason": reason,
                "binding_status": binding_status,
                "explicit_witness": bool(ledger.get("explicit_witness")),
                "witness_kind": ledger.get("witness_kind"),
                "M_contra": ledger.get("M_contra"),
                "M_restore": ledger.get("M_restore"),
                "U_edge": ledger.get("U_edge"),
            }
        )
    return predictions


def evidence_statistics(rows: dict[str, dict]) -> dict[str, object]:
    proposal_counts = []
    provenance = Counter()
    segment = Counter()
    object_rows = reference_rows = full_rows = inverse_rows = long_labels = 0
    for row in rows.values():
        proposals = row["detector_evidence"].get("proposals") or []
        proposal_counts.append(len(proposals))
        object_rows += any(p.get("segment_id") == "object" for p in proposals)
        reference_rows += any(str(p.get("segment_id") or "").startswith("reference:") for p in proposals)
        full_rows += any(p.get("segment_id") == "full" for p in proposals)
        inverse_rows += any(p.get("prompt_provenance") == "inverse" for p in proposals)
        long_labels += sum(len(str(p.get("label") or "").split()) >= 5 for p in proposals)
        for proposal in proposals:
            provenance[str(proposal.get("prompt_provenance") or "claim")] += 1
            segment[str(proposal.get("segment_id") or "").split(":", 1)[0]] += 1
    total = len(rows)
    return {
        "records": total,
        "mean_proposals": statistics.mean(proposal_counts),
        "object_coverage": object_rows / total,
        "reference_coverage": reference_rows / total,
        "full_coverage": full_rows / total,
        "inverse_proposal_coverage": inverse_rows / total,
        "long_label_proposals": long_labels,
        "provenance_counts": dict(provenance),
        "segment_counts": dict(segment),
    }


def witness_statistics(rows: list[dict]) -> dict[str, object]:
    eligible = [row for row in rows if row["verifier_eligible"]]
    witnessed = [row for row in eligible if bool(row.get("explicit_witness"))]
    typed_rejects = [
        row
        for row in rows
        if row["reason"] == "typed_binding_counterfactual_contradiction"
    ]
    return {
        "eligible": len(eligible),
        "explicit_witnesses": len(witnessed),
        "explicit_witness_rate": len(witnessed) / len(eligible) if eligible else None,
        "witness_kinds": dict(
            Counter(
                str(
                    row.get("witness_kind")
                    or (((row.get("atom_evidence") or {}).get("ledger") or {}).get("witness_kind"))
                )
                for row in witnessed
            )
        ),
        "typed_rejects": len(typed_rejects),
        "typed_reject_precision": (
            sum(not row["label_exists"] for row in typed_rejects) / len(typed_rejects)
            if typed_rejects else None
        ),
        "typed_reject_false_positives": sum(row["label_exists"] for row in typed_rejects),
    }


def variant_summary(rows: list[dict]) -> dict[str, object]:
    grouped = evaluate_grouped_safety(rows)
    return {
        "overall": evaluate_safety(rows),
        "heldout_model_task": grouped,
        "safety_gate": audit_safety_gates(grouped),
        "witness": witness_statistics(rows),
        "reason_counts": dict(Counter(str(row["reason"]) for row in rows)),
        "binding_status_counts": dict(Counter(str(row["binding_status"]) for row in rows)),
    }


def paired_changes(left: list[dict], right: list[dict]) -> dict[str, int]:
    output = Counter()
    for prior, current in zip(left, right, strict=True):
        key = (prior["model"], prior["task"], prior["sample_id"])
        if key != (current["model"], current["task"], current["sample_id"]):
            raise ValueError("paired alignment outputs are not ordered identically")
        if prior["action"] != current["action"]:
            stratum = "positive" if current["label_exists"] else current["hallucination_type"]
            output[f"{prior['action']}->{current['action']}|{stratum}"] += 1
    return dict(output)


def main() -> int:
    args = arguments()
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "paired_predictions.jsonl.gz"
    if not args.force and (summary_path.exists() or predictions_path.exists()):
        raise FileExistsError("alignment pilot output exists; pass --force")
    legacy_evidence = evidence_map(args.legacy_evidence)
    token_evidence = evidence_map(args.token_evidence)
    if set(legacy_evidence) != set(token_evidence):
        raise ValueError("paired evidence record ids differ")
    sources, frozen = selected_sources(
        args.features, args.baseline_predictions, set(legacy_evidence)
    )
    legacy = run_variant(sources, legacy_evidence)
    token = run_variant(sources, token_evidence)
    summary = {
        "schema_version": "vsight_cable_alignment_pilot_v2",
        "semantic_queries": len(legacy_evidence),
        "crossmodel_records": len(sources),
        "evidence": {
            "legacy_label": evidence_statistics(legacy_evidence),
            "token_span": evidence_statistics(token_evidence),
        },
        "variants": {
            "frozen_v2_before_role_fix": variant_summary(frozen),
            "role_fix_legacy_label": variant_summary(legacy),
            "role_fix_token_span": variant_summary(token),
        },
        "action_changes": {
            "legacy_to_token_span": paired_changes(legacy, token),
            "frozen_v2_to_token_span": paired_changes(frozen, token),
        },
        "dataset_role": "development_pilot_not_final_test",
        "detector_image_encoder_forwards_per_query": 1,
        "sealed_heldout_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with deterministic_gzip(predictions_path) as handle:
        for source, legacy_row, token_row in zip(sources, legacy, token, strict=True):
            handle.write(
                json.dumps(
                    {"source": source, "legacy_label": legacy_row, "token_span": token_row},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    summary["paired_predictions_sha256"] = hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "semantic_queries": summary["semantic_queries"],
                "crossmodel_records": summary["crossmodel_records"],
                "evidence": summary["evidence"],
                "overall": {
                    name: value["overall"]
                    for name, value in summary["variants"].items()
                },
                "action_changes": summary["action_changes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
