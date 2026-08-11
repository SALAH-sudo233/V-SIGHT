#!/usr/bin/env python3
"""Audit CABLE-RAFT attention ledgers without changing action policy."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    args = arguments()
    if args.output.exists() and not args.force:
        raise FileExistsError("audit output exists; pass --force")
    rows = read_jsonl(args.evidence)
    statuses = Counter()
    role_counts = Counter()
    role_sources = Counter()
    latencies = []
    forwards = Counter()
    edge_count = 0
    role_confidences = []
    transports = []
    uncertainties = []
    decoder_transports = []
    decoder_uncertainties = []
    decoder_edges = 0
    causal_transports = []
    causal_uncertainties = []
    causal_edges = 0
    causal_relative_drops = []
    influence_transports = []
    influence_uncertainties = []
    influence_edges = 0
    hidden_drifts = []
    influence_swap_margins = []
    intervention_statuses = Counter()
    intervention_latencies = []
    metadata = Counter()
    for row in rows:
        detector = row.get("detector_evidence") or row
        ledger = detector.get("attention_ledger") or {}
        statuses[str(ledger.get("status") or "missing")] += 1
        roles = ledger.get("role_spans") or []
        role_counts[len(roles)] += 1
        for role in roles:
            role_sources[str(role.get("source") or "unknown")] += 1
            if role.get("confidence") is not None:
                role_confidences.append(float(role["confidence"]))
        if ledger.get("best_edge") is not None:
            edge_count += 1
        if ledger.get("top_transport") is not None:
            transports.append(float(ledger["top_transport"]))
        if ledger.get("edge_uncertainty") is not None:
            uncertainties.append(float(ledger["edge_uncertainty"]))
        if ledger.get("best_decoder_edge") is not None:
            decoder_edges += 1
        if ledger.get("decoder_top_transport") is not None:
            decoder_transports.append(float(ledger["decoder_top_transport"]))
        if ledger.get("decoder_edge_uncertainty") is not None:
            decoder_uncertainties.append(float(ledger["decoder_edge_uncertainty"]))
        if ledger.get("best_causal_edge") is not None:
            causal_edges += 1
        if ledger.get("causal_top_transport") is not None:
            causal_transports.append(float(ledger["causal_top_transport"]))
        if ledger.get("causal_edge_uncertainty") is not None:
            causal_uncertainties.append(float(ledger["causal_edge_uncertainty"]))
        for effect in ledger.get("causal_role_effects") or []:
            if effect.get("relative_drop") is not None:
                causal_relative_drops.append(float(effect["relative_drop"]))
            if effect.get("hidden_cosine_drift") is not None:
                hidden_drifts.append(float(effect["hidden_cosine_drift"]))
        if ledger.get("best_causal_influence_edge") is not None:
            influence_edges += 1
        if ledger.get("causal_influence_top_transport") is not None:
            influence_transports.append(
                float(ledger["causal_influence_top_transport"])
            )
        if ledger.get("causal_influence_edge_uncertainty") is not None:
            influence_uncertainties.append(
                float(ledger["causal_influence_edge_uncertainty"])
            )
        for edge in ledger.get("causal_influence_edges") or []:
            if edge.get("role_swap_margin") is not None:
                influence_swap_margins.append(float(edge["role_swap_margin"]))
        if detector.get("latency_ms") is not None:
            latencies.append(float(detector["latency_ms"]))
        forwards[str(detector.get("image_encoder_forwards"))] += 1
        for key, value in (ledger.get("attention_metadata") or {}).items():
            if value is not None:
                metadata[key] += 1
        attention_metadata = ledger.get("attention_metadata") or {}
        intervention_statuses[
            str(attention_metadata.get("decoder_intervention_status") or "disabled")
        ] += 1
        if attention_metadata.get("decoder_counterfactual_latency_ms") is not None:
            intervention_latencies.append(
                float(attention_metadata["decoder_counterfactual_latency_ms"])
            )
    total = len(rows)
    summary = {
        "schema_version": "vsight_cable_raft_attention_audit_v1",
        "records": total,
        "evidence_v4_records": statuses.get("ok", 0) + statuses.get("attention_unavailable", 0),
        "status_counts": dict(statuses),
        "role_count_counts": {str(key): value for key, value in sorted(role_counts.items())},
        "three_role_rate": role_counts.get(3, 0) / total if total else None,
        "role_source_counts": dict(role_sources),
        "edge_rate": edge_count / total if total else None,
        "mean_role_confidence": statistics.mean(role_confidences) if role_confidences else None,
        "mean_top_transport": statistics.mean(transports) if transports else None,
        "mean_edge_uncertainty": statistics.mean(uncertainties) if uncertainties else None,
        "decoder_edge_rate": decoder_edges / total if total else None,
        "mean_decoder_top_transport": (
            statistics.mean(decoder_transports) if decoder_transports else None
        ),
        "mean_decoder_edge_uncertainty": (
            statistics.mean(decoder_uncertainties) if decoder_uncertainties else None
        ),
        "causal_edge_rate": causal_edges / total if total else None,
        "mean_causal_top_transport": (
            statistics.mean(causal_transports) if causal_transports else None
        ),
        "mean_causal_edge_uncertainty": (
            statistics.mean(causal_uncertainties) if causal_uncertainties else None
        ),
        "mean_causal_relative_drop": (
            statistics.mean(causal_relative_drops) if causal_relative_drops else None
        ),
        "positive_causal_relative_drop_rate": (
            sum(value > 0 for value in causal_relative_drops)
            / len(causal_relative_drops)
            if causal_relative_drops else None
        ),
        "causal_influence_edge_rate": influence_edges / total if total else None,
        "mean_causal_influence_top_transport": (
            statistics.mean(influence_transports) if influence_transports else None
        ),
        "mean_causal_influence_edge_uncertainty": (
            statistics.mean(influence_uncertainties)
            if influence_uncertainties else None
        ),
        "mean_hidden_cosine_drift": (
            statistics.mean(hidden_drifts) if hidden_drifts else None
        ),
        "hidden_cosine_drift": {
            "p50": statistics.median(hidden_drifts) if hidden_drifts else None,
            "p95": percentile(hidden_drifts, 0.95),
        },
        "positive_causal_influence_swap_margin_rate": (
            sum(value > 0 for value in influence_swap_margins)
            / len(influence_swap_margins)
            if influence_swap_margins else None
        ),
        "mean_causal_influence_swap_margin": (
            statistics.mean(influence_swap_margins)
            if influence_swap_margins else None
        ),
        "decoder_intervention_status_counts": dict(intervention_statuses),
        "decoder_counterfactual_latency_ms": {
            "p50": (
                statistics.median(intervention_latencies)
                if intervention_latencies else None
            ),
            "p95": percentile(intervention_latencies, 0.95),
        },
        "image_encoder_forwards": dict(forwards),
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
        },
        "attention_metadata_presence": dict(metadata),
        "action_policy_applied": False,
        "dataset_role": "development_diagnostic_not_final_test",
        "sealed_heldout_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
