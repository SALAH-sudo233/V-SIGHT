import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class TracePilotSummaryTest(unittest.TestCase):
    def test_auroc_uses_average_ranks_for_ties(self):
        from scripts import summarize_trace_bind_pilot as module

        self.assertEqual(module.auroc([0.5, 0.5, 0.1], [True, False, False]), 0.75)

    def test_status_metrics_treat_unavailable_as_abstention(self):
        from scripts import summarize_trace_bind_pilot as module

        metrics = module._status_metrics(
            [
                {"label": True, "trace_status": "supported"},
                {"label": False, "trace_status": "unsupported"},
                {"label": False, "trace_status": "unavailable"},
            ]
        )
        self.assertEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["covered_accuracy"], 1.0)
        self.assertEqual(metrics["covered_records"], 2)

    def test_summary_is_diagnostic_without_labels(self):
        from scripts import summarize_trace_bind_pilot as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run.jsonl"
            output = root / "summary.json"
            row = {
                "record_id": "r1",
                "image_filename": "image.jpg",
                "detector_evidence": {
                    "trace_ledger": {
                        "evidence_status": "supported",
                        "action_policy_applied": False,
                        "statistics": {
                            "rank_stability": 1.0,
                            "layer_agreement": 1.0,
                            "stabilization_layer": 1,
                            "swap_persistence": 0.0,
                            "alternative_dominance": 0.0,
                            "trajectory_entropy": 0.1,
                            "bbox_convergence": 1.0,
                            "edge_uncertainty": 0.1,
                        },
                        "proposal_trajectories": [
                            {"states": [{"span_scores": {"target": 0.8}}]}
                        ],
                        "latency_ms": {"detector_only": 10.0, "trace_added": 2.0},
                    }
                },
            }
            run.write_text(json.dumps(row) + "\n", encoding="utf-8")
            original_argv = module.arguments
            module.arguments = lambda: type(
                "Args",
                (),
                {
                    "run": run,
                    "labels": None,
                    "output": output,
                    "bootstrap_samples": 10,
                    "seed": 1,
                    "force": False,
                },
            )()
            try:
                self.assertEqual(module.main(), 0)
            finally:
                module.arguments = original_argv
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(summary["diagnostic_only"])
            self.assertEqual(summary["matched_metrics"]["records"], 0)
            self.assertEqual(summary["action_policy_applied_records"], 0)
            self.assertEqual(summary["field_coverage"]["trace_ledger"], 1.0)
            payload_sha256 = summary.pop("payload_sha256")
            payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
            self.assertEqual(
                payload_sha256,
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
