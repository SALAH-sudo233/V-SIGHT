import json
import unittest

from vsight.trace_bind import TraceSpan, build_trace_bind_ledger
from vsight.trajectory_ledger import build_trajectory_ledger


class TraceBindTest(unittest.TestCase):
    def setUp(self):
        boxes = [
            [[0.20, 0.20, 0.20, 0.20], [0.70, 0.20, 0.20, 0.20], [0.50, 0.70, 0.20, 0.20]],
            [[0.21, 0.20, 0.20, 0.20], [0.69, 0.20, 0.20, 0.20], [0.50, 0.69, 0.20, 0.20]],
            [[0.21, 0.20, 0.20, 0.20], [0.69, 0.20, 0.20, 0.20], [0.50, 0.69, 0.20, 0.20]],
        ]
        scores = {
            "target": [[0.8, 0.1, 0.2], [0.85, 0.1, 0.2], [0.9, 0.1, 0.2]],
            "predicate": [[0.5, 0.4, 0.2], [0.6, 0.5, 0.2], [0.7, 0.6, 0.2]],
            "reference": [[0.1, 0.8, 0.2], [0.1, 0.85, 0.2], [0.1, 0.9, 0.2]],
            "full_query": [[0.8, 0.2, 0.2], [0.85, 0.2, 0.2], [0.9, 0.2, 0.2]],
        }
        self.trajectory = build_trajectory_ledger(
            layer_boxes_cxcywh=boxes,
            span_scores=scores,
            candidate_cap=2,
        )
        self.spans = (
            TraceSpan("target", "the cup", (1,), ((4, 7),), 0.9, "test"),
            TraceSpan("predicate", "next to", (2, 3), ((8, 12), (13, 15)), 0.8, "test"),
            TraceSpan("reference", "the plate", (5,), ((20, 25),), 0.9, "test"),
            TraceSpan("full_query", "the cup next to the plate", (1, 2, 3, 5), ((4, 7), (8, 12), (13, 15), (20, 25)), 1.0, "raw_query"),
        )

    def test_builds_typed_assignment_and_fixed_swap_witnesses(self):
        ledger = build_trace_bind_ledger(
            sample_id="sample-1",
            query="the cup next to the plate",
            upstream_box_xyxy=[10, 10, 30, 30],
            spans=self.spans,
            trajectory_ledger=self.trajectory,
            image_width=100,
            image_height=100,
            candidate_cap=2,
            detector_only_latency_ms=10.0,
            trace_added_latency_ms=2.0,
        )
        self.assertEqual(ledger["schema_version"], "vsight_trace_bind_pilot_v1")
        self.assertEqual(ledger["upstream_target_object_query_id"], 0)
        self.assertEqual(len(ledger["assignment_ledger"]), 3)
        self.assertFalse(ledger["action_policy_applied"])
        self.assertEqual(ledger["budget"]["detector_image_encoder_forwards"], 1)
        for row in ledger["assignment_ledger"]:
            self.assertEqual(row["upstream"]["target_object_query_id"], 0)
            self.assertIsNotNone(row["role_swap"])
        for key in (
            "rank_stability",
            "layer_agreement",
            "stabilization_layer",
            "swap_persistence",
            "alternative_dominance",
            "trajectory_entropy",
            "bbox_convergence",
            "edge_uncertainty",
        ):
            self.assertIn(key, ledger["statistics"])
        json.dumps(ledger, allow_nan=False)

    def test_missing_upstream_box_is_explicitly_unavailable(self):
        ledger = build_trace_bind_ledger(
            sample_id="sample-1",
            query="the cup next to the plate",
            upstream_box_xyxy=None,
            spans=self.spans,
            trajectory_ledger=self.trajectory,
            image_width=100,
            image_height=100,
        )
        self.assertEqual(ledger["evidence_status"], "unavailable")
        self.assertEqual(ledger["assignment_ledger"], [])

    def test_rejects_budget_violation(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_trace_bind_ledger(
                sample_id="sample-1",
                query="the cup next to the plate",
                upstream_box_xyxy=[10, 10, 30, 30],
                spans=self.spans,
                trajectory_ledger=self.trajectory,
                image_width=100,
                image_height=100,
                image_encoder_forwards=2,
            )


if __name__ == "__main__":
    unittest.main()
