import json
import unittest

from vsight.trajectory_ledger import (
    build_trajectory_ledger,
    select_candidate_query_ids,
)


class TrajectoryLedgerTest(unittest.TestCase):
    def setUp(self):
        self.boxes = [
            [[0.20, 0.20, 0.20, 0.20], [0.70, 0.20, 0.20, 0.20], [0.50, 0.70, 0.20, 0.20]],
            [[0.21, 0.20, 0.20, 0.20], [0.69, 0.20, 0.20, 0.20], [0.50, 0.69, 0.20, 0.20]],
            [[0.21, 0.20, 0.20, 0.20], [0.69, 0.20, 0.20, 0.20], [0.50, 0.69, 0.20, 0.20]],
        ]
        self.scores = {
            "target": [[0.8, 0.2, 0.1], [0.85, 0.2, 0.1], [0.9, 0.2, 0.1]],
            "reference": [[0.1, 0.8, 0.2], [0.1, 0.85, 0.2], [0.1, 0.9, 0.2]],
            "predicate": [[0.4, 0.4, 0.1], [0.5, 0.5, 0.1], [0.6, 0.6, 0.1]],
            "full_query": [[0.7, 0.3, 0.1], [0.8, 0.3, 0.1], [0.9, 0.3, 0.1]],
        }

    def test_tracks_query_ids_and_emits_json_safe_summaries(self):
        ledger = build_trajectory_ledger(
            layer_boxes_cxcywh=self.boxes,
            span_scores=self.scores,
            candidate_cap=2,
            hidden_l2_norms=[[1.0, 2.0, 3.0]] * 3,
            hidden_cosine_to_previous=[
                [None, None, None],
                [0.9, 0.8, 0.7],
                [0.99, 0.98, 0.97],
            ],
        )
        self.assertEqual(ledger["schema_version"], "vsight_trajectory_ledger_v1")
        self.assertEqual(ledger["tracking_key"], "detector_object_query_index")
        self.assertFalse(ledger["hidden_tensors_serialized"])
        self.assertEqual(ledger["candidate_query_ids"], [0, 1, 2])
        first = ledger["proposal_trajectories"][0]
        self.assertEqual(first["object_query_id"], 0)
        self.assertEqual([row["object_query_id"] for row in first["states"]], [0, 0, 0])
        self.assertEqual(first["states"][0]["rank_by_role"]["target"], 1)
        self.assertIsNone(first["states"][0]["hidden_summary"]["cosine_to_previous"])
        self.assertGreater(ledger["statistics"]["bbox_convergence"], 0.9)
        json.dumps(ledger, allow_nan=False)

    def test_candidate_selection_is_final_layer_role_union(self):
        selected = select_candidate_query_ids(
            self.scores, roles=("target", "reference"), candidate_cap=1
        )
        self.assertEqual(selected, (0, 1))

    def test_rank_switching_reduces_stability_and_delays_stabilization(self):
        scores = dict(self.scores)
        scores["target"] = [[0.2, 0.9, 0.1], [0.9, 0.2, 0.1], [0.9, 0.2, 0.1]]
        ledger = build_trajectory_ledger(
            layer_boxes_cxcywh=self.boxes,
            span_scores=scores,
            candidate_cap=2,
        )
        target = ledger["role_statistics"]["target"]
        self.assertLess(target["rank_stability"], 1.0)
        self.assertEqual(target["stabilization_layer"], 1)

    def test_rejects_more_than_five_candidates_per_role(self):
        with self.assertRaisesRegex(ValueError, "candidate cap"):
            build_trajectory_ledger(
                layer_boxes_cxcywh=self.boxes,
                span_scores=self.scores,
                candidate_cap=6,
            )


if __name__ == "__main__":
    unittest.main()
