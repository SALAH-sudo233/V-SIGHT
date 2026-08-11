import unittest

from vsight.ccv_learning import (
    IsotonicCalibrator,
    MonotonicActionHead,
    apply_action_head,
    eligible_cable_review_record,
    eligible_review_record,
)
from vsight.ccv import CCVAction, CCVResult
from vsight.ccv_metrics import evaluate_router, evaluate_safety, latency_summary


class CCVLearningTest(unittest.TestCase):
    def test_isotonic_pooling_is_monotone(self):
        calibrator = IsotonicCalibrator.fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
        values = calibrator.predict_many([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(values, sorted(values))

    def test_action_head_respects_declared_weight_signs(self):
        rows = [
            {"absence": 0.0, "margin": 0.0},
            {"absence": 0.1, "margin": 0.2},
            {"absence": 0.9, "margin": 0.0},
            {"absence": 1.0, "margin": 0.8},
        ]
        actions = ["ACCEPT", "RELOCALIZE", "REJECT", "REJECT"]
        head = MonotonicActionHead.fit(
            rows,
            actions,
            reject_directions={"absence": 1},
            binding_directions={"margin": 1},
        )
        self.assertTrue(head.reject_model.audit_monotonicity())
        self.assertTrue(head.binding_model.audit_monotonicity())
        self.assertGreaterEqual(head.reject_model.weights[0], 0)
        self.assertGreaterEqual(head.binding_model.weights[0], 0)
        restored = MonotonicActionHead.from_dict(head.to_dict())
        self.assertEqual(restored.predict(rows[0])[0], head.predict(rows[0])[0])

    def test_learned_relocalize_without_alternative_preserves_upstream(self):
        rows = [{"absence": 0.0, "margin": value} for value in (0.0, 0.1, 0.9, 1.0)]
        head = MonotonicActionHead.fit(
            rows,
            ["ACCEPT", "ACCEPT", "RELOCALIZE", "RELOCALIZE"],
            reject_directions={"absence": 1},
            binding_directions={"margin": 1},
        )
        result = CCVResult(
            action=CCVAction.ACCEPT,
            corrected_bbox=None,
            atom_evidence={"null_support": 0.0, "contradiction_support": 0.0},
            existence_margin=0.0,
            binding_margin=1.0,
            calibrated_risk=0.5,
            latency_metadata={},
            reason="test",
            original_bbox=(0, 0, 1, 1),
            alternative_bbox=None,
            claim_support=0.0,
            alternative_support=0.0,
            evidence_sufficient=True,
        )
        # Use a small adapter head whose feature names match action_features.
        compatible = MonotonicActionHead.fit(
            [
                {"claim_support": 0.0, "alternative_support": 0.0, "binding_margin": 0.0},
                {"claim_support": 0.0, "alternative_support": 0.0, "binding_margin": 1.0},
            ],
            ["ACCEPT", "RELOCALIZE"],
            reject_directions={"claim_support": -1},
            binding_directions={"binding_margin": 1},
        )
        applied = apply_action_head(result, compatible)
        self.assertEqual(applied.action, CCVAction.ACCEPT)
        self.assertIsNone(applied.corrected_bbox)

    def test_training_eligibility_excludes_unreviewed_or_sensitive(self):
        row = {
            "verifier_action": "ACCEPT",
            "reviewer_count": 1,
            "annotation_protocol": "single_project_owner",
            "minimum_confidence": 0.95,
            "source_queue_sha256": "abc",
        }
        self.assertTrue(eligible_review_record(row))
        self.assertFalse(eligible_review_record({**row, "sensitive_attribute": True}))
        self.assertFalse(eligible_review_record({**row, "verifier_action": "UNCERTAIN"}))
        self.assertFalse(eligible_review_record({**row, "reviewer_count": 2}))
        self.assertFalse(
            eligible_review_record({**row, "annotation_protocol": "dual_review"})
        )

    def test_single_owner_relocalize_requires_one_corrected_box(self):
        row = {
            "verifier_action": "RELOCALIZE",
            "reviewer_count": 1,
            "annotation_protocol": "single_project_owner",
            "minimum_confidence": 0.95,
            "source_queue_sha256": "abc",
            "router_only_corrected_boxes_xyxy": [[1, 2, 10, 20]],
        }
        self.assertTrue(eligible_review_record(row))
        self.assertFalse(
            eligible_review_record(
                {**row, "router_only_corrected_boxes_xyxy": []}
            )
        )

    def test_single_owner_relation_requires_one_reference_box(self):
        row = {
            "verifier_action": "ACCEPT",
            "reviewer_count": 1,
            "annotation_protocol": "single_project_owner",
            "minimum_confidence": 0.95,
            "source_queue_sha256": "abc",
            "schema_gate_passed": True,
            "stage_b_applicable_atoms": ["identity", "relation"],
            "stage_b_authoritative_atoms": {
                "identity": "supported",
                "relation": "supported",
            },
            "independent_reference_boxes_xyxy": [[20, 2, 30, 20]],
        }
        self.assertTrue(eligible_cable_review_record(row))
        self.assertFalse(
            eligible_cable_review_record(
                {**row, "independent_reference_boxes_xyxy": []}
            )
        )


class CCVMetricsTest(unittest.TestCase):
    def test_safety_reports_strata_gap_and_no_blanket_rejection(self):
        rows = [
            {"label_exists": True, "action": "ACCEPT", "original_iou": 0.8},
            {"label_exists": True, "action": "REJECT", "original_iou": 0.2},
            {"label_exists": False, "action": "REJECT", "hallucination_type": "object"},
            {"label_exists": False, "action": "ACCEPT", "hallucination_type": "co_occurrence"},
            {"label_exists": False, "action": "REJECT", "hallucination_type": "attribute"},
            {"label_exists": False, "action": "ACCEPT", "hallucination_type": "relation"},
        ]
        result = evaluate_safety(rows)
        self.assertEqual(result["post_far"], 0.5)
        self.assertEqual(result["added_fnr"], 0.5)
        self.assertEqual(result["roh_minus_boh_gap_delta"], 0.0)
        self.assertFalse(result["blanket_rejection"])

    def test_router_metrics_keep_trigger_and_applied_separate(self):
        rows = [
            {
                "action": "RELOCALIZE", "corrected_bbox": [1, 1, 2, 2],
                "relocalize_correct": True, "should_relocalize": True,
                "original_iou": 0.0, "corrected_iou": 0.7,
                "upstream_caption": "caption",
            },
            {
                "action": "RELOCALIZE", "corrected_bbox": None,
                "should_relocalize": True, "original_iou": 0.4,
            },
            {"action": "ACCEPT", "should_relocalize": False, "original_iou": 0.8},
        ]
        result = evaluate_router(rows)
        self.assertEqual(result["triggered"], 2)
        self.assertEqual(result["applied"], 1)
        self.assertEqual(result["iou_zero_repair_rate"], 1.0)
        self.assertEqual(result["caption_preservation_rate"], 1.0)
        self.assertEqual(latency_summary([{"detector_latency_ms": 10}, {"detector_latency_ms": 20}])["p50_ms"], 10)


if __name__ == "__main__":
    unittest.main()
