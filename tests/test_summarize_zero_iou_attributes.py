import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_zero_iou_attributes",
    ROOT / "scripts" / "summarize_zero_iou_attributes.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AttributeSummaryTest(unittest.TestCase):
    def test_selects_only_valid_box_zero_iou_groups(self):
        rows = [
            {
                "base_sample_id": "valid",
                "cases": [{"baseline_box": [0, 0, 1, 1], "baseline_iou": 0.0}],
            },
            {
                "base_sample_id": "false_rejection",
                "cases": [{"baseline_box": None, "baseline_iou": 0.0}],
            },
            {
                "base_sample_id": "nonzero",
                "cases": [{"baseline_box": [0, 0, 1, 1], "baseline_iou": 0.1}],
            },
        ]
        self.assertEqual(set(MODULE.valid_zero_iou_groups(rows)), {"valid"})

    def test_latest_human_status_and_model_success_take_precedence(self):
        reviews = [
            {"base_sample_id": "a", "reviewer_id": "r", "status": "completed"},
            {"base_sample_id": "a", "reviewer_id": "r", "status": "draft"},
            {"base_sample_id": "b", "reviewer_id": "r", "status": "completed"},
        ]
        self.assertEqual(MODULE.completed_human_ids(reviews), {"b"})

        successes, errors = MODULE.model_results([
            {"base_sample_id": "a", "status": "error"},
            {"base_sample_id": "a", "status": "ok"},
            {"base_sample_id": "a", "status": "error"},
        ])
        self.assertEqual(set(successes), {"a"})
        self.assertNotIn("a", errors)

    def test_flags_sensitive_gender_evidence_and_arrearage(self):
        identity = {
            "apparent_gender": "male_presenting",
            "gender_evidence": "clothing style and body build",
        }
        self.assertEqual(
            MODULE.gender_review_flag(identity), "stereotype_sensitive_cue"
        )
        self.assertEqual(
            MODULE.error_class("type: Arrearage, overdue-payment"), "arrearage"
        )


if __name__ == "__main__":
    unittest.main()
