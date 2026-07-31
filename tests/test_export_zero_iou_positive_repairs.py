import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_zero_iou_positive_repairs",
    ROOT / "scripts" / "export_zero_iou_positive_repairs.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PositiveExportTest(unittest.TestCase):
    def test_latest_successes_and_compact_record_preserve_review_gate(self):
        row = {
            "status": "ok",
            "base_sample_id": "a",
            "image_filename": "a.jpg",
            "source_expression": "the chair",
            "repair_prompt_sha256": "prompt",
            "repair": {
                "decision": "rewrite",
                "repaired_expression": "the blue chair",
                "source_expression_truth": "ambiguous",
                "head_object": "chair",
                "added_atoms": [],
                "removed_or_replaced_atoms": [],
                "evidence_citations": [],
                "confidence": "high",
                "reason": "blue distinguishes it",
            },
            "request": {
                "target_category_hint": "chair",
                "expression_structure": "attribute",
                "same_category_distractors_hint": 2,
                "human_review": None,
            },
        }
        record = MODULE.compact_record(row, "source")
        self.assertFalse(record["eligible_for_training"])
        self.assertEqual(record["review_status"], "pending_human_confirmation")
        self.assertEqual(record["repaired_expression"], "the blue chair")


if __name__ == "__main__":
    unittest.main()
