import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repair_zero_iou_positives_bailian",
    ROOT / "scripts" / "repair_zero_iou_positives_bailian.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PositiveRepairTest(unittest.TestCase):
    def test_keep_requires_exact_source_and_no_edits(self):
        value = {
            "decision": "keep",
            "repaired_expression": "the red chair",
            "head_object": "chair",
            "source_expression_truth": "supported",
            "added_atoms": [],
            "removed_or_replaced_atoms": [],
            "evidence_citations": [],
            "confidence": "high",
            "reason": "the expression uniquely binds the target",
            "rejection_reason": "",
        }
        MODULE.validate_repair(value, "the red chair")
        value["repaired_expression"] = "the chair"
        with self.assertRaisesRegex(ValueError, "copy source"):
            MODULE.validate_repair(value, "the red chair")

    def test_rewrite_requires_a_traceable_edit(self):
        value = {
            "decision": "rewrite",
            "repaired_expression": "the blue chair by the window",
            "head_object": "chair",
            "source_expression_truth": "ambiguous",
            "added_atoms": [
                {"text": "blue", "type": "color", "evidence_cue": "blue upholstery"}
            ],
            "removed_or_replaced_atoms": [],
            "evidence_citations": ["blue upholstery"],
            "confidence": "high",
            "reason": "the blue upholstery distinguishes the chair",
            "rejection_reason": "",
        }
        MODULE.validate_repair(value, "the chair")
        value["repaired_expression"] = "the green box chair"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            MODULE.validate_repair(value, "the chair")

    def test_reject_cannot_emit_a_positive(self):
        value = {
            "decision": "reject",
            "repaired_expression": "",
            "head_object": "",
            "source_expression_truth": "contradicted",
            "added_atoms": [],
            "removed_or_replaced_atoms": [],
            "evidence_citations": [],
            "confidence": "high",
            "reason": "the relation is contradicted",
            "rejection_reason": "no truthful expression can be formed",
        }
        MODULE.validate_repair(value, "the chair")


if __name__ == "__main__":
    unittest.main()
