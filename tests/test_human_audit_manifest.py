import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_human_audit_manifest", ROOT / "scripts" / "build_human_audit_manifest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HumanAuditManifestTest(unittest.TestCase):
    def test_only_valid_zero_and_regressions_are_grouped(self):
        base = {
            "task": "t2",
            "base_sample_id": "g1",
            "image_filename": "a.jpg",
            "query": "person on left",
            "expression_structure": "relation",
            "target_category": "person",
            "same_category_distractors": "2",
            "baseline_state": "valid_zero",
            "transition": "valid_zero_unresolved",
            "baseline_box": "[0, 0, 1, 1]",
            "result_box": "[1, 1, 2, 2]",
            "gt_box": "[2, 2, 3, 3]",
            "baseline_iou": "0",
            "result_iou": "0",
            "baseline_zero_box_class": "wrong_same_category_instance",
            "result_zero_box_class": "wrong_same_category_instance",
        }
        companion = {
            **base,
            "task": "t4",
            "baseline_state": "overlap",
            "transition": "nonzero_remained_nonzero",
        }
        ignored = {
            **base,
            "base_sample_id": "g2",
            "baseline_state": "overlap",
            "transition": "nonzero_remained_nonzero",
        }
        manifest = MODULE.build([base, companion, ignored])
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["base_sample_id"], "g1")
        self.assertEqual(len(manifest[0]["cases"]), 2)
        self.assertEqual(manifest[0]["review"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
