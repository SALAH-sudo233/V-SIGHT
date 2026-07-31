import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_zero_iou_strata", ROOT / "scripts" / "analyze_zero_iou_strata.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StratifiedAnalysisTest(unittest.TestCase):
    def test_valid_zero_cases_excludes_false_rejections(self):
        group = {
            "cases": [
                {"baseline_box": None, "baseline_iou": 0.0},
                {"baseline_box": [0, 0, 1, 1], "baseline_iou": 0.0},
                {"baseline_box": [0, 0, 1, 1], "baseline_iou": 0.2},
            ]
        }
        self.assertEqual(len(MODULE.valid_zero_cases(group)), 1)

    def test_latest_model_keeps_only_successes(self):
        rows = [
            {"base_sample_id": "a", "status": "error"},
            {"base_sample_id": "a", "status": "ok"},
            {"base_sample_id": "b", "status": "error"},
        ]
        self.assertEqual(set(MODULE.latest_model(rows)), {"a"})

    def test_percentage_handles_zero(self):
        self.assertEqual(MODULE.percentage(0, 0), "0.0%")
        self.assertEqual(MODULE.percentage(1, 4), "25.0%")


if __name__ == "__main__":
    unittest.main()
