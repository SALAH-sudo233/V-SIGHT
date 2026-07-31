import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e2_verifier import (  # noqa: E402
    candidate_geometry,
    choose_safe_threshold,
    selector_metrics,
)


class CandidateGeometryTest(unittest.TestCase):
    def test_geometry_is_candidate_order_equivariant(self):
        first = [0, 0, 20, 40]
        second = [40, 10, 80, 50]
        a = candidate_geometry(first, second, 100, 100)
        b = candidate_geometry(second, first, 100, 100)
        self.assertEqual(len(a), 18)
        self.assertAlmostEqual(a[14], b[14])
        self.assertAlmostEqual(a[15], -b[15])
        self.assertAlmostEqual(a[16], -b[16])
        self.assertAlmostEqual(a[17], -b[17])


class SelectorMetricTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "query_id": "gain",
                "selector_eligible": True,
                "selector_action": "switch",
                "baseline_iou": 0.0,
                "challenger_iou": 0.8,
            },
            {
                "query_id": "risk",
                "selector_eligible": True,
                "selector_action": "keep",
                "baseline_iou": 0.7,
                "challenger_iou": 0.0,
            },
            {
                "query_id": "locked",
                "selector_eligible": False,
                "selector_action": None,
                "baseline_iou": 0.0,
                "challenger_iou": 0.9,
            },
        ]

    def test_locked_rows_are_never_recovered(self):
        result = selector_metrics(
            self.rows, {"gain": 2.0, "risk": 1.0, "locked": 99.0}, 0.0
        )
        self.assertEqual(result["switches"], 2)
        self.assertEqual(result["nonzero_to_zero_regressions"], 1)
        self.assertAlmostEqual(result["selector_miou"], 0.8 / 3)
        self.assertAlmostEqual(result["state_preserving_challenger_miou"], 0.8 / 3)
        self.assertAlmostEqual(result["raw_forced_challenger_miou"], 1.7 / 3)

    def test_safe_threshold_respects_regression_budget(self):
        threshold, result = choose_safe_threshold(
            self.rows, {"gain": 2.0, "risk": 1.0}, max_nonzero_to_zero=0
        )
        self.assertEqual(threshold, 1.0)
        self.assertEqual(result["switches"], 1)
        self.assertEqual(result["nonzero_to_zero_regressions"], 0)
        self.assertAlmostEqual(result["oracle_gap_capture_fraction"], 1.0)

    def test_all_keep_boundary_is_available(self):
        _, result = choose_safe_threshold(
            self.rows, {"gain": -2.0, "risk": 2.0}, max_nonzero_to_zero=0
        )
        self.assertTrue(math.isfinite(result["selector_miou"]))
        self.assertEqual(result["nonzero_to_zero_regressions"], 0)


if __name__ == "__main__":
    unittest.main()
