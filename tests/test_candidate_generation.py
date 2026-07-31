import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.candidate_generation import (  # noqa: E402
    CHALLENGER_PROMPT,
    generation_spec,
    parse_baseline_output,
    parse_challenger_output,
)


class CandidateGenerationTest(unittest.TestCase):
    def test_baseline_parses_box_and_rejection(self):
        box = parse_baseline_output("[10,20,30,40]", (100, 80))
        self.assertTrue(box["pred_found"])
        self.assertEqual(box["pred_bbox_xyxy"], [10.0, 20.0, 30.0, 40.0])
        rejected = parse_baseline_output("not found", (100, 80))
        self.assertFalse(rejected["pred_found"])
        self.assertTrue(rejected["parse_valid"])

    def test_challenger_selects_first_ordered_box(self):
        result = parse_challenger_output(
            '{"boxes":[[1,2,30,40],[40,2,70,40]]}', (100, 80)
        )
        self.assertEqual(result["selected_bbox_xyxy"], [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(len(result["candidate_boxes_xyxy"]), 2)

    def test_normalized_coordinates_are_scaled(self):
        result = parse_baseline_output("[0.1,0.2,0.5,0.75]", (100, 80))
        self.assertEqual(result["pred_bbox_xyxy"], [10.0, 16.0, 50.0, 60.0])

    def test_prompt_hashes_are_frozen(self):
        spec = generation_spec()
        self.assertEqual(len(spec["baseline_prompt_sha256"]), 64)
        self.assertEqual(len(spec["challenger_prompt_sha256"]), 64)
        self.assertIn('{"boxes":', CHALLENGER_PROMPT.format(expr="the red car"))


if __name__ == "__main__":
    unittest.main()
