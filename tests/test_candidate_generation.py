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
    parse_t4_output,
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

    def test_t4_requires_consistent_json_decision_and_bbox(self):
        found = parse_t4_output(
            '{"description":"A person sits by a desk.","exists":"yes","bbox":[10,20,100,120]}',
            (200, 150),
        )
        self.assertTrue(found["parse_valid"])
        self.assertEqual(found["pred_bbox_xyxy"], [10.0, 20.0, 100.0, 120.0])
        rejected = parse_t4_output(
            '{"description":"An empty room.","exists":"no","bbox":"not found"}',
            (200, 150),
        )
        self.assertTrue(rejected["parse_valid"])
        self.assertFalse(rejected["pred_exists"])
        inconsistent = parse_t4_output(
            '{"description":"A room.","exists":"yes","bbox":"not found"}',
            (200, 150),
        )
        self.assertFalse(inconsistent["parse_valid"])

    def test_t4_accepts_qwen_bbox_alias_and_trailing_brace(self):
        parsed = parse_t4_output(
            '```json\n{"description":"A person by a desk.","exists":"yes","bbox_2d":[1,2,30,40]}}\n```',
            (100, 100),
        )
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["pred_bbox_xyxy"], [1.0, 2.0, 30.0, 40.0])
        stray_quote = parse_t4_output(
            '```json\n{"description":"A carriage.","exists":"yes","bbox_2d":[1,2,30,40]"}\n```',
            (100, 100),
        )
        self.assertTrue(stray_quote["parse_valid"])
        self.assertIn('{"boxes":', CHALLENGER_PROMPT.format(expr="the red car"))


if __name__ == "__main__":
    unittest.main()
