import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.relation_supervision import (  # noqa: E402
    category_mentions,
    extract_reference_phrase,
    candidate_reference_features,
    normalized_box_features,
    relation_context_features,
    parse_relation,
)


class RelationSupervisionTest(unittest.TestCase):
    categories = ("person", "chair", "dining table", "dog", "umbrella")

    def test_extracts_distinct_reference_category(self):
        parsed = parse_relation(
            "the chair next to the person", "chair", self.categories
        )
        self.assertEqual(parsed.relation, "next_to")
        self.assertEqual(parsed.reference_categories, ("person",))

    def test_same_category_requires_two_mentions(self):
        parsed = parse_relation(
            "the person behind the other person", "person", self.categories
        )
        self.assertEqual(parsed.relation, "behind")
        self.assertEqual(parsed.reference_categories, ("person",))

    def test_long_alias_wins_over_overlapping_alias(self):
        mentions = category_mentions("cup on dining table", self.categories)
        self.assertEqual([value.category for value in mentions], ["dining table"])

    def test_attribute_only_query_has_no_relation(self):
        parsed = parse_relation("the woman in red", "person", self.categories)
        self.assertIsNone(parsed.relation)
        self.assertEqual(parsed.reference_categories, ())

    def test_extracts_reference_suffix_without_determiner(self):
        self.assertEqual(
            extract_reference_phrase("the chair next to the red table"), "red table"
        )

    def test_candidate_reference_geometry_has_direction_and_overlap(self):
        candidate = [0, 0, 20, 20]
        reference = [20, 0, 40, 20]
        features = candidate_reference_features(candidate, reference, 0.8, 100, 100)
        self.assertEqual(len(features), 32)
        self.assertLess(features[21], 0)
        self.assertEqual(features[27], 0.0)
        self.assertEqual(len(normalized_box_features(candidate, 100, 100)), 10)

    def test_relation_context_is_fixed_width_and_order_invariant(self):
        proposals = [
            {"bbox_xyxy": [20, 0, 40, 20], "score": 0.8},
            {"bbox_xyxy": [50, 0, 70, 20], "score": 0.4},
        ]
        first = relation_context_features([0, 0, 20, 20], proposals, "next_to", 100, 100)
        second = relation_context_features(
            [0, 0, 20, 20], list(reversed(proposals)), "next_to", 100, 100
        )
        self.assertEqual(len(first), 85)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
