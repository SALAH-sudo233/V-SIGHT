import unittest

from vsight.single_pass import (
    candidate_reference_geometry_features,
    proposal_support_features,
    target_phrases,
)


class SinglePassFeatureTest(unittest.TestCase):
    def test_target_and_reference_phrases_are_query_only(self):
        phrases = target_phrases("The man to the left of the red car")
        self.assertEqual(phrases["head"], "person")
        self.assertEqual(phrases["full"], "man")
        self.assertEqual(phrases["reference"], "red car")

    def test_missing_candidate_has_zero_joint_support(self):
        features = proposal_support_features(
            None,
            [{"score": 0.8, "bbox_xyxy": [10, 10, 30, 30]}],
        )
        self.assertEqual(features["proposal_count"], 1.0)
        self.assertEqual(features["max_score"], 0.8)
        self.assertEqual(features["max_candidate_iou"], 0.0)
        self.assertEqual(features["max_score_iou_product"], 0.0)
        self.assertEqual(features["overlap_count_010"], 0.0)

    def test_joint_support_and_reference_direction(self):
        proposals = [
            {"score": 0.8, "bbox_xyxy": [10, 10, 30, 30]},
            {"score": 0.4, "bbox_xyxy": [60, 10, 80, 30]},
        ]
        support = proposal_support_features([10, 10, 30, 30], proposals)
        self.assertAlmostEqual(support["max_score_iou_product"], 0.8)
        self.assertEqual(support["overlap_count_050"], 1.0)

        geometry = candidate_reference_geometry_features(
            [10, 10, 30, 30], proposals[1:], 100, 100
        )
        self.assertEqual(geometry["valid"], 1.0)
        self.assertLess(geometry["score_weighted_dx"], 0.0)
        self.assertEqual(geometry["candidate_left_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
