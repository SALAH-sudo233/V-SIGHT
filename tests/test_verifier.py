import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.verifier import CandidateEvidence, VerifierWeights, score_candidates  # noqa: E402


class VerifierScoreTest(unittest.TestCase):
    def test_contradiction_is_subtracted_from_candidate_logit(self):
        weights = VerifierWeights(
            object_weight=1,
            relation_weight=2,
            action_attribute_weight=0,
            localization_weight=0,
            contradiction_weight=3,
        )
        evidence = CandidateEvidence(2, 1, 9, 9, contradiction_score=1)
        self.assertEqual(weights.candidate_logit(evidence), 1.0)

    def test_difficulty_changes_null_not_candidate_preference(self):
        weights = VerifierWeights(null_difficulty_weight=2)
        evidence = CandidateEvidence(1, 1, 1, 1)
        candidate = weights.candidate_logit(evidence)
        easy_null = weights.null_logit(difficulty_score=0)
        hard_null = weights.null_logit(difficulty_score=2)
        self.assertEqual(candidate, weights.candidate_logit(evidence))
        self.assertGreater(hard_null, easy_null)

    def test_invalid_scores_and_empty_batch_fail(self):
        with self.assertRaises(ValueError):
            CandidateEvidence(float("nan"), 0, 0, 0)
        with self.assertRaises(ValueError):
            score_candidates({})
        self.assertTrue(math.isfinite(score_candidates({"b": CandidateEvidence(1, 0, 0, 0)})["b"]))


if __name__ == "__main__":
    unittest.main()
