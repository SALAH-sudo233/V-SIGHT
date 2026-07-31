import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.decision import (  # noqa: E402
    Action,
    DecisionPolicy,
    RegionCandidate,
    SupportScores,
)


def candidate(candidate_id: str, score: float) -> RegionCandidate:
    return RegionCandidate(
        candidate_id=candidate_id,
        source=candidate_id,
        box=(0, 0, 10, 10),
        support=SupportScores(score, 0.0, 0.0),
    )


class DecisionPolicyTest(unittest.TestCase):
    def test_keep_switch_and_reject(self):
        policy = DecisionPolicy()
        keep = policy.decide(
            baseline=candidate("b", 3),
            challenger=candidate("c", 2),
            null_logit=1,
        )
        switch = policy.decide(
            baseline=candidate("b", 2),
            challenger=candidate("c", 3),
            null_logit=1,
        )
        reject = policy.decide(
            baseline=candidate("b", 2),
            challenger=candidate("c", 1),
            null_logit=3,
        )
        self.assertEqual(keep.action, Action.KEEP)
        self.assertEqual(switch.action, Action.SWITCH)
        self.assertEqual(reject.action, Action.REJECT)
        self.assertIsNone(reject.selected_box)

    def test_margins_default_to_baseline(self):
        policy = DecisionPolicy(switch_margin=0.5, reject_margin=0.5)
        switch_blocked = policy.decide(
            baseline=candidate("b", 2.0),
            challenger=candidate("c", 2.2),
            null_logit=0.0,
        )
        reject_blocked = policy.decide(
            baseline=candidate("b", 2.0),
            challenger=None,
            null_logit=2.2,
        )
        self.assertEqual(switch_blocked.action, Action.KEEP)
        self.assertEqual(switch_blocked.reason, "switch_margin_not_met")
        self.assertEqual(reject_blocked.action, Action.KEEP)
        self.assertEqual(reject_blocked.reason, "reject_margin_not_met")

    def test_recovery_from_baseline_null_is_explicit(self):
        conservative = DecisionPolicy().decide(
            baseline=None,
            challenger=candidate("c", 5),
            null_logit=0,
        )
        recovery = DecisionPolicy(allow_recovery_from_null=True).decide(
            baseline=None,
            challenger=candidate("c", 5),
            null_logit=0,
        )
        self.assertEqual(conservative.action, Action.REJECT)
        self.assertEqual(recovery.action, Action.SWITCH)

    def test_probabilities_are_stable_and_normalized(self):
        decision = DecisionPolicy(temperature=0.5).decide(
            baseline=candidate("b", 1000),
            challenger=candidate("c", 999),
            null_logit=-1000,
        )
        self.assertTrue(all(math.isfinite(x) for x in decision.probabilities.values()))
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0)

    def test_invalid_box_and_score_fail(self):
        with self.assertRaises(ValueError):
            RegionCandidate("b", (0, 0, 0, 10), SupportScores(1, 1, 1), "base")
        with self.assertRaises(ValueError):
            SupportScores(float("nan"), 1, 1)


if __name__ == "__main__":
    unittest.main()
