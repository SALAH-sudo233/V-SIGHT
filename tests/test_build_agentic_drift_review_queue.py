import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "build_agentic_drift_review_queue",
    Path(__file__).parents[1] / "scripts/build_agentic_drift_review_queue.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def audit_row(record_id, state, action="ABSTAIN", priority=0.5):
    return {
        "record_id": record_id,
        "query": "the red car next to the person",
        "original_bbox": [1, 2, 30, 40],
        "alternative_bbox": [40, 2, 70, 40],
        "evidence_state": state,
        "action": action,
        "drift_risk_raw": priority,
        "review_priority": priority,
        "reason_codes": ["ALTERNATIVE_DOMINATES"],
        "claim_support": 0.2,
        "alternative_support": 0.5,
        "binding_margin": 0.3,
        "memory_status": "REVIEW_REQUIRED",
        "context": {
            "group_id": f"group-{record_id}",
            "image_filename": "image.jpg",
            "model": "must-not-leak",
            "task": "must-not-leak",
        },
    }


class AgenticReviewQueueTest(unittest.TestCase):
    def test_selection_keeps_wrong_instance_and_blinds_hypotheses(self):
        rows = [
            audit_row("wrong", "WRONG_INSTANCE", priority=0.9),
            audit_row("amb-high", "UNOBSERVABLE_AMBIGUOUS", priority=0.8),
            audit_row("amb-low", "UNOBSERVABLE_AMBIGUOUS", priority=0.1),
            audit_row("accept", "SUPPORTED_CORRECT", "ACCEPT", 0.2),
        ]
        selected = MODULE.select_rows(
            rows, ambiguous_high_priority=1, accept_controls=1, low_risk_controls=1
        )
        queue, sidecar = MODULE.build_rows(selected)
        self.assertEqual(len(queue), 4)
        self.assertEqual(len(sidecar), 4)
        for row in queue:
            self.assertFalse(MODULE.QUEUE_FORBIDDEN_FIELDS & set(row))
            self.assertNotIn("model", row)
            self.assertEqual(
                row["stage_b_applicable_atoms"],
                ["identity", "attribute", "relation"],
            )
        self.assertEqual(
            {row["selection_band"] for row in sidecar},
            {
                "provisional_wrong_instance", "ambiguous_high_priority",
                "accept_control", "low_risk_control",
            },
        )

    def test_selection_excludes_rows_without_upstream_box(self):
        row = audit_row("missing", "WRONG_INSTANCE")
        row["original_bbox"] = None
        selected = MODULE.select_rows(
            [row], ambiguous_high_priority=1, accept_controls=1, low_risk_controls=1
        )
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
