import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "summarize_agentic_drift_shadow",
    Path(__file__).parents[1] / "scripts" / "summarize_agentic_drift_shadow.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(model, task, state, action, risk=0.2):
    return {
        "schema_version": "vsight_agentic_drift_audit_v2",
        "context": {"model": model, "task": task},
        "evidence_state": state,
        "action": action,
        "drift_risk_raw": risk,
        "review_priority": risk,
        "latency_ms": 2.0,
        "reason_codes": ["CLAIM_SUPPORT_WEAK"],
        "trace": [],
        "rounds_executed": 3,
        "stop_reason": "MAX_ROUNDS_REACHED",
        "audit_disposition": "REVIEW_REQUIRED" if action == "ABSTAIN" else "PROVISIONAL_ACCEPT",
        "effective_config_sha256": "a" * 64,
        "budget_ledger": {"additional_detector_image_encoder_forwards": 0},
    }


class AgenticShadowSummaryTest(unittest.TestCase):
    def test_groups_by_model_and_task_without_human_labels(self):
        report = MODULE.summarize(
            [
                row("a", "t2", "WRONG_INSTANCE", "ABSTAIN"),
                row("a", "t2", "SUPPORTED_CORRECT", "ACCEPT"),
                row("b", "t4", "UNOBSERVABLE_AMBIGUOUS", "ABSTAIN"),
            ],
            input_sha256="hash",
        )
        self.assertEqual(report["model_task_groups"], 2)
        self.assertEqual(report["by_model_task"]["a|t2"]["records"], 2)
        self.assertEqual(report["by_model_task"]["a|t2"]["rates"]["accept"], 0.5)
        self.assertFalse(report["human_labels_used"])
        self.assertFalse(report["self_memory_written"])
        self.assertEqual(report["overall"]["mean_rounds_executed"], 3.0)
        self.assertEqual(report["additional_detector_image_encoder_forwards_max"], 0)
        self.assertEqual(report["effective_config_sha256"], "a" * 64)

    def test_memory_retriever_steps_are_explicitly_counted(self):
        item = row("a", "t2", "UNOBSERVABLE_AMBIGUOUS", "ABSTAIN")
        item["trace"] = [{"name": "memory_retriever"}]
        report = MODULE.summarize([item], input_sha256="hash")
        self.assertEqual(report["memory_retriever_steps"], 1)

    def test_rejects_missing_or_invalid_effective_config_hash(self):
        invalid_values = (None, "", "   ", "not-a-sha256", 123)
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                item = row("a", "t2", "UNOBSERVABLE_AMBIGUOUS", "ABSTAIN")
                item["effective_config_sha256"] = invalid_value
                with self.assertRaisesRegex(ValueError, "effective_config_sha256"):
                    MODULE.summarize([item], input_sha256="hash")

        item = row("a", "t2", "UNOBSERVABLE_AMBIGUOUS", "ABSTAIN")
        del item["effective_config_sha256"]
        with self.assertRaisesRegex(ValueError, "effective_config_sha256"):
            MODULE.summarize([item], input_sha256="hash")

    def test_rejects_mixed_effective_config_hashes_across_models(self):
        first = row("a", "t2", "UNOBSERVABLE_AMBIGUOUS", "ABSTAIN")
        second = row("b", "t4", "SUPPORTED_CORRECT", "ACCEPT")
        second["effective_config_sha256"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "mixed effective_config_sha256"):
            MODULE.summarize([first, second], input_sha256="hash")

    def test_rejects_empty_audit_without_an_effective_config_hash(self):
        with self.assertRaisesRegex(ValueError, "at least one row"):
            MODULE.summarize([], input_sha256="hash")


if __name__ == "__main__":
    unittest.main()
