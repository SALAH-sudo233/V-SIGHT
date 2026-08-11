import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "summarize_agentic_drift_reviews",
    Path(__file__).parents[1] / "scripts" / "summarize_agentic_drift_reviews.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AgenticReviewSummaryTest(unittest.TestCase):
    def test_adjacent_summary_path_supports_natural_queue_name(self):
        queue = Path("/tmp/agentic_drift_natural_review_queue.jsonl.gz")
        self.assertEqual(
            MODULE.adjacent_queue_summary_path(queue),
            Path("/tmp/agentic_drift_natural_review_queue.summary.json"),
        )

    def test_validated_natural_manifest_drives_report_semantics(self):
        queue = [
            {
                "schema_version": "vsight_agentic_drift_natural_review_input_v1",
                "data_role": "development_agentic_natural_audit_not_training",
            }
        ]
        summary = {
            "schema_version": "vsight_agentic_drift_natural_review_queue_summary_v1",
            "dataset_role": "development_agentic_natural_audit_not_training",
            "selection_rule": "hash stratified; no agent outcome fields used",
        }
        cohort = MODULE.resolve_cohort(queue, queue_summary=summary)
        self.assertEqual(cohort["cohort_kind"], "natural")
        self.assertEqual(
            cohort["dataset_role"],
            "development_agentic_natural_audit_not_training",
        )
        self.assertEqual(cohort["metadata_source"], "validated_queue_summary")
        self.assertIn(summary["selection_rule"], cohort["interpretation_warning"])
        self.assertNotIn("risk-enriched", cohort["selection_bias_warning"])

    def test_default_diagnostic_queue_semantics_remain_compatible(self):
        cohort = MODULE.resolve_cohort(
            [
                {
                    "schema_version": "vsight_agentic_drift_review_input_v1",
                    "data_role": "development_agentic_audit_not_training",
                }
            ]
        )
        self.assertEqual(cohort["cohort_kind"], "diagnostic")
        self.assertEqual(
            cohort["dataset_role"], "development_agentic_audit_not_training"
        )
        self.assertIn("risk-enriched", cohort["selection_bias_warning"])

    def test_explicit_cohort_cannot_override_conflicting_queue_metadata(self):
        with self.assertRaisesRegex(ValueError, "冲突"):
            MODULE.resolve_cohort(
                [
                    {
                        "schema_version": "vsight_agentic_drift_natural_review_input_v1",
                        "data_role": "development_agentic_natural_audit_not_training",
                    }
                ],
                requested="diagnostic",
            )

    def test_queue_summary_validation_binds_hash_count_and_natural_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.summary.json"
            summary = {
                "schema_version": "vsight_agentic_drift_natural_review_queue_summary_v1",
                "dataset_role": "development_agentic_natural_audit_not_training",
                "queue_sha256": "a" * 64,
                "records": 2,
                "selection_rule": "no agent outcome fields used",
            }
            path.write_text(json.dumps(summary), encoding="utf-8")
            loaded = MODULE.validated_queue_summary(
                path, queue_hash="a" * 64, queue_records=2
            )
            self.assertEqual(loaded, summary)
            with self.assertRaisesRegex(ValueError, "记录数"):
                MODULE.validated_queue_summary(
                    path, queue_hash="a" * 64, queue_records=1
                )

    def test_effective_config_identity_is_unique_or_explicit_legacy(self):
        current = MODULE.effective_config_identity(
            [{"effective_config_sha256": "a" * 64} for _ in range(2)]
        )
        self.assertEqual(current["effective_config_sha256"], "a" * 64)
        self.assertEqual(current["status"], "verified_unique")

        legacy = MODULE.effective_config_identity([{}, {}])
        self.assertIsNone(legacy["effective_config_sha256"])
        self.assertEqual(legacy["status"], "legacy_all_rows_missing")
        self.assertTrue(legacy["legacy_compatibility"])

        with self.assertRaisesRegex(ValueError, "部分缺失"):
            MODULE.effective_config_identity(
                [{"effective_config_sha256": "a" * 64}, {}]
            )
        with self.assertRaisesRegex(ValueError, "多个"):
            MODULE.effective_config_identity(
                [
                    {"effective_config_sha256": "a" * 64},
                    {"effective_config_sha256": "b" * 64},
                ]
            )

    def test_config_hash_propagates_to_report_and_verified_memory(self):
        queue_hash = "q" * 64
        config_hash = "a" * 64
        queue = [
            {
                "schema_version": "vsight_agentic_drift_natural_review_input_v1",
                "annotation_id": "annotation-a",
                "data_role": "development_agentic_natural_audit_not_training",
                "image_group_id": "group-a",
                "image_filename": "image.jpg",
                "query": "the dog",
                "query_stratum": "object",
                "relation_family": None,
                "reference_phrase": None,
                "stage_b_applicable_atoms": ["identity"],
                "original_bbox_xyxy": [0, 0, 10, 10],
            }
        ]
        sidecar = [
            {"annotation_id": "annotation-a", "source_record_id": "record-a"}
        ]
        reviews = [
            {
                "annotation_id": "annotation-a",
                "reviewer_id": MODULE.REVIEWER,
                "status": MODULE.COMPLETED,
                "source_queue_sha256": queue_hash,
                "evidence_state": "SUPPORTED_CORRECT",
                "parser_status": "correct",
                "confidence": 0.99,
            }
        ]
        audit = [
            {
                "record_id": "record-a",
                "detector_evidence_record_id": "detector-a",
                "effective_config_sha256": config_hash,
                "evidence_state": "SUPPORTED_CORRECT",
                "action": "ACCEPT",
                "reason_codes": [],
                "drift_risk_raw": 0.1,
                "review_priority": 0.1,
            }
        ]
        evidence = {
            "detector-a": {
                "detector_evidence": {
                    "proposals": [
                        {
                            "proposal_id": "target-a",
                            "is_reference": False,
                            "score": 0.9,
                            "bbox_xyxy": [0, 0, 10, 10],
                        }
                    ]
                }
            }
        }
        verified, report = MODULE.summarize(
            queue,
            sidecar,
            reviews,
            audit,
            evidence,
            queue_hash=queue_hash,
        )
        self.assertEqual(report["cohort_kind"], "natural")
        self.assertEqual(report["effective_config_sha256"], config_hash)
        self.assertEqual(
            report["effective_config_validation"]["status"], "verified_unique"
        )
        self.assertEqual(verified[0]["effective_config_sha256"], config_hash)
        self.assertEqual(
            verified[0]["effective_config_sha256_status"], "verified_unique"
        )
        markdown = MODULE.markdown_report(report)
        self.assertIn("correctness-blind 自然队列", markdown)
        self.assertNotIn("本队列按 agent 风险和控制带富集", markdown)

    def test_ranking_metrics_are_dependency_free(self):
        self.assertEqual(MODULE.auroc([0.1, 0.9], [False, True]), 1.0)
        self.assertEqual(
            MODULE.average_precision([0.1, 0.9], [False, True]), 1.0
        )

    def test_current_replay_precedes_selection_time_hypothesis(self):
        audit = {
            "evidence_state": "UNOBSERVABLE_AMBIGUOUS",
            "action": "ABSTAIN",
            "reason_codes": [],
        }
        sidecar = {
            "agent_evidence_state": "WRONG_INSTANCE",
            "agent_action": "RELOCALIZE",
            "reason_codes": ["STALE"],
        }
        self.assertEqual(
            MODULE.current_agent_field(
                audit, sidecar, "evidence_state", "agent_evidence_state"
            ),
            "UNOBSERVABLE_AMBIGUOUS",
        )
        self.assertEqual(
            MODULE.current_agent_field(
                audit, sidecar, "reason_codes", "reason_codes"
            ),
            [],
        )

    def test_latest_review_is_reviewer_specific_and_last_write_wins(self):
        latest = MODULE.latest_reviews(
            [
                {"annotation_id": "a", "reviewer_id": "项目负责人", "status": "completed"},
                {"annotation_id": "a", "reviewer_id": "项目负责人", "status": "draft"},
                {"annotation_id": "a", "reviewer_id": "other", "status": "completed"},
            ]
        )
        self.assertEqual(latest[("a", "项目负责人")]["status"], "draft")
        self.assertEqual(latest[("a", "other")]["status"], "completed")

    def test_topk_coverage_uses_role_and_score_order(self):
        proposals = [
            {"proposal_id": "ref", "is_reference": True, "score": 0.99, "bbox_xyxy": [0, 0, 10, 10]},
            {"proposal_id": "target-low", "is_reference": False, "score": 0.1, "bbox_xyxy": [0, 0, 10, 10]},
            {"proposal_id": "target-high", "is_reference": False, "score": 0.9, "bbox_xyxy": [10, 10, 20, 20]},
        ]
        result = MODULE.topk_coverage([10, 10, 20, 20], proposals, is_reference=False, k=1)
        self.assertTrue(result["covered"])
        self.assertEqual(result["topk"][0]["proposal_id"], "target-high")
        self.assertFalse(MODULE.topk_coverage([10, 10, 20, 20], proposals, is_reference=True)["covered"])

    def test_policy_does_not_relocalize_without_human_corrected_box(self):
        action, reason = MODULE.derive_action(
            {"evidence_state": "WRONG_INSTANCE", "confidence": 0.99},
            target={"available": False, "covered": None},
            reference={"available": True, "covered": True},
            reference_needed=True,
        )
        self.assertEqual((action, reason), ("ABSTAIN", "NO_HUMAN_CORRECTED_TARGET"))

    def test_verified_memory_coverage_omits_iou(self):
        memory = MODULE.memory_coverage(
            {"available": True, "covered": True, "best_iou": 0.8, "topk": [{"proposal_id": "p", "iou": 0.8}]}
        )
        self.assertNotIn("iou", str(memory).casefold())
        self.assertEqual(memory["topk_proposal_ids"], ["p"])


if __name__ == "__main__":
    unittest.main()
