import json
import tempfile
import unittest
from pathlib import Path

from vsight.agentic_drift_loop import AgenticDriftConfig, AgenticDriftLoop
from vsight.ccv import DetectorEvidence, DetectorProposal, PromptProvenance
from vsight.drift_memory import DriftMemory


def evidence(*proposals, prompted=("object", "attribute:red")):
    return DetectorEvidence(
        proposals=tuple(proposals),
        image_width=100,
        image_height=100,
        evidence_complete=True,
        prompted_atom_ids=frozenset(prompted),
    )


class AgenticDriftLoopTest(unittest.TestCase):
    def test_supported_case_accepts_only_with_strong_evidence(self):
        box = (10, 10, 40, 40)
        detector = evidence(
            DetectorProposal(
                "object:0", box, 0.95,
                atom_scores={"object": 0.95, "attribute:red": 0.95},
                prompt_provenance=PromptProvenance.CLAIM,
            )
        )
        result = AgenticDriftLoop().audit(
            record_id="supported", query="the red car", upstream_bbox=box,
            detector_evidence=detector,
        )
        self.assertEqual(result.evidence_state.value, "SUPPORTED_CORRECT")
        self.assertEqual(result.action.value, "ACCEPT")
        self.assertEqual(result.legacy_action, "ACCEPT")
        self.assertLess(result.drift_risk, 0.35)
        self.assertEqual(result.rounds_executed, 3)
        self.assertEqual(result.stop_reason, "CONFIDENT_ACCEPT")
        self.assertEqual(
            result.budget_ledger["candidate_caps_evaluated"], [1, 3, 5]
        )
        self.assertEqual(
            result.budget_ledger["additional_detector_image_encoder_forwards"],
            0,
        )

    def test_object_only_confident_case_stops_after_static_round(self):
        box = (10, 10, 40, 40)
        detector = evidence(
            DetectorProposal(
                "object:0", box, 0.95,
                atom_scores={"object": 0.95},
                prompt_provenance=PromptProvenance.CLAIM,
            ),
            prompted=("object",),
        )
        result = AgenticDriftLoop().audit(
            record_id="object-supported", query="the car", upstream_bbox=box,
            detector_evidence=detector,
        )
        self.assertEqual(result.action.value, "ACCEPT")
        self.assertEqual(result.rounds_executed, 1)
        self.assertEqual(result.stop_reason, "CONFIDENT_ACCEPT")
        self.assertEqual(
            result.budget_ledger["candidate_caps_evaluated"], [1]
        )

    def test_typed_candidate_swap_relocalizes_in_round_two(self):
        original = (10, 10, 30, 30)
        alternative = (50, 10, 70, 30)
        detector = evidence(
            DetectorProposal(
                "object-up", original, 0.9,
                {"object": 0.9}, prompt_provenance="claim",
            ),
            DetectorProposal(
                "attr-up", original, 0.1,
                {"object": 0.1, "attribute:red": 0.1},
                prompt_provenance="claim",
            ),
            DetectorProposal(
                "full-up", original, 0.1,
                full_score=0.1, prompt_provenance="claim",
            ),
            DetectorProposal(
                "object-alt", alternative, 0.9,
                {"object": 0.9}, prompt_provenance="claim",
            ),
            DetectorProposal(
                "attr-alt", alternative, 0.9,
                {"object": 0.9, "attribute:red": 0.9},
                prompt_provenance="claim",
            ),
            DetectorProposal(
                "full-alt", alternative, 0.9,
                full_score=0.9, prompt_provenance="claim",
            ),
        )
        result = AgenticDriftLoop().audit(
            record_id="swap", query="the red cup", upstream_bbox=original,
            detector_evidence=detector,
        )
        self.assertEqual(result.evidence_state.value, "WRONG_INSTANCE")
        self.assertEqual(result.action.value, "RELOCALIZE")
        self.assertEqual(result.rounds_executed, 2)
        self.assertEqual(result.stop_reason, "SAFE_TYPED_RELOCALIZATION")
        self.assertEqual(result.alternative_bbox, alternative)

    def test_score_margin_without_typed_witness_requires_review(self):
        original = (10, 10, 30, 30)
        detector = evidence(
            DetectorProposal(
                "weak", original, 0.1, {"object": 0.1},
                full_score=0.1, prompt_provenance="claim",
            ),
            DetectorProposal(
                "strong", (50, 10, 70, 30), 0.95, {"object": 0.95},
                full_score=0.95, prompt_provenance="claim",
            ),
            prompted=("object",),
        )
        result = AgenticDriftLoop().audit(
            record_id="margin-only", query="the cup", upstream_bbox=original,
            detector_evidence=detector,
        )
        self.assertNotEqual(result.evidence_state.value, "WRONG_INSTANCE")
        self.assertEqual(result.action.value, "ABSTAIN")
        self.assertEqual(result.audit_disposition, "REVIEW_REQUIRED")
        self.assertIn("REVIEW_REQUIRED_UNCERTAIN", result.reason_codes)
        self.assertNotIn("PRESERVE_UPSTREAM_UNCERTAIN", result.reason_codes)

    def test_max_rounds_controls_weak_evidence_escalation(self):
        box = (10, 10, 40, 40)
        detector = evidence(
            DetectorProposal(
                "object:0", box, 0.30,
                atom_scores={"object": 0.30, "attribute:red": 0.30},
                prompt_provenance=PromptProvenance.CLAIM,
            )
        )
        result = AgenticDriftLoop(AgenticDriftConfig(max_rounds=2)).audit(
            record_id="weak", query="the red car", upstream_bbox=box,
            detector_evidence=detector,
        )
        self.assertEqual(result.rounds_executed, 2)
        self.assertEqual(result.stop_reason, "MAX_ROUNDS_REACHED")
        self.assertEqual(
            result.budget_ledger["candidate_caps_evaluated"], [1, 3]
        )

    def test_configured_binder_thresholds_replace_hardcoded_values(self):
        class Result:
            atom_evidence = {
                "ledger": {
                    "explicit_witness": True,
                    "M_contra": 0.2,
                    "witness_support": 0.3,
                    "U_edge": 0.2,
                }
            }
            claim_support = 0.1
            alternative_support = 0.5
            binding_margin = 0.2
            alternative_bbox = (50, 10, 70, 30)

        strict = AgenticDriftLoop(
            AgenticDriftConfig(
                alternative_support=0.8,
                binding_margin=0.7,
                counterfactual_margin=0.6,
                witness_support=0.5,
                edge_uncertainty_max=0.1,
            )
        )._bind(Result(), round_number=3)
        self.assertNotIn("ALTERNATIVE_DOMINATES", strict.findings)
        self.assertNotIn("TYPED_COUNTERFACTUAL_WITNESS", strict.findings)
        self.assertIn("EDGE_UNCERTAIN", strict.findings)

    def test_missing_upstream_fails_closed_without_relocalization(self):
        detector = evidence(
            DetectorProposal(
                "object:0", (10, 10, 40, 40), 0.95,
                atom_scores={"object": 0.95, "attribute:red": 0.95},
                prompt_provenance=PromptProvenance.CLAIM,
            )
        )
        result = AgenticDriftLoop().audit(
            record_id="missing", query="the red car", upstream_bbox=None,
            detector_evidence=detector,
        )
        self.assertEqual(result.evidence_state.value, "UNOBSERVABLE_AMBIGUOUS")
        self.assertEqual(result.action.value, "ABSTAIN")
        self.assertEqual(result.legacy_action, "REJECT")

    def test_memory_rejects_supervision_fields_and_retrieves_reason_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = DriftMemory(Path(directory) / "memory.jsonl")
            row = memory.append({
                "record_id": "a", "reason_codes": ["ALTERNATIVE_DOMINATES"],
                "review_priority": 0.8, "memory_status": "REVIEW_REQUIRED",
            })
            self.assertTrue(row["episode_hash"])
            self.assertEqual(len(memory.retrieve(reason_codes=["ALTERNATIVE_DOMINATES"])), 1)
            with self.assertRaisesRegex(ValueError, "forbidden supervision"):
                memory.append({"record_id": "b", "original_iou": 0.4})
            self.assertEqual(len(json.loads(json.dumps(memory.rows))), 1)

    def test_verified_feedback_only_boosts_review_priority(self):
        box = (10, 10, 40, 40)
        detector = evidence(
            DetectorProposal(
                "object:0", box, 0.95,
                atom_scores={"object": 0.95, "attribute:red": 0.95},
                prompt_provenance=PromptProvenance.CLAIM,
            )
        )
        baseline = AgenticDriftLoop().audit(
            record_id="feedback-baseline", query="the red car", upstream_bbox=box,
            detector_evidence=detector,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            memory = DriftMemory(path)
            memory.append_verified({
                "annotation_id": "review-1",
                "query": "the red car",
                "evidence_state": "WRONG_INSTANCE",
                "derived_action": "ABSTAIN",
                "confidence": 0.95,
                "agent_reason_codes": list(baseline.reason_codes),
            })
            replay = AgenticDriftLoop(memory=DriftMemory(path, read_only=True)).audit(
                record_id="feedback-replay", query="the red car", upstream_bbox=box,
                detector_evidence=detector,
            )
            self.assertEqual(replay.action, baseline.action)
            self.assertEqual(replay.evidence_state, baseline.evidence_state)
            self.assertGreaterEqual(replay.feedback_match_count, 1)
            self.assertTrue(replay.feedback_conflict)
            self.assertGreaterEqual(replay.review_priority, baseline.review_priority)

    def test_read_only_memory_cannot_be_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            path.write_text("", encoding="utf-8")
            memory = DriftMemory(path, read_only=True)
            with self.assertRaisesRegex(ValueError, "read-only"):
                memory.append({"record_id": "blocked", "reason_codes": []})


if __name__ == "__main__":
    unittest.main()
