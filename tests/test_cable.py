import importlib.util
import unittest
from pathlib import Path

from vsight.ccv import (
    CCVAction,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
    DetectorEvidence,
    DetectorProposal,
    PromptProvenance,
    TypedClaimParser,
)
from vsight.ccv_learning import GapBalancedRiskController
from vsight.cable_crop import ConditionalCropExtension
from vsight.composite_detector import build_composite_prompt


class CABLEPromptTest(unittest.TestCase):
    def test_compiles_only_safe_hard_counterfactuals(self):
        binary = build_composite_prompt(TypedClaimParser().parse("the open door"))
        inverse = [row for row in binary.segments if row.provenance is PromptProvenance.INVERSE]
        self.assertEqual(len(inverse), 1)
        self.assertIn("closed", inverse[0].text)
        self.assertEqual(inverse[0].counterfactual_of, "attribute:open")

        color = build_composite_prompt(TypedClaimParser().parse("the red cup"))
        self.assertFalse(
            any(row.provenance is PromptProvenance.INVERSE for row in color.segments)
        )

    def test_relation_inverse_retains_atom_provenance(self):
        prompt = build_composite_prompt(
            TypedClaimParser().parse("the person left of the car")
        )
        inverse = next(
            row for row in prompt.segments
            if row.counterfactual_of == "relation:to_the_left_of"
        )
        self.assertEqual(inverse.provenance, PromptProvenance.INVERSE)
        self.assertEqual(inverse.relation, "to_the_right_of")
        self.assertIn("right of", inverse.text)


class CABLELedgerPolicyTest(unittest.TestCase):
    def verifier(self):
        return ClaimConditionedCounterfactualVerifier(
            CCVThresholds(conjunction="minimum")
        )

    def test_localized_inverse_is_explicit_relation_rejection_witness(self):
        proposals = (
            DetectorProposal("object", (60, 10, 80, 30), 0.9, {"object": 0.9}),
            DetectorProposal("full", (60, 10, 80, 30), 0.8, full_score=0.8),
            DetectorProposal("reference", (10, 10, 30, 30), 0.9, is_reference=True),
            DetectorProposal(
                "inverse", (60, 10, 80, 30), 0.9,
                prompt_provenance="inverse",
                counterfactual_of="relation:to_the_left_of",
                relation="to_the_right_of",
            ),
        )
        evidence = DetectorEvidence(
            proposals, 100, 100, True,
            frozenset({"object", "relation:to_the_left_of"}),
            counterfactual_atom_ids=frozenset({"relation:to_the_left_of"}),
        )
        result = self.verifier().verify(
            None, "the person left of the car", (60, 10, 80, 30), evidence
        )
        self.assertEqual(result.action, CCVAction.REJECT)
        self.assertEqual(result.reason, "typed_binding_counterfactual_contradiction")
        ledger = result.atom_evidence["ledger"]
        self.assertEqual(ledger["witness_kind"], "inverse_relation")
        self.assertTrue(ledger["explicit_witness"])

    def test_between_without_two_references_is_unknown_not_negative(self):
        proposals = (
            DetectorProposal("object", (40, 10, 60, 30), 0.9, {"object": 0.9}),
            DetectorProposal("full", (40, 10, 60, 30), 0.8, full_score=0.8),
            DetectorProposal("reference", (10, 10, 30, 30), 0.9, is_reference=True),
        )
        evidence = DetectorEvidence(
            proposals, 100, 100, True,
            frozenset({"object", "relation:between"}),
        )
        result = self.verifier().verify(
            None, "the person between the cars", (40, 10, 60, 30), evidence
        )
        self.assertEqual(result.action, CCVAction.ACCEPT)
        self.assertEqual(result.binding_status, "BINDING_UNCERTAIN")
        edge = result.atom_evidence["ledger"]["relation_edges"][0]
        self.assertFalse(edge["interpretable"])
        self.assertIn("two_distinguishable", edge["unknown_reason"])

    def test_nonbinary_attribute_uses_candidate_swap_for_restoration(self):
        verifier = ClaimConditionedCounterfactualVerifier(
            CCVThresholds(
                conjunction="minimum", require_typed_relocalization=True
            )
        )
        proposals = (
            DetectorProposal("object-up", (10, 10, 30, 30), 0.9, {"object": 0.9}),
            DetectorProposal("attr-up", (10, 10, 30, 30), 0.1, {"object": 0.1, "attribute:red": 0.1}),
            DetectorProposal("full-up", (10, 10, 30, 30), 0.1, full_score=0.1),
            DetectorProposal("object-alt", (50, 10, 70, 30), 0.9, {"object": 0.9}),
            DetectorProposal("attr-alt", (50, 10, 70, 30), 0.9, {"object": 0.9, "attribute:red": 0.9}),
            DetectorProposal("full-alt", (50, 10, 70, 30), 0.9, full_score=0.9),
        )
        evidence = DetectorEvidence(
            proposals, 100, 100, True, frozenset({"object", "attribute:red"})
        )
        result = verifier.verify(None, "the red cup", (10, 10, 30, 30), evidence)
        self.assertEqual(result.action, CCVAction.RELOCALIZE)
        self.assertEqual(result.corrected_bbox, (50.0, 10.0, 70.0, 30.0))
        self.assertEqual(
            result.atom_evidence["ledger"]["witness_kind"],
            "attribute_candidate_swap",
        )


class GapControllerTest(unittest.TestCase):
    def test_lookup_excludes_model_and_requires_witness_for_roh(self):
        controller = GapBalancedRiskController({"t2|relation|1": 0.6})
        common = dict(
            task="t2", atom_type="relation", calibration_fold=1,
            absence_risk=0.0, contradiction_risk=0.9,
            restoration_probability=0.1, edge_ambiguity=0.1,
            has_complete_alternative=False,
        )
        self.assertEqual(
            controller.decide(**common, explicit_witness=False), CCVAction.ACCEPT
        )
        self.assertEqual(
            controller.decide(**common, explicit_witness=True), CCVAction.REJECT
        )
        artifact = controller.to_dict()
        self.assertNotIn("model", artifact["lookup_features"])
        self.assertNotIn("hallucination_type", artifact["lookup_features"])


class ConditionalCropTest(unittest.TestCase):
    def test_gray_zone_adds_at_most_one_target_crop_without_reference(self):
        class Image:
            size = (100, 100)

            def crop(self, box):
                result = Image()
                result.size = (box[2] - box[0], box[3] - box[1])
                return result

        class Detector:
            calls = 0

            def infer(self, image, query):
                self.calls += 1
                evidence = DetectorEvidence(
                    (
                        DetectorProposal("object", (5, 5, 25, 25), 0.9, {"object": 0.9}),
                        DetectorProposal("full", (5, 5, 25, 25), 0.8, full_score=0.8),
                    ),
                    int(image.size[0]), int(image.size[1]), True, frozenset({"object"}),
                )
                return TypedClaimParser().parse(query), evidence

        verifier = ClaimConditionedCounterfactualVerifier()
        initial_evidence = DetectorEvidence((), 100, 100, False, frozenset())
        initial = verifier.verify(
            None, "the cup", (10, 10, 30, 30), initial_evidence
        )
        self.assertEqual(initial.binding_status, "BINDING_UNCERTAIN")
        detector = Detector()
        result = ConditionalCropExtension().verify(
            image=Image(), query="the cup", upstream_bbox=(10, 10, 30, 30),
            initial_result=initial, initial_evidence=initial_evidence,
            detector=detector, verifier=verifier,
        )
        self.assertEqual(detector.calls, 1)
        self.assertEqual(result.latency_metadata["image_encoder_forwards"], 2)
        self.assertFalse(result.reason.startswith("conditional_crops_remain_uncertain"))

    def test_reference_union_is_second_and_final_crop(self):
        class Image:
            size = (100, 100)

            def crop(self, box):
                result = Image()
                result.size = (box[2] - box[0], box[3] - box[1])
                return result

        class Detector:
            calls = 0

            def infer(self, image, query):
                self.calls += 1
                evidence = DetectorEvidence(
                    (), int(image.size[0]), int(image.size[1]), False,
                    frozenset(), latency_ms=5.0,
                )
                return TypedClaimParser().parse(query), evidence

        verifier = ClaimConditionedCounterfactualVerifier()
        initial_evidence = DetectorEvidence(
            (
                DetectorProposal(
                    "reference", (60, 10, 90, 40), 0.9, {}, is_reference=True
                ),
            ),
            100, 100, False, frozenset(), latency_ms=7.0,
        )
        initial = verifier.verify(
            None, "the cup next to the table", (10, 10, 30, 30), initial_evidence
        )
        detector = Detector()
        result = ConditionalCropExtension().verify(
            image=Image(), query="the cup next to the table",
            upstream_bbox=(10, 10, 30, 30), initial_result=initial,
            initial_evidence=initial_evidence, detector=detector, verifier=verifier,
        )
        self.assertEqual(detector.calls, 2)
        self.assertEqual(result.latency_metadata["image_encoder_forwards"], 3)
        self.assertEqual(result.latency_metadata["conditional_crop_pass"], 2)
        self.assertEqual(
            result.latency_metadata["conditional_crop_additional_detector_ms"], 10.0
        )
        self.assertEqual(result.latency_metadata["conditional_crop_pass_1_ms"], 5.0)
        self.assertEqual(result.latency_metadata["conditional_crop_pass_2_ms"], 5.0)
        self.assertEqual(result.reason, "conditional_crops_remain_uncertain_preserve_upstream")

    def test_extreme_aspect_crop_is_expanded_before_detector(self):
        class Image:
            size = (100, 100)

            def crop(self, box):
                result = Image()
                result.size = (box[2] - box[0], box[3] - box[1])
                return result

        class Detector:
            crop_size = None

            def infer(self, image, query):
                self.crop_size = image.size
                evidence = DetectorEvidence(
                    (
                        DetectorProposal("object", (1, 1, 10, 20), 0.9, {"object": 0.9}),
                        DetectorProposal("full", (1, 1, 10, 20), 0.8, full_score=0.8),
                    ),
                    int(image.size[0]), int(image.size[1]), True, frozenset({"object"}),
                )
                return TypedClaimParser().parse(query), evidence

        verifier = ClaimConditionedCounterfactualVerifier()
        initial_evidence = DetectorEvidence((), 100, 100, False, frozenset())
        initial = verifier.verify(None, "the cup", (10, 10, 12, 90), initial_evidence)
        detector = Detector()
        ConditionalCropExtension().verify(
            image=Image(), query="the cup", upstream_bbox=(10, 10, 12, 90),
            initial_result=initial, initial_evidence=initial_evidence,
            detector=detector, verifier=verifier,
        )
        width, height = detector.crop_size
        self.assertGreaterEqual(width / height, 0.75)
        self.assertLessEqual(width / height, 4 / 3)


SPEC = importlib.util.spec_from_file_location(
    "review_e3_binding",
    Path(__file__).parents[1] / "scripts/review_e3_binding.py",
)
REVIEW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REVIEW)


class StageBV2ValidationTest(unittest.TestCase):
    def base(self):
        known = {
            "a": {
                "annotation_id": "a",
                "image_filename": "image.jpg",
                "stage_b_applicable_atoms": ["identity", "relation"],
            }
        }
        payload = {
            "annotation_id": "a",
            "reviewer_id": "project_owner",
            "status": "completed",
            "stage_a_target_status": "supported",
            "stage_b_atoms": {"identity": "supported", "relation": "supported"},
            "stage_c_action": "ACCEPT",
            "reference_bbox_xyxy": [50, 10, 80, 40],
            "reference_visibility": "visible",
            "confidence": 0.95,
        }
        return known, payload

    def test_observable_relation_requires_independent_reference_box(self):
        known, payload = self.base()
        row = REVIEW.validate_submission(payload, known, {"image.jpg": (100, 100)})
        self.assertEqual(row["schema_version"], "vsight_e3_binding_review_v2")
        self.assertEqual(row["annotation_protocol"], "single_project_owner")
        self.assertEqual(row["stage_b_atoms"]["relation"], "supported")

        payload["reference_bbox_xyxy"] = None
        with self.assertRaisesRegex(ValueError, "reference bbox"):
            REVIEW.validate_submission(payload, known, {"image.jpg": (100, 100)})


if __name__ == "__main__":
    unittest.main()
