import unittest

from vsight.ccv import (
    CCVAction,
    CCVThresholds,
    ClaimConditionedCounterfactualVerifier,
    ConditionalRelocalizationRouter,
    DetectorEvidence,
    DetectorProposal,
    TypedClaimParser,
)


def proposal(proposal_id, box, score, atom_scores=None, full=None, reference=False):
    return DetectorProposal(
        proposal_id,
        tuple(box),
        score,
        atom_scores or {},
        full,
        reference,
    )


def evidence(proposals, prompted, complete=True, contradiction=0.0):
    return DetectorEvidence(
        tuple(proposals),
        100,
        100,
        complete,
        frozenset(prompted),
        contradiction_support=contradiction,
    )


class TypedClaimParserTest(unittest.TestCase):
    def test_parses_typed_atoms_without_supervision(self):
        claim = TypedClaimParser().parse("the red man holding a cup left of the car")
        ids = {atom.atom_id: atom for atom in claim.atoms}
        self.assertTrue(ids["object"].known)
        self.assertEqual(ids["object"].text, "person")
        self.assertTrue(ids["attribute:red"].known)
        self.assertTrue(ids["action:holding"].known)
        self.assertEqual(ids["relation:to_the_left_of"].reference_text, "car")

    def test_unknown_modifier_is_never_known_negative_evidence(self):
        claim = TypedClaimParser().parse("the translucent cup")
        unknown = [atom for atom in claim.atoms if not atom.known]
        self.assertEqual([atom.atom_id for atom in unknown], ["modifier:unknown"])
        self.assertNotIn("modifier:unknown", {atom.atom_id for atom in claim.known_atoms})

    def test_open_vocabulary_relation_target_is_not_replaced_by_reference(self):
        claim = TypedClaimParser().parse("the helmet next to the red car")
        ids = {atom.atom_id: atom for atom in claim.atoms}
        self.assertEqual(ids["object"].text, "helmet")
        self.assertEqual(ids["relation:next_to"].reference_text, "red car")
        self.assertNotIn("attribute:red", ids)


class CCVPolicyTest(unittest.TestCase):
    def setUp(self):
        self.verifier = ClaimConditionedCounterfactualVerifier(
            CCVThresholds(conjunction="minimum")
        )

    def test_accepts_claim_when_all_known_atoms_bind_to_upstream(self):
        rows = [
            proposal("object", [10, 10, 30, 30], 0.9, {"object": 0.9}),
            proposal("attribute", [10, 10, 30, 30], 0.8, {"attribute:red": 0.8}),
            proposal("full", [10, 10, 30, 30], 0.85, full=0.85),
        ]
        result = self.verifier.verify(
            None,
            "the red cup",
            [10, 10, 30, 30],
            evidence(rows, {"object", "attribute:red"}),
        )
        self.assertEqual(result.action, CCVAction.ACCEPT)
        self.assertGreaterEqual(result.claim_support, 0.8)
        self.assertIsNone(result.corrected_bbox)

    def test_successful_empty_evidence_rejects_but_failed_evidence_preserves(self):
        complete = self.verifier.verify(
            None,
            "the cup",
            [10, 10, 30, 30],
            evidence([], {"object"}, complete=True),
        )
        self.assertEqual(complete.action, CCVAction.REJECT)

        incomplete = self.verifier.verify(
            None,
            "the cup",
            [10, 10, 30, 30],
            evidence([], set(), complete=False),
        )
        self.assertEqual(incomplete.action, CCVAction.ACCEPT)
        self.assertFalse(incomplete.evidence_sufficient)
        self.assertEqual(incomplete.calibrated_risk, 0.5)

    def test_relocalizes_only_to_stronger_complete_claim(self):
        alternative = [50, 10, 70, 30]
        rows = [
            proposal("object", alternative, 0.9, {"object": 0.9}),
            proposal("attribute", alternative, 0.8, {"attribute:red": 0.8}),
            proposal("full", alternative, 0.85, full=0.85),
        ]
        result = self.verifier.verify(
            None,
            "the red cup",
            [10, 10, 30, 30],
            evidence(rows, {"object", "attribute:red"}),
        )
        self.assertEqual(result.action, CCVAction.RELOCALIZE)
        self.assertEqual(result.corrected_bbox, tuple(alternative))
        self.assertGreaterEqual(result.binding_margin, 0.12)

    def test_full_expression_support_prevents_false_absence_reject(self):
        rows = [
            proposal("object", [50, 10, 70, 30], 0.7, {"object": 0.7}),
            proposal("attribute", [10, 10, 30, 30], 0.7, {"attribute:red": 0.7}),
            proposal("full", [50, 10, 70, 30], 0.5, full=0.5),
        ]
        result = self.verifier.verify(
            None,
            "the red cup",
            [10, 10, 30, 30],
            evidence(rows, {"object", "attribute:red"}),
        )
        self.assertEqual(result.action, CCVAction.ACCEPT)
        self.assertEqual(result.reason, "binding_uncertain_preserve_upstream")
        self.assertGreater(result.existence_margin, 0.0)

    def test_relation_support_uses_reference_geometry(self):
        rows = [
            proposal("object", [10, 10, 30, 30], 0.9, {"object": 0.9}),
            proposal("full", [10, 10, 30, 30], 0.8, full=0.8),
            proposal("reference", [60, 10, 80, 30], 0.9, reference=True),
        ]
        result = self.verifier.verify(
            None,
            "the person left of the car",
            [10, 10, 30, 30],
            evidence(rows, {"object", "relation:to_the_left_of"}),
        )
        claim = result.atom_evidence["claim"]
        self.assertGreaterEqual(claim["atom_support"]["relation:to_the_left_of"], 0.8)
        self.assertEqual(result.action, CCVAction.ACCEPT)

    def test_object_presence_does_not_become_relation_absence(self):
        rows = [
            proposal("object", [10, 10, 30, 30], 0.8, {"object": 0.8}),
            proposal("reference", [70, 70, 90, 90], 0.9, reference=True),
        ]
        result = self.verifier.verify(
            None,
            "the umbrella held by the person",
            [10, 10, 30, 30],
            evidence(rows, {"object", "relation:held_by"}),
        )
        self.assertEqual(result.action, CCVAction.ACCEPT)
        self.assertEqual(result.reason, "binding_uncertain_preserve_upstream")
        self.assertGreater(result.existence_margin, 0.0)

    def test_router_retains_box_when_absolute_support_is_unsafe(self):
        verifier = ClaimConditionedCounterfactualVerifier(
            CCVThresholds(
                conjunction="minimum",
                alternative_support=0.2,
                binding_margin=0.1,
            )
        )
        rows = [
            proposal("object", [50, 10, 70, 30], 0.25, {"object": 0.25}),
            proposal("full", [50, 10, 70, 30], 0.25, full=0.25),
        ]
        flagged = verifier.verify(
            None,
            "the cup",
            [10, 10, 30, 30],
            evidence(rows, {"object"}),
        )
        self.assertEqual(flagged.action, CCVAction.RELOCALIZE)
        routed = ConditionalRelocalizationRouter(0.3).route(flagged)
        self.assertEqual(routed.action, CCVAction.RELOCALIZE)
        self.assertIsNone(routed.corrected_bbox)

    def test_white_box_attention_is_candidate_and_atom_localized(self):
        verifier = ClaimConditionedCounterfactualVerifier(
            CCVThresholds(
                conjunction="minimum",
                attention_weight=1.0,
                alternative_support=0.3,
                binding_margin=0.12,
            )
        )
        rows = [
            proposal("upstream-proposal", [10, 10, 30, 30], 0.5, {"object": 0.5}, full=0.5),
            proposal("alternative", [50, 10, 70, 30], 0.5, {"object": 0.5}, full=0.5),
        ]
        result = verifier.verify(
            None,
            "the cup",
            [10, 10, 30, 30],
            evidence(rows, {"object"}),
            optional_localized_attention={
                "upstream": {"object": 0.1},
                "alternative": {"object": 0.9},
            },
        )
        self.assertEqual(result.action, CCVAction.RELOCALIZE)
        self.assertEqual(result.alternative_bbox, (50.0, 10.0, 70.0, 30.0))


if __name__ == "__main__":
    unittest.main()
