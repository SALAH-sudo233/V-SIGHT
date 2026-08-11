import unittest

from vsight.binding_taxonomy import (
    BindingAction,
    EvidenceState,
    OracleAttribution,
    OracleReplacementEvidence,
    action_for_evidence,
    attribute_oracle_error,
    legacy_ccv_action,
)


class BindingTaxonomyTest(unittest.TestCase):
    def test_evidence_and_action_are_independent(self) -> None:
        self.assertIs(
            action_for_evidence(EvidenceState.SUPPORTED_CORRECT),
            BindingAction.ACCEPT,
        )
        self.assertIs(
            action_for_evidence(EvidenceState.WRONG_INSTANCE),
            BindingAction.ABSTAIN,
        )
        self.assertIs(
            action_for_evidence(
                EvidenceState.WRONG_INSTANCE, safe_topk_replacement=True
            ),
            BindingAction.RELOCALIZE,
        )
        self.assertIs(
            action_for_evidence(EvidenceState.UNOBSERVABLE_AMBIGUOUS),
            BindingAction.ABSTAIN,
        )

    def test_supported_row_rejects_unneeded_replacement(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not need"):
            action_for_evidence(
                EvidenceState.SUPPORTED_CORRECT, safe_topk_replacement=True
            )

    def test_oracle_attribution_precedence(self) -> None:
        cases = [
            (
                OracleReplacementEvidence(False, False, False),
                OracleAttribution.PARSE_ERR,
            ),
            (
                OracleReplacementEvidence(True, False, False),
                OracleAttribution.SEE_ERR,
            ),
            (
                OracleReplacementEvidence(
                    True,
                    True,
                    True,
                    reference_topk_covered=True,
                    target_replacement_restores=True,
                ),
                OracleAttribution.TARGET_ERR,
            ),
            (
                OracleReplacementEvidence(
                    True,
                    True,
                    True,
                    reference_topk_covered=True,
                    target_replacement_restores=False,
                    reference_replacement_restores=True,
                ),
                OracleAttribution.REF_ERR,
            ),
            (
                OracleReplacementEvidence(
                    True,
                    True,
                    True,
                    reference_topk_covered=True,
                    target_replacement_restores=False,
                    reference_replacement_restores=False,
                    relation_edge_supported=False,
                ),
                OracleAttribution.REL_ERR,
            ),
        ]
        for evidence, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(attribute_oracle_error(evidence), expected)

    def test_complete_and_incomplete_attribution(self) -> None:
        complete = OracleReplacementEvidence(
            True,
            True,
            False,
            target_replacement_restores=False,
        )
        incomplete = OracleReplacementEvidence(None, True, False)
        relation_complete = OracleReplacementEvidence(
            True,
            True,
            True,
            reference_topk_covered=True,
            target_replacement_restores=False,
            reference_replacement_restores=False,
            relation_edge_supported=True,
        )
        self.assertIs(attribute_oracle_error(complete), OracleAttribution.NONE)
        self.assertIs(
            attribute_oracle_error(relation_complete), OracleAttribution.NONE
        )
        self.assertIs(
            attribute_oracle_error(incomplete), OracleAttribution.UNRESOLVED
        )

    def test_invalid_oracle_combinations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirmed Top-K"):
            OracleReplacementEvidence(
                True,
                False,
                False,
                target_replacement_restores=True,
            )
        with self.assertRaisesRegex(ValueError, "reference-free"):
            OracleReplacementEvidence(
                True,
                True,
                False,
                reference_replacement_restores=True,
            )

    def test_legacy_abstain_mapping(self) -> None:
        self.assertEqual(legacy_ccv_action(BindingAction.ABSTAIN), "REJECT")
        self.assertEqual(legacy_ccv_action(BindingAction.ACCEPT), "ACCEPT")


if __name__ == "__main__":
    unittest.main()
