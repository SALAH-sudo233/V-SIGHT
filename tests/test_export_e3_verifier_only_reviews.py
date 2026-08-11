import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "export_e3_verifier_only_reviews",
    Path(__file__).parents[1] / "scripts/export_e3_verifier_only_reviews.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def queue_row(annotation_id="a1"):
    return {
        "annotation_id": annotation_id,
        "data_split": "train",
        "group_id": f"group-{annotation_id}",
        "image_id": 1,
        "image_filename": "image.jpg",
        "query": "the red cup",
        "original_bbox_xyxy": [1, 2, 30, 40],
        "query_stratum": "attribute",
    }


def review(
    reviewer="project_owner",
    action="ACCEPT",
    stage_a="supported",
    confidence=0.95,
    queue_hash="hash",
):
    return {
        "annotation_id": "a1",
        "reviewer_id": reviewer,
        "status": "completed",
        "stage_a_target_status": stage_a,
        "stage_b_atoms": ["identity", "attribute"],
        "stage_c_action": action,
        "corrected_bbox_xyxy": [5, 6, 35, 45] if action == "RELOCALIZE" else None,
        "confidence": confidence,
        "source_queue_sha256": queue_hash,
    }


class ExportVerifierOnlyReviewsTest(unittest.TestCase):
    def test_v2_relation_requires_one_authoritative_reference_box(self):
        source = {
            **queue_row(),
            "schema_version": "vsight_e3_binding_annotation_input_v2",
            "stage_b_applicable_atoms": ["identity", "relation"],
            "relation_family": "directional",
        }
        reviews = [review()]
        reviews[0]["stage_b_atoms"] = {
            "identity": "supported", "relation": "supported"
        }
        reviews[0]["reference_bbox_xyxy"] = [50, 10, 80, 40]
        reviews[0]["reference_visibility"] = "visible"
        rows, audit = MODULE.adjudicate([source], reviews, "hash")
        self.assertEqual(audit["exported_accept"], 1)
        self.assertEqual(rows[0]["schema_version"], "vsight_e3_verifier_only_training_v2")
        self.assertTrue(rows[0]["schema_gate_passed"])
        self.assertEqual(len(rows[0]["independent_reference_boxes_xyxy"]), 1)
        self.assertEqual(rows[0]["annotation_protocol"], "single_project_owner")
        self.assertFalse(rows[0]["inter_reviewer_agreement_computed"])

    def test_exports_one_project_owner_accept(self):
        rows, audit = MODULE.adjudicate(
            [queue_row()], [review()], "hash"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verifier_action"], "ACCEPT")
        self.assertEqual(rows[0]["reviewer_ids"], ["project_owner"])
        self.assertEqual(rows[0]["reviewer_count"], 1)
        self.assertEqual(audit["exported_accept"], 1)

    def test_ignores_non_authoritative_review_and_preserves_relocalize(self):
        rows, audit = MODULE.adjudicate(
            [queue_row()],
            [review(), review("observer", "REJECT", "contradicted")],
            "hash",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verifier_action"], "ACCEPT")

        rows, audit = MODULE.adjudicate(
            [queue_row()],
            [review(action="RELOCALIZE")],
            "hash",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verifier_action"], "RELOCALIZE")
        self.assertEqual(len(rows[0]["router_only_corrected_boxes_xyxy"]), 1)
        self.assertEqual(audit["exported_relocalize"], 1)

    def test_requires_hash_confidence_and_action_stage_consistency(self):
        rows, audit = MODULE.adjudicate(
            [queue_row()],
            [review(queue_hash="wrong")],
            "hash",
        )
        self.assertEqual(rows, [])
        self.assertEqual(audit["insufficient_completed_reviewers"], 1)

        rows, audit = MODULE.adjudicate(
            [queue_row()],
            [review(action="REJECT", stage_a="supported")],
            "hash",
        )
        self.assertEqual(rows, [])
        self.assertEqual(audit["invalid_action_existence_pair"], 1)

    def test_excludes_uncertain_and_invalid_relocalize_box(self):
        rows, audit = MODULE.adjudicate(
            [queue_row()],
            [review(action="UNCERTAIN", stage_a="ambiguous")],
            "hash",
        )
        self.assertEqual(rows, [])
        self.assertEqual(audit["reserved_uncertain"], 1)

        invalid = [review(action="RELOCALIZE")]
        invalid[0]["corrected_bbox_xyxy"] = None
        rows, audit = MODULE.adjudicate([queue_row()], invalid, "hash")
        self.assertEqual(rows, [])
        self.assertEqual(audit["invalid_relocalize_box"], 1)

    def test_single_relocalize_box_does_not_use_pairwise_iou(self):
        rows, audit = MODULE.adjudicate(
            [queue_row()],
            [review(action="RELOCALIZE")],
            "hash",
            min_corrected_box_iou=1.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["exported_relocalize"], 1)

    def test_sensitive_rows_are_excluded(self):
        rows, audit = MODULE.adjudicate(
            [{**queue_row(), "sensitive_attribute": True}], [review()], "hash"
        )
        self.assertEqual(rows, [])
        self.assertEqual(audit["excluded_sensitive_attribute"], 1)

    def test_rejects_dual_reviewer_protocol_setting(self):
        with self.assertRaisesRegex(ValueError, "min_reviewers=1"):
            MODULE.adjudicate([queue_row()], [review()], "hash", min_reviewers=2)


if __name__ == "__main__":
    unittest.main()
