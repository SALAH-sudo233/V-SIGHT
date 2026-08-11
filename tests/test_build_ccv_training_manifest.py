import importlib.util
import unittest
from pathlib import Path

from vsight.ccv import TypedClaimParser
from vsight.composite_detector import (
    build_composite_prompt,
    evidence_to_dict,
    normalize_detector_output,
)


SPEC = importlib.util.spec_from_file_location(
    "build_ccv_training_manifest",
    Path(__file__).parents[1] / "scripts/build_ccv_training_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CCVTrainingManifestTest(unittest.TestCase):
    def test_joins_only_eligible_reviews_and_preserves_audit_fields(self):
        query = "the red cup"
        prompt = build_composite_prompt(TypedClaimParser().parse(query))
        evidence = normalize_detector_output(
            boxes=[[10, 10, 30, 30], [10, 10, 30, 30]],
            scores=[0.9, 0.8],
            labels=["cup", "red cup"],
            prompt=prompt,
            image_width=100,
            image_height=100,
        )
        review = {
            "annotation_id": "a1",
            "group_id": "g1",
            "image_id": 1,
            "image_filename": "image.jpg",
            "query": query,
            "original_bbox_xyxy": [10, 10, 30, 30],
            "verifier_action": "ACCEPT",
            "router_only_corrected_boxes_xyxy": [],
            "reviewer_ids": ["project_owner"],
            "reviewer_count": 1,
            "authoritative_reviewer_id": "project_owner",
            "annotation_protocol": "single_project_owner",
            "minimum_confidence": 0.95,
            "mean_confidence": 0.96,
            "source_queue_sha256": "queue-hash",
        }
        outputs = MODULE.build_rows(
            [review],
            [{"record_id": "a1", "detector_evidence": evidence_to_dict(evidence)}],
            0.2,
        )
        rows = outputs[MODULE.split_for("g1", 0.2)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verifier_action"], "ACCEPT")
        self.assertIn("atom_evidence", rows[0])
        self.assertIn("binding_margin", rows[0]["ccv_features"])
        self.assertEqual(rows[0]["source_queue_sha256"], "queue-hash")
        self.assertEqual(rows[0]["annotation_protocol"], "single_project_owner")
        self.assertFalse(rows[0]["inter_reviewer_agreement_computed"])

    def test_missing_evidence_is_not_silently_dropped(self):
        review = {
            "annotation_id": "missing",
            "group_id": "g1",
            "verifier_action": "ACCEPT",
            "reviewer_count": 1,
            "annotation_protocol": "single_project_owner",
            "minimum_confidence": 0.95,
            "source_queue_sha256": "hash",
        }
        with self.assertRaisesRegex(ValueError, "missing composite evidence"):
            MODULE.build_rows([review], [], 0.2)


if __name__ == "__main__":
    unittest.main()
