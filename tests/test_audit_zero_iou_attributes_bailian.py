import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_zero_iou_attributes_bailian",
    ROOT / "scripts" / "audit_zero_iou_attributes_bailian.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_audit():
    return {
        "target_visible": True,
        "query_binds_gt_target": "yes",
        "target_identity": {
            "category": "person",
            "apparent_gender": "unclear",
            "gender_evidence": "",
        },
        "attributes": {
            "colors": [
                {
                    "part": "shirt",
                    "value": "blue",
                    "confidence": "high",
                    "evidence": "visible blue pixels",
                }
            ],
            "materials": [],
            "other": [],
            "actions_or_states": [],
        },
        "spatial_and_relational_facts": [],
        "query_attribute_checks": [
            {
                "atom": "person",
                "type": "object",
                "verdict": "supported",
                "evidence": "visible person",
            }
        ],
        "disambiguating_cues": ["blue shirt"],
        "instance_confusion_risk": "high",
        "likely_zero_iou_cause": "same_category_instance_confusion",
        "reason": "the baseline selected another person",
    }


class AttributeAuditTest(unittest.TestCase):
    def test_selects_unique_groups_with_valid_baseline_zero(self):
        groups = MODULE.select_groups(MODULE.DEFAULT_MANIFEST)
        self.assertEqual(len(groups), 114)
        self.assertEqual(sum(len(group["zero_iou_cases"]) for group in groups), 180)
        self.assertTrue(
            all(
                case["baseline_box"] is not None and case["baseline_iou"] == 0.0
                for group in groups
                for case in group["zero_iou_cases"]
            )
        )

    def test_validates_expected_attribute_schema(self):
        MODULE.validate_audit(valid_audit())
        broken = valid_audit()
        broken["target_identity"]["apparent_gender"] = "man"
        with self.assertRaisesRegex(ValueError, "apparent_gender"):
            MODULE.validate_audit(broken)
        broken = valid_audit()
        broken["query_attribute_checks"][0]["type"] = "category"
        with self.assertRaisesRegex(ValueError, "type"):
            MODULE.validate_audit(broken)

    def test_gender_presentation_requires_visible_evidence(self):
        broken = valid_audit()
        broken["target_identity"]["apparent_gender"] = "female_presenting"
        with self.assertRaisesRegex(ValueError, "gender_evidence"):
            MODULE.validate_audit(broken)

    def test_vision_stage_preserves_uncertain_verdict(self):
        evidence = {
            "target_visible": True,
            "target_identity_observation": {
                "category": "person",
                "apparent_gender": "unclear",
                "gender_evidence": "",
            },
            "visible_attributes": {
                "colors": [],
                "materials": [],
                "other": [],
                "actions_or_states": [],
            },
            "spatial_and_relational_facts": [],
            "query_visual_checks": [
                {
                    "atom": "left person",
                    "type": "relation",
                    "verdict": "uncertain",
                    "evidence": "depth ordering is unclear",
                }
            ],
            "gt_disambiguating_cues": [],
            "zero_iou_baseline_observations": [],
            "limitations": ["low resolution"],
        }
        MODULE.validate_vision_evidence(evidence)

    def test_normalizes_only_explicit_query_type_aliases(self):
        audit = {"query_attribute_checks": [{"type": "category"}, {"type": "made_up"}]}
        corrections = MODULE.normalize_audit_types(audit)
        self.assertEqual(audit["query_attribute_checks"][0]["type"], "object")
        self.assertEqual(audit["query_attribute_checks"][1]["type"], "made_up")
        self.assertEqual(corrections, ["query_attribute_checks[0].type:category->object"])

    def test_uses_latest_human_review_status_per_reviewer(self):
        rows = [
            {"base_sample_id": "a", "reviewer_id": "r1", "status": "completed"},
            {"base_sample_id": "a", "reviewer_id": "r1", "status": "draft"},
            {"base_sample_id": "b", "reviewer_id": "r1", "status": "draft"},
            {"base_sample_id": "b", "reviewer_id": "r2", "status": "completed"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(MODULE.completed_human_review_ids(path), {"b"})

    def test_renders_boxed_full_image_and_target_crop(self):
        group = MODULE.select_groups(MODULE.DEFAULT_MANIFEST)[0]
        views = MODULE.render_views(group, MODULE.DEFAULT_IMAGES)
        self.assertEqual([label for label, _ in views], [
            "full image with GT and zero-IoU baseline boxes",
            "unmarked GT target crop",
        ])
        self.assertTrue(all(data.startswith("data:image/jpeg;base64,") for _, data in views))


if __name__ == "__main__":
    unittest.main()
