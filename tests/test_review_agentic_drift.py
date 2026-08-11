import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "review_agentic_drift",
    Path(__file__).parents[1] / "scripts/review_agentic_drift.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AgenticDriftReviewValidationTest(unittest.TestCase):
    def setUp(self):
        self.known = {
            "a": {
                "annotation_id": "a",
                "image_filename": "image.jpg",
                "stage_b_applicable_atoms": ["identity", "relation"],
            }
        }
        self.sizes = {"image.jpg": (100, 100)}
        self.payload = {
            "annotation_id": "a",
            "reviewer_id": "项目负责人",
            "status": "completed",
            "parser_status": "correct",
            "evidence_state": "WRONG_INSTANCE",
            "wrong_instance_subtype": "target_reference_relation",
            "atom_states": {"identity": "supported", "relation": "contradicted"},
            "reference_bbox_xyxy": [50, 10, 80, 40],
            "reference_visibility": "visible",
            "confidence": 0.95,
        }

    def test_wrong_instance_can_be_retained_without_replacement_box(self):
        row = MODULE.validate_submission(self.payload, self.known, self.sizes)
        self.assertEqual(row["evidence_state"], "WRONG_INSTANCE")
        self.assertIsNone(row["corrected_target_bbox_xyxy"])
        self.assertFalse(row["policy_action_assigned"])
        self.assertEqual(row["annotation_protocol"], "single_project_owner")
        self.assertEqual(row["review_authority"], "project_owner")

    def test_supported_correct_requires_supported_atoms(self):
        self.payload.update(
            evidence_state="SUPPORTED_CORRECT",
            wrong_instance_subtype=None,
            atom_states={"identity": "supported", "relation": "contradicted"},
        )
        with self.assertRaisesRegex(ValueError, "全部适用原子"):
            MODULE.validate_submission(self.payload, self.known, self.sizes)

    def test_observable_relation_requires_reference_box(self):
        self.payload["reference_bbox_xyxy"] = None
        with self.assertRaisesRegex(ValueError, "独立的关系参照框"):
            MODULE.validate_submission(self.payload, self.known, self.sizes)

    def test_non_wrong_state_rejects_subtype_and_corrected_box(self):
        self.payload.update(
            evidence_state="ABSENT_UNSUPPORTED",
            wrong_instance_subtype="other",
            atom_states={"identity": "contradicted", "relation": "unobservable"},
            corrected_target_bbox_xyxy=[1, 1, 20, 20],
        )
        with self.assertRaisesRegex(ValueError, "只有错误实例绑定状态"):
            MODULE.validate_submission(self.payload, self.known, self.sizes)

    def test_page_exposes_clear_three_step_chinese_workflow(self):
        for text in (
            "步骤一：核对查询解析",
            "步骤二：标注红框内的证据",
            "步骤三：给出证据结论",
            "上游原框",
            "目标校正框",
            "关系参照框",
        ):
            self.assertIn(text, MODULE.HTML)
        self.assertIn("setupReviewWorkflow();refresh(true);", MODULE.HTML)

    def test_incorrect_parser_can_recover_omitted_atoms(self):
        self.payload.update(
            parser_status="incorrect",
            atom_states={
                "identity": "supported",
                "attribute": "contradicted",
                "action": "unobservable",
                "relation": "unobservable",
            },
            reference_bbox_xyxy=None,
            reference_visibility=None,
            wrong_instance_subtype="target_attribute",
        )
        row = MODULE.validate_submission(self.payload, self.known, self.sizes)
        self.assertEqual(
            row["atom_states"],
            {
                "identity": "supported",
                "attribute": "contradicted",
                "action": "unobservable",
                "relation": "unobservable",
            },
        )

    def test_canvas_is_anchored_to_full_image_layer(self):
        self.assertIn("imageLayer.append($('image'),$('canvas'))", MODULE.HTML)
        self.assertIn(".image-layer canvas", MODULE.HTML)

    def test_natural_agentic_queue_role_is_supported(self):
        self.assertIn(
            "development_agentic_natural_audit_not_training",
            MODULE.SUPPORTED_QUEUE_ROLES,
        )


if __name__ == "__main__":
    unittest.main()
