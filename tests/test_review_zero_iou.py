import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_zero_iou", ROOT / "scripts" / "review_zero_iou.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReviewZeroIouTest(unittest.TestCase):
    def test_html_page_has_resolved_schema_values(self):
        page = MODULE.html_page().decode("utf-8")
        self.assertNotIn("%FAILURE_MODES%", page)
        self.assertIn("same_category_wrong_instance", page)
        self.assertIn('id="complete"', page)

    def test_real_manifest_has_complete_image_coverage(self):
        groups = MODULE.load_groups(MODULE.DEFAULT_MANIFEST, MODULE.DEFAULT_IMAGES)
        self.assertEqual(len(groups), 127)
        self.assertEqual(sum(len(group["cases"]) for group in groups), 254)
        self.assertTrue(all(group["image_exists"] for group in groups))

    def test_latest_review_is_reviewer_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.jsonl"
            MODULE.append_jsonl(path, {"base_sample_id": "g", "reviewer_id": "r1", "status": "draft"})
            MODULE.append_jsonl(path, {"base_sample_id": "g", "reviewer_id": "r2", "status": "completed"})
            MODULE.append_jsonl(path, {"base_sample_id": "g", "reviewer_id": "r1", "status": "completed"})
            latest = MODULE.load_latest_reviews(path)
        self.assertEqual(latest[("g", "r1")]["status"], "completed")
        self.assertEqual(latest[("g", "r2")]["status"], "completed")

    def test_completed_review_requires_both_task_labels(self):
        request = {
            "base_sample_id": "g",
            "reviewer_id": "reviewer_1",
            "status": "completed",
            "query_support": "supported",
            "case_reviews": {
                task: {
                    "failure_mode": "same_category_wrong_instance",
                    "preferred_action": "switch",
                    "binding_evidence": ["target_reference_relation"],
                    "ambiguity": "clear",
                    "notes": "",
                }
                for task in MODULE.TASKS
            },
            "group_notes": "",
        }
        cleaned = MODULE.validate_submission(request, {"g"})
        self.assertEqual(cleaned["status"], "completed")
        request["case_reviews"][MODULE.TASKS[1]]["preferred_action"] = None
        with self.assertRaisesRegex(ValueError, "preferred_action"):
            MODULE.validate_submission(request, {"g"})


if __name__ == "__main__":
    unittest.main()
