import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "build_agentic_drift_natural_queue",
    Path(__file__).parents[1] / "scripts" / "build_agentic_drift_natural_queue.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(group, model, query, task="t2_vqa_grounding"):
    return {
        "record_id": f"{model}:{task}:{group}",
        "query": query,
        "original_bbox": [1, 2, 10, 20],
        "evidence_state": "WRONG_INSTANCE",
        "action": "ABSTAIN",
        "reason_codes": ["CLAIM_SUPPORT_WEAK"],
        "review_priority": 0.8,
        "context": {
            "group_id": group,
            "model": model,
            "task": task,
            "image_filename": f"{group}.jpg",
        },
    }


class NaturalAgenticQueueTest(unittest.TestCase):
    def test_selects_one_model_per_unique_group_task_and_is_deterministic(self):
        rows = []
        for index in range(8):
            query = "the red car next to the person" if index % 2 else "the car"
            for model in ("LENS", "Seg-R1", "visual-rft"):
                rows.append(row(f"g{index}", model, query))
        targets = {"relation": 2, "attribute": 0, "action": 0, "object": 2}
        first = MODULE.select_natural_rows(rows, seed="test", stratum_targets=targets)
        second = MODULE.select_natural_rows(rows, seed="test", stratum_targets=targets)
        self.assertEqual([item["record_id"] for item in first], [item["record_id"] for item in second])
        self.assertEqual(len(first), 4)
        self.assertEqual(len({item["context"]["group_id"] for item in first}), 4)

    def test_queue_rows_do_not_expose_agent_fields(self):
        selected = MODULE.select_natural_rows(
            [row("g1", "LENS", "the car")],
            stratum_targets={"relation": 0, "attribute": 0, "action": 0, "object": 1},
        )
        queue, sidecar = MODULE.build_rows(selected, source_audit_sha256="hash")
        self.assertEqual(len(queue), 1)
        self.assertFalse(MODULE.FORBIDDEN_QUEUE_FIELDS & set(queue[0]))
        self.assertNotIn("source_model", queue[0])
        self.assertEqual(sidecar[0]["source_model"], "LENS")


if __name__ == "__main__":
    unittest.main()
