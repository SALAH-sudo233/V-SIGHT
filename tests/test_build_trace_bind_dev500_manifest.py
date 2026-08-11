import gzip
import json
import tempfile
import unittest
from pathlib import Path


class TraceDev500ManifestTest(unittest.TestCase):
    def test_freezes_supervision_free_queue_and_proxy_labels(self):
        from scripts import build_trace_bind_dev500_manifest as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.json"
            records = root / "records.jsonl"
            output = root / "output"
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").touch()
            (images / "b.jpg").touch()
            benchmark.write_text(
                json.dumps(
                    [
                        {
                            "base_sample_id": "g1",
                            "image_filename": "a.jpg",
                            "positive_text": "object next to person",
                            "gt_bbox_xyxy": [0, 0, 10, 10],
                        },
                        {
                            "base_sample_id": "g2",
                            "image_filename": "b.jpg",
                            "positive_text": "cup on table",
                            "gt_bbox_xyxy": [0, 0, 10, 10],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "model": "LENS",
                    "task": "t2_vqa_grounding",
                    "query_role": "positive",
                    "base_sample_id": "g1",
                    "query": "object next to person",
                    "pred_found": True,
                    "pred_bbox_xyxy": [0, 0, 10, 10],
                },
                {
                    "model": "LENS",
                    "task": "t2_vqa_grounding",
                    "query_role": "positive",
                    "base_sample_id": "g2",
                    "query": "cup on table",
                    "pred_found": False,
                    "pred_bbox_xyxy": None,
                },
            ]
            records.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            original_arguments = module.arguments
            module.arguments = lambda: type(
                "Args",
                (),
                {
                    "benchmark": benchmark,
                    "records": records,
                    "upstream_model": "LENS",
                    "upstream_task": "t2_vqa_grounding",
                    "label_iou_threshold": 0.5,
                    "expected_records": 2,
                    "image_root": images,
                    "output_dir": output,
                    "force": False,
                },
            )()
            try:
                self.assertEqual(module.main(), 0)
            finally:
                module.arguments = original_arguments

            with gzip.open(output / "trace_bind_dev500_queue.jsonl.gz", "rt") as handle:
                queue = [json.loads(line) for line in handle]
            with gzip.open(output / "trace_bind_dev500_labels.jsonl.gz", "rt") as handle:
                labels = [json.loads(line) for line in handle]
            self.assertEqual(len(queue), 2)
            self.assertIsNone(queue[1]["upstream_box_xyxy"])
            self.assertFalse(
                {"binding_label", "gt_bbox_xyxy", "iou", "model"} & set(queue[0])
            )
            self.assertEqual(
                [row["binding_label"] for row in labels],
                ["supported", "contradicted"],
            )
            self.assertEqual(
                {row["source_queue_sha256"] for row in labels},
                {
                    json.loads(
                        (output / "trace_bind_dev500_queue.jsonl.gz.summary.json").read_text()
                    )["queue_sha256"]
                },
            )


if __name__ == "__main__":
    unittest.main()
