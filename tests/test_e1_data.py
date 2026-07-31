import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.e1_data import (  # noqa: E402
    CocoIndex,
    SourceSpec,
    annotation_candidate_record,
    assign_image_splits,
    bbox_iou,
    extract_coco_image_id,
    iter_query_records,
    localization_candidates,
    protected_image_ids,
    inference_queue_record,
    select_compute_subset,
    xywh_to_clipped_xyxy,
)


class E1DataTest(unittest.TestCase):
    def test_extracts_integer_and_filename_image_ids(self):
        self.assertEqual(extract_coco_image_id(445503), 445503)
        self.assertEqual(
            extract_coco_image_id("COCO_train2014_000000445503.jpg"), 445503
        )

    def test_image_split_is_deterministic_exact_and_disjoint(self):
        first = assign_image_splits(range(100), 0.1, "seed")
        second = assign_image_splits(reversed(range(100)), 0.1, "seed")
        self.assertEqual(first, second)
        train, calibration = first
        self.assertEqual(len(train), 90)
        self.assertEqual(len(calibration), 10)
        self.assertFalse(train & calibration)

    def test_bbox_is_clipped_to_image(self):
        self.assertEqual(
            xywh_to_clipped_xyxy([-2, 3, 10, 20], 6, 12),
            [0.0, 3.0, 6.0, 12.0],
        )

    def test_protected_ids_are_read_without_other_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protected.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "image_filename": "COCO_train2014_000000445503.jpg",
                            "hallucination_type": "must_not_matter",
                        }
                    ]
                )
            )
            result = protected_image_ids([path])
        self.assertEqual(result, frozenset({445503}))

    def test_query_records_resolve_coco_and_exclude_protected_images(self):
        coco = CocoIndex(
            images={
                1: {"id": 1, "file_name": "one.jpg", "width": 100, "height": 80},
                2: {"id": 2, "file_name": "two.jpg", "width": 100, "height": 80},
            },
            annotations={
                10: {"id": 10, "image_id": 1, "category_id": 1, "bbox": [1, 2, 20, 30]},
                11: {"id": 11, "image_id": 1, "category_id": 1, "bbox": [40, 2, 20, 30]},
                20: {"id": 20, "image_id": 2, "category_id": 1, "bbox": [1, 2, 20, 30]},
            },
            categories={1: "person"},
            category_annotations={(1, 1): (10, 11), (2, 1): (20,)},
        )
        refs = [
            {
                "split": "train",
                "image_id": 1,
                "ann_id": 10,
                "category_id": 1,
                "ref_id": 5,
                "sentences": [{"sent_id": 7, "sent": "left person", "raw": "Left person"}],
            },
            {
                "split": "train",
                "image_id": 2,
                "ann_id": 20,
                "category_id": 1,
                "ref_id": 6,
                "sentences": [{"sent_id": 8, "sent": "other person"}],
            },
        ]
        records = list(
            iter_query_records(
                refs,
                SourceSpec("refcoco_unc", "RefCOCO", "unc", Path("unused")),
                coco,
                frozenset({2}),
                frozenset({1}),
                frozenset(),
            )
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["query_id"], "refcoco_unc:sent:7")
        self.assertEqual(records[0]["same_category_distractor_count"], 1)
        self.assertEqual(records[0]["gt_bbox_xyxy"], [1.0, 2.0, 21.0, 32.0])

    def test_annotation_candidates_rank_same_category_and_localization(self):
        coco = CocoIndex(
            images={1: {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}},
            annotations={
                10: {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                11: {"id": 11, "image_id": 1, "category_id": 1, "bbox": [40, 10, 20, 20]},
                12: {"id": 12, "image_id": 1, "category_id": 1, "bbox": [80, 80, 5, 5]},
            },
            categories={1: "person"},
            category_annotations={(1, 1): (10, 11, 12)},
        )
        record = annotation_candidate_record(10, "train", coco, max_same_category=1)
        self.assertEqual(record["same_category_available"], 2)
        self.assertEqual(record["same_category_candidates"][0]["ann_id"], 11)
        self.assertGreaterEqual(len(record["localization_candidates"]), 4)
        self.assertTrue(
            all(row["iou_to_gt"] < 0.95 for row in record["localization_candidates"])
        )

    def test_iou_and_localization_candidates(self):
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)
        candidates = localization_candidates([0, 0, 100, 100], 100, 100)
        self.assertEqual(
            {row["candidate_type"] for row in candidates},
            {"partial_horizontal", "partial_vertical"},
        )

    def test_annotation_candidate_excludes_indistinguishable_overlap(self):
        coco = CocoIndex(
            images={1: {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}},
            annotations={
                10: {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                11: {"id": 11, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            },
            categories={1: "person"},
            category_annotations={(1, 1): (10, 11)},
        )
        record = annotation_candidate_record(10, "train", coco)
        self.assertEqual(record["same_category_raw"], 1)
        self.assertEqual(record["same_category_overlap_excluded"], 1)
        self.assertEqual(record["same_category_candidates"], [])

    def test_compute_subset_deduplicates_and_strips_supervision(self):
        def item(source, index, image, ann, query):
            return {
                "query_id": f"{source}:{index}",
                "group_id": f"{source}:ref:{index}",
                "data_split": "train",
                "source_dataset": source,
                "source_split_by": "test",
                "image_id": image,
                "image_filename": f"{image}.jpg",
                "image_width": 100,
                "image_height": 80,
                "ann_id": ann,
                "query": query,
                "gt_bbox_xyxy": [1, 2, 3, 4],
            }

        records = {
            "a": [item("a", 1, 1, 10, "left person"), item("a", 2, 2, 20, "cat")],
            "b": [item("b", 1, 1, 10, "left person"), item("b", 2, 3, 30, "dog")],
        }
        selected, stats = select_compute_subset(records, {"a": 1, "b": 1}, "seed")
        semantic = {(row["image_id"], row["ann_id"], row["query"]) for row in selected}
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(semantic), 2)
        self.assertEqual(stats["selected_by_source"], {"a": 1, "b": 1})
        queue = inference_queue_record(selected[0], 0)
        self.assertNotIn("ann_id", queue)
        self.assertNotIn("gt_bbox_xyxy", queue)


if __name__ == "__main__":
    unittest.main()
