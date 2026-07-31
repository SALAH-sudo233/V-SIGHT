import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsight.data_isolation import audit_splits, identity  # noqa: E402


def row(group: str, image: str, pair: str) -> dict:
    return {
        "base_sample_id": group,
        "image_filename": image,
        "pair_id": pair,
    }


class DataIsolationTest(unittest.TestCase):
    def test_identity_counts_grouped_pairs(self):
        result = identity([row("g1", "a.jpg", "p1"), row("g1", "a.jpg", "p2")])
        self.assertEqual(result.records, 2)
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(len(result.pairs), 2)

    def test_audit_detects_image_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            dev = root / "dev.json"
            train.write_text(json.dumps([row("train", "same.jpg", "train-p")]))
            dev.write_text(json.dumps([row("dev", "same.jpg", "dev-p")]))
            report = audit_splits({"train": train, "dev": dev})
        self.assertFalse(report["all_disjoint"])
        self.assertEqual(
            report["comparisons"]["train__dev"]["image_overlap_count"], 1
        )

    def test_coco_integer_and_filename_are_the_same_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            dev = root / "dev.json"
            train.write_text(
                json.dumps(
                    [
                        {
                            "base_sample_id": "train",
                            "image_id": 445503,
                            "pair_id": "train-p",
                        }
                    ]
                )
            )
            dev.write_text(
                json.dumps(
                    [
                        row(
                            "dev",
                            "COCO_train2014_000000445503.jpg",
                            "dev-p",
                        )
                    ]
                )
            )
            report = audit_splits({"train": train, "dev": dev})
        self.assertFalse(report["all_disjoint"])
        self.assertEqual(
            report["comparisons"]["train__dev"]["image_overlap_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
