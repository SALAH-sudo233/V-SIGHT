import argparse
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vsight.ccv import DetectorEvidence, DetectorProposal
from vsight.composite_detector import evidence_to_dict


SPEC = importlib.util.spec_from_file_location(
    "run_agentic_drift_audit",
    Path(__file__).parents[1] / "scripts" / "run_agentic_drift_audit.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AgenticAuditReplayIndexTest(unittest.TestCase):
    def test_default_config_is_strict_single_owner_policy(self):
        config, effective, digest = MODULE.load_agentic_config(
            Path(__file__).parents[1] / "configs" / "agentic_drift_loop_v1.json"
        )
        self.assertEqual(config.max_rounds, 3)
        self.assertEqual(config.proposal_cap, 5)
        self.assertEqual(
            effective["promotion"]["annotation_protocol"],
            "single_project_owner",
        )
        self.assertEqual(effective["promotion"]["minimum_reviewers"], 1)
        self.assertEqual(len(digest), 64)

    def test_effective_config_hash_ignores_json_format_and_key_order(self):
        source = Path(__file__).parents[1] / "configs" / "agentic_drift_loop_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps(payload, indent=4), encoding="utf-8")
            second.write_text(
                json.dumps(dict(reversed(list(payload.items()))), separators=(",", ":")),
                encoding="utf-8",
            )
            _, _, first_hash = MODULE.load_agentic_config(first)
            _, _, second_hash = MODULE.load_agentic_config(second)
        self.assertEqual(first_hash, second_hash)

    def test_config_rejects_unknown_fields_and_invalid_budget(self):
        source = Path(__file__).parents[1] / "configs" / "agentic_drift_loop_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            payload["silent_override"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown=.*silent_override"):
                MODULE.load_agentic_config(path)

            payload.pop("silent_override")
            payload["inference_budget"]["detector_image_encoder_forwards"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one detector"):
                MODULE.load_agentic_config(path)

    def test_custom_thresholds_reach_loop_and_binder(self):
        source = Path(__file__).parents[1] / "configs" / "agentic_drift_loop_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["alternative_support"] = 0.91
        payload["binding_margin"] = 0.81
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config, _, _ = MODULE.load_agentic_config(path)
        loop = MODULE.AgenticDriftLoop(config)
        self.assertEqual(loop.verifier.thresholds.alternative_support, 0.91)
        self.assertEqual(loop.verifier.thresholds.binding_margin, 0.81)

    def test_replay_writes_effective_hash_and_round_budget_to_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.jsonl"
            features_path = root / "features.jsonl.gz"
            record_root = root / "records"
            model_dir = record_root / "model-a"
            output_dir = root / "output"
            model_dir.mkdir(parents=True)

            detector = DetectorEvidence(
                proposals=(
                    DetectorProposal(
                        "object:0", (10, 10, 30, 30), 0.95,
                        {"object": 0.95},
                    ),
                ),
                image_width=100,
                image_height=100,
                evidence_complete=True,
                prompted_atom_ids=frozenset({"object"}),
            )
            evidence_path.write_text(
                json.dumps(
                    {
                        "record_id": "semantic-1",
                        "detector_evidence": evidence_to_dict(detector),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with gzip.open(features_path, "wt", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "group_id": "g1",
                            "image_filename": "image.jpg",
                            "model": "model-a",
                            "task": "t2_vqa_grounding",
                            "sample_id": "s1",
                            "query": "the cup",
                            "semantic_query_id": "semantic-1",
                        }
                    )
                    + "\n"
                )
            (model_dir / "records.jsonl").write_text(
                json.dumps(
                    {
                        "model": "model-a",
                        "task": "t2_vqa_grounding",
                        "sample_id": "s1",
                        "pred_found": True,
                        "pred_bbox_xyxy": [10, 10, 30, 30],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            options = argparse.Namespace(
                evidence_glob=str(evidence_path),
                features=features_path,
                record_root=record_root,
                config=Path(__file__).parents[1]
                / "configs"
                / "agentic_drift_loop_v1.json",
                model="model-a",
                limit=1,
                output_dir=output_dir,
                no_memory=True,
                memory_path=None,
                memory_readonly=False,
                force=False,
            )
            with mock.patch.object(MODULE, "args", return_value=options):
                self.assertEqual(MODULE.main(), 0)
            with gzip.open(output_dir / "audit.jsonl.gz", "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(row["schema_version"], "vsight_agentic_drift_audit_v2")
        self.assertEqual(
            row["effective_config_sha256"], summary["effective_config_sha256"]
        )
        self.assertEqual(row["budget_ledger"]["additional_detector_image_encoder_forwards"], 0)
        self.assertEqual(row["rounds_executed"], 1)

    def test_all_model_prediction_index_keeps_models_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("model-a", "model-b"):
                folder = root / model
                folder.mkdir()
                (folder / "records.jsonl").write_text(
                    json.dumps(
                        {
                            "model": model,
                            "task": "t2_vqa_grounding",
                            "sample_id": "same-sample",
                            "pred_found": True,
                            "pred_bbox_xyxy": [1, 1, 2, 2],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            rows = MODULE.load_predictions(root, "all")
            self.assertEqual(len(rows), 2)
            self.assertIn(("model-a", "t2_vqa_grounding", "same-sample"), rows)
            self.assertIn(("model-b", "t2_vqa_grounding", "same-sample"), rows)

    def test_single_model_index_keeps_legacy_key_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "model-a"
            folder.mkdir()
            (folder / "records.jsonl").write_text(
                json.dumps(
                    {
                        "model": "model-a",
                        "task": "t2_vqa_grounding",
                        "sample_id": "sample",
                        "pred_found": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = MODULE.load_predictions(Path(directory), "model-a")
            self.assertIn(("t2_vqa_grounding", "sample"), rows)


if __name__ == "__main__":
    unittest.main()
