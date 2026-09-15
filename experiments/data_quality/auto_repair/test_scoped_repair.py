import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).parent


def load_module():
    path = ROOT / "scoped_repair.py"
    spec = importlib.util.spec_from_file_location("scoped_repair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScopedRepairTest(unittest.TestCase):
    def test_scope_excludes_500dev_and_keeps_2000_components(self):
        m = load_module()
        rows = [
            {"set": "1996", "sid": "a", "ht": "object"},
            {"set": "added4", "sid": "b", "ht": "relation"},
            {"set": "500dev", "sid": "c", "ht": "object"},
        ]
        selected = m.scope_rows(rows)
        self.assertEqual([(x["set"], x["sid"]) for x in selected], [("1996", "a"), ("added4", "b")])

    def test_secondary_gate_fails_closed_per_check(self):
        m = load_module()
        enum_fields = {
            f"{role}_{suffix}"
            for role in ("A", "B")
            for suffix in ("matches_box", "has_match_anywhere", "unique")
        }
        enforced = m.SECONDARY_CHECKS | enum_fields
        verdict = {key: True for key in m.SECONDARY_CHECKS}
        verdict.update(
            A_matches_box="yes",
            A_unique="yes",
            A_has_match_anywhere="yes",
            B_matches_box="no",
            B_unique="yes",
            B_has_match_anywhere="no",
            reason="visible evidence",
        )
        self.assertTrue(m.accept_secondary(verdict, "A"))
        for key in enforced:
            bad = dict(verdict)
            bad[key] = False
            self.assertFalse(m.accept_secondary(bad, "A"))

    def test_secondary_never_accepts_uncertain_labels(self):
        m = load_module()
        verdict = {key: True for key in m.SECONDARY_CHECKS}
        verdict.update(
            A_matches_box="uncertain",
            A_unique="yes",
            A_has_match_anywhere="yes",
            B_matches_box="no",
            B_unique="yes",
            B_has_match_anywhere="no",
            reason="uncertain",
        )
        self.assertFalse(m.accept_secondary(verdict, "A"))

    def test_evidence_page_accepts_tuple_key_without_crashing(self):
        m = load_module()
        row = {"set": "1996", "sid": "x", "ht": "object", "pos": "a", "neg": "b"}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            m.evidence_page(folder, row, {"pos": "c", "neg": "d"}, {}, "validation_failed")
            self.assertTrue((folder / "before_after.html").exists())

    def test_load_candidate_reuses_first_generation_response(self):
        m = load_module()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            payload = {
                "raw_response": {
                    "choices": [{
                        "message": {
                            "content": '{"status":"proposed","pos":"new positive","neg":"new negative","reason":"visible"}'
                        }
                    }]
                }
            }
            path = folder / "generation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            candidate = m.load_generation_candidate({"evidence_dir": str(folder)})
            self.assertEqual(candidate["pos"], "new positive")
            self.assertEqual(candidate["neg"], "new negative")

    def test_fine_grained_run_is_separate_from_historical_run(self):
        m = load_module()
        self.assertTrue(str(m.RUN).endswith("2000held_finegrained_v2"))
        self.assertNotEqual(m.RUN.name, "20260911T083146Z")

    def test_final_validation_failed_only_uses_latest_event(self):
        m = load_module()
        events = [
            {"content_id": "a", "status": "validation_failed"},
            {"content_id": "a", "status": "accepted"},
            {"content_id": "b", "status": "validation_failed"},
        ]
        final = m.final_validation_failed(events)
        self.assertEqual([x["content_id"] for x in final], ["b"])


if __name__ == "__main__":
    unittest.main()
