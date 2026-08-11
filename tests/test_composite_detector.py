import unittest
from types import SimpleNamespace

from vsight.ccv import TypedClaimParser
from vsight.composite_detector import (
    GroundingDINOCompositeDetector,
    build_raft_prompt,
    build_composite_prompt,
    evidence_from_dict,
    evidence_to_dict,
    normalize_detector_output,
    normalize_detector_token_output,
    _build_trace_ledger_from_prediction,
)
from vsight.attention_transport import build_raft_evidence, infer_role_spans


class CompositeDetectorTest(unittest.TestCase):
    def test_detector_defaults_to_legacy_label_alignment(self):
        detector = GroundingDINOCompositeDetector("unused-local-model")
        self.assertEqual(detector.alignment, "legacy_label")

    def test_raft_intervention_requires_raft_alignment(self):
        with self.assertRaisesRegex(ValueError, "requires alignment='raft'"):
            GroundingDINOCompositeDetector(
                "unused-local-model", raft_intervention=True
            )

    def test_trace_capture_is_opt_in_and_requires_raw_query_alignment(self):
        detector = GroundingDINOCompositeDetector("unused-local-model")
        self.assertFalse(detector.trace_bind)
        with self.assertRaisesRegex(ValueError, "requires alignment='raft'"):
            GroundingDINOCompositeDetector(
                "unused-local-model", trace_bind=True
            )

    def test_trace_decoder_reduction_builds_versioned_ledger(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        class ClassHead:
            def __init__(self, logits):
                self.logits = torch.tensor(logits, dtype=torch.float32)

            def __call__(self, **kwargs):
                query_count = kwargs["vision_hidden_state"].shape[1]
                return self.logits[:query_count].unsqueeze(0)

        query = "cup next to plate"
        offsets = [[0, 3], [4, 8], [9, 11], [12, 17]]
        roles = (
            type("Role", (), {
                "role": "target", "text": "cup", "token_indices": (0,),
                "confidence": 0.9, "source": "test",
            })(),
            type("Role", (), {
                "role": "predicate", "text": "next to", "token_indices": (1, 2),
                "confidence": 0.8, "source": "test",
            })(),
            type("Role", (), {
                "role": "reference", "text": "plate", "token_indices": (3,),
                "confidence": 0.9, "source": "test",
            })(),
        )
        layer_logits = [
            [[2.0, 0.5, 0.5, -1.0], [-1.0, 0.5, 0.5, 2.0], [0.0, 0.0, 0.0, 0.0]],
            [[3.0, 1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]],
        ]
        model = SimpleNamespace(class_embed=[ClassHead(value) for value in layer_logits])
        prediction = SimpleNamespace(
            intermediate_hidden_states=torch.tensor(
                [[
                    [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                    [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]],
                ]]
            ),
            intermediate_reference_points=torch.tensor(
                [[
                    [[0.2, 0.2, 0.2, 0.2], [0.7, 0.2, 0.2, 0.2], [0.5, 0.7, 0.2, 0.2]],
                    [[0.21, 0.2, 0.2, 0.2], [0.69, 0.2, 0.2, 0.2], [0.5, 0.69, 0.2, 0.2]],
                ]]
            ),
            encoder_last_hidden_state_text=torch.zeros((1, 4, 2)),
        )
        ledger = _build_trace_ledger_from_prediction(
            model=model,
            prediction=prediction,
            inputs=SimpleNamespace(attention_mask=torch.ones((1, 4), dtype=torch.long)),
            query=query,
            token_offsets=offsets,
            roles=roles,
            upstream_box_xyxy=[10, 10, 30, 30],
            sample_id="synthetic",
            image_width=100,
            image_height=100,
            detector_only_latency_ms=10.0,
            candidate_cap=2,
        )
        self.assertEqual(ledger["schema_version"], "vsight_trace_bind_pilot_v1")
        self.assertEqual(ledger["upstream_target_object_query_id"], 0)
        self.assertEqual(len(ledger["proposal_trajectories"]), 3)
        self.assertFalse(ledger["action_policy_applied"])

    def test_raft_prompt_does_not_expand_ontology(self):
        prompt = build_raft_prompt("the gizmo next to the brass object")
        self.assertEqual(prompt.text, "the gizmo next to the brass object.")
        self.assertEqual([segment.segment_id for segment in prompt.segments], ["full"])

    def test_raft_induces_roles_from_attention_without_vocab(self):
        query = "the helmet next to the red car"
        prompt = build_raft_prompt(query)
        offsets = []
        cursor = 0
        for token in query.split():
            start = query.index(token, cursor)
            offsets.append([start, start + len(token)])
            cursor = start + len(token)
        maps = []
        for start, end in offsets:
            token = query[start:end]
            if token == "helmet":
                maps.append([0.9, 0.05, 0.03, 0.02])
            elif token in {"next", "to"}:
                maps.append([0.1, 0.4, 0.4, 0.1])
            elif token in {"red", "car"}:
                maps.append([0.02, 0.03, 0.05, 0.9])
            else:
                maps.append([0.01, 0.01, 0.01, 0.01])
        roles = infer_role_spans(query=query, token_offsets=offsets, token_maps=maps)
        self.assertEqual([role.role for role in roles], ["target", "predicate", "reference"])
        self.assertEqual([role.text for role in roles], ["the helmet", "next to", "the red car"])
        self.assertEqual(prompt.segments[0].text, query)

    def test_raft_does_not_invent_relation_for_single_node_query(self):
        query = "the translucent gizmo"
        offsets = []
        cursor = 0
        for token in query.split():
            start = query.index(token, cursor)
            offsets.append([start, start + len(token)])
            cursor = start + len(token)
        roles = infer_role_spans(
            query=query,
            token_offsets=offsets,
            token_maps=[[0.8, 0.1], [0.6, 0.4], [0.7, 0.3]],
        )
        self.assertEqual([role.role for role in roles], ["target"])

    def test_raft_evidence_round_trips_attention_ledger(self):
        query = "the gizmo next to the brass object"
        offsets = []
        cursor = 0
        for token in query.split():
            start = query.index(token, cursor)
            offsets.append([start, start + len(token)])
            cursor = start + len(token)
        maps = [
            [0.01, 0.01, 0.01, 0.01],
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.4, 0.4, 0.1],
            [0.1, 0.4, 0.4, 0.1],
            [0.01, 0.01, 0.01, 0.01],
            [0.02, 0.03, 0.05, 0.9],
            [0.02, 0.03, 0.05, 0.9],
        ]
        probabilities = [
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        ]
        build_kwargs = dict(
            query=query,
            token_offsets=offsets,
            token_maps=maps,
            token_probabilities=probabilities,
            boxes=[[5, 5, 25, 25], [70, 5, 95, 30]],
            global_scores=[0.9, 0.8],
            coordinates=[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]],
            image_width=100,
            image_height=100,
            prompted_atom_ids=frozenset(),
            text_threshold=0.2,
            proposal_cap_per_segment=3,
        )
        result = build_raft_evidence(**build_kwargs)
        payload = evidence_to_dict(result)
        self.assertEqual(payload["schema_version"], "vsight_ccv_composite_evidence_v4")
        self.assertEqual(result.attention_ledger["status"], "ok")
        self.assertEqual(len(result.attention_ledger["role_spans"]), 3)
        restored = evidence_from_dict(payload)
        self.assertEqual(restored.attention_ledger, result.attention_ledger)

    def test_decoder_conditioned_ledger_uses_distinct_query_bindings(self):
        query = "the gizmo next to the brass object"
        offsets = []
        cursor = 0
        for token in query.split():
            start = query.index(token, cursor)
            offsets.append([start, start + len(token)])
            cursor = start + len(token)
        maps = [
            [0.01, 0.01, 0.01, 0.01],
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.4, 0.4, 0.1],
            [0.1, 0.4, 0.4, 0.1],
            [0.01, 0.01, 0.01, 0.01],
            [0.02, 0.03, 0.05, 0.9],
            [0.02, 0.03, 0.05, 0.9],
        ]
        probabilities = [
            [0.1, 0.9, 0.4, 0.4, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        ]
        decoder_text = [
            [0.02, 0.40, 0.12, 0.12, 0.04, 0.15, 0.15],
            [0.02, 0.04, 0.08, 0.08, 0.04, 0.37, 0.37],
            [0.05, 0.15, 0.15, 0.15, 0.05, 0.225, 0.225],
        ]
        decoder_visual = [
            [[0.15, 0.15, 0.8], [0.90, 0.90, 0.2]],
            [[0.82, 0.15, 0.8], [0.10, 0.90, 0.2]],
            [[0.50, 0.50, 1.0]],
        ]
        build_kwargs = dict(
            query=query,
            token_offsets=offsets,
            token_maps=maps,
            token_probabilities=probabilities,
            boxes=[[5, 5, 25, 25], [70, 5, 95, 30], [40, 40, 60, 60]],
            global_scores=[0.9, 0.8, 0.3],
            coordinates=[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]],
            image_width=100,
            image_height=100,
            prompted_atom_ids=frozenset(),
            text_threshold=0.2,
            proposal_cap_per_segment=3,
            proposal_query_indices=[10, 11, 12],
            decoder_text_attention=decoder_text,
            decoder_visual_attention=decoder_visual,
        )
        result = build_raft_evidence(**build_kwargs)
        ledger = result.attention_ledger
        self.assertEqual(ledger["schema_version"], "vsight_cable_raft_attention_v2")
        self.assertTrue(ledger["decoder_edges"])
        self.assertTrue(0.0 <= ledger["decoder_transport_relative_margin"] <= 1.0)
        self.assertTrue(0.0 <= ledger["decoder_edge_uncertainty"] <= 1.0)
        for edge in ledger["decoder_edges"]:
            self.assertNotEqual(
                edge["target_proposal_id"].rsplit(":", 1)[1],
                edge["reference_proposal_id"].rsplit(":", 1)[1],
            )
        causal_effects = [
            {
                "model_query_index": query_index,
                "masked_role": role,
                "relative_drop": value,
                "original_score": 0.8,
                "counterfactual_score": 0.8 * (1.0 - value),
                "causal_drop": 0.8 * value,
                "hidden_cosine_drift": 0.1 * value,
                "hidden_drift_selectivity": value,
            }
            for query_index, values in {
                10: {"target": 0.6, "predicate": 0.3, "reference": 0.05},
                11: {"target": 0.05, "predicate": 0.3, "reference": 0.6},
                12: {"target": 0.1, "predicate": 0.1, "reference": 0.1},
            }.items()
            for role, value in values.items()
        ]
        causal = build_raft_evidence(
            **build_kwargs,
            causal_role_effects=causal_effects,
        ).attention_ledger
        self.assertEqual(causal["schema_version"], "vsight_cable_raft_attention_v3")
        self.assertIsNotNone(causal["best_causal_edge"])
        self.assertGreater(causal["best_causal_edge"]["role_swap_margin"], 0.0)
        self.assertTrue(0.0 <= causal["causal_edge_uncertainty"] <= 1.0)
        self.assertIsNotNone(causal["best_causal_influence_edge"])
        self.assertGreater(
            causal["best_causal_influence_edge"]["role_swap_margin"], 0.0
        )

    def test_composite_prompt_contains_all_views_in_one_text(self):
        claim = TypedClaimParser().parse("the red person left of the blue car")
        prompt = build_composite_prompt(claim)
        self.assertIn("person", prompt.text)
        self.assertIn("red person", prompt.text)
        self.assertIn("blue car", prompt.text)
        self.assertIn(claim.full_expression, prompt.text)
        self.assertEqual(
            prompt.prompted_atom_ids,
            {"object", "attribute:red", "relation:to_the_left_of"},
        )

    def test_open_vocabulary_target_and_known_reference_keep_roles(self):
        claim = TypedClaimParser().parse("the helmet next to the red car")
        prompt = build_composite_prompt(claim)
        segments = {row.segment_id: row.text for row in prompt.segments}
        self.assertEqual(segments["object"], "helmet")
        self.assertEqual(segments["reference:relation:next_to"], "red car")

    def test_normalizes_phrase_alignment_and_round_trips(self):
        claim = TypedClaimParser().parse("the red cup")
        prompt = build_composite_prompt(claim)
        result = normalize_detector_output(
            boxes=[[1, 2, 30, 40], [1, 2, 30, 40]],
            scores=[0.9, 0.8],
            labels=["cup", "red cup"],
            prompt=prompt,
            image_width=100,
            image_height=80,
            latency_ms=12.5,
        )
        self.assertEqual(result.image_encoder_forwards, 1)
        self.assertEqual(result.proposals[0].atom_scores["object"], 0.9)
        self.assertIn("attribute:red", result.proposals[1].atom_scores)
        self.assertEqual(evidence_to_dict(result)["schema_version"], "vsight_ccv_composite_evidence_v3")
        self.assertNotIn("attention_ledger", evidence_to_dict(result))
        restored = evidence_from_dict(evidence_to_dict(result))
        self.assertEqual(restored, result)

    def test_trace_ledger_round_trips_as_v5_evidence(self):
        claim = TypedClaimParser().parse("the red cup")
        prompt = build_composite_prompt(claim)
        result = normalize_detector_output(
            boxes=[[1, 2, 30, 40]],
            scores=[0.9],
            labels=["cup"],
            prompt=prompt,
            image_width=100,
            image_height=80,
        )
        from dataclasses import replace

        result = replace(
            result,
            trace_ledger={
                "schema_version": "vsight_trace_bind_pilot_v1",
                "action_policy_applied": False,
            },
        )
        payload = evidence_to_dict(result)
        self.assertEqual(payload["schema_version"], "vsight_ccv_composite_evidence_v5")
        self.assertEqual(evidence_from_dict(payload), result)

    def test_rejects_non_single_pass_metadata(self):
        payload = {
            "image_width": 10,
            "image_height": 10,
            "evidence_complete": True,
            "prompted_atom_ids": [],
            "image_encoder_forwards": 3,
            "proposals": [],
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evidence_from_dict(payload)

    def test_token_span_alignment_keeps_segments_independent(self):
        claim = TypedClaimParser().parse("the helmet next to the red car")
        prompt = build_composite_prompt(claim)
        offsets = []
        cursor = 0
        for token in prompt.text.split():
            start = prompt.text.index(token, cursor)
            clean_end = start + len(token.rstrip("."))
            offsets.append([start, clean_end])
            cursor = start + len(token)
        scores = [0.0] * len(offsets)
        for index, (start, end) in enumerate(offsets):
            token = prompt.text[start:end]
            scores[index] = {
                "helmet": 0.8,
                "red": 0.6,
                "car": 0.6,
                "next": 0.4,
                "to": 0.4,
            }.get(token, 0.1)
        result = normalize_detector_token_output(
            boxes=[[1, 2, 30, 40]],
            global_scores=[0.9],
            token_scores=[scores],
            token_offsets=offsets,
            prompt=prompt,
            image_width=100,
            image_height=80,
            text_threshold=0.2,
        )
        by_segment = {row.segment_id: row for row in result.proposals}
        self.assertEqual(
            set(by_segment), {"object", "reference:relation:next_to", "full"}
        )
        self.assertEqual(by_segment["object"].label, "helmet")
        self.assertTrue(by_segment["reference:relation:next_to"].is_reference)
        self.assertGreater(by_segment["object"].score, by_segment["full"].score)


if __name__ == "__main__":
    unittest.main()
