import pytest

from hallucination_defense.core.candidate_recovery import (
    CANDIDATE_PROMPTS,
    Candidate,
    box_iou,
    deduplicate_candidates,
    evaluate_record_oracle,
    normalize_box,
    parse_candidate_boxes,
)
from run_v023_oracle import enrich_external_candidates


def test_candidate_prompts_render_literal_json_schema():
    for prompt in CANDIDATE_PROMPTS:
        rendered = prompt.render("the red car")
        assert '"boxes"' in rendered
        assert "the red car" in rendered


def test_normalize_box_orders_clamps_and_rejects_degenerate():
    assert normalize_box([12, -3, 2, 8], image_size=(10, 10)) == (2.0, 0.0, 10.0, 8.0)
    assert normalize_box([2, 2, 2, 9]) is None
    assert normalize_box([0, 0, float("nan"), 4]) is None


def test_parse_candidate_boxes_from_controlled_json_and_fallback():
    text = '```json\n{"boxes":[[1,2,9,10],[12,3,20,18]]}\n```'
    assert parse_candidate_boxes(text, image_size=(16, 16)) == [
        (1.0, 2.0, 9.0, 10.0),
        (12.0, 3.0, 16.0, 16.0),
    ]
    assert parse_candidate_boxes("answer: (4, 5, 8, 9)") == [(4.0, 5.0, 8.0, 9.0)]


def test_deduplicate_candidates_preserves_diverse_boxes():
    candidates = [
        Candidate((0, 0, 10, 10), "a"),
        Candidate((0.1, 0.1, 10.1, 10.1), "b"),
        Candidate((20, 20, 30, 30), "c"),
    ]
    kept = deduplicate_candidates(candidates, iou_threshold=0.9)
    assert [candidate.source for candidate in kept] == ["a", "c"]
    assert box_iou(kept[0].box, kept[1].box) == 0.0


def test_record_oracle_reports_recovery_over_baseline():
    records = [
        {
            "base_sample_id": "a",
            "query_role": "positive",
            "candidate_source": "base",
            "gt_bbox_xyxy": [0, 0, 10, 10],
            "pred_bbox_xyxy": None,
        },
        {
            "base_sample_id": "a",
            "query_role": "positive",
            "candidate_source": "alt",
            "gt_bbox_xyxy": [0, 0, 10, 10],
            "candidate_boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
        },
        {
            "base_sample_id": "b",
            "query_role": "positive",
            "candidate_source": "base",
            "gt_bbox_xyxy": [0, 0, 10, 10],
            "pred_bbox_xyxy": [0, 0, 5, 10],
        },
    ]
    summary = evaluate_record_oracle(records, ["base", "alt"], "base")
    assert summary["n_groups"] == 2
    assert summary["sources"]["base"]["top1"]["mean_iou"] == pytest.approx(0.25)
    assert summary["sources"]["alt"]["source_oracle"]["mean_iou"] == pytest.approx(0.5)
    assert summary["oracle"]["mean_iou"] == pytest.approx(0.75)
    assert summary["gain_over_baseline"]["recovered_baseline_zero_count"] == 1
    assert summary["gain_over_baseline"]["recovered_to_iou_0_5_count"] == 1


def test_external_candidate_records_join_exact_repaired_positive():
    canonical = {
        ("group-a", "the red car"): {
            "sample_id": "group-a",
            "base_sample_id": "group-a",
            "query": "the red car",
            "gt_bbox_xyxy": [1, 2, 11, 12],
        }
    }
    external = [
        {
            "model": "REC",
            "task": "t2_vqa_grounding",
            "sample_id": "group-a",
            "query_role": "positive",
            "query": "the red car",
            "pred_bbox_xyxy": [1, 2, 11, 12],
        },
        {
            "model": "REC",
            "task": "t2_vqa_grounding",
            "sample_id": "unknown",
            "query_role": "positive",
            "query": "the red car",
            "pred_bbox_xyxy": [1, 2, 11, 12],
        },
    ]
    joined = list(enrich_external_candidates(external, canonical))
    assert len(joined) == 1
    assert joined[0]["gt_bbox_xyxy"] == [1, 2, 11, 12]
    assert joined[0]["candidate_source"] == "REC"
