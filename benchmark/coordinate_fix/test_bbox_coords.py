"""Test bbox_coords: rescale raw_numbers, never touch pred_bbox_xyxy."""
import pytest
from bbox_coords import rescale_raw_numbers, apply_coordinate_system, clamp_bbox


class TestRescaleRawNumbers:
    def test_pixels_returns_none(self):
        # pixels and norm_01 are already handled by parser; return None = no-op
        assert rescale_raw_numbers([100, 200, 300, 400], "pixels", (640, 480)) is None
    
    def test_norm_01_returns_none(self):
        assert rescale_raw_numbers([0.5, 0.5, 0.8, 0.8], "norm_01", (640, 480)) is None
    
    def test_none_passthrough(self):
        assert rescale_raw_numbers(None, "norm_1000", (640, 480)) is None
    
    def test_wrong_length_returns_none(self):
        assert rescale_raw_numbers([10, 20, 30], "norm_1000", (640, 480)) is None
    
    def test_norm_1000_square_image(self):
        # [500, 250, 750, 625] on 800×800 → [400, 200, 600, 500]
        result = rescale_raw_numbers([500, 250, 750, 625], "norm_1000", (800, 800))
        assert result == pytest.approx([400.0, 200.0, 600.0, 500.0])
    
    def test_norm_1000_qwen35_real_case(self):
        # Qwen3.5 real: [17, 38, 355, 988] on 640×422 → y2=988 is out-of-bounds
        # Should scale to [10.88, 16.036, 227.2, 416.936] then clamp y2 to 422
        result = rescale_raw_numbers([17, 38, 355, 988], "norm_1000", (640, 422))
        assert result == pytest.approx([10.88, 16.036, 227.2, 416.936], abs=1e-2)
    
    def test_norm_1000_internvl_real_case(self):
        # InternVL real: [598, 492, 919, 989] on 640×480 → both x2,y2 out-of-bounds
        result = rescale_raw_numbers([598, 492, 919, 989], "norm_1000", (640, 480))
        # Expected: [382.72, 236.16, 588.16, 474.72], all in-bounds after scale
        assert result == pytest.approx([382.72, 236.16, 588.16, 474.72], abs=1e-2)
    
    def test_invalid_system_raises(self):
        with pytest.raises(ValueError, match="Unknown coord_system"):
            rescale_raw_numbers([100, 200, 300, 400], "invalid", (640, 480))


class TestApplyCoordinateSystem:
    def test_not_found_unchanged(self):
        parsed = {"pred_found": False, "pred_bbox_xyxy": None, "raw_numbers": None}
        result = apply_coordinate_system(parsed, "norm_1000", (640, 480))
        assert result["bbox_coord_system"] == "norm_1000"
        assert result["bbox_rescaled"] is False
        assert result["pred_bbox_xyxy"] is None
    
    def test_pixels_no_rescale(self):
        parsed = {"pred_found": True, "pred_bbox_xyxy": [100, 200, 300, 400],
                  "raw_numbers": [100, 200, 300, 400]}
        result = apply_coordinate_system(parsed, "pixels", (640, 480))
        assert result["bbox_coord_system"] == "pixels"
        assert result["bbox_rescaled"] is False
        assert result["pred_bbox_xyxy"] == [100, 200, 300, 400]
        assert "pred_bbox_xyxy_preclamp" not in result
    
    def test_norm_01_no_rescale(self):
        # llava case: parser already converted [0.636, ...] → pixels
        parsed = {"pred_found": True, "pred_bbox_xyxy": [407.04, 46.008, 547.2, 253.896],
                  "raw_numbers": [0.636, 0.108, 0.855, 0.596]}
        result = apply_coordinate_system(parsed, "norm_01", (640, 426))
        assert result["bbox_coord_system"] == "norm_01"
        assert result["bbox_rescaled"] is False
        assert result["pred_bbox_xyxy"] == pytest.approx([407.04, 46.008, 547.2, 253.896])
    
    def test_norm_1000_rescale_qwen35(self):
        # Qwen3.5: parser saw [17, 38, 355, 988], clamped y2 to 422 → [17, 38, 355, 422]
        parsed = {"pred_found": True, "pred_bbox_xyxy": [17.0, 38.0, 355.0, 422.0],
                  "raw_numbers": [17, 38, 355, 988], "parse_valid": True}
        result = apply_coordinate_system(parsed, "norm_1000", (640, 422))
        assert result["bbox_coord_system"] == "norm_1000"
        assert result["bbox_rescaled"] is True
        assert result["pred_bbox_xyxy_preclamp"] == [17.0, 38.0, 355.0, 422.0]
        # After rescale: [10.88, 16.036, 227.2, 416.936]
        assert result["pred_bbox_xyxy"] == pytest.approx([10.88, 16.036, 227.2, 416.936], abs=1e-2)
    
    def test_norm_1000_rescale_internvl(self):
        # InternVL: raw [598, 492, 919, 989] on 640×480, parser clamped → [598, 480, 640, 480]
        parsed = {"pred_found": True, "pred_bbox_xyxy": [598.0, 480.0, 640.0, 480.0],
                  "raw_numbers": [598, 492, 919, 989], "parse_valid": True}
        result = apply_coordinate_system(parsed, "norm_1000", (640, 480))
        assert result["bbox_coord_system"] == "norm_1000"
        assert result["bbox_rescaled"] is True
        # After rescale: [382.72, 236.16, 588.16, 474.72]
        assert result["pred_bbox_xyxy"] == pytest.approx([382.72, 236.16, 588.16, 474.72], abs=1e-2)
    
    def test_original_parsed_unchanged(self):
        # Verify apply_coordinate_system does not mutate the input dict
        parsed = {"pred_found": True, "pred_bbox_xyxy": [17.0, 38.0, 355.0, 422.0],
                  "raw_numbers": [17, 38, 355, 988]}
        original_bbox = parsed["pred_bbox_xyxy"]
        result = apply_coordinate_system(parsed, "norm_1000", (640, 422))
        assert parsed["pred_bbox_xyxy"] is original_bbox  # not modified
        assert result["pred_bbox_xyxy"] != original_bbox


class TestClampBbox:
    def test_already_in_bounds(self):
        assert clamp_bbox([100, 200, 300, 400], 640, 480) == [100, 200, 300, 400]
    
    def test_clamp_right_and_bottom(self):
        assert clamp_bbox([500, 400, 700, 550], 640, 480) == [500, 400, 640, 480]
    
    def test_clamp_negative(self):
        assert clamp_bbox([-10, -5, 50, 60], 640, 480) == [0, 0, 50, 60]
    
    def test_clamp_all_sides(self):
        assert clamp_bbox([-10, -10, 700, 600], 640, 480) == [0, 0, 640, 480]
