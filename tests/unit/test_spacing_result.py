"""
tests/unit/test_spacing_result.py

Tests for SpacingResult dataclass field contracts.
Guards against accidental field removal or type changes that would break
batch assembly in well_spacing_stats.py.
"""

import math

import pytest

from src.well_data.well_spacing_stats import SpacingResult

pytestmark = pytest.mark.unit


class TestSpacingResultFields:
    def test_required_fields_present(self, spacing_result_stub: SpacingResult):
        required = [
            "well_i", "well_k",
            "horizontal_dist", "horizontal_dist_median",
            "vertical_dist", "dist3d",
            "overlap_len_common_ft", "LL_i", "LL_k",
            "tvd_i", "tvd_k",
            "overlap_pct_i", "overlap_pct_k",
            "n_samples", "dy_p5",
            "angle_deg", "pair_alignment",
            "min_distance_ft", "mean_windowed_ft",
            "reject_reason",
            "direction_axis", "direction_to_k_from_i_axis",
            "direction_axis_confidence", "direction_axis_distribution",
            "drill_direction_i", "drill_direction_k",
            "axis_forced",
        ]
        for field in required:
            assert hasattr(spacing_result_stub, field), f"Missing field: {field}"

    def test_valid_pair_alignment_string(self, spacing_result_stub: SpacingResult):
        valid = {"parallel_like", "oblique", "perpendicular", "misaligned"}
        assert spacing_result_stub.pair_alignment in valid

    def test_n_samples_positive_for_valid_pair(self, spacing_result_stub: SpacingResult):
        assert spacing_result_stub.n_samples >= 1

    def test_empty_reject_reason_for_valid_pair(self, spacing_result_stub: SpacingResult):
        assert spacing_result_stub.reject_reason == ""

    def test_dist3d_consistent_with_components(self, spacing_result_stub: SpacingResult):
        """dist3d ≈ sqrt(horizontal_dist² + vertical_dist²) within 1%."""
        expected = math.sqrt(
            spacing_result_stub.horizontal_dist ** 2
            + spacing_result_stub.vertical_dist ** 2
        )
        assert abs(spacing_result_stub.dist3d - expected) / expected < 0.01

    def test_overlap_pct_range(self, spacing_result_stub: SpacingResult):
        assert 0.0 <= spacing_result_stub.overlap_pct_i <= 1.0
        assert 0.0 <= spacing_result_stub.overlap_pct_k <= 1.0

    def test_direction_axis_values(self, spacing_result_stub: SpacingResult):
        assert spacing_result_stub.direction_axis in {"EW", "NS", None}

    def test_drill_direction_values(self, spacing_result_stub: SpacingResult):
        assert spacing_result_stub.drill_direction_i in {"EW", "NS"}
        assert spacing_result_stub.drill_direction_k in {"EW", "NS"}

    def test_parallel_like_has_nan_oblique_fields(self, spacing_result_stub: SpacingResult):
        """For parallel_like pairs, min_distance_ft and mean_windowed_ft should be NaN."""
        assert math.isnan(spacing_result_stub.min_distance_ft)
        assert math.isnan(spacing_result_stub.mean_windowed_ft)

    def test_optional_contact_fields_default_nan(self, spacing_result_stub: SpacingResult):
        """Contact fields default to NaN for parallel-like pairs."""
        assert math.isnan(spacing_result_stub.contact_threshold_ft)
        assert math.isnan(spacing_result_stub.contact_len_i_ft)
        assert math.isnan(spacing_result_stub.contact_pct_i)
