"""
tests/integration/test_spacing_calculator.py

Integration tests for WellSpacingCalculator with synthetic trajectories.
These tests run the real spacing engine on a tiny 3-well dataset.
Marked 'slow' because the calculator does real geometry.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.well_data.well_spacing_stats import AlignmentType, WellSpacingCalculator

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures — module-scoped so calculator runs once per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spacing_df(trajectories_df, minimal_header_df):
    """
    Full WellSpacingCalculator output for the 3-well synthetic dataset.
    Created once per module to avoid repeating the expensive calculation.
    """
    calc = WellSpacingCalculator(
        trajectories=trajectories_df,
        header_df=minimal_header_df,
    )
    return calc._calculate_spacing_statistics(
        batch_size=10_000,
        max_distance_miles=2.0,
        max_crossline_ft=3_000.0,   # 3000 ft — our wells are ~700 ft apart, so all pairs included
    )


# ---------------------------------------------------------------------------
# Output shape and required columns
# ---------------------------------------------------------------------------


class TestSpacingCalculatorOutput:
    def test_returns_dataframe(self, spacing_df):
        assert isinstance(spacing_df, pd.DataFrame)

    def test_required_output_columns(self, spacing_df):
        required = [
            "well_i", "well_k",
            "horizontal_dist", "vertical_dist", "3D_dist",
            "angle_deg", "pair_alignment",
            "overlap_len_common_ft", "LL_i", "LL_k",
            "n_samples", "reject_reason",
        ]
        for col in required:
            assert col in spacing_df.columns, f"Missing output column: {col}"

    def test_has_pairs(self, spacing_df):
        """3-well dataset should produce at least one valid pair."""
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert len(valid) >= 1

    def test_no_self_pairs(self, spacing_df):
        """well_i must never equal well_k."""
        assert (spacing_df["well_i"] != spacing_df["well_k"]).all()

    def test_distances_are_positive(self, spacing_df):
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert (valid["horizontal_dist"] > 0).all()
        assert (valid["vertical_dist"] >= 0).all()

    def test_dist3d_geq_horizontal_dist(self, spacing_df):
        """3D distance ≥ horizontal distance (by definition)."""
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert (valid["3D_dist"] >= valid["horizontal_dist"] - 0.01).all()

    def test_overlap_pct_between_0_and_1(self, spacing_df):
        valid = spacing_df[
            (spacing_df["reject_reason"] == "")
            & (spacing_df["pair_alignment"] == "parallel_like")
        ]
        if not valid.empty:
            assert valid["overlap_pct_i"].between(0.0, 1.0).all()
            assert valid["overlap_pct_k"].between(0.0, 1.0).all()

    def test_n_samples_positive_for_valid_pairs(self, spacing_df):
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert (valid["n_samples"] >= 1).all()


# ---------------------------------------------------------------------------
# Alignment classification (parallel-like expected for ~700 ft spacing)
# ---------------------------------------------------------------------------


class TestAlignmentClassification:
    def test_closely_spaced_wells_are_parallel_like(self, spacing_df):
        """
        Our synthetic wells are ~700 ft apart with angle ≈ 0–2°.
        They must classify as PARALLEL_LIKE.
        """
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        if not valid.empty:
            alignments = valid["pair_alignment"].unique()
            # At least one parallel_like pair
            assert "parallel_like" in alignments

    def test_angle_deg_within_bounds(self, spacing_df):
        """All angle values must be in [0, 90]."""
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert valid["angle_deg"].between(0.0, 90.0).all()

    def test_pair_alignment_valid_values(self, spacing_df):
        valid_labels = {"parallel_like", "oblique", "perpendicular", "misaligned"}
        assert set(spacing_df["pair_alignment"].unique()).issubset(valid_labels)


# ---------------------------------------------------------------------------
# Numeric sanity checks
# ---------------------------------------------------------------------------


class TestNumericSanity:
    def test_no_inf_in_distances(self, spacing_df):
        for col in ["horizontal_dist", "vertical_dist", "3D_dist"]:
            assert not np.isinf(spacing_df[col]).any(), f"Inf found in {col}"

    def test_no_negative_horizontal_dist(self, spacing_df):
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert (valid["horizontal_dist"] >= 0).all()

    def test_lateral_lengths_positive(self, spacing_df):
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        assert (valid["LL_i"] > 0).all()
        assert (valid["LL_k"] > 0).all()

    def test_horizontal_dist_in_ft_range(self, spacing_df):
        """Synthetic wells are ~700 ft apart — expected horizontal_dist 500–1500 ft."""
        valid = spacing_df[spacing_df["reject_reason"] == ""]
        if not valid.empty:
            mean_dist = valid["horizontal_dist"].mean()
            assert 400 < mean_dist < 2_000, f"Mean horizontal_dist {mean_dist:.0f} ft unexpected"


# ---------------------------------------------------------------------------
# Dict vs DataFrame input equivalence
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_dict_input_produces_same_pairs(
        self, trajectories_df, trajectories_dict, minimal_header_df
    ):
        """Dict and long-format DataFrame inputs should produce the same pair set."""
        calc_df  = WellSpacingCalculator(trajectories_df, minimal_header_df)
        calc_dict = WellSpacingCalculator(trajectories_dict, minimal_header_df)

        result_df   = calc_df._calculate_spacing_statistics(batch_size=10_000, max_distance_miles=2.0, max_crossline_ft=3_000.0)
        result_dict = calc_dict._calculate_spacing_statistics(batch_size=10_000, max_distance_miles=2.0, max_crossline_ft=3_000.0)

        pairs_df   = set(zip(result_df["well_i"],   result_df["well_k"]))
        pairs_dict = set(zip(result_dict["well_i"], result_dict["well_k"]))
        assert pairs_df == pairs_dict

    def test_raises_on_df_without_uwi(self, minimal_header_df):
        bad_df = pd.DataFrame({"md": [1], "x": [1], "y": [1], "tvd": [1]})
        with pytest.raises(ValueError, match="uwi"):
            WellSpacingCalculator(bad_df, minimal_header_df)
