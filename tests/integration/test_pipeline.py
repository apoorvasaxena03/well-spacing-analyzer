"""
tests/integration/test_pipeline.py

Integration tests for dashboard/pipeline.py — the src/ wrapper layer.
Tests UTM detection, gun barrel data prep, and cache round-trips.
Uses real src/ classes; no mocking.

Note: load_from_files and project_and_extract_laterals are covered by
smoke tests here. Full column-map and validation tests live in
test_geo_survey.py.
"""

import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from dashboard.pipeline import (
    _auto_detect_utm,
    _merge_roles_into_header,
    compute_gun_barrel,
    load_cached_pipeline,
    project_and_extract_laterals,
)

pytestmark = pytest.mark.integration


class TestMergeRolesIntoHeader:
    """The on-demand 'Tune Roles' flow re-merges roles into an already-merged
    header; the merge must be idempotent (no role_x/role_y duplicates)."""

    @staticmethod
    def _roles():
        return pd.DataFrame({
            "uwi": ["A", "B"],
            "role": ["parent", "child"],
            "parent_uwi": [None, "A"],
            "parent_dist_ft": [None, 640.0],
            "parent_vertical_ft": [None, 30.0],
            "days_since_parent": [None, 200.0],
            "child_gen": [None, "gen1_child"],
        })

    def test_first_merge_adds_roles_and_fills_unmatched(self):
        header = pd.DataFrame({"uwi": ["A", "B", "C"], "bench": ["W", "W", "X"]})
        out = _merge_roles_into_header(header, self._roles())
        assert out.set_index("uwi")["role"].to_dict() == {
            "A": "parent", "B": "child", "C": "no_eligible_neighbor"}
        assert "parent_dist_ft" in out.columns

    def test_re_merge_is_idempotent(self):
        header = pd.DataFrame({"uwi": ["A", "B", "C"], "bench": ["W", "W", "X"]})
        once = _merge_roles_into_header(header, self._roles())
        twice = _merge_roles_into_header(once, self._roles())  # the re-tune case
        assert not [c for c in twice.columns if c.endswith(("_x", "_y"))]
        assert list(once.columns) == list(twice.columns)
        assert twice.set_index("uwi")["role"].to_dict() == once.set_index("uwi")["role"].to_dict()


# ---------------------------------------------------------------------------
# project_and_extract_laterals
# ---------------------------------------------------------------------------


class TestProjectAndExtractLaterals:
    def test_returns_three_items(self, minimal_header_df, minimal_directional_df):
        header, lateral, crs = project_and_extract_laterals(
            minimal_header_df, minimal_directional_df
        )
        assert isinstance(header, pd.DataFrame)
        assert isinstance(lateral, pd.DataFrame)
        assert isinstance(crs, str)

    def test_crs_is_epsg_string(self, minimal_header_df, minimal_directional_df):
        _, _, crs = project_and_extract_laterals(minimal_header_df, minimal_directional_df)
        assert crs.startswith("EPSG:")

    def test_lateral_df_smaller_than_full_directional(
        self, minimal_header_df, minimal_directional_df
    ):
        """Lateral = only sections after heel — fewer rows than raw directional."""
        _, lateral, _ = project_and_extract_laterals(minimal_header_df, minimal_directional_df)
        assert len(lateral) < len(minimal_directional_df)

    def test_lateral_has_x_y_columns(self, minimal_header_df, minimal_directional_df):
        _, lateral, _ = project_and_extract_laterals(minimal_header_df, minimal_directional_df)
        assert "x" in lateral.columns
        assert "y" in lateral.columns

    def test_explicit_crs_override(self, minimal_header_df, minimal_directional_df):
        _, _, crs = project_and_extract_laterals(
            minimal_header_df, minimal_directional_df, crs_to="EPSG:32613"
        )
        assert crs == "EPSG:32613"


# ---------------------------------------------------------------------------
# compute_gun_barrel — pipeline wrapper
# ---------------------------------------------------------------------------


class TestComputeGunBarrelIntegration:
    def test_cum_dist_increases_monotonically(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        diffs = gb["cum_dist"].diff().dropna()
        assert (diffs >= 0).all()

    def test_section_dist_spans_both_sides_of_zero(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        assert gb["sectionDist"].min() < 0
        assert gb["sectionDist"].max() > 0

    def test_elevation_i_column_present(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        assert "elevation_i" in gb.columns

    def test_wells_sorted_by_lat_for_ew_drill(self, ik_spacing_df, heeltoe_df):
        """
        Fixture drill_direction_i = 'EW' → sorted S→N by mid_Lat.
        """
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        lats = gb["mid_Lat"].tolist()
        assert lats == sorted(lats)


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_load_cached_pipeline_restores_data(
        self, minimal_header_df, trajectories_df, tmp_path
    ):
        """Simulate what run_spacing_calculation writes, then verify load_cached_pipeline reads."""
        cache_file = tmp_path / "pipeline_test.pkl"
        payload = {
            "df_spacing":    pd.DataFrame({"a": [1, 2]}),
            "header_df":     minimal_header_df,
            "lateral_df":    trajectories_df,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(payload, f)

        result = load_cached_pipeline(str(cache_file))
        assert "df_spacing" in result
        assert "header_df"  in result
        assert "lateral_df" in result
        assert len(result["df_spacing"]) == 2

    def test_load_cached_pipeline_raises_on_missing_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_cached_pipeline("/nonexistent/path/pipeline.pkl")
