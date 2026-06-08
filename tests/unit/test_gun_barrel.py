"""
tests/unit/test_gun_barrel.py

Unit tests for dashboard/pipeline.py::compute_gun_barrel() and
dashboard/components/gun_barrel.py::build_gun_barrel_figure().

No database, no file I/O. Uses session-scoped fixtures from conftest.
"""

import math

import pandas as pd
import pytest
import plotly.graph_objects as go

from dashboard.pipeline import compute_gun_barrel
from dashboard.components.gun_barrel import (
    build_gun_barrel_figure,
    empty_figure,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# compute_gun_barrel — data logic
# ---------------------------------------------------------------------------


class TestComputeGunBarrel:
    def test_returns_dataframe(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        assert isinstance(gb, pd.DataFrame)

    def test_row_count_equals_unique_wells(self, ik_spacing_df, heeltoe_df):
        """One row per unique well_i in sorted order."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        unique_wells = ik_spacing_df["well_i"].nunique()
        assert len(gb) == unique_wells

    def test_cum_dist_first_well_is_zero(self, ik_spacing_df, heeltoe_df):
        """First well in sorted order must start at cum_dist = 0."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        assert gb["cum_dist"].iloc[0] == pytest.approx(0.0)

    def test_cum_dist_monotonically_nondecreasing(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        diffs = gb["cum_dist"].diff().dropna()
        assert (diffs >= 0).all(), "cum_dist must be non-decreasing"

    def test_section_dist_centered_at_zero(self, ik_spacing_df, heeltoe_df):
        """sectionDist = cum_dist - cum_dist.max()/2 → midpoint is ~0."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        mid = gb["sectionDist"].mean()
        max_val = gb["cum_dist"].max()
        assert abs(mid) <= max_val / 2 + 1.0  # within half-range

    def test_section_dist_column_present(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        assert "sectionDist" in gb.columns

    def test_empty_ik_returns_empty_df(self, heeltoe_df):
        gb = compute_gun_barrel(pd.DataFrame(), heeltoe_df)
        assert gb.empty

    def test_single_well_cum_dist_zero(self, ik_spacing_df, heeltoe_df, well_uwis):
        """Single-well IK (no pairs) → cum_dist = 0."""
        single_ik = ik_spacing_df[ik_spacing_df["well_i"] == well_uwis[0]].head(1).copy()
        single_heel = heeltoe_df[heeltoe_df["uwi"] == well_uwis[0]].copy()
        gb = compute_gun_barrel(single_ik, single_heel)
        assert gb["cum_dist"].iloc[0] == pytest.approx(0.0)

    def test_cum_dist_matches_horizontal_dist(self, ik_spacing_df, heeltoe_df):
        """Second well's cum_dist should equal first pair's horizontal_dist."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        # horizontal_dist of the first pair
        first_h_dist = ik_spacing_df["horizontal_dist"].iloc[0]
        assert gb["cum_dist"].iloc[1] == pytest.approx(first_h_dist, rel=1e-3)

    def test_ew_wells_sort_by_lat_south_to_north(self, ik_spacing_df, heeltoe_df):
        """EW wells → sorted by mid_Lat (S→N); northernmost well has highest cum_dist."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        # fixture uses EW wells; mid_Lat increases with each well
        lats = gb["mid_Lat"].tolist()
        assert lats == sorted(lats), "EW wells should be sorted S→N by mid_Lat"

    def test_3d_dist_col_preserved(self, ik_spacing_df, heeltoe_df):
        """3D_dist or dist3d column should be preserved in merged output."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        has_3d = "3D_dist" in gb.columns or "dist3d" in gb.columns
        assert has_3d


# ---------------------------------------------------------------------------
# empty_figure
# ---------------------------------------------------------------------------


class TestEmptyFigure:
    def test_returns_go_figure(self):
        fig = empty_figure("No data")
        assert isinstance(fig, go.Figure)

    def test_annotation_contains_message(self):
        msg = "Click a well on the map."
        fig = empty_figure(msg)
        annotations = fig.layout.annotations
        assert any(msg in (a.text or "") for a in annotations)

    def test_axes_hidden(self):
        fig = empty_figure()
        assert not fig.layout.xaxis.visible
        assert not fig.layout.yaxis.visible


# ---------------------------------------------------------------------------
# build_gun_barrel_figure
# ---------------------------------------------------------------------------


class TestBuildGunBarrelFigure:
    def test_returns_go_figure(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        fig = build_gun_barrel_figure(gb, ik_spacing_df)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        fig = build_gun_barrel_figure(gb, ik_spacing_df)
        assert len(fig.data) > 0

    def test_yaxis_reversed(self, ik_spacing_df, heeltoe_df):
        """TVD must be reversed (deeper = lower on chart)."""
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        fig = build_gun_barrel_figure(gb, ik_spacing_df)
        assert fig.layout.yaxis.autorange == "reversed"

    def test_xaxis_label_section_dist(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        fig = build_gun_barrel_figure(gb, ik_spacing_df, x_col="sectionDist")
        assert "Section" in fig.layout.xaxis.title.text

    def test_xaxis_label_cum_dist(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        fig = build_gun_barrel_figure(gb, ik_spacing_df, x_col="cum_dist")
        assert "Cumulative" in fig.layout.xaxis.title.text

    def test_empty_gb_returns_empty_figure(self, ik_spacing_df):
        fig = build_gun_barrel_figure(pd.DataFrame(), ik_spacing_df)
        # empty figure has no real data traces
        data_traces = [t for t in fig.data if len(t.x or []) > 0]
        assert len(data_traces) == 0

    def test_formation_tops_layer_added_when_provided(self, ik_spacing_df, heeltoe_df, well_uwis):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        df_tops = pd.DataFrame({
            "uwi":       [well_uwis[0], well_uwis[1]],
            "formation": ["FORM A1", "FORM A1"],
            "top_tvd":   [-10_500.0, -10_510.0],
        })
        n_before = len(build_gun_barrel_figure(gb, ik_spacing_df).data)
        n_after  = len(build_gun_barrel_figure(gb, ik_spacing_df, df_formation_tops=df_tops).data)
        assert n_after > n_before

    def test_formation_tops_not_added_when_none(self, ik_spacing_df, heeltoe_df):
        gb = compute_gun_barrel(ik_spacing_df, heeltoe_df)
        fig = build_gun_barrel_figure(gb, ik_spacing_df, df_formation_tops=None)
        # formation trace names should not appear
        trace_names = [t.name for t in fig.data]
        assert not any("FORM" in (n or "") for n in trace_names)
