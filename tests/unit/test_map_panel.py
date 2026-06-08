"""
tests/unit/test_map_panel.py

Unit tests for dashboard/components/map_panel.py
  - build_trajectory_geodataframe()
  - build_bottomhole_geodataframe()

No database, no file I/O.
"""

import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point

from dashboard.components.map_panel import (
    build_trajectory_geodataframe,
    build_bottomhole_geodataframe,
)

pytestmark = pytest.mark.unit


class TestBuildTrajectoryGeodataframe:
    def test_returns_geodataframe(self, trajectories_df, minimal_header_df):
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_one_row_per_well(self, trajectories_df, minimal_header_df, well_uwis):
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        assert len(gdf) == len(well_uwis)

    def test_geometry_is_linestring(self, trajectories_df, minimal_header_df):
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        for geom in gdf.geometry:
            assert isinstance(geom, LineString)

    def test_crs_is_wgs84(self, trajectories_df, minimal_header_df):
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        assert gdf.crs.to_epsg() == 4326

    def test_header_columns_joined(self, trajectories_df, minimal_header_df):
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        assert "well_name" in gdf.columns
        assert "bench" in gdf.columns

    def test_uwi_column_present(self, trajectories_df, minimal_header_df):
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        assert "uwi" in gdf.columns

    def test_bottomhole_coords_stored(self, trajectories_df, minimal_header_df):
        """bh_lon and bh_lat must be populated (last survey station per well)."""
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        assert "bh_lon" in gdf.columns
        assert "bh_lat" in gdf.columns
        assert gdf["bh_lon"].notna().all()
        assert gdf["bh_lat"].notna().all()

    def test_linestring_has_multiple_vertices(self, trajectories_df, minimal_header_df):
        """Each wellbore stick must have at least 2 coordinate pairs."""
        gdf = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        for geom in gdf.geometry:
            assert len(geom.coords) >= 2

    def test_skips_single_station_wells(self, minimal_header_df):
        """Wells with only one survey station cannot form a LineString — should be skipped."""
        import pandas as pd
        single_station = pd.DataFrame({
            "uwi":       ["SINGLE"],
            "md":        [5000.0],
            "tvd":       [3200.0],
            "x":         [700_000.0],
            "y":         [3_490_000.0],
            "latitude":  [31.51],
            "longitude": [-101.90],
            "azimuth":   [90.0],
            "inclination": [90.0],
        })
        gdf = build_trajectory_geodataframe(single_station, minimal_header_df)
        assert len(gdf) == 0

    def test_empty_lateral_df_returns_empty_gdf(self, minimal_header_df):
        import pandas as pd
        empty_df = pd.DataFrame(columns=["uwi", "md", "tvd", "x", "y",
                                          "latitude", "longitude", "azimuth"])
        gdf = build_trajectory_geodataframe(empty_df, minimal_header_df)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0


class TestBuildBottomholeGeodataframe:
    def test_returns_geodataframe(self, trajectories_df, minimal_header_df):
        gdf_traj = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        gdf_bh   = build_bottomhole_geodataframe(gdf_traj)
        assert isinstance(gdf_bh, gpd.GeoDataFrame)

    def test_same_row_count_as_trajectories(self, trajectories_df, minimal_header_df):
        gdf_traj = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        gdf_bh   = build_bottomhole_geodataframe(gdf_traj)
        assert len(gdf_bh) == len(gdf_traj)

    def test_geometry_is_point(self, trajectories_df, minimal_header_df):
        gdf_traj = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        gdf_bh   = build_bottomhole_geodataframe(gdf_traj)
        for geom in gdf_bh.geometry:
            assert isinstance(geom, Point)

    def test_point_coords_match_bh_lon_lat(self, trajectories_df, minimal_header_df):
        gdf_traj = build_trajectory_geodataframe(trajectories_df, minimal_header_df)
        gdf_bh   = build_bottomhole_geodataframe(gdf_traj)
        for _, row in gdf_bh.iterrows():
            pt = row.geometry
            assert pt.x == pytest.approx(row["bh_lon"], abs=1e-9)
            assert pt.y == pytest.approx(row["bh_lat"], abs=1e-9)
