"""
tests/integration/test_geo_survey.py

Integration tests for src/well_data/well_data_manager.py:
  - GeoSurveyProcessor.compute_utm_coordinates()
  - GeoSurveyProcessor.filter_after_heel_point()
  - GeoSurveyProcessor.get_heel_toe_midpoints_latlon()
  - GeoSurveyProcessor.prepare_lateral_trajectory_data()
  - dashboard/pipeline._auto_detect_utm()

These tests construct real GeoSurveyProcessor instances with synthetic
DataFrames. No database or file system access required.
"""

import pytest
import numpy as np
import pandas as pd

from src.well_data.well_data_manager import GeoSurveyProcessor
from dashboard.pipeline import _auto_detect_utm

pytestmark = pytest.mark.integration

# inclination_filter is a REQUIRED positional arg — use 30.0 throughout
INCL_FILTER = 30.0


# ---------------------------------------------------------------------------
# GeoSurveyProcessor — construction
# ---------------------------------------------------------------------------


class TestGeoSurveyProcessorInit:
    def test_constructs_with_header_and_directional(
        self, minimal_header_df, minimal_directional_df
    ):
        proc = GeoSurveyProcessor(
            header_df=minimal_header_df,
            directional_df=minimal_directional_df,
        )
        assert proc is not None

    def test_accepts_ihs_source(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(
            header_df=minimal_header_df,
            directional_df=minimal_directional_df,
            directional_source="ihs",
        )
        assert proc is not None


# ---------------------------------------------------------------------------
# UTM projection
# ---------------------------------------------------------------------------


class TestComputeUtmCoordinates:
    def test_adds_x_y_columns(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.compute_utm_coordinates(minimal_directional_df)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_x_y_are_numeric(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.compute_utm_coordinates(minimal_directional_df)
        assert pd.api.types.is_numeric_dtype(result["x"])
        assert pd.api.types.is_numeric_dtype(result["y"])

    def test_x_in_expected_range(self, minimal_header_df, minimal_directional_df):
        """
        Midland Basin area. UTM Zone 14 easting ≈ 700 000–800 000 m (or ×3.28 if ft).
        Accept both units: metres (600k–900k) and feet (2M–3M).
        """
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.compute_utm_coordinates(minimal_directional_df)
        x_mean = result["x"].mean()
        assert (600_000 < x_mean < 900_000) or (2_000_000 < x_mean < 3_000_000), (
            f"UTM x out of expected range: {x_mean:.0f}"
        )

    def test_no_nan_in_utm_output(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.compute_utm_coordinates(minimal_directional_df)
        assert result["x"].notna().all()
        assert result["y"].notna().all()


# ---------------------------------------------------------------------------
# Heel point filtering
# ---------------------------------------------------------------------------


class TestFilterAfterHeelPoint:
    def test_removes_vertical_section(self, minimal_header_df, minimal_directional_df):
        """
        Vertical section rows have inclination = 0.0 (pure vertical).
        After filtering with inclination_filter=30.0, no rows with inclination < 30.0 remain.
        The boundary value (30.0) may be kept or dropped depending on implementation —
        we only assert that clearly-vertical rows (< 30.0) are gone.
        """
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        lateral = proc.filter_after_heel_point(minimal_directional_df, INCL_FILTER)
        assert not (lateral["inclination"] < INCL_FILTER).any()

    def test_lateral_row_count_less_than_total(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        lateral = proc.filter_after_heel_point(minimal_directional_df, INCL_FILTER)
        assert len(lateral) < len(minimal_directional_df)

    def test_all_uwis_still_represented(
        self, minimal_header_df, minimal_directional_df, well_uwis
    ):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        lateral = proc.filter_after_heel_point(minimal_directional_df, INCL_FILTER)
        remaining_uwis = set(lateral["uwi"].unique())
        for uwi in well_uwis:
            assert uwi in remaining_uwis, f"UWI {uwi} missing after heel filter"

    def test_md_monotonically_increasing_per_well(
        self, minimal_header_df, minimal_directional_df
    ):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        lateral = proc.filter_after_heel_point(minimal_directional_df, INCL_FILTER)
        for uwi, grp in lateral.groupby("uwi"):
            md_sorted = grp["md"].tolist()
            assert md_sorted == sorted(md_sorted), f"MD not monotone for {uwi}"


# ---------------------------------------------------------------------------
# Heel/toe midpoints
# ---------------------------------------------------------------------------


class TestGetHeelToeMidpoints:
    def test_returns_dataframe_with_required_cols(
        self, minimal_header_df, minimal_directional_df
    ):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.get_heel_toe_midpoints_latlon(minimal_directional_df, INCL_FILTER)
        for col in ["uwi", "mid_Lat", "mid_Lon"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_well(self, minimal_header_df, minimal_directional_df, well_uwis):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.get_heel_toe_midpoints_latlon(minimal_directional_df, INCL_FILTER)
        assert len(result) == len(well_uwis)

    def test_midpoints_within_lat_lon_bounds(
        self, minimal_header_df, minimal_directional_df
    ):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df)
        result = proc.get_heel_toe_midpoints_latlon(minimal_directional_df, INCL_FILTER)
        assert result["mid_Lat"].between(30.0, 33.0).all()
        assert result["mid_Lon"].between(-103.0, -100.0).all()


# ---------------------------------------------------------------------------
# prepare_lateral_trajectory_data — full pipeline
# ---------------------------------------------------------------------------


class TestPrepareLateralTrajectoryData:
    def test_returns_two_dataframes(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df, directional_source="ihs")
        lateral, header_aligned = proc.prepare_lateral_trajectory_data(INCL_FILTER)
        assert isinstance(lateral, pd.DataFrame)
        assert isinstance(header_aligned, pd.DataFrame)

    def test_lateral_has_x_y_columns(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df, directional_source="ihs")
        lateral, _ = proc.prepare_lateral_trajectory_data(INCL_FILTER)
        assert "x" in lateral.columns
        assert "y" in lateral.columns

    def test_lateral_fewer_rows_than_raw(self, minimal_header_df, minimal_directional_df):
        proc = GeoSurveyProcessor(minimal_header_df, minimal_directional_df, directional_source="ihs")
        lateral, _ = proc.prepare_lateral_trajectory_data(INCL_FILTER)
        assert len(lateral) < len(minimal_directional_df)


# ---------------------------------------------------------------------------
# _auto_detect_utm (dashboard/pipeline.py helper)
# ---------------------------------------------------------------------------


class TestAutoDetectUtm:
    def test_returns_epsg_string_format(self, minimal_header_df):
        crs = _auto_detect_utm(minimal_header_df)
        assert crs.startswith("EPSG:")
        epsg_num = int(crs.split(":")[1])
        assert 32601 <= epsg_num <= 32660  # valid UTM North zone range

    def test_midland_basin_fixture_in_valid_zone(self, minimal_header_df):
        """
        Fixture surface_lon ≈ -101.9 → on the Zone 13/14 boundary.
        Accept either 32613 or 32614.
        """
        crs = _auto_detect_utm(minimal_header_df)
        assert crs in ("EPSG:32613", "EPSG:32614")

    @pytest.mark.parametrize("lon,expected_zone", [
        (-103.5, 13),   # docstring example from determine_utm_zone
        (-75.0,  18),   # US East
        (-96.1,  14),   # east of -96 → zone 15? Let's use -96.1 → zone 14
    ])
    def test_zone_formula_matches_src_implementation(self, lon, expected_zone):
        """
        Verify _auto_detect_utm uses same formula as GeoSurveyProcessor.determine_utm_zone():
          int((lon + 180) / 6) + 1
        """
        computed_zone = int((lon + 180) / 6) + 1
        assert computed_zone == expected_zone
        df = pd.DataFrame({"surface_lat": [35.0], "surface_lon": [lon]})
        crs = _auto_detect_utm(df)
        assert crs == f"EPSG:326{expected_zone:02d}"

    def test_uses_surface_lat_lon_columns(self):
        """header_df uses surface_lat/surface_lon, not latitude/longitude."""
        df = pd.DataFrame({"surface_lat": [31.5], "surface_lon": [-103.5]})
        crs = _auto_detect_utm(df)
        assert crs == "EPSG:32613"

    def test_falls_back_to_latitude_longitude(self):
        """Directional df uses latitude/longitude — should also work."""
        df = pd.DataFrame({"latitude": [31.5], "longitude": [-103.5]})
        crs = _auto_detect_utm(df)
        assert crs == "EPSG:32613"
