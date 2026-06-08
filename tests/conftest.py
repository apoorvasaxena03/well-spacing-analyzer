"""
tests/conftest.py — shared pytest fixtures for well-spacing-analyzer.

Fixture hierarchy (fastest → slowest):
  scalar helpers
    → minimal_header_df / minimal_directional_df   (in-memory DataFrames)
    → lateral_df                                    (heel-filtered subset)
    → trajectories_dict / trajectories_df           (UTM-projected, ft)
    → spacing_result_stub                           (one SpacingResult instance)
  file fixtures
    → header_csv_path / directional_csv_path        (paths to tests/fixtures/)
  module-scope (expensive, created once per test module)
    → spacing_df                                    (full WellSpacingCalculator output)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def header_csv_path(fixture_dir: Path) -> Path:
    return fixture_dir / "header.csv"


@pytest.fixture(scope="session")
def directional_csv_path(fixture_dir: Path) -> Path:
    return fixture_dir / "directional.csv"


@pytest.fixture(scope="session")
def production_csv_path(fixture_dir: Path) -> Path:
    return fixture_dir / "production.csv"


# ---------------------------------------------------------------------------
# Minimal DataFrames — realistic but tiny (5 wells, ~3 bench EW laterals)
#
# Geometry:
#   3 EW-drilled wells (azimuth ≈ 90°) stacked N-S, same bench
#   Lateral spans ~6000 ft EW in UTM-ft
#   Wells are ~700 ft apart N-S (horizontal_dist ≈ 700 ft → PARALLEL_LIKE)
#
# UTM Zone 13N origin (EPSG:32613) for Midland Basin area:
#   x (Easting)  ≈ 700 000 m  → *3.28084 ≈ 2 296 590 ft
#   y (Northing)  ≈ 3 490 000 m → *3.28084 ≈ 11 450 800 ft
# ---------------------------------------------------------------------------

# Conversion factor used throughout fixtures
M2FT = 3.28084

# UTM base in feet
_X0 = 700_000 * M2FT   # easting base
_Y0 = 3_490_000 * M2FT  # northing base
_TVD = 10_800.0          # TVD ≈ 3200 m * 3.28084 ft
_LATERAL_LEN = 6_000.0   # 6000 ft lateral
_EW_SPACING = 700.0      # ft between wells N-S
_N_STATIONS = 10          # survey stations per lateral


def _make_lateral_stations(
    uwi: str,
    x_start: float,
    y0: float,
    tvd: float,
    lat0: float,
    lon0: float,
    n: int = _N_STATIONS,
) -> pd.DataFrame:
    """Build survey stations for one EW lateral (azimuth=90°)."""
    xs = np.linspace(x_start, x_start + _LATERAL_LEN, n)
    ys = np.full(n, y0)
    tvds = np.full(n, tvd)
    # synthetic lat/lon from UTM (rough approximation for test purposes)
    lats = np.full(n, lat0)
    lons = np.linspace(lon0, lon0 + 0.06, n)  # ~0.06° lon ≈ 6000 ft at lat 31.5
    return pd.DataFrame({
        "uwi": uwi,
        "md": np.linspace(3_000, 9_000, n),  # md starts after vertical section
        "tvd": tvds,
        "x": xs,
        "y": ys,
        "latitude": lats,
        "longitude": lons,
        "azimuth": 90.0,
        "inclination": 90.0,
    })


@pytest.fixture(scope="session")
def well_uwis() -> list[str]:
    return [
        "42-329-40001-0000",
        "42-329-40002-0000",
        "42-329-40003-0000",
    ]


@pytest.fixture(scope="session")
def minimal_header_df(well_uwis: list[str]) -> pd.DataFrame:
    """Minimal valid header DataFrame with canonical column names."""
    return pd.DataFrame({
        "uwi":            well_uwis,
        "well_name":      ["REAGAN 1H", "REAGAN 2H", "REAGAN 3H"],
        "operator":       ["OPERATOR A"] * 3,
        "bench":          ["WOLFCAMP A", "WOLFCAMP A", "WOLFCAMP B"],
        "first_prod_date": pd.to_datetime(["2018-01-15", "2018-03-20", "2019-07-10"]),
        "last_prod_date":  pd.to_datetime(["2023-06-01"] * 3),
        "hole_direction": ["H", "H", "H"],
        "surface_lat":    [31.510, 31.510, 31.510],
        "surface_lon":    [-101.910, -101.904, -101.897],
        "lease_name":     ["REAGAN RANCH"] * 3,
        "well_status":    ["ACTIVE"] * 3,
    })


@pytest.fixture(scope="session")
def trajectories_df(well_uwis: list[str]) -> pd.DataFrame:
    """
    UTM-projected lateral-only trajectories (long format).
    Columns: uwi, md, tvd, x, y, latitude, longitude, azimuth, inclination
    3 EW wells, ~700 ft N-S spacing, ~6000 ft laterals.
    """
    frames = []
    for i, uwi in enumerate(well_uwis):
        y0   = _Y0 + i * _EW_SPACING
        lat0 = 31.510 + i * 0.006   # ~700 ft N increments
        lon0 = -101.910
        frames.append(_make_lateral_stations(uwi, _X0, y0, _TVD + i * 50, lat0, lon0))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def trajectories_dict(trajectories_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Dict[uwi → trajectory DataFrame] (no uwi column inside)."""
    return {
        uwi: grp.drop(columns="uwi").reset_index(drop=True)
        for uwi, grp in trajectories_df.groupby("uwi")
    }


@pytest.fixture(scope="session")
def minimal_directional_df(well_uwis: list[str]) -> pd.DataFrame:
    """
    Directional survey DataFrame with canonical IHS columns.
    Includes vertical section (inclination < 30°) before the lateral.
    """
    frames = []
    for i, uwi in enumerate(well_uwis):
        lat0 = 31.510 + i * 0.006
        lon0 = -101.910
        # Vertical section
        vert = pd.DataFrame({
            "uwi":          uwi,
            "md":           [0, 1000, 2000, 3000],
            "tvd":          [0, 1000, 2000, 2900],
            "inclination":  [0.0, 5.0, 15.0, 30.0],
            "azimuth":      [90.0] * 4,
            "latitude":     [lat0] * 4,
            "longitude":    [lon0] * 4,
            "deviation_E/W": [0.0] * 4,
            "E/W":          ["E"] * 4,
            "deviation_N/S": [0.0] * 4,
            "N/S":          ["N"] * 4,
        })
        # Lateral section
        lat = _make_lateral_stations(uwi, _X0, _Y0 + i * _EW_SPACING, _TVD, lat0, lon0)
        lat["inclination"] = 90.0
        lat["deviation_E/W"] = np.linspace(0, 1800, len(lat))
        lat["E/W"] = "E"
        lat["deviation_N/S"] = 0.0
        lat["N/S"] = "N"
        frames.append(pd.concat([vert, lat], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def ik_spacing_df(well_uwis: list[str]) -> pd.DataFrame:
    """
    Synthetic IK spacing pairs DataFrame — mirrors WellSpacingCalculator output columns.
    Used to test gun barrel and map-panel components without running the full calculator.
    Two pairs: (well0, well1) and (well1, well2).
    """
    w = well_uwis
    return pd.DataFrame({
        "well_i":                [w[0],      w[1]],
        "well_k":                [w[1],      w[2]],
        "horizontal_dist":       [700.0,     720.0],
        "horizontal_dist_median":[695.0,     715.0],
        "vertical_dist":         [50.0,      100.0],
        "dist3d":                [701.8,     724.9],
        "overlap_len_common_ft": [5_000.0,   5_200.0],
        "LL_i":                  [6_000.0,   6_000.0],
        "LL_k":                  [6_000.0,   6_000.0],
        "tvd_i":                 [10_800.0,  10_850.0],
        "tvd_k":                 [10_850.0,  10_900.0],
        "overlap_pct_i":         [0.83,      0.87],
        "overlap_pct_k":         [0.83,      0.87],
        "n_samples":             [50,        50],
        "dy_p5":                 [680.0,     700.0],
        "angle_deg":             [1.2,       1.5],
        "pair_alignment":        ["parallel_like", "parallel_like"],
        "min_distance_ft":       [float("nan"), float("nan")],
        "mean_windowed_ft":      [float("nan"), float("nan")],
        "reject_reason":         ["", ""],
        "direction_axis":        ["NS", "NS"],
        "direction_to_k_from_i_axis": ["N", "N"],
        "direction_axis_confidence": [0.95, 0.94],
        "direction_axis_distribution": ["N:0.95,S:0.05", "N:0.94,S:0.06"],
        "drill_direction_i":     ["EW", "EW"],
        "drill_direction_k":     ["EW", "EW"],
        "axis_forced":           [True, True],
        "elevation_i":           [-10_800.0, -10_850.0],
        "bench":                 ["WOLFCAMP A", "WOLFCAMP A"],
    })


@pytest.fixture(scope="session")
def heeltoe_df(well_uwis: list[str]) -> pd.DataFrame:
    """HeelToe midpoint DataFrame: uwi, mid_Lat, mid_Lon."""
    return pd.DataFrame({
        "uwi":     well_uwis,
        "mid_Lat": [31.510, 31.516, 31.522],
        "mid_Lon": [-101.880, -101.874, -101.868],
    })


@pytest.fixture(scope="session")
def spacing_result_stub():
    """One valid SpacingResult instance for dataclass field tests."""
    from src.well_data.well_spacing_stats import SpacingResult
    return SpacingResult(
        well_i="42-329-40001-0000",
        well_k="42-329-40002-0000",
        horizontal_dist=700.0,
        horizontal_dist_median=695.0,
        vertical_dist=50.0,
        dist3d=701.8,
        overlap_len_common_ft=5_000.0,
        LL_i=6_000.0,
        LL_k=6_000.0,
        tvd_i=10_800.0,
        tvd_k=10_850.0,
        overlap_pct_i=0.83,
        overlap_pct_k=0.83,
        n_samples=50,
        dy_p5=680.0,
        angle_deg=1.2,
        pair_alignment="parallel_like",
        min_distance_ft=float("nan"),
        mean_windowed_ft=float("nan"),
        reject_reason="",
        direction_axis="NS",
        direction_to_k_from_i_axis="N",
        direction_axis_confidence=0.95,
        direction_axis_distribution="N:0.95,S:0.05",
        drill_direction_i="EW",
        drill_direction_k="EW",
    )
