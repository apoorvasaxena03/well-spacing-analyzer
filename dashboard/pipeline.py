"""
dashboard/pipeline.py — thin wrapper around src/ library.

This module is the ONLY place that imports from src/.
All dashboard callbacks call functions here — never src/ directly.

The src/ library is never modified for dashboard purposes.
If a bug is found in src/, fix it there. If a new feature is needed,
add it to src/ first, then expose it here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.well_data.well_data_manager import WellDataLoader, GeoSurveyProcessor
from src.well_data.well_spacing_stats import (
    WellSpacingCalculator,
    DirectionalBenchNeighbors,
    SpacingNeighborEnricher,
)

# ---------------------------------------------------------------------------
# Cache directory for pipeline outputs
# ---------------------------------------------------------------------------
PIPELINE_CACHE_DIR = Path("./dashboard/.pipeline_cache")
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_files(
    header_path: str,
    directional_path: str,
    column_map_header: dict[str, str],
    column_map_directional: dict[str, str],
    production_path: str | None = None,
    column_map_production: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load well data from uploaded CSV/Excel files using src/ WellDataLoader.

    Args:
        header_path: Path to header CSV/Excel file.
        directional_path: Path to directional survey CSV/Excel file.
        column_map_header: {source_col: canonical_col} mapping for header.
        column_map_directional: {source_col: canonical_col} mapping for directional.
        production_path: Optional path to production CSV/Excel file.
        column_map_production: Optional mapping for production file.

    Returns:
        dict with keys: 'header_df', 'directional_df', 'production_df' (or None)
    """
    loader = WellDataLoader(
        source=header_path,
        header_column_map=column_map_header,
        directional_source=directional_path,
        directional_column_map=column_map_directional,
    )
    header_df, directional_df = loader.load()

    production_df = None
    if production_path and column_map_production:
        prod_loader = WellDataLoader(
            source=production_path,
            header_column_map=column_map_production,
        )
        production_df, _ = prod_loader.load()

    return {
        "header_df": header_df,
        "directional_df": directional_df,
        "production_df": production_df,
    }


def load_from_database(
    header_query: str,
    directional_query: str,
    db_config: dict[str, Any],
    column_map_header: dict[str, str],
    column_map_directional: dict[str, str],
    production_query: str | None = None,
    column_map_production: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load well data from a database using src/ WellDataLoader with SQL queries.

    Args:
        header_query: SQL SELECT for header data.
        directional_query: SQL SELECT for directional survey data.
        db_config: Connection config dict passed to WellDataLoader.
        column_map_header: {source_col: canonical_col} for header.
        column_map_directional: {source_col: canonical_col} for directional.
        production_query: Optional SQL SELECT for production data.
        column_map_production: Optional mapping for production.

    Returns:
        dict with keys: 'header_df', 'directional_df', 'production_df' (or None)
    """
    loader = WellDataLoader(
        header_query=header_query,
        directional_query=directional_query,
        db_config=db_config,
        header_column_map=column_map_header,
        directional_column_map=column_map_directional,
    )
    header_df, directional_df = loader.load()

    production_df = None
    if production_query and column_map_production:
        prod_loader = WellDataLoader(
            header_query=production_query,
            db_config=db_config,
            header_column_map=column_map_production,
        )
        production_df, _ = prod_loader.load()

    return {
        "header_df": header_df,
        "directional_df": directional_df,
        "production_df": production_df,
    }


# ---------------------------------------------------------------------------
# Projection + lateral extraction
# ---------------------------------------------------------------------------

def project_and_extract_laterals(
    header_df: pd.DataFrame,
    directional_df: pd.DataFrame,
    crs_to: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Project lat/lon to UTM and extract lateral (horizontal) sections only.

    UTM zone is auto-detected from the centroid of surface locations when
    crs_to is None — no user input needed.

    Returns:
        (header_df, lateral_df, crs_used)
    """
    if crs_to is None:
        crs_to = _auto_detect_utm(header_df)

    processor = GeoSurveyProcessor(crs_to=crs_to)
    header_df, lateral_df = processor.process(header_df, directional_df)
    return header_df, lateral_df, crs_to


def _auto_detect_utm(header_df: pd.DataFrame) -> str:
    """Detect UTM zone from centroid of surface lat/lon."""
    lat = header_df["latitude"].dropna().mean()
    lon = header_df["longitude"].dropna().mean()
    zone = int((lon + 180) / 6) + 1
    hemisphere = "326" if lat >= 0 else "327"
    return f"EPSG:{hemisphere}{zone:02d}"


# ---------------------------------------------------------------------------
# Spacing calculation (designed to run in dash.long_callback)
# ---------------------------------------------------------------------------

def run_spacing_calculation(
    header_df: pd.DataFrame,
    lateral_df: pd.DataFrame,
    max_distance_miles: float = 4.0,
    cutoff_ft: float = 5280.0,
    batch_size: int = 200_000,
    run_id: str | None = None,
) -> str:
    """
    Run the full spacing pipeline and cache results to disk.

    Designed to be called from a dash.long_callback background job.
    Returns a cache key (file path string) — never the DataFrame itself
    (too large for dcc.Store).

    Args:
        header_df: Processed header DataFrame.
        lateral_df: Lateral-only directional survey DataFrame.
        max_distance_miles: Pre-filter spatial radius.
        cutoff_ft: Maximum spacing distance to include in results.
        batch_size: Pairs per batch (memory control).
        run_id: Optional run identifier for logging.

    Returns:
        Path to cached pipeline output (pickle file).
    """
    calculator = WellSpacingCalculator(
        header_df=header_df,
        directional_df=lateral_df,
    )
    df_spacing = calculator._calculate_spacing_statistics(
        batch_size=batch_size,
        max_distance_miles=max_distance_miles,
        cutoff_ft=cutoff_ft,
    )

    neighbors = DirectionalBenchNeighbors(
        df_spacing=df_spacing,
        header_df=header_df,
    )
    df_neighbors = neighbors.identify()

    enricher = SpacingNeighborEnricher(
        df_spacing=df_spacing,
        df_neighbors=df_neighbors,
        header_df=header_df,
    )
    df_enriched = enricher.enrich()

    # Cache to disk
    cache_key = PIPELINE_CACHE_DIR / f"pipeline_{run_id or 'latest'}.pkl"
    with open(cache_key, "wb") as f:
        pickle.dump(
            {
                "df_spacing": df_enriched,
                "header_df": header_df,
                "lateral_df": lateral_df,
            },
            f,
        )

    return str(cache_key)


# ---------------------------------------------------------------------------
# Cache loaders (used by callbacks in Explore step)
# ---------------------------------------------------------------------------

def load_cached_pipeline(cache_path: str) -> dict[str, pd.DataFrame]:
    """Load cached pipeline output from disk."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_cached_ik_heeltoe(
    pipeline_result: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IK spacing pairs and HeelToe midpoints from cached pipeline output.

    Returns empty DataFrames if cache is not yet available.
    """
    if not pipeline_result or not pipeline_result.get("cache_path"):
        return pd.DataFrame(), pd.DataFrame()

    data = load_cached_pipeline(pipeline_result["cache_path"])
    df_spacing = data["df_spacing"]
    header_df = data["header_df"]
    lateral_df = data["lateral_df"]

    # HeelToe: midpoint of each well's lateral (mid_Lat, mid_Lon)
    heel_toe = (
        lateral_df.groupby("uwi")
        .agg(
            mid_Lat=("latitude", "mean"),
            mid_Lon=("longitude", "mean"),
        )
        .reset_index()
    )

    return df_spacing, heel_toe


# ---------------------------------------------------------------------------
# Gun barrel data preparation
# ---------------------------------------------------------------------------

def compute_gun_barrel(
    IK: pd.DataFrame,
    HeelToe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute gun barrel positioning data.

    Replicates the Spotfire GB Python data function exactly.

    Args:
        IK: Filtered spacing pairs DataFrame (intra-neighborhood only).
            Required columns: well_i, well_k, horizontal_dist, vertical_dist,
            dist3d (or 3D_dist), elevation_i, drill_direction_i
        HeelToe: Midpoint data. Required columns: uwi, mid_Lat, mid_Lon

    Returns:
        GB DataFrame with cum_dist and sectionDist for x-axis positioning.
        Sorted W→E (NS wells) or S→N (EW wells).
    """
    if IK.empty:
        return pd.DataFrame()

    HeelToe = HeelToe.copy()
    HeelToe["mid_Lat"] = np.round(HeelToe["mid_Lat"], 9)
    HeelToe["mid_Lon"] = np.round(HeelToe["mid_Lon"], 9)

    # Unique well_i entries
    GB = (
        IK.drop_duplicates(subset=["well_i"])
        [["well_i", "elevation_i", "drill_direction_i"]]
        .copy()
    )

    # Join heel/toe midpoints
    GB = pd.merge(
        GB,
        HeelToe.rename(columns={"uwi": "well_i"}),
        how="left",
        on="well_i",
    ).reset_index(drop=True)

    # Sort wells for gun barrel positioning
    drill_dir = GB["drill_direction_i"].mode()
    if not drill_dir.empty and drill_dir.item() == "NS":
        GB["mid_Lon"] = np.round(GB["mid_Lon"], 9)
        GB = GB.sort_values("mid_Lon").reset_index(drop=True)
        GB["E_to_W_Rank"] = GB.index + 1
    else:
        GB["mid_Lat"] = np.round(GB["mid_Lat"], 9)
        GB = GB.sort_values("mid_Lat").reset_index(drop=True)
        GB["N_to_S_Rank"] = GB.index + 1

    if len(GB) == 1:
        GB["horizontal_dist"] = 0.0
        GB["cum_dist"] = 0.0
    else:
        # Normalise 3D_dist column name
        dist3d_col = "3D_dist" if "3D_dist" in IK.columns else "dist3d"

        GB["next_i_uwi"] = GB["well_i"].shift(-1)
        GB = GB.merge(
            IK[["well_i", "well_k", "horizontal_dist", "vertical_dist", dist3d_col]],
            left_on=["well_i", "next_i_uwi"],
            right_on=["well_i", "well_k"],
            how="left",
        )
        # First well at x=0; each subsequent well = cumulative horizontal_dist
        GB["cum_dist"] = GB["horizontal_dist"].shift(1, fill_value=0).cumsum()

    # Centered axis option (sectionDist = 0 at midpoint of the group)
    max_cum = GB["cum_dist"].max()
    GB["sectionDist"] = GB["cum_dist"] - (max_cum / 2 if max_cum else 0)

    return GB
