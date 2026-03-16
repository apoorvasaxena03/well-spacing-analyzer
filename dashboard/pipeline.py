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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.well_data.well_data_manager import WellDataLoader, GeoSurveyProcessor
from src.well_data.well_spacing_stats import (
    WellSpacingCalculator,
    DirectionalBenchNeighbors,
    AvgSpacingCalculator,
)
from src.utils.custom_logger import get_logger, new_run_id, set_run_id
from src.utils.utils import drop_uwi_duplicates_keep_max_last_prod, compute_rsv_cat

# ---------------------------------------------------------------------------
# Module-level logger — writes to logs/dashboard.log + terminal
# ---------------------------------------------------------------------------
logger = get_logger("dashboard", log_to_console=True)
logger.info("Dashboard pipeline ready — logging to logs/dashboard.log")

# ---------------------------------------------------------------------------
# Cache directory for pipeline outputs
# ---------------------------------------------------------------------------
PIPELINE_CACHE_DIR = Path("./dashboard/.pipeline_cache")
PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline stats tracker — records well/pair counts at each step
# ---------------------------------------------------------------------------

class PipelineStats:
    """Accumulates step-by-step counts through the pipeline.

    Each step is recorded as (label, count, delta_from_previous).
    The final report is a list of dicts suitable for display in the UI.
    """

    def __init__(self):
        self._steps: list[dict] = []
        self._prev_count: int | None = None

    def record(self, label: str, count: int, *, unit: str = "wells"):
        delta = None
        if self._prev_count is not None:
            delta = count - self._prev_count
        self._steps.append({
            "label": label,
            "count": count,
            "delta": delta,
            "unit": unit,
        })
        self._prev_count = count

    def record_independent(self, label: str, count: int, *, unit: str = "wells"):
        """Record a count that is NOT part of the running funnel (no delta)."""
        self._steps.append({
            "label": label,
            "count": count,
            "delta": None,
            "unit": unit,
        })

    def to_list(self) -> list[dict]:
        return list(self._steps)


# ---------------------------------------------------------------------------
# RSV category col_map for canonical column names
# ---------------------------------------------------------------------------
_RSV_COL_MAP = {
    "status": "well_status",
    "last_prod": "last_prod_date",
    "first_prod": "first_prod_date",
    "spud": "spud_date",
    "comp": "comp_date",
    "permit_date": "permit_date",
}

# Default RSV categories to keep (matches notebook)
DEFAULT_RSV_KEEP = {"01PDP", "02PA", "02PDNP", "03PUD"}


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
    dtype_map_header: dict[str, str] | None = None,
    dtype_map_directional: dict[str, str] | None = None,
    dtype_map_production: dict[str, str] | None = None,
    directional_source: str | None = None,
    rsv_categories: list[str] | None = None,
    prod_cutoff_months: int = 6,
    duc_age_years: int = 3,
    permit_window_years: int = 2,
    stats: PipelineStats | None = None,
) -> dict[str, Any]:
    """
    Load well data from uploaded CSV/Excel files using src/ WellDataLoader.

    Pipeline steps (matching notebook):
      1. Load header → column map
      2. Deduplicate header by UWI (keep max last_prod_date)
      3. Compute RSV category (if status/spud columns present)
      4. Filter by RSV category (if rsv_categories provided)
      5. Load directional → column map → merge uwi (Enverus)
      6. Load production (optional)

    Returns:
        dict with keys: 'header_df', 'directional_df', 'production_df',
                        'directional_source', 'stats'
    """
    if stats is None:
        stats = PipelineStats()

    # --- Auto-detect directional source ---
    if directional_source is None:
        dir_canonical = set((column_map_directional or {}).values())
        if "uwi12" in dir_canonical and "uwi" not in dir_canonical:
            directional_source = "enverus"
        else:
            directional_source = "ihs"
        logger.info("Auto-detected directional_source=%s from column mapping", directional_source)

    # --- Step 1: Load header ---
    logger.info("Loading header from: %s", Path(header_path).name)
    loader = WellDataLoader(directional_source=directional_source)
    header_df = loader.get_header_data(
        source=header_path,
        column_map=column_map_header or None,
        dtype_map=dtype_map_header or None,
    )
    stats.record("Header loaded (raw)", len(header_df))
    logger.info("Header loaded: %d wells", len(header_df))

    # --- Step 2: Deduplicate header ---
    if "uwi" in header_df.columns and "last_prod_date" in header_df.columns:
        before = len(header_df)
        header_df = drop_uwi_duplicates_keep_max_last_prod(header_df)
        removed = before - len(header_df)
        stats.record("After UWI deduplication", len(header_df))
        if removed:
            logger.info("Deduplication: removed %d duplicate UWIs → %d remain", removed, len(header_df))
        else:
            logger.info("Deduplication: no duplicates found (%d wells)", len(header_df))
    else:
        logger.info("Skipping deduplication (missing 'uwi' or 'last_prod_date' column)")

    # --- Step 3: Compute RSV category ---
    has_rsv_cols = (
        "well_status" in header_df.columns
        and "spud_date" in header_df.columns
    )
    if has_rsv_cols:
        if "rsv_cat" not in header_df.columns:
            header_df["rsv_cat"] = compute_rsv_cat(
                header_df,
                col_map=_RSV_COL_MAP,
                prod_cutoff_months=prod_cutoff_months,
                duc_age_years=duc_age_years,
                permit_window_years=permit_window_years,
            )
            logger.info("RSV categorization computed (cutoffs: prod=%d mo, DUC=%d yr, permit=%d yr): %s",
                        prod_cutoff_months, duc_age_years, permit_window_years,
                        header_df["rsv_cat"].value_counts().to_dict())
        else:
            logger.info("RSV category already present: %s", header_df["rsv_cat"].value_counts().to_dict())
    else:
        logger.info("Skipping RSV categorization (missing 'well_status' or 'spud_date')")

    # --- Step 4: Filter by RSV category ---
    if rsv_categories and "rsv_cat" in header_df.columns:
        rsv_set = set(rsv_categories)
        before = len(header_df)
        header_df = header_df[header_df["rsv_cat"].isin(rsv_set)].copy()
        removed = before - len(header_df)
        stats.record("After RSV category filter", len(header_df))
        logger.info("RSV filter (keep %s): removed %d → %d remain",
                     sorted(rsv_set), removed, len(header_df))
    elif "rsv_cat" in header_df.columns:
        stats.record("RSV computed (no filter)", len(header_df))

    # --- Step 5: Load directional ---
    logger.info("Loading directional survey from: %s", Path(directional_path).name)
    directional_df = loader.get_directional_data(
        source=directional_path,
        column_map=column_map_directional or None,
        dtype_map=dtype_map_directional or None,
    )

    # Enverus merge: directional has uwi12 but not uwi
    if "uwi12" in directional_df.columns and "uwi" not in directional_df.columns:
        if "uwi12" in header_df.columns and "uwi" in header_df.columns:
            uwi_map = (
                header_df[["uwi", "uwi12"]]
                .sort_values("uwi")
                .drop_duplicates(subset=["uwi12"], keep="first")
            )
            directional_df = directional_df.merge(uwi_map, on="uwi12", how="left")
            logger.info("Merged uwi from header into directional via uwi12 (%d matched)",
                        directional_df["uwi"].notna().sum())

    uwi_col = "uwi" if "uwi" in directional_df.columns else "uwi12"
    dir_well_count = directional_df[uwi_col].nunique() if uwi_col in directional_df.columns else 0
    stats.record_independent("Directional surveys loaded", dir_well_count)
    logger.info("Directional loaded: %d survey stations across %d wells",
                len(directional_df), dir_well_count)

    # Report wells in header but missing from directional
    if "uwi" in header_df.columns and "uwi" in directional_df.columns:
        header_uwis = set(header_df["uwi"].dropna().unique())
        dir_uwis = set(directional_df["uwi"].dropna().unique())
        missing_dir = header_uwis - dir_uwis
        if missing_dir:
            stats.record_independent("Header wells missing directional data", len(missing_dir))
            logger.warning("%d wells in header have NO directional survey data", len(missing_dir))

    # --- Step 6: Load production (optional) ---
    production_df = None
    if production_path and column_map_production:
        logger.info("Loading production from: %s", Path(production_path).name)
        # Build dtype dict for UWI columns (always string)
        prod_dtypes = {}
        if dtype_map_production:
            prod_dtypes.update(dtype_map_production)
        for src, canon in column_map_production.items():
            if "uwi" in src.lower() or "api" in src.lower() or "uwi" in canon.lower():
                prod_dtypes.setdefault(src, str)
        suffix = Path(production_path).suffix.lower()
        if suffix == ".csv":
            raw = pd.read_csv(production_path, usecols=list(column_map_production.keys()),
                              dtype=prod_dtypes or None)
        else:
            raw = pd.read_excel(production_path, usecols=list(column_map_production.keys()),
                                dtype=prod_dtypes or None)
        production_df = raw.rename(columns=column_map_production)
        prod_wells = production_df["uwi"].nunique() if "uwi" in production_df.columns else 0
        stats.record_independent("Production data loaded", prod_wells)
        logger.info("Production loaded: %d rows, %d wells", len(production_df), prod_wells)

    return {
        "header_df": header_df,
        "directional_df": directional_df,
        "production_df": production_df,
        "directional_source": directional_source,
        "stats": stats,
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
    inclination_filter: float = 30.0,
    directional_source: str = "ihs",
    stats: PipelineStats | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Project lat/lon to UTM and extract lateral (horizontal) sections only.

    Uses GeoSurveyProcessor.prepare_lateral_trajectory_data() which:
      1. Computes UTM (x, y, z) in feet per survey station
      2. Filters to lateral section only (inclination > inclination_filter)
      3. Cross-aligns header and directional to the shared well set

    UTM zone is auto-detected from the centroid of surface locations when
    crs_to is None — layman users never need to know EPSG codes.

    Returns:
        (header_aligned, lateral_df, crs_used)
    """
    if stats is None:
        stats = PipelineStats()

    crs_used = crs_to or _auto_detect_utm(header_df)
    logger.info("UTM projection: %s (inclination filter: %.1f°)", crs_used, inclination_filter)

    wells_before = header_df["uwi"].nunique() if "uwi" in header_df.columns else len(header_df)

    processor = GeoSurveyProcessor(
        header_df=header_df,
        directional_df=directional_df,
        directional_source=directional_source,
        logger=logger,
    )
    lateral_df, header_aligned = processor.prepare_lateral_trajectory_data(
        inclination_filter=inclination_filter,
    )

    wells_after = lateral_df["uwi"].nunique() if "uwi" in lateral_df.columns else 0
    wells_lost = wells_before - wells_after
    stats.record("After UTM + lateral extraction", wells_after)
    if wells_lost > 0:
        logger.info("Lateral extraction: %d wells lost (no heel point or missing data) → %d remain",
                     wells_lost, wells_after)
    logger.info("Lateral extraction complete: %d stations, %d wells",
                len(lateral_df), wells_after)
    return header_aligned, lateral_df, crs_used


def _auto_detect_utm(header_df: pd.DataFrame) -> str:
    """Detect UTM zone from centroid of surface lat/lon.

    Checks for canonical column names in order of preference:
    surface_lat/surface_lon (header canonical) → latitude/longitude (directional).
    """
    lat_col = "surface_lat" if "surface_lat" in header_df.columns else "latitude"
    lon_col = "surface_lon" if "surface_lon" in header_df.columns else "longitude"
    lat = header_df[lat_col].dropna().mean()
    lon = header_df[lon_col].dropna().mean()
    zone = int((lon + 180) / 6) + 1
    hemisphere = "326" if lat >= 0 else "327"
    crs = f"EPSG:{hemisphere}{zone:02d}"
    logger.info("Auto-detected UTM zone: %s (centroid lat=%.4f, lon=%.4f)", crs, lat, lon)
    return crs


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
    stats: PipelineStats | None = None,
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
        stats: PipelineStats tracker.

    Returns:
        Path to cached pipeline output (pickle file).
    """
    if stats is None:
        stats = PipelineStats()

    # Set run_id so all log lines for this calculation are correlated
    rid = run_id or new_run_id()
    set_run_id(rid)

    n_wells = lateral_df["uwi"].nunique() if "uwi" in lateral_df.columns else "?"
    logger.info("=== Spacing calculation START | wells=%s | max_dist=%.1f mi | cutoff=%g ft | batch=%d ===",
                n_wells, max_distance_miles, cutoff_ft, batch_size)

    t0 = time.perf_counter()
    calculator = WellSpacingCalculator(
        trajectories=lateral_df,
        header_df=header_df,
        logger=logger,
    )
    df_spacing = calculator._calculate_spacing_statistics(
        batch_size=batch_size,
        max_distance_miles=max_distance_miles,
    )
    total_pairs = len(df_spacing)
    stats.record_independent("Spacing pairs computed (raw)", total_pairs, unit="pairs")
    logger.info("Spacing stats done: %d pairs in %.1fs", total_pairs, time.perf_counter() - t0)

    # --- Filter out rejected pairs (matches notebook) ---
    if "reject_reason" in df_spacing.columns:
        valid_mask = (df_spacing["reject_reason"] == "") | df_spacing["reject_reason"].isna()
        rejected_count = (~valid_mask).sum()
        df_spacing = df_spacing[valid_mask].copy()
        stats.record_independent("Valid pairs (reject_reason filtered)", len(df_spacing), unit="pairs")
        if rejected_count:
            logger.info("Reject filter: removed %d pairs → %d valid remain", rejected_count, len(df_spacing))

    t1 = time.perf_counter()
    neighbors = DirectionalBenchNeighbors(logger=logger)
    df_enriched = neighbors.summarize(
        spacing_df=df_spacing,
        header_df=header_df,
        cutoff_ft=cutoff_ft,
    )
    stats.record_independent("Neighbor-enriched wells", len(df_enriched), unit="wells")
    logger.info("Neighbor enrichment done: %d enriched rows in %.1fs",
                len(df_enriched), time.perf_counter() - t1)

    # Cache to disk (include stats)
    cache_key = PIPELINE_CACHE_DIR / f"pipeline_{rid}.pkl"
    with open(cache_key, "wb") as f:
        pickle.dump(
            {
                "df_spacing": df_enriched,
                "header_df": header_df,
                "lateral_df": lateral_df,
                "stats": stats.to_list(),
            },
            f,
        )
    logger.info("=== Spacing calculation DONE | total=%.1fs | cache=%s ===",
                time.perf_counter() - t0, cache_key.name)

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

    # Unique well_i entries — carry bench column if present (set by enrichment step)
    base_cols = ["well_i", "elevation_i", "drill_direction_i"]
    optional_cols = [c for c in ["bench", "well_name", "first_prod_date"] if c in IK.columns]
    GB = (
        IK.drop_duplicates(subset=["well_i"])
        [base_cols + optional_cols]
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
