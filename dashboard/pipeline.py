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
    FloatingSectionWPS,
    BoxSpec,
    CorridorSpec,
)
from src.well_data.well_role_assignment import OverlappingNeighborhoodRoles
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
        # Build dtype dict — separate datetime columns (pd.read_csv dtype doesn't accept "datetime")
        prod_dtypes = {}
        prod_date_cols = []
        if dtype_map_production:
            for col, dt in dtype_map_production.items():
                if dt in ("datetime", "datetime64", "datetime64[ns]"):
                    prod_date_cols.append(col)
                else:
                    prod_dtypes[col] = dt
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
        for col in prod_date_cols:
            if col in raw.columns:
                raw[col] = pd.to_datetime(raw[col], errors="coerce")
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
    production_df: pd.DataFrame | None = None,
    directional_df: pd.DataFrame | None = None,
    # Advanced spacing engine params
    step_ft: int = 100,
    max_crossline_ft: float | None = 2000.0,
    crossline_percentile: float = 5.0,
    theta_parallel_deg: float = 25.0,
    theta_perp_deg: float = 65.0,
    contact_threshold_ft: float = 300.0,
    use_pca_axis: bool = True,
    emit_rejected: bool = True,
    reject_misaligned: bool = False,
    # Role assignment params
    role_parallel_eps_ft: float = 1_320,
    role_perp_eps_ft: float = 1_000,
    role_oblique_eps_ft: float = 1_100,
    role_cross_bench_eps_ft: float = 1_320,
    role_cross_bench_vertical_gate_ft: float = 200,
    role_inner_zone_ft: float = 660,
    role_child_window_months: float = 18,
    role_infill_min_older: int = 2,
    role_treat_no_neighbor_as_parent: bool = False,
    role_pair_types: dict[str, bool] | None = None,
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
        step_ft=step_ft,
        max_crossline_ft=max_crossline_ft,
        crossline_percentile=crossline_percentile,
        theta_parallel_deg=theta_parallel_deg,
        theta_perp_deg=theta_perp_deg,
        contact_threshold_ft=contact_threshold_ft,
        use_pca_axis=use_pca_axis,
        emit_rejected=emit_rejected,
        reject_misaligned=reject_misaligned,
    )
    total_pairs = len(df_spacing)
    stats.record_independent("Spacing pairs computed (raw)", total_pairs, unit="pairs")
    logger.info("Spacing stats done: %d pairs in %.1fs", total_pairs, time.perf_counter() - t0)

    # Preserve ALL pairs (including rejected) for optional export
    df_spacing_raw = df_spacing.copy()

    # --- Filter out rejected pairs (matches notebook) ---
    if "reject_reason" in df_spacing.columns:
        valid_mask = (df_spacing["reject_reason"] == "") | df_spacing["reject_reason"].isna()
        rejected_count = (~valid_mask).sum()
        df_spacing = df_spacing[valid_mask].copy()
        stats.record_independent("Valid pairs (reject_reason filtered)", len(df_spacing), unit="pairs")
        if rejected_count:
            logger.info("Reject filter: removed %d pairs → %d valid remain", rejected_count, len(df_spacing))

    # --- Run role assignment (V2 — all pair types) ---
    logger.info("Running overlapping neighborhood role assignment...")
    t_roles = time.perf_counter()
    df_well_roles = assign_well_roles(
        df_spacing_raw, header_df,
        parallel_eps_ft=role_parallel_eps_ft,
        perp_eps_ft=role_perp_eps_ft,
        oblique_eps_ft=role_oblique_eps_ft,
        cross_bench_eps_ft=role_cross_bench_eps_ft,
        cross_bench_vertical_gate_ft=role_cross_bench_vertical_gate_ft,
        inner_zone_ft=role_inner_zone_ft,
        child_window_months=role_child_window_months,
        infill_min_older=role_infill_min_older,
        treat_no_neighbor_as_parent=role_treat_no_neighbor_as_parent,
        role_pair_types=role_pair_types,
        logger=logger,
    )
    logger.info("Role assignment done in %.1fs — %d wells assigned",
                time.perf_counter() - t_roles, len(df_well_roles))

    # Merge role into header_df so it's available as a GeoJSON property on the map
    role_cols = ["uwi", "role"]
    header_df = header_df.merge(
        df_well_roles[role_cols], on="uwi", how="left",
    )
    header_df["role"] = header_df["role"].fillna("no_eligible_neighbor")

    # Cache to disk (include stats)
    # NOTE: DirectionalBenchNeighbors, AvgSpacingCalculator, and FloatingSectionWPS
    # are now run on-demand from the Explore page (not during batch calculation).
    cache_key = PIPELINE_CACHE_DIR / f"pipeline_{rid}.pkl"
    with open(cache_key, "wb") as f:
        pickle.dump(
            {
                "df_ik_pairs": df_spacing,        # reject-filtered valid IK pairs
                "df_ik_pairs_raw": df_spacing_raw,  # ALL IK pairs (including rejected)
                "header_df": header_df,            # now includes 'role' column
                "lateral_df": lateral_df,
                "directional_df": directional_df, # full survey (for map trajectories)
                "production_df": production_df,   # may be None if not uploaded
                "df_well_roles": df_well_roles,   # full role table (parent info, confidence, etc.)
                "stats": stats.to_list(),
            },
            f,
        )
    logger.info("=== Spacing calculation DONE | total=%.1fs | cache=%s ===",
                time.perf_counter() - t0, cache_key.name)

    return str(cache_key)


# ---------------------------------------------------------------------------
# Role assignment (V2 — overlapping neighborhoods)
# ---------------------------------------------------------------------------

def assign_well_roles(
    df_ik_pairs_raw: pd.DataFrame,
    header_df: pd.DataFrame,
    *,
    parallel_eps_ft: float = 1_320,
    perp_eps_ft: float = 1_000,
    oblique_eps_ft: float = 1_100,
    cross_bench_eps_ft: float = 1_320,
    cross_bench_vertical_gate_ft: float = 200,
    inner_zone_ft: float = 660,
    child_window_months: float = 18,
    infill_min_older: int = 2,
    treat_no_neighbor_as_parent: bool = False,
    role_pair_types: dict[str, bool] | None = None,
    logger: Any = None,
) -> pd.DataFrame:
    """
    Assign parent / child / infill_candidate roles to every well.

    Wraps OverlappingNeighborhoodRoles from src/. Uses the raw IK pairs
    (including rejected) so the class can filter internally.

    Returns one-row-per-well DataFrame with role, parent info, confidence, etc.
    """
    roles_engine = OverlappingNeighborhoodRoles(logger=logger)
    return roles_engine.assign_roles(
        pairs_df=df_ik_pairs_raw,
        header_df=header_df,
        parallel_eps_ft=parallel_eps_ft,
        perp_eps_ft=perp_eps_ft,
        oblique_eps_ft=oblique_eps_ft,
        cross_bench_eps_ft=cross_bench_eps_ft,
        cross_bench_vertical_gate_ft=cross_bench_vertical_gate_ft,
        inner_zone_ft=inner_zone_ft,
        child_window_months=child_window_months,
        infill_min_older=infill_min_older,
        treat_no_neighbor_as_parent=treat_no_neighbor_as_parent,
        role_pair_types=role_pair_types,
    )


# ---------------------------------------------------------------------------
# Cache loaders (used by callbacks in Explore step)
# ---------------------------------------------------------------------------

def load_cached_pipeline(cache_path: str) -> dict[str, pd.DataFrame]:
    """Load cached pipeline output from disk."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_ui_state(cache_path: str, ui_state: dict) -> None:
    """Persist UI state (colors, settings, field selections) into the cache pickle.

    Loads the existing cache, updates the 'ui_state' key, and writes back.
    This is called on every UI settings change so the user never loses state.
    """
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        data["ui_state"] = ui_state
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass  # silently ignore — UI state is non-critical


def load_ui_state(cache_path: str) -> dict:
    """Load saved UI state from the cache pickle, if any."""
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data.get("ui_state", {})
    except Exception:
        return {}


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
    # Prefer raw IK pairs (well_i, well_k); fall back to enriched summary
    df_ik = data.get("df_ik_pairs")
    if df_ik is None:
        df_ik = data["df_spacing"]
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

    return df_ik, heel_toe


# ---------------------------------------------------------------------------
# Gun barrel data preparation
# ---------------------------------------------------------------------------

def compute_gun_barrel(
    IK: pd.DataFrame,
    HeelToe: pd.DataFrame,
    header_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute gun barrel positioning data.

    Replicates the Spotfire GB Python data function exactly.

    Args:
        IK: Filtered spacing pairs DataFrame (well_i in selected wells).
            Required columns: well_i, well_k, horizontal_dist, vertical_dist,
            dist3d (or 3D_dist), elevation_i, drill_direction_i
        HeelToe: Midpoint data. Required columns: uwi, mid_Lat, mid_Lon
        header_df: Optional well header for enriching GB with bench, operator, etc.

    Returns:
        GB DataFrame with cum_dist and sectionDist for x-axis positioning.
        Sorted W→E (NS wells) or S→N (EW wells).
    """
    if IK.empty:
        return pd.DataFrame()

    HeelToe = HeelToe.copy()
    HeelToe["mid_Lat"] = np.round(HeelToe["mid_Lat"], 9)
    HeelToe["mid_Lon"] = np.round(HeelToe["mid_Lon"], 9)

    # One row per unique well_i (core columns from IK)
    base_cols = ["well_i", "elevation_i", "drill_direction_i"]
    GB = (
        IK.drop_duplicates(subset=["well_i"])
        [base_cols]
        .copy()
    )

    # Enrich with header data (bench, well_name, operator, etc.)
    if header_df is not None and not header_df.empty and "uwi" in header_df.columns:
        hdr_cols = [c for c in header_df.columns if c not in GB.columns and c != "uwi"]
        if hdr_cols:
            GB = GB.merge(
                header_df[["uwi"] + hdr_cols].rename(columns={"uwi": "well_i"}),
                on="well_i", how="left",
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
    if not drill_dir.empty and drill_dir.iloc[0] == "NS":
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


# ---------------------------------------------------------------------------
# On-demand analysis functions (called from Explore page, NOT batch calc)
# ---------------------------------------------------------------------------

def run_directional_bench_neighbors(
    df_ik_pairs: pd.DataFrame,
    header_df: pd.DataFrame,
    *,
    cutoff_ft: float,
    vertical_cutoff_ft: float | None = None,
    overlap_pct_k_min: float | None = None,
    axis_mode: str = "any",
    prefer_axis: str | None = None,
    proj_coverage_min: float | None = None,
    overrides_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run DirectionalBenchNeighbors.summarize() with full parameters.

    Called on-demand from the Explore page Analysis tab.
    """
    t0 = time.perf_counter()
    neighbors = DirectionalBenchNeighbors(logger=logger)
    result = neighbors.summarize(
        spacing_df=df_ik_pairs,
        header_df=header_df,
        cutoff_ft=cutoff_ft,
        vertical_cutoff_ft=vertical_cutoff_ft,
        overlap_pct_k_min=overlap_pct_k_min,
        axis_mode=axis_mode,
        prefer_axis=prefer_axis,
        proj_coverage_min=proj_coverage_min,
        overrides_df=overrides_df,
    )
    logger.info("DirectionalBenchNeighbors done: %d rows in %.1fs",
                len(result), time.perf_counter() - t0)
    return result


def run_avg_spacing(
    df_ik_pairs: pd.DataFrame,
    lateral_df: pd.DataFrame,
    *,
    cutoff_ft: float,
    vertical_cutoff_ft: float | None = None,
    overlap_pct_k_min: float | None = None,
    overrides_df: pd.DataFrame | None = None,
    axis_mode: str = "any",
    neighborhood_mode: str = "chain",
    chain_sort_mode: str = "pca",
    edge_pick: str = "min",
) -> pd.DataFrame:
    """Run AvgSpacingCalculator.summarize() with full parameters.

    Called on-demand from the Explore page Analysis tab.
    """
    t0 = time.perf_counter()
    calc = AvgSpacingCalculator(logger=logger)
    result = calc.summarize(
        spacing_df=df_ik_pairs,
        cutoff_ft=cutoff_ft,
        vertical_cutoff_ft=vertical_cutoff_ft,
        overlap_pct_k_min=overlap_pct_k_min,
        overrides_df=overrides_df,
        axis_mode=axis_mode,
        neighborhood_mode=neighborhood_mode,
        chain_sort_mode=chain_sort_mode,
        trajectories=lateral_df,
        edge_pick=edge_pick,
    )
    logger.info("AvgSpacingCalculator done: %d rows in %.1fs",
                len(result), time.perf_counter() - t0)
    return result


def run_floating_wps(
    lateral_df: pd.DataFrame,
    *,
    box_half_width_ft: float = 2640.0,
    box_half_height_ft: float = 2640.0,
    corridor_half_width_ft: float = 2640.0,
    corridor_extra_along_ft: float = 0.0,
    min_inside_ft: float = 660.0,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Run FloatingSectionWPS.summarize_per_well().

    Called on-demand from the Explore page Analysis tab.
    """
    t0 = time.perf_counter()
    wells_df = FloatingSectionWPS.ds_to_lateral_endpoints(lateral_df)
    box = BoxSpec(
        half_width_ft=box_half_width_ft,
        half_height_ft=box_half_height_ft,
    )
    corridor = CorridorSpec(
        half_width_ft=corridor_half_width_ft,
        extra_along_ft=corridor_extra_along_ft,
    )
    wps = FloatingSectionWPS(
        wells_df=wells_df,
        box=box,
        min_inside_ft=min_inside_ft,
        exclude_self=exclude_self,
        corridor=corridor,
        logger=logger,
    )
    result = wps.summarize_per_well()
    logger.info("FloatingSectionWPS done: %d rows in %.1fs",
                len(result), time.perf_counter() - t0)
    return result


# ---------------------------------------------------------------------------
# Phase B: Diagnostic visualization wrappers
# ---------------------------------------------------------------------------

def run_debug_pair_spacing(
    lateral_df: pd.DataFrame,
    header_df: pd.DataFrame,
    uwi_i: str,
    uwi_k: str,
    *,
    step_ft: int = 100,
    use_pca_axis: bool = True,
    theta_parallel_deg: float = 25.0,
    theta_perp_deg: float = 65.0,
    contact_threshold_ft: float = 300.0,
) -> tuple[dict, "list[plt.Figure]"]:
    """Run WellSpacingCalculator.debug_pair_spacing() and capture figures.

    Returns:
        (metrics_dict, list_of_matplotlib_figures)
    """
    import matplotlib.pyplot as plt
    import matplotlib.figure

    plt.close("all")  # start clean

    # debug_pair_spacing calls plt.close(fig) after each panel,
    # so intercept close to capture figures before they're destroyed.
    captured_figs: list[plt.Figure] = []
    _original_close = plt.close

    def _intercept_close(fig_or_num=None):
        if fig_or_num is not None and fig_or_num != "all":
            if isinstance(fig_or_num, matplotlib.figure.Figure):
                captured_figs.append(fig_or_num)
                return  # keep figure alive
        _original_close(fig_or_num)

    plt.close = _intercept_close
    try:
        calculator = WellSpacingCalculator(
            trajectories=lateral_df,
            header_df=header_df,
            logger=logger,
        )
        metrics = calculator.debug_pair_spacing(
            uwi_i=uwi_i,
            uwi_k=uwi_k,
            step_ft=step_ft,
            use_pca_axis=use_pca_axis,
            theta_parallel_deg=theta_parallel_deg,
            theta_perp_deg=theta_perp_deg,
            contact_threshold_ft=contact_threshold_ft,
            show=False,       # don't call plt.show() in server
            save_dir=None,    # don't save to disk
        )
    finally:
        plt.close = _original_close

    # Use captured figures; fall back to any still-open figures
    figs = captured_figs or [plt.figure(n) for n in plt.get_fignums()]
    return metrics, figs


def run_wps_visualize(
    lateral_df: pd.DataFrame,
    uwi_ref: str,
    *,
    orientation: str = "cardinal",
    box_half_width_ft: float = 2640.0,
    box_half_height_ft: float = 2640.0,
    corridor_half_width_ft: float = 2640.0,
    corridor_extra_along_ft: float = 0.0,
    min_inside_ft: float = 660.0,
    exclude_self: bool = True,
    figsize: tuple[float, float] = (10.0, 10.0),
) -> "plt.Figure":
    """Run FloatingSectionWPS.visualize() for a single well.

    Returns:
        matplotlib Figure showing the floating section + neighbors.
    """
    wells_df = FloatingSectionWPS.ds_to_lateral_endpoints(lateral_df)
    box = BoxSpec(
        half_width_ft=box_half_width_ft,
        half_height_ft=box_half_height_ft,
    )
    corridor = CorridorSpec(
        half_width_ft=corridor_half_width_ft,
        extra_along_ft=corridor_extra_along_ft,
    )
    wps = FloatingSectionWPS(
        wells_df=wells_df,
        box=box,
        min_inside_ft=min_inside_ft,
        exclude_self=exclude_self,
        corridor=corridor,
        logger=logger,
    )
    fig = wps.visualize(
        uwi_ref=uwi_ref,
        orientation=orientation,
        figsize=figsize,
    )
    return fig


def run_avg_spacing_plot(
    df_ik_pairs: pd.DataFrame,
    lateral_df: pd.DataFrame,
    well_i: str,
    *,
    cutoff_ft: float,
    vertical_cutoff_ft: float | None = None,
    overlap_pct_k_min: float | None = None,
    overrides_df: pd.DataFrame | None = None,
    axis_mode: str = "any",
    neighborhood_mode: str = "chain",
    chain_sort_mode: str = "pca",
    edge_pick: str = "min",
    plot_mode: str = "both",
    figsize: tuple[float, float] = (12.5, 8.0),
) -> "plt.Figure":
    """Run AvgSpacingCalculator.summarize() then plot_neighborhood() for one well.

    Must run summarize() first (plot_neighborhood requires it).

    Returns:
        matplotlib Figure showing the neighborhood diagnostic.
    """
    import matplotlib.pyplot as plt

    calc = AvgSpacingCalculator(logger=logger)
    calc.summarize(
        spacing_df=df_ik_pairs,
        cutoff_ft=cutoff_ft,
        vertical_cutoff_ft=vertical_cutoff_ft,
        overlap_pct_k_min=overlap_pct_k_min,
        overrides_df=overrides_df,
        axis_mode=axis_mode,
        neighborhood_mode=neighborhood_mode,
        chain_sort_mode=chain_sort_mode,
        trajectories=lateral_df,
        edge_pick=edge_pick,
    )
    plt.close("all")
    # plot_neighborhood() calls plt.close(fig) internally, so we
    # intercept close to capture the figure before it's destroyed.
    captured_figs = []
    _original_close = plt.close

    def _intercept_close(fig_or_num=None):
        if fig_or_num is not None and fig_or_num != "all":
            import matplotlib.figure
            if isinstance(fig_or_num, matplotlib.figure.Figure):
                captured_figs.append(fig_or_num)
                return  # don't actually close — we need it
        _original_close(fig_or_num)

    plt.close = _intercept_close
    try:
        calc.plot_neighborhood(
            well_i=well_i,
            trajectories=lateral_df,
            mode=plot_mode,
            show=False,
            figsize=figsize,
        )
    finally:
        plt.close = _original_close

    if captured_figs:
        return captured_figs[-1]
    # Fallback: check if any figures are still open
    figs = plt.get_fignums()
    if figs:
        return plt.figure(figs[-1])
    raise ValueError(f"No plot generated for well {well_i}")
