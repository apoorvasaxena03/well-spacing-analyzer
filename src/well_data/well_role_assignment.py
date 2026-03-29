# %% Imports
from __future__ import annotations

import logging
import time
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd

from src.utils.custom_logger import get_logger


# ══════════════════════════════════════════════════════════════════════════════
#  Overlapping Neighborhood Role Assignment  (V2 — universal pair-type support)
# ══════════════════════════════════════════════════════════════════════════════


class OverlappingNeighborhoodRoles:
    """
    Assigns parent / child / infill_candidate roles to every well based on
    **overlapping neighborhoods** — each well's local view of who is nearby.

    Unlike hard clustering (DBSCAN, graph partitioning), neighborhoods
    *overlap*: well C can appear in both A's and E's neighborhoods without
    transitively linking A and E into one cluster.  This eliminates the
    chaining problem inherent in density-based clustering.

    ----------
    V2 Design — Universal Pair-Type Support
    ----------
    All five relationship types participate in role assignment by default.
    The user controls inclusion/exclusion per type via ``role_pair_types``.

    | rel_type              | Default | Distance metric          | Confidence |
    |-----------------------|---------|--------------------------|------------|
    | parallel_same_bench   | ON      | hz_effective (crossline) | high       |
    | cross_formation       | ON      | 3D_dist_effective (gated)| medium     |
    | cross_orientation     | ON      | hz_effective (min near.) | medium     |
    | oblique_same_bench    | ON      | hz_effective (min near.) | medium     |
    | oblique_cross_bench   | ON      | hz_effective (min near.) | low        |

    Cross-bench pairs (cross_formation, oblique_cross_bench) pass through an
    additional **vertical gate**: ``vertical_dist <= cross_bench_vertical_gate_ft``.
    The gate is user-configurable globally and per-well via ``overrides_df``,
    following the same pattern as ``DirectionalBenchNeighbors``.

    ----------
    Required Inputs
    ----------
    pairs_df : pd.DataFrame
        Pairwise spacing table (one row per ordered (i, k) pair) with columns:
            - well_i, well_k : str
            - bench_i, bench_k : str
            - hz_effective : float (ft) — alignment-aware horizontal distance
            - 3D_dist_effective : float (ft) — sqrt(hz^2 + vertical_dist^2)
            - vertical_dist : float (ft) — |tvd_i - tvd_k|
            - angle_deg : float
            - pair_alignment : str ('parallel_like' | 'perpendicular' | 'oblique')
            - LL_i, LL_k : float (lateral lengths)
            - overlap_pct_i, overlap_pct_k : float [0, 1]
            - tvd_i, tvd_k : float (ft) — absolute landing TVD per well
        Optional:
            - reject_reason : str — if present, rows with non-empty values are filtered out
            - min_distance_ft : float

    header_df : pd.DataFrame
        One row per well with:
            - uwi : str
            - comp_date : datetime — completion date (NaT if unknown)
            - bench : str
        Plus any extra columns to join onto the output (well_name, operator, etc.).

    ----------
    Parameters (at call time via ``assign_roles()``)
    ----------
    parallel_eps_ft : float, default 1320
        Max hz_effective for parallel_same_bench pairs.
    perp_eps_ft : float, default 1000
        Max hz_effective for cross_orientation (perpendicular) pairs.
    oblique_eps_ft : float, default 1100
        Max hz_effective for oblique pairs.
    cross_bench_eps_ft : float, default 1320
        Max distance for cross-bench parallel pairs.
    cross_bench_vertical_gate_ft : float, default 200
        Max vertical_dist for cross-bench pairs to qualify for role assignment.
    use_3d_for_cross_bench : bool, default True
        If True, cross_formation pairs compare 3D_dist_effective to the threshold;
        otherwise hz_effective.
    inner_zone_ft : float, default 660
        Wells within this distance are flagged as inner-zone (high severity).
    child_window_months : float, default 18
        Max months after parent completion to label as gen1_child (vs late_child).
    infill_min_older : int, default 2
        Minimum older eligible neighbors for infill_candidate classification.
    role_pair_types : dict[str, bool], optional
        Toggle which pair types participate in role assignment.
        Default: all True.
    overrides_df : pd.DataFrame, optional
        Per-well overrides. Recognized columns:
            - well_i (or uwi) : str — required identifier
            - parallel_eps_ft : float
            - cross_bench_vertical_gate_ft : float
            - cross_bench_eps_ft : float
            - perp_eps_ft : float
            - oblique_eps_ft : float

    ----------
    Output Columns
    ----------
    Identity: uwi
    Role: role, child_gen
    Parent info: parent_uwi, parent_dist_ft, parent_pair_type, parent_confidence,
                 parent_vertical_ft, child_above_parent, parent_comp_date, days_since_parent
    Neighborhood: n_eligible_neighbors, n_older_eligible, n_psb_neighbors,
                  n_cross_formation, n_cross_orientation, n_oblique,
                  n_inner_zone_neighbors, role_contributing_types, dominant_pair_type

    ----------
    Examples
    ----------
    >>> roles = OverlappingNeighborhoodRoles()
    >>> df_out = roles.assign_roles(
    ...     pairs_df, header_df,
    ...     parallel_eps_ft=1320,
    ...     cross_bench_vertical_gate_ft=200,
    ... )

    # Exclude perpendicular pairs from role assignment
    >>> df_out = roles.assign_roles(
    ...     pairs_df, header_df,
    ...     role_pair_types={'cross_orientation': False},
    ... )

    # Per-well vertical gate overrides
    >>> overrides = pd.DataFrame({
    ...     'well_i': ['42501367160100', '42501370590000'],
    ...     'cross_bench_vertical_gate_ft': [300.0, 100.0],
    ... })
    >>> df_out = roles.assign_roles(
    ...     pairs_df, header_df,
    ...     overrides_df=overrides,
    ... )
    """

    # --- Relationship type constants ---
    REL_PARALLEL_SAME_BENCH = "parallel_same_bench"
    REL_CROSS_FORMATION = "cross_formation"
    REL_CROSS_ORIENTATION = "cross_orientation"
    REL_OBLIQUE_SAME_BENCH = "oblique_same_bench"
    REL_OBLIQUE_CROSS_BENCH = "oblique_cross_bench"

    ALL_REL_TYPES: List[str] = [
        REL_PARALLEL_SAME_BENCH,
        REL_CROSS_FORMATION,
        REL_CROSS_ORIENTATION,
        REL_OBLIQUE_SAME_BENCH,
        REL_OBLIQUE_CROSS_BENCH,
    ]

    # Cross-bench types that need a vertical gate
    _CROSS_BENCH_TYPES: Set[str] = {REL_CROSS_FORMATION, REL_OBLIQUE_CROSS_BENCH}

    # --- Confidence mapping ---
    CONFIDENCE_MAP: Dict[str, str] = {
        REL_PARALLEL_SAME_BENCH: "high",
        REL_CROSS_FORMATION: "medium",
        REL_CROSS_ORIENTATION: "medium",
        REL_OBLIQUE_SAME_BENCH: "medium",
        REL_OBLIQUE_CROSS_BENCH: "low",
    }

    # --- Role constants ---
    ROLE_PARENT = "parent"
    ROLE_CHILD = "child"
    ROLE_INFILL = "infill_candidate"
    ROLE_NO_NEIGHBOR = "no_eligible_neighbor"

    # --- Default thresholds ---
    _DEFAULT_ROLE_PAIR_TYPES: Dict[str, bool] = {
        REL_PARALLEL_SAME_BENCH: True,
        REL_CROSS_FORMATION: True,
        REL_CROSS_ORIENTATION: True,
        REL_OBLIQUE_SAME_BENCH: True,
        REL_OBLIQUE_CROSS_BENCH: True,
    }

    def __init__(
        self,
        *,
        overrides_df: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Set up hierarchical logger using project's custom_logger convention.
        # If no logger is injected, create one via get_logger() (writes to logs/,
        # includes run_id, consistent format with all other modules).
        # If a parent logger IS injected, create a child logger that propagates
        # into the parent's handlers.
        if logger is None:
            self.logger = get_logger("overlapping_neighborhood_roles", log_to_console=True)
        else:
            self.logger = logging.getLogger(
                f"{logger.name}.overlapping_neighborhood_roles"
            )
            self.logger.propagate = True

        self.logger.info("Initializing OverlappingNeighborhoodRoles")

        self._overrides_df_default = overrides_df

        if overrides_df is not None:
            n_overrides = len(overrides_df)
            self.logger.info(
                f"Initialized with default overrides for {n_overrides:,} wells"
            )
        else:
            self.logger.debug("  No default overrides configured")

        # Intermediate results for inspection / diagnostics
        self._df_sym: Optional[pd.DataFrame] = None
        self._df_nbhd: Optional[pd.DataFrame] = None
        self._df_roles: Optional[pd.DataFrame] = None
        self._df_output: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _resolve_overrides(
        self, overrides_df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Return call-time overrides if provided, else fall back to init default."""
        return overrides_df if overrides_df is not None else self._overrides_df_default

    def _clear_intermediates(self) -> None:
        """Clear stored intermediate results from a previous run."""
        self._df_sym = None
        self._df_nbhd = None
        self._df_roles = None
        self._df_output = None

    @staticmethod
    def _symmetrize_pairs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a fully symmetric pair table.

        For each (i -> k) row, produce the reverse (k -> i) by swapping i/k
        columns, then concatenate and deduplicate on (well_i, well_k).
        """
        swap = {
            "well_i": "well_k",
            "well_k": "well_i",
            "bench_i": "bench_k",
            "bench_k": "bench_i",
            "LL_i": "LL_k",
            "LL_k": "LL_i",
            "overlap_pct_i": "overlap_pct_k",
            "overlap_pct_k": "overlap_pct_i",
            "tvd_i": "tvd_k",
            "tvd_k": "tvd_i",
        }
        sym_cols = [
            "hz_effective",
            "3D_dist_effective",
            "vertical_dist",
            "angle_deg",
            "pair_alignment",
            "min_distance_ft",
        ]
        keep = list(swap.keys()) + [c for c in sym_cols if c in df.columns]

        df_fwd = df[keep].copy()
        df_rev = df[keep].rename(columns=swap)

        df_sym = pd.concat([df_fwd, df_rev], ignore_index=True)
        df_sym = df_sym.drop_duplicates(subset=["well_i", "well_k"])
        return df_sym

    @staticmethod
    def _classify_rel_type(
        alignment: pd.Series, same_bench: pd.Series
    ) -> pd.Series:
        """Vectorized classification of neighbor relationship type."""
        rel = pd.Series("oblique_cross_bench", index=alignment.index)
        is_parallel = alignment == "parallel_like"
        is_perp = alignment == "perpendicular"
        is_oblique = alignment == "oblique"

        rel[is_parallel & same_bench] = "parallel_same_bench"
        rel[is_parallel & ~same_bench] = "cross_formation"
        rel[is_perp] = "cross_orientation"
        rel[is_oblique & same_bench] = "oblique_same_bench"
        return rel

    def _build_threshold_series(
        self,
        df: pd.DataFrame,
        *,
        parallel_eps_ft: float,
        perp_eps_ft: float,
        oblique_eps_ft: float,
        cross_bench_eps_ft: float,
        cross_bench_vertical_gate_ft: float,
        overrides_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Assign per-row distance metric, threshold, and vertical gate.

        Returns the input dataframe with added columns:
        - dist_for_threshold
        - threshold_ft
        - vert_gate_ft  (NaN for same-bench types)
        """
        # --- Distance metric selection ---
        df["dist_for_threshold"] = df["hz_effective"].copy()
        # Cross-bench parallel: use 3D distance if available
        cf_mask = df["rel_type"] == self.REL_CROSS_FORMATION
        if "3D_dist_effective" in df.columns:
            df.loc[cf_mask, "dist_for_threshold"] = (
                df.loc[cf_mask, "3D_dist_effective"]
                .fillna(df.loc[cf_mask, "hz_effective"])
            )

        # --- Threshold mapping (global defaults) ---
        threshold_map = {
            self.REL_PARALLEL_SAME_BENCH: parallel_eps_ft,
            self.REL_CROSS_FORMATION: cross_bench_eps_ft,
            self.REL_CROSS_ORIENTATION: perp_eps_ft,
            self.REL_OBLIQUE_SAME_BENCH: oblique_eps_ft,
            self.REL_OBLIQUE_CROSS_BENCH: oblique_eps_ft,
        }
        df["threshold_ft"] = df["rel_type"].map(threshold_map)

        # --- Vertical gate for cross-bench types ---
        df["vert_gate_ft"] = np.nan
        for cb_type in self._CROSS_BENCH_TYPES:
            mask = df["rel_type"] == cb_type
            df.loc[mask, "vert_gate_ft"] = cross_bench_vertical_gate_ft

        # --- Per-well overrides ---
        if overrides_df is not None and len(overrides_df) > 0:
            ov = overrides_df.copy()
            if "uwi" in ov.columns and "well_i" not in ov.columns:
                ov = ov.rename(columns={"uwi": "well_i"})
            ov["well_i"] = ov["well_i"].astype(str)

            n_before = len(ov)
            self.logger.info(f"Applying per-well overrides for {n_before:,} wells")

            # Merge override columns onto df (left join on well_i)
            override_cols = [
                "parallel_eps_ft",
                "cross_bench_vertical_gate_ft",
                "cross_bench_eps_ft",
                "perp_eps_ft",
                "oblique_eps_ft",
            ]
            present_cols = ["well_i"] + [c for c in override_cols if c in ov.columns]
            ov_slim = ov[present_cols].drop_duplicates(subset=["well_i"])
            df = df.merge(ov_slim, on="well_i", how="left", suffixes=("", "_ov"))

            # Apply overrides — update threshold_ft per type if override is present
            if "parallel_eps_ft" in ov_slim.columns:
                m = (df["rel_type"] == self.REL_PARALLEL_SAME_BENCH) & df["parallel_eps_ft"].notna()
                df.loc[m, "threshold_ft"] = df.loc[m, "parallel_eps_ft"]
            if "cross_bench_eps_ft" in ov_slim.columns:
                m = (df["rel_type"] == self.REL_CROSS_FORMATION) & df["cross_bench_eps_ft"].notna()
                df.loc[m, "threshold_ft"] = df.loc[m, "cross_bench_eps_ft"]
            if "perp_eps_ft" in ov_slim.columns:
                m = (df["rel_type"] == self.REL_CROSS_ORIENTATION) & df["perp_eps_ft"].notna()
                df.loc[m, "threshold_ft"] = df.loc[m, "perp_eps_ft"]
            if "oblique_eps_ft" in ov_slim.columns:
                for ot in [self.REL_OBLIQUE_SAME_BENCH, self.REL_OBLIQUE_CROSS_BENCH]:
                    m = (df["rel_type"] == ot) & df["oblique_eps_ft"].notna()
                    df.loc[m, "threshold_ft"] = df.loc[m, "oblique_eps_ft"]
            if "cross_bench_vertical_gate_ft" in ov_slim.columns:
                for cb_type in self._CROSS_BENCH_TYPES:
                    m = (df["rel_type"] == cb_type) & df["cross_bench_vertical_gate_ft"].notna()
                    df.loc[m, "vert_gate_ft"] = df.loc[m, "cross_bench_vertical_gate_ft"]

            # Drop temporary override columns
            drop_cols = [c for c in df.columns if c in override_cols]
            df = df.drop(columns=drop_cols, errors="ignore")

        return df

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def assign_roles(
        self,
        pairs_df: pd.DataFrame,
        header_df: pd.DataFrame,
        *,
        parallel_eps_ft: float = 1_320,
        perp_eps_ft: float = 1_000,
        oblique_eps_ft: float = 1_100,
        cross_bench_eps_ft: float = 1_320,
        cross_bench_vertical_gate_ft: float = 200,
        use_3d_for_cross_bench: bool = True,
        inner_zone_ft: float = 660,
        child_window_months: float = 18,
        infill_min_older: int = 2,
        role_pair_types: Optional[Dict[str, bool]] = None,
        overrides_df: Optional[pd.DataFrame] = None,
        header_join_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run the full role-assignment pipeline and return one row per well.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            Pairwise spacing table. See class docstring for required columns.
        header_df : pd.DataFrame
            Well header table. Must have 'uwi' and 'comp_date'.
        parallel_eps_ft : float
            Max hz_effective for parallel_same_bench pairs.
        perp_eps_ft : float
            Max hz_effective for perpendicular pairs.
        oblique_eps_ft : float
            Max hz_effective for oblique pairs.
        cross_bench_eps_ft : float
            Max distance for cross-bench parallel pairs.
        cross_bench_vertical_gate_ft : float
            Max vertical_dist for cross-bench pairs.
        use_3d_for_cross_bench : bool
            Use 3D_dist_effective for cross-bench parallel pairs.
        inner_zone_ft : float
            Inner severity zone radius (ft).
        child_window_months : float
            Months after parent to label gen1_child.
        infill_min_older : int
            Min older neighbors for infill_candidate.
        role_pair_types : dict, optional
            Toggle pair types. Keys: rel_type names. Values: bool.
            Missing keys default to True.
        overrides_df : pd.DataFrame, optional
            Per-well threshold overrides.
        header_join_cols : list[str], optional
            Header columns to include in the output. If None, a sensible
            default set is used.

        Returns
        -------
        pd.DataFrame
            One row per well with role, parent info, and neighborhood stats.
        """
        self.logger.info("=" * 80)
        self.logger.info("OVERLAPPING NEIGHBORHOOD ROLE ASSIGNMENT (V2)")
        self.logger.info("=" * 80)
        pipeline_start = time.time()

        self._clear_intermediates()
        overrides_df = self._resolve_overrides(overrides_df)

        # --- Resolve role_pair_types ---
        rpt = dict(self._DEFAULT_ROLE_PAIR_TYPES)
        if role_pair_types is not None:
            rpt.update(role_pair_types)
        enabled_types = [k for k, v in rpt.items() if v]
        disabled_types = [k for k, v in rpt.items() if not v]

        self.logger.info("Configuration:")
        self.logger.info(f"  parallel_eps_ft:               {parallel_eps_ft:,.0f}")
        self.logger.info(f"  perp_eps_ft:                   {perp_eps_ft:,.0f}")
        self.logger.info(f"  oblique_eps_ft:                {oblique_eps_ft:,.0f}")
        self.logger.info(f"  cross_bench_eps_ft:            {cross_bench_eps_ft:,.0f}")
        self.logger.info(f"  cross_bench_vertical_gate_ft:  {cross_bench_vertical_gate_ft:,.0f}")
        self.logger.info(f"  use_3d_for_cross_bench:        {use_3d_for_cross_bench}")
        self.logger.info(f"  inner_zone_ft:                 {inner_zone_ft:,.0f}")
        self.logger.info(f"  child_window_months:           {child_window_months}")
        self.logger.info(f"  infill_min_older:              {infill_min_older}")
        self.logger.info(f"  Enabled pair types:  {enabled_types}")
        if disabled_types:
            self.logger.info(f"  Disabled pair types: {disabled_types}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Prepare pairs
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 1: Preparing pairs")

        df = pairs_df.copy()
        df["well_i"] = df["well_i"].astype(str)
        df["well_k"] = df["well_k"].astype(str)

        # Filter rejected pairs if reject_reason column exists
        if "reject_reason" in df.columns:
            n_before = len(df)
            df = df[df["reject_reason"] == ""].drop(columns="reject_reason").copy()
            n_rejected = n_before - len(df)
            self.logger.info(f"  Filtered {n_rejected:,} rejected pairs ({n_before:,} -> {len(df):,})")
        else:
            self.logger.debug("  No reject_reason column; using all rows")

        self.logger.info(f"  Valid pairs: {len(df):,}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Symmetrize
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 2: Symmetrizing pairs")

        n_before_sym = len(df)
        df_sym = self._symmetrize_pairs(df)
        self.logger.info(f"  {n_before_sym:,} -> {len(df_sym):,} symmetric pairs")

        # ══════════════════════════════════════════════════════════════════
        # STEP 3: Classify relationship type
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 3: Classifying relationship types")

        df_sym["same_bench"] = df_sym["bench_i"] == df_sym["bench_k"]
        df_sym["rel_type"] = self._classify_rel_type(
            df_sym["pair_alignment"], df_sym["same_bench"]
        )

        for rt in self.ALL_REL_TYPES:
            count = (df_sym["rel_type"] == rt).sum()
            if count > 0:
                self.logger.info(f"  {rt:<25s} {count:>8,}")

        self._df_sym = df_sym

        # ══════════════════════════════════════════════════════════════════
        # STEP 4: Apply thresholds & vertical gate
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 4: Applying distance thresholds & vertical gate")

        df_sym = self._build_threshold_series(
            df_sym,
            parallel_eps_ft=parallel_eps_ft,
            perp_eps_ft=perp_eps_ft,
            oblique_eps_ft=oblique_eps_ft,
            cross_bench_eps_ft=cross_bench_eps_ft,
            cross_bench_vertical_gate_ft=cross_bench_vertical_gate_ft,
            overrides_df=overrides_df,
        )

        # Distance gate
        dist_ok = df_sym["dist_for_threshold"] <= df_sym["threshold_ft"]

        # Vertical gate for cross-bench types
        vert_ok = pd.Series(True, index=df_sym.index)
        has_vert_gate = df_sym["vert_gate_ft"].notna()
        vert_ok[has_vert_gate] = (
            df_sym.loc[has_vert_gate, "vertical_dist"]
            <= df_sym.loc[has_vert_gate, "vert_gate_ft"]
        )

        df_sym["in_neighborhood"] = dist_ok & vert_ok

        # Filter to neighborhood pairs
        df_nbhd = df_sym[df_sym["in_neighborhood"]].copy()

        self.logger.info(f"  Total symmetric pairs:     {len(df_sym):,}")
        self.logger.info(f"  Within neighborhood:       {len(df_nbhd):,}")
        self.logger.info(f"  Neighborhood pairs by type:")
        for rt in self.ALL_REL_TYPES:
            count = (df_nbhd["rel_type"] == rt).sum()
            if count > 0:
                self.logger.info(f"    {rt:<25s} {count:>8,}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 5: Join completion dates
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 5: Joining completion dates")

        hdr = header_df.copy()
        hdr["uwi"] = hdr["uwi"].astype(str)
        date_lookup = hdr.set_index("uwi")["comp_date"].to_dict()

        df_nbhd["comp_date_i"] = df_nbhd["well_i"].map(date_lookup)
        df_nbhd["comp_date_k"] = df_nbhd["well_k"].map(date_lookup)
        df_nbhd["k_is_older"] = df_nbhd["comp_date_k"] < df_nbhd["comp_date_i"]
        df_nbhd["days_diff"] = (
            df_nbhd["comp_date_i"] - df_nbhd["comp_date_k"]
        ).dt.days

        n_missing_i = df_nbhd["comp_date_i"].isna().sum()
        n_missing_k = df_nbhd["comp_date_k"].isna().sum()
        self.logger.info(f"  Pairs missing comp_date_i: {n_missing_i:,}")
        self.logger.info(f"  Pairs missing comp_date_k: {n_missing_k:,}")
        self.logger.info(f"  Pairs where k is older:    {df_nbhd['k_is_older'].sum():,}")

        # Add confidence tag per pair
        df_nbhd["confidence"] = df_nbhd["rel_type"].map(self.CONFIDENCE_MAP)

        self._df_nbhd = df_nbhd

        # ══════════════════════════════════════════════════════════════════
        # STEP 6: Role assignment (universal — all enabled pair types)
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 6: Assigning roles (universal pair-type support)")

        # Filter to enabled pair types only
        eligible = df_nbhd[df_nbhd["rel_type"].isin(enabled_types)].copy()
        self.logger.info(f"  Eligible neighborhood pairs (enabled types): {len(eligible):,}")

        # --- Per-well aggregations ---
        n_eligible_all = eligible.groupby("well_i").size().rename("n_eligible_neighbors")

        # Older eligible neighbors
        elig_older = eligible[eligible["k_is_older"]].copy()
        n_older = elig_older.groupby("well_i").size().rename("n_older_eligible")

        # Nearest older eligible neighbor (= parent candidate)
        if len(elig_older) > 0:
            idx_nearest = elig_older.groupby("well_i")["dist_for_threshold"].idxmin()
            df_parent = (
                elig_older.loc[idx_nearest][
                    [
                        "well_i",
                        "well_k",
                        "dist_for_threshold",
                        "vertical_dist",
                        "comp_date_k",
                        "days_diff",
                        "rel_type",
                        "confidence",
                        "tvd_i",
                        "tvd_k",
                    ]
                ]
                .rename(
                    columns={
                        "well_k": "parent_uwi",
                        "dist_for_threshold": "parent_dist_ft",
                        "vertical_dist": "parent_vertical_ft",
                        "comp_date_k": "parent_comp_date",
                        "days_diff": "days_since_parent",
                        "rel_type": "parent_pair_type",
                        "confidence": "parent_confidence",
                    }
                )
                .set_index("well_i")
            )
            # Child above parent = child TVD < parent TVD (shallower)
            df_parent["child_above_parent"] = df_parent["tvd_i"] < df_parent["tvd_k"]
            df_parent = df_parent.drop(columns=["tvd_i", "tvd_k"])
        else:
            df_parent = pd.DataFrame()

        # Inner zone flag (any neighbor within inner_zone_ft)
        inner_flag = (
            eligible[eligible["dist_for_threshold"] <= inner_zone_ft]
            .groupby("well_i")
            .size()
            .rename("n_inner_zone_neighbors")
        )

        # Per-type neighbor counts
        type_counts = {}
        for rt in self.ALL_REL_TYPES:
            count_s = (
                eligible[eligible["rel_type"] == rt]
                .groupby("well_i")
                .size()
                .rename(f"n_{rt}")
            )
            type_counts[rt] = count_s

        # Contributing types per well (which types have older neighbors)
        contributing = (
            elig_older.groupby("well_i")["rel_type"]
            .apply(lambda s: ",".join(sorted(s.unique())))
            .rename("role_contributing_types")
        )
        dominant = (
            elig_older.groupby("well_i")["rel_type"]
            .apply(lambda s: s.value_counts().idxmax())
            .rename("dominant_pair_type")
        )

        # --- Assemble role table ---
        all_uwis = hdr["uwi"].unique()
        df_roles = pd.DataFrame({"uwi": all_uwis}).set_index("uwi")

        df_roles = df_roles.join(n_eligible_all.rename_axis("uwi"), how="left")
        df_roles = df_roles.join(n_older.rename_axis("uwi"), how="left")
        if len(df_parent) > 0:
            df_roles = df_roles.join(df_parent, how="left")
        df_roles = df_roles.join(inner_flag.rename_axis("uwi"), how="left")
        for rt in self.ALL_REL_TYPES:
            df_roles = df_roles.join(type_counts[rt].rename_axis("uwi"), how="left")
        df_roles = df_roles.join(contributing.rename_axis("uwi"), how="left")
        df_roles = df_roles.join(dominant.rename_axis("uwi"), how="left")

        # Fill NaN counts with 0
        count_cols = (
            ["n_eligible_neighbors", "n_older_eligible", "n_inner_zone_neighbors"]
            + [f"n_{rt}" for rt in self.ALL_REL_TYPES]
        )
        for col in count_cols:
            if col in df_roles.columns:
                df_roles[col] = df_roles[col].fillna(0).astype(int)

        # --- Assign role ---
        def _assign_role(row: pd.Series) -> str:
            if row["n_eligible_neighbors"] == 0:
                return self.ROLE_NO_NEIGHBOR
            if row["n_older_eligible"] == 0:
                return self.ROLE_PARENT
            if row["n_older_eligible"] >= infill_min_older:
                return self.ROLE_INFILL
            return self.ROLE_CHILD

        df_roles["role"] = df_roles.apply(_assign_role, axis=1)

        # --- Child generation sub-label ---
        child_window_days = child_window_months * 30.44
        is_child = df_roles["role"] == self.ROLE_CHILD
        if "days_since_parent" in df_roles.columns:
            df_roles.loc[is_child, "child_gen"] = np.where(
                df_roles.loc[is_child, "days_since_parent"] <= child_window_days,
                "gen1_child",
                "late_child",
            )
        else:
            df_roles.loc[is_child, "child_gen"] = np.nan

        df_roles = df_roles.reset_index()
        self._df_roles = df_roles

        # Log role distribution
        self.logger.info("  Role distribution:")
        for role_name, count in df_roles["role"].value_counts().items():
            self.logger.info(f"    {role_name:<25s} {count:>6,}")

        if "child_gen" in df_roles.columns:
            gen_counts = df_roles["child_gen"].value_counts(dropna=True)
            if len(gen_counts) > 0:
                self.logger.info("  Child generation breakdown:")
                for gen_name, count in gen_counts.items():
                    self.logger.info(f"    {gen_name:<25s} {count:>6,}")

        # Log parent pair-type breakdown
        if "parent_pair_type" in df_roles.columns:
            ppt = df_roles["parent_pair_type"].value_counts(dropna=True)
            if len(ppt) > 0:
                self.logger.info("  Parent assigned via pair type:")
                for pt, count in ppt.items():
                    self.logger.info(f"    {pt:<25s} {count:>6,}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 7: Cross-interference summary (for ALL types, not just enabled)
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 7: Cross-interference summary")

        for rt in [
            self.REL_CROSS_FORMATION,
            self.REL_CROSS_ORIENTATION,
            self.REL_OBLIQUE_SAME_BENCH,
        ]:
            sub = df_nbhd[df_nbhd["rel_type"] == rt]
            if len(sub) == 0:
                continue
            total = sub.groupby("well_i").size().rename(f"n_{rt}")
            pre = (
                sub[sub["k_is_older"]]
                .groupby("well_i")
                .size()
                .rename(f"n_{rt}_pre_existing")
            )
            post = (
                sub[~sub["k_is_older"]]
                .groupby("well_i")
                .size()
                .rename(f"n_{rt}_post")
            )
            min_d = (
                sub.groupby("well_i")["dist_for_threshold"]
                .min()
                .rename(f"min_dist_{rt}_ft")
            )

            for s in [total, pre, post, min_d]:
                col_name = s.name
                s_indexed = s.rename_axis("uwi")
                # Avoid overwriting type count columns already in df_roles
                if col_name in df_roles.columns:
                    continue
                df_roles = df_roles.set_index("uwi").join(s_indexed, how="left").reset_index()

            self.logger.info(
                f"  {rt}: {(total > 0).sum():,} wells with >= 1 neighbor"
            )

        # Fill cross-interference NaN counts
        cross_count_cols = [
            c
            for c in df_roles.columns
            if c.startswith("n_cross") or c.startswith("n_oblique")
        ]
        # Only fill integer count columns (not distance columns)
        int_cross_cols = [c for c in cross_count_cols if "dist" not in c]
        for col in int_cross_cols:
            df_roles[col] = df_roles[col].fillna(0).astype(int)

        # ══════════════════════════════════════════════════════════════════
        # STEP 8: Join header attributes
        # ══════════════════════════════════════════════════════════════════
        self.logger.info("-" * 60)
        self.logger.info("Step 8: Joining header attributes")

        if header_join_cols is None:
            # Default set of useful header columns
            default_cols = [
                "uwi", "well_name", "lease_name", "operator", "bench",
                "comp_date", "first_prod_date",
                "lateral_length_ft", "tvd",
                "surface_lat", "surface_lon",
                "fluid_intensity_bbl_per_ft", "proppant_intensity_lbs_per_ft",
                "cum_oil_180d", "cum_oil_365d",
                "cum_oil_180d_per_ft", "cum_oil_365d_per_ft",
                "rsv_cat",
            ]
            header_join_cols = [c for c in default_cols if c in hdr.columns]
        else:
            # Always ensure uwi is included
            if "uwi" not in header_join_cols:
                header_join_cols = ["uwi"] + header_join_cols

        df_out = df_roles.merge(
            hdr[header_join_cols], on="uwi", how="left"
        )

        self.logger.info(f"  Joined {len(header_join_cols) - 1} header columns")

        # ══════════════════════════════════════════════════════════════════
        # STEP 9: Order output columns
        # ══════════════════════════════════════════════════════════════════
        # Identity -> Role -> Parent -> Neighborhood -> Cross-interference -> Header -> Coords
        priority_cols = [
            # Identity
            "uwi", "well_name", "lease_name", "operator", "bench", "comp_date",
            # Role
            "role", "child_gen",
            # Parent info
            "parent_uwi", "parent_dist_ft", "parent_pair_type", "parent_confidence",
            "parent_vertical_ft", "child_above_parent",
            "parent_comp_date", "days_since_parent",
            # Neighborhood
            "n_eligible_neighbors", "n_older_eligible", "n_inner_zone_neighbors",
            "n_parallel_same_bench",
            "n_cross_formation", "n_cross_formation_pre_existing", "n_cross_formation_post",
            "min_dist_cross_formation_ft",
            "n_cross_orientation", "n_cross_orientation_pre_existing", "n_cross_orientation_post",
            "min_dist_cross_orientation_ft",
            "n_oblique_same_bench", "n_oblique_same_bench_pre_existing", "n_oblique_same_bench_post",
            "min_dist_oblique_same_bench_ft",
            "n_oblique_cross_bench",
            "role_contributing_types", "dominant_pair_type",
        ]
        # Columns that exist in df_out, in priority order, then remaining
        ordered = [c for c in priority_cols if c in df_out.columns]
        remaining = [c for c in df_out.columns if c not in ordered]
        df_out = df_out[ordered + remaining]

        self._df_output = df_out

        elapsed = time.time() - pipeline_start
        self.logger.info("=" * 80)
        self.logger.info(
            f"DONE — {len(df_out):,} wells, {len(df_nbhd):,} neighborhood pairs, "
            f"{elapsed:.1f}s"
        )
        self.logger.info("=" * 80)

        return df_out

    # ------------------------------------------------------------------
    #  Diagnostic helpers
    # ------------------------------------------------------------------

    def print_neighborhood(
        self, uwi: str, *, max_rows: int = 30
    ) -> None:
        """Pretty-print the full neighborhood of a focal well for validation."""
        if self._df_nbhd is None or self._df_output is None:
            raise RuntimeError("Run assign_roles() before calling print_neighborhood()")

        uwi = str(uwi)
        row = self._df_output[self._df_output["uwi"] == uwi]
        if row.empty:
            print(f"UWI {uwi} not found in output table.")
            return
        row = row.iloc[0]

        print(f"\n{'=' * 60}")
        print(f"Focal well: {uwi}")
        print(f"{'=' * 60}")
        print(f"  Name:        {row.get('well_name', '—')}")
        print(f"  Bench:       {row.get('bench', '—')}")
        print(f"  Comp date:   {str(row.get('comp_date', '—'))[:10]}")
        print(f"  Role:        {row['role']}", end="")
        if pd.notna(row.get("child_gen")):
            print(f" ({row['child_gen']})", end="")
        print()

        if pd.notna(row.get("parent_uwi")):
            print(
                f"  Parent:      {row['parent_uwi']} @ {row['parent_dist_ft']:.0f} ft"
                f"  [{row.get('parent_pair_type', '?')}]"
                f"  (conf: {row.get('parent_confidence', '?')})"
            )
            if pd.notna(row.get("child_above_parent")):
                pos = "ABOVE" if row["child_above_parent"] else "BELOW"
                print(f"  Child is {pos} parent  (vert sep: {row.get('parent_vertical_ft', 0):.0f} ft)")

        print(f"  Eligible neighbors:  {int(row.get('n_eligible_neighbors', 0))}")
        print(f"  Older eligible:      {int(row.get('n_older_eligible', 0))}")
        print(f"  Inner zone (<{int(660)} ft): {int(row.get('n_inner_zone_neighbors', 0))}")

        for rt in self.ALL_REL_TYPES:
            col = f"n_{rt}"
            if col in row.index and row[col] > 0:
                print(f"  {rt}: {int(row[col])}")

        # Neighbor detail
        nbrs = self._df_nbhd[self._df_nbhd["well_i"] == uwi].sort_values(
            "dist_for_threshold"
        )
        if nbrs.empty:
            print("  (no neighbors within threshold)")
            return

        print(f"\n  Neighbor detail ({len(nbrs)} total):")
        display_cols = [
            "well_k", "rel_type", "confidence", "dist_for_threshold",
            "vertical_dist", "angle_deg", "bench_k",
            "comp_date_k", "k_is_older",
        ]
        display_cols = [c for c in display_cols if c in nbrs.columns]
        print(nbrs[display_cols].head(max_rows).to_string(index=False))

    @property
    def neighborhood_pairs(self) -> Optional[pd.DataFrame]:
        """Return the neighborhood edge table from the last run."""
        return self._df_nbhd

    @property
    def roles_table(self) -> Optional[pd.DataFrame]:
        """Return the per-well role table (before header join) from the last run."""
        return self._df_roles

    @property
    def symmetric_pairs(self) -> Optional[pd.DataFrame]:
        """Return the full symmetric pair table from the last run."""
        return self._df_sym
