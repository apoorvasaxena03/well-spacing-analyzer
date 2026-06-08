#%% Imports
from __future__ import annotations 

import os
from pathlib import Path

import time
import warnings

import logging

import pandas as pd 

pd.set_option('display.max_columns', None) # show all columns when printing DataFrames

import numpy as np

import math

from typing import Dict, Tuple, List, Union, Optional, ClassVar, Any, Literal, Iterable, Set

from tqdm import tqdm 


from matplotlib import pyplot as plt 
from matplotlib.patches import Circle

from pyproj import Geod 

from dataclasses import dataclass 
from enum import Enum, auto

from scipy.spatial import ConvexHull

#%% # ==================== Well Spacing Calculator ====================

class AlignmentType(Enum):
    """
    Classification of well pair alignment based on the angle between their lateral axes.

    Values
    ------
    PARALLEL_LIKE : Wells are roughly parallel (angle <= theta_parallel_deg, typically 25°)
    OBLIQUE : Wells are at an intermediate angle (between parallel and perpendicular thresholds)
    PERPENDICULAR : Wells are roughly perpendicular (angle >= theta_perp_deg, typically 65°)
    MISALIGNED : Wells that don't fit other categories (used when reject_misaligned=True)
    """
    PARALLEL_LIKE = auto()
    OBLIQUE = auto()
    PERPENDICULAR = auto()
    MISALIGNED = auto()


@dataclass
class SpacingResult:
    """
    Container for well-pair spacing metrics computed by WellSpacingCalculator.

    This dataclass holds all computed metrics for a single (well_i, well_k) pair,
    including spacing distances, overlap metrics, alignment classification, and
    directional information.

    Attributes
    ----------
    well_i, well_k : Any
        Unique identifiers for the well pair.
    horizontal_dist : float
        Mean horizontal spacing (ft). For parallel-like pairs: crossline mean |Δy(x)|.
        For oblique/perp: mean nearest-projection distance.
    horizontal_dist_median : float
        Median horizontal spacing (ft).
    vertical_dist : float
        Absolute vertical separation between well midpoints (ft).
    dist3d : float
        3D distance: sqrt(horizontal_dist² + vertical_dist²) (ft).
    overlap_len_common_ft : float
        Length of lateral overlap in i-frame (ft). NaN for oblique/perp pairs.
    LL_i, LL_k : float
        Lateral lengths of well_i and well_k respectively (ft).
    overlap_pct_i, overlap_pct_k : float
        Overlap length as percentage of each well's lateral (0-1). NaN for oblique/perp.
    n_samples : int
        Number of sample points used in spacing calculation.
    dy_p5 : float
        5th percentile of crossline distances (ft). Used as quality filter.
    angle_deg : float
        Angle between well lateral axes (degrees, 0-90).
    pair_alignment : str
        Classification: 'parallel_like', 'oblique', 'perpendicular', or 'misaligned'.
    min_distance_ft : float
        Minimum nearest-projection distance (ft). For oblique/perp only; NaN for parallel.
    mean_windowed_ft : float
        Mean distance in ±window_ft around closest approach (ft). For oblique/perp only.
    reject_reason : str
        If pair was rejected, reason string (e.g., 'no_overlap_x', 'coarse_far'). Empty if valid.
    direction_axis : str
        The axis used for direction classification ('EW' or 'NS').
    direction_to_k_from_i_axis : str
        Cardinal direction from i to k constrained to direction_axis ('E','W','N', or 'S').
    direction_axis_confidence : float
        Confidence score for direction classification (0-1).
    direction_axis_distribution : str
        Distribution of directional votes (e.g., 'E:0.83,W:0.17').
    drill_direction_i, drill_direction_k : str
        Drill direction classification for each well ('EW' or 'NS').
    axis_forced : bool
        Whether direction was forced to axis (always True in current implementation).
    contact_threshold_ft : float
        Distance threshold T used for contact metrics (ft). For oblique/perp only.
    contact_len_i_ft : float
        Length of well_i within T ft of well_k (ft). For oblique/perp only.
    contact_pct_i : float
        contact_len_i_ft as percentage of LL_i (0-1). For oblique/perp only.
    contact_len_i_interior_ft : float
        Contact length excluding endpoints (interior only). For oblique/perp only.
    contact_pct_i_interior : float
        Interior contact as percentage of LL_i (0-1). For oblique/perp only.
    proj_coverage_i_pct : float
        Percentage of well_i samples with valid interior projection onto well_k (0-1).
    """
    # core identifiers
    well_i: Any
    well_k: Any

    # spacing metrics (same names you already emit)
    horizontal_dist: float
    horizontal_dist_median: float
    vertical_dist: float
    dist3d: float

    # overlap + LL (parallel-like; NaN otherwise)
    overlap_len_common_ft: float
    LL_i: float
    LL_k: float
    tvd_i: float
    tvd_k: float
    overlap_pct_i: float
    overlap_pct_k: float

    # diagnostics
    n_samples: int
    dy_p5: float
    angle_deg: float
    pair_alignment: str
    min_distance_ft: float       # for oblique/perp (NaN for parallel-like)
    mean_windowed_ft: float      # for oblique/perp (NaN for parallel-like)
    reject_reason: str

    # direction fields
    direction_axis: Optional[str]
    direction_to_k_from_i_axis: Optional[str]
    direction_axis_confidence: float
    direction_axis_distribution: str

    # drill directions
    drill_direction_i: str
    drill_direction_k: str

    # marker
    axis_forced: bool = True

    # contact + coverage metrics for oblique/perp ---
    contact_threshold_ft: float = float("nan")
    contact_len_i_ft: float = float("nan")
    contact_pct_i: float = float("nan")
    contact_len_i_interior_ft: float = float("nan")
    contact_pct_i_interior: float = float("nan")
    proj_coverage_i_pct: float = float("nan")
@dataclass
class PairArtifacts:
    """
    Container for intermediate computation artifacts used in debug visualizations.

    This dataclass stores arrays and metadata generated during spacing calculations
    that are useful for diagnostic plots (e.g., debug_pair_spacing). Most fields
    are None unless want_artifacts=True is passed to the computation method.

    Attributes (Common)
    -------------------
    Xi_utm, Xk_utm : np.ndarray or None
        Full-resolution UTM coordinates for well_i and well_k. Shape: (N, 2).
    Xi_if, Xk_if : np.ndarray or None
        Coordinates projected into well_i's local frame. Shape: (N, 2).
    dir_axis : str or None
        Direction axis used ('EW' or 'NS').
    has_latlon : bool
        Whether lat/lon coordinates were available for geodetic calculations.

    Attributes (Parallel-like pairs)
    --------------------------------
    overlap_band : Tuple[float, float] or None
        (x_lo, x_hi) bounds of the overlap region in i-frame coordinates.
    Xi_clip, Xk_clip : np.ndarray or None
        Clipped polylines within the overlap band. Shape: (M, 2).
    xgrid : np.ndarray or None
        X-coordinates of sample stations in overlap region.
    yi, yk : np.ndarray or None
        Interpolated y-coordinates at each x-station for wells i and k.
    Xi_utml, Xk_utml : np.ndarray or None
        UTM coordinates of sample points for quiver/arrow plots. Shape: (N, 2).
    theta_rose : np.ndarray or None
        Azimuth angles (radians) for polar rose plot of i→k directions.

    Attributes (Oblique/Perpendicular pairs)
    ----------------------------------------
    s_targets : np.ndarray or None
        Arclength sample positions along well_i (ft).
    Pi : np.ndarray or None
        Sample points on well_i in UTM coordinates. Shape: (N, 2).
    Q : np.ndarray or None
        Nearest points on well_k for each sample in Pi. Shape: (N, 2).
    d_series : np.ndarray or None
        Distance from each Pi sample to its nearest point Q on well_k (ft).
    """
    # Always useful
    Xi_utm: Optional[np.ndarray] = None
    Xk_utm: Optional[np.ndarray] = None
    Xi_if: Optional[np.ndarray] = None
    Xk_if: Optional[np.ndarray] = None
    dir_axis: Optional[str] = None
    has_latlon: bool = False

    # Parallel-like only
    overlap_band: Optional[Tuple[float, float]] = None
    Xi_clip: Optional[np.ndarray] = None
    Xk_clip: Optional[np.ndarray] = None
    xgrid: Optional[np.ndarray] = None
    yi: Optional[np.ndarray] = None
    yk: Optional[np.ndarray] = None
    Xi_utml: Optional[np.ndarray] = None
    Xk_utml: Optional[np.ndarray] = None
    theta_rose: Optional[np.ndarray] = None  # radians for polar plot

    # Oblique/Perp only
    s_targets: Optional[np.ndarray] = None
    Pi: Optional[np.ndarray] = None          # samples on i (UTM)
    Q: Optional[np.ndarray] = None           # nearest on k (UTM)
    d_series: Optional[np.ndarray] = None    # nearest distances

class WellSpacingCalculator:
    """
    Advanced calculator for computing well spacing metrics and directional relationships
    between horizontal well pairs using alignment-aware algorithms and 3D geometric analysis.

    This class implements a sophisticated framework for analyzing spatial relationships
    between horizontal wells, providing metrics essential for evaluating well interference,
    reservoir drainage patterns, and regulatory compliance. The calculator uses alignment-based
    routing to apply the most appropriate spacing algorithm for each well pair.

    Key Features
    ------------
    1. **Alignment-Based Routing**: Automatically classifies well pairs as parallel-like,
       oblique, or perpendicular based on lateral axis angles, then routes to the appropriate
       spacing computation method.

    2. **Curvature-Aware Distance Calculation**: Uses full 3D well trajectories rather than
       simplified heel-to-toe lines, ensuring accurate spacing for curved laterals.

    3. **Midpoint Normalization**: Computes metrics at normalized midpoints (default 50% MD)
       to eliminate lateral-length bias in vertical distance calculations.

    4. **Local Coordinate Frame Projection**: Projects wells into a local i-frame aligned
       with the reference well's lateral axis for accurate crossline distance computation.

    5. **Direction Classification**: Determines cardinal directions (N/S/E/W) between wells
       using geodetic azimuth calculations constrained to drill direction axes.

    6. **Comprehensive Contact Metrics**: For non-parallel wells, computes contact length
       (how much of well_i is within threshold distance of well_k) and projection coverage.

    Spacing Algorithms by Alignment
    --------------------------------
    **Parallel-Like Pairs (angle ≤ 25°)**:
        - Computes crossline distance |Δy(x)| over x-overlap region in i-frame
        - Returns: mean, median, overlap length, overlap percentages
        - Appropriate for child wells in same pad, parallel infill wells

    **Oblique Pairs (25° < angle < 65°)**:
        - Computes nearest-projection distance series from well_i samples to well_k polyline
        - Returns: mean, median, min distance, contact metrics
        - Appropriate for wells at intermediate angles

    **Perpendicular Pairs (angle ≥ 65°)**:
        - Same as oblique, with optional windowed mean around closest approach
        - Returns: all oblique metrics plus windowed mean
        - Appropriate for crossing wells, offset wells

    Typical Workflow
    ----------------
    1. Initialize with trajectory data (DataFrame or Dict)
    2. Call _calculate_spacing_statistics() to compute all pairwise metrics
    3. Use results for neighbor identification, regulatory analysis, or visualization
    4. Optionally use debug_pair_spacing() for detailed diagnostics of specific pairs

    Attributes
    ----------
    trajectories : Dict[str, pd.DataFrame]
        Dictionary mapping well identifiers (uwi) to trajectory DataFrames.
        Each DataFrame contains columns: ['md', 'x', 'y', 'tvd', 'latitude', 'longitude', 'azimuth'].
    _trajectory_df : pd.DataFrame
        Combined trajectory data in long format with 'uwi' column for vectorized operations.
    _paircache : Dict[str, Any], optional
        Internal cache storing precomputed local frames, lateral lengths, and coarse arrays
        for fast pairwise distance filtering. Built on first call to spacing calculations.

    Examples
    --------
    **Basic Usage:**

    >>> import pandas as pd
    >>> from well_spacing_stats import WellSpacingCalculator
    >>>
    >>> # Load trajectory data
    >>> traj_df = pd.read_parquet("trajectories.parquet")
    >>> # Required columns: uwi, md, x, y, tvd, latitude, longitude, azimuth
    >>>
    >>> # Initialize calculator
    >>> wsc = WellSpacingCalculator(traj_df)
    >>>
    >>> # Compute spacing for all pairs within 20 miles
    >>> spacing_df = wsc._calculate_spacing_statistics(
    ...     frac=0.5,                    # Use midpoint (50% MD)
    ...     max_distance_miles=20.0,     # Geographic prefilter
    ...     theta_parallel_deg=25.0,     # Parallel threshold
    ...     theta_perp_deg=65.0,         # Perpendicular threshold
    ...     step_ft=100,                 # Sample every 100 ft
    ...     contact_threshold_ft=300.0   # Contact distance threshold
    ... )
    >>>
    >>> # Results contain ~40 columns per well pair:
    >>> print(spacing_df.columns)
    >>> # ['well_i', 'well_k', 'horizontal_dist', 'vertical_dist', '3D_dist',
    >>> #  'overlap_len_common_ft', 'LL_i', 'LL_k', 'tvd_i', 'tvd_k',
    >>> #  'pair_alignment', 'angle_deg', 'direction_to_k_from_i_axis', ...]

    **Debug Specific Pair:**

    >>> # Visualize detailed spacing computation for one pair
    >>> metrics = wsc.debug_pair_spacing(
    ...     uwi_i="42-123-12345-00-00",
    ...     uwi_k="42-123-12346-00-00",
    ...     save_dir="./debug_plots",
    ...     show=True
    ... )
    >>> # Creates plots: map_view, projected_iframe, crossline_dy_series, etc.

    **Filter to Nearest Neighbors:**

    >>> # Find wells within 500 ft horizontal, 200 ft vertical
    >>> neighbors = spacing_df[
    ...     (spacing_df['horizontal_dist'] <= 500) &
    ...     (spacing_df['vertical_dist'] <= 200) &
    ...     (spacing_df['reject_reason'] == '')  # Valid pairs only
    ... ].sort_values('horizontal_dist')

    Performance Characteristics
    ---------------------------
    - Time Complexity: O(N²) for N wells, parallelized across batches
    - Memory: Processes in configurable batches (default 500k pairs) to control RAM usage
    - Typical Runtime: ~10-30 minutes for 1000 wells (500k pairs) on modern hardware
    - Scalability: Can handle 2000+ wells with batch processing and saved Parquet outputs

    Notes
    -----
    - All distances are in feet (ft) unless otherwise specified
    - Geographic filtering uses approximate planar distance for speed
    - Direction axis is always constrained based on drill direction:
      * NS-drilled wells → E/W direction axis
      * EW-drilled wells → N/S direction axis
    - Effective horizontal distance is normalized across alignment types for consistent filtering
    - TVD values (tvd_i, tvd_k) represent true vertical depth at the specified midpoint fraction

    See Also
    --------
    DirectionalBenchNeighbors : Uses spacing results to identify directional neighbors
    SpacingResult : Dataclass containing all computed metrics for a well pair
    AlignmentType : Enum defining well pair alignment classifications

    References
    ----------
    This implementation is based on principles from:
    - Well interference analysis in unconventional reservoirs
    - Geodetic azimuth calculation using WGS84 ellipsoid
    - PCA-based local coordinate frame definition for curved trajectories
    """

    # Class-level constant (shared, immutable-by-convention)
    _DIR4_LABELS: ClassVar[np.ndarray] = np.array(["N", "E", "S", "W"], dtype=object)

    def __init__(self, trajectories: Union[Dict[str, pd.DataFrame], pd.DataFrame], header_df: pd.DataFrame, logger: Optional[logging.Logger] = None,):
        """
        Initialize WellSpacingCalculator with well trajectory data.

        This constructor accepts trajectory data in either DataFrame (long format) or
        dictionary format, automatically converting and storing both representations
        internally for optimal performance in different operations.

        Parameters
        ----------
        trajectories : Union[Dict[str, pd.DataFrame], pd.DataFrame]
            Well trajectory data in one of two formats:

            **Format 1: DataFrame (long format)** - Must contain columns:
               - 'uwi' : str
                   Unique well identifier (e.g., API number, company well ID)
               - 'md' : float
                   Measured depth along wellbore (ft). Should be monotonically increasing.
               - 'x' : float
                   UTM Easting coordinate (ft or meters, must be consistent)
               - 'y' : float
                   UTM Northing coordinate (ft or meters, must be consistent)
               - 'tvd' : float
                   True vertical depth (ft, positive down). Used for vertical separation.
               - 'latitude' : float, optional
                   Decimal degrees latitude (WGS84). Required for geodetic direction calculation.
               - 'longitude' : float, optional
                   Decimal degrees longitude (WGS84). Required for geodetic direction calculation.
               - 'azimuth' : float, optional
                   Wellbore azimuth in degrees (0-360, 0=North, 90=East). Used for drill direction
                   classification (EW vs NS). If not provided, direction inference is limited.

            **Format 2: Dict[str, pd.DataFrame]**
               - Keys: Well identifiers (uwi as strings)
               - Values: Trajectory DataFrames with columns above (except 'uwi')
               - Useful when loading pre-grouped trajectories or from APIs

        header_df : pd.DataFrame
            Well header data containing at minimum 'uwi' and 'bench' columns.
            Used to map bench_i and bench_k columns into spacing output
            (positioned right after well_k).

        Raises
        ------
        ValueError
            - If trajectories is a DataFrame without 'uwi' column
            - If trajectories is neither DataFrame nor Dict
            - If required columns (md, x, y, tvd) are missing from any trajectory

        Notes
        -----
        **Internal Storage:**
        The calculator maintains two representations:
        - `self.trajectories` : Dict[str, pd.DataFrame]
            Fast per-well lookups for pairwise computations. Used in local frame projection
            and polyline operations.
        - `self._trajectory_df` : pd.DataFrame
            Combined long-format data for vectorized operations like midpoint computation
            and drill direction classification.

        **Coordinate System Requirements:**
        - x, y should be in a projected coordinate system (UTM recommended)
        - Units should be consistent (feet or meters). Output distances will match input units.
        - tvd should use sign convention appropriate for your data (typically positive down)

        **Data Quality Recommendations:**
        - Sample trajectories at ≤50 ft intervals for accurate curvature representation
        - Ensure trajectories extend through full lateral (not just vertical section)
        - Remove duplicate MD values or out-of-order points before initialization
        - Verify latitude/longitude are present if direction classification is needed

        Examples
        --------
        **Example 1: Initialize from DataFrame**

        >>> import pandas as pd
        >>> from well_spacing_stats import WellSpacingCalculator
        >>>
        >>> # Load trajectory data with all wells stacked
        >>> traj_df = pd.DataFrame({
        ...     'uwi': ['WELL_001', 'WELL_001', 'WELL_001', 'WELL_002', 'WELL_002'],
        ...     'md': [10000, 10500, 11000, 10000, 10500],
        ...     'x': [500000, 501000, 502000, 500100, 501100],
        ...     'y': [4000000, 4000050, 4000100, 4000200, 4000250],
        ...     'tvd': [10000, 10020, 10040, 10000, 10020],
        ...     'latitude': [36.0, 36.001, 36.002, 36.002, 36.003],
        ...     'longitude': [-102.0, -102.001, -102.002, -102.001, -102.002],
        ...     'azimuth': [90, 92, 89, 88, 91]
        ... })
        >>>
        >>> wsc = WellSpacingCalculator(traj_df)
        >>> print(f"Loaded {len(wsc.trajectories)} wells")
        Loaded 2 wells

        **Example 2: Initialize from Dictionary**

        >>> # Pre-grouped trajectories (e.g., from database query)
        >>> trajectories_dict = {
        ...     'WELL_001': pd.DataFrame({
        ...         'md': [10000, 10500, 11000],
        ...         'x': [500000, 501000, 502000],
        ...         'y': [4000000, 4000050, 4000100],
        ...         'tvd': [10000, 10020, 10040],
        ...         'latitude': [36.0, 36.001, 36.002],
        ...         'longitude': [-102.0, -102.001, -102.002],
        ...         'azimuth': [90, 92, 89]
        ...     }),
        ...     'WELL_002': pd.DataFrame({
        ...         'md': [10000, 10500],
        ...         'x': [500100, 501100],
        ...         'y': [4000200, 4000250],
        ...         'tvd': [10000, 10020],
        ...         'latitude': [36.002, 36.003],
        ...         'longitude': [-102.001, -102.002],
        ...         'azimuth': [88, 91]
        ...     })
        ... }
        >>>
        >>> wsc = WellSpacingCalculator(trajectories_dict)
        >>> # Both formats yield identical internal representation

        **Example 3: Loading from File**

        >>> # From Parquet (recommended for large datasets)
        >>> traj_df = pd.read_parquet("trajectories.parquet")
        >>> wsc = WellSpacingCalculator(traj_df)
        >>>
        >>> # From CSV
        >>> traj_df = pd.read_csv("trajectories.csv")
        >>> wsc = WellSpacingCalculator(traj_df)

        **Example 4: Verify Data After Loading**

        >>> wsc = WellSpacingCalculator(traj_df)
        >>> print(f"Wells loaded: {len(wsc.trajectories)}")
        >>> print(f"Total trajectory points: {len(wsc._trajectory_df)}")
        >>>
        >>> # Check a specific well's trajectory
        >>> well_001_traj = wsc.trajectories['WELL_001']
        >>> print(f"WELL_001 trajectory points: {len(well_001_traj)}")
        >>> print(f"MD range: {well_001_traj['md'].min():.0f} - {well_001_traj['md'].max():.0f} ft")

        See Also
        --------
        _calculate_spacing_statistics : Main method to compute pairwise spacing metrics
        _compute_normalized_midpoints : Extracts midpoint coordinates from trajectories
        """
        # Set up hierarchical logger
        parent_logger = logger or logging.getLogger("well_spacing_calculator")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.well_spacing_calculator")
        self.logger.propagate = True

        self.logger.info("Initializing WellSpacingCalculator")

        if isinstance(trajectories, pd.DataFrame):
            if "uwi" not in trajectories.columns:
                raise ValueError("Trajectory DataFrame must contain 'uwi' column.")
            self.logger.debug("  Input format: DataFrame (long format)")
            self._trajectory_df = trajectories.reset_index(drop=True)
            self.trajectories = {
                cid: group for cid, group in self._trajectory_df.groupby("uwi")
            }
        elif isinstance(trajectories, dict):
            self.logger.debug("  Input format: Dict[str, DataFrame]")
            self.trajectories = trajectories
            # Inject 'uwi' from the dict keys when the per-well frames don't
            # already carry it (e.g. pre-grouped trajectories). Concatenating
            # with reset_index alone would drop the key, leaving _trajectory_df
            # without the 'uwi' column the engine relies on downstream.
            frames = []
            for uwi, frame in trajectories.items():
                if "uwi" not in frame.columns:
                    frame = frame.assign(uwi=uwi)
                frames.append(frame)
            self._trajectory_df = pd.concat(frames, ignore_index=True)
        else:
            raise ValueError("Invalid type for trajectories. Must be DataFrame or Dict.")

        # Log initialization summary
        n_wells = len(self.trajectories)
        n_points = len(self._trajectory_df)
        avg_points_per_well = n_points / n_wells if n_wells > 0 else 0

        # Check for optional columns
        has_latlon = "latitude" in self._trajectory_df.columns and "longitude" in self._trajectory_df.columns
        has_azimuth = "azimuth" in self._trajectory_df.columns

        self.logger.info(f"✓ Loaded {n_wells:,} wells with {n_points:,} trajectory points")
        self.logger.debug(f"  Average points per well: {avg_points_per_well:.1f}")
        self.logger.debug(f"  Lat/lon available: {'Yes' if has_latlon else 'No (direction calculation limited)'}")
        self.logger.debug(f"  Azimuth available: {'Yes' if has_azimuth else 'No (drill direction inference limited)'}")

        # Store bench mapping from header_df
        if not {'uwi', 'bench'}.issubset(header_df.columns):
            raise ValueError("header_df must contain 'uwi' and 'bench' columns.")
        self._bench_map = header_df.set_index("uwi")["bench"]
        self.logger.debug(f"  Bench mapping loaded: {len(self._bench_map)} wells")

    def _apply_effective_horizontal(
        self,
        spacing_df: pd.DataFrame,
        *,
        inplace: bool = True,
        keep_effective_3d_audit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize the horizontal metric across alignments and recompute 3D distance (vectorized).

        This post-processing step makes the *live* ``horizontal_dist`` semantically correct for all
        alignment classes so downstream consumers (e.g., :class:`DirectionalBenchNeighbors`) can
        rank/filter on a single column:

        - ``pair_alignment == "parallel_like"``  → keep the original crossline-mean spacing
        (computed over the true overlap band in the i-frame).
        - ``pair_alignment in {"oblique", "perpendicular", "misaligned"}`` → use
        ``min_distance_ft`` (closest approach from the nearest-projection series).

        Audit columns are added/preserved to keep full transparency, and ``3D_dist`` is overwritten
        to reflect the effective horizontal spacing.

        Parameters
        ----------
        spacing_df : pandas.DataFrame
            Spacing output with **one row per (well_i, well_k)**. Must contain at least:
            ``["horizontal_dist", "vertical_dist", "pair_alignment", "min_distance_ft"]``.
            If ``"3D_dist"`` exists, it will be **overwritten**.
        inplace : bool, default True
            If True, modify ``spacing_df`` in place and return it. If False, operate on a copy.
        keep_effective_3d_audit : bool, default True
            If True, also write an audit copy of the recomputed 3D as ``"3D_dist_effective"`` before
            overwriting ``"3D_dist"``.

        Returns
        -------
        pandas.DataFrame
            The modified DataFrame (same object if ``inplace=True``) with these columns updated/added:
            - ``horizontal_crossline_mean_ft`` : float
                Audit copy of the *pre-modification* ``horizontal_dist``.
                *Note*: for oblique/perp/misaligned rows, this reflects the **nearest-projection mean**
                (because that was the original definition of ``horizontal_dist`` for those rows).
            - ``hz_effective`` : float
                Effective horizontal used for selection/cutoffs (crossline mean for parallel_like;
                min nearest for oblique/perpendicular/misaligned).
            - ``hz_basis`` : {"crossline_mean", "min_nearest"}
                Provenance flag for ``hz_effective``.
            - ``horizontal_dist`` : float
                **Overwritten** with ``hz_effective`` so all downstream logic uses the right metric.
            - ``3D_dist`` : float
                **Overwritten** as ``hypot(horizontal_dist, vertical_dist)`` using the effective horizontal.
            - ``3D_dist_effective`` : float, optional
                Present only when ``keep_effective_3d_audit=True``; duplicate of the recomputed 3D.

        Raises
        ------
        ValueError
            If any required column is missing from ``spacing_df``.

        Notes
        -----
        - This method is intended to be called **once** at the end of the spacing computation
        (e.g., inside ``_calculate_spacing_statistics`` right before returning or saving batches),
        so parquet outputs are already normalized.
        - Rows with non-parallel alignment but missing/NaN ``min_distance_ft`` will keep NaN
        in the effective fields and naturally fall out under standard distance cutoffs.

        Examples
        --------
        >>> out = wsc._process_batch(...)          # internal pipeline step
        >>> out = wsc._apply_effective_horizontal(out, inplace=False)
        >>> out.filter(items=["well_i","well_k","pair_alignment",
        ...                   "horizontal_crossline_mean_ft","hz_effective",
        ...                   "hz_basis","horizontal_dist","vertical_dist","3D_dist"]).head()
        """
        required = {"horizontal_dist", "vertical_dist", "pair_alignment", "min_distance_ft"}
        missing = required - set(spacing_df.columns)
        if missing:
            raise ValueError(f"spacing_df is missing required columns: {sorted(missing)}")

        df = spacing_df if inplace else spacing_df.copy()

        # Preserve original horizontal_dist for QA (name kept for continuity)
        if "horizontal_crossline_mean_ft" not in df.columns:
            df["horizontal_crossline_mean_ft"] = df["horizontal_dist"]

        # Vectorized effective horizontal
        align = (
            df["pair_alignment"].astype(str).str.lower().str.replace(" ", "_", regex=False)
        )
        is_parallel = align.eq("parallel_like")

        hz_eff = np.where(
            is_parallel.to_numpy(),
            df["horizontal_dist"].to_numpy(),   # crossline mean (parallel_like)
            df["min_distance_ft"].to_numpy(),   # min nearest (oblique/perp)
        )
        df["hz_effective"] = hz_eff
        df["hz_basis"] = np.where(is_parallel.to_numpy(), "crossline_mean", "min_nearest")

        # Overwrite live horizontal_dist so downstream consumers (e.g., DBN) use the right number
        df["horizontal_dist"] = df["hz_effective"]

        # Recompute and overwrite 3D using the effective horizontal
        eff_3d = np.hypot(df["horizontal_dist"].to_numpy(), df["vertical_dist"].to_numpy())
        if keep_effective_3d_audit:
            df["3D_dist_effective"] = eff_3d
        df["3D_dist"] = eff_3d

        return df

    def _calculate_spacing_statistics(
        self,
        *,
        frac: float = 0.5,
        batch_size: int = 500_000,
        max_distance_miles: Optional[float] = 20.0,
        save_batches_dir: Optional[str] = None,
        use_interpolation: bool = False,
        # --- existing crossline knobs ---
        step_ft: int = 100,
        n_samples: Optional[int] = None,
        max_crossline_ft: Optional[float] = 2000.0,
        crossline_percentile: float = 5.0,
        ds_crossline_step_ft: int = 200,
        emit_rejected: bool = True,
        use_pca_axis: bool = True,
        # --- NEW: angle-based routing knobs ---
        theta_parallel_deg: float = 25.0,
        theta_perp_deg: float = 65.0,
        reject_misaligned: bool = False,
        # perpendicular optional smoothing
        use_windowed_mean: bool = False,
        window_ft: float = 300.0,
        # --- NEW: oblique/perp adjacency metrics ---
        contact_threshold_ft: float = 300.0,     # distance T for "contact" along i
        coverage_epsilon: float = 1e-3,          # interior cutoff for t in (ε, 1-ε)
    ) -> Optional[pd.DataFrame]:
        """
        Compute comprehensive spacing metrics for all well pairs using alignment-aware routing.

        This is the main entry point for pairwise well spacing analysis. The method automatically
        classifies each well pair by alignment (parallel/oblique/perpendicular), routes to the
        appropriate spacing algorithm, and returns a DataFrame with ~40 columns of metrics per pair.

        The computation uses parallel processing with configurable batch sizes to handle large
        datasets efficiently. Results can be returned as a single DataFrame or saved incrementally
        to Parquet files for very large analyses.

        Algorithm Overview
        ------------------
        1. **Precomputation**: Build pair cache with local coordinate frames and lateral lengths
        2. **Midpoint Extraction**: Compute normalized midpoints at specified frac (default 50% MD)
        3. **Direction Classification**: Determine drill directions (EW/NS) from azimuth data
        4. **Geographic Filtering**: Prefilter pairs beyond max_distance_miles (default 20 mi)
        5. **Batch Processing**: Process pairs in parallel batches with progress bar
        6. **Alignment Routing**:
           - **Parallel-like (≤25°)**: Crossline distance |Δy(x)| over x-overlap in i-frame
           - **Oblique (25°-65°)**: Nearest-projection distance series from i to k
           - **Perpendicular (≥65°)**: Same as oblique + optional windowed mean
        7. **Normalization**: Apply effective horizontal distance across alignment types
        8. **Output**: Return combined DataFrame or save batches to Parquet

        Parameters
        ----------
        frac : float, default 0.5
            Fractional position along lateral (0.0-1.0) for midpoint computation.
            0.5 = midpoint, 0.0 = heel, 1.0 = toe. Used for vertical distance and TVD values.

        batch_size : int, default 500_000
            Number of well pairs to process per batch. Controls memory usage.
            Larger batches = faster but more RAM. Reduce if encountering memory errors.

        max_distance_miles : float or None, default 20.0
            Maximum geographic distance (miles) for prefiltering well pairs.
            Uses fast planar approximation. Set to None to compute all N² pairs (slow).

        save_batches_dir : str or None, default None
            If provided, saves each batch as Parquet file to this directory and returns None.
            Useful for very large analyses (>1M pairs) to avoid memory limits.
            Files named: "spacing_batch_0000.parquet", "spacing_batch_0001.parquet", etc.

        use_interpolation : bool, default False
            If True, interpolate midpoint by MD fraction (curvature-aware).
            If False, use simple geometric midpoint between heel and toe (faster).

        step_ft : int, default 100
            Sampling interval (ft) along laterals for spacing computation.
            Smaller = more accurate but slower. Typical range: 50-200 ft.

        n_samples : int or None, default None
            If specified, overrides step_ft to use exactly this many samples per lateral.
            Useful for consistent sampling regardless of lateral length.

        max_crossline_ft : float or None, default 2000.0
            Maximum crossline distance threshold (ft) for parallel pairs.
            Pairs exceeding this in coarse precheck are rejected with "coarse_far".
            Set to None to disable crossline filtering.

        crossline_percentile : float, default 5.0
            Percentile (0-100) for crossline distance quality filter (dy_p5).
            Pairs where p5 > max_crossline_ft are rejected (helps filter bad overlaps).

        ds_crossline_step_ft : int, default 200
            Downsampling step (ft) for coarse crossline precheck on parallel pairs.
            Larger = faster precheck but less accurate filtering. Typical: 100-300 ft.

        emit_rejected : bool, default True
            If True, include rejected pairs in output with reject_reason populated.
            If False, only return valid pairs (reject_reason == '').

        use_pca_axis : bool, default True
            If True, use PCA to define local i-frame axis (handles curved laterals well).
            If False, use simple heel-to-toe vector (faster but less accurate for curves).

        theta_parallel_deg : float, default 25.0
            Maximum angle (degrees) between lateral axes to classify as PARALLEL_LIKE.
            Typical range: 15-30°. Lower = stricter parallel definition.

        theta_perp_deg : float, default 65.0
            Minimum angle (degrees) between lateral axes to classify as PERPENDICULAR.
            Typical range: 60-75°. Higher = stricter perpendicular definition.
            Pairs between theta_parallel and theta_perp are OBLIQUE.

        reject_misaligned : bool, default False
            If True, reject all oblique/perpendicular pairs (only process parallel pairs).
            Useful for analyses focused solely on parallel child wells.

        use_windowed_mean : bool, default False
            If True, compute mean_windowed_ft (mean distance in ±window around closest approach)
            for oblique/perpendicular pairs. Provides localized spacing around contact zone.

        window_ft : float, default 300.0
            Half-width of window (ft) for windowed mean calculation.
            Only used if use_windowed_mean=True.

        contact_threshold_ft : float, default 300.0
            Distance threshold T (ft) for defining "contact" in oblique/perpendicular pairs.
            Computes length of well_i within T ft of well_k (contact_len_i_ft, contact_pct_i).
            Typical range: 200-500 ft.

        coverage_epsilon : float, default 1e-3
            Epsilon value (0-0.5) for defining "interior" projection in coverage metrics.
            Points with projection parameter t in (ε, 1-ε) are considered interior.
            Excludes endpoint projections for more robust coverage calculation.

        Returns
        -------
        pd.DataFrame or None
            If save_batches_dir is None:
                Returns DataFrame with one row per well pair, containing ~40 columns:
                - Identifiers: well_i, well_k
                - Core metrics: horizontal_dist, horizontal_dist_median, vertical_dist,
                  3D_dist, tvd_i, tvd_k
                - Lateral info: LL_i, LL_k, drill_direction_i, drill_direction_k
                - Overlap (parallel): overlap_len_common_ft, overlap_pct_i, overlap_pct_k
                - Alignment: pair_alignment, angle_deg
                - Oblique/perp: min_distance_ft, mean_windowed_ft, contact metrics
                - Direction: direction_axis, direction_to_k_from_i_axis, direction_axis_confidence
                - Quality: n_samples, dy_p5, reject_reason
                - Audit: horizontal_crossline_mean_ft, hz_effective, hz_basis

            If save_batches_dir is provided:
                Returns None. Batches saved as Parquet files in save_batches_dir.
                Use _load_saved_batches(save_batches_dir) to load combined results later.

        Raises
        ------
        ValueError
            If trajectory data is invalid or required columns are missing.
        MemoryError
            If batch_size is too large for available RAM. Reduce batch_size or use save_batches_dir.

        Notes
        -----
        **Performance Considerations:**
        - Time complexity: O(N² * M) where N = wells, M = samples per lateral
        - Memory usage: O(batch_size) due to batch processing
        - Parallelization: Uses all available CPU cores (joblib n_jobs=-1)
        - Typical runtime: ~10-30 min for 1000 wells (500k pairs) on modern hardware

        **Distance Semantics:**
        - horizontal_dist is normalized across alignments (effective horizontal):
          * Parallel: crossline mean from overlap region
          * Oblique/Perp: min_distance_ft (closest approach)
        - vertical_dist is always computed at midpoint TVD (frac parameter)
        - All distances in feet (or input coordinate units)

        **Filtering and Rejection:**
        Pairs can be rejected for multiple reasons (see reject_reason column):
        - "no_overlap_x": No x-overlap in i-frame (parallel pairs only)
        - "coarse_far": Failed coarse downsampled distance check
        - "dy_p5_too_high": 5th percentile exceeds max_crossline_ft
        - "misaligned_angle": Oblique/perp pair when reject_misaligned=True

        **Direction Classification:**
        - Direction axis is constrained based on drill direction:
          * NS-drilled wells → E/W direction axis
          * EW-drilled wells → N/S direction axis
        - Requires 'azimuth' column in trajectory data for best results

        Examples
        --------
        **Example 1: Basic Usage (All Defaults)**

        >>> wsc = WellSpacingCalculator(trajectories_df)
        >>> spacing_df = wsc._calculate_spacing_statistics()
        >>> print(f"Computed spacing for {len(spacing_df)} well pairs")
        >>> print(f"Valid pairs: {(spacing_df['reject_reason'] == '').sum()}")

        **Example 2: Custom Thresholds for Parallel Wells**

        >>> spacing_df = wsc._calculate_spacing_statistics(
        ...     theta_parallel_deg=20.0,      # Stricter parallel definition
        ...     max_crossline_ft=1500.0,      # Tighter crossline filter
        ...     step_ft=50,                   # Higher resolution sampling
        ...     reject_misaligned=True        # Only process parallel pairs
        ... )
        >>> parallel_only = spacing_df[spacing_df['pair_alignment'] == 'parallel_like']

        **Example 3: Large Dataset with Batch Saving**

        >>> # For 2000+ wells, save batches to avoid memory issues
        >>> wsc._calculate_spacing_statistics(
        ...     batch_size=300_000,
        ...     save_batches_dir="./spacing_batches",
        ...     max_distance_miles=15.0
        ... )
        >>> # Returns None, batches saved to ./spacing_batches/
        >>>
        >>> # Later, load all batches
        >>> spacing_df = wsc._load_saved_batches("./spacing_batches")

        **Example 4: High-Resolution Analysis with Contact Metrics**

        >>> spacing_df = wsc._calculate_spacing_statistics(
        ...     step_ft=50,                     # 50 ft sampling
        ...     frac=0.5,                       # Use midpoint
        ...     use_interpolation=True,         # MD-based interpolation
        ...     contact_threshold_ft=250.0,     # 250 ft contact threshold
        ...     use_windowed_mean=True,         # Compute windowed mean
        ...     window_ft=400.0,                # ±400 ft window
        ...     theta_perp_deg=70.0             # Stricter perpendicular (70°+)
        ... )

        **Example 5: Filter Results for Nearest Neighbors**

        >>> spacing_df = wsc._calculate_spacing_statistics()
        >>>
        >>> # Find valid pairs within regulatory spacing
        >>> neighbors = spacing_df[
        ...     (spacing_df['horizontal_dist'] <= 500) &   # 500 ft horizontal
        ...     (spacing_df['vertical_dist'] <= 200) &     # 200 ft vertical
        ...     (spacing_df['reject_reason'] == '')        # Valid only
        ... ].sort_values('horizontal_dist')
        >>>
        >>> print(f"Wells with neighbors: {neighbors['well_i'].nunique()}")

        **Example 6: Analyze by Alignment Type**

        >>> spacing_df = wsc._calculate_spacing_statistics()
        >>>
        >>> # Group by alignment
        >>> by_alignment = spacing_df.groupby('pair_alignment').agg({
        ...     'horizontal_dist': ['mean', 'median', 'std'],
        ...     'vertical_dist': ['mean', 'median'],
        ...     'angle_deg': ['mean', 'min', 'max']
        ... })
        >>> print(by_alignment)

        **Example 7: Export for GIS/Visualization**

        >>> spacing_df = wsc._calculate_spacing_statistics(emit_rejected=False)
        >>>
        >>> # Select key columns for export
        >>> export_cols = [
        ...     'well_i', 'well_k', 'horizontal_dist', 'vertical_dist',
        ...     'pair_alignment', 'direction_to_k_from_i_axis',
        ...     'overlap_pct_i', 'contact_pct_i_T300'
        ... ]
        >>> spacing_df[export_cols].to_csv("well_spacing_results.csv", index=False)

        See Also
        --------
        _load_saved_batches : Load and combine saved batch Parquet files
        debug_pair_spacing : Detailed visualization for single well pair
        _compute_normalized_midpoints : Midpoint extraction logic
        _process_batch : Core batch processing logic
        _apply_effective_horizontal : Distance normalization logic
        DirectionalBenchNeighbors : Downstream neighbor identification using spacing results

        References
        ----------
        - Alignment classification based on lateral axis dot product
        - Crossline spacing computed in local i-frame (PCA or heel-toe axis)
        - Nearest-projection uses polyline distance algorithms from computational geometry
        - Contact metrics inspired by well interference analysis in unconventional reservoirs
        """
        # ═══════════════════════════════════════════════════════════════════════
        self.logger.info("=" * 80)
        self.logger.info("WELL SPACING CALCULATOR - PAIRWISE ANALYSIS PIPELINE")
        self.logger.info("=" * 80)

        pipeline_start = time.time()
        n_wells = len(self.trajectories)
        n_max_pairs = (n_wells * (n_wells - 1)) // 2

        self.logger.info(f"Input: {n_wells:,} wells → {n_max_pairs:,} potential well pairs")
        self.logger.debug(f"Configuration:")
        self.logger.debug(f"  Midpoint fraction: {frac:.1%}")
        self.logger.debug(f"  Batch size: {batch_size:,} pairs")
        self.logger.debug(f"  Geographic filter: {max_distance_miles} miles" if max_distance_miles else "  Geographic filter: None (all pairs)")
        self.logger.debug(f"  Alignment thresholds: parallel ≤{theta_parallel_deg}°, perpendicular ≥{theta_perp_deg}°")
        self.logger.debug(f"  Sampling: step={step_ft} ft" + (f", n_samples={n_samples}" if n_samples else ""))
        self.logger.debug(f"  Contact threshold: {contact_threshold_ft} ft")

        # Step 1: Build the pair cache once (local frames + coarse arrays for precheck)
        self.logger.info("─" * 80)
        self.logger.info("Step 1/5: Building pair cache (local frames + coarse arrays)")
        step1_start = time.time()
        self._build_pair_cache(use_pca_axis, ds_crossline_step_ft)
        step1_elapsed = time.time() - step1_start
        self.logger.info(f"✓ Pair cache built for {n_wells:,} wells ({step1_elapsed:.2f}s)")

        # Step 2: Midpoints + drill directions (vertical distances remain midpoint-based)
        self.logger.info("─" * 80)
        self.logger.info(f"Step 2/5: Computing normalized midpoints (frac={frac:.1%})")
        step2_start = time.time()
        midpoint_df = self._compute_normalized_midpoints(frac=frac, use_interpolation=use_interpolation)
        drill_dirs = self._compute_drill_directions()
        midpoint_df["drill_direction"] = drill_dirs
        step2_elapsed = time.time() - step2_start

        # Direction distribution
        dir_counts = drill_dirs.value_counts()
        dir_summary = ", ".join([f"{dir}={count}" for dir, count in dir_counts.items()])
        self.logger.info(f"✓ Midpoints computed: {len(midpoint_df):,} wells ({step2_elapsed:.2f}s)")
        self.logger.debug(f"  Drill direction distribution: {dir_summary}")

        # 2) Arrays
        ids = midpoint_df.index.to_numpy()
        coords = midpoint_df[["x", "y", "tvd"]].to_numpy()
        lat_lon = midpoint_df[["latitude", "longitude"]].to_numpy()
        directions = midpoint_df["drill_direction"].to_numpy()

        # Step 3: Prefilter pairs (miles)
        self.logger.info("─" * 80)
        self.logger.info("Step 3/5: Geographic prefiltering")
        step3_start = time.time()

        if max_distance_miles is not None:
            lat = lat_lon[:, 0]; lon = lat_lon[:, 1]
            i_idx, k_idx = self._filter_close_pairs(lat, lon, max_distance_miles)
        else:
            i_idx, k_idx = self._get_pairwise_indices(ids)

        step3_elapsed = time.time() - step3_start
        n_filtered_pairs = len(i_idx)
        filter_pct = (n_filtered_pairs / n_max_pairs * 100) if n_max_pairs > 0 else 0

        if max_distance_miles is not None:
            self.logger.info(f"✓ Geographic filter complete: {n_filtered_pairs:,} pairs within {max_distance_miles} mi "
                           f"({filter_pct:.1f}% of total, {step3_elapsed:.2f}s)")
        else:
            self.logger.info(f"✓ No geographic filter: processing all {n_filtered_pairs:,} pairs ({step3_elapsed:.2f}s)")

        pairs = list(zip(i_idx, k_idx))
        batch_generator = list(self._batch_filtered_indices(pairs, batch_size=batch_size))
        n_batches = len(batch_generator)

        self.logger.debug(f"  Batch configuration: {n_batches} batches of ~{batch_size:,} pairs each")

        # Step 4: Batch processing (parallel)
        self.logger.info("─" * 80)
        self.logger.info(f"Step 4/5: Processing {n_batches} batches in parallel")
        if save_batches_dir:
            os.makedirs(save_batches_dir, exist_ok=True)
            self.logger.info(f"  Output mode: Saving batches to {save_batches_dir}")
        else:
            self.logger.info(f"  Output mode: In-memory (returning combined DataFrame)")

        step4_start = time.time()

        def process_and_save(batch_number: int, i_idx: np.ndarray, k_idx: np.ndarray):
            batch_df = self._process_batch(
                i_idx, k_idx, ids, coords, directions,
                step_ft=step_ft,
                n_samples=n_samples,
                max_crossline_ft=max_crossline_ft,
                crossline_percentile=crossline_percentile,
                ds_crossline_step_ft=ds_crossline_step_ft,
                emit_rejected=emit_rejected,
                use_pca_axis=use_pca_axis,
                theta_parallel_deg=theta_parallel_deg,
                theta_perp_deg=theta_perp_deg,
                reject_misaligned=reject_misaligned,
                use_windowed_mean=use_windowed_mean,
                window_ft=window_ft,
                contact_threshold_ft=contact_threshold_ft,
                coverage_epsilon=coverage_epsilon,
            )

            # ▶ Apply effective horizontal ONCE per batch (so saved parquet is already normalized)
            self._apply_effective_horizontal(batch_df, inplace=True, keep_effective_3d_audit=True)

            # ▶ Insert bench columns right after well_k
            well_k_pos = batch_df.columns.get_loc("well_k") + 1
            batch_df.insert(well_k_pos, "bench_i", batch_df["well_i"].map(self._bench_map))
            batch_df.insert(well_k_pos + 1, "bench_k", batch_df["well_k"].map(self._bench_map))

            if save_batches_dir:
                filepath = os.path.join(save_batches_dir, f"spacing_batch_{batch_number:04d}.parquet")
                batch_df.to_parquet(filepath, index=False)
            return batch_df

        tqdm_desc = "Calculating & Saving Batches" if save_batches_dir else "Calculating Spacing"
        tqdm_kwargs = dict(
            desc=f"🚀 {tqdm_desc}",
            dynamic_ncols=True,
            smoothing=0.3,
            bar_format="{desc}: |{bar:40}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ascii="░▒█",
            leave=True,
        )

        results = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
            for batch_num, (i_i, k_i) in tqdm(enumerate(batch_generator), total=n_batches, **tqdm_kwargs):
                batch_result = process_and_save(batch_num, i_i, k_i)
                results.append(batch_result)

        step4_elapsed = time.time() - step4_start

        # Step 5: Finalize output
        self.logger.info("─" * 80)
        self.logger.info("Step 5/5: Finalizing results")
        step5_start = time.time()

        if save_batches_dir:
            self.logger.info(f"✓ All {n_batches} batches saved to {save_batches_dir}")
            self.logger.info(f"  Use _load_saved_batches('{save_batches_dir}') to load combined results")
            step5_elapsed = time.time() - step5_start
            pipeline_elapsed = time.time() - pipeline_start

            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info(f"Total time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.1f} min)")
            self.logger.debug(f"  Step breakdown: Cache={step1_elapsed:.2f}s, Midpoints={step2_elapsed:.2f}s, "
                            f"Filter={step3_elapsed:.2f}s, Batch={step4_elapsed:.2f}s, Finalize={step5_elapsed:.2f}s")
            self.logger.info("=" * 80)
            return None
        else:
            out = pd.concat(results, ignore_index=True)
            # ▶ Apply effective horizontal ONCE globally if not saving batches
            out = self._apply_effective_horizontal(out, inplace=False, keep_effective_3d_audit=True)
            step5_elapsed = time.time() - step5_start

            # Summary statistics
            total_pairs = len(out)
            valid_pairs = (out['reject_reason'] == '').sum()
            rejected_pairs = total_pairs - valid_pairs
            valid_pct = (valid_pairs / total_pairs * 100) if total_pairs > 0 else 0

            alignment_counts = out['pair_alignment'].value_counts()
            alignment_summary = ", ".join([f"{align}={count:,}" for align, count in alignment_counts.items()])

            pipeline_elapsed = time.time() - pipeline_start
            throughput = total_pairs / pipeline_elapsed if pipeline_elapsed > 0 else 0

            self.logger.info(f"✓ Results combined: {total_pairs:,} total pairs")
            self.logger.info(f"  Valid: {valid_pairs:,} ({valid_pct:.1f}%)")
            if rejected_pairs > 0:
                self.logger.debug(f"  Rejected: {rejected_pairs:,} ({100-valid_pct:.1f}%)")
            self.logger.debug(f"  Alignment distribution: {alignment_summary}")

            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info(f"Total time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.1f} min)")
            self.logger.info(f"Throughput: {throughput:.0f} pairs/sec")
            self.logger.debug(f"  Step breakdown: Cache={step1_elapsed:.2f}s, Midpoints={step2_elapsed:.2f}s, "
                            f"Filter={step3_elapsed:.2f}s, Batch={step4_elapsed:.2f}s, Finalize={step5_elapsed:.2f}s")
            self.logger.info("=" * 80)

            return out
    
    def _load_saved_batches(self, batch_folder: str) -> pd.DataFrame:
        """
        Load and combine all spacing batch Parquet files from a directory into a single DataFrame.

        This utility method is designed to work with batch outputs from _calculate_spacing_statistics()
        when save_batches_dir is specified. It automatically discovers all Parquet files in the
        target folder, loads them efficiently, and concatenates them into a unified result set.

        Useful for:
        - Loading results from very large spacing analyses (1M+ pairs)
        - Resuming analysis after batch processing completes
        - Incremental processing workflows
        - Memory-efficient handling of massive datasets

        Parameters
        ----------
        batch_folder : str
            Path to directory containing batch Parquet files.
            Expected filename pattern: "spacing_batch_####.parquet" (e.g., spacing_batch_0000.parquet).
            All .parquet files in the directory will be loaded regardless of naming.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame containing all well pairs from all batches.
            Schema matches output of _calculate_spacing_statistics():
            - ~40 columns per row (well_i, well_k, horizontal_dist, vertical_dist, etc.)
            - One row per well pair
            - Index is reset (not preserved from individual batches)

        Raises
        ------
        FileNotFoundError
            If batch_folder does not exist or is not a directory.
        ValueError
            If no .parquet files are found in batch_folder.
        pandas.errors.ParserError
            If Parquet files are corrupted or have incompatible schemas.

        Notes
        -----
        - Files are loaded in sorted order (lexicographic sort by filename)
        - All Parquet files must have compatible schemas (same columns)
        - Memory usage: O(total_rows × row_size). For very large datasets (10M+ rows),
          consider using Dask or chunked processing instead.
        - Effective horizontal normalization is already applied in saved batches
        - Progress printed to console: number of files found and total rows loaded

        Performance
        -----------
        - Load speed: ~1-5 seconds per 100k rows (depends on disk I/O)
        - Typical batch: 500k pairs ≈ 50-100 MB Parquet ≈ 2-5 seconds load time
        - Concatenation: O(N) where N = total rows across all batches

        Examples
        --------
        **Example 1: Basic Usage After Batch Processing**

        >>> wsc = WellSpacingCalculator(trajectories_df)
        >>>
        >>> # Step 1: Compute and save batches (returns None)
        >>> wsc._calculate_spacing_statistics(
        ...     batch_size=500_000,
        ...     save_batches_dir="./spacing_batches"
        ... )
        >>> # Output: ✅ All batches saved to ./spacing_batches
        >>>
        >>> # Step 2: Load combined results
        >>> spacing_df = wsc._load_saved_batches("./spacing_batches")
        >>> # Output: 🔍 Found 10 batch files. Loading and combining...
        >>> #         ✅ Loaded 5,234,567 rows from all batches.
        >>>
        >>> print(f"Total pairs: {len(spacing_df):,}")
        >>> print(f"Valid pairs: {(spacing_df['reject_reason'] == '').sum():,}")

        **Example 2: Filter After Loading**

        >>> spacing_df = wsc._load_saved_batches("./spacing_batches")
        >>>
        >>> # Filter to valid parallel pairs within 500 ft
        >>> parallel_neighbors = spacing_df[
        ...     (spacing_df['pair_alignment'] == 'parallel_like') &
        ...     (spacing_df['horizontal_dist'] <= 500) &
        ...     (spacing_df['reject_reason'] == '')
        ... ]
        >>>
        >>> print(f"Parallel neighbors: {len(parallel_neighbors):,}")
        >>> parallel_neighbors.to_parquet("parallel_neighbors_500ft.parquet")

        **Example 3: Incremental Processing Workflow**

        >>> # Day 1: Run expensive computation, save batches
        >>> wsc._calculate_spacing_statistics(
        ...     max_distance_miles=20,
        ...     save_batches_dir="./results/run_20260206"
        ... )
        >>>
        >>> # Day 2: Load and analyze without recomputing
        >>> spacing_df = wsc._load_saved_batches("./results/run_20260206")
        >>> summary_stats = spacing_df.groupby('pair_alignment').agg({
        ...     'horizontal_dist': ['mean', 'median', 'std', 'count']
        ... })
        >>> print(summary_stats)

        **Example 4: Verify Batch Integrity**

        >>> import os
        >>> batch_folder = "./spacing_batches"
        >>>
        >>> # Check files before loading
        >>> parquet_files = [f for f in os.listdir(batch_folder) if f.endswith('.parquet')]
        >>> print(f"Found {len(parquet_files)} batch files:")
        >>> for f in sorted(parquet_files):
        ...     size_mb = os.path.getsize(os.path.join(batch_folder, f)) / 1e6
        ...     print(f"  {f}: {size_mb:.1f} MB")
        >>>
        >>> # Load and verify
        >>> spacing_df = wsc._load_saved_batches(batch_folder)
        >>> assert len(spacing_df) > 0, "No data loaded!"
        >>> assert 'horizontal_dist' in spacing_df.columns, "Missing required column!"

        **Example 5: Handle Missing or Corrupted Batches**

        >>> try:
        ...     spacing_df = wsc._load_saved_batches("./nonexistent_folder")
        ... except FileNotFoundError as e:
        ...     print(f"Error: {e}")
        ...     # Fallback: recompute or use alternate source
        >>>
        >>> try:
        ...     spacing_df = wsc._load_saved_batches("./empty_folder")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        ...     # Handle no Parquet files found

        **Example 6: Memory-Efficient Column Selection**

        >>> # For very large datasets, load only needed columns
        >>> batch_folder = "./spacing_batches"
        >>> batch_files = sorted([
        ...     os.path.join(batch_folder, f)
        ...     for f in os.listdir(batch_folder)
        ...     if f.endswith(".parquet")
        ... ])
        >>>
        >>> # Load with column selection (using pandas directly)
        >>> import pandas as pd
        >>> cols = ['well_i', 'well_k', 'horizontal_dist', 'vertical_dist', 'pair_alignment']
        >>> dfs = [pd.read_parquet(f, columns=cols) for f in batch_files]
        >>> spacing_subset = pd.concat(dfs, ignore_index=True)
        >>> print(f"Memory footprint reduced by loading only {len(cols)} columns")

        See Also
        --------
        _calculate_spacing_statistics : Batch computation with save_batches_dir parameter
        pd.read_parquet : Underlying Pandas method for loading Parquet files
        pd.concat : Concatenation method used to combine batches

        Tips
        ----
        - Use Parquet format for 5-10x compression vs CSV
        - Batch files are independent and can be analyzed separately if needed
        - For multi-TB datasets, consider using Dask: dask.dataframe.read_parquet()
        - Verify disk space: batch outputs typically 10-50 MB per 100k pairs
        """

        self.logger.info(f"Loading saved batches from: {batch_folder}")

        if not os.path.isdir(batch_folder):
            raise FileNotFoundError(f"Batch folder '{batch_folder}' not found.")

        batch_files = sorted([
            os.path.join(batch_folder, f)
            for f in os.listdir(batch_folder)
            if f.endswith(".parquet")
        ])

        if not batch_files:
            raise ValueError(f"No Parquet files found in folder '{batch_folder}'.")

        n_files = len(batch_files)
        self.logger.info(f"Found {n_files} batch file(s)")
        print(f"🔍 Found {n_files} batch files. Loading and combining...")

        load_start = time.time()
        dfs = []
        for idx, file in enumerate(batch_files, 1):
            self.logger.debug(f"  Loading batch {idx}/{n_files}: {os.path.basename(file)}")
            dfs.append(pd.read_parquet(file))

        self.logger.debug("  Concatenating batches...")
        combined_df = pd.concat(dfs, ignore_index=True)
        load_elapsed = time.time() - load_start

        n_rows = len(combined_df)
        throughput = n_rows / load_elapsed if load_elapsed > 0 else 0

        self.logger.info(f"✓ Loaded {n_rows:,} rows from {n_files} batches in {load_elapsed:.2f}s ({throughput:.0f} rows/sec)")
        print(f"✅ Loaded {n_rows:,} rows from all batches.")

        return combined_df

    def _filter_close_pairs(self, lat: np.ndarray, lon: np.ndarray, max_distance_miles: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prefilter well pairs by approximate geographic distance to reduce computation.

        Uses a fast planar approximation (lat/lon degree-to-miles conversion) to identify
        pairs within max_distance_miles. This is much faster than computing exact geodetic
        distances for all N² pairs.

        Parameters
        ----------
        lat : np.ndarray
            Latitude coordinates for all wells (degrees). Shape: (N,).
        lon : np.ndarray
            Longitude coordinates for all wells (degrees). Shape: (N,).
        max_distance_miles : float, default 20.0
            Maximum distance threshold (miles). Pairs farther apart are excluded.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (i_idx, k_idx) arrays of well index pairs that pass the distance filter.
            Self-pairs (i == k) are excluded.

        Notes
        -----
        Uses 69 miles/degree for latitude and cos(lat)-adjusted for longitude.
        This is approximate but sufficient for filtering at 20-mile scale.
        """
        n_wells = len(lat)
        n_max_pairs = (n_wells * (n_wells - 1)) // 2
        self.logger.debug(f"    Applying geographic filter: {max_distance_miles} mi threshold")

        lat1, lat2 = np.meshgrid(lat, lat, indexing="ij")
        lon1, lon2 = np.meshgrid(lon, lon, indexing="ij")

        delta_lat = np.abs(lat1 - lat2)
        delta_lon = np.abs(lon1 - lon2)

        miles_per_lat_degree = 69.0
        miles_per_lon_degree = 69.0 * np.cos(np.radians(lat))
        miles_per_lon_degree_matrix = np.add.outer(miles_per_lon_degree, miles_per_lon_degree) / 2.0

        rough_dist_miles = np.sqrt(
            (delta_lat * miles_per_lat_degree)**2 + (delta_lon * miles_per_lon_degree_matrix)**2
        )

        mask = (rough_dist_miles <= max_distance_miles) & (delta_lat + delta_lon > 0)
        i_idx, k_idx = np.where(mask)

        n_filtered = len(i_idx)
        self.logger.debug(f"    ✓ {n_filtered:,} pairs within {max_distance_miles} mi ({n_filtered/n_max_pairs*100:.1f}% of total)")

        return i_idx, k_idx

    def _get_pairwise_indices(self, uwis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate all valid pairwise (i, k) UWI combinations from an array of well IDs,
        excluding self-comparisons (i != k).

        Parameters
        ----------
        uwis : np.ndarray
            Array of unique well identifiers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two 1D arrays of i and k UWIs representing all valid (i, k) pairs.
        """
        # Generate meshgrid of all possible UWI pairs
        n = len(uwis)
        i_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            
        # Exclude self-comparisons (where i_uwi == k_uwi)
        valid_mask = i_idx != k_idx

        return i_idx[valid_mask], k_idx[valid_mask]

    def _batch_filtered_indices(self, pairs: List[Tuple[int, int]], batch_size: int = 1_000_000):
        """
        Vectorized batching of prefiltered well pairs.

        Parameters
        ----------
        pairs : List[Tuple[int, int]]
            List of (i_idx, k_idx) pairs.
        batch_size : int
            Number of pairs per batch.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            i_idx and k_idx arrays for each batch.
        """
        pairs_array = np.array(pairs)  # Convert list of tuples directly to 2D array (N, 2)
        n_pairs = pairs_array.shape[0]

        # Vectorized slicing
        split_indices = np.arange(0, n_pairs, batch_size)

        for start_idx in split_indices:
            end_idx = min(start_idx + batch_size, n_pairs)
            batch = pairs_array[start_idx:end_idx]
            yield batch[:, 0], batch[:, 1]

    def _compute_normalized_midpoints(self, frac: float = 0.5, use_interpolation: bool = True) -> pd.DataFrame:
        """
        Compute normalized position coordinates for each well at a specified fractional MD location.

        This method extracts a representative point along each well's lateral trajectory, used
        for vertical distance calculations and TVD values in spacing analysis. The "normalized"
        position eliminates lateral-length bias by ensuring all wells are compared at equivalent
        relative positions (e.g., all at 50% MD, not at absolute MD values).

        Two computation modes are available:
        1. **Interpolation mode (use_interpolation=True)**: Uses MD-based linear interpolation
           to find the exact position at the fractional MD. Curvature-aware and accurate.
        2. **Geometric mode (use_interpolation=False)**: Simple average of heel and toe positions.
           Faster but less accurate for curved wells or irregular sampling.

        Parameters
        ----------
        frac : float, default 0.5
            Fractional position along measured depth (0.0 to 1.0):
            - 0.0 = heel (starting point of lateral)
            - 0.5 = midpoint (default, most common for regulatory compliance)
            - 1.0 = toe (end point of lateral)
            Values outside [0, 1] are technically allowed but may extrapolate beyond data.

        use_interpolation : bool, default True
            Computation mode:
            - True: MD-based interpolation (recommended for accurate TVD and curvature handling)
            - False: Simple geometric average of heel/toe (faster, ~2-3x speedup)

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by 'uwi' with the following columns:
            - 'x' : float
                UTM Easting coordinate at fractional position (same units as input, typically ft)
            - 'y' : float
                UTM Northing coordinate at fractional position
            - 'tvd' : float
                True vertical depth at fractional position (ft, positive down)
            - 'latitude' : float
                Decimal degrees latitude (WGS84) at fractional position. May be NaN if
                lat/lon not present in trajectory data.
            - 'longitude' : float
                Decimal degrees longitude (WGS84) at fractional position. May be NaN if
                lat/lon not present in trajectory data.

        Notes
        -----
        **Interpolation Mode Details (use_interpolation=True):**
        - Normalizes MD to [0, 1] range per well
        - Uses binary search to find bounding survey stations
        - Linear interpolation between stations for all attributes (x, y, tvd, lat, lon)
        - Respects well curvature and irregular survey spacing
        - Recommended for production analyses

        **Geometric Mode Details (use_interpolation=False):**
        - Simply averages first and last points: (heel + toe) / 2
        - Ignores intermediate trajectory points
        - Fast but inaccurate for curved wells (can be off by 100+ ft for deviated laterals)
        - Useful for quick testing or when exact midpoint accuracy isn't critical

        **TVD Calculation:**
        - TVD is interpolated/averaged same as x, y coordinates
        - Used for vertical_dist computation in spacing analysis
        - Critical for regulatory compliance (many rules specify TVD separation)

        **Missing Lat/Lon Handling:**
        - If latitude/longitude columns absent, returns NaN for those columns
        - Direction classification will fall back to UTM-based azimuth estimation
        - Geodetic direction calculations unavailable without lat/lon

        **Edge Cases:**
        - Wells with single survey point: returns that point (interpolation = geometric)
        - Wells with frac outside [0,1]: extrapolates (may yield unexpected results)
        - Wells with non-monotonic MD: behavior undefined (preprocess to clean data)

        Performance
        -----------
        - Interpolation mode: O(N log M) per well (N = wells, M = avg points per well)
        - Geometric mode: O(N) per well
        - Typical runtime: <1 second for 1000 wells with 100 points each

        Examples
        --------
        **Example 1: Standard Midpoint (50% MD)**

        >>> wsc = WellSpacingCalculator(trajectories_df)
        >>> midpoints = wsc._compute_normalized_midpoints(frac=0.5)
        >>> print(midpoints.head())
        >>>                       x           y       tvd  latitude  longitude
        >>> uwi
        >>> WELL_001      501000.0  4000050.0   10020.0     36.001    -102.001
        >>> WELL_002      501100.0  4000250.0   10020.0     36.003    -102.002

        **Example 2: Compare Interpolation vs Geometric**

        >>> # Interpolation mode (default, accurate)
        >>> midpts_interp = wsc._compute_normalized_midpoints(
        ...     frac=0.5, use_interpolation=True
        ... )
        >>>
        >>> # Geometric mode (fast, approximate)
        >>> midpts_geom = wsc._compute_normalized_midpoints(
        ...     frac=0.5, use_interpolation=False
        ... )
        >>>
        >>> # Compare TVD differences
        >>> tvd_diff = (midpts_interp['tvd'] - midpts_geom['tvd']).abs()
        >>> print(f"Max TVD difference: {tvd_diff.max():.1f} ft")
        >>> print(f"Mean TVD difference: {tvd_diff.mean():.1f} ft")

        **Example 3: Heel, Midpoint, and Toe Comparison**

        >>> heel = wsc._compute_normalized_midpoints(frac=0.0)   # Start
        >>> mid = wsc._compute_normalized_midpoints(frac=0.5)    # Middle
        >>> toe = wsc._compute_normalized_midpoints(frac=1.0)    # End
        >>>
        >>> well_id = 'WELL_001'
        >>> print(f"Heel TVD: {heel.loc[well_id, 'tvd']:.0f} ft")
        >>> print(f"Mid TVD:  {mid.loc[well_id, 'tvd']:.0f} ft")
        >>> print(f"Toe TVD:  {toe.loc[well_id, 'tvd']:.0f} ft")

        **Example 4: Extract Quarter-Points for Analysis**

        >>> # Get positions at 25%, 50%, 75% along lateral
        >>> positions = {}
        >>> for frac_val in [0.25, 0.5, 0.75]:
        ...     positions[f'{int(frac_val*100)}pct'] = wsc._compute_normalized_midpoints(
        ...         frac=frac_val
        ...     )
        >>>
        >>> # Analyze TVD variation along lateral
        >>> well = 'WELL_001'
        >>> tvds = [positions[k].loc[well, 'tvd'] for k in positions.keys()]
        >>> print(f"TVD progression for {well}: {tvds}")

        **Example 5: Benchmark Interpolation vs Geometric**

        >>> import time
        >>>
        >>> # Time interpolation mode
        >>> start = time.time()
        >>> _ = wsc._compute_normalized_midpoints(use_interpolation=True)
        >>> time_interp = time.time() - start
        >>>
        >>> # Time geometric mode
        >>> start = time.time()
        >>> _ = wsc._compute_normalized_midpoints(use_interpolation=False)
        >>> time_geom = time.time() - start
        >>>
        >>> print(f"Interpolation: {time_interp:.3f}s")
        >>> print(f"Geometric: {time_geom:.3f}s")
        >>> print(f"Speedup: {time_interp/time_geom:.1f}x")

        **Example 6: Handle Missing Lat/Lon**

        >>> midpoints = wsc._compute_normalized_midpoints()
        >>>
        >>> # Check if lat/lon available
        >>> has_latlon = midpoints['latitude'].notna().all()
        >>> if has_latlon:
        ...     print("✓ Geodetic coordinates available for direction calculation")
        ... else:
        ...     print("⚠ Missing lat/lon - direction classification will be limited")

        **Example 7: Custom Fractional Position for Regulatory Analysis**

        >>> # Some regulations specify 40% or 60% positions
        >>> midpoints_40pct = wsc._compute_normalized_midpoints(frac=0.4)
        >>> midpoints_60pct = wsc._compute_normalized_midpoints(frac=0.6)
        >>>
        >>> # Compute vertical separation at both positions
        >>> tvd_diff_40 = midpoints_40pct['tvd'].values - midpoints_40pct['tvd'].iloc[0]
        >>> tvd_diff_60 = midpoints_60pct['tvd'].values - midpoints_60pct['tvd'].iloc[0]

        See Also
        --------
        _calculate_spacing_statistics : Uses midpoints for vertical distance computation
        _compute_drill_directions : Complementary method for direction classification
        pd.DataFrame.interpolate : Not used, but similar concept for time series

        References
        ----------
        - Fractional MD concept standard in well trajectory analysis
        - Linear interpolation between survey stations per SPE best practices
        - Midpoint (50% MD) most common for spacing regulations (e.g., Texas RRC)
        """
        n_wells = len(self.trajectories)
        mode_str = "MD interpolation" if use_interpolation else "geometric (heel+toe)/2"
        self.logger.debug(f"  Computing normalized midpoints: frac={frac:.1%}, mode={mode_str}")

        df = self._trajectory_df.copy()
        df = df.sort_values(["uwi", "md"]).reset_index(drop=True)

        if not use_interpolation:
            # Simple geometric midpoint (fast)
            heel_toe_df = (
                df.groupby("uwi")
                .agg(
                    heel_x=("x", "first"),
                    heel_y=("y", "first"),
                    heel_tvd=("tvd", "first"),
                    heel_lat=("latitude", "first"),
                    heel_lon=("longitude", "first"),
                    toe_x=("x", "last"),
                    toe_y=("y", "last"),
                    toe_tvd=("tvd", "last"),
                    toe_lat=("latitude", "last"),
                    toe_lon=("longitude", "last"),
                )
            )

            midpoint_df = pd.DataFrame({
                "x": (heel_toe_df["heel_x"] + heel_toe_df["toe_x"]) / 2,
                "y": (heel_toe_df["heel_y"] + heel_toe_df["toe_y"]) / 2,
                "tvd": (heel_toe_df["heel_tvd"] + heel_toe_df["toe_tvd"]) / 2,
                "latitude": (heel_toe_df["heel_lat"] + heel_toe_df["toe_lat"]) / 2,
                "longitude": (heel_toe_df["heel_lon"] + heel_toe_df["toe_lon"]) / 2,
            })
            midpoint_df.index.name = "uwi"

            has_latlon = midpoint_df["latitude"].notna().sum()
            self.logger.debug(f"  ✓ Midpoints computed for {n_wells:,} wells ({has_latlon}/{n_wells} with lat/lon)")
            return midpoint_df

        # Interpolated midpoint (MD-based)
        min_md = df.groupby("uwi")["md"].transform("min")
        max_md = df.groupby("uwi")["md"].transform("max")
        df["normalized_md"] = (df["md"] - min_md) / (max_md - min_md)

        df["row_index"] = df.groupby("uwi").cumcount()
        df["prev_idx"] = df.groupby("uwi")["normalized_md"].transform(lambda x: x.searchsorted(frac, side="right") - 1)
        df["next_idx"] = df["prev_idx"] + 1
        df["next_idx"] = np.minimum(df["next_idx"], df["row_index"].groupby(df["uwi"]).transform("max"))

        df_prev = df.groupby("uwi").apply(lambda g: g.loc[g["row_index"] == g["prev_idx"].iloc[0]]).reset_index(drop=True)
        df_next = df.groupby("uwi").apply(lambda g: g.loc[g["row_index"] == g["next_idx"].iloc[0]]).reset_index(drop=True)

        merged = pd.merge(df_prev, df_next, on="uwi", suffixes=("_prev", "_next"))

        delta = merged["normalized_md_next"] - merged["normalized_md_prev"]
        delta = delta.replace(0, np.nan)
        ratio = (frac - merged["normalized_md_prev"]) / delta

        midpoint_df = pd.DataFrame({
            "x": merged["x_prev"] + ratio * (merged["x_next"] - merged["x_prev"]),
            "y": merged["y_prev"] + ratio * (merged["y_next"] - merged["y_prev"]),
            "tvd": merged["tvd_prev"] + ratio * (merged["tvd_next"] - merged["tvd_prev"]),
            "latitude": merged["latitude_prev"] + ratio * (merged["latitude_next"] - merged["latitude_prev"]),
            "longitude": merged["longitude_prev"] + ratio * (merged["longitude_next"] - merged["longitude_prev"]),
        })
        midpoint_df["uwi"] = merged["uwi"]
        result = midpoint_df.set_index("uwi")

        has_latlon = result["latitude"].notna().sum()
        self.logger.debug(f"  ✓ Midpoints computed for {n_wells:,} wells ({has_latlon}/{n_wells} with lat/lon)")
        return result
    
    def _compute_drill_directions(self) -> pd.Series:
        """
        Classify each well's drill direction as 'EW' (East-West) or 'NS' (North-South).

        Uses the median azimuth of each well's trajectory to determine the primary
        drilling direction. Wells drilled predominantly east-west (azimuth 45-135° or
        225-315°) are classified as 'EW'; others are classified as 'NS'.

        Returns
        -------
        pd.Series
            Series indexed by 'uwi' with values 'EW' or 'NS'.

        Notes
        -----
        The drill direction is used to determine the axis for spacing direction
        classification: NS-drilled wells use E/W direction labels, and vice versa.
        """
        self.logger.debug("  Computing drill directions from azimuth data")
        median_azimuth = self._trajectory_df.groupby("uwi")["azimuth"].median()
        is_ew = ((median_azimuth >= 45) & (median_azimuth <= 135)) | ((median_azimuth >= 225) & (median_azimuth <= 315))
        result = pd.Series(np.where(is_ew, "EW", "NS"), index=median_azimuth.index, name="drill_direction")

        # Log distribution
        n_ew = (result == "EW").sum()
        n_ns = (result == "NS").sum()
        self.logger.debug(f"  ✓ Drill directions classified: EW={n_ew}, NS={n_ns}")

        return result

    def _axis_component_from_az(self, az_deg: np.ndarray, want_axis: str) -> np.ndarray:
        """
        Convert geodetic azimuth(s) (0°=N, 90°=E) to the signed component on a target axis.

        want_axis:
        - "EW" -> east-west component (sin), + = E, - = W
        - "NS" -> north-south component (cos), + = N, - = S
        """
        az = np.deg2rad(az_deg % 360.0)
        if want_axis == "EW":
            return np.sin(az)       # +E / -W
        elif want_axis == "NS":
            return np.cos(az)       # +N / -S
        else:
            raise ValueError("want_axis must be 'EW' or 'NS'")

    def _axis_label_from_component(self, comp: np.ndarray, want_axis: str,
                                deadband: float = 0.15, tie_tol: float = 0.05
    ) -> tuple[str, float, str]:
        """
        Turn signed axis components into a label + confidence + distribution string.

        - Ignores samples with |component| < deadband (neutral, avoids jitter).
        - Weighted vote by |component|; confidence = winner_weight / total_weight.
        - If near tie, fall back to the sign of the median of kept components.

        Returns:
        label: "E"/"W" if want_axis="EW", or "N"/"S" if want_axis="NS"
        confidence: float in [0,1]
        distribution: e.g. "E:0.83,W:0.17" or "N:0.91,S:0.09"
        """
        comp = np.asarray(comp, dtype=float)
        keep = np.abs(comp) >= float(deadband)
        if not keep.any():
            # Indeterminate; return low-confidence neutral toward positive side
            if want_axis == "EW":
                return "E", 0.0, "E:0.00,W:0.00"
            else:
                return "N", 0.0, "N:0.00,S:0.00"

        c = comp[keep]
        w_pos = float(np.abs(c[c > 0]).sum())
        w_neg = float(np.abs(c[c < 0]).sum())
        tot = w_pos + w_neg if (w_pos + w_neg) > 0 else 1.0

        # provisional winner
        if w_pos >= w_neg:
            winner, conf = ("E" if want_axis == "EW" else "N"), w_pos / tot
            loser = "W" if want_axis == "EW" else "S"
        else:
            winner, conf = ("W" if want_axis == "EW" else "S"), w_neg / tot
            loser = "E" if want_axis == "EW" else "N"

        # tie-break if very close
        if abs(w_pos - w_neg) / tot <= float(tie_tol):
            med = float(np.median(c))
            if med == 0.0:
                # keep original (essentially a tie)
                pass
            else:
                winner = ("E" if want_axis == "EW" else "N") if med > 0 else ("W" if want_axis == "EW" else "S")
                conf = max(w_pos, w_neg) / tot

        dist = (f"E:{w_pos/tot:.2f},W:{w_neg/tot:.2f}" if want_axis == "EW"
                else f"N:{w_pos/tot:.2f},S:{w_neg/tot:.2f}")
        return winner, conf, dist

    def _axis_constrained_direction_from_pairs(
        self,
        lat_i: np.ndarray, lon_i: np.ndarray,
        lat_k: np.ndarray, lon_k: np.ndarray,
        want_axis: str,
        *,
        deadband: float = 0.15,
        tie_tol: float = 0.05,
    ) -> tuple[str, float, str]:
        """
        Given arrays of matched i→k points (same length), compute geodetic azimuths and
        return axis-constrained label/conf/dist using the helpers above.
        """
        geod = Geod(ellps="WGS84")
        az12, _, _ = geod.inv(lon_i, lat_i, lon_k, lat_k)   # 0°=N, 90°=E
        comp = self._axis_component_from_az(az12, want_axis)
        return self._axis_label_from_component(comp, want_axis, deadband=deadband, tie_tol=tie_tol)

    def _axis_constrained_direction_over_overlap(
        self,
        Xi_seg: np.ndarray, Xk_seg: np.ndarray,
        lat_i_seg: np.ndarray, lon_i_seg: np.ndarray,
        lat_k_seg: np.ndarray, lon_k_seg: np.ndarray,
        want_axis: str,
        *,
        step_ft: int,
        n_samples: int | None,
        deadband: float = 0.15,
        tie_tol: float = 0.05,
    ) -> tuple[str, float, str]:
        """
        Use the same stationing you use for spacing over the clipped overlap and
        produce an axis-constrained direction label/conf/dist.
        """
        # choose sample count exactly as your spacing logic does
        si = self._arclength(Xi_seg); Li = si[-1]
        sk = self._arclength(Xk_seg); Lk = sk[-1]
        Lmin = max(min(Li, Lk), 1e-6)

        if n_samples is None:
            step = max(int(step_ft or 100), 1)
            n = max(int(np.floor(Lmin / step)) + 1, 2)
        else:
            n = max(int(n_samples), 2)
        t = np.linspace(0.0, 1.0, n)

        # interpolate lat/lon by arclength on both segments
        lat_i = self._interp_attr_by_arclength(Xi_seg, lat_i_seg, t * Li)
        lon_i = self._interp_attr_by_arclength(Xi_seg, lon_i_seg, t * Li)
        lat_k = self._interp_attr_by_arclength(Xk_seg, lat_k_seg, t * Lk)
        lon_k = self._interp_attr_by_arclength(Xk_seg, lon_k_seg, t * Lk)

        return self._axis_constrained_direction_from_pairs(
            lat_i, lon_i, lat_k, lon_k, want_axis,
            deadband=deadband, tie_tol=tie_tol
        )

    def _build_pair_cache(
        self,
        use_pca_axis: bool,
        ds_crossline_step_ft: int,
        ds_min_points: int = 16,
        ds_max_points: int = 64,
    ):
        """
        Precompute per-well:
        - origin, ex, ey (local frame metadata)
        - coarse resample XY (same length across wells) for vectorized cross-line precheck
        """
        n_wells = len(self.trajectories)
        self.logger.debug(f"  Building pair cache for {n_wells:,} wells")
        self.logger.debug(f"    Axis mode: {'PCA' if use_pca_axis else 'heel-to-toe'}")
        self.logger.debug(f"    Downsampling step: {ds_crossline_step_ft} ft")

        # Compute each well's lateral length to pick a *global* coarse point count
        lengths = {}
        for uwi, df in self.trajectories.items():
            XY = df.sort_values("md")[["x","y"]].to_numpy()
            d = np.hypot(np.diff(XY[:,0]), np.diff(XY[:,1]))
            lengths[uwi] = float(d.sum())

        # Choose a *single* coarse count using median length / step, then clamp
        median_len = np.median(list(lengths.values())) if lengths else 3000.0
        min_len = min(lengths.values()) if lengths else 0.0
        max_len = max(lengths.values()) if lengths else 0.0
        m_guess = int(np.ceil(max(median_len, 1.0) / max(ds_crossline_step_ft, 1))) + 1
        M_ds = int(np.clip(m_guess, ds_min_points, ds_max_points))

        self.logger.debug(f"    Lateral length stats: min={min_len:.0f} ft, median={median_len:.0f} ft, max={max_len:.0f} ft")
        self.logger.debug(f"    Coarse resampling: {M_ds} points per well")

        cache = {
            "origin": {},
            "ex": {},
            "ey": {},
            "XY_coarse": {},
            "M_ds": M_ds,
            "use_pca_axis": use_pca_axis,
            "LL": lengths      # <-- ADD: per-well lateral length in feet
        }

        for uwi, df in self.trajectories.items():
            df = df.sort_values("md")
            XY = df[["x","y"]].to_numpy()

            # local frame metadata
            pts = XY
            heel, toe = pts[0], pts[-1]
            origin = heel.copy()
            if use_pca_axis:
                C = pts - pts.mean(0)
                _, _, Vt = np.linalg.svd(C, full_matrices=False)
                ex = Vt[0]
                if np.dot(ex, toe - heel) < 0:
                    ex = -ex
            else:
                v = toe - heel
                ex = v / (np.linalg.norm(v) + 1e-12)
            ey = np.array([-ex[1], ex[0]])

            cache["origin"][uwi] = origin
            cache["ex"][uwi] = ex
            cache["ey"][uwi] = ey

            # coarse resample to common length M_ds by arclength
            d = np.hypot(np.diff(XY[:,0]), np.diff(XY[:,1]))
            s = np.concatenate([[0.0], np.cumsum(d)])
            L = s[-1] if s.size else 0.0
            if L <= 0.0:
                XYc = np.repeat(XY[:1], M_ds, axis=0)
            else:
                t = np.linspace(0.0, 1.0, M_ds)
                sx = np.interp(t*L, s, XY[:,0])
                sy = np.interp(t*L, s, XY[:,1])
                XYc = np.column_stack([sx, sy])

            cache["XY_coarse"][uwi] = XYc

        self._paircache = cache
        self.logger.debug(f"  ✓ Pair cache built: {n_wells:,} wells with local frames and coarse arrays")

    def _build_local_frame_from_i(self, df_i: pd.DataFrame, use_pca_axis: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build well_i's local frame from its UTM x,y points.
        origin: heel (first by MD); ex: along-lateral (PCA or heel→toe); ey: 90° CCW from ex.
        """
        pts = df_i.sort_values("md")[["x","y"]].to_numpy()
        heel, toe = pts[0], pts[-1]
        origin = heel.copy()

        if use_pca_axis:
            C = pts - pts.mean(0)
            _, _, Vt = np.linalg.svd(C, full_matrices=False)
            ex = Vt[0]
            if np.dot(ex, toe - heel) < 0:
                ex = -ex
        else:
            v = toe - heel
            ex = v / (np.linalg.norm(v) + 1e-12)

        ey = np.array([-ex[1], ex[0]])
        return origin, ex, ey
    
    def _project_xy_to_frame(self, df: pd.DataFrame, origin: np.ndarray, ex: np.ndarray, ey: np.ndarray
    ) -> np.ndarray:
        """Return Nx2 array of (x_local, y_local) from UTM x,y."""
        XY = df[["x","y"]].to_numpy()
        R = XY - origin
        return np.column_stack([R @ ex, R @ ey])
    
    def _clip_polyline_by_x_band(self, X: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """
        Clip polyline X[:,0]=x, X[:,1]=y to x ∈ [x_lo, x_hi] and INSERT boundary
        intersection points by linear interpolation. Guarantees ≥2 points if the
        band intersects the polyline.
        """
        x_lo, x_hi = band
        x = X[:, 0]
        pts = []

        for j in range(len(X) - 1):
            x0, x1 = x[j], x[j + 1]
            P0, P1 = X[j], X[j + 1]

            seg_min, seg_max = (x0, x1) if x0 <= x1 else (x1, x0)
            if seg_max < x_lo or seg_min > x_hi:
                continue  # segment outside band

            # keep start if inside
            if x_lo <= x0 <= x_hi:
                pts.append(P0)

            # intersections with boundaries (strict crossing)
            for xb in (x_lo, x_hi):
                denom = (x1 - x0)
                if denom != 0.0 and (x0 - xb) * (x1 - xb) < 0.0:
                    t = (xb - x0) / denom
                    pts.append(P0 + t * (P1 - P0))

            # keep end if last seg and inside
            if j == len(X) - 2 and (x_lo <= x1 <= x_hi):
                pts.append(P1)

        if not pts:
            return np.empty((0, 2))

        P = np.vstack(pts)
        # drop exact duplicates while preserving order
        keep = np.ones(len(P), dtype=bool)
        if len(P) > 1:
            dup = np.all(np.isclose(np.diff(P, axis=0), 0.0, atol=1e-9), axis=1)
            keep[1:] = ~dup
        return P[keep]
    
    def _clip_polyline_with_latlon_by_x_band(
        self,
        X: np.ndarray,         # (N,2) local-frame (x,y)
        lat: np.ndarray,       # (N,)
        lon: np.ndarray,       # (N,)
        band: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (X_clip, lat_clip, lon_clip), each with >=2 points if band intersects.
        Boundary points are linearly interpolated (for both XY and lat/lon).
        """
        x_lo, x_hi = band
        x = X[:, 0]
        pts_xy, pts_lat, pts_lon = [], [], []

        for j in range(len(X) - 1):
            x0, x1 = x[j], x[j + 1]
            P0, P1 = X[j], X[j + 1]
            lat0, lat1 = lat[j], lat[j + 1]
            lon0, lon1 = lon[j], lon[j + 1]

            seg_min, seg_max = (x0, x1) if x0 <= x1 else (x1, x0)
            if seg_max < x_lo or seg_min > x_hi:
                continue  # segment entirely outside band

            # keep start if inside
            if x_lo <= x0 <= x_hi:
                pts_xy.append(P0); pts_lat.append(lat0); pts_lon.append(lon0)

            # intersections with x_lo and x_hi (strict crossings)
            for xb in (x_lo, x_hi):
                denom = (x1 - x0)
                if denom != 0.0 and (x0 - xb) * (x1 - xb) < 0.0:
                    t = (xb - x0) / denom
                    pts_xy.append(P0 + t * (P1 - P0))
                    pts_lat.append(lat0 + t * (lat1 - lat0))
                    pts_lon.append(lon0 + t * (lon1 - lon0))

            # keep end if last seg and inside
            if j == len(X) - 2 and (x_lo <= x1 <= x_hi):
                pts_xy.append(P1); pts_lat.append(lat1); pts_lon.append(lon1)

        if not pts_xy:
            return np.empty((0, 2)), np.empty((0,)), np.empty((0,))

        XYc = np.vstack(pts_xy)
        latc = np.asarray(pts_lat, dtype=float)
        lonc = np.asarray(pts_lon, dtype=float)

        # drop adjacent duplicates (numeric noise)
        if len(XYc) > 1:
            dup = np.all(np.isclose(np.diff(XYc, axis=0), 0.0, atol=1e-9), axis=1)
            keep = np.ones(len(XYc), dtype=bool); keep[1:] = ~dup
            XYc, latc, lonc = XYc[keep], latc[keep], lonc[keep]
        return XYc, latc, lonc
    
    def _interp_y_of_x(self, X: np.ndarray, x_targets: np.ndarray) -> np.ndarray:
        """
        Piecewise-linear interpolation of y(x) on a 2D polyline X[:,0]=x, X[:,1]=y.
        Assumes x is roughly monotonic *within the clipped overlap*.
        Handles duplicate x by collapsing them.
        """
        x = X[:, 0]; y = X[:, 1]
        order = np.argsort(x)
        x_sorted = x[order]; y_sorted = y[order]
        keep = np.concatenate([[True], np.diff(x_sorted) > 1e-9])  # drop duplicate x
        x_sorted = x_sorted[keep]; y_sorted = y_sorted[keep]

        if x_sorted.size == 1:
            # Degenerate segment: constant y
            return np.full_like(x_targets, y_sorted[0], dtype=float)

        x_lo, x_hi = x_sorted[0], x_sorted[-1]
        xq = np.clip(x_targets, x_lo, x_hi)
        return np.interp(xq, x_sorted, y_sorted)

    def _nearest_distances_to_polyline(
        self, P: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized nearest point from many points P (n,2) to a polyline X (m,2).
        Returns (d, j, t) where:
        d: (n,) distances,
        j: (n,) segment indices (X[j] -> X[j+1]) or -1 if no segments,
        t: (n,) segment parameters in [0,1] or NaN if undefined.
        """
        m = X.shape[0]
        n = P.shape[0]
        # Degenerate: no segments
        if m == 0:
            return np.full(n, np.nan), np.full(n, -1, dtype=int), np.full(n, np.nan)
        # Degenerate: single-point "polyline" — nearest to that point
        if m == 1:
            diff = P - X[0]
            d = np.sqrt((diff ** 2).sum(axis=1))
            j = np.zeros(n, dtype=int)
            t_sel = np.zeros(n, dtype=float)
            return d, j, t_sel

        A = X[:-1]                  # (m-1,2)
        B = X[1:]                   # (m-1,2)
        AB = B - A                  # (m-1,2)

        AP = P[:, None, :] - A[None, :, :]              # (n,m-1,2)
        denom = (AB[None, :, 0]**2 + AB[None, :, 1]**2) # (1,m-1)
        denom = np.where(denom == 0.0, 1e-12, denom)

        t = (AP[..., 0]*AB[None, :, 0] + AP[..., 1]*AB[None, :, 1]) / denom  # (n,m-1)
        t = np.clip(t, 0.0, 1.0)

        Q = A[None, :, :] + t[..., None] * AB[None, :, :]  # (n,m-1,2)
        diff = Q - P[:, None, :]
        d2 = diff[..., 0]**2 + diff[..., 1]**2             # (n,m-1)

        j = np.argmin(d2, axis=1)                          # (n,)
        d = np.sqrt(d2[np.arange(P.shape[0]), j])
        t_sel = t[np.arange(P.shape[0]), j]
        return d, j, t_sel

    def _crossline_spacing_from_overlap(
        self,
        Xi_seg: np.ndarray, Xk_seg: np.ndarray,
        *,
        step_ft: int,
        n_samples: Optional[int] = None,
    ) -> Tuple[float, int, float]:
        """
        Map-style crossline spacing:
        - Build a common x-grid across the TRUE overlap band (in i-frame).
        - Interpolate y_i(x) and y_k(x).
        - Return mean(|Δy(x)|), number of stations, and overlap length in x.

        Returns: (cross_mean, n_stations, overlap_len_x)
        """
        if Xi_seg.size == 0 or Xk_seg.size == 0:
            return float("nan"), 0, 0.0

        xi_min, xi_max = float(Xi_seg[:,0].min()), float(Xi_seg[:,0].max())
        xk_min, xk_max = float(Xk_seg[:,0].min()), float(Xk_seg[:,0].max())
        x_lo, x_hi = max(xi_min, xk_min), min(xi_max, xk_max)
        if not (x_hi > x_lo):
            return float("nan"), 0, 0.0

        step = max(int(step_ft or 100), 1)
        n = max(int(np.floor((x_hi - x_lo)/step)) + 1, 2)  # works even if overlap < step
        xgrid = np.linspace(x_lo, x_hi, n)

        yi = self._interp_y_of_x(Xi_seg, xgrid)
        yk = self._interp_y_of_x(Xk_seg, xgrid)
        dy = np.abs(yk - yi)

        cross_mean = float(dy.mean())
        return cross_mean, int(n), float(x_hi - x_lo)
    
    def _crossline_spacing_median_from_overlap(
        self,
        Xi_seg: np.ndarray, Xk_seg: np.ndarray,
        *,
        step_ft: int,
        n_samples: Optional[int] = None,
    ) -> Tuple[float, int, float]:
        """
        Map-style crossline spacing (median):
        - Build a common x-grid across the TRUE overlap band (in i-frame).
        - Interpolate y_i(x) and y_k(x).
        - Return median(|Δy(x)|), number of stations, and overlap length in x.

        Returns: (cross_median, n_stations, overlap_len_x)
        """
        if Xi_seg.size == 0 or Xk_seg.size == 0:
            return float("nan"), 0, 0.0

        xi_min, xi_max = float(Xi_seg[:,0].min()), float(Xi_seg[:,0].max())
        xk_min, xk_max = float(Xk_seg[:,0].min()), float(Xk_seg[:,0].max())
        x_lo, x_hi = max(xi_min, xk_min), min(xi_max, xk_max)
        if not (x_hi > x_lo):
            return float("nan"), 0, 0.0

        step = max(int(step_ft or 100), 1)
        n = max(int(np.floor((x_hi - x_lo)/step)) + 1, 2)
        xgrid = np.linspace(x_lo, x_hi, n)

        yi = self._interp_y_of_x(Xi_seg, xgrid)
        yk = self._interp_y_of_x(Xk_seg, xgrid)
        dy = np.abs(yk - yi)

        cross_median = float(np.median(dy))
        return cross_median, int(n), float(x_hi - x_lo)
    
    def _spacing_from_overlap(
        self,
        Xi_seg: np.ndarray,           # (Ni,2)   i-frame clipped segment (x=along-i, y=crossline)
        Xk_seg: np.ndarray,           # (Nk,2)   i-frame clipped segment for k
        step_ft: Optional[int],
        n_samples: Optional[int],
    ) -> Tuple[float, int, float]:
        """
        Map-style crossline spacing over the clipped overlap.
        - Sample both segments by arclength (same # of samples).
        - Return mean |Δy| (crossline), the number of samples, and the length used (min of segment lengths).
        """
        # --- helpers used elsewhere in the class ---
        # self._arclength(X): cumulative length of 2D polyline
        # self._interp_by_arclength(X, s_targets): interpolate XY at arclengths

        si = self._arclength(Xi_seg); Li = float(si[-1]) if si.size else 0.0
        sk = self._arclength(Xk_seg); Lk = float(sk[-1]) if sk.size else 0.0

        # if either segment collapsed, treat as no usable overlap
        if Li <= 0.0 or Lk <= 0.0:
            return float("nan"), 0, 0.0

        # use the smaller physical length as the "overlap length" for reporting
        Lmin = min(Li, Lk)

        # choose sample count
        if n_samples is None:
            step = max(int(step_ft or 100), 1)
            n = max(int(np.floor(Lmin / step)) + 1, 2)   # at least 2 samples
        else:
            n = max(int(n_samples), 2)

        t = np.linspace(0.0, 1.0, n)
        # arclength sampling within each segment
        Pi = self._interp_by_arclength(Xi_seg, t * Li)
        Pk = self._interp_by_arclength(Xk_seg, t * Lk)

        # crossline difference is simply Δy in the i-frame
        dy = np.abs(Pk[:, 1] - Pi[:, 1])
        mean_crossline = float(np.nanmean(dy))

        return mean_crossline, int(n), float(Lmin)

    def _bin4_from_geod_az(self, az_deg_from_north: np.ndarray) -> np.ndarray:
        """
        Bin geodetic azimuths (0°=N, 90°=E, 180°=S, 270°=W) into 4 compass labels.
        Edges at 45°,135°,225°,315°. Returns array of {"N","E","S","W"}.
        """
        a = az_deg_from_north % 360.0
        idx = ((a + 45.0) // 90.0).astype(int) % 4  # 0:N, 1:E, 2:S, 3:W
        return self._DIR4_LABELS[idx]

    def _arclength(self, X: np.ndarray) -> np.ndarray:
        """Cumulative arclength of a 2D polyline X[:,0]=x, X[:,1]=y."""
        X = np.asarray(X)
        if X.size == 0 or len(X) < 2:
            return np.array([0.0], dtype=float)
        d = np.hypot(np.diff(X[:, 0]), np.diff(X[:, 1]))
        return np.concatenate([[0.0], np.cumsum(d)])

    def _interp_by_arclength(self, X: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
        """
        Interpolate XY by arclength along polyline X at distances s_targets.
        Collapses zero-length segments to avoid duplicates.
        """
        X = np.asarray(X)
        s = self._arclength(X)
        # drop zero-length steps to keep interp well-defined
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        s, Xc = s[keep], X[keep]
        if s.size == 1:  # degenerate — return the single point repeated
            xi = np.full_like(s_targets, Xc[0, 0], dtype=float)
            yi = np.full_like(s_targets, Xc[0, 1], dtype=float)
        else:
            xi = np.interp(s_targets, s, Xc[:, 0])
            yi = np.interp(s_targets, s, Xc[:, 1])
        return np.column_stack([xi, yi])

    def _interp_attr_by_arclength(self, X: np.ndarray, attr: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
        """
        Interpolate a 1D attribute sampled along polyline X at distances s_targets.
        """
        X = np.asarray(X); attr = np.asarray(attr)
        s = self._arclength(X)
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        s, a = s[keep], attr[keep]
        if s.size == 1:
            return np.full_like(s_targets, a[0], dtype=float)
        return np.interp(s_targets, s, a)

    def _modal_direction_geodetic_over_overlap(
        self,
        Xi_seg: np.ndarray, Xk_seg: np.ndarray,          # local XY clipped segments
        lat_i_seg: np.ndarray, lon_i_seg: np.ndarray,    # clipped lat/lon for i
        lat_k_seg: np.ndarray, lon_k_seg: np.ndarray,    # clipped lat/lon for k
        step_ft: Optional[int], n_samples: Optional[int]
    ) -> Tuple[str, float, str]:
        """
        Returns (modal_direction_8way, confidence, distribution_string)
        using geodetic bearings at the same sample grid as spacing.
        """
        # Sample count (match spacing logic)
        si = self._arclength(Xi_seg); Li = si[-1]
        sk = self._arclength(Xk_seg); Lk = sk[-1]
        Lmin = max(min(Li, Lk), 1e-6)

        if n_samples is None:
            step = max(int(step_ft or 100), 1)
            n = max(int(np.floor(Lmin / step)) + 1, 2)
        else:
            n = max(int(n_samples), 2)
        t = np.linspace(0.0, 1.0, n)

        # Interpolate lat/lon by arclength within each clipped segment
        s_i = t * Li
        s_k = t * Lk
        lat_i = self._interp_attr_by_arclength(Xi_seg, lat_i_seg, s_i)
        lon_i = self._interp_attr_by_arclength(Xi_seg, lon_i_seg, s_i)
        lat_k = self._interp_attr_by_arclength(Xk_seg, lat_k_seg, s_k)
        lon_k = self._interp_attr_by_arclength(Xk_seg, lon_k_seg, s_k)

        # Geodetic forward azimuth i→k at each sample
        geod = Geod(ellps="WGS84")
        az12, _, _ = geod.inv(lon_i, lat_i, lon_k, lat_k)  # degrees, vectorized

        # 8-way bins + aggregation
        labels = self._bin4_from_geod_az(az12)  # az12 is 0°=N CW from Geod.inv
        # mode + confidence
        uniq, counts = np.unique(labels, return_counts=True)
        best_idx = np.argmax(counts)
        mode = uniq[best_idx]
        conf = counts[best_idx] / float(n)

        # compact distribution string for QA
        parts = [f"{u}:{c/float(n):.2f}" for u, c in sorted(zip(uniq, counts), key=lambda z: -z[1])]
        dist_str = ",".join(parts)
        return mode, conf, dist_str
    
    def _classify_alignment(
        self,
        ex_i: np.ndarray,
        ex_k: np.ndarray,
        theta_parallel_deg: float,
        theta_perp_deg: float
    ) -> Tuple[AlignmentType, float]:
        """
        Classify the alignment between two wells based on their lateral axis directions.

        Parameters
        ----------
        ex_i : np.ndarray
            Unit vector along well_i's lateral axis. Shape: (2,).
        ex_k : np.ndarray
            Unit vector along well_k's lateral axis. Shape: (2,).
        theta_parallel_deg : float
            Maximum angle (degrees) to be classified as PARALLEL_LIKE.
        theta_perp_deg : float
            Minimum angle (degrees) to be classified as PERPENDICULAR.

        Returns
        -------
        Tuple[AlignmentType, float]
            (alignment_type, angle_deg) where angle_deg is the actual angle between axes.

        Notes
        -----
        Uses absolute dot product to be direction-agnostic (heel-to-toe vs toe-to-heel).
        Angle ranges: [0, theta_parallel] → PARALLEL_LIKE, [theta_perp, 90] → PERPENDICULAR,
        otherwise → OBLIQUE.
        """
        # use absolute dot (direction-agnostic)
        dot_abs = float(np.abs(float(np.dot(ex_i, ex_k))))
        angle_deg = float(np.degrees(np.arccos(np.clip(dot_abs, -1.0, 1.0))))
        if angle_deg <= float(theta_parallel_deg):
            return AlignmentType.PARALLEL_LIKE, angle_deg
        if angle_deg >= float(theta_perp_deg):
            return AlignmentType.PERPENDICULAR, angle_deg
        return AlignmentType.OBLIQUE, angle_deg

    def _compute_pair_metrics_and_artifacts(
        self,
        uwi_i: Any,
        uwi_k: Any,
        *,
        step_ft: int,
        n_samples: Optional[int],
        max_crossline_ft: Optional[float],
        crossline_percentile: float,
        ds_crossline_step_ft: int,
        use_pca_axis: bool,
        theta_parallel_deg: float,
        theta_perp_deg: float,
        reject_misaligned: bool,
        use_windowed_mean: bool,
        window_ft: float,
        contact_threshold_ft: float,
        coverage_epsilon: float,
        drill_direction_i: Optional[str] = None,
        drill_direction_k: Optional[str] = None,
        tvd_i: Optional[float] = None,
        tvd_k: Optional[float] = None,
        want_artifacts: bool = True,
    ) -> Tuple[SpacingResult, Optional[PairArtifacts]]:
        """
        Compute all spacing and direction metrics for a single well pair.

        This is the unified computation engine that handles both parallel-like pairs
        (crossline spacing over overlap) and oblique/perpendicular pairs (nearest-projection
        distance series). It produces consistent results for both the batch processor
        and the debug_pair_spacing visualizer.

        Parameters
        ----------
        uwi_i, uwi_k : Any
            Unique identifiers for the well pair.
        step_ft : int
            Sampling step size along laterals (ft).
        n_samples : int or None
            If specified, overrides step_ft to use exactly this many samples.
        max_crossline_ft : float or None
            Maximum crossline distance threshold for filtering.
        crossline_percentile : float
            Percentile (0-100) for crossline distance guardrail (e.g., 5.0 for p5).
        ds_crossline_step_ft : int
            Downsampling step for coarse-resolution prefiltering.
        use_pca_axis : bool
            If True, use PCA to define the i-frame axis; else use heel-to-toe vector.
        theta_parallel_deg : float
            Maximum angle for PARALLEL_LIKE classification (degrees).
        theta_perp_deg : float
            Minimum angle for PERPENDICULAR classification (degrees).
        reject_misaligned : bool
            If True, reject oblique/perpendicular pairs (return NaN metrics).
        use_windowed_mean : bool
            If True, compute mean distance in ±window_ft around closest approach.
        window_ft : float
            Half-width of window for windowed mean (ft).
        contact_threshold_ft : float
            Distance threshold T for contact length metrics (ft).
        coverage_epsilon : float
            Fraction of segment ends to exclude for "interior" projection (0-0.5).
        drill_direction_i, drill_direction_k : str or None
            Pre-computed drill directions ('EW' or 'NS'). If None, computed from azimuth.
        tvd_i, tvd_k : float or None
            Pre-computed TVD values. If None, computed as midpoint TVD from trajectory.
        want_artifacts : bool, default True
            If True, populate PairArtifacts for debug visualization.

        Returns
        -------
        Tuple[SpacingResult, Optional[PairArtifacts]]
            (result, artifacts) where artifacts is None if want_artifacts=False.
        """
        # DEBUG-level logging (called many times per batch)
        self.logger.debug(f"      Computing pair: {uwi_i} → {uwi_k}")

        # Cache build if needed
        if getattr(self, "_paircache", None) is None or self._paircache.get("use_pca_axis", None) != use_pca_axis:
            self._build_pair_cache(use_pca_axis, ds_crossline_step_ft)
        cache = self._paircache
        LL_cache = cache["LL"]

        # Pull full-resolution trajectories
        df_i = self.trajectories[uwi_i].sort_values("md")
        df_k = self.trajectories[uwi_k].sort_values("md")
        Xi_utm = df_i[["x", "y"]].to_numpy()
        Xk_utm = df_k[["x", "y"]].to_numpy()

        # Lat/lon availability
        has_ll = ("latitude" in df_i.columns) and ("longitude" in df_i.columns) and \
                ("latitude" in df_k.columns) and ("longitude" in df_k.columns)
        geod = Geod(ellps="WGS84") if has_ll else None

        # Build i-frame metadata (origin, ex, ey)
        origin_i, ex_i, ey_i = self._build_local_frame_from_i(df_i, use_pca_axis=use_pca_axis)
        # ex for k in its own frame (or global PCA) to classify
        ex_k_global = cache["ex"][uwi_k]  # already consistent with use_pca_axis in cache

        # Alignment class
        align_type, angle_deg = self._classify_alignment(ex_i, ex_k_global, theta_parallel_deg, theta_perp_deg)
        self.logger.debug(f"        Alignment: {align_type.name.lower()}, angle={angle_deg:.1f}°")

        # Decide direction axis (same rule you use elsewhere)
        # If i is NS → E/W; if i is EW → N/S. If not provided, compute.
        if (drill_direction_i is None) or (drill_direction_k is None):
            # Compute median az by group (like your _compute_drill_directions)
            median_az = self._trajectory_df.groupby("uwi")["azimuth"].median()
            is_ew = ((median_az >= 45) & (median_az <= 135)) | ((median_az >= 225) & (median_az <= 315))
            dir_map = pd.Series(np.where(is_ew, "EW", "NS"), index=median_az.index).to_dict()
            dir_i = dir_map.get(uwi_i, "EW")
            dir_k = dir_map.get(uwi_k, "EW")
        else:
            # directions array indexed by self._compute_normalized_midpoints() index order (ids)
            # If you call this from debug (uwi strings), we won't have the index; fallback above is fine.
            # From batch, pass in the 2 labels explicitly:
            dir_i = drill_direction_i
            dir_k = drill_direction_k
        # axis used for direction classification
        dir_axis = "EW" if dir_i == "NS" else "NS"

        # Project full polylines to i-frame
        df_i_sorted = df_i  # already sorted
        df_k_sorted = df_k
        Xi_if = self._project_xy_to_frame(df_i_sorted, origin_i, ex_i, ey_i)
        Xk_if = self._project_xy_to_frame(df_k_sorted, origin_i, ex_i, ey_i)

        # Defaults for direction fields
        dir_mode_axis = None
        dir_conf_axis = np.nan
        dir_dist_axis = ""
        theta_rose = np.array([], dtype=float)

        # vertical
        if (tvd_i is not None) and (tvd_k is not None):
            vertical = abs(float(tvd_k) - float(tvd_i))
        else:
            # same fallback
            tvd_i = float((df_i["tvd"].iloc[0] + df_i["tvd"].iloc[-1]) / 2.0) if "tvd" in df_i.columns else 0.0
            tvd_k = float((df_k["tvd"].iloc[0] + df_k["tvd"].iloc[-1]) / 2.0) if "tvd" in df_k.columns else 0.0
            vertical = abs(tvd_k - tvd_i)

        # Prepare artifacts container if requested
        artifacts = PairArtifacts() if want_artifacts else None
        if want_artifacts:
            artifacts.Xi_utm = Xi_utm
            artifacts.Xk_utm = Xk_utm
            artifacts.Xi_if = Xi_if
            artifacts.Xk_if = Xk_if
            artifacts.dir_axis = dir_axis
            artifacts.has_latlon = bool(has_ll)

        # Parallel-like route: x-overlap + crossline |Δy(x)|
        if align_type is AlignmentType.PARALLEL_LIKE:
            xi_min, xi_max = float(Xi_if[:, 0].min()), float(Xi_if[:, 0].max())
            xk_min, xk_max = float(Xk_if[:, 0].min()), float(Xk_if[:, 0].max())
            x_lo, x_hi = max(xi_min, xk_min), min(xi_max, xk_max)

            if not (x_hi > x_lo):
                # no overlap; treat as reject in parallel bucket
                result = SpacingResult(
                    well_i=uwi_i, well_k=uwi_k,
                    horizontal_dist=np.nan, horizontal_dist_median=np.nan,
                    vertical_dist=float(vertical), dist3d=np.nan,
                    overlap_len_common_ft=0.0, LL_i=float(LL_cache[uwi_i]), LL_k=float(LL_cache[uwi_k]),
                    tvd_i=float(tvd_i), tvd_k=float(tvd_k),
                    overlap_pct_i=np.nan, overlap_pct_k=np.nan,
                    n_samples=0, dy_p5=np.nan, angle_deg=float(angle_deg),
                    pair_alignment="parallel_like",
                    min_distance_ft=np.nan, mean_windowed_ft=np.nan,
                    reject_reason="no_overlap_x",
                    direction_axis=dir_axis,
                    direction_to_k_from_i_axis=None,
                    direction_axis_confidence=np.nan,
                    direction_axis_distribution="",
                    drill_direction_i=dir_i, drill_direction_k=dir_k,
                    axis_forced=True
                )
                if want_artifacts:
                    artifacts.overlap_band = None
                    artifacts.theta_rose = np.array([], dtype=float)
                return result, artifacts

            # clip to [x_lo, x_hi]
            band = (x_lo, x_hi)
            Xi_seg = self._clip_polyline_by_x_band(Xi_if, band)
            Xk_seg = self._clip_polyline_by_x_band(Xk_if, band)

            # spacing over overlap
            cross_mean, n_used, Lmin = self._crossline_spacing_from_overlap(
                Xi_seg, Xk_seg, step_ft=step_ft, n_samples=n_samples
            )
            cross_median, _, _ = self._crossline_spacing_median_from_overlap(
                Xi_seg, Xk_seg, step_ft=step_ft, n_samples=n_samples
            )

            # dy percentile guardrail (optional)
            # compute dy at same grid to obtain dy_p
            if n_used > 0:
                # rebuild x-grid to get dy series
                xi_min2, xi_max2 = float(Xi_seg[:, 0].min()), float(Xi_seg[:, 0].max())
                xk_min2, xk_max2 = float(Xk_seg[:, 0].min()), float(Xk_seg[:, 0].max())
                x_lo2, x_hi2 = max(xi_min2, xk_min2), min(xi_max2, xk_max2)
                step = max(int(step_ft or 100), 1)
                N = max(int(np.floor((x_hi2 - x_lo2) / step)) + 1, 2)
                xgrid = np.linspace(x_lo2, x_hi2, N)
                yi = self._interp_y_of_x(Xi_seg, xgrid)
                yk = self._interp_y_of_x(Xk_seg, xgrid)
                dy = np.abs(yk - yi)
                dy_p = float(np.percentile(dy, crossline_percentile))
            else:
                dy_p = np.nan
                xgrid = np.array([], dtype=float)
                yi = np.array([], dtype=float)
                yk = np.array([], dtype=float)

            # direction over overlap (axis-constrained)
            dir_mode_axis = None; dir_conf_axis = np.nan; dir_dist_axis = ""
            if has_ll and (n_used > 0):
                lat_i_full = df_i["latitude"].to_numpy(float)
                lon_i_full = df_i["longitude"].to_numpy(float)
                lat_k_full = df_k["latitude"].to_numpy(float)
                lon_k_full = df_k["longitude"].to_numpy(float)

                Xi_seg_ll, lat_i_seg, lon_i_seg = self._clip_polyline_with_latlon_by_x_band(Xi_if, lat_i_full, lon_i_full, band)
                Xk_seg_ll, lat_k_seg, lon_k_seg = self._clip_polyline_with_latlon_by_x_band(Xk_if, lat_k_full, lon_k_full, band)

                dir_mode_axis, dir_conf_axis, dir_dist_axis = self._axis_constrained_direction_over_overlap(
                    Xi_seg, Xk_seg,
                    lat_i_seg, lon_i_seg, lat_k_seg, lon_k_seg,
                    want_axis=dir_axis, step_ft=step_ft, n_samples=n_samples,
                    deadband=0.15, tie_tol=0.05
                )

                # for rose plot (parallel-like): compute az per xgrid
                lat_i_x = np.interp(xgrid, Xi_seg_ll[:, 0], lat_i_seg) if xgrid.size else np.array([])
                lon_i_x = np.interp(xgrid, Xi_seg_ll[:, 0], lon_i_seg) if xgrid.size else np.array([])
                lat_k_x = np.interp(xgrid, Xk_seg_ll[:, 0], lat_k_seg) if xgrid.size else np.array([])
                lon_k_x = np.interp(xgrid, Xk_seg_ll[:, 0], lon_k_seg) if xgrid.size else np.array([])
                if xgrid.size:
                    az12, _, _ = geod.inv(lon_i_x, lat_i_x, lon_k_x, lat_k_x)
                    theta_rose = np.deg2rad((90.0 - az12) % 360.0)
                else:
                    theta_rose = np.array([], dtype=float)
            else:
                theta_rose = np.array([], dtype=float)

            horiz_mean = float(cross_mean)
            horiz_med = float(cross_median)
            dist3d = float(np.hypot(horiz_mean, vertical))

            # overlap % against full LLs
            Li = float(self._arclength(Xi_seg)[-1]) if Xi_seg.size else 0.0
            Lk = float(self._arclength(Xk_seg)[-1]) if Xk_seg.size else 0.0
            overlap_len_common = float(min(Li, Lk))
            LL_i = float(LL_cache[uwi_i]); LL_k = float(LL_cache[uwi_k])
            overlap_pct_i = (overlap_len_common / LL_i) if LL_i > 0 else np.nan
            overlap_pct_k = (overlap_len_common / LL_k) if LL_k > 0 else np.nan

            result = SpacingResult(
                well_i=uwi_i, well_k=uwi_k,
                horizontal_dist=horiz_mean, horizontal_dist_median=horiz_med,
                vertical_dist=float(vertical), dist3d=dist3d,
                overlap_len_common_ft=overlap_len_common,
                LL_i=LL_i, LL_k=LL_k,
                tvd_i=float(tvd_i), tvd_k=float(tvd_k),
                overlap_pct_i=overlap_pct_i, overlap_pct_k=overlap_pct_k,
                n_samples=int(n_used), dy_p5=float(dy_p), angle_deg=float(angle_deg),
                pair_alignment="parallel_like",
                min_distance_ft=np.nan, mean_windowed_ft=np.nan,
                reject_reason="",
                direction_axis=dir_axis,
                direction_to_k_from_i_axis=dir_mode_axis,
                direction_axis_confidence=float(dir_conf_axis) if np.isfinite(dir_conf_axis) else np.nan,
                direction_axis_distribution=dir_dist_axis or "",
                drill_direction_i=dir_i, drill_direction_k=dir_k,
                axis_forced=True
            )

            if want_artifacts:
                # Rebuild UTM pairs at xgrid for arrows
                if xgrid.size:
                    Xi_utml = origin_i[None, :] + xgrid[:, None]*ex_i[None, :] + yi[:, None]*ey_i[None, :]
                    Xk_utml = origin_i[None, :] + xgrid[:, None]*ex_i[None, :] + yk[:, None]*ey_i[None, :]
                else:
                    Xi_utml = np.empty((0, 2)); Xk_utml = np.empty((0, 2))

                artifacts.overlap_band = (float(x_lo), float(x_hi))
                artifacts.Xi_clip = Xi_seg
                artifacts.Xk_clip = Xk_seg
                artifacts.xgrid = xgrid
                artifacts.yi = yi
                artifacts.yk = yk
                artifacts.Xi_utml = Xi_utml
                artifacts.Xk_utml = Xk_utml
                artifacts.theta_rose = theta_rose

            return result, artifacts

        # Oblique/Perpendicular: nearest projection series
        # sample along i
        si = self._arclength(Xi_utm); Li = float(si[-1]) if si.size else 0.0
        step = max(int(step_ft or 100), 1)
        n = max(int(np.floor(Li / step)) + 1, 2)
        s_targets = np.linspace(0.0, max(Li, 1e-6), n)
        Pi = self._interp_by_arclength(Xi_utm, s_targets)
        d, j_arr, t_arr = self._nearest_distances_to_polyline(Pi, Xk_utm)

        # --- NEW: contact length + projection coverage (vectorized) ---
        T = float(contact_threshold_ft) if contact_threshold_ft is not None else float("nan")
        eps = max(float(coverage_epsilon), 0.0)

        if d.size:
            # thresholded "contact" along i
            mask_contact = d <= T if np.isfinite(T) else np.zeros_like(d, dtype=bool)
            contact_pct = float(mask_contact.mean()) if np.isfinite(T) else np.nan
            contact_len = contact_pct * Li if np.isfinite(T) else np.nan

            # interior-only mask based on orthogonal projection parameter t∈(ε,1-ε)
            mask_interior = (t_arr > eps) & (t_arr < (1.0 - eps))
            proj_cov_pct = float(mask_interior.mean())

            # combined: contact AND interior
            mask_contact_int = mask_contact & mask_interior if np.isfinite(T) else np.zeros_like(mask_interior, dtype=bool)
            contact_pct_int = float(mask_contact_int.mean()) if np.isfinite(T) else np.nan
            contact_len_int = contact_pct_int * Li if np.isfinite(T) else np.nan
        else:
            contact_pct = contact_len = proj_cov_pct = contact_pct_int = contact_len_int = np.nan


        # direction over nearest points if lat/lon available
        if has_ll and n > 0:
            lat_i = df_i["latitude"].to_numpy(float)
            lon_i = df_i["longitude"].to_numpy(float)
            lat_k = df_k["latitude"].to_numpy(float)
            lon_k = df_k["longitude"].to_numpy(float)
            lat_i_s = self._interp_attr_by_arclength(Xi_utm, lat_i, s_targets)
            lon_i_s = self._interp_attr_by_arclength(Xi_utm, lon_i, s_targets)
            # Guard: if well k has only 1 survey point, we cannot interpolate along segments
            if lat_k.size > 1:
                lat_k_s = lat_k[j_arr] + t_arr*(lat_k[j_arr+1] - lat_k[j_arr])
                lon_k_s = lon_k[j_arr] + t_arr*(lon_k[j_arr+1] - lon_k[j_arr])
            else:
                # Single point: use that point for all samples
                lat_k_s = np.full_like(lat_i_s, lat_k[0])
                lon_k_s = np.full_like(lon_i_s, lon_k[0])
            dir_mode_axis, dir_conf_axis, dir_dist_axis = self._axis_constrained_direction_from_pairs(
                lat_i_s, lon_i_s, lat_k_s, lon_k_s, want_axis=dir_axis,
                deadband=0.15, tie_tol=0.05
            )
            az12, _, _ = geod.inv(lon_i_s, lat_i_s, lon_k_s, lat_k_s)
            theta_rose = np.deg2rad((90.0 - az12) % 360.0)
        else:
            theta_rose = np.array([], dtype=float)

        mean_d = float(d.mean())
        median_d = float(np.median(d))
        min_d = float(d.min()) if d.size else np.nan

        mean_windowed = np.nan
        if use_windowed_mean and d.size and np.isfinite(d).any():
            idx_min = int(np.nanargmin(d))
            s0 = s_targets[idx_min]
            mask = (s_targets >= s0 - window_ft) & (s_targets <= s0 + window_ft)
            if mask.any():
                mean_windowed = float(np.nanmean(d[mask]))

        dist3d = float(np.hypot(mean_d, vertical))

        result = SpacingResult(
            well_i=uwi_i, well_k=uwi_k,
            horizontal_dist=mean_d, horizontal_dist_median=median_d,
            vertical_dist=float(vertical), dist3d=dist3d,
            overlap_len_common_ft=float("nan"),
            LL_i=float(LL_cache[uwi_i]), LL_k=float(LL_cache[uwi_k]),
            tvd_i=float(tvd_i), tvd_k=float(tvd_k),
            overlap_pct_i=float("nan"), overlap_pct_k=float("nan"),
            n_samples=int(n), dy_p5=np.nan, angle_deg=float(angle_deg),
            pair_alignment=("oblique" if align_type is AlignmentType.OBLIQUE else "perpendicular"),
            min_distance_ft=min_d, mean_windowed_ft=mean_windowed,
            reject_reason="",
            direction_axis=dir_axis,
            direction_to_k_from_i_axis=dir_mode_axis,
            direction_axis_confidence=float(dir_conf_axis) if np.isfinite(dir_conf_axis) else np.nan,
            direction_axis_distribution=dir_dist_axis or "",
            drill_direction_i=dir_i, drill_direction_k=dir_k,
            axis_forced=True,
            contact_threshold_ft=T,
            contact_len_i_ft=contact_len,
            contact_pct_i=contact_pct,
            contact_len_i_interior_ft=contact_len_int,
            contact_pct_i_interior=contact_pct_int,
            proj_coverage_i_pct=proj_cov_pct,
        )

        if want_artifacts:
            # nearest targets Q on k (robust to degenerate k polylines)
            if Xk_utm.shape[0] >= 2:
                A = Xk_utm[:-1]; B = Xk_utm[1:]; AB = B - A
                Q = A[j_arr] + t_arr[:, None]*(AB[j_arr])
            elif Xk_utm.shape[0] == 1:
                Q = np.repeat(Xk_utm[0][None, :], len(Pi), axis=0)
            else:
                Q = np.empty((0, 2))

            artifacts.s_targets = s_targets
            artifacts.Pi = Pi
            artifacts.Q = Q
            artifacts.d_series = d
            artifacts.theta_rose = theta_rose

        return result, artifacts

    def debug_pair_spacing(
        self,
        uwi_i: Any,
        uwi_k: Any,
        *,
        step_ft: int = 100,
        use_pca_axis: bool = True,
        theta_parallel_deg: float = 25.0,
        theta_perp_deg: float = 65.0,
        arrow_stride: int = 5,
        save_dir: Optional[os.PathLike] = None,
        figure_prefix: str = "spacing",
        show: bool = True,
        contact_threshold_ft: float = 300,
        coverage_epsilon: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic visualizations and metrics for a single well pair.

        This method provides detailed visual debugging for understanding spacing calculations
        between two specific wells. It uses the same core computation engine as the batch
        processor (_calculate_spacing_statistics), ensuring consistency, but generates
        extensive diagnostic plots and returns intermediate artifacts.

        Particularly useful for:
        - Validating spacing calculations for specific well pairs
        - Understanding why certain pairs were rejected
        - Investigating unexpected spacing values
        - Creating publication-quality figures for reports
        - Training and education on spacing methodology

        The method automatically detects pair alignment (parallel/oblique/perpendicular)
        and generates alignment-specific visualizations.

        Parameters
        ----------
        uwi_i : Any (typically str)
            Unique well identifier for reference well (well_i).
            Must exist in self.trajectories keys.

        uwi_k : Any (typically str)
            Unique well identifier for comparison well (well_k).
            Must exist in self.trajectories keys.

        step_ft : int, default 100
            Sampling interval (ft) along laterals for spacing computation.
            Smaller values = higher resolution but slower. Typical: 50-200 ft.

        use_pca_axis : bool, default True
            If True, use PCA to define local i-frame axis (handles curved laterals).
            If False, use simple heel-to-toe vector (faster, less accurate for curves).

        theta_parallel_deg : float, default 25.0
            Maximum angle (degrees) for PARALLEL_LIKE classification.
            Pairs with angle ≤ this value use crossline spacing algorithm.

        theta_perp_deg : float, default 65.0
            Minimum angle (degrees) for PERPENDICULAR classification.
            Pairs with angle ≥ this value treated as perpendicular (not oblique).

        arrow_stride : int, default 5
            Stride for decimating arrows/segments in quiver/projection plots.
            Higher = fewer arrows (cleaner plots). Use 1 for full resolution.

        save_dir : str, Path, or None, default None
            If provided, saves all generated figures to this directory as PNG files.
            Directory is created if it doesn't exist. Files named with figure_prefix.
            If None, figures are displayed interactively (show=True) or closed (show=False).

        figure_prefix : str, default "spacing"
            Prefix for saved figure filenames. Figures saved as:
            "{figure_prefix}_map_view.png", "{figure_prefix}_projected_iframe.png", etc.

        show : bool, default True
            If True and save_dir is None, display figures interactively using plt.show().
            If False or save_dir is provided, figures are closed after saving.

        contact_threshold_ft : float, default 300
            Distance threshold T (ft) for contact metrics in oblique/perpendicular pairs.
            Visualized as horizontal line on nearest-distance plots.

        coverage_epsilon : float, default 1e-3
            Epsilon for interior projection classification. Points with parameter
            t ∈ (ε, 1-ε) are considered interior (excludes endpoint projections).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all computed metrics for the well pair:
            - 'well_i', 'well_k' : Well identifiers
            - 'horizontal_dist', 'horizontal_dist_median' : Mean/median spacing (ft)
            - 'vertical_dist', '3D_dist' : Vertical and 3D distances (ft)
            - 'tvd_i', 'tvd_k' : True vertical depths at midpoint (ft)
            - 'LL_i', 'LL_k' : Lateral lengths (ft)
            - 'pair_alignment' : 'parallel_like', 'oblique', or 'perpendicular'
            - 'angle_deg' : Angle between lateral axes (0-90°)
            - 'overlap_len_common_ft', 'overlap_pct_i', 'overlap_pct_k' : Parallel metrics
            - 'min_distance_ft', 'mean_windowed_ft' : Oblique/perp metrics
            - 'contact_len_i_ft', 'contact_pct_i' : Contact metrics
            - 'direction_to_k_from_i_axis' : Cardinal direction ('N', 'S', 'E', 'W')
            - 'direction_axis_confidence' : Direction confidence (0-1)
            - 'reject_reason' : Empty if valid, reason string if rejected
            - 'n_samples', 'dy_p5' : Quality metrics
            - Plus 20+ additional fields (see SpacingResult dataclass)

        Raises
        ------
        KeyError
            If uwi_i or uwi_k not found in self.trajectories
        ValueError
            If trajectory data is invalid or insufficient for spacing calculation

        Generated Figures
        -----------------
        **Common to All Pairs:**
        1. **map_view.png**: Original UTM map view showing both well trajectories
        2. **projected_iframe.png**: Wells projected into reference well's local i-frame

        **Parallel-Like Pairs Only:**
        3. **overlap_band.png**: X-overlap region highlighted in i-frame
        4. **clipped_segments.png**: Clipped polylines used for crossline calculation
        5. **crossline_dy_series.png**: |Δy(x)| distance series along overlap
        6. **pairs_utm_crossline_dir.png**: UTM map with i→k direction arrows
        7. **direction_rose.png**: Polar histogram of i→k directions

        **Oblique/Perpendicular Pairs Only:**
        8. **nearest_projection_series.png**: Distance vs. arclength along well_i,
           with contact threshold T visualized
        9. **utm_nearest_segments.png**: UTM map showing nearest-point segments
           connecting samples on well_i to closest points on well_k

        Notes
        -----
        - All figures are 8×5 or 8×6 inches at 150 DPI
        - Distances in feet (or trajectory coordinate units)
        - Uses same algorithms as _calculate_spacing_statistics for consistency
        - Color scheme: well_i typically blue, well_k typically orange
        - Axes use equal aspect ratio for map/projection views
        - Progress bars and parallel processing disabled (single pair)

        Performance
        -----------
        - Typical runtime: 0.5-2 seconds per pair (includes figure generation)
        - Memory: ~50-100 MB per pair with artifacts
        - Figure file sizes: ~100-300 KB each (PNG, 150 DPI)

        Examples
        --------
        **Example 1: Basic Interactive Debug**

        >>> wsc = WellSpacingCalculator(trajectories_df)
        >>> metrics = wsc.debug_pair_spacing(
        ...     uwi_i="42-123-12345-00-00",
        ...     uwi_k="42-123-12346-00-00"
        ... )
        >>> # Figures displayed interactively
        >>> print(f"Horizontal spacing: {metrics['horizontal_dist']:.1f} ft")
        >>> print(f"Alignment: {metrics['pair_alignment']}")

        **Example 2: Save Figures for Report**

        >>> metrics = wsc.debug_pair_spacing(
        ...     uwi_i="WELL_001",
        ...     uwi_k="WELL_002",
        ...     save_dir="./debug_plots/well001_vs_well002",
        ...     figure_prefix="pair_analysis",
        ...     show=False  # Don't display, just save
        ... )
        >>> # Figures saved to ./debug_plots/well001_vs_well002/pair_analysis_*.png

        **Example 3: High-Resolution Analysis**

        >>> metrics = wsc.debug_pair_spacing(
        ...     uwi_i="WELL_A",
        ...     uwi_k="WELL_B",
        ...     step_ft=50,              # 50 ft sampling (vs default 100)
        ...     arrow_stride=3,          # More arrows in plots
        ...     contact_threshold_ft=250, # 250 ft contact threshold
        ...     save_dir="./high_res_analysis"
        ... )

        **Example 4: Batch Debug Multiple Pairs**

        >>> # Debug all pairs for a specific reference well
        >>> reference_well = "42-123-12345-00-00"
        >>> spacing_df = wsc._calculate_spacing_statistics()
        >>>
        >>> # Get top 5 nearest neighbors
        >>> neighbors = spacing_df[
        ...     (spacing_df['well_i'] == reference_well) &
        ...     (spacing_df['reject_reason'] == '')
        ... ].nsmallest(5, 'horizontal_dist')
        >>>
        >>> # Debug each neighbor
        >>> for _, row in neighbors.iterrows():
        ...     uwi_k = row['well_k']
        ...     metrics = wsc.debug_pair_spacing(
        ...         uwi_i=reference_well,
        ...         uwi_k=uwi_k,
        ...         save_dir=f"./debug/{reference_well}_vs_{uwi_k}",
        ...         show=False
        ...     )
        ...     print(f"{uwi_k}: {metrics['horizontal_dist']:.1f} ft")

        **Example 5: Investigate Rejected Pair**

        >>> # Find why a pair was rejected
        >>> rejected = spacing_df[spacing_df['reject_reason'] == 'coarse_far'].iloc[0]
        >>> metrics = wsc.debug_pair_spacing(
        ...     uwi_i=rejected['well_i'],
        ...     uwi_k=rejected['well_k'],
        ...     save_dir="./rejected_investigation"
        ... )
        >>> print(f"Reject reason: {metrics['reject_reason']}")
        >>> # Examine figures to understand why spacing exceeded threshold

        **Example 6: Compare PCA vs Heel-Toe Axis**

        >>> # With PCA (default, handles curves)
        >>> metrics_pca = wsc.debug_pair_spacing(
        ...     uwi_i="WELL_X", uwi_k="WELL_Y",
        ...     use_pca_axis=True,
        ...     save_dir="./compare/pca",
        ...     show=False
        ... )
        >>>
        >>> # With heel-to-toe axis (simpler)
        >>> metrics_heeltoe = wsc.debug_pair_spacing(
        ...     uwi_i="WELL_X", uwi_k="WELL_Y",
        ...     use_pca_axis=False,
        ...     save_dir="./compare/heeltoe",
        ...     show=False
        ... )
        >>>
        >>> print(f"PCA spacing: {metrics_pca['horizontal_dist']:.1f} ft")
        >>> print(f"Heel-toe spacing: {metrics_heeltoe['horizontal_dist']:.1f} ft")

        **Example 7: Extract Specific Metrics**

        >>> metrics = wsc.debug_pair_spacing("WELL_1", "WELL_2")
        >>>
        >>> # Build summary report
        >>> if metrics['pair_alignment'] == 'parallel_like':
        ...     print(f"Parallel Wells: {metrics['angle_deg']:.1f}° angle")
        ...     print(f"Overlap: {metrics['overlap_len_common_ft']:.0f} ft")
        ...     print(f"Spacing: {metrics['horizontal_dist']:.1f} ± "
        ...           f"{metrics['horizontal_dist'] - metrics['horizontal_dist_median']:.1f} ft")
        ... else:
        ...     print(f"Non-Parallel Wells: {metrics['angle_deg']:.1f}° angle")
        ...     print(f"Closest approach: {metrics['min_distance_ft']:.1f} ft")
        ...     print(f"Contact (T={metrics['contact_threshold_ft']:.0f}): "
        ...           f"{metrics['contact_pct_i']*100:.1f}% of well_i")

        See Also
        --------
        _calculate_spacing_statistics : Batch computation for all pairs
        _compute_pair_metrics_and_artifacts : Core computation engine (called internally)
        SpacingResult : Dataclass defining all returned metrics
        PairArtifacts : Dataclass containing visualization artifacts

        Tips
        ----
        - Use save_dir for creating documentation/presentations
        - Set show=False when generating many figures programmatically
        - Increase arrow_stride for cleaner plots with dense trajectories
        - Compare use_pca_axis=True vs False for curved wells
        - Check reject_reason to understand why pairs fail QC filters
        """

        self.logger.info("=" * 80)
        self.logger.info(f"DEBUG PAIR SPACING: {uwi_i} → {uwi_k}")
        self.logger.info("=" * 80)
        debug_start = time.time()

        self.logger.debug(f"Configuration:")
        self.logger.debug(f"  Sampling step: {step_ft} ft")
        self.logger.debug(f"  Axis mode: {'PCA' if use_pca_axis else 'heel-to-toe'}")
        self.logger.debug(f"  Alignment thresholds: parallel ≤{theta_parallel_deg}°, perp ≥{theta_perp_deg}°")
        self.logger.debug(f"  Contact threshold: {contact_threshold_ft} ft")
        self.logger.debug(f"  Arrow stride: {arrow_stride}")
        if save_dir is not None:
            self.logger.debug(f"  Output: {save_dir}/{figure_prefix}_*.png")
        else:
            self.logger.debug(f"  Output: interactive display (show={show})")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.logger.info("─" * 80)
        self.logger.info("Computing pair metrics and artifacts...")
        compute_start = time.time()

        # compute once, request artifacts
        res, art = self._compute_pair_metrics_and_artifacts(
            uwi_i=uwi_i, uwi_k=uwi_k,
            step_ft=step_ft, n_samples=None,
            max_crossline_ft=None, crossline_percentile=5.0, ds_crossline_step_ft=200,
            use_pca_axis=use_pca_axis,
            theta_parallel_deg=theta_parallel_deg, theta_perp_deg=theta_perp_deg,
            reject_misaligned=False, use_windowed_mean=True, window_ft=300.0,
            want_artifacts=True,
            contact_threshold_ft=contact_threshold_ft,
            coverage_epsilon=coverage_epsilon,
        )

        compute_elapsed = time.time() - compute_start
        self.logger.info(f"✓ Metrics computed in {compute_elapsed:.2f}s")
        self.logger.debug(f"  Alignment: {res.pair_alignment}, angle={res.angle_deg:.1f}°")
        if res.reject_reason:
            self.logger.info(f"  ⚠ Pair rejected: {res.reject_reason}")
        else:
            if res.pair_alignment == "parallel_like":
                self.logger.debug(f"  Horizontal spacing: {res.horizontal_dist:.1f} ft (median={res.horizontal_dist_median:.1f} ft)")
                self.logger.debug(f"  Overlap: {res.overlap_len_common_ft:.0f} ft ({res.overlap_pct_i*100:.1f}% of well_i)")
            else:
                self.logger.debug(f"  Min distance: {res.min_distance_ft:.1f} ft")
                self.logger.debug(f"  Contact length: {res.contact_len_i_ft:.0f} ft ({res.contact_pct_i*100:.1f}% of well_i)")

        self.logger.info("─" * 80)
        self.logger.info("Generating diagnostic visualizations...")
        viz_start = time.time()

        paths: Dict[str, str] = {}
        def _save(fig, key):
            if save_dir is not None:
                p = os.path.join(save_dir, f"{figure_prefix}_{key}.png")
                fig.savefig(p, dpi=150, bbox_inches="tight")
                paths[key] = p
                plt.close(fig)
            else:
                plt.show()   # show immediately in notebook
                plt.close(fig)

        # Panel 1: UTM map
        if art and art.Xi_utm is not None and art.Xk_utm is not None:
            fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
            ax.plot(art.Xi_utm[:, 0], art.Xi_utm[:, 1], label=f"{uwi_i}")
            ax.plot(art.Xk_utm[:, 0], art.Xk_utm[:, 1], label=f"{uwi_k}")
            ax.axis("equal"); ax.legend()
            ax.set_title("Original UTM map view"); ax.set_xlabel("Easting (ft)"); ax.set_ylabel("Northing (ft)")
            _save(fig, "map_view")

        # Panel 2: projected i-frame
        if art and art.Xi_if is not None and art.Xk_if is not None:
            fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
            ax.plot(art.Xi_if[:, 0], art.Xi_if[:, 1], label="i in i-frame")
            ax.plot(art.Xk_if[:, 0], art.Xk_if[:, 1], label="k in i-frame")
            ax.axis("equal"); ax.legend()
            ax.set_title("Projected into i-frame"); ax.set_xlabel("x (along i)"); ax.set_ylabel("y (crossline)")
            _save(fig, "projected_iframe")

        # Parallel-like specifics
        if res.pair_alignment == "parallel_like":
            # Overlap band
            if art.overlap_band is not None:
                x_lo, x_hi = art.overlap_band
                fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
                ax.plot(art.Xi_if[:, 0], art.Xi_if[:, 1], label="i (full)")
                ax.plot(art.Xk_if[:, 0], art.Xk_if[:, 1], label="k (full)")
                ax.axvline(x_lo, ls="--"); ax.axvline(x_hi, ls="--")
                ax.axis("equal"); ax.legend()
                ax.set_title("Overlap band (x in i-frame)"); ax.set_xlabel("x (along i)"); ax.set_ylabel("y (crossline)")
                _save(fig, "overlap_band")

            # Clipped segments
            if art.Xi_clip is not None and art.Xi_clip.size and art.Xk_clip is not None and art.Xk_clip.size:
                fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
                ax.plot(art.Xi_clip[:, 0], art.Xi_clip[:, 1], label="i (clipped)")
                ax.plot(art.Xk_clip[:, 0], art.Xk_clip[:, 1], label="k (clipped)")
                ax.axis("equal"); ax.legend()
                ax.set_title("Clipped segments used for spacing")
                ax.set_xlabel("x (along i)"); ax.set_ylabel("y (crossline)")
                _save(fig, "clipped_segments")

            # Crossline dy series + quiver in UTM
            if art.xgrid is not None and art.xgrid.size and art.yi is not None and art.yk is not None:
                dy = np.abs(art.yk - art.yi)
                fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
                ax.plot(art.xgrid - art.overlap_band[0], dy)
                ax.set_title(f"|Δy(x)|; mean={res.horizontal_dist:.1f} ft, median={res.horizontal_dist_median:.1f} ft")
                ax.set_xlabel("Along-overlap x (ft from band start)"); ax.set_ylabel("|Δy| (ft)")
                _save(fig, "crossline_dy_series")

                # UTM arrows (i → k)
                if art.Xi_utml is not None and art.Xi_utml.size and art.Xk_utml is not None and art.Xk_utml.size:
                    U = art.Xk_utml[:, 0] - art.Xi_utml[:, 0]
                    V = art.Xk_utml[:, 1] - art.Xi_utml[:, 1]
                    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
                    ax.plot(art.Xi_utm[:, 0], art.Xi_utm[:, 1], label=f"{uwi_i}")
                    ax.plot(art.Xk_utm[:, 0], art.Xk_utm[:, 1], label=f"{uwi_k}")
                    ax.quiver(art.Xi_utml[::arrow_stride, 0], art.Xi_utml[::arrow_stride, 1],
                            U[::arrow_stride], V[::arrow_stride],
                            angles='xy', scale_units='xy', scale=1, width=0.002)
                    ax.axis("equal"); ax.legend()
                    ax.set_title("Crossline sampling pairs in UTM (i → k)")
                    ax.set_xlabel("Easting (ft)"); ax.set_ylabel("Northing (ft)")
                    _save(fig, "pairs_utm_crossline_dir")

        # Oblique/Perp specifics
        if res.pair_alignment in ("oblique", "perpendicular"):
            if art.s_targets is not None and art.d_series is not None:
                fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
                ax.plot(art.s_targets, art.d_series)
                # --- NEW: visualize T ---
                if np.isfinite(res.contact_threshold_ft):
                    ax.axhline(res.contact_threshold_ft, linestyle="--")
                ax.set_title(f"Nearest distance along i; mean={res.horizontal_dist:.1f} ft, "
                            f"median={res.horizontal_dist_median:.1f} ft")
                ax.set_xlabel("Distance along well i (ft)"); ax.set_ylabel("Nearest dist to k (ft)")
                _save(fig, "nearest_projection_series")

            # draw decimated nearest segments
            if art.Pi is not None and art.Q is not None:
                fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
                ax.plot(art.Xi_utm[:, 0], art.Xi_utm[:, 1], label=f"{uwi_i}")
                ax.plot(art.Xk_utm[:, 0], art.Xk_utm[:, 1], label=f"{uwi_k}")
                stride = max(int(arrow_stride), 1)
                for idx in range(0, len(art.Pi), stride):
                    ax.plot([art.Pi[idx, 0], art.Q[idx, 0]], [art.Pi[idx, 1], art.Q[idx, 1]])
                ax.axis("equal"); ax.legend()
                ax.set_title("Nearest-projection segments in UTM (i samples → nearest on k)")
                ax.set_xlabel("Easting (ft)"); ax.set_ylabel("Northing (ft)")
                _save(fig, "utm_nearest_segments")

        # Direction rose (if any)
        if art.theta_rose is not None and art.theta_rose.size:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='polar')
            bins = np.linspace(0.0, 2*np.pi, 5)
            counts_b, _ = np.histogram(art.theta_rose % (2*np.pi), bins=bins)
            widths = np.diff(bins); centers_b = bins[:-1] + widths/2
            ax.bar(centers_b, counts_b, width=widths, align='center')
            ax.set_title(f"Direction i→k on {art.dir_axis} (modal={res.direction_to_k_from_i_axis}, conf={res.direction_axis_confidence:.2f})")
            _save(fig, "direction_rose")

        if show and paths:
            # quick gallery grid in a simple order
            keys = list(paths.keys())
            n = len(keys); ncols = 2; nrows = (n + ncols - 1)//ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5*nrows))
            axes = np.atleast_2d(axes)
            for idx, key in enumerate(keys):
                r, c = divmod(idx, ncols)
                img = plt.imread(paths[key])
                axes[r, c].imshow(img); axes[r, c].set_title(key.replace("_", " "))
                axes[r, c].axis("off")
            # hide extra
            for j in range(len(keys), nrows*ncols):
                r, c = divmod(j, ncols); axes[r, c].axis("off")
            plt.show()

        viz_elapsed = time.time() - viz_start
        n_figures = len(paths)
        self.logger.info(f"✓ Generated {n_figures} diagnostic figure(s) in {viz_elapsed:.2f}s")
        if save_dir is not None:
            self.logger.debug(f"  Figures saved to: {save_dir}")
            for key in sorted(paths.keys()):
                self.logger.debug(f"    - {figure_prefix}_{key}.png")

        debug_elapsed = time.time() - debug_start
        self.logger.info("─" * 80)
        self.logger.info(f"✓ Debug complete in {debug_elapsed:.2f}s (compute: {compute_elapsed:.2f}s, viz: {viz_elapsed:.2f}s)")
        self.logger.info("=" * 80)

        # Return a compact metrics dict (same fields as before to avoid breaking downstream)
        return {
            "pair_alignment": res.pair_alignment,
            "angle_deg": res.angle_deg,
            "horizontal_dist": res.horizontal_dist,
            "horizontal_dist_median": res.horizontal_dist_median,
            "vertical_dist": res.vertical_dist,
            "3D_dist": res.dist3d,
            "overlap_len_common_ft": res.overlap_len_common_ft,
            "LL_i": res.LL_i, "LL_k": res.LL_k,
            "overlap_pct_i": res.overlap_pct_i, "overlap_pct_k": res.overlap_pct_k,
            "n_samples": res.n_samples,
            "dy_p5": res.dy_p5,
            "min_distance_ft": res.min_distance_ft,
            "mean_windowed_ft": res.mean_windowed_ft,
            "direction_axis": res.direction_axis,
            "direction_to_k_from_i_axis": res.direction_to_k_from_i_axis,
            "direction_axis_confidence": res.direction_axis_confidence,
            "direction_axis_distribution": res.direction_axis_distribution,
            "paths": paths,
            "proj_coverage_i_pct": res.proj_coverage_i_pct,
            "contact_threshold_ft": res.contact_threshold_ft,
            "contact_len_i_ft": res.contact_len_i_ft,
            "contact_pct_i": res.contact_pct_i,
            "contact_len_i_interior_ft": res.contact_len_i_interior_ft,
            "contact_pct_i_interior": res.contact_pct_i_interior,
        }

    def _process_batch(
        self,
        i_idx: np.ndarray,
        k_idx: np.ndarray,
        ids: np.ndarray,
        coords: np.ndarray,        # midpoint coords [["x","y","tvd"]]
        directions: np.ndarray,    # drill dirs per well (EW/NS)
        *,
        step_ft: int,
        n_samples: Optional[int]=None,
        max_crossline_ft: Optional[float],
        crossline_percentile: float,
        ds_crossline_step_ft: int,
        emit_rejected: bool,
        use_pca_axis: bool,
        # angle routing knobs
        theta_parallel_deg: float,
        theta_perp_deg: float,
        reject_misaligned: bool,
        use_windowed_mean: bool,
        window_ft: float,
        contact_threshold_ft: float,
        coverage_epsilon: float,
    ) -> pd.DataFrame:
        """
        Process a batch of well pairs with angle-aware routing to appropriate spacing methods.

        This method is the main batch processor that handles all candidate well pairs,
        routing them to the appropriate spacing computation based on their alignment:

        Routing Logic
        -------------
        - PARALLEL_LIKE (angle ≤ theta_parallel_deg): Uses crossline |Δy(x)| computed over
          the x-overlap region in the reference well's i-frame.
        - OBLIQUE (theta_parallel_deg < angle < theta_perp_deg): Uses nearest-projection
          distance series from samples along well_i to the nearest point on well_k.
        - PERPENDICULAR (angle ≥ theta_perp_deg): Same as oblique, with additional
          min_distance_ft and mean_windowed_ft metrics.

        Parameters
        ----------
        i_idx, k_idx : np.ndarray
            Index arrays for (well_i, well_k) pairs to process.
        ids : np.ndarray
            Array of well identifiers (uwi), indexed by i_idx/k_idx.
        coords : np.ndarray
            Midpoint coordinates array with columns [x, y, tvd]. Shape: (N_wells, 3).
        directions : np.ndarray
            Drill direction array ('EW' or 'NS') for each well. Shape: (N_wells,).
        step_ft : int
            Sampling step size (ft).
        n_samples : int or None
            Override step_ft with fixed sample count.
        max_crossline_ft : float or None
            Maximum crossline threshold for filtering.
        crossline_percentile : float
            Percentile for crossline guardrail.
        ds_crossline_step_ft : int
            Downsampling step for coarse prefiltering.
        emit_rejected : bool
            If True, include rejected pairs in output with reject_reason populated.
        use_pca_axis : bool
            Use PCA for i-frame axis definition.
        theta_parallel_deg, theta_perp_deg : float
            Alignment classification thresholds (degrees).
        reject_misaligned : bool
            If True, reject oblique/perpendicular pairs entirely.
        use_windowed_mean : bool
            Compute windowed mean around closest approach for oblique/perp.
        window_ft : float
            Half-width of window (ft).
        contact_threshold_ft : float
            Distance threshold T for contact metrics (ft).
        coverage_epsilon : float
            Fraction for interior-only projection mask.

        Returns
        -------
        pd.DataFrame
            Pairwise spacing results with columns matching SpacingResult fields.

        Notes
        -----
        Direction axis is always constrained based on drill direction:
        NS-drilled wells → E/W direction axis; EW-drilled wells → N/S direction axis.
        """

        batch_start = time.time()
        n_pairs = len(i_idx)

        self.logger.debug(f"  Processing batch: {n_pairs:,} pairs")

        rows: List[Dict] = []
        rejection_counts = {}  # Track rejection reasons

        def _res_to_row(res) -> Dict[str, Any]:
            row = {
                "well_i": res.well_i, "well_k": res.well_k,
                "horizontal_dist": res.horizontal_dist,
                "horizontal_dist_median": res.horizontal_dist_median,
                "vertical_dist": res.vertical_dist,
                "3D_dist": res.dist3d,
                "drill_direction_i": res.drill_direction_i,
                "drill_direction_k": res.drill_direction_k,
                "overlap_len_common_ft": res.overlap_len_common_ft,
                "LL_i": res.LL_i, "LL_k": res.LL_k,
                "tvd_i": res.tvd_i, "tvd_k": res.tvd_k,
                "overlap_pct_i": res.overlap_pct_i, "overlap_pct_k": res.overlap_pct_k,
                "n_samples": res.n_samples,
                "dy_p5": res.dy_p5,
                "angle_deg": res.angle_deg,
                "pair_alignment": res.pair_alignment,
                "min_distance_ft": res.min_distance_ft,
                "mean_windowed_ft": res.mean_windowed_ft,
                "reject_reason": res.reject_reason,
                "direction_axis": res.direction_axis,
                "direction_to_k_from_i_axis": res.direction_to_k_from_i_axis,
                "direction_axis_confidence": res.direction_axis_confidence,
                "direction_axis_distribution": res.direction_axis_distribution,
                "axis_forced": res.axis_forced,
                "proj_coverage_i_pct": res.proj_coverage_i_pct,
                "contact_threshold_ft": res.contact_threshold_ft,
                "contact_len_i_ft": res.contact_len_i_ft,
                "contact_pct_i": res.contact_pct_i,
                "contact_len_i_interior_ft": res.contact_len_i_interior_ft,
                "contact_pct_i_interior": res.contact_pct_i_interior,
            }

            # Threshold-tagged duplicates (e.g., T300) for instant readability
            if np.isfinite(res.contact_threshold_ft):
                Ttag = f"T{int(round(res.contact_threshold_ft))}"
                row[f"contact_len_i_ft_{Ttag}"] = res.contact_len_i_ft
                row[f"contact_pct_i_{Ttag}"] = res.contact_pct_i
                row[f"contact_len_i_interior_ft_{Ttag}"] = res.contact_len_i_interior_ft
                row[f"contact_pct_i_interior_{Ttag}"] = res.contact_pct_i_interior

            # return combined
            base = {
                "well_i": res.well_i, "well_k": res.well_k,
                "horizontal_dist": res.horizontal_dist,
                "horizontal_dist_median": res.horizontal_dist_median,
                "vertical_dist": res.vertical_dist,
                "3D_dist": res.dist3d,
                "drill_direction_i": res.drill_direction_i,
                "drill_direction_k": res.drill_direction_k,
                "overlap_len_common_ft": res.overlap_len_common_ft,
                "LL_i": res.LL_i, "LL_k": res.LL_k,
                "tvd_i": res.tvd_i, "tvd_k": res.tvd_k,
                "overlap_pct_i": res.overlap_pct_i, "overlap_pct_k": res.overlap_pct_k,
                "n_samples": res.n_samples,
                "dy_p5": res.dy_p5,
                "angle_deg": res.angle_deg,
                "pair_alignment": res.pair_alignment,
                "min_distance_ft": res.min_distance_ft,
                "mean_windowed_ft": res.mean_windowed_ft,
                "reject_reason": res.reject_reason,
                "direction_axis": res.direction_axis,
                "direction_to_k_from_i_axis": res.direction_to_k_from_i_axis,
                "direction_axis_confidence": res.direction_axis_confidence,
                "direction_axis_distribution": res.direction_axis_distribution,
                "axis_forced": res.axis_forced,
            }
            base.update(row)
            return base


        if getattr(self, "_paircache", None) is None or self._paircache.get("use_pca_axis", None) != use_pca_axis:
            self._build_pair_cache(use_pca_axis, ds_crossline_step_ft)

        cache = self._paircache

        # --- process pairs grouped by i (vectorized precheck on parallel-like branch) ---
        for i in np.unique(i_idx):
            mask_i = (i_idx == i)
            k_list = k_idx[mask_i]
            if k_list.size == 0:
                continue

            uwi_i = ids[i]
            origin_i = cache["origin"][uwi_i]
            ex_i = cache["ex"][uwi_i]
            ey_i = cache["ey"][uwi_i]

            # Axis used for ALL rows where i is the reference
            dir_axis_i = "EW" if directions[i] == "NS" else "NS"

            # Coarse arrays for i + Ks
            Pi_coarse = cache["XY_coarse"][uwi_i]                                 # (M_ds, 2)
            Pk_coarse = np.stack([cache["XY_coarse"][ids[k]] for k in k_list])    # (K, M_ds, 2)

            # Precompute angle for each (i,k)
            ex_k_mat = np.stack([cache["ex"][ids[k]] for k in k_list])            # (K,2)
            dot_abs = np.abs(ex_k_mat @ ex_i)                                     # (K,)
            angles = np.degrees(np.arccos(np.clip(dot_abs, -1.0, 1.0)))           # (K,)

            # Alignment classes
            is_parallel = angles <= theta_parallel_deg
            is_perp = angles >= theta_perp_deg
            is_oblique = (~is_parallel) & (~is_perp)

            # Optionally reject misaligned pairs outright
            if reject_misaligned and (is_oblique.any() or is_perp.any()):
                if emit_rejected:
                    for idx_k, k in enumerate(k_list):
                        if is_oblique[idx_k] or is_perp[idx_k]:
                            rejection_counts["misaligned_angle"] = rejection_counts.get("misaligned_angle", 0) + 1
                            rows.append({
                                "well_i": uwi_i, "well_k": ids[k],
                                "horizontal_dist": np.nan, "horizontal_dist_median": np.nan,
                                "vertical_dist": np.nan, "3D_dist": np.nan,
                                "drill_direction_i": directions[i], "drill_direction_k": directions[k],
                                "n_samples": np.nan, "dy_p5": np.nan,
                                "angle_deg": float(angles[idx_k]),
                                "pair_alignment": "misaligned",
                                "min_distance_ft": np.nan,
                                "mean_windowed_ft": np.nan,
                                "reject_reason": "misaligned_angle",
                                "direction_axis": dir_axis_i,
                                "direction_to_k_from_i_axis": None,
                                "direction_axis_confidence": np.nan,
                                "direction_axis_distribution": "",
                                "axis_forced": True
                            })
                # keep only parallel-like in this branch
                keep_mask = is_parallel
            else:
                keep_mask = np.ones_like(is_parallel, dtype=bool)

            # -------------------- PARALLEL-LIKE branch (vectorized precheck) --------------------
            par_idx = np.where(keep_mask & is_parallel)[0]
            if par_idx.size:
                # Project to i-frame
                Ri = Pi_coarse - origin_i
                Rk = Pk_coarse - origin_i[None, None, :]
                xi = Ri @ ex_i                      # (M_ds,)
                yi = Ri @ ey_i
                xk = np.einsum("kmd,d->km", Rk, ex_i)   # (K, M_ds)
                yk = np.einsum("kmd,d->km", Rk, ey_i)

                xi_min, xi_max = float(xi.min()), float(xi.max())
                xk_min = xk.min(axis=1); xk_max = xk.max(axis=1)
                x_lo = np.maximum(xi_min, xk_min)
                x_hi = np.minimum(xi_max, xk_max)
                has_overlap = x_hi > x_lo

                # Emit no-overlap_x rejects for the affected indices
                if emit_rejected:
                    for local, k in enumerate(k_list):
                        if (local in par_idx) and (not has_overlap[local]):
                            rejection_counts["no_overlap_x"] = rejection_counts.get("no_overlap_x", 0) + 1
                            rows.append({
                                "well_i": uwi_i, "well_k": ids[k],
                                "horizontal_dist": np.nan, "horizontal_dist_median": np.nan,
                                "vertical_dist": np.nan, "3D_dist": np.nan,
                                "drill_direction_i": directions[i], "drill_direction_k": directions[k],
                                "n_samples": np.nan, "dy_p5": np.nan,
                                "angle_deg": float(angles[local]),
                                "pair_alignment": "parallel_like",
                                "min_distance_ft": np.nan,
                                "mean_windowed_ft": np.nan,
                                "reject_reason": "no_overlap_x",
                                "direction_axis": dir_axis_i,
                                "direction_to_k_from_i_axis": None,
                                "direction_axis_confidence": np.nan,
                                "direction_axis_distribution": "",
                                "axis_forced": True
                            })

                # Crossline guardrail
                mask_kj = (xk >= x_lo[:, None]) & (xk <= x_hi[:, None])
                mask_km = (xi[None, :] >= x_lo[:, None]) & (xi[None, :] <= x_hi[:, None])

                YK = yk[:, :, None]
                YI = yi[None, None, :]
                D = np.abs(YK - YI)
                mask_grid = mask_kj[:, :, None] & mask_km[:, None, :]
                D[~mask_grid] = np.inf
                Dmin_km = np.min(D, axis=1)
                with np.errstate(invalid="ignore"):
                    Dmin_km = np.where(np.isfinite(Dmin_km), Dmin_km, np.nan)
                    dy_p = np.nanpercentile(Dmin_km, crossline_percentile, axis=1)

                keep = has_overlap.copy()
                if max_crossline_ft is not None:
                    keep &= np.isfinite(dy_p) & (dy_p <= max_crossline_ft)

                for local, k in enumerate(k_list):
                    if not (local in par_idx and keep[local]):
                        continue

                    # Hand off exact math to the unified pair engine
                    res, _ = self._compute_pair_metrics_and_artifacts(
                        uwi_i=uwi_i,
                        uwi_k=ids[k],
                        step_ft=step_ft,
                        n_samples=n_samples,
                        max_crossline_ft=max_crossline_ft,
                        crossline_percentile=crossline_percentile,
                        ds_crossline_step_ft=ds_crossline_step_ft,
                        use_pca_axis=use_pca_axis,
                        theta_parallel_deg=theta_parallel_deg,
                        theta_perp_deg=theta_perp_deg,
                        reject_misaligned=reject_misaligned,
                        use_windowed_mean=use_windowed_mean,
                        window_ft=window_ft,
                        drill_direction_i=directions[i],
                        drill_direction_k=directions[k],
                        tvd_i=float(coords[i, 2]), tvd_k=float(coords[k, 2]),
                        contact_threshold_ft=contact_threshold_ft,
                        coverage_epsilon=coverage_epsilon,
                        want_artifacts=False
                    )
                    rows.append(_res_to_row(res))


            # -------------------- OBLIQUE / PERPENDICULAR branch (nearest projection) --------------------
            # full-resolution UTM polylines once
            df_i_full = self.trajectories[uwi_i].sort_values("md")
            Xi_utm = df_i_full[["x", "y"]].to_numpy()
            lat_i_full = df_i_full.get("latitude", pd.Series(dtype=float)).to_numpy(float) if "latitude" in df_i_full.columns else None
            lon_i_full = df_i_full.get("longitude", pd.Series(dtype=float)).to_numpy(float) if "longitude" in df_i_full.columns else None

            geod = Geod(ellps="WGS84") if (lat_i_full is not None) else None

            for idx_k, k in enumerate(k_list):
                if not (keep_mask[idx_k] and (is_oblique[idx_k] or is_perp[idx_k])):
                    continue

                # cheap coarse distance cull
                coarse_skip_ft = (max_crossline_ft if max_crossline_ft is not None else 4000.0) + 1000.0
                Pi_c = cache["XY_coarse"][uwi_i]          # (M_ds, 2)
                Pk_c = cache["XY_coarse"][ids[k]]         # (M_ds, 2)
                diff = Pi_c[:, None, :] - Pk_c[None, :, :]       # (M,M,2)
                coarse_min = float(np.sqrt((diff**2).sum(axis=2)).min())

                if coarse_min > coarse_skip_ft:
                    if emit_rejected:
                        rejection_counts["coarse_far"] = rejection_counts.get("coarse_far", 0) + 1
                        rows.append({
                            "well_i": uwi_i, "well_k": ids[k],
                            "horizontal_dist": np.nan, "horizontal_dist_median": np.nan,
                            "vertical_dist": np.nan, "3D_dist": np.nan,
                            "drill_direction_i": directions[i], "drill_direction_k": directions[k],
                            "n_samples": np.nan, "dy_p5": np.nan,
                            "angle_deg": float(angles[idx_k]),
                            "pair_alignment": "oblique" if is_oblique[idx_k] else "perpendicular",
                            "min_distance_ft": np.nan,
                            "mean_windowed_ft": np.nan,
                            "reject_reason": "coarse_far",
                            "direction_axis": dir_axis_i,
                            "direction_to_k_from_i_axis": None,
                            "direction_axis_confidence": np.nan,
                            "direction_axis_distribution": "",
                            "axis_forced": True
                        })
                    continue

                # Survivors after coarse cull → unified pair engine
                res, _ = self._compute_pair_metrics_and_artifacts(
                    uwi_i=uwi_i,
                    uwi_k=ids[k],
                    step_ft=step_ft,
                    n_samples=n_samples,
                    max_crossline_ft=max_crossline_ft,
                    crossline_percentile=crossline_percentile,
                    ds_crossline_step_ft=ds_crossline_step_ft,
                    use_pca_axis=use_pca_axis,
                    theta_parallel_deg=theta_parallel_deg,
                    theta_perp_deg=theta_perp_deg,
                    reject_misaligned=reject_misaligned,
                    use_windowed_mean=use_windowed_mean,
                    window_ft=window_ft,
                    drill_direction_i=directions[i],
                    drill_direction_k=directions[k],
                    tvd_i=float(coords[i, 2]), tvd_k=float(coords[k, 2]),
                    contact_threshold_ft=contact_threshold_ft,
                    coverage_epsilon=coverage_epsilon,
                    want_artifacts=False
                )
                rows.append(_res_to_row(res))

        # Batch summary logging
        batch_elapsed = time.time() - batch_start
        df_batch = pd.DataFrame(rows)

        if len(df_batch) > 0:
            n_valid = (df_batch['reject_reason'] == '').sum() if 'reject_reason' in df_batch.columns else len(df_batch)
            n_rejected = len(df_batch) - n_valid
            valid_pct = (n_valid / len(df_batch) * 100) if len(df_batch) > 0 else 0

            # Alignment distribution
            if 'pair_alignment' in df_batch.columns:
                alignment_counts = df_batch['pair_alignment'].value_counts()
                alignment_summary = ", ".join([f"{align}={count}" for align, count in alignment_counts.items()])
            else:
                alignment_summary = "N/A"

            throughput = n_pairs / batch_elapsed if batch_elapsed > 0 else 0

            self.logger.debug(f"  Batch complete: {len(df_batch):,} results ({n_valid:,} valid, {n_rejected:,} rejected) "
                            f"in {batch_elapsed:.2f}s ({throughput:.0f} pairs/sec)")

            if rejection_counts:
                rejection_summary = ", ".join([f"{reason}={count}" for reason, count in rejection_counts.items()])
                self.logger.debug(f"    Rejections: {rejection_summary}")

            self.logger.debug(f"    Alignments: {alignment_summary}")

        return df_batch
    

#%% # ==================== Directional Bench Neighbors ====================

class DirectionalBenchNeighbors:
    """
    Vectorized nearest-neighbor summarizer by bench *and* opposite direction.

    This class reduces a well-to-well spacing table to **one row per `well_i`**, picking:
      1) SAME-bench closest neighbor (axis-limited or any) → `*_same_1`
      2) SAME-bench closest neighbor in the **opposite** compass direction of `same_1` → `*_same_2`
      3) DIFFERENT-bench closest neighbor (axis-limited or any) → `*_near_1`
      4) DIFFERENT-bench closest neighbor in the **opposite** compass direction of `near_1` → `*_near_2`

    The selection is done in three vectorized phases per category (`same` / `near`):
      (a) For each `(well_i, direction)`, select the argmin pair by `horizontal_dist`
          (ties broken stably by `tie_break_on`, default `well_k`).
      (b) Among **eligible directions** (controlled by `axis_mode`), pick the overall best for `*_1`.
          If `prefer_axis` is set and there’s a tie, bias toward that axis family.
      (c) For `*_2`, pick the best in the **opposite** direction of `*_1` (E↔W, N↔S), if any.

    All filtering (horizontal, vertical, overlap) is vectorized and applied *before* the above selection.

    ----------
    Required inputs
    ----------
    spacing_df : pandas.DataFrame
        Pairwise spacing table with **one row per (well_i, well_k)** and at least:
          - 'well_i' : str/int (will be coerced to str)
          - 'well_k' : str/int (will be coerced to str)
          - 'horizontal_dist' : float (feet)
          - 'vertical_dist'   : float (feet)
          - '3D_dist'         : float (feet)
          - 'direction_to_k_from_i_axis' : {'E','W','N','S'}

        If `overlap_pct_k_min` is provided globally or via overrides, you must also include:
          - 'overlap_pct_k' : float in [0, 1]

    header_df : pandas.DataFrame
        Map of well → bench with:
          - 'uwi'   : str/int (will be coerced to str; must match `well_i`/`well_k`)
          - 'bench' : str (formation/bench label)

    ----------
    Parameters (at call time)
    ----------
    cutoff_ft : float
        Global horizontal cutoff (ft). Pairs with `horizontal_dist > cutoff_ft` are excluded,
        unless a **per-well override** supplies a different cutoff for that `well_i`.

    vertical_cutoff_ft : float, optional
        Global vertical cutoff (ft). If provided, a row must also satisfy `vertical_dist <= vertical_cutoff_ft`.
        If None, *no vertical rule* is applied globally (but per-well overrides can still add one).

    axis_mode : {'any', 'EW', 'NS'}, default 'any'
        Direction eligibility for the initial pick (`*_1`):
          - 'EW' → only E/W are eligible
          - 'NS' → only N/S are eligible
          - 'any' → all four directions are eligible

    prefer_axis : {'EW', 'NS'} or None, default None
        If there is a tie in `horizontal_dist` for `*_1` across directions, prefer the axis family here.

    overlap_pct_k_min : float, optional
        Global overlap rule for *k relative to itself* (k’s own lateral length):
        require `overlap_pct_k >= overlap_pct_k_min`. If None, *no global overlap rule* is applied
        (but per-well overrides can still add one).

    overrides_df : pandas.DataFrame, optional
        **Per-well_i rule overrides** (vectorized). Lets you tailor cutoffs for selected wells while keeping
        a global default for the rest. Recognized columns (all optional):

          - 'well_i'  (or 'uwi'): well identifier (required if `overrides_df` is supplied)
          - 'cutoff_ft'          : horizontal cutoff for this well (ft)
          - 'vertical_cutoff_ft' : vertical cutoff for this well (ft)
          - 'overlap_pct_k_min'  : overlap minimum for this well (unit fraction, e.g., 0.80)

        **Fallback semantics (per well):**
          - For each field, if the per-well value is NaN/missing, it falls back to the global value.
          - If both per-well **and** global are missing for a rule (vertical/overlap), that rule is *not applied* to that well.

        **Axis behavior** (`axis_mode`, `prefer_axis`) remains **global** for all wells.

    ----------
    Output columns
    ----------
    Bound category (positioned after well_i):
        bound_category : str
            Classification of how surrounded the well is by neighbors. One of:
            - 'Unbounded'         : No neighbors at all (isolated well)
            - 'Partially Bounded' : Has neighbor(s) but only on ONE directional side
                                    (e.g., same_1 only, or near_1 only, or same_1+near_1 but no _2's)
            - 'Mostly Bounded'    : Has both sides filled in at least one category (same OR near),
                                    but not both categories fully bounded
                                    (e.g., same_1+same_2 but no near_1+near_2, or vice versa)
            - 'Fully Bounded'     : Has neighbors on both sides in BOTH same-bench and different-bench
                                    (same_1+same_2+near_1+near_2 all present)

    Well_i attributes (positioned after bound_category):
        tvd_i : float
            True vertical depth of well_i at the normalized midpoint position (ft).
            Sourced from spacing_df (computed by WellSpacingCalculator at frac parameter).
        LL_i : float
            Lateral length of well_i (ft). Full measured length of the horizontal lateral section.

    Neighbor picks (per category 'same'/'near'):
        uwi_same_1,  hz_ft_to_same_1,  vt_ft_to_same_1,  3d_ft_to_same_1
        uwi_same_2,  hz_ft_to_same_2,  vt_ft_to_same_2,  3d_ft_to_same_2
        uwi_near_1,  hz_ft_to_near_1,  vt_ft_to_near_1,  3d_ft_to_near_1
        uwi_near_2,  hz_ft_to_near_2,  vt_ft_to_near_2,  3d_ft_to_near_2

    Audit / transparency columns (one row per `well_i`):
        override_applied           : bool
            True if *any* override field was specified for this well in `overrides_df`.
        eff_cutoff_ft              : float
            The **effective** horizontal cutoff used for this well (override → global).
        eff_vertical_cutoff_ft     : float or NaN
            Effective vertical cutoff (ft), or NaN if **no vertical rule** applied to this well.
        eff_overlap_pct_k_min      : float or NaN
            Effective overlap-min (unit fraction), or NaN if **no overlap rule** applied to this well.

    Notes:
        - A well can have all neighbor columns NaN (no survivors after filters) and still carry the audit columns.
        - Distances are in **feet**.

    ----------
    Selection & filtering logic (concise)
    ----------
      1) Build per-row effective thresholds by mapping overrides onto `well_i` and falling back to globals.
      2) Apply masks:
            horizontal_dist <= eff_cutoff_ft
            if eff_vertical_cutoff_ft present → vertical_dist <= eff_vertical_cutoff_ft
            if eff_overlap_pct_k_min present → overlap_pct_k >= eff_overlap_pct_k_min
      3) Split survivors into SAME (bench_i == bench_k) and NEAR (bench_i != bench_k).
      4) Per category:
           a) For each (well_i, direction), keep the argmin by horizontal_dist (tie → `tie_break_on`).
           b) Among eligible directions (axis_mode), pick best overall → `*_1`
              (tie bias → `prefer_axis`).
           c) Opposite direction (E↔W, N↔S) best → `*_2` (if exists).

    ----------
    Error handling
    ----------
      - Missing required columns raise `ValueError`.
      - If any overlap rule (global or per-well) is active but `spacing_df` lacks `overlap_pct_k`, raise `ValueError`.
      - If `overrides_df` is provided without an id column ('well_i' or 'uwi'), raise `ValueError`.

    ----------
    Examples
    ----------
    >>> nb = DirectionalBenchNeighbors()

    # 1) Global horizontal cutoff only
    >>> out1 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,
    ... )
    >>> out1.filter(regex="^uwi_|^hz_ft_|^override_|^eff_").head()

    # 2) Add a global vertical rule (200 ft)
    >>> out2 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,
    ...     vertical_cutoff_ft=200.0,
    ... )

    # 3) Restrict the initial pick to EW directions only
    >>> out3 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,
    ...     axis_mode="EW",
    ... )

    # 4) Prefer NS when ties occur (still allow any direction)
    >>> out4 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,
    ...     axis_mode="any",
    ...     prefer_axis="NS",
    ... )

    # 5) Require at least 80% overlap of k relative to *k* (k's own lateral length)
    >>> out5 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,
    ...     overlap_pct_k_min=0.80,
    ... )

    # 6) Per-well overrides (horizontal only for some wells; others fall back to 1320 ft)
    >>> overrides = pd.DataFrame({
    ...     "well_i": ["30025410040100", "30025421210000"],
    ...     "cutoff_ft": [2200.0, 900.0],
    ... })
    >>> out6 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,                # global fallback for non-listed wells
    ...     overrides_df=overrides,
    ... )
    >>> out6.loc[out6["well_i"].isin(overrides["well_i"]),
    ...          ["well_i", "override_applied", "eff_cutoff_ft"]]

    # 7) Mixed overrides with vertical and overlap rules, plus global defaults
    >>> overrides2 = pd.DataFrame({
    ...     "uwi": ["30025426220000", "30025428730000"],   # 'uwi' also accepted
    ...     "cutoff_ft": [1800.0, np.nan],                 # second uses global 1320.0
    ...     "vertical_cutoff_ft": [250.0, 200.0],          # both have per-well vertical
    ...     "overlap_pct_k_min": [0.70, np.nan],           # second falls back to global 0.60
    ... })
    >>> out7 = nb.summarize(
    ...     spacing_df, header_df,
    ...     cutoff_ft=1320.0,
    ...     vertical_cutoff_ft=150.0,       # global vertical default
    ...     overlap_pct_k_min=0.60,         # global overlap default
    ...     overrides_df=overrides2,
    ... )
    >>> out7.loc[out7["well_i"].isin(overrides2["uwi"]),
    ...          ["well_i", "override_applied",
    ...           "eff_cutoff_ft", "eff_vertical_cutoff_ft", "eff_overlap_pct_k_min"]]

    ----------
    Performance
    ----------
      - All operations are vectorized (`Series.map`, boolean masks, grouped argmin via sort+drop_duplicates).
      - No Python loops over wells; suitable for basin-scale runs.
      - Deterministic outputs via stable tie-breaks (`tie_break_on`).

    Parameters
    ----------
    tie_break_on : str, default 'well_k'
        Secondary stable key when distances tie (used in per-direction argmin).

    """

    # --- Type aliases inside the class ---
    Direction = Literal["E", "W", "N", "S"]
    AxisMode = Literal["any", "EW", "NS"]
    AxisPref = Literal["EW", "NS"]

    _OPPOSITE: Dict[Direction, Direction] = {"E": "W", "W": "E", "N": "S", "S": "N"}

    def __init__(self, *, tie_break_on: str = "well_k", overrides_df: Optional[pd.DataFrame] = None, logger: Optional[logging.Logger] = None,) -> None:
        """
        Parameters
        ----------
        tie_break_on : str, default 'well_k'
            Secondary stable key when distances tie.
        """
        # Set up hierarchical logger
        parent_logger = logger or logging.getLogger("directional_bench_neighbors")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.directional_bench_neighbors")
        self.logger.propagate = True

        self.logger.info("Initializing DirectionalBenchNeighbors")
        self.logger.debug(f"  Tie-break column: {tie_break_on}")

        self.tie_break_on = tie_break_on
        self._overrides_df_default = overrides_df

        if overrides_df is not None:
            n_overrides = len(overrides_df)
            self.logger.info(f"✓ Initialized with default overrides for {n_overrides:,} wells")
        else:
            self.logger.debug("  No default overrides configured")

    def _resolve_overrides(self, overrides_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return overrides_df if overrides_df is not None else self._overrides_df_default

    # ---------- Public API ----------

    def summarize(
        self,
        spacing_df: pd.DataFrame,
        header_df: pd.DataFrame,
        *,
        cutoff_ft: float,
        vertical_cutoff_ft: Optional[float] = None,
        axis_mode: AxisMode = "any",
        prefer_axis: Optional[AxisPref] = None,
        overlap_pct_k_min: Optional[float] = None,
        overrides_df: Optional[pd.DataFrame] = None,
        proj_coverage_min: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return one summary row per well_i with *_same_{1,2} and *_near_{1,2}.

        Output includes well_i attributes (tvd_i, LL_i), bound category classification,
        neighbor selections in four slots (same_1, same_2, near_1, near_2), and
        effective cutoff audit columns.

        See class docstring for complete details on behavior, parameters, and examples.

        Returns
        -------
        pd.DataFrame
            One row per well_i with columns:
            - well_i : str
            - bound_category : str (Unbounded/Partially/Mostly/Fully Bounded)
            - tvd_i : float (true vertical depth at midpoint, ft)
            - LL_i : float (lateral length, ft)
            - uwi_same_1, hz_ft_to_same_1, vt_ft_to_same_1, 3d_ft_to_same_1
            - uwi_same_2, hz_ft_to_same_2, vt_ft_to_same_2, 3d_ft_to_same_2
            - uwi_near_1, hz_ft_to_near_1, vt_ft_to_near_1, 3d_ft_to_near_1
            - uwi_near_2, hz_ft_to_near_2, vt_ft_to_near_2, 3d_ft_to_near_2
            - override_applied, eff_cutoff_ft, eff_vertical_cutoff_ft, eff_overlap_pct_k_min

        Raises
        ------
        ValueError
            If required columns are missing or inputs are inconsistent.
        """

        # ═══════════════════════════════════════════════════════════════════════
        self.logger.info("=" * 80)
        self.logger.info("DIRECTIONAL BENCH NEIGHBORS - IDENTIFICATION PIPELINE")
        self.logger.info("=" * 80)

        pipeline_start = time.time()

        overrides_df = self._resolve_overrides(overrides_df)

        self._validate_inputs(spacing_df, header_df)

        n_input_pairs = len(spacing_df)
        n_input_wells = spacing_df["well_i"].nunique()

        self.logger.info(f"Input: {n_input_pairs:,} spacing pairs across {n_input_wells:,} wells")
        self.logger.debug(f"Configuration:")
        self.logger.debug(f"  Horizontal cutoff: {cutoff_ft} ft")
        self.logger.debug(f"  Vertical cutoff: {vertical_cutoff_ft} ft" if vertical_cutoff_ft else "  Vertical cutoff: None")
        self.logger.debug(f"  Overlap min: {overlap_pct_k_min:.1%}" if overlap_pct_k_min else "  Overlap min: None")
        self.logger.debug(f"  Axis mode: {axis_mode}")
        self.logger.debug(f"  Direction preference: {prefer_axis}" if prefer_axis else "  Direction preference: None")
        if proj_coverage_min:
            self.logger.debug(f"  Coverage quality gate: {proj_coverage_min:.1%}")

        if overrides_df is not None:
            n_overrides = len(overrides_df)
            self.logger.info(f"Applying per-well overrides for {n_overrides:,} wells")

        # Normalize IDs to strings
        spacing = spacing_df.copy()
        spacing["well_i"] = spacing["well_i"].astype(str)
        spacing["well_k"] = spacing["well_k"].astype(str)

        header = header_df.copy()
        header["uwi"] = header["uwi"].astype(str)

        # Map benches
        bench_map = header.set_index("uwi")["bench"]
        spacing["bench_i"] = spacing["well_i"].map(bench_map)
        spacing["bench_k"] = spacing["well_k"].map(bench_map)

        n_benches = header["bench"].nunique()
        self.logger.debug(f"Mapped {n_benches} unique bench(es) to wells")


        # --- NEW: adaptive adjacency percentage (parallel uses overlap; oblique/perp use contact)
        # Expect these columns to exist from your updated spacing:
        #   'pair_alignment', 'overlap_pct_k', 'contact_pct_i_interior', 'contact_pct_i', 'proj_coverage_i_pct'
        if "pair_alignment" not in spacing.columns:
            raise ValueError("spacing_df is missing 'pair_alignment' needed for adaptive adjacency.")

        # Prefer interior contact percentage for oblique/perp; fallback to contact_pct_i if interior is NaN
        contact_interior = spacing.get("contact_pct_i_interior")
        contact_any      = spacing.get("contact_pct_i")

        adj_oblique_perp = None
        if contact_interior is not None:
            adj_oblique_perp = contact_interior.copy()
            if contact_any is not None:
                adj_oblique_perp = adj_oblique_perp.fillna(contact_any)
        else:
            # If interior metric not present, fall back entirely to contact_pct_i (may be all-NaN for parallel rows)
            adj_oblique_perp = contact_any

        # Build unified adjacency percentage
        spacing["adj_pct"] = np.where(
            spacing["pair_alignment"].eq("parallel_like"),
            spacing.get("overlap_pct_k", np.nan),
            (adj_oblique_perp if adj_oblique_perp is not None else np.nan)
        ).astype(float)


        # ---------- NEW: set up per-well effective rules (independent of filtering) ----------
        all_wells = spacing_df["well_i"].astype(str).drop_duplicates().reset_index(drop=True)

        # ---------- NEW: Extract well_i attributes (tvd_i, LL_i) from spacing_df ----------
        # These are constant per well_i, so we take the first occurrence
        well_attrs = spacing_df[["well_i", "tvd_i", "LL_i"]].copy()
        well_attrs["well_i"] = well_attrs["well_i"].astype(str)
        well_attrs = well_attrs.drop_duplicates(subset=["well_i"]).reset_index(drop=True)
        well_attrs_df = well_attrs  # Will be merged into final output

        # Defaults: no overrides
        override_wells = set()
        ov_cut = ov_vcut = ov_omin = pd.Series(dtype=float)

        if overrides_df is not None:
            ov = overrides_df.copy()
            if "well_i" not in ov.columns and "uwi" in ov.columns:
                ov = ov.rename(columns={"uwi": "well_i"})
            if "well_i" not in ov.columns:
                raise ValueError("overrides_df must contain 'well_i' (or 'uwi').")

            ov["well_i"] = ov["well_i"].astype(str)
            ov = ov.set_index("well_i")

            # Which wells have ANY override specified?
            ov_any = ov.reindex(columns=["cutoff_ft", "vertical_cutoff_ft", "overlap_pct_k_min"])
            ov_any_flag = ov_any.notna().any(axis=1)
            override_wells = set(ov_any_flag.index[ov_any_flag])

            # Pull series (may be empty if column absent)
            ov_cut  = ov["cutoff_ft"]           if "cutoff_ft" in ov.columns           else pd.Series(dtype=float)
            ov_vcut = ov["vertical_cutoff_ft"]  if "vertical_cutoff_ft" in ov.columns  else pd.Series(dtype=float)
            ov_omin = ov["overlap_pct_k_min"]   if "overlap_pct_k_min" in ov.columns   else pd.Series(dtype=float)

        # Effective horizontal cutoff per WELL (always defined via override→global)
        eff_cutoff_well = all_wells.map(ov_cut) if not ov_cut.empty else pd.Series(np.nan, index=all_wells.index)
        eff_cutoff_well = eff_cutoff_well.fillna(cutoff_ft)

        # Effective vertical cutoff per WELL (NaN means "no vertical rule")
        if vertical_cutoff_ft is not None or not ov_vcut.empty:
            eff_vcut_well = all_wells.map(ov_vcut) if not ov_vcut.empty else pd.Series(np.nan, index=all_wells.index)
            if vertical_cutoff_ft is not None:
                eff_vcut_well = eff_vcut_well.fillna(vertical_cutoff_ft)
        else:
            eff_vcut_well = pd.Series(np.nan, index=all_wells.index)

        # Effective overlap min per WELL (NaN means "no overlap rule")
        if overlap_pct_k_min is not None or not ov_omin.empty:
            eff_omin_well = all_wells.map(ov_omin) if not ov_omin.empty else pd.Series(np.nan, index=all_wells.index)
            if overlap_pct_k_min is not None:
                eff_omin_well = eff_omin_well.fillna(overlap_pct_k_min)
        else:
            eff_omin_well = pd.Series(np.nan, index=all_wells.index)

        # Build a tiny audit frame we’ll merge into the final output
        audit_df = pd.DataFrame({
            "well_i": all_wells,
            "override_applied": all_wells.isin(override_wells),
            "eff_cutoff_ft": eff_cutoff_well.values,
            "eff_vertical_cutoff_ft": eff_vcut_well.values,
            "eff_overlap_pct_k_min": eff_omin_well.values,
        })

        # ---------- Existing per-row effective rules for filtering (unchanged behavior) ----------
        # (We could reuse the *_well series by mapping, but keeping row-wise is fine & clear.)
        if overrides_df is not None:
            eff_hcut = spacing["well_i"].map(ov_cut) if not ov_cut.empty else pd.Series(np.nan, index=spacing.index)
            eff_hcut = eff_hcut.fillna(cutoff_ft)

            if vertical_cutoff_ft is not None or not ov_vcut.empty:
                eff_vcut = spacing["well_i"].map(ov_vcut) if not ov_vcut.empty else pd.Series(np.nan, index=spacing.index)
                if vertical_cutoff_ft is not None:
                    eff_vcut = eff_vcut.fillna(vertical_cutoff_ft)
            else:
                eff_vcut = pd.Series(np.nan, index=spacing.index)

            if overlap_pct_k_min is not None or not ov_omin.empty:
                eff_omin = spacing["well_i"].map(ov_omin) if not ov_omin.empty else pd.Series(np.nan, index=spacing.index)
                if overlap_pct_k_min is not None:
                    eff_omin = eff_omin.fillna(overlap_pct_k_min)
            else:
                eff_omin = pd.Series(np.nan, index=spacing.index)
        else:
            eff_hcut = pd.Series(cutoff_ft, index=spacing.index)
            eff_vcut = pd.Series(vertical_cutoff_ft, index=spacing.index)  # may be all-None/NaN
            eff_omin = pd.Series(overlap_pct_k_min, index=spacing.index)   # may be all-None/NaN

        # Apply filters (same logic as before)
        mask = spacing["horizontal_dist"] <= eff_hcut
        if eff_vcut.notna().any():
            has_vrule = eff_vcut.notna()
            mask &= (~has_vrule) | (spacing["vertical_dist"] <= eff_vcut)
        # --- NEW: adaptive adjacency rule (reuses eff_omin threshold for all alignments)
        if eff_omin.notna().any():
            has_orule = eff_omin.notna()
            # If adj_pct is NaN for a row with a rule, it fails (same behavior as before).
            mask &= (~has_orule) | (spacing["adj_pct"] >= eff_omin)

        # --- NEW: optional coverage quality gate for oblique/perp only
        if proj_coverage_min is not None:
            if "proj_coverage_i_pct" not in spacing.columns:
                raise ValueError("proj_coverage_min provided but 'proj_coverage_i_pct' missing in spacing_df.")
            # Keep parallel_like rows as-is; gate only oblique/perp
            ok_cov = np.where(
                spacing["pair_alignment"].eq("parallel_like"),
                True,
                spacing["proj_coverage_i_pct"] >= float(proj_coverage_min)
            )
            mask &= ok_cov

        spacing = spacing.loc[mask].copy()

        n_survivors = len(spacing)
        n_filtered_out = n_input_pairs - n_survivors
        survival_rate = (n_survivors / n_input_pairs * 100) if n_input_pairs > 0 else 0

        self.logger.info("─" * 80)
        self.logger.info(f"Filtering complete: {n_survivors:,} pairs survived ({survival_rate:.1f}%)")
        if n_filtered_out > 0:
            self.logger.debug(f"  Filtered out: {n_filtered_out:,} pairs ({100-survival_rate:.1f}%)")

        if spacing.empty:
            self.logger.warning("No pairs survived filtering - returning empty summary")
            out = self._empty_summary(all_wells)
            # Merge well attributes and audit columns even when no neighbors survive
            out = out.merge(well_attrs_df, on="well_i", how="left")
            out = out.merge(audit_df, on="well_i", how="left")

            pipeline_elapsed = time.time() - pipeline_start
            self.logger.info("=" * 80)
            self.logger.info(f"PIPELINE COMPLETE: {pipeline_elapsed:.2f}s")
            self.logger.info("=" * 80)
            return out

        # SAME vs NEAR summaries (unchanged)
        self.logger.info("─" * 80)
        self.logger.info("Computing directional neighbor summaries")
        spacing["is_same"] = spacing["bench_i"] == spacing["bench_k"]
        same = spacing.loc[spacing["is_same"]].copy()
        near = spacing.loc[~spacing["is_same"]].copy()

        self.logger.debug(f"  Same-bench pairs: {len(same):,}")
        self.logger.debug(f"  Different-bench pairs: {len(near):,}")

        same_summary = self._compute_category_summary(same, category="same", axis_mode=axis_mode, prefer_axis=prefer_axis)
        near_summary = self._compute_category_summary(near, category="near", axis_mode=axis_mode, prefer_axis=prefer_axis)

        out = (
            all_wells.to_frame(name="well_i")
            .merge(well_attrs_df, on="well_i", how="left")  # Add tvd_i and LL_i
            .merge(same_summary, on="well_i", how="left")
            .merge(near_summary, on="well_i", how="left")
            # ---------- NEW: audit columns ----------
            .merge(audit_df, on="well_i", how="left")
        )

        # ---------- Add bound_category column ----------
        has_same_1 = out["uwi_same_1"].notna()
        has_same_2 = out["uwi_same_2"].notna()
        has_near_1 = out["uwi_near_1"].notna()
        has_near_2 = out["uwi_near_2"].notna()

        has_any_neighbor = has_same_1 | has_same_2 | has_near_1 | has_near_2
        same_both_sides = has_same_1 & has_same_2
        near_both_sides = has_near_1 & has_near_2

        conditions = [
            ~has_any_neighbor,                      # Unbounded
            same_both_sides & near_both_sides,      # Fully Bounded
            same_both_sides | near_both_sides,      # Mostly Bounded
        ]
        choices = ["Unbounded", "Fully Bounded", "Mostly Bounded"]
        out["bound_category"] = np.select(conditions, choices, default="Partially Bounded")

        ordered_cols = (
            ["well_i", "bound_category", "tvd_i", "LL_i"]  # Added tvd_i and LL_i after bound_category
            + self._category_cols("same", 1)
            + self._category_cols("same", 2)
            + self._category_cols("near", 1)
            + self._category_cols("near", 2)
            + ["override_applied", "eff_cutoff_ft", "eff_vertical_cutoff_ft", "eff_overlap_pct_k_min"]
        )
        existing_cols = [c for c in ordered_cols if c in out.columns]
        final_out = out[existing_cols + [c for c in out.columns if c not in existing_cols]]

        # Final summary
        pipeline_elapsed = time.time() - pipeline_start

        n_wells_with_neighbors = final_out[
            final_out["uwi_same_1"].notna() |
            final_out["uwi_same_2"].notna() |
            final_out["uwi_near_1"].notna() |
            final_out["uwi_near_2"].notna()
        ].shape[0]

        pct_with_neighbors = (n_wells_with_neighbors / len(final_out) * 100) if len(final_out) > 0 else 0

        # Bound category distribution
        bound_counts = final_out["bound_category"].value_counts()
        bound_summary = ", ".join([f"{cat}={count}" for cat, count in bound_counts.items()])

        self.logger.info("─" * 80)
        self.logger.info(f"✓ Summary complete: {len(final_out):,} wells analyzed")
        self.logger.info(f"  Wells with neighbors: {n_wells_with_neighbors:,} ({pct_with_neighbors:.1f}%)")
        self.logger.debug(f"  Bound categories: {bound_summary}")

        self.logger.info("=" * 80)
        self.logger.info(f"PIPELINE COMPLETE: {pipeline_elapsed:.2f}s")
        self.logger.info("=" * 80)

        return final_out

    @staticmethod
    def classify_bound_category(summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each well into a bound category based on neighbor configuration.

        This method implements a 4-category system based on the presence/absence
        of neighbors in same-bench (lateral) and different-bench (vertical/stacked) directions.

        Categories:
        -----------
        1. Unbounded: No neighbors at all (isolated well)
           - same_1=❌, same_2=❌, near_1=❌, near_2=❌

        2. Partially Bounded: Has neighbor(s) but only on ONE directional side
           - Has at least one neighbor, but no category has both _1 and _2 filled
           - Examples: only same_1, only near_1, same_1+near_1 (but no _2's)

        3. Mostly Bounded: Has both sides filled in at least one category, but not both
           - Has same_1+same_2 OR near_1+near_2, but not BOTH pairs complete
           - Examples: same_1+same_2 (laterally bounded), near_1+near_2 (vertically bounded)

        4. Fully Bounded: Has neighbors on both sides in BOTH same and different bench
           - same_1=✅, same_2=✅, near_1=✅, near_2=✅

        Parameters
        ----------
        summary_df : pd.DataFrame
            Output from the `summarize()` method containing columns:
            uwi_same_1, uwi_same_2, uwi_near_1, uwi_near_2

        Returns
        -------
        pd.DataFrame
            Input dataframe with additional column 'bound_category'

        Examples
        --------
        >>> nb = DirectionalBenchNeighbors()
        >>> summary = nb.summarize(spacing_df, header_df, cutoff_ft=1800)
        >>> summary_with_category = DirectionalBenchNeighbors.classify_bound_category(summary)
        >>> summary_with_category['bound_category'].value_counts()
        """
        df = summary_df.copy()

        # Check presence of each neighbor type
        has_same_1 = df["uwi_same_1"].notna()
        has_same_2 = df["uwi_same_2"].notna()
        has_near_1 = df["uwi_near_1"].notna()
        has_near_2 = df["uwi_near_2"].notna()

        # Derived flags
        has_any_neighbor = has_same_1 | has_same_2 | has_near_1 | has_near_2
        same_both_sides = has_same_1 & has_same_2  # Laterally bounded (same bench)
        near_both_sides = has_near_1 & has_near_2  # Vertically bounded (different bench)

        # Classification logic
        def classify(row_idx):
            if not has_any_neighbor.iloc[row_idx]:
                return "Unbounded"
            elif same_both_sides.iloc[row_idx] and near_both_sides.iloc[row_idx]:
                return "Fully Bounded"
            elif same_both_sides.iloc[row_idx] or near_both_sides.iloc[row_idx]:
                return "Mostly Bounded"
            else:
                return "Partially Bounded"

        # Vectorized approach for performance
        conditions = [
            ~has_any_neighbor,                                      # Unbounded
            same_both_sides & near_both_sides,                      # Fully Bounded
            same_both_sides | near_both_sides,                      # Mostly Bounded
        ]
        choices = ["Unbounded", "Fully Bounded", "Mostly Bounded"]

        df["bound_category"] = np.select(conditions, choices, default="Partially Bounded")

        return df

    @staticmethod
    def get_bound_category_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary count of wells by bound category.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Output from `classify_bound_category()` containing 'bound_category' column

        Returns
        -------
        pd.DataFrame
            Summary with counts and percentages for each category
        """
        if "bound_category" not in summary_df.columns:
            summary_df = DirectionalBenchNeighbors.classify_bound_category(summary_df)

        # Define category order
        cat_order = ["Unbounded", "Partially Bounded", "Mostly Bounded", "Fully Bounded"]

        counts = summary_df["bound_category"].value_counts()
        total = len(summary_df)

        result = pd.DataFrame({
            "bound_category": cat_order,
            "count": [counts.get(cat, 0) for cat in cat_order],
            "percentage": [counts.get(cat, 0) / total * 100 for cat in cat_order]
        })

        return result

    def summarize_avg_spacing(
        self,
        spacing_df: pd.DataFrame,
        *,
        cutoff_ft: float,
        vertical_cutoff_ft: Optional[float] = None,
        overlap_pct_k_min: Optional[float] = None,
        overrides_df: Optional[pd.DataFrame] = None,
        proj_coverage_min: Optional[float] = None,
        axis_mode: "DirectionalBenchNeighbors.AxisMode" = "any",
        neighborhood_mode: Literal["i2nbr", "chain", "dense"] = "chain",
        chain_sort_mode: Literal["x", "pca"] = "pca",
        trajectories: Optional[pd.DataFrame] = None,
        edge_pick: Literal["min", "mean", "forward"] = "min",
        include_unweighted: bool = False,
        # --- Debug / diagnostic plotting ---
        debug_well_i: Optional[str] = None,
        debug_mode: Literal["A", "B", "both"] = "both",
        debug_show: bool = False,
        debug_save_dir: Optional[str] = None,
        debug_max_members: int = 30,
        debug_label_members: bool = True,
    ) -> pd.DataFrame:
        """
        Summarize per-well average spacing by building neighborhoods and computing aggregate metrics.

        For each well_i, this method:
        1. Builds a neighborhood S(i) = {well_i} ∪ {eligible neighbors k}
        2. Computes an average spacing metric using one of three aggregation modes
        3. Returns one row per well_i with the computed metrics

        Parameters
        ----------
        spacing_df : pd.DataFrame
            Pairwise spacing table from WellSpacingCalculator.calculate_spacing().
            Required columns: well_i, well_k, horizontal_dist, vertical_dist,
            pair_alignment, direction_to_k_from_i_axis, LL_i.
        cutoff_ft : float
            Global horizontal cutoff (ft). Pairs with horizontal_dist > cutoff are excluded.
        vertical_cutoff_ft : float, optional
            Global vertical cutoff (ft). If provided, pairs must satisfy vertical_dist <= cutoff.
        overlap_pct_k_min : float, optional
            Minimum overlap/adjacency percentage (0-1). For parallel pairs: overlap_pct_k.
            For oblique/perp: contact_pct_i_interior.
        overrides_df : pd.DataFrame, optional
            Per-well override table with columns: well_i (or uwi), cutoff_ft,
            vertical_cutoff_ft, overlap_pct_k_min. Overrides global values for specific wells.
        proj_coverage_min : float, optional
            Minimum projection coverage (0-1) for oblique/perp pairs.
        axis_mode : {'any', 'EW', 'NS'}, default 'any'
            Direction eligibility: 'EW' = only E/W directions, 'NS' = only N/S, 'any' = all.
        neighborhood_mode : {'i2nbr', 'chain', 'dense'}, default 'chain'
            Aggregation mode:
            - 'i2nbr': Mean of i→k survivor distances (simple average).
            - 'chain': Chain adjacency A→B→C→D sorted by chain_sort_mode (no repeats).
            - 'dense': Mean of all unique undirected member-member pairs in S(i).
        chain_sort_mode : {'x', 'pca'}, default 'pca'
            For chain mode, how to order members: 'x' = by UTM X coordinate,
            'pca' = by projection onto first principal component (handles any orientation).
        trajectories : pd.DataFrame, optional
            Required for chain/dense modes. Long-format trajectory table with columns:
            uwi, x, y (and optionally md).
        edge_pick : {'min', 'mean', 'forward'}, default 'min'
            For chain mode, how to resolve bidirectional edges:
            - 'min': min(u→v, v→u) distance
            - 'mean': average of both directions
            - 'forward': use u→v only (chain order)
        include_unweighted : bool, default False
            If True, include hz_mean_i2nbr_ft and vt_mean_i2nbr_ft columns.
        debug_well_i : str, optional
            If specified, generate debug plot for this well's neighborhood.
        debug_mode : {'A', 'B', 'both'}, default 'both'
            Debug plot style: 'A' = chain edges, 'B' = dense pairs, 'both' = overlay both.
        debug_show : bool, default False
            If True, display debug plot interactively.
        debug_save_dir : str, optional
            Directory to save debug plot PNG.
        debug_max_members : int, default 30
            Maximum neighborhood members to show in debug plot.
        debug_label_members : bool, default True
            If True, label members in debug plot legend.

        Returns
        -------
        pd.DataFrame
            One row per well_i with columns:
            - well_i : str
            - group_size : int, size of neighborhood S(i)
            - neighbors_in_group : int, group_size - 1
            - survivor_rows_i_to_k : int, number of eligible i→k pairs
            - avg_hz_spacing_ft : float, aggregated horizontal spacing
            - avg_vt_spacing_ft : float, aggregated vertical spacing
            - neighborhood_mode_used : str
            - (mode-specific columns for chain/dense statistics)

        Notes
        -----
        proj_coverage_min fallback: If spacing_df lacks proj_coverage_i_pct, the method
        derives it from overlap_pct_i (preferred) or overlap_len_common_ft / LL_i.
        The coverage gate applies only to non-parallel pairs.
        """
        overrides_df = self._resolve_overrides(overrides_df)

        spacing = spacing_df.copy()
        spacing["well_i"] = spacing["well_i"].astype(str)
        spacing["well_k"] = spacing["well_k"].astype(str)

        req_base = {
            "well_i",
            "well_k",
            "horizontal_dist",
            "vertical_dist",
            "pair_alignment",
            "direction_to_k_from_i_axis",
            "LL_i",
        }
        missing = req_base - set(spacing.columns)
        if missing:
            raise ValueError(f"summarize_avg_spacing: spacing_df missing required columns: {sorted(missing)}")

        all_wells = spacing["well_i"].drop_duplicates().reset_index(drop=True).astype(str)
        do_debug = (debug_well_i is not None) and (debug_show or (debug_save_dir is not None))

        # ---------------- Overrides (per well_i) ----------------
        ov_cut = ov_vcut = ov_omin = None
        if overrides_df is not None and not overrides_df.empty:
            ov = overrides_df.copy()
            if "well_i" not in ov.columns and "uwi" in ov.columns:
                ov = ov.rename(columns={"uwi": "well_i"})
            if "well_i" in ov.columns:
                ov["well_i"] = ov["well_i"].astype(str)
                ov = ov.set_index("well_i")
                ov_cut = ov["cutoff_ft"] if "cutoff_ft" in ov.columns else None
                ov_vcut = ov["vertical_cutoff_ft"] if "vertical_cutoff_ft" in ov.columns else None
                ov_omin = ov["overlap_pct_k_min"] if "overlap_pct_k_min" in ov.columns else None

        # Effective horizontal cutoff per row
        if ov_cut is not None:
            eff_hcut = spacing["well_i"].map(ov_cut).astype(float).fillna(float(cutoff_ft))
        else:
            eff_hcut = pd.Series(float(cutoff_ft), index=spacing.index)

        # Effective vertical cutoff per row (NaN = no rule)
        if (vertical_cutoff_ft is not None) or (ov_vcut is not None):
            eff_vcut = spacing["well_i"].map(ov_vcut).astype(float) if ov_vcut is not None else pd.Series(np.nan, index=spacing.index)
            if vertical_cutoff_ft is not None:
                eff_vcut = eff_vcut.fillna(float(vertical_cutoff_ft))
        else:
            eff_vcut = pd.Series(np.nan, index=spacing.index)

        # Effective overlap/adjacency min per row (NaN = no rule)
        if (overlap_pct_k_min is not None) or (ov_omin is not None):
            eff_omin = spacing["well_i"].map(ov_omin).astype(float) if ov_omin is not None else pd.Series(np.nan, index=spacing.index)
            if overlap_pct_k_min is not None:
                eff_omin = eff_omin.fillna(float(overlap_pct_k_min))
        else:
            eff_omin = pd.Series(np.nan, index=spacing.index)

        # ---------------- Adaptive adjacency percentage ----------------
        contact_interior = spacing["contact_pct_i_interior"].astype(float) if "contact_pct_i_interior" in spacing.columns else pd.Series(np.nan, index=spacing.index)
        contact_any = spacing["contact_pct_i"].astype(float) if "contact_pct_i" in spacing.columns else pd.Series(np.nan, index=spacing.index)
        adj_oblique_perp = contact_interior.fillna(contact_any)

        overlap_pct_k = spacing["overlap_pct_k"].astype(float) if "overlap_pct_k" in spacing.columns else pd.Series(np.nan, index=spacing.index)

        spacing["adj_pct"] = np.where(
            spacing["pair_alignment"].eq("parallel_like"),
            overlap_pct_k,
            adj_oblique_perp,
        ).astype(float)

        # ---------------- Eligibility mask (survivors) ----------------
        mask = spacing["horizontal_dist"].astype(float) <= eff_hcut

        if eff_vcut.notna().any():
            has_vrule = eff_vcut.notna()
            mask &= (~has_vrule) | (spacing["vertical_dist"].astype(float) <= eff_vcut)

        if eff_omin.notna().any():
            has_orule = eff_omin.notna()
            mask &= (~has_orule) | (spacing["adj_pct"] >= eff_omin)

        # --- proj_coverage_i_pct fallback (FIX FOR YOUR ERROR) ---
        if proj_coverage_min is not None:
            if "proj_coverage_i_pct" not in spacing.columns:
                # Prefer overlap_pct_i if present (your sample has overlap_pct_i)
                if "overlap_pct_i" in spacing.columns:
                    spacing["proj_coverage_i_pct"] = spacing["overlap_pct_i"].astype(float)
                # Else approximate from overlap length / LL_i if possible
                elif ("overlap_len_common_ft" in spacing.columns) and ("LL_i" in spacing.columns):
                    denom = spacing["LL_i"].astype(float).replace(0.0, np.nan)
                    spacing["proj_coverage_i_pct"] = (spacing["overlap_len_common_ft"].astype(float) / denom).replace([np.inf, -np.inf], np.nan)
                else:
                    # No info to compute; keep as NaN (non-parallel rows will fail the gate)
                    spacing["proj_coverage_i_pct"] = np.nan

            ok_cov = np.where(
                spacing["pair_alignment"].eq("parallel_like"),
                True,
                spacing["proj_coverage_i_pct"].astype(float) >= float(proj_coverage_min),
            )
            mask &= ok_cov

        if axis_mode == "EW":
            mask &= spacing["direction_to_k_from_i_axis"].isin({"E", "W"})
        elif axis_mode == "NS":
            mask &= spacing["direction_to_k_from_i_axis"].isin({"N", "S"})

        survivors = spacing.loc[mask].copy()

        # ---------------- Baseline i->eligible k summaries ----------------
        g_i = survivors.groupby("well_i", sort=False)
        survivor_rows = g_i.size().rename("survivor_rows_i_to_k").rename_axis("group_i")

        hz_mean_i2nbr = g_i["horizontal_dist"].mean().rename("hz_mean_i2nbr_ft").rename_axis("group_i")
        vt_mean_i2nbr = g_i["vertical_dist"].mean().rename("vt_mean_i2nbr_ft").rename_axis("group_i")

        # ---------------- Build membership table for S(i) ----------------
        members_self = pd.DataFrame({"group_i": all_wells.values, "member": all_wells.values})
        members_nbrs = survivors[["well_i", "well_k"]].rename(columns={"well_i": "group_i", "well_k": "member"})
        members = pd.concat([members_self, members_nbrs], ignore_index=True).drop_duplicates()

        group_size = members.groupby("group_i", sort=False)["member"].nunique().rename("group_size").rename_axis("group_i")
        neighbors_in_group = (group_size - 1).clip(lower=0).rename("neighbors_in_group").rename_axis("group_i")

        want_chain = (neighborhood_mode == "chain") or (do_debug and debug_mode in ("A", "both"))
        want_dense = (neighborhood_mode == "dense") or (do_debug and debug_mode in ("B", "both"))

        # -----------------------------
        # CHAIN MODE (A->B->C->D)
        # -----------------------------
        chain_axis = None
        chain_edges_used = chain_edges_missing = None
        hz_mean_chain = vt_mean_chain = None
        chain_p50 = chain_p25 = chain_p75 = chain_min = chain_max = None
        edges_chain = None
        ordered_members = None

        if want_chain:
            if trajectories is None:
                raise ValueError("summarize_avg_spacing: chain mode (or chain debug) requires `trajectories` with columns ['uwi','x','y'].")

            need_cols = {"uwi", "x", "y"}
            miss_t = need_cols - set(trajectories.columns)
            if miss_t:
                raise ValueError(f"summarize_avg_spacing: trajectories missing columns: {sorted(miss_t)}")

            traj = trajectories.copy()
            traj["uwi"] = traj["uwi"].astype(str)

            centroids = (
                traj.groupby("uwi", sort=False)[["x", "y"]]
                .median()
                .rename(columns={"x": "x_mid", "y": "y_mid"})
            )

            mem_xy = members.merge(
                centroids.reset_index().rename(columns={"uwi": "member"}),
                on="member",
                how="left",
            )

            if mem_xy[["x_mid", "y_mid"]].isna().any(axis=1).any():
                bad = (
                    mem_xy.loc[mem_xy[["x_mid", "y_mid"]].isna().any(axis=1), "member"]
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                )
                raise ValueError(
                    "summarize_avg_spacing: trajectories missing x/y for some neighborhood members. "
                    f"Examples: {bad}"
                )

            mem_xy["chain_sort_mode"] = chain_sort_mode

            def _order_members_one_group(g: pd.DataFrame) -> pd.DataFrame:
                xy = g[["x_mid", "y_mid"]].to_numpy(dtype=float)

                if g.shape[0] <= 1:
                    g["chain_axis_dx"] = 1.0
                    g["chain_axis_dy"] = 0.0
                    g["_score"] = 0.0
                    return g

                if chain_sort_mode == "x":
                    axis = np.array([1.0, 0.0], dtype=float)
                    score = xy[:, 0].copy()
                else:
                    C = xy - xy.mean(axis=0, keepdims=True)
                    if np.allclose(C, 0.0):
                        axis = np.array([1.0, 0.0], dtype=float)
                        score = C @ axis
                    else:
                        _, _, Vt = np.linalg.svd(C, full_matrices=False)
                        axis = Vt[0]

                        # deterministic axis sign
                        if abs(axis[0]) >= abs(axis[1]):
                            if axis[0] < 0:
                                axis = -axis
                        else:
                            if axis[1] < 0:
                                axis = -axis

                        score = (xy - xy.mean(axis=0, keepdims=True)) @ axis

                g["chain_axis_dx"] = float(axis[0])
                g["chain_axis_dy"] = float(axis[1])
                g["_score"] = score
                return g.sort_values(["_score", "member"], ascending=[True, True]).drop(columns=["_score"])

            ordered_members = (
                mem_xy.groupby("group_i", sort=False, group_keys=False)
                .apply(_order_members_one_group)
                .reset_index(drop=True)
            )

            chain_edges = ordered_members[["group_i", "member"]].copy()
            chain_edges["next_member"] = ordered_members.groupby("group_i", sort=False)["member"].shift(-1)
            chain_edges = chain_edges.dropna(subset=["next_member"]).rename(columns={"member": "u", "next_member": "v"})
            chain_edges["u"] = chain_edges["u"].astype(str)
            chain_edges["v"] = chain_edges["v"].astype(str)

            chain_axis = (
                ordered_members.groupby("group_i", sort=False)
                .agg(
                    chain_sort_mode=("chain_sort_mode", "first"),
                    chain_axis_dx=("chain_axis_dx", "first"),
                    chain_axis_dy=("chain_axis_dy", "first"),
                )
                .rename_axis("group_i")
            )

            lookup = spacing[["well_i", "well_k", "horizontal_dist", "vertical_dist"]].copy()
            lookup["well_i"] = lookup["well_i"].astype(str)
            lookup["well_k"] = lookup["well_k"].astype(str)

            uv = lookup.rename(columns={"well_i": "u", "well_k": "v", "horizontal_dist": "hz_uv", "vertical_dist": "vt_uv"})
            vu = lookup.rename(columns={"well_i": "v", "well_k": "u", "horizontal_dist": "hz_vu", "vertical_dist": "vt_vu"})

            edges_chain = chain_edges.merge(uv, on=["u", "v"], how="left").merge(vu, on=["u", "v"], how="left")

            hz_uv = edges_chain["hz_uv"].astype(float)
            hz_vu = edges_chain["hz_vu"].astype(float)
            vt_uv = edges_chain["vt_uv"].astype(float)
            vt_vu = edges_chain["vt_vu"].astype(float)

            if edge_pick == "forward":
                edges_chain["hz_edge_ft"] = hz_uv
                edges_chain["vt_edge_ft"] = vt_uv
            elif edge_pick == "mean":
                edges_chain["hz_edge_ft"] = np.where(hz_uv.notna() & hz_vu.notna(), 0.5 * (hz_uv + hz_vu), hz_uv.fillna(hz_vu))
                edges_chain["vt_edge_ft"] = np.where(vt_uv.notna() & vt_vu.notna(), 0.5 * (vt_uv + vt_vu), vt_uv.fillna(vt_vu))
            else:  # "min"
                edges_chain["hz_edge_ft"] = np.where(hz_uv.notna() & hz_vu.notna(), np.minimum(hz_uv, hz_vu), hz_uv.fillna(hz_vu))
                edges_chain["vt_edge_ft"] = np.where(vt_uv.notna() & vt_vu.notna(), np.minimum(vt_uv, vt_vu), vt_uv.fillna(vt_vu))

            ge_chain = edges_chain.groupby("group_i", sort=False)

            chain_edges_used = ge_chain["hz_edge_ft"].count().rename("chain_edges_used").rename_axis("group_i")
            hz_mean_chain = ge_chain["hz_edge_ft"].mean().rename("hz_mean_chain_ft").rename_axis("group_i")
            vt_mean_chain = ge_chain["vt_edge_ft"].mean().rename("vt_mean_chain_ft").rename_axis("group_i")

            chain_p50 = ge_chain["hz_edge_ft"].median().rename("p50_hz_chain_edge_ft").rename_axis("group_i")
            chain_p25 = ge_chain["hz_edge_ft"].quantile(0.25).rename("p25_hz_chain_edge_ft").rename_axis("group_i")
            chain_p75 = ge_chain["hz_edge_ft"].quantile(0.75).rename("p75_hz_chain_edge_ft").rename_axis("group_i")
            chain_min = ge_chain["hz_edge_ft"].min().rename("min_hz_chain_edge_ft").rename_axis("group_i")
            chain_max = ge_chain["hz_edge_ft"].max().rename("max_hz_chain_edge_ft").rename_axis("group_i")

            chain_edges_missing = ((group_size - 1) - chain_edges_used).rename("chain_edges_missing").rename_axis("group_i")

        # -----------------------------
        # DENSE MODE (all member-member pairs)
        # -----------------------------
        cand_cnt = expected_dir_pairs = candidate_pair_density = None
        pairs_used_dense = hz_mean_dense = vt_mean_dense = None
        dense_p50 = dense_p25 = dense_p75 = dense_min = dense_max = None
        pairs_dense = None

        if want_dense:
            base_cols = ["well_i", "well_k", "horizontal_dist", "vertical_dist", "direction_to_k_from_i_axis"]
            spacing_base = spacing.loc[:, [c for c in base_cols if c in spacing.columns]].copy()

            cand = spacing_base.merge(
                members.rename(columns={"member": "well_i"})[["group_i", "well_i"]],
                on="well_i",
                how="inner",
            )
            cand = cand.merge(
                members.rename(columns={"member": "well_k"})[["group_i", "well_k"]],
                on=["group_i", "well_k"],
                how="inner",
            )

            cand = cand.loc[cand["well_i"] != cand["well_k"]].copy()

            if axis_mode == "EW":
                cand = cand.loc[cand["direction_to_k_from_i_axis"].isin(["E", "W"])]
            elif axis_mode == "NS":
                cand = cand.loc[cand["direction_to_k_from_i_axis"].isin(["N", "S"])]

            cand_cnt = cand.groupby("group_i", sort=False).size().rename("member_member_rows_found").rename_axis("group_i")
            expected_dir_pairs = (group_size * (group_size - 1)).rename("expected_directed_pairs").rename_axis("group_i")
            candidate_pair_density = (cand_cnt / expected_dir_pairs.replace(0, np.nan)).rename("candidate_pair_density").rename_axis("group_i")

            cand2 = cand.rename(
                columns={"well_i": "src", "well_k": "dst", "direction_to_k_from_i_axis": "direction"}
            )

            uu = np.minimum(cand2["src"].astype(str), cand2["dst"].astype(str))
            vv = np.maximum(cand2["src"].astype(str), cand2["dst"].astype(str))

            pairs_dense = cand2.assign(pair_u=uu, pair_v=vv).drop_duplicates(subset=["group_i", "pair_u", "pair_v"], keep="first")

            gp = pairs_dense.groupby("group_i", sort=False)
            pairs_used_dense = gp.size().rename("pairs_used_dense").rename_axis("group_i")
            hz_mean_dense = gp["horizontal_dist"].mean().rename("hz_mean_dense_ft").rename_axis("group_i")
            vt_mean_dense = gp["vertical_dist"].mean().rename("vt_mean_dense_ft").rename_axis("group_i")

            dense_p50 = gp["horizontal_dist"].median().rename("p50_hz_dense_pair_ft").rename_axis("group_i")
            dense_p25 = gp["horizontal_dist"].quantile(0.25).rename("p25_hz_dense_pair_ft").rename_axis("group_i")
            dense_p75 = gp["horizontal_dist"].quantile(0.75).rename("p75_hz_dense_pair_ft").rename_axis("group_i")
            dense_min = gp["horizontal_dist"].min().rename("min_hz_dense_pair_ft").rename_axis("group_i")
            dense_max = gp["horizontal_dist"].max().rename("max_hz_dense_pair_ft").rename_axis("group_i")

        # ---------------- Merge into output (one row per well_i) ----------------
        out = pd.DataFrame({"group_i": all_wells.astype(str)})

        merges = [group_size, neighbors_in_group, survivor_rows]

        if include_unweighted:
            merges += [hz_mean_i2nbr, vt_mean_i2nbr]

        if want_chain:
            merges += [
                chain_axis,
                chain_edges_used,
                chain_edges_missing,
                hz_mean_chain,
                vt_mean_chain,
                chain_p50,
                chain_p25,
                chain_p75,
                chain_min,
                chain_max,
            ]

        if want_dense:
            merges += [
                cand_cnt,
                expected_dir_pairs,
                candidate_pair_density,
                pairs_used_dense,
                hz_mean_dense,
                vt_mean_dense,
                dense_p50,
                dense_p25,
                dense_p75,
                dense_min,
                dense_max,
            ]

        for s in merges:
            if s is None:
                continue
            if getattr(s, "index", None) is not None and s.index.name != "group_i":
                s = s.rename_axis("group_i")
            out = out.merge(s.reset_index(), on="group_i", how="left")

        out = out.rename(columns={"group_i": "well_i"})

        out["group_size"] = out["group_size"].fillna(1).astype(int)
        out["neighbors_in_group"] = out["neighbors_in_group"].fillna(0).astype(int)
        out["survivor_rows_i_to_k"] = out["survivor_rows_i_to_k"].fillna(0).astype(int)

        if want_chain:
            out["chain_edges_used"] = out["chain_edges_used"].fillna(0).astype(int)
            out["chain_edges_missing"] = out["chain_edges_missing"].fillna(out["neighbors_in_group"]).astype(int)

        if want_dense:
            out["member_member_rows_found"] = out["member_member_rows_found"].fillna(0).astype(int)
            out["pairs_used_dense"] = out["pairs_used_dense"].fillna(0).astype(int)

        if neighborhood_mode == "i2nbr":
            out["avg_hz_spacing_ft"] = out.get("hz_mean_i2nbr_ft")
            out["avg_vt_spacing_ft"] = out.get("vt_mean_i2nbr_ft")
            out["neighborhood_mode_used"] = "i2nbr"
        elif neighborhood_mode == "chain":
            out["avg_hz_spacing_ft"] = out.get("hz_mean_chain_ft")
            out["avg_vt_spacing_ft"] = out.get("vt_mean_chain_ft")
            out["neighborhood_mode_used"] = "chain"
        else:
            out["avg_hz_spacing_ft"] = out.get("hz_mean_dense_ft")
            out["avg_vt_spacing_ft"] = out.get("vt_mean_dense_ft")
            out["neighborhood_mode_used"] = "dense"

        # ---------------- Debug plot (optional) ----------------
        def _traj_lookup(uwi: str) -> Optional[pd.DataFrame]:
            if trajectories is None:
                return None
            df = trajectories.loc[trajectories["uwi"].astype(str) == str(uwi)].copy()
            return None if df.empty else df

        def _midpoint_xy(df_traj: pd.DataFrame) -> Tuple[float, float]:
            d = df_traj.sort_values("md") if "md" in df_traj.columns else df_traj
            x0, y0 = float(d["x"].iloc[0]), float(d["y"].iloc[0])
            x1, y1 = float(d["x"].iloc[-1]), float(d["y"].iloc[-1])
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0

        def _plot_group(group_i: str) -> None:

            g = str(group_i)

            mem = (
                members.loc[members["group_i"].astype(str) == g, "member"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            if not mem:
                return
            if len(mem) > int(debug_max_members):
                mem = mem[: int(debug_max_members)]

            surv_k = (
                survivors.loc[survivors["well_i"].astype(str) == g, "well_k"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            surv_k = [x for x in surv_k if x in set(mem)]

            # Load trajectories + midpoints
            mids: Dict[str, Tuple[float, float]] = {}
            trajs: Dict[str, pd.DataFrame] = {}
            for u in mem:
                dfu = _traj_lookup(u)
                if dfu is None:
                    continue
                dfu = dfu.sort_values("md") if "md" in dfu.columns else dfu
                trajs[u] = dfu
                mids[u] = _midpoint_xy(dfu)

            if g not in trajs:
                return

            fig = plt.figure(figsize=(12.5, 8))  # a bit wider for the side legend
            ax = fig.add_subplot(111)

            # Plot each well; keep the returned Line2D so legend shows the *same* color
            line_by_uwi: Dict[str, any] = {}
            for u, dfu in trajs.items():
                x = dfu["x"].to_numpy(float)
                y = dfu["y"].to_numpy(float)
                lw = 2.7 if u == g else (2.0 if u in surv_k else 1.2)
                (ln,) = ax.plot(x, y, linewidth=lw)
                line_by_uwi[u] = ln

            # Midpoints (no text labels on map)
            for u, (mx, my) in mids.items():
                ax.scatter([mx], [my], s=35)

            # Convex hull (optional)
            pts = np.array(list(mids.values()), dtype=float)
            if pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts)
                    poly = pts[hull.vertices]
                    poly = np.vstack([poly, poly[0]])
                    ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=1.5)
                except Exception:
                    pass

            # Cutoff circle around well_i midpoint
            if g in mids:
                mx, my = mids[g]
                hcut_i = float(cutoff_ft)
                if ov_cut is not None and g in ov_cut.index:
                    v = ov_cut.loc[g]
                    if pd.notna(v):
                        hcut_i = float(v)
                ax.add_patch(Circle((mx, my), radius=hcut_i, fill=False, linestyle=":", linewidth=1.5))

            # Chain edges (A): annotate only distances (no well labels)
            if debug_mode in ("A", "both") and edges_chain is not None and ordered_members is not None:
                e = edges_chain.loc[edges_chain["group_i"].astype(str) == g].copy()
                for _, r in e.iterrows():
                    u = str(r["u"]); v = str(r["v"])
                    hz = r.get("hz_edge_ft", np.nan)
                    if (u in mids) and (v in mids):
                        x1, y1 = mids[u]; x2, y2 = mids[v]
                        ax.plot([x1, x2], [y1, y2], linewidth=2.2, linestyle="--")
                        xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        if pd.notna(hz):
                            ax.text(xm, ym, f"{float(hz):.0f} ft", fontsize=8)

            # Dense pairs (B): light lines, no annotations
            if debug_mode in ("B", "both") and pairs_dense is not None:
                p = pairs_dense.loc[pairs_dense["group_i"].astype(str) == g].copy()
                if len(p) > 160:
                    p = p.nsmallest(160, "horizontal_dist")
                for _, r in p.iterrows():
                    u = str(r["pair_u"]); v = str(r["pair_v"])
                    if (u in mids) and (v in mids):
                        x1, y1 = mids[u]; x2, y2 = mids[v]
                        ax.plot([x1, x2], [y1, y2], linewidth=0.9, alpha=0.22)

            # Title
            row = out.loc[out["well_i"].astype(str) == g]
            if not row.empty:
                r0 = row.iloc[0].to_dict()
                title = (
                    f"Neighborhood S(i) diagnostics for well_i={g}\n"
                    f"group_size={r0.get('group_size')}, survivors(i->k)={r0.get('survivor_rows_i_to_k')}"
                )
                if want_dense:
                    title += (
                        f", member_member_rows_found={r0.get('member_member_rows_found')}, "
                        f"candidate_pair_density={r0.get('candidate_pair_density')}"
                    )
                ax.set_title(title)

            ax.set_xlabel("UTM Easting (ft)")
            ax.set_ylabel("UTM Northing (ft)")
            ax.axis("equal")

            # ---------- SIDE LEGEND (color -> UWI) ----------
            # Put well_i first, then neighbors; show short label + full UWI
            legend_order = [g] + [u for u in mem if u != g and u in line_by_uwi]
            handles = [line_by_uwi[u] for u in legend_order if u in line_by_uwi]

            # Legend text: mark i, mark survivors, show full UWI
            labels = []
            for u in legend_order:
                if u not in line_by_uwi:
                    continue
                tag = "i" if u == g else ("k*" if u in set(surv_k) else "k")
                labels.append(f"{tag}: {u}")

            # Make room for legend on the right
            fig.subplots_adjust(right=0.72)
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                fontsize=8,
                title="Legend (color → well)",
                title_fontsize=9,
            )

            if debug_save_dir is not None:
                os.makedirs(debug_save_dir, exist_ok=True)
                fp = os.path.join(debug_save_dir, f"avg_spacing_debug_{g}.png")
                fig.savefig(fp, dpi=160, bbox_inches="tight")

            if debug_show:
                plt.show()
            plt.close(fig)

        if do_debug:
            if trajectories is None:
                raise ValueError("Debug plotting requires trajectories=... (long DataFrame with uwi/x/y).")
            _plot_group(str(debug_well_i))

        return out


    # ---------- Internals: category computation ----------

    def _compute_category_summary(
        self,
        df: pd.DataFrame,
        *,
        category: Literal["same", "near"],
        axis_mode: AxisMode,
        prefer_axis: Optional[AxisPref],
    ) -> pd.DataFrame:
        """
        Build a summary for one category (same/near) using vectorized reductions.

          1) Best-per-direction (well_i, direction) by horizontal_dist
             (tie-break: well_k for determinism).
          2) *_1: among eligible directions (based on axis_mode), pick best overall.
             If `prefer_axis` is set, bias ties toward that axis family.
          3) *_2: pick best entry in the opposite direction of *_1 (if exists).

        Returns
        -------
        pd.DataFrame with:
          ['well_i'] + uwi_{cat}_1, hz_ft_to_{cat}_1, vt_ft_to_{cat}_1, 3d_ft_to_{cat}_1,
                       uwi_{cat}_2, hz_ft_to_{cat}_2, vt_ft_to_{cat}_2, 3d_ft_to_{cat}_2
        """
        if df.empty:
            return pd.DataFrame(columns=["well_i"] + self._category_cols(category, 1) + self._category_cols(category, 2))

        # Step 1: best per-direction (argmin via sort + drop_duplicates)
        per_dir = (
            df.assign(direction=df["direction_to_k_from_i_axis"])
              .sort_values(["well_i", "direction", "horizontal_dist", self.tie_break_on])
              .drop_duplicates(subset=["well_i", "direction"], keep="first")
              .loc[:, ["well_i", "direction", "well_k", "horizontal_dist", "vertical_dist", "3D_dist"]]
        )

        if per_dir.empty:
            return pd.DataFrame(columns=["well_i"] + self._category_cols(category, 1) + self._category_cols(category, 2))

        # Eligible directions for *_1 based on axis_mode
        if axis_mode == "EW":
            eligible = {"E", "W"}
        elif axis_mode == "NS":
            eligible = {"N", "S"}
        else:
            eligible = {"E", "W", "N", "S"}

        per_dir_eligible = per_dir.loc[per_dir["direction"].isin(eligible)].copy()
        if per_dir_eligible.empty:
            # No eligible directions for *_1 => no *_1 and consequently no *_2
            return pd.DataFrame({"well_i": per_dir["well_i"].unique()}).astype({"well_i": str})

        # Step 2: choose *_1 across eligible directions with optional axis preference
        if prefer_axis is None:
            # Pure distance, then well_k
            best_overall = (
                per_dir_eligible.sort_values(["well_i", "horizontal_dist", "well_k"])
                                .drop_duplicates(subset=["well_i"], keep="first")
            )
        else:
            prefer_set = {"E", "W"} if prefer_axis == "EW" else {"N", "S"}
            # Priority 0 if in preferred axis family; 1 otherwise
            per_dir_eligible = per_dir_eligible.assign(
                _axis_priority=np.where(per_dir_eligible["direction"].isin(prefer_set), 0, 1)
            )
            best_overall = (
                per_dir_eligible.sort_values(["well_i", "horizontal_dist", "_axis_priority", "well_k"])
                                .drop_duplicates(subset=["well_i"], keep="first")
            )

        best_overall = best_overall.rename(columns={
            "well_k": f"uwi_{category}_1",
            "horizontal_dist": f"hz_ft_to_{category}_1",
            "vertical_dist": f"vt_ft_to_{category}_1",
            "3D_dist": f"3d_ft_to_{category}_1",
            "direction": f"direction_{category}_1",
        })

        keep_cols_1 = ["well_i",
                       f"uwi_{category}_1",
                       f"hz_ft_to_{category}_1",
                       f"vt_ft_to_{category}_1",
                       f"3d_ft_to_{category}_1",
                       f"direction_{category}_1"]
        best_overall = best_overall.loc[:, keep_cols_1]

        if best_overall.empty:
            return best_overall.drop(columns=[f"direction_{category}_1"], errors="ignore")

        # Step 3: *_2 is best in the opposite direction of *_1 (if present)
        best_overall = best_overall.copy()
        best_overall[f"opp_dir_{category}"] = best_overall[f"direction_{category}_1"].map(self._OPPOSITE)

        # Build lookup for opposite-direction minima (from all per_dir, not just eligible)
        per_dir_for_merge = per_dir.rename(columns={
            "direction": f"opp_dir_{category}",
            "well_k": f"uwi_{category}_2",
            "horizontal_dist": f"hz_ft_to_{category}_2",
            "vertical_dist": f"vt_ft_to_{category}_2",
            "3D_dist": f"3d_ft_to_{category}_2",
        })

        merged = best_overall.merge(
            per_dir_for_merge[["well_i",
                               f"opp_dir_{category}",
                               f"uwi_{category}_2",
                               f"hz_ft_to_{category}_2",
                               f"vt_ft_to_{category}_2",
                               f"3d_ft_to_{category}_2"]],
            on=["well_i", f"opp_dir_{category}"],
            how="left",
        )

        return merged.drop(columns=[f"direction_{category}_1", f"opp_dir_{category}"])

    # ---------- Internals: utilities ----------

    @staticmethod
    def _category_cols(cat: Literal["same", "near"], idx: Literal[1, 2]) -> List[str]:
        """
        Generate column names for a neighbor category and index.

        Parameters
        ----------
        cat : {'same', 'near'}
            Category: 'same' for same-bench, 'near' for different-bench.
        idx : {1, 2}
            Index: 1 for primary direction, 2 for opposite direction.

        Returns
        -------
        List[str]
            Column names: [uwi_{cat}_{idx}, hz_ft_to_{cat}_{idx},
            vt_ft_to_{cat}_{idx}, 3d_ft_to_{cat}_{idx}].
        """
        return [
            f"uwi_{cat}_{idx}",
            f"hz_ft_to_{cat}_{idx}",
            f"vt_ft_to_{cat}_{idx}",
            f"3d_ft_to_{cat}_{idx}",
        ]

    @staticmethod
    def _empty_summary(wells: Iterable[str]) -> pd.DataFrame:
        """
        Create an empty summary DataFrame with all neighbor columns set to NaN.

        Used when there are no eligible neighbors for a set of wells.

        Parameters
        ----------
        wells : Iterable[str]
            Well identifiers to include in the summary.

        Returns
        -------
        pd.DataFrame
            DataFrame with well_i column and all neighbor columns
            (uwi_same_1, hz_ft_to_same_1, etc.) set to NaN.
        """
        cols = ["well_i"]
        for cat in ("same", "near"):
            for idx in (1, 2):
                cols += [
                    f"uwi_{cat}_{idx}",
                    f"hz_ft_to_{cat}_{idx}",
                    f"vt_ft_to_{cat}_{idx}",
                    f"3d_ft_to_{cat}_{idx}",
                ]
        out = pd.DataFrame({"well_i": list(map(str, wells))})
        for c in cols:
            if c != "well_i":
                out[c] = np.nan
        return out

    @staticmethod
    def _validate_inputs(spacing_df: pd.DataFrame, header_df: pd.DataFrame) -> None:
        """
        Validate that required columns are present in input DataFrames.

        Parameters
        ----------
        spacing_df : pd.DataFrame
            Pairwise spacing table to validate.
        header_df : pd.DataFrame
            Well header table with bench assignments.

        Raises
        ------
        ValueError
            If any required columns are missing from either DataFrame.

        Notes
        -----
        Required spacing_df columns: well_i, well_k, horizontal_dist, vertical_dist,
        3D_dist, direction_to_k_from_i_axis.
        Required header_df columns: uwi, bench.
        """
        spacing_required = {
            "well_i",
            "well_k",
            "horizontal_dist",
            "vertical_dist",
            "3D_dist",
            "direction_to_k_from_i_axis",
        }
        header_required = {"uwi", "bench"}

        missing_s = spacing_required - set(spacing_df.columns)
        missing_h = header_required - set(header_df.columns)
        if missing_s:
            raise ValueError(f"spacing_df missing required columns: {sorted(missing_s)}")
        if missing_h:
            raise ValueError(f"header_df missing required columns: {sorted(missing_h)}")

#%% # ==================== Floating Section WPS ====================

Orientation = Literal["cardinal", "i_frame", "corridor"]
@dataclass(frozen=True)
class BoxSpec:
    """Half-sizes of the floating section box (ft)."""
    half_width_ft: float   # x-half-size (east-west extent / 2)
    half_height_ft: float  # y-half-size (north-south extent / 2)
@dataclass(frozen=True)
class CorridorSpec:
    """
    Corridor-style floating region around the reference lateral.

    The corridor lives in the reference well's own azimuth-aligned frame:

      - Along-well half-length = 0.5 * lateral_length + extra_along_ft
      - Cross-well half-width  = half_width_ft
    """
    half_width_ft: float          # cross-well half-width (normal to lateral)
    extra_along_ft: float = 0.0   # margin beyond heel/toe along the lateral

class FloatingSectionWPS:
    """
    Floating section Well-Per-Section (WPS) counter on projected feet coordinates.

    This class counts, for each reference well_i, how many other wells have at least
    `min_inside_ft` lateral **inside** a 2D rectangular window (the "floating section")
    centered on well_i.

    Supported orientations
    ----------------------
    - 'cardinal':
        Box is axis-aligned to map (north-up, east-right). Uses BoxSpec.
    - 'i_frame':
        Box is rotated so the reference lateral is along +x. Uses the same BoxSpec
        half-sizes but in the ref-well frame.
    - 'corridor':
        A lateral-following corridor: rectangle in the ref-well frame whose
        along-direction half-length is derived from that well's lateral length plus
        an optional margin, and whose cross-direction half-width is set by
        CorridorSpec.half_width_ft.

    Required input DataFrame columns (projected feet, e.g., UTM):
        - 'uwi' : unique id (str/int)
        - 'heel_x','heel_y','toe_x','toe_y' : lateral endpoints in feet
      Optional:
        - 'mid_x','mid_y' : midpoint in feet (computed if missing)
        - 'azimuth_deg'   : geodetic-style: 0°=N, 90°=E (computed if missing)

    Parameters
    ----------
    wells_df : pd.DataFrame
        Table of well laterals in projected feet.
    box : BoxSpec
        Box half-sizes (ft). For a 1×1 mile box, pass BoxSpec(2640, 2640).
        Used for 'cardinal' and 'i_frame' orientations.
    min_inside_ft : float, default 660.0
        Minimum **inside-lateral length** required to count a neighbor.
    exclude_self : bool, default True
        Whether to exclude well_i when counting its own box.
    corridor : CorridorSpec or None, default None
        Corridor geometry used only when orientation='corridor'.
    """

    def __init__(
        self,
        wells_df: pd.DataFrame,
        box: BoxSpec = BoxSpec(half_width_ft=2640.0, half_height_ft=2640.0),
        *,
        min_inside_ft: float = 660.0,
        exclude_self: bool = True,
        corridor: Optional[CorridorSpec] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.box = box
        self.corridor = corridor
        self.min_inside_ft = float(min_inside_ft)
        self.exclude_self = bool(exclude_self)
    
        # Set up hierarchical logger
        parent_logger = logger or logging.getLogger("wps")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.wps")
        self.logger.propagate = True

        df = wells_df.copy()

        # Basic validation
        need = {"uwi", "heel_x", "heel_y", "toe_x", "toe_y"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"wells_df missing required columns: {sorted(missing)}")

        # Ensure dtypes
        df["uwi"] = df["uwi"].astype(str)
        for c in ["heel_x", "heel_y", "toe_x", "toe_y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Midpoints if missing
        if not {"mid_x", "mid_y"}.issubset(df.columns):
            df["mid_x"] = 0.5 * (df["heel_x"] + df["toe_x"])
            df["mid_y"] = 0.5 * (df["heel_y"] + df["toe_y"])

        # Azimuth if missing (0°=N, 90°=E). Uses heel->toe direction.
        if "azimuth_deg" not in df.columns:
            dx = df["toe_x"].to_numpy(float) - df["heel_x"].to_numpy(float)
            dy = df["toe_y"].to_numpy(float) - df["heel_y"].to_numpy(float)
            # arctan2(x,y) to get 0°=N,90°=E (swap args)
            az = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
            df["azimuth_deg"] = az

        self.df = df.reset_index(drop=True)
        # Prebuild arrays for vectorized ops
        self._x0 = self.df["heel_x"].to_numpy(float)
        self._y0 = self.df["heel_y"].to_numpy(float)
        self._x1 = self.df["toe_x"].to_numpy(float)
        self._y1 = self.df["toe_y"].to_numpy(float)
        self._mx = self.df["mid_x"].to_numpy(float)
        self._my = self.df["mid_y"].to_numpy(float)
        self._az = self.df["azimuth_deg"].to_numpy(float)
        self._uwi = self.df["uwi"].to_numpy(str)


    # -------------------- helpers: directional survey -> endpoints -----------
    @staticmethod
    def ds_to_lateral_endpoints(ds: pd.DataFrame, *, assume_units: str = "ft") -> pd.DataFrame:
        """
        Collapse a directional-survey style table (many rows per UWI with 'md','x','y')
        into one row per UWI with heel/toe UTM coords (in feet), plus mid & azimuth.

        Required columns in `ds`: 'uwi','md','x','y'
        """
        required = {"uwi", "md", "x", "y"}
        if not required <= set(ds.columns):
            missing = required - set(ds.columns)
            raise ValueError(f"ds_to_lateral_endpoints: missing columns {sorted(missing)}")

        # Sort so that 'first' and 'last' line up with heel & toe
        base = (
            ds.sort_values(["uwi", "md"])
            .groupby("uwi", as_index=False)
            .agg(
                heel_x=("x", "first"),
                heel_y=("y", "first"),
                toe_x=("x", "last"),
                toe_y=("y", "last"),
            )
        )

        out = base.copy()

        # If your x/y are meters, convert once here (the class works in FEET)
        if assume_units == "m":
            out[["heel_x", "toe_x", "heel_y", "toe_y"]] *= 3.28084

        out["mid_x"] = 0.5 * (out["heel_x"] + out["toe_x"])
        out["mid_y"] = 0.5 * (out["heel_y"] + out["toe_y"])

        # azimuth from North (deg), CW
        dx = out["toe_x"] - out["heel_x"]
        dy = out["toe_y"] - out["heel_y"]
        out["azimuth_deg"] = (np.degrees(np.arctan2(dx, dy)) % 360.0).astype(float)

        return out[["uwi", "heel_x", "heel_y", "toe_x", "toe_y", "mid_x", "mid_y", "azimuth_deg"]]


    # ----------------------- math helpers (vectorized) -----------------------

    @staticmethod
    def _rotate_points(
        x: np.ndarray,
        y: np.ndarray,
        theta_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate points by -theta_deg around origin in the usual math sense:
        - 0° is the +X axis (Easting),
        - positive angles are counter-clockwise.

        This is a low-level helper. Higher-level helpers take care of converting
        from geodetic azimuth (0°=N, 90°=E).
        """
        t = math.radians(theta_deg)
        c, s = math.cos(t), math.sin(t)
        xr = c * x + s * y
        yr = -s * x + c * y
        return xr, yr

    # --- azimuth / frame helpers --------------------------------------------

    @staticmethod
    def _az_to_theta_iframe(az_deg: float) -> float:
        """
        Convert geodetic azimuth (0°=N, 90°=E) into a theta (deg) for
        `_rotate_points` such that the well's azimuth maps to +X in the
        local i-frame.

        `_rotate_points(x, y, theta)` always rotates by -theta.
        """
        return 90.0 - az_deg

    @classmethod
    def _world_to_iframe(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        az_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate world (map) coordinates into the i-frame where:

          +X = along the well (heel→toe, azimuth az_deg)
          +Y = left-of-well (cross-well)
        """
        theta = cls._az_to_theta_iframe(az_deg)
        return cls._rotate_points(x, y, theta)

    @classmethod
    def _iframe_to_world(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        az_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate from the i-frame (aligned with the well) back to world
        map coordinates.
        """
        theta = -cls._az_to_theta_iframe(az_deg)  # = az_deg - 90
        return cls._rotate_points(x, y, theta)

    @staticmethod
    def _liang_barsky_clip_len(
        x0: np.ndarray,
        y0: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        a: float,
        b: float,
    ) -> np.ndarray:
        """
        Compute clipped segment lengths inside a rectangle using Liang-Barsky algorithm.

        This vectorized implementation clips line segments against an axis-aligned
        rectangle centered at the origin and returns the inside-length of each segment.

        Parameters
        ----------
        x0, y0 : np.ndarray
            Start coordinates of segments. Shape: (N,).
        x1, y1 : np.ndarray
            End coordinates of segments. Shape: (N,).
        a : float
            Half-width of clipping rectangle (x in [-a, a]).
        b : float
            Half-height of clipping rectangle (y in [-b, b]).

        Returns
        -------
        np.ndarray
            Inside-length for each segment (ft). Shape: (N,).
            Returns 0 if segment is fully outside the rectangle.

        Notes
        -----
        Uses parametric form of line segments and edge-by-edge clipping to find
        the portion of each segment that lies within the rectangle.
        """
        dx = x1 - x0
        dy = y1 - y0

        p = np.stack([-dx, dx, -dy, dy], axis=0)
        q = np.stack([x0 + a, a - x0, y0 + b, b - y0], axis=0)

        u0 = np.zeros_like(dx, dtype=float)
        u1 = np.ones_like(dx, dtype=float)

        for i in range(4):
            pi = p[i]
            qi = q[i]
            mask_zero = np.isclose(pi, 0.0)
            # If pi == 0 and qi < 0 → parallel outside → reject (len=0)
            reject = mask_zero & (qi < 0.0)
            if np.any(reject):
                u0[reject] = 1.0
                u1[reject] = 0.0

            # Non-parallel edges
            nz = ~mask_zero
            ti = qi[nz] / pi[nz]
            neg = (pi[nz] < 0.0)
            pos = ~neg
            # entering boundary
            u0_n = u0[nz]
            u1_n = u1[nz]
            u0_n = np.maximum(u0_n, np.where(neg, ti, u0_n))
            # leaving boundary
            u1_n = np.minimum(u1_n, np.where(pos, ti, u1_n))
            u0[nz] = u0_n
            u1[nz] = u1_n

        # invalid where u1<u0
        invalid = u1 < u0
        seg_len = np.hypot(dx, dy)
        inside_len = seg_len * np.clip(u1 - u0, 0.0, None)
        inside_len[invalid] = 0.0
        return inside_len

    # ----------------------- geometry helpers -----------------------

    def _get_half_sizes(
        self,
        ref_idx: int,
        orientation: Orientation,
    ) -> Tuple[float, float]:
        """
        Determine the half-sizes of the floating region for a reference well.

        Parameters
        ----------
        ref_idx : int
            Index of the reference well in the internal arrays.
        orientation : {'cardinal', 'i_frame', 'corridor'}
            The orientation mode determining box geometry.

        Returns
        -------
        Tuple[float, float]
            (a, b) where a is the half-width along the primary axis and b is the
            half-height along the secondary axis (both in feet).

        Notes
        -----
        - 'cardinal'/'i_frame': Uses global BoxSpec (e.g., 2640 ft for 1-mile section).
        - 'corridor': Uses per-well lateral length + CorridorSpec margins.
          a = 0.5 * lateral_length + extra_along_ft, b = half_width_ft.
        """
        if orientation == "corridor":
            if self.corridor is None:
                raise ValueError(
                    "orientation='corridor' requires FloatingSectionWPS.corridor to be set"
                )
            dx = self._x1[ref_idx] - self._x0[ref_idx]
            dy = self._y1[ref_idx] - self._y0[ref_idx]
            L = float(np.hypot(dx, dy))  # ref lateral length
            a = 0.5 * L + float(self.corridor.extra_along_ft)   # along-well
            b = float(self.corridor.half_width_ft)              # cross-well
            return a, b

        # DSU-style fixed box for 'cardinal' and 'i_frame'
        return self.box.half_width_ft, self.box.half_height_ft

    # ----------------------- core counters -----------------------

    def _inside_lengths_for_ref(
        self,
        ref_idx: int,
        *,
        orientation: Orientation,
    ) -> np.ndarray:
        """
        Compute the inside-lengths of all well segments within the floating region.

        This is the core vectorized computation that determines how much of each
        well's lateral lies within the floating region centered on a reference well.

        Parameters
        ----------
        ref_idx : int
            Index of the reference well in the internal arrays.
        orientation : {'cardinal', 'i_frame', 'corridor'}
            Region orientation:
            - 'cardinal': Axis-aligned box (north-up, east-right) in map frame.
            - 'i_frame': Box rotated so reference lateral is along +X axis.
            - 'corridor': Same rotation as 'i_frame' but with dynamic half-sizes
              derived from the reference lateral's length + CorridorSpec.

        Returns
        -------
        np.ndarray
            Inside-length (ft) for each well's lateral segment. Shape: (N_wells,).
            Zero if the lateral is entirely outside the region.

        Notes
        -----
        The computation translates all segments so the region is centered at origin,
        optionally rotates into the reference well's frame, then applies Liang-Barsky
        clipping to compute inside lengths.
        """
        cx = self._mx[ref_idx]
        cy = self._my[ref_idx]
        a, b = self._get_half_sizes(ref_idx, orientation)

        # translate all segments so the region center is at origin
        x0 = self._x0 - cx
        y0 = self._y0 - cy
        x1 = self._x1 - cx
        y1 = self._y1 - cy

        # Any azimuth-aligned orientations rotate the world into the ref frame
        if orientation in ("i_frame", "corridor"):
            az_ref = float(self._az[ref_idx])
            x0, y0 = self._world_to_iframe(x0, y0, az_ref)
            x1, y1 = self._world_to_iframe(x1, y1, az_ref)

        return self._liang_barsky_clip_len(x0, y0, x1, y1, a, b)

    def count_for_reference(
        self,
        uwi_ref: str,
        *,
        orientation: Orientation = "cardinal",
        include_lengths: bool = True,
    ) -> pd.DataFrame:
        """
        Return a neighbor table for one reference well showing which wells contribute
        and by how much.

        Columns: ['well_k','inside_len_ft','passes_min_inside','orientation','uwi_ref']

        orientation may be 'cardinal', 'i_frame', or 'corridor'.
        """
        ref_idx = int(np.where(self._uwi == str(uwi_ref))[0][0])
        inside = self._inside_lengths_for_ref(ref_idx, orientation=orientation)

        # apply 660-ft rule
        passes = inside >= self.min_inside_ft
        if self.exclude_self:
            passes[ref_idx] = False

        out = pd.DataFrame(
            {
                "uwi_ref": str(uwi_ref),
                "orientation": orientation,
                "well_k": self._uwi,
                "inside_len_ft": inside if include_lengths else np.nan,
                "passes_min_inside": passes,
            }
        )
        return out.loc[passes].sort_values("inside_len_ft", ascending=False).reset_index(drop=True)

    # ----------------------- per-well summaries -----------------------

    def summarize_per_well(
        self
    ) -> pd.DataFrame:
        """
        Compute WPS for every well in both DSU-style modes + anisotropy index.

        wps_iframe > wps_cardinal → more crowding along the well direction.
        wps_iframe < wps_cardinal → more stacked / across-strike crowding.

        If `include_corridor=True`, also compute `wps_corridor` using the
        corridor definition for each well as the reference.

        Returns a DataFrame with one row per well_i:

            ['well_i',
            'wps_cardinal',
            'wps_iframe',
            'anisotropy_ratio',
            'anisotropy_delta',
            'azimuth_deg',
            'mid_x',
            'mid_y',
            ('wps_corridor' if requested)]

        Note: corridor-based WPS requires `self.corridor` to be set.
        """
        n = len(self._uwi)
        wps_card = np.empty(n, dtype=int)
        wps_ifrm = np.empty(n, dtype=int)
        wps_corr = np.empty(n, dtype=int)

        for i in range(n):
            # Cardinal box
            inside_c = self._inside_lengths_for_ref(i, orientation="cardinal")
            # i-frame box
            inside_i = self._inside_lengths_for_ref(i, orientation="i_frame")

            if self.exclude_self:
                inside_c[i] = 0.0
                inside_i[i] = 0.0

            wps_card[i] = int(np.sum(inside_c >= self.min_inside_ft))
            wps_ifrm[i] = int(np.sum(inside_i >= self.min_inside_ft))

            # Corridor box
            inside_corr = self._inside_lengths_for_ref(i, orientation="corridor")
            if self.exclude_self:
                inside_corr[i] = 0.0
            wps_corr[i] = int(np.sum(inside_corr >= self.min_inside_ft))

        # Anisotropy metrics (cardinal vs i-frame)
        eps = 1.0  # guard small denominators for a stable ratio
        ratio = (wps_ifrm.astype(float) / np.maximum(eps, wps_card.astype(float)))
        delta = wps_ifrm.astype(float) - wps_card.astype(float)

        data = {
            "well_i": self._uwi,
            "wps_cardinal": wps_card,
            "wps_iframe": wps_ifrm,
            "wps_corridor": wps_corr,
            "anisotropy_ratio": ratio,
            "anisotropy_delta": delta,
            "azimuth_deg": self._az,
            "mid_x": self._mx,
            "mid_y": self._my,
        }

        return pd.DataFrame(data)

    # ----------------------- diagnostics -----------------------

    def plot_map_for(
        self,
        uwi_ref: str,
        *,
        show_boxes: bool = True,
        arrow_stride: int = 1,
        figsize: Tuple[int, int] = (8, 7),
    ) -> plt.Figure:
        """
        Map diagnostic for a single reference well:

          - Plots all laterals,
          - Draws the **cardinal** and **i-frame** floating boxes centered at uwi_ref,
          - Annotates WPS counts in both DSU-style modes.

        (This helper does not draw the corridor; use `plot_map_with_neighbors` or
        `visualize` with orientation='corridor' instead.)
        """
        ref_idx = int(np.where(self._uwi == str(uwi_ref))[0][0])
        cx, cy = self._mx[ref_idx], self._my[ref_idx]
        az = float(self._az[ref_idx])

        # counts
        inside_c = self._inside_lengths_for_ref(ref_idx, orientation="cardinal")
        inside_i = self._inside_lengths_for_ref(ref_idx, orientation="i_frame")

        if self.exclude_self:
            inside_c[ref_idx] = 0.0
            inside_i[ref_idx] = 0.0

        c_cnt = int(np.sum(inside_c >= self.min_inside_ft))
        i_cnt = int(np.sum(inside_i >= self.min_inside_ft))

        fig, ax = plt.subplots(figsize=figsize)

        # all laterals
        for i in range(len(self._uwi)):
            ax.plot([self._x0[i], self._x1[i]], [self._y0[i], self._y1[i]])

        if show_boxes:
            a, b = self.box.half_width_ft, self.box.half_height_ft

            # cardinal box
            rx = np.array([-a, a, a, -a, -a]) + cx
            ry = np.array([-b, -b, b, b, -b]) + cy
            ax.plot(rx, ry)

            # i-frame box: draw rotated rectangle aligned with ref azimuth
            rect = np.array([[-a, -b], [a, -b], [a, b], [-a, b], [-a, -b]], dtype=float)
            rr_x, rr_y = self._iframe_to_world(rect[:, 0], rect[:, 1], az)
            ax.plot(rr_x + cx, rr_y + cy)

            ax.annotate(
                f"cardinal={c_cnt}, i-frame={i_cnt}",
                xy=(cx, cy),
                xytext=(cx + 0.6 * a, cy + 0.6 * b),
                arrowprops=dict(arrowstyle="->"),
            )

        ax.set_title(f"Floating section boxes for {uwi_ref}")
        ax.set_xlabel("Easting (ft)")
        ax.set_ylabel("Northing (ft)")
        ax.grid(True, alpha=0.2)
        return fig

    def plot_polar_azimuth_hist(
        self,
        *,
        bins: int = 12,
        figsize: Tuple[int, int] = (6, 6),
    ) -> plt.Figure:
        """
        Polar histogram of lateral azimuths (0°=N, 90°=E). Useful for **stratification**.
        """
        th = np.deg2rad(self._az % 360.0)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
        edges = np.linspace(0.0, 2.0 * math.pi, bins + 1)
        counts, edges = np.histogram(th, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=np.diff(edges), align="center")
        ax.set_title("Azimuth distribution (for stratification)")
        return fig

    def plot_wps_by_azimuth_bins(
        self,
        *,
        bin_edges_deg: Iterable[float] = (0, 60, 120, 180),
        figsize: Tuple[int, int] = (7, 5),
    ) -> plt.Figure:
        """
        Bar chart comparing mean WPS across user-specified azimuth bins,
        for both **cardinal** and **i-frame** counts (a normalization view).
        """
        summary = self.summarize_per_well()
        az = summary["azimuth_deg"].to_numpy(float) % 180.0  # mirror symmetry
        wps_c = summary["wps_cardinal"].to_numpy(int)
        wps_i = summary["wps_iframe"].to_numpy(int)

        edges = np.array(list(bin_edges_deg), dtype=float)
        means_c, means_i, labels = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (az >= lo) & (az < hi)
            labels.append(f"{int(lo)}–{int(hi)}°")
            if np.any(mask):
                means_c.append(np.mean(wps_c[mask]))
                means_i.append(np.mean(wps_i[mask]))
            else:
                means_c.append(0.0)
                means_i.append(0.0)

        xpos = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(xpos - width / 2, means_c, width, label="cardinal")
        ax.bar(xpos + width / 2, means_i, width, label="i_frame")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean WPS")
        ax.set_title("Mean WPS by azimuth bins (normalization view)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        return fig

    # ----------------------- diagnostics (enhanced) -----------------------

    def visualize(
        self,
        uwi_ref: str,
        *,
        orientation: Orientation = "cardinal",
        min_len_ft: Optional[float] = None,
        annotate: bool = True,
        max_labels: int = 30,
        figsize: Tuple[float, float] = (7.5, 7.5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Floating-section view for one reference well, with neighbors colored by category.

        Categories are based on the floating box or corridor centered on `uwi_ref`:

          - \">=min_len_ft\" : inside-lateral length >= min_len_ft
          - \"intersect\"    : segment intersects region but inside length < min_len_ft
          - \"midpoint\"     : midpoint lies inside region, but no intersection
          - \"outside\"      : none of the above (not drawn)

        The PLOT is in a **map-style frame** centered on the reference midpoint:
          - X = Easting (ft, local)
          - Y = Northing (ft, local)

        For `orientation` in {'i_frame','corridor'}, we *only* rotate into the
        ref-lateral frame internally for classification and clipping; wells are
        still drawn in map orientation.
        """
        threshold = float(min_len_ft) if min_len_ft is not None else self.min_inside_ft

        # Locate reference well
        uwi_ref = str(uwi_ref)
        try:
            ref_idx = int(np.where(self._uwi == uwi_ref)[0][0])
        except IndexError:
            raise ValueError(f"Reference UWI {uwi_ref!r} not found in FloatingSectionWPS.df")

        # Region half-sizes (a = along, b = cross for corridor / i_frame)
        a, b = self._get_half_sizes(ref_idx, orientation)

        # Center in world coords
        cx, cy = self._mx[ref_idx], self._my[ref_idx]

        # Inside lengths (already uses the correct frame internally)
        inside = self._inside_lengths_for_ref(ref_idx, orientation=orientation)
        if self.exclude_self:
            inside[ref_idx] = 0.0

        # ---------- Coordinates for plotting vs classification ----------

        # World coordinates, translated so ref midpoint is at (0,0) – used for plotting
        x0_plot = self._x0 - cx
        y0_plot = self._y0 - cy
        x1_plot = self._x1 - cx
        y1_plot = self._y1 - cy
        mid_x_plot = self._mx - cx
        mid_y_plot = self._my - cy

        # Local (box-aligned) coordinates for classification:
        #   cardinal → same as plot coords
        #   i_frame / corridor → rotate world into the ref-lateral frame
        if orientation == "cardinal":
            x0_loc, y0_loc = x0_plot, y0_plot
            x1_loc, y1_loc = x1_plot, y1_plot
            mid_x_loc, mid_y_loc = mid_x_plot, mid_y_plot
        else:
            az_ref = float(self._az[ref_idx])
            x0_loc, y0_loc = self._world_to_iframe(x0_plot, y0_plot, az_ref)
            x1_loc, y1_loc = self._world_to_iframe(x1_plot, y1_plot, az_ref)
            mid_x_loc, mid_y_loc = self._world_to_iframe(mid_x_plot, mid_y_plot, az_ref)

        # Midpoint inside region? (computed in the box-aligned frame)
        mid_inside = (np.abs(mid_x_loc) <= a) & (np.abs(mid_y_loc) <= b)

        # Segment intersects region? (from Liang–Barsky inside lengths)
        intersects = inside > 0.0

        # Long enough inside
        long_enough = inside >= threshold

        # Assign categories
        n = len(self._uwi)
        cat = np.full(n, "outside", dtype=object)
        cat[mid_inside] = "midpoint"
        cat[intersects] = "intersect"
        cat[long_enough] = f">={int(threshold)}ft"
        cat[ref_idx] = "self"

        # Working DataFrame (use PLOT coords for label positions)
        idx_arr = np.arange(n)
        df = pd.DataFrame(
            {
                "idx": idx_arr,
                "uwi": self._uwi,
                "len_in_box_ft": inside,
                "mid_x_plot": mid_x_plot,
                "mid_y_plot": mid_y_plot,
                "category": cat,
            }
        )

        # ---------- Create axes & draw region ----------

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Region polygon at origin in box-aligned frame
        rect = np.array(
            [[-a, -b], [a, -b], [a, b], [-a, b], [-a, -b]],
            dtype=float,
        )

        if orientation == "cardinal":
            # No rotation – same as cardinal box
            rect_x, rect_y = rect[:, 0], rect[:, 1]
        else:
            az_ref = float(self._az[ref_idx])
            rect_x, rect_y = self._iframe_to_world(rect[:, 0], rect[:, 1], az_ref)

        region_label = (
            f"Floating box ({int(2*a)}×{int(2*b)} ft)"
            if orientation != "corridor"
            else f"Corridor ({int(2*a)}×{int(2*b)} ft)"
        )
        ax.plot(rect_x, rect_y, lw=1.5, label=region_label)

        # Draw reference lateral (in map frame)
        ax.plot(
            [x0_plot[ref_idx], x1_plot[ref_idx]],
            [y0_plot[ref_idx], y1_plot[ref_idx]],
            color="black",
            lw=2.25,
            label=f"{uwi_ref} (ref)",
        )

        # ---------- Draw neighbors by category (using map coords) ----------

        def _cat_order(c: str) -> int:
            if c.startswith(">="):
                return 0
            if c == "intersect":
                return 1
            if c == "midpoint":
                return 2
            return 3

        df_sorted = df.sort_values(
            "category",
            key=lambda s: s.map(_cat_order),
        )

        labels_drawn = 0
        for _, r in df_sorted.iterrows():
            i = int(r["idx"])
            if i == ref_idx:
                continue
            category = r["category"]
            if category == "outside":
                continue

            x0p, y0p = x0_plot[i], y0_plot[i]
            x1p, y1p = x1_plot[i], y1_plot[i]

            if category.startswith(">="):
                ax.plot([x0p, x1p], [y0p, y1p], lw=1.5)
            elif category == "intersect":
                ax.plot([x0p, x1p], [y0p, y1p], lw=1.2, linestyle="--")
            elif category == "midpoint":
                ax.plot([x0p, x1p], [y0p, y1p], lw=1.0, linestyle=":")
            else:
                continue

            if annotate and labels_drawn < max_labels:
                mx, my = r["mid_x_plot"], r["mid_y_plot"]
                ax.text(
                    mx,
                    my,
                    f"{r['uwi']}\n{int(round(r['len_in_box_ft']))} ft",
                    fontsize=8,
                    ha="center",
                    va="center",
                    alpha=0.9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="0.4",
                        alpha=0.8,
                    ),
                )
                labels_drawn += 1

        # ---------- Counts & axes ----------
        n_ge = int(np.sum(df["category"] == f">={int(threshold)}ft"))
        n_int = int(np.sum(df["category"] == "intersect"))
        n_mid = int(np.sum(df["category"] == "midpoint"))

        ax.set_xlabel("Easting (ft, local)")
        ax.set_ylabel("Northing (ft, local)")
        ax.set_title(
            f"Floating-section view for {uwi_ref}  ({orientation})\n"
            f"≥{int(threshold)}ft: {n_ge}, "
            f"intersect: {n_int}, "
            f"midpoint: {n_mid}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        return fig

    def plot_map_with_neighbors(
        self,
        uwi_ref: str,
        *,
        orientation: Orientation = "i_frame",
        label_top_n: Optional[int] = None,
        fontsize: int = 8,
        figsize: Tuple[int, int] = (8, 7),
        show_other_box: bool = True,
    ) -> plt.Figure:
        """
        Map diagnostic for one reference well with neighbor labels.

        - Draws all laterals.
        - Draws the active floating region (box or corridor, per `orientation`).
        - Optionally also draws the *cardinal* DSU-style box for comparison.
        - Labels neighbors that pass the 660-ft rule (sorted by inside length).
        - Returns the Matplotlib Figure (caller can save/close).

        Parameters
        ----------
        uwi_ref : str
            Reference well to center the floating section on.
        orientation : {'cardinal','i_frame','corridor'}, default 'i_frame'
            Which region alignment to use for counting and labeling.
        label_top_n : int or None
            If set, annotate only the top-N neighbors by inside length.
            If None, annotate all neighbors that pass the threshold.
        fontsize : int
            Label font size.
        figsize : (w,h)
            Figure size in inches.
        show_other_box : bool
            If True, draws the *cardinal* box faintly for DSU context.
        """
        ref_idx = int(np.where(self._uwi == str(uwi_ref))[0][0])
        cx, cy = self._mx[ref_idx], self._my[ref_idx]
        az = float(self._az[ref_idx])

        # Active region geometry for this orientation
        a, b = self._get_half_sizes(ref_idx, orientation)

        # Compute inside lengths for the requested orientation
        inside = self._inside_lengths_for_ref(ref_idx, orientation=orientation)
        if self.exclude_self:
            inside[ref_idx] = 0.0
        passes = inside >= self.min_inside_ft

        # Sort by inside length (desc) and optionally cap to top-N
        order = np.argsort(-inside)
        order = [i for i in order if passes[i]]
        if label_top_n is not None:
            order = order[:int(label_top_n)]

        fig, ax = plt.subplots(figsize=figsize)

        # plot all laterals
        for i in range(len(self._uwi)):
            ax.plot([self._x0[i], self._x1[i]], [self._y0[i], self._y1[i]], lw=1, alpha=0.9)

        # helper: draw a rotated rectangle
        def _draw_rot_rect(ax, cx, cy, a, b, az_deg, **kw):
            rect = np.array([[-a, -b], [a, -b], [a, b], [-a, b], [-a, -b]], dtype=float)
            rr_x, rr_y = self._iframe_to_world(rect[:, 0], rect[:, 1], az_deg)
            ax.plot(rr_x + cx, rr_y + cy, **kw)

        # draw active region
        if orientation == "cardinal":
            ax.plot(
                [cx - a, cx + a, cx + a, cx - a, cx - a],
                [cy - b, cy - b, cy + b, cy + b, cy - b],
                lw=2,
                color="black",
                label="cardinal box",
            )
        elif orientation == "i_frame":
            _draw_rot_rect(ax, cx, cy, a, b, az, lw=2, color="black", label="i-frame box")
        else:  # corridor
            _draw_rot_rect(ax, cx, cy, a, b, az, lw=2, color="black", label="corridor")

        # Optional DSU-style cardinal box for context
        if show_other_box:
            a_c, b_c = self.box.half_width_ft, self.box.half_height_ft
            ax.plot(
                [cx - a_c, cx + a_c, cx + a_c, cx - a_c, cx - a_c],
                [cy - b_c, cy - b_c, cy + b_c, cy + b_c, cy - b_c],
                lw=1,
                color="gray",
                alpha=0.7,
                label="cardinal DSU box",
            )

        # annotate neighbors
        for i in order:
            # label at the midpoint of the segment (approximate clipped midpoint)
            xlab = 0.5 * (self._x0[i] + self._x1[i])
            ylab = 0.5 * (self._y0[i] + self._y1[i])
            ax.scatter([xlab], [ylab], s=15)
            ax.text(
                xlab,
                ylab,
                f"{self._uwi[i]}\n{inside[i]:.0f} ft",
                fontsize=fontsize,
                ha="center",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="0.4",
                    alpha=0.8,
                ),
            )

        # counts for caption
        count_pass = int(np.sum(passes))
        if self.exclude_self and passes[ref_idx]:
            count_pass -= 1

        ax.set_title(
            f"{orientation} region for {uwi_ref}  |  "
            f"neighbors ≥{int(self.min_inside_ft)} ft: {count_pass}"
        )
        ax.set_xlabel("Easting (ft)")
        ax.set_ylabel("Northing (ft)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        return fig

    def save_diagnostics_for(
        self,
        uwi_refs: Iterable[str],
        out_dir: os.PathLike,
        *,
        orientations: Tuple[Orientation, ...] = ("cardinal", "i_frame"),
        label_top_n: Optional[int] = 10,
        include_polar: bool = True,
        include_bin_chart: bool = True,
        bin_edges_deg: Iterable[float] = (0, 45, 90, 135, 180),
        export_neighbors_csv: bool = True,
        dpi: int = 150,
    ) -> dict:
        """
        Save a complete diagnostics pack to `out_dir`.

        - For each `uwi_ref`: saves map PNG(s) with neighbor labels for each orientation.
          Also writes a neighbors CSV (who was counted, inside length, threshold flag).
        - Basin-wide: optionally saves polar azimuth histogram and WPS-by-azimuth-bin chart.
        - Writes a README.md explaining artifacts and parameters used.

        Returns a dict with lists of created file paths.

        Parameters
        ----------
        uwi_refs : iterable of str
            Reference wells to render.
        out_dir : path-like
            Destination folder. Created if missing.
        orientations : tuple of Orientation
            Which region alignments to render per ref (e.g. ('cardinal','i_frame','corridor')).
        label_top_n : int or None
            Annotate up to N neighbors per map (sorted by inside length). None → annotate all.
        include_polar : bool
            Save polar azimuth histogram for the dataset.
        include_bin_chart : bool
            Save WPS mean-by-azimuth-bin bar chart.
        bin_edges_deg : iterable
            Bin edges for the azimuth comparison chart.
        export_neighbors_csv : bool
            Save per-ref neighbors tables (one CSV per orientation).
        dpi : int
            Image save DPI.
        """
        out_paths = {"maps": [], "csvs": [], "polar": None, "bins": None, "readme": None}
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Global artifacts
        if include_polar:
            fig = self.plot_polar_azimuth_hist()
            p = out_dir / "azimuth_polar.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out_paths["polar"] = str(p)

        if include_bin_chart:
            fig = self.plot_wps_by_azimuth_bins(bin_edges_deg=bin_edges_deg)
            p = out_dir / "wps_by_azimuth_bins.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out_paths["bins"] = str(p)

        # Per-reference artifacts
        for u in uwi_refs:
            for ori in orientations:
                fig = self.plot_map_with_neighbors(
                    u,
                    orientation=ori,
                    label_top_n=label_top_n,
                    show_other_box=True,
                )
                p = out_dir / f"map_{u}_{ori}.png"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                out_paths["maps"].append(str(p))

                if export_neighbors_csv:
                    tbl = self.count_for_reference(u, orientation=ori, include_lengths=True)
                    c = out_dir / f"neighbors_{u}_{ori}.csv"
                    tbl.to_csv(c, index=False)
                    out_paths["csvs"].append(str(c))

        # README.md with a compact explanation
        readme = out_dir / "README.md"
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
f"""# Floating Section Diagnostics

**Box:** {int(self.box.half_width_ft*2)} ft × {int(self.box.half_height_ft*2)} ft  
**Inside-length threshold:** {int(self.min_inside_ft)} ft  
**Exclude self:** {self.exclude_self}
**Corridor:** {self.corridor!r}

## What’s included
- Per-well maps (`map_<uwi>_<orientation>.png`) showing:
  - all laterals,
  - the active floating region (**black**) centered on the reference well
    (cardinal box, i-frame box, or corridor),
  - the cardinal DSU box (**gray**) for context,
  - labels for neighbors with inside length ≥ threshold (`<uwi>`, `<inside_len_ft> ft`).
- Neighbor tables (`neighbors_<uwi>_<orientation>.csv`) listing counted wells and inside lengths.
- Basin-wide polar azimuth histogram (`azimuth_polar.png`) for **stratification**.
- Mean WPS by azimuth bins chart (`wps_by_azimuth_bins.png`) for **normalization**
  (cardinal vs i-frame).

## Notes
- *cardinal* keeps the box north-up, east-right.
- *i-frame* aligns the box with the reference well’s azimuth (0°=N, 90°=E).
- *corridor* builds a lateral-following corridor aligned to the reference well’s azimuth,
  with length based on the well’s heel–toe distance and width set by `CorridorSpec`.
- A neighbor is counted only if its lateral length **inside** the region is ≥ the threshold.
"""
            )
        out_paths["readme"] = str(readme)

        return out_paths
    


#%% # ======================= Average Spacing Calculator ======================= #
"""
Average Spacing Calculator.

Extracted from DirectionalBenchNeighbors to provide a standalone class for
computing per-well average spacing metrics using chain, dense, or i2nbr modes.
"""
class AvgSpacingCalculator:
    """
    Summarizes per-well spacing by building neighborhoods S(i) and computing
    spacing metrics via chain, dense, or i2nbr modes.

    For each well_i, the neighborhood S(i) consists of well_i itself plus all
    eligible neighbor wells (survivors after filtering). The spacing metric
    is then computed within this neighborhood.

    ----------
    Neighborhood modes
    ----------
    - "i2nbr": Mean of eligible i→k survivor distances (simple radial average)
    - "chain": Chain adjacency inside S(i): A→B→C→D (no repeats), ordered by
               chain_sort_mode ("x" or "pca")
    - "dense": Mean of all unique undirected member-member pairs inside S(i)

    ----------
    Usage
    ----------
    >>> summarizer = AvgSpacingCalculator()
    >>> result_df = summarizer.summarize(
    ...     spacing_df,
    ...     cutoff_ft=1320.0,
    ...     neighborhood_mode="chain",
    ...     trajectories=trajectories,
    ... )
    
    # Optional: plot diagnostics for a specific well
    >>> summarizer.plot_neighborhood(
    ...     well_i="30025410040100",
    ...     trajectories=trajectories,
    ...     mode="both",
    ...     show=True,
    ... )

    ----------
    Parameters (at init)
    ----------
    overrides_df : pandas.DataFrame, optional
        Default per-well override rules. Can be overridden at call time.

    ----------
    Output columns
    ----------
    - well_i: well identifier
    - group_size: number of members in S(i) including well_i
    - neighbors_in_group: group_size - 1
    - survivor_rows_i_to_k: number of eligible i→k pairs
    - avg_hz_spacing_ft: the computed horizontal spacing metric
    - avg_vt_spacing_ft: the computed vertical spacing metric
    - neighborhood_mode_used: which mode was applied

    Plus mode-specific diagnostics (chain edges, dense pairs, percentiles, etc.)
    """

    AxisMode = Literal["any", "EW", "NS"]
    NeighborhoodMode = Literal["i2nbr", "chain", "dense"]
    ChainSortMode = Literal["x", "pca"]
    EdgePick = Literal["min", "mean", "forward"]
    PlotMode = Literal["chain", "dense", "both"]

    def __init__(self, *, overrides_df: Optional[pd.DataFrame] = None, logger: Optional[logging.Logger] = None) -> None:
        """
        Parameters
        ----------
        overrides_df : pandas.DataFrame, optional
            Default per-well override rules. Can be overridden at call time.
        """
        # Set up hierarchical logger
        parent_logger = logger or logging.getLogger("avg_Spacing_neighborhoods")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.avg_Spacing_neighborhoods")
        self.logger.propagate = True

        self.logger.info("Initializing AvgSpacingCalculator")

        self._overrides_df_default = overrides_df

        if overrides_df is not None:
            n_overrides = len(overrides_df)
            self.logger.info(f"✓ Initialized with default overrides for {n_overrides:,} wells")
        else:
            self.logger.debug("  No default overrides configured")

        # Intermediate results stored for optional debug plotting
        self._members: Optional[pd.DataFrame] = None
        self._survivors: Optional[pd.DataFrame] = None
        self._edges_chain: Optional[pd.DataFrame] = None
        self._pairs_dense: Optional[pd.DataFrame] = None
        self._ordered_members: Optional[pd.DataFrame] = None
        self._output: Optional[pd.DataFrame] = None
        self._cutoff_ft: Optional[float] = None
        self._ov_cut: Optional[pd.Series] = None
        self._all_wells: Optional[pd.Series] = None

    def _resolve_overrides(
        self, overrides_df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Return call-time overrides if provided, else fall back to init default."""
        return overrides_df if overrides_df is not None else self._overrides_df_default

    def _clear_intermediates(self) -> None:
        """Clear stored intermediate results from previous run."""
        self._members = None
        self._survivors = None
        self._edges_chain = None
        self._pairs_dense = None
        self._ordered_members = None
        self._output = None
        self._cutoff_ft = None
        self._ov_cut = None
        self._all_wells = None

    def summarize(
        self,
        spacing_df: pd.DataFrame,
        *,
        cutoff_ft: float,
        vertical_cutoff_ft: Optional[float] = None,
        overlap_pct_k_min: Optional[float] = None,
        overrides_df: Optional[pd.DataFrame] = None,
        proj_coverage_min: Optional[float] = None,
        axis_mode: AxisMode = "any",
        neighborhood_mode: NeighborhoodMode = "chain",
        chain_sort_mode: ChainSortMode = "pca",
        trajectories: Optional[pd.DataFrame] = None,
        edge_pick: EdgePick = "min",
        include_unweighted: bool = False,
    ) -> pd.DataFrame:
        """
        Summarize per-well spacing from a pairwise spacing table by building a
        neighborhood S(i) for each well_i and computing a neighborhood spacing metric.

        Parameters
        ----------
        spacing_df : pandas.DataFrame
            Pairwise spacing table with required columns:
              - 'well_i', 'well_k': str/int (well identifiers)
              - 'horizontal_dist': float (feet)
              - 'vertical_dist': float (feet)
              - 'pair_alignment': str ('parallel_like' or other)
              - 'direction_to_k_from_i_axis': {'E','W','N','S'}
              - 'LL_i': float (lateral length of well_i)

        cutoff_ft : float
            Global horizontal cutoff (ft).

        vertical_cutoff_ft : float, optional
            Global vertical cutoff (ft). If None, no vertical rule applied globally.

        overlap_pct_k_min : float, optional
            Global overlap/adjacency minimum. If None, no overlap rule applied globally.

        overrides_df : pandas.DataFrame, optional
            Per-well rule overrides. Columns: 'well_i' (or 'uwi'), plus optional
            'cutoff_ft', 'vertical_cutoff_ft', 'overlap_pct_k_min'.

        proj_coverage_min : float, optional
            Coverage quality gate for oblique/perpendicular pairs only.

        axis_mode : {'any', 'EW', 'NS'}, default 'any'
            Direction eligibility filter for survivors.

        neighborhood_mode : {'i2nbr', 'chain', 'dense'}, default 'chain'
            How to compute the neighborhood spacing metric.

        chain_sort_mode : {'x', 'pca'}, default 'pca'
            For chain mode, how to order members along the chain.

        trajectories : pandas.DataFrame, optional
            Required for chain mode. Must have columns ['uwi', 'x', 'y'].

        edge_pick : {'min', 'mean', 'forward'}, default 'min'
            For chain mode, how to resolve directional edge distances.

        include_unweighted : bool, default False
            If True, include hz_mean_i2nbr_ft and vt_mean_i2nbr_ft columns.

        Returns
        -------
        pandas.DataFrame
            One row per well_i with spacing metrics and diagnostics.

        Raises
        ------
        ValueError
            If required columns are missing or inputs are inconsistent.

        Notes
        -----
        After calling this method, you can use `plot_neighborhood()` to generate
        diagnostic plots for specific wells.

        If proj_coverage_min is provided but spacing_df does not contain
        proj_coverage_i_pct, this method will derive a proxy from overlap_pct_i
        or overlap_len_common_ft / LL_i if available.
        """
        # ═══════════════════════════════════════════════════════════════════════
        self.logger.info("=" * 80)
        self.logger.info("AVERAGE SPACING CALCULATOR - NEIGHBORHOOD ANALYSIS")
        self.logger.info("=" * 80)

        pipeline_start = time.time()

        # Clear any previous run's intermediates
        self._clear_intermediates()

        overrides_df = self._resolve_overrides(overrides_df)

        spacing = spacing_df.copy()
        spacing["well_i"] = spacing["well_i"].astype(str)
        spacing["well_k"] = spacing["well_k"].astype(str)

        # Validate required columns
        req_base = {
            "well_i",
            "well_k",
            "horizontal_dist",
            "vertical_dist",
            "pair_alignment",
            "direction_to_k_from_i_axis",
            "LL_i",
        }
        missing = req_base - set(spacing.columns)
        if missing:
            raise ValueError(
                f"summarize: spacing_df missing required columns: {sorted(missing)}"
            )

        all_wells = (
            spacing["well_i"].drop_duplicates().reset_index(drop=True).astype(str)
        )
        self._all_wells = all_wells
        self._cutoff_ft = cutoff_ft

        n_input_pairs = len(spacing)
        n_input_wells = len(all_wells)

        self.logger.info(f"Input: {n_input_pairs:,} spacing pairs across {n_input_wells:,} wells")
        self.logger.debug(f"Configuration:")
        self.logger.debug(f"  Neighborhood mode: {neighborhood_mode}")
        if neighborhood_mode == "chain":
            self.logger.debug(f"  Chain sort mode: {chain_sort_mode}")
            self.logger.debug(f"  Edge pick: {edge_pick}")
        self.logger.debug(f"  Horizontal cutoff: {cutoff_ft} ft")
        self.logger.debug(f"  Vertical cutoff: {vertical_cutoff_ft} ft" if vertical_cutoff_ft else "  Vertical cutoff: None")
        self.logger.debug(f"  Overlap min: {overlap_pct_k_min:.1%}" if overlap_pct_k_min else "  Overlap min: None")
        self.logger.debug(f"  Axis mode: {axis_mode}")
        if proj_coverage_min:
            self.logger.debug(f"  Coverage quality gate: {proj_coverage_min:.1%}")

        if overrides_df is not None and not overrides_df.empty:
            n_overrides = len(overrides_df)
            self.logger.info(f"Applying per-well overrides for {n_overrides:,} wells")

        # ---------------- Overrides (per well_i) ----------------
        ov_cut = ov_vcut = ov_omin = None
        if overrides_df is not None and not overrides_df.empty:
            ov = overrides_df.copy()
            if "well_i" not in ov.columns and "uwi" in ov.columns:
                ov = ov.rename(columns={"uwi": "well_i"})
            if "well_i" in ov.columns:
                ov["well_i"] = ov["well_i"].astype(str)
                ov = ov.set_index("well_i")
                ov_cut = ov["cutoff_ft"] if "cutoff_ft" in ov.columns else None
                ov_vcut = (
                    ov["vertical_cutoff_ft"]
                    if "vertical_cutoff_ft" in ov.columns
                    else None
                )
                ov_omin = (
                    ov["overlap_pct_k_min"]
                    if "overlap_pct_k_min" in ov.columns
                    else None
                )

        self._ov_cut = ov_cut

        # Effective horizontal cutoff per row
        if ov_cut is not None:
            eff_hcut = (
                spacing["well_i"].map(ov_cut).astype(float).fillna(float(cutoff_ft))
            )
        else:
            eff_hcut = pd.Series(float(cutoff_ft), index=spacing.index)

        # Effective vertical cutoff per row (NaN = no rule)
        if (vertical_cutoff_ft is not None) or (ov_vcut is not None):
            eff_vcut = (
                spacing["well_i"].map(ov_vcut).astype(float)
                if ov_vcut is not None
                else pd.Series(np.nan, index=spacing.index)
            )
            if vertical_cutoff_ft is not None:
                eff_vcut = eff_vcut.fillna(float(vertical_cutoff_ft))
        else:
            eff_vcut = pd.Series(np.nan, index=spacing.index)

        # Effective overlap/adjacency min per row (NaN = no rule)
        if (overlap_pct_k_min is not None) or (ov_omin is not None):
            eff_omin = (
                spacing["well_i"].map(ov_omin).astype(float)
                if ov_omin is not None
                else pd.Series(np.nan, index=spacing.index)
            )
            if overlap_pct_k_min is not None:
                eff_omin = eff_omin.fillna(float(overlap_pct_k_min))
        else:
            eff_omin = pd.Series(np.nan, index=spacing.index)

        # ---------------- Adaptive adjacency percentage ----------------
        contact_interior = (
            spacing["contact_pct_i_interior"].astype(float)
            if "contact_pct_i_interior" in spacing.columns
            else pd.Series(np.nan, index=spacing.index)
        )
        contact_any = (
            spacing["contact_pct_i"].astype(float)
            if "contact_pct_i" in spacing.columns
            else pd.Series(np.nan, index=spacing.index)
        )
        adj_oblique_perp = contact_interior.fillna(contact_any)

        overlap_pct_k = (
            spacing["overlap_pct_k"].astype(float)
            if "overlap_pct_k" in spacing.columns
            else pd.Series(np.nan, index=spacing.index)
        )

        spacing["adj_pct"] = np.where(
            spacing["pair_alignment"].eq("parallel_like"),
            overlap_pct_k,
            adj_oblique_perp,
        ).astype(float)

        # ---------------- Eligibility mask (survivors) ----------------
        mask = spacing["horizontal_dist"].astype(float) <= eff_hcut

        if eff_vcut.notna().any():
            has_vrule = eff_vcut.notna()
            mask &= (~has_vrule) | (spacing["vertical_dist"].astype(float) <= eff_vcut)

        if eff_omin.notna().any():
            has_orule = eff_omin.notna()
            mask &= (~has_orule) | (spacing["adj_pct"] >= eff_omin)

        # --- proj_coverage_i_pct fallback ---
        if proj_coverage_min is not None:
            if "proj_coverage_i_pct" not in spacing.columns:
                # Prefer overlap_pct_i if present
                if "overlap_pct_i" in spacing.columns:
                    spacing["proj_coverage_i_pct"] = spacing["overlap_pct_i"].astype(
                        float
                    )
                # Else approximate from overlap length / LL_i if possible
                elif ("overlap_len_common_ft" in spacing.columns) and (
                    "LL_i" in spacing.columns
                ):
                    denom = spacing["LL_i"].astype(float).replace(0.0, np.nan)
                    spacing["proj_coverage_i_pct"] = (
                        spacing["overlap_len_common_ft"].astype(float) / denom
                    ).replace([np.inf, -np.inf], np.nan)
                else:
                    # No info to compute; keep as NaN (non-parallel rows will fail)
                    spacing["proj_coverage_i_pct"] = np.nan

            ok_cov = np.where(
                spacing["pair_alignment"].eq("parallel_like"),
                True,
                spacing["proj_coverage_i_pct"].astype(float) >= float(proj_coverage_min),
            )
            mask &= ok_cov

        if axis_mode == "EW":
            mask &= spacing["direction_to_k_from_i_axis"].isin({"E", "W"})
        elif axis_mode == "NS":
            mask &= spacing["direction_to_k_from_i_axis"].isin({"N", "S"})

        survivors = spacing.loc[mask].copy()
        self._survivors = survivors

        n_survivors = len(survivors)
        n_filtered_out = n_input_pairs - n_survivors
        survival_rate = (n_survivors / n_input_pairs * 100) if n_input_pairs > 0 else 0

        self.logger.info("─" * 80)
        self.logger.info(f"Filtering complete: {n_survivors:,} survivor pairs ({survival_rate:.1f}%)")
        if n_filtered_out > 0:
            self.logger.debug(f"  Filtered out: {n_filtered_out:,} pairs ({100-survival_rate:.1f}%)")

        # ---------------- Baseline i->eligible k summaries ----------------
        g_i = survivors.groupby("well_i", sort=False)
        survivor_rows = (
            g_i.size().rename("survivor_rows_i_to_k").rename_axis("group_i")
        )

        hz_mean_i2nbr = (
            g_i["horizontal_dist"].mean().rename("hz_mean_i2nbr_ft").rename_axis("group_i")
        )
        vt_mean_i2nbr = (
            g_i["vertical_dist"].mean().rename("vt_mean_i2nbr_ft").rename_axis("group_i")
        )

        # ---------------- Build membership table for S(i) ----------------
        self.logger.info("─" * 80)
        self.logger.info("Building neighborhood memberships S(i)")
        members_self = pd.DataFrame(
            {"group_i": all_wells.values, "member": all_wells.values}
        )
        members_nbrs = survivors[["well_i", "well_k"]].rename(
            columns={"well_i": "group_i", "well_k": "member"}
        )
        members = pd.concat([members_self, members_nbrs], ignore_index=True).drop_duplicates()
        self._members = members

        group_size = (
            members.groupby("group_i", sort=False)["member"]
            .nunique()
            .rename("group_size")
            .rename_axis("group_i")
        )
        neighbors_in_group = (
            (group_size - 1).clip(lower=0).rename("neighbors_in_group").rename_axis("group_i")
        )

        avg_group_size = group_size.mean()
        max_group_size = group_size.max()
        self.logger.info(f"✓ Neighborhoods built: avg size={avg_group_size:.1f}, max size={max_group_size:.0f}")

        want_chain = neighborhood_mode == "chain"
        want_dense = neighborhood_mode == "dense"

        self.logger.info("─" * 80)
        self.logger.info(f"Computing spacing metrics using '{neighborhood_mode}' mode")

        # -----------------------------
        # CHAIN MODE (A->B->C->D)
        # -----------------------------
        chain_axis = None
        chain_edges_used = chain_edges_missing = None
        hz_mean_chain = vt_mean_chain = None
        chain_p50 = chain_p25 = chain_p75 = chain_min = chain_max = None

        if want_chain:
            if trajectories is None:
                raise ValueError(
                    "summarize: chain mode requires `trajectories` "
                    "with columns ['uwi','x','y']."
                )

            need_cols = {"uwi", "x", "y"}
            miss_t = need_cols - set(trajectories.columns)
            if miss_t:
                raise ValueError(
                    f"summarize: trajectories missing columns: {sorted(miss_t)}"
                )

            traj = trajectories.copy()
            traj["uwi"] = traj["uwi"].astype(str)

            centroids = (
                traj.groupby("uwi", sort=False)[["x", "y"]]
                .median()
                .rename(columns={"x": "x_mid", "y": "y_mid"})
            )

            mem_xy = members.merge(
                centroids.reset_index().rename(columns={"uwi": "member"}),
                on="member",
                how="left",
            )

            if mem_xy[["x_mid", "y_mid"]].isna().any(axis=1).any():
                bad = (
                    mem_xy.loc[
                        mem_xy[["x_mid", "y_mid"]].isna().any(axis=1), "member"
                    ]
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                )
                raise ValueError(
                    "summarize: trajectories missing x/y for some neighborhood members. "
                    f"Examples: {bad}"
                )

            mem_xy["chain_sort_mode"] = chain_sort_mode

            def _order_members_one_group(g: pd.DataFrame) -> pd.DataFrame:
                xy = g[["x_mid", "y_mid"]].to_numpy(dtype=float)

                if g.shape[0] <= 1:
                    g = g.copy()
                    g["chain_axis_dx"] = 1.0
                    g["chain_axis_dy"] = 0.0
                    g["_score"] = 0.0
                    return g

                if chain_sort_mode == "x":
                    axis = np.array([1.0, 0.0], dtype=float)
                    score = xy[:, 0].copy()
                else:
                    C = xy - xy.mean(axis=0, keepdims=True)
                    if np.allclose(C, 0.0):
                        axis = np.array([1.0, 0.0], dtype=float)
                        score = C @ axis
                    else:
                        _, _, Vt = np.linalg.svd(C, full_matrices=False)
                        axis = Vt[0]

                        # Deterministic axis sign
                        if abs(axis[0]) >= abs(axis[1]):
                            if axis[0] < 0:
                                axis = -axis
                        else:
                            if axis[1] < 0:
                                axis = -axis

                        score = (xy - xy.mean(axis=0, keepdims=True)) @ axis

                g = g.copy()
                g["chain_axis_dx"] = float(axis[0])
                g["chain_axis_dy"] = float(axis[1])
                g["_score"] = score
                return g.sort_values(["_score", "member"], ascending=[True, True]).drop(
                    columns=["_score"]
                )

            ordered_members = (
                mem_xy.groupby("group_i", sort=False, group_keys=False)
                .apply(_order_members_one_group)
                .reset_index(drop=True)
            )
            self._ordered_members = ordered_members

            chain_edges = ordered_members[["group_i", "member"]].copy()
            chain_edges["next_member"] = ordered_members.groupby("group_i", sort=False)[
                "member"
            ].shift(-1)
            chain_edges = chain_edges.dropna(subset=["next_member"]).rename(
                columns={"member": "u", "next_member": "v"}
            )
            chain_edges["u"] = chain_edges["u"].astype(str)
            chain_edges["v"] = chain_edges["v"].astype(str)

            chain_axis = (
                ordered_members.groupby("group_i", sort=False)
                .agg(
                    chain_sort_mode=("chain_sort_mode", "first"),
                    chain_axis_dx=("chain_axis_dx", "first"),
                    chain_axis_dy=("chain_axis_dy", "first"),
                )
                .rename_axis("group_i")
            )

            lookup = spacing[
                ["well_i", "well_k", "horizontal_dist", "vertical_dist"]
            ].copy()
            lookup["well_i"] = lookup["well_i"].astype(str)
            lookup["well_k"] = lookup["well_k"].astype(str)

            uv = lookup.rename(
                columns={
                    "well_i": "u",
                    "well_k": "v",
                    "horizontal_dist": "hz_uv",
                    "vertical_dist": "vt_uv",
                }
            )
            vu = lookup.rename(
                columns={
                    "well_i": "v",
                    "well_k": "u",
                    "horizontal_dist": "hz_vu",
                    "vertical_dist": "vt_vu",
                }
            )

            edges_chain = chain_edges.merge(uv, on=["u", "v"], how="left").merge(
                vu, on=["u", "v"], how="left"
            )

            hz_uv = edges_chain["hz_uv"].astype(float)
            hz_vu = edges_chain["hz_vu"].astype(float)
            vt_uv = edges_chain["vt_uv"].astype(float)
            vt_vu = edges_chain["vt_vu"].astype(float)

            if edge_pick == "forward":
                edges_chain["hz_edge_ft"] = hz_uv
                edges_chain["vt_edge_ft"] = vt_uv
            elif edge_pick == "mean":
                edges_chain["hz_edge_ft"] = np.where(
                    hz_uv.notna() & hz_vu.notna(),
                    0.5 * (hz_uv + hz_vu),
                    hz_uv.fillna(hz_vu),
                )
                edges_chain["vt_edge_ft"] = np.where(
                    vt_uv.notna() & vt_vu.notna(),
                    0.5 * (vt_uv + vt_vu),
                    vt_uv.fillna(vt_vu),
                )
            else:  # "min"
                edges_chain["hz_edge_ft"] = np.where(
                    hz_uv.notna() & hz_vu.notna(),
                    np.minimum(hz_uv, hz_vu),
                    hz_uv.fillna(hz_vu),
                )
                edges_chain["vt_edge_ft"] = np.where(
                    vt_uv.notna() & vt_vu.notna(),
                    np.minimum(vt_uv, vt_vu),
                    vt_uv.fillna(vt_vu),
                )

            self._edges_chain = edges_chain

            ge_chain = edges_chain.groupby("group_i", sort=False)

            chain_edges_used = (
                ge_chain["hz_edge_ft"]
                .count()
                .rename("chain_edges_used")
                .rename_axis("group_i")
            )
            hz_mean_chain = (
                ge_chain["hz_edge_ft"]
                .mean()
                .rename("hz_mean_chain_ft")
                .rename_axis("group_i")
            )
            vt_mean_chain = (
                ge_chain["vt_edge_ft"]
                .mean()
                .rename("vt_mean_chain_ft")
                .rename_axis("group_i")
            )

            chain_p50 = (
                ge_chain["hz_edge_ft"]
                .median()
                .rename("p50_hz_chain_edge_ft")
                .rename_axis("group_i")
            )
            chain_p25 = (
                ge_chain["hz_edge_ft"]
                .quantile(0.25)
                .rename("p25_hz_chain_edge_ft")
                .rename_axis("group_i")
            )
            chain_p75 = (
                ge_chain["hz_edge_ft"]
                .quantile(0.75)
                .rename("p75_hz_chain_edge_ft")
                .rename_axis("group_i")
            )
            chain_min = (
                ge_chain["hz_edge_ft"]
                .min()
                .rename("min_hz_chain_edge_ft")
                .rename_axis("group_i")
            )
            chain_max = (
                ge_chain["hz_edge_ft"]
                .max()
                .rename("max_hz_chain_edge_ft")
                .rename_axis("group_i")
            )

            chain_edges_missing = (
                ((group_size - 1) - chain_edges_used)
                .rename("chain_edges_missing")
                .rename_axis("group_i")
            )

        # -----------------------------
        # DENSE MODE (all member-member pairs)
        # -----------------------------
        cand_cnt = expected_dir_pairs = candidate_pair_density = None
        pairs_used_dense = hz_mean_dense = vt_mean_dense = None
        dense_p50 = dense_p25 = dense_p75 = dense_min = dense_max = None

        if want_dense:
            base_cols = [
                "well_i",
                "well_k",
                "horizontal_dist",
                "vertical_dist",
                "direction_to_k_from_i_axis",
            ]
            spacing_base = spacing.loc[
                :, [c for c in base_cols if c in spacing.columns]
            ].copy()

            cand = spacing_base.merge(
                members.rename(columns={"member": "well_i"})[["group_i", "well_i"]],
                on="well_i",
                how="inner",
            )
            cand = cand.merge(
                members.rename(columns={"member": "well_k"})[["group_i", "well_k"]],
                on=["group_i", "well_k"],
                how="inner",
            )

            cand = cand.loc[cand["well_i"] != cand["well_k"]].copy()

            if axis_mode == "EW":
                cand = cand.loc[cand["direction_to_k_from_i_axis"].isin(["E", "W"])]
            elif axis_mode == "NS":
                cand = cand.loc[cand["direction_to_k_from_i_axis"].isin(["N", "S"])]

            cand_cnt = (
                cand.groupby("group_i", sort=False)
                .size()
                .rename("member_member_rows_found")
                .rename_axis("group_i")
            )
            expected_dir_pairs = (
                (group_size * (group_size - 1))
                .rename("expected_directed_pairs")
                .rename_axis("group_i")
            )
            candidate_pair_density = (
                (cand_cnt / expected_dir_pairs.replace(0, np.nan))
                .rename("candidate_pair_density")
                .rename_axis("group_i")
            )

            cand2 = cand.rename(
                columns={
                    "well_i": "src",
                    "well_k": "dst",
                    "direction_to_k_from_i_axis": "direction",
                }
            )

            uu = np.minimum(cand2["src"].astype(str), cand2["dst"].astype(str))
            vv = np.maximum(cand2["src"].astype(str), cand2["dst"].astype(str))

            pairs_dense = cand2.assign(pair_u=uu, pair_v=vv).drop_duplicates(
                subset=["group_i", "pair_u", "pair_v"], keep="first"
            )
            self._pairs_dense = pairs_dense

            gp = pairs_dense.groupby("group_i", sort=False)
            pairs_used_dense = (
                gp.size().rename("pairs_used_dense").rename_axis("group_i")
            )
            hz_mean_dense = (
                gp["horizontal_dist"]
                .mean()
                .rename("hz_mean_dense_ft")
                .rename_axis("group_i")
            )
            vt_mean_dense = (
                gp["vertical_dist"]
                .mean()
                .rename("vt_mean_dense_ft")
                .rename_axis("group_i")
            )

            dense_p50 = (
                gp["horizontal_dist"]
                .median()
                .rename("p50_hz_dense_pair_ft")
                .rename_axis("group_i")
            )
            dense_p25 = (
                gp["horizontal_dist"]
                .quantile(0.25)
                .rename("p25_hz_dense_pair_ft")
                .rename_axis("group_i")
            )
            dense_p75 = (
                gp["horizontal_dist"]
                .quantile(0.75)
                .rename("p75_hz_dense_pair_ft")
                .rename_axis("group_i")
            )
            dense_min = (
                gp["horizontal_dist"]
                .min()
                .rename("min_hz_dense_pair_ft")
                .rename_axis("group_i")
            )
            dense_max = (
                gp["horizontal_dist"]
                .max()
                .rename("max_hz_dense_pair_ft")
                .rename_axis("group_i")
            )

        # ---------------- Merge into output (one row per well_i) ----------------
        out = pd.DataFrame({"group_i": all_wells.astype(str)})

        merges = [group_size, neighbors_in_group, survivor_rows]

        if include_unweighted:
            merges += [hz_mean_i2nbr, vt_mean_i2nbr]

        if want_chain:
            merges += [
                chain_axis,
                chain_edges_used,
                chain_edges_missing,
                hz_mean_chain,
                vt_mean_chain,
                chain_p50,
                chain_p25,
                chain_p75,
                chain_min,
                chain_max,
            ]

        if want_dense:
            merges += [
                cand_cnt,
                expected_dir_pairs,
                candidate_pair_density,
                pairs_used_dense,
                hz_mean_dense,
                vt_mean_dense,
                dense_p50,
                dense_p25,
                dense_p75,
                dense_min,
                dense_max,
            ]

        for s in merges:
            if s is None:
                continue
            if getattr(s, "index", None) is not None and s.index.name != "group_i":
                s = s.rename_axis("group_i")
            out = out.merge(s.reset_index(), on="group_i", how="left")

        out = out.rename(columns={"group_i": "well_i"})

        out["group_size"] = out["group_size"].fillna(1).astype(int)
        out["neighbors_in_group"] = out["neighbors_in_group"].fillna(0).astype(int)
        out["survivor_rows_i_to_k"] = out["survivor_rows_i_to_k"].fillna(0).astype(int)

        if want_chain:
            out["chain_edges_used"] = out["chain_edges_used"].fillna(0).astype(int)
            out["chain_edges_missing"] = out["chain_edges_missing"].fillna(
                out["neighbors_in_group"]
            ).astype(int)

        if want_dense:
            out["member_member_rows_found"] = out["member_member_rows_found"].fillna(0).astype(int)
            out["pairs_used_dense"] = out["pairs_used_dense"].fillna(0).astype(int)

        if neighborhood_mode == "i2nbr":
            out["avg_hz_spacing_ft"] = out.get("hz_mean_i2nbr_ft")
            out["avg_vt_spacing_ft"] = out.get("vt_mean_i2nbr_ft")
            out["neighborhood_mode_used"] = "i2nbr"
        elif neighborhood_mode == "chain":
            out["avg_hz_spacing_ft"] = out.get("hz_mean_chain_ft")
            out["avg_vt_spacing_ft"] = out.get("vt_mean_chain_ft")
            out["neighborhood_mode_used"] = "chain"
        else:
            out["avg_hz_spacing_ft"] = out.get("hz_mean_dense_ft")
            out["avg_vt_spacing_ft"] = out.get("vt_mean_dense_ft")
            out["neighborhood_mode_used"] = "dense"

        self._output = out

        # Final summary
        pipeline_elapsed = time.time() - pipeline_start

        n_wells_analyzed = len(out)
        avg_hz = out["avg_hz_spacing_ft"].mean()
        avg_vt = out["avg_vt_spacing_ft"].mean()

        self.logger.info("─" * 80)
        self.logger.info(f"✓ Analysis complete: {n_wells_analyzed:,} wells")
        self.logger.info(f"  Average horizontal spacing: {avg_hz:.1f} ft")
        self.logger.info(f"  Average vertical spacing: {avg_vt:.1f} ft")
        self.logger.debug(f"  Mode used: {neighborhood_mode}")

        self.logger.info("=" * 80)
        self.logger.info(f"PIPELINE COMPLETE: {pipeline_elapsed:.2f}s")
        self.logger.info("=" * 80)

        return out

    # =========================================================================
    # Debug / Diagnostic Plotting (separate from summarize)
    # =========================================================================

    def plot_neighborhood(
        self,
        well_i: str,
        trajectories: pd.DataFrame,
        *,
        mode: PlotMode = "both",
        show: bool = True,
        save_path: Optional[str] = None,
        max_members: int = 30,
        figsize: Tuple[float, float] = (12.5, 8),
        dpi: int = 160,
    ) -> None:
        """
        Generate a diagnostic plot for a specific well's neighborhood.

        Must be called after `summarize()` has been run.

        Parameters
        ----------
        well_i : str
            The well identifier to plot the neighborhood for.

        trajectories : pandas.DataFrame
            Well trajectories with columns ['uwi', 'x', 'y'] (and optionally 'md').

        mode : {'chain', 'dense', 'both'}, default 'both'
            What to overlay on the plot:
            - 'chain': Show chain edges (A→B→C→D) with distance annotations
            - 'dense': Show all member-member pairs as light lines
            - 'both': Show both overlays

        show : bool, default True
            If True, display the plot interactively (plt.show()).

        save_path : str, optional
            If provided, save the plot to this file path.

        max_members : int, default 30
            Maximum number of neighborhood members to include in the plot.

        figsize : tuple, default (12.5, 8)
            Figure size in inches (width, height).

        dpi : int, default 160
            Resolution for saved figure.

        Raises
        ------
        RuntimeError
            If called before `summarize()` has been run.

        ValueError
            If trajectories is missing required columns or well_i data.

        Examples
        --------
        >>> summarizer = NeighborhoodSpacingSummarizer()
        >>> result = summarizer.summarize(spacing_df, cutoff_ft=1320, ...)
        >>> summarizer.plot_neighborhood(
        ...     well_i="30025410040100",
        ...     trajectories=trajectories,
        ...     mode="chain",
        ...     show=True,
        ...     save_path="./debug/well_30025410040100.png"
        ... )
        """
        # Check that summarize() has been called
        if self._members is None or self._output is None:
            raise RuntimeError(
                "plot_neighborhood() must be called after summarize(). "
                "No intermediate data found."
            )

        # Validate trajectories
        if trajectories is None:
            raise ValueError("trajectories DataFrame is required for plotting.")

        need_cols = {"uwi", "x", "y"}
        miss_t = need_cols - set(trajectories.columns)
        if miss_t:
            raise ValueError(f"trajectories missing columns: {sorted(miss_t)}")


        # Helper functions
        def _traj_lookup(uwi: str) -> Optional[pd.DataFrame]:
            df = trajectories.loc[trajectories["uwi"].astype(str) == str(uwi)].copy()
            return None if df.empty else df

        def _midpoint_xy(df_traj: pd.DataFrame) -> Tuple[float, float]:
            d = df_traj.sort_values("md") if "md" in df_traj.columns else df_traj
            x0, y0 = float(d["x"].iloc[0]), float(d["y"].iloc[0])
            x1, y1 = float(d["x"].iloc[-1]), float(d["y"].iloc[-1])
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0

        g = str(well_i)

        # Get neighborhood members
        mem = (
            self._members.loc[self._members["group_i"].astype(str) == g, "member"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if not mem:
            print(f"No neighborhood members found for well_i={g}")
            return
        if len(mem) > int(max_members):
            mem = mem[: int(max_members)]

        # Get survivors
        surv_k: List[str] = []
        if self._survivors is not None:
            surv_k = (
                self._survivors.loc[self._survivors["well_i"].astype(str) == g, "well_k"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            surv_k = [x for x in surv_k if x in set(mem)]

        # Load trajectories + midpoints
        mids: Dict[str, Tuple[float, float]] = {}
        trajs: Dict[str, pd.DataFrame] = {}
        for u in mem:
            dfu = _traj_lookup(u)
            if dfu is None:
                continue
            dfu = dfu.sort_values("md") if "md" in dfu.columns else dfu
            trajs[u] = dfu
            mids[u] = _midpoint_xy(dfu)

        if g not in trajs:
            print(f"No trajectory data found for well_i={g}")
            return

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Plot each well trajectory
        line_by_uwi: Dict[str, any] = {}
        for u, dfu in trajs.items():
            x = dfu["x"].to_numpy(float)
            y = dfu["y"].to_numpy(float)
            lw = 2.7 if u == g else (2.0 if u in surv_k else 1.2)
            (ln,) = ax.plot(x, y, linewidth=lw)
            line_by_uwi[u] = ln

        # Midpoints
        for u, (mx, my) in mids.items():
            ax.scatter([mx], [my], s=35)

        # Convex hull
        pts = np.array(list(mids.values()), dtype=float)
        if pts.shape[0] >= 3:
            try:

                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                poly = np.vstack([poly, poly[0]])
                ax.plot(poly[:, 0], poly[:, 1], linestyle="--", linewidth=1.5)
            except Exception:
                pass

        # Cutoff circle around well_i midpoint
        if g in mids and self._cutoff_ft is not None:
            mx, my = mids[g]
            hcut_i = float(self._cutoff_ft)
            if self._ov_cut is not None and g in self._ov_cut.index:
                v = self._ov_cut.loc[g]
                if pd.notna(v):
                    hcut_i = float(v)
            ax.add_patch(
                Circle((mx, my), radius=hcut_i, fill=False, linestyle=":", linewidth=1.5)
            )

        # Chain edges overlay
        if mode in ("chain", "both") and self._edges_chain is not None:
            e = self._edges_chain.loc[self._edges_chain["group_i"].astype(str) == g].copy()
            for _, r in e.iterrows():
                u = str(r["u"])
                v = str(r["v"])
                hz = r.get("hz_edge_ft", np.nan)
                if (u in mids) and (v in mids):
                    x1, y1 = mids[u]
                    x2, y2 = mids[v]
                    ax.plot([x1, x2], [y1, y2], linewidth=2.2, linestyle="--")
                    xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if pd.notna(hz):
                        ax.text(xm, ym, f"{float(hz):.0f} ft", fontsize=8)

        # Dense pairs overlay
        if mode in ("dense", "both") and self._pairs_dense is not None:
            p = self._pairs_dense.loc[self._pairs_dense["group_i"].astype(str) == g].copy()
            if len(p) > 160:
                p = p.nsmallest(160, "horizontal_dist")
            for _, r in p.iterrows():
                u = str(r["pair_u"])
                v = str(r["pair_v"])
                if (u in mids) and (v in mids):
                    x1, y1 = mids[u]
                    x2, y2 = mids[v]
                    ax.plot([x1, x2], [y1, y2], linewidth=0.9, alpha=0.22)

        # Title from output
        if self._output is not None:
            row = self._output.loc[self._output["well_i"].astype(str) == g]
            if not row.empty:
                r0 = row.iloc[0].to_dict()
                title = (
                    f"Neighborhood S(i) diagnostics for well_i={g}\n"
                    f"group_size={r0.get('group_size')}, survivors(i->k)={r0.get('survivor_rows_i_to_k')}"
                )
                if "member_member_rows_found" in r0 and pd.notna(r0.get("member_member_rows_found")):
                    title += (
                        f", member_member_rows_found={int(r0.get('member_member_rows_found'))}, "
                        f"candidate_pair_density={r0.get('candidate_pair_density'):.2f}"
                        if pd.notna(r0.get('candidate_pair_density')) else ""
                    )
                ax.set_title(title)

        ax.set_xlabel("UTM Easting (ft)")
        ax.set_ylabel("UTM Northing (ft)")
        ax.axis("equal")

        # Side legend
        legend_order = [g] + [u for u in mem if u != g and u in line_by_uwi]
        handles = [line_by_uwi[u] for u in legend_order if u in line_by_uwi]

        labels = []
        surv_k_set: Set[str] = set(surv_k)
        for u in legend_order:
            if u not in line_by_uwi:
                continue
            tag = "i" if u == g else ("k*" if u in surv_k_set else "k")
            labels.append(f"{tag}: {u}")

        fig.subplots_adjust(right=0.72)
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            title="Legend (color → well)",
            title_fontsize=9,
        )

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        
        plt.close(fig)

    def plot_multiple_neighborhoods(
        self,
        well_ids: List[str],
        trajectories: pd.DataFrame,
        *,
        save_dir: str,
        mode: PlotMode = "both",
        max_members: int = 30,
        figsize: Tuple[float, float] = (12.5, 8),
        dpi: int = 160,
    ) -> None:
        """
        Generate diagnostic plots for multiple wells, saving each to a file.

        Convenience method that calls `plot_neighborhood()` for each well.

        Parameters
        ----------
        well_ids : list of str
            List of well identifiers to plot.

        trajectories : pandas.DataFrame
            Well trajectories with columns ['uwi', 'x', 'y'].

        save_dir : str
            Directory to save plot files (created if doesn't exist).

        mode : {'chain', 'dense', 'both'}, default 'both'
            What to overlay on the plots.

        max_members : int, default 30
            Maximum number of neighborhood members per plot.

        figsize : tuple, default (12.5, 8)
            Figure size in inches.

        dpi : int, default 160
            Resolution for saved figures.

        Examples
        --------
        >>> summarizer.plot_multiple_neighborhoods(
        ...     well_ids=["well_001", "well_002", "well_003"],
        ...     trajectories=trajectories,
        ...     save_dir="./debug_plots/",
        ...     mode="chain"
        ... )
        """
        os.makedirs(save_dir, exist_ok=True)

        for well_id in well_ids:
            save_path = os.path.join(save_dir, f"neighborhood_{well_id}.png")
            try:
                self.plot_neighborhood(
                    well_i=well_id,
                    trajectories=trajectories,
                    mode=mode,
                    show=False,
                    save_path=save_path,
                    max_members=max_members,
                    figsize=figsize,
                    dpi=dpi,
                )
            except Exception as e:
                print(f"Warning: Failed to plot well_i={well_id}: {e}")