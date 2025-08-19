#%%
from __future__ import annotations # Enabling future annotations for type hinting

import os # Importing os module for operating system dependent functionality

import pandas as pd # Importing pandas package

# Set the maximum number of columns to display to None
pd.set_option('display.max_columns', None)

import numpy as np # Importing numpy package

from typing import Dict, Tuple, List, Union, Optional, ClassVar, Any, Literal, Iterable # Importing specific types from typing module

from tqdm import tqdm # Importing tqdm for progress bar functionality

from joblib import Parallel, delayed # Importing Parallel and delayed for parallel processing

from matplotlib import pyplot as plt # Importing pyplot from matplotlib for plotting

from pyproj import Geod # Importing Geod class from pyproj for geodetic calculations

#%%
class WellSpacingCalculator:
    """
    Class for calculating well spacing metrics and directional relationships using
    3D lateral midpoint alignment and curvature-aware distances.
    Midpoints are projected in 2D space to remove lateral-length bias when calculating spacing.
    """

    # Class-level constant (shared, immutable-by-convention)
    _DIR4_LABELS: ClassVar[np.ndarray] = np.array(["N", "E", "S", "W"], dtype=object)

    def __init__(self, trajectories: Union[Dict[str, pd.DataFrame], pd.DataFrame]):
        if isinstance(trajectories, pd.DataFrame):
            if "uwi" not in trajectories.columns:
                raise ValueError("Trajectory DataFrame must contain 'uwi' column.")
            self._trajectory_df = trajectories.reset_index(drop=True)
            self.trajectories = {
                cid: group for cid, group in self._trajectory_df.groupby("uwi")
            }
        elif isinstance(trajectories, dict):
            self.trajectories = trajectories
            self._trajectory_df = pd.concat(
                trajectories.values(), keys=trajectories.keys()
            ).reset_index(drop=True)
        else:
            raise ValueError("Invalid type for trajectories. Must be DataFrame or Dict.")

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
    ) -> Optional[pd.DataFrame]:
        """
        Compute spacing metrics for all well pairs, now with angle-aware routing:
        - parallel-like pairs: crossline |Δy(x)| over true i-frame overlap (your existing method)
        - oblique pairs: nearest-projection mean/median (walk along i, project to nearest point on k)
        - perpendicular pairs: closest approach (min distance) and optional ±window_ft mean
        """
        # Build the pair cache once (local frames + coarse arrays for precheck)
        self._build_pair_cache(use_pca_axis, ds_crossline_step_ft)

        # 1) Midpoints + drill directions (vertical distances remain midpoint-based)
        midpoint_df = self._compute_normalized_midpoints(frac=frac, use_interpolation=use_interpolation)
        drill_dirs = self._compute_drill_directions()
        midpoint_df["drill_direction"] = drill_dirs

        # 2) Arrays
        ids = midpoint_df.index.to_numpy()
        coords = midpoint_df[["x", "y", "tvd"]].to_numpy()
        lat_lon = midpoint_df[["latitude", "longitude"]].to_numpy()
        directions = midpoint_df["drill_direction"].to_numpy()

        # 3) Prefilter pairs (miles)
        if max_distance_miles is not None:
            lat = lat_lon[:, 0]; lon = lat_lon[:, 1]
            i_idx, k_idx = self._filter_close_pairs(lat, lon, max_distance_miles)
        else:
            i_idx, k_idx = self._get_pairwise_indices(ids)

        pairs = list(zip(i_idx, k_idx))
        batch_generator = list(self._batch_filtered_indices(pairs, batch_size=batch_size))
        n_batches = len(batch_generator)

        if save_batches_dir:
            os.makedirs(save_batches_dir, exist_ok=True)

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
            )
            if save_batches_dir:
                filepath = os.path.join(save_batches_dir, f"spacing_batch_{batch_number:04d}.parquet")
                batch_df.to_parquet(filepath, index=False)
            return batch_df

        tqdm_kwargs = dict(
            desc="🚀 Calculating Spacing (Parallel)",
            dynamic_ncols=True,
            smoothing=0.3,
            bar_format="{desc}: |{bar:40}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ascii="░▒█",
            leave=True,
        )

        results = Parallel(n_jobs=-1)(
            delayed(process_and_save)(batch_num, i_i, k_i)
            for batch_num, (i_i, k_i) in tqdm(enumerate(batch_generator), total=n_batches, **tqdm_kwargs)
        )

        if save_batches_dir:
            print(f"✅ All batches saved to {save_batches_dir}")
            return None
        else:
            return pd.concat(results, ignore_index=True)
    
    def _load_saved_batches(self, batch_folder: str) -> pd.DataFrame:
        """
        Load all saved spacing batch Parquet files from a folder and combine into a single DataFrame.

        Parameters
        ----------
        batch_folder : str
            Path to the folder where batch Parquet files are stored.

        Returns
        -------
        pd.DataFrame
            Combined spacing DataFrame.
        """
        if not os.path.isdir(batch_folder):
            raise FileNotFoundError(f"Batch folder '{batch_folder}' not found.")

        batch_files = sorted([
            os.path.join(batch_folder, f)
            for f in os.listdir(batch_folder)
            if f.endswith(".parquet")
        ])

        if not batch_files:
            raise ValueError(f"No Parquet files found in folder '{batch_folder}'.")

        print(f"🔍 Found {len(batch_files)} batch files. Loading and combining...")

        dfs = []
        for file in batch_files:
            dfs.append(pd.read_parquet(file))

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(combined_df):,} rows from all batches.")
        return combined_df

    def _filter_close_pairs(self, lat: np.ndarray, lon: np.ndarray, max_distance_miles: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:

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
        Computes midpoints for each well either by interpolating along the well trajectory
        using MD-based fractional position or by averaging heel and toe coordinates.

        Parameters:
        -----------
        frac : float
            Fractional location along the lateral to compute the midpoint (0.0 to 1.0).
        use_interpolation : bool
            If True, uses curvature-aware interpolation along MD.
            If False, uses geometric midpoint between heel and toe.

        Returns:
        --------
        pd.DataFrame indexed by 'uwi', containing:
            ['x', 'y', 'tvd', 'latitude', 'longitude']
        """
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
        return midpoint_df.set_index("uwi")
    
    def _compute_drill_directions(self) -> pd.Series:
        median_azimuth = self._trajectory_df.groupby("uwi")["azimuth"].median()
        is_ew = ((median_azimuth >= 45) & (median_azimuth <= 135)) | ((median_azimuth >= 225) & (median_azimuth <= 315))
        return pd.Series(np.where(is_ew, "EW", "NS"), index=median_azimuth.index, name="drill_direction")

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
        from pyproj import Geod
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
        # Compute each well's lateral length to pick a *global* coarse point count
        lengths = {}
        for uwi, df in self.trajectories.items():
            XY = df.sort_values("md")[["x","y"]].to_numpy()
            d = np.hypot(np.diff(XY[:,0]), np.diff(XY[:,1]))
            lengths[uwi] = float(d.sum())

        # Choose a *single* coarse count using median length / step, then clamp
        median_len = np.median(list(lengths.values())) if lengths else 3000.0
        m_guess = int(np.ceil(max(median_len, 1.0) / max(ds_crossline_step_ft, 1))) + 1
        M_ds = int(np.clip(m_guess, ds_min_points, ds_max_points))

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
    
    def _arclength(self, X: np.ndarray) -> np.ndarray:
        """Cumulative arclength for 2D polyline (x,y)."""
        d = np.hypot(np.diff(X[:,0]), np.diff(X[:,1]))
        return np.concatenate([[0.0], np.cumsum(d)])
    
    def _interp_by_arclength(self, X: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
        """
        Interpolate 2D polyline X to arclength grid s_targets. Returns Mx2 array.
        """
        s = self._arclength(X)
        keep = np.concatenate([[True], np.diff(s) > 1e-9])  # drop zero-length steps
        s, X = s[keep], X[keep]
        xi = np.interp(s_targets, s, X[:,0])
        yi = np.interp(s_targets, s, X[:,1])
        return np.column_stack([xi, yi])
    
    def _interp_attr_by_arclength(
        self,
        X: np.ndarray,          # (N,2) polyline
        attr: np.ndarray,       # (N,) attribute values (lat or lon)
        s_targets: np.ndarray   # (M,) arclength positions
    ) -> np.ndarray:
        s = self._arclength(X)
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        s, Xattr = s[keep], attr[keep]
        return np.interp(s_targets, s, Xattr)

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
        j: (n,) segment indices (X[j] -> X[j+1]),
        t: (n,) segment parameters in [0,1].
        """
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
    ) -> pd.DataFrame:
        """
        Angle-aware routing:
        - parallel-like (Δθ ≤ theta_parallel_deg): crossline |Δy(x)| over i-frame overlap (existing flow)
        - oblique (theta_parallel_deg < Δθ < theta_perp_deg): nearest-projection mean/median
        - perpendicular (Δθ ≥ theta_perp_deg): closest-approach min (+ optional ±window mean)
        Axis-constrained direction is always reported:
        NS reference -> E/W
        EW reference -> N/S
        """
        rows: List[Dict] = []

        if getattr(self, "_paircache", None) is None or self._paircache.get("use_pca_axis", None) != use_pca_axis:
            self._build_pair_cache(use_pca_axis, ds_crossline_step_ft)

        cache = self._paircache
        LL_cache = cache["LL"]    # <-- ADD

        # --- tiny helpers used inside this function ---
        def _arclength(X: np.ndarray) -> np.ndarray:
            d = np.hypot(np.diff(X[:, 0]), np.diff(X[:, 1]))
            return np.concatenate([[0.0], np.cumsum(d)])

        def _interp_by_arclength(X: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
            s = _arclength(X)
            keep = np.concatenate([[True], np.diff(s) > 1e-9])
            s, Xc = s[keep], X[keep]
            xi = np.interp(s_targets, s, Xc[:, 0])
            yi = np.interp(s_targets, s, Xc[:, 1])
            return np.column_stack([xi, yi])

        def _interp_attr_by_arclength(X: np.ndarray, attr: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
            s = _arclength(X)
            keep = np.concatenate([[True], np.diff(s) > 1e-9])
            s, a = s[keep], attr[keep]
            return np.interp(s_targets, s, a)

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

                # survivors → exact clip + spacing + axis-constrained direction
                df_i = self.trajectories[uwi_i].sort_values("md")
                Xi_full = self._project_xy_to_frame(df_i, origin_i, ex_i, ey_i)
                lat_i_full = df_i["latitude"].to_numpy(float) if "latitude" in df_i.columns else None
                lon_i_full = df_i["longitude"].to_numpy(float) if "longitude" in df_i.columns else None

                from pyproj import Geod
                geod = Geod(ellps="WGS84") if (lat_i_full is not None) else None

                for local, k in enumerate(k_list):
                    if not (local in par_idx and keep[local]):
                        continue
                    uwi_k = ids[k]
                    band = (float(x_lo[local]), float(x_hi[local]))

                    df_k = self.trajectories[uwi_k].sort_values("md")
                    Xk_full = self._project_xy_to_frame(df_k, origin_i, ex_i, ey_i)

                    Xi_seg = self._clip_polyline_by_x_band(Xi_full, band)
                    Xk_seg = self._clip_polyline_by_x_band(Xk_full, band)

                    if Xi_seg.shape[0] == 0 or Xk_seg.shape[0] == 0:
                        if emit_rejected:
                            rows.append({
                                "well_i": uwi_i, "well_k": uwi_k,
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
                        continue

                    # direction over overlap (axis-constrained, geodetic if available)
                    if (lat_i_full is not None) and ("latitude" in df_k.columns):
                        lat_k_full = df_k["latitude"].to_numpy(float)
                        lon_k_full = df_k["longitude"].to_numpy(float)

                        Xi_seg_ll, lat_i_seg, lon_i_seg = self._clip_polyline_with_latlon_by_x_band(
                            Xi_full, lat_i_full, lon_i_full, band
                        )
                        Xk_seg_ll, lat_k_seg, lon_k_seg = self._clip_polyline_with_latlon_by_x_band(
                            Xk_full, lat_k_full, lon_k_full, band
                        )

                        dir_mode_axis, dir_conf_axis, dir_dist_axis = self._axis_constrained_direction_over_overlap(
                            Xi_seg, Xk_seg,
                            lat_i_seg, lon_i_seg, lat_k_seg, lon_k_seg,
                            want_axis=dir_axis_i,
                            step_ft=step_ft, n_samples=n_samples,
                            deadband=0.15, tie_tol=0.05
                        )
                    else:
                        dir_mode_axis, dir_conf_axis, dir_dist_axis = None, np.nan, ""

                    # spacing stats
                    horiz_mean, n_used, Lmin = self._crossline_spacing_from_overlap(
                        Xi_seg, Xk_seg, step_ft=step_ft, n_samples=n_samples
                    )
                    horiz_med,  _,     _  = self._crossline_spacing_median_from_overlap(
                        Xi_seg, Xk_seg, step_ft=step_ft, n_samples=n_samples
                    )

                    A = coords[i]; B = coords[k]
                    vertical = float(abs(B[2] - A[2])); dist3d = float(np.hypot(horiz_mean, vertical))

                    # --- NEW: derive per-segment lengths used, full LLs, and overlap% ---
                    Li = float(self._arclength(Xi_seg)[-1]) if Xi_seg.size else 0.0          # NEW
                    Lk = float(self._arclength(Xk_seg)[-1]) if Xk_seg.size else 0.0          # NEW
                    overlap_len_common = float(min(Li, Lk))                                   # NEW

                    LL_i = float(LL_cache[uwi_i])                                             # NEW
                    LL_k = float(LL_cache[uwi_k])                                             # NEW
                    overlap_pct_i = (overlap_len_common / LL_i) if LL_i > 0 else float("nan") # NEW
                    overlap_pct_k = (overlap_len_common / LL_k) if LL_k > 0 else float("nan") # NEW

                    rows.append({
                        "well_i": uwi_i, "well_k": uwi_k,
                        "horizontal_dist": horiz_mean, "horizontal_dist_median": horiz_med, "vertical_dist": vertical,
                        "3D_dist": dist3d,
                        "drill_direction_i": directions[i], "drill_direction_k": directions[k],

                        # NEW: clearer alias with same value
                        "overlap_len_common_ft": overlap_len_common,                           # NEW

                        # NEW: full laterals + overlap%
                        "LL_i": LL_i, "LL_k": LL_k,                                           # NEW
                        "overlap_pct_i": overlap_pct_i,                                       # NEW
                        "overlap_pct_k": overlap_pct_k,                                       # NEW

                        "n_samples": n_used,
                        "dy_p5": float(dy_p[local]) if np.isfinite(dy_p[local]) else np.nan,
                        "angle_deg": float(angles[local]),
                        "pair_alignment": "parallel_like",
                        "min_distance_ft": np.nan,
                        "mean_windowed_ft": np.nan,
                        "reject_reason": "",
                        "direction_axis": dir_axis_i,
                        "direction_to_k_from_i_axis": dir_mode_axis,
                        "direction_axis_confidence": dir_conf_axis,
                        "direction_axis_distribution": dir_dist_axis,
                        "axis_forced": True
                    })


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

                uwi_k = ids[k]
                df_k_full = self.trajectories[uwi_k].sort_values("md")
                Xk_utm = df_k_full[["x", "y"]].to_numpy()

                # sample along i
                si = _arclength(Xi_utm); Li = si[-1]
                if n_samples is None:
                    step = max(int(step_ft), 1)
                    n = max(int(np.floor(Li / step)) + 1, 2)
                else:
                    n = max(int(n_samples), 2)
                s_targets = np.linspace(0.0, Li, n)
                Pi = _interp_by_arclength(Xi_utm, s_targets)

                # nearest on k for each Pi (vectorized)
                d, j_arr, t_arr = self._nearest_distances_to_polyline(Pi, Xk_utm)

                # geodetic directions (axis-constrained) if lat/lon available
                dir_mode_axis, dir_conf_axis, dist_str_axis = None, np.nan, ""
                if (geod is not None) and ("latitude" in df_k_full.columns):
                    lat_k_full = df_k_full["latitude"].to_numpy(float)
                    lon_k_full = df_k_full["longitude"].to_numpy(float)

                    # i: arclength-interpolated lat/lon at s_targets
                    lat_i_s = _interp_attr_by_arclength(Xi_utm, lat_i_full, s_targets)
                    lon_i_s = _interp_attr_by_arclength(Xi_utm, lon_i_full, s_targets)

                    # k: nearest segment interpolation at (j,t)
                    lat_k_s = lat_k_full[j_arr] + t_arr*(lat_k_full[j_arr+1] - lat_k_full[j_arr])
                    lon_k_s = lon_k_full[j_arr] + t_arr*(lon_k_full[j_arr+1] - lon_k_full[j_arr])

                    dir_mode_axis, dir_conf_axis, dist_str_axis = self._axis_constrained_direction_from_pairs(
                        lat_i_s, lon_i_s, lat_k_s, lon_k_s, want_axis=dir_axis_i,
                        deadband=0.15, tie_tol=0.05
                    )

                mean_d = float(d.mean())
                median_d = float(np.median(d))
                min_d = float(d.min())

                mean_windowed = np.nan
                if use_windowed_mean:
                    # average distance in a ±window_ft band around the minimum
                    idx_min = int(np.argmin(d))
                    s0 = s_targets[idx_min]
                    win_mask = (s_targets >= s0 - window_ft) & (s_targets <= s0 + window_ft)
                    if win_mask.any():
                        mean_windowed = float(d[win_mask].mean())

                A = coords[i]; B = coords[k]
                vertical = float(abs(B[2] - A[2]))
                dist3d = float(np.hypot(mean_d, vertical))

                rows.append({
                    "well_i": uwi_i, "well_k": uwi_k,
                    "horizontal_dist": mean_d,
                    "horizontal_dist_median": median_d,
                    "vertical_dist": vertical, "3D_dist": dist3d,
                    "drill_direction_i": directions[i], "drill_direction_k": directions[k],

                    # keep NaN for no x-overlap concept
                    "overlap_len_common_ft": float("nan"),          # NEW

                    # NEW: still emit full LLs; overlap% not applicable here
                    "LL_i": float(LL_cache[uwi_i]),                 # NEW
                    "LL_k": float(LL_cache[uwi_k]),                 # NEW
                    "overlap_pct_i": float("nan"),                  # NEW
                    "overlap_pct_k": float("nan"),                  # NEW

                    "n_samples": int(n),
                    "dy_p5": np.nan,
                    "angle_deg": float(angles[idx_k]),
                    "pair_alignment": "oblique" if is_oblique[idx_k] else "perpendicular",
                    "min_distance_ft": min_d,
                    "mean_windowed_ft": mean_windowed,
                    "reject_reason": "",
                    "direction_axis": dir_axis_i,
                    "direction_to_k_from_i_axis": dir_mode_axis,
                    "direction_axis_confidence": dir_conf_axis,
                    "direction_axis_distribution": dist_str_axis,
                    "axis_forced": True
                })


        return pd.DataFrame(rows)
#%%

class DirectionalBenchNeighbors:
    """
    Summarize nearest neighbors per well_i by bench and opposite direction.

    For each well_i, produce:
      - SAME-bench closest (any direction, or axis-limited) -> *_same_1
      - SAME-bench closest in the OPPOSITE direction of same_1 -> *_same_2
      - DIFFERENT-bench closest (any direction, or axis-limited) -> *_near_1
      - DIFFERENT-bench closest in the OPPOSITE direction of near_1 -> *_near_2

    Required columns
    ----------------
    spacing_df:
        'well_i', 'well_k', 'horizontal_dist', 'vertical_dist', '3D_dist',
        'direction_to_k_from_i_axis', 'overlap_pct_k'
        # overlap_pct_k is only required if overlap_pct_k_min is provided.
    header_df:
        'uwi', 'bench'

    Parameters (at call-time)
    -------------------------
    cutoff_ft : float
        Horizontal cutoff distance (feet).
    vertical_cutoff_ft : float, optional
        If provided, also require vertical_dist <= vertical_cutoff_ft.
        If None, no vertical filter is applied.
    axis_mode : {'any', 'EW', 'NS'}, default 'any'
        If 'EW', only E/W directions are eligible for the *_1 pick.
        If 'NS', only N/S directions are eligible for the *_1 pick.
        If 'any', all four directions can compete for *_1.
    prefer_axis : {'EW', 'NS'} | None, default None
        If set and there is a tie in horizontal distance for *_1 across axes,
        prefer the candidate whose direction belongs to that axis family.
    overlap_pct_k_min : float, optional
        If provided, require that overlap_pct_k >= this value for a pair
        to be considered. This ensures that well_k overlaps well_i by at
        least the given percentage of k’s own lateral length.

    Output columns
    --------------
      uwi_same_1,  hz_ft_to_same_1,  vt_ft_to_same_1,  3d_ft_to_same_1,
      uwi_same_2,  hz_ft_to_same_2,  vt_ft_to_same_2,  3d_ft_to_same_2,
      uwi_near_1,  hz_ft_to_near_1,  vt_ft_to_near_1,  3d_ft_to_near_1,
      uwi_near_2,  hz_ft_to_near_2,  vt_ft_to_near_2,  3d_ft_to_near_2

    Examples
    --------
    >>> nb = DirectionalBenchNeighbors()

    # Example 1: horizontal cutoff only
    >>> out1 = nb.summarize(spacing_df, header_df, cutoff_ft=1320.0)
    >>> out1.filter(regex="^uwi_|^hz_ft_").head()

    # Example 2: add vertical cutoff of 200 ft
    >>> out2 = nb.summarize(spacing_df, header_df,
    ...                     cutoff_ft=1320.0, vertical_cutoff_ft=200.0)

    # Example 3: restrict initial pick to EW directions only
    >>> out3 = nb.summarize(spacing_df, header_df,
    ...                     cutoff_ft=1320.0, axis_mode="EW")

    # Example 4: allow any axis but prefer NS in case of ties
    >>> out4 = nb.summarize(spacing_df, header_df,
    ...                     cutoff_ft=1320.0, axis_mode="any", prefer_axis="NS")

    # Example 5: require at least 80% overlap of k relative to i
    >>> out5 = nb.summarize(spacing_df, header_df,
    ...                     cutoff_ft=1320.0, overlap_pct_k_min=0.80)
    """

    # --- Type aliases inside the class ---
    Direction = Literal["E", "W", "N", "S"]
    AxisMode = Literal["any", "EW", "NS"]
    AxisPref = Literal["EW", "NS"]

    _OPPOSITE: Dict[Direction, Direction] = {"E": "W", "W": "E", "N": "S", "S": "N"}

    def __init__(self, *, tie_break_on: str = "well_k") -> None:
        """
        Parameters
        ----------
        tie_break_on : str, default 'well_k'
            Secondary stable key when distances tie.
        """
        self.tie_break_on = tie_break_on

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
        overlap_pct_k_min: Optional[float] = None,  # <-- NEW
    ) -> pd.DataFrame:
        self._validate_inputs(spacing_df, header_df)
        """Return one summary row per well_i with *_same_{1,2} and *_near_{1,2}.

        See class docstring for details on behavior and examples.

        Raises
        ------
        ValueError
            If required columns are missing or inputs are inconsistent.
        """
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

        # --- Apply cutoffs ---
        mask = spacing["horizontal_dist"] <= cutoff_ft
        if vertical_cutoff_ft is not None:
            mask &= spacing["vertical_dist"] <= vertical_cutoff_ft

        # NEW: optional overlap filter on k
        if overlap_pct_k_min is not None:
            if "overlap_pct_k" not in spacing.columns:
                raise ValueError(
                    "spacing_df missing required column 'overlap_pct_k' when overlap_pct_k_min is provided."
                )
            mask &= spacing["overlap_pct_k"] >= overlap_pct_k_min

        spacing = spacing.loc[mask].copy()

        if spacing.empty:
            wells = spacing_df["well_i"].astype(str).unique()
            return self._empty_summary(wells)

        # SAME vs NEAR
        spacing["is_same"] = spacing["bench_i"] == spacing["bench_k"]
        same = spacing.loc[spacing["is_same"]].copy()
        near = spacing.loc[~spacing["is_same"]].copy()

        same_summary = self._compute_category_summary(
            same, category="same", axis_mode=axis_mode, prefer_axis=prefer_axis
        )
        near_summary = self._compute_category_summary(
            near, category="near", axis_mode=axis_mode, prefer_axis=prefer_axis
        )

        all_wells = spacing_df["well_i"].astype(str).drop_duplicates().to_frame(name="well_i")
        out = (
            all_wells.merge(same_summary, on="well_i", how="left")
                     .merge(near_summary, on="well_i", how="left")
        )

        ordered_cols = (
            ["well_i"]
            + self._category_cols("same", 1)
            + self._category_cols("same", 2)
            + self._category_cols("near", 1)
            + self._category_cols("near", 2)
        )
        existing_cols = [c for c in ordered_cols if c in out.columns]
        return out[existing_cols + [c for c in out.columns if c not in existing_cols]]
    
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
        return [
            f"uwi_{cat}_{idx}",
            f"hz_ft_to_{cat}_{idx}",
            f"vt_ft_to_{cat}_{idx}",
            f"3d_ft_to_{cat}_{idx}",
        ]

    @staticmethod
    def _empty_summary(wells: Iterable[str]) -> pd.DataFrame:
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

#%%

def debug_pair_spacing(
    df: pd.DataFrame,
    uwi_i: Union[str, int],
    uwi_k: Union[str, int],
    *,
    step_ft: int = 100,
    use_pca_axis: bool = True,
    theta_parallel_deg: float = 25.0,
    theta_perp_deg: float = 65.0,
    arrow_stride: int = 5,
    save_dir: Optional[Union[str, os.PathLike]] = None,
    figure_prefix: str = "spacing",
    show: bool = True,
) -> Dict[str, Any]:
    # ---------- checks ----------
    for col in ("uwi", "md", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"df must contain column '{col}'")
    has_ll = ("latitude" in df.columns) and ("longitude" in df.columns)

    geod = None
    if has_ll:
        try:
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
        except Exception:
            geod = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    df_i = df[df["uwi"] == uwi_i].sort_values("md").copy()
    df_k = df[df["uwi"] == uwi_k].sort_values("md").copy()
    if df_i.empty or df_k.empty:
        raise ValueError("One or both UWIs not found in df.")

    # ---------- helpers ----------
    def build_ex_from_df(df_: pd.DataFrame, use_pca: bool) -> np.ndarray:
        XY = df_[["x", "y"]].to_numpy()
        heel, toe = XY[0], XY[-1]
        if use_pca and len(XY) >= 2:
            C = XY - XY.mean(0)
            _, _, Vt = np.linalg.svd(C, full_matrices=False)
            ex_ = Vt[0]
            if np.dot(ex_, toe - heel) < 0:
                ex_ = -ex_
        else:
            v = toe - heel
            ex_ = v / (np.linalg.norm(v) + 1e-12)
        return ex_

    def build_local_frame_from_i(df_i_: pd.DataFrame, use_pca: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        XY = df_i_[["x", "y"]].to_numpy()
        origin_ = XY[0].copy()
        ex_ = build_ex_from_df(df_i_, use_pca)
        ey_ = np.array([-ex_[1], ex_[0]])
        return origin_, ex_, ey_

    def project_xy_to_frame(df_: pd.DataFrame, origin_: np.ndarray, ex_: np.ndarray, ey_: np.ndarray) -> np.ndarray:
        XY = df_[["x", "y"]].to_numpy()
        R = XY - origin_
        return np.column_stack([R @ ex_, R @ ey_])

    def arclength(X: np.ndarray) -> np.ndarray:
        d = np.hypot(np.diff(X[:, 0]), np.diff(X[:, 1]))
        return np.concatenate([[0.0], np.cumsum(d)])

    def interp_by_arclength(X: np.ndarray, s_targets: np.ndarray) -> np.ndarray:
        s = arclength(X)
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        s, Xc = s[keep], X[keep]
        xi = np.interp(s_targets, s, Xc[:, 0])
        yi = np.interp(s_targets, s, Xc[:, 1])
        return np.column_stack([xi, yi])

    def x_overlap_interval(Xi_: np.ndarray, Xk_: np.ndarray) -> Optional[Tuple[float, float]]:
        xi_min, xi_max = Xi_[:, 0].min(), Xi_[:, 0].max()
        xk_min, xk_max = Xk_[:, 0].min(), Xk_[:, 0].max()
        x_lo_, x_hi_ = max(xi_min, xk_min), min(xi_max, xk_max)
        return (x_lo_, x_hi_) if (x_hi_ > x_lo_) else None

    def clip_by_x_band(X: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        x_lo_, x_hi_ = band
        x = X[:, 0]
        pts = []
        for j in range(len(X) - 1):
            x0, x1 = x[j], x[j + 1]
            P0, P1 = X[j], X[j + 1]
            seg_min, seg_max = (x0, x1) if x0 <= x1 else (x1, x0)
            if seg_max < x_lo_ or seg_min > x_hi_:
                continue
            if x_lo_ <= x0 <= x_hi_:
                pts.append(P0)
            for xb in (x_lo_, x_hi_):
                denom = (x1 - x0)
                if denom != 0.0 and (x0 - xb) * (x1 - xb) < 0.0:
                    t = (xb - x0) / denom
                    pts.append(P0 + t * (P1 - P0))
            if j == len(X) - 2 and (x_lo_ <= x1 <= x_hi_):
                pts.append(P1)
        if not pts:
            return np.empty((0, 2))
        P = np.vstack(pts)
        if len(P) > 1:
            dup = np.all(np.isclose(np.diff(P, axis=0), 0.0, atol=1e-9), axis=1)
            keep = np.ones(len(P), dtype=bool); keep[1:] = ~dup
            P = P[keep]
        return P

    def interp_y_of_x(X: np.ndarray, x_targets: np.ndarray) -> np.ndarray:
        x = X[:, 0]; y = X[:, 1]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        keep = np.concatenate([[True], np.diff(xs) > 1e-9])
        xs, ys = xs[keep], ys[keep]
        if xs.size == 0: return np.zeros_like(x_targets, dtype=float)
        if xs.size == 1: return np.full_like(x_targets, ys[0], dtype=float)
        xq = np.clip(x_targets, xs[0], xs[-1])
        return np.interp(xq, xs, ys)

    def clip_with_latlon_by_x_band(
        X: np.ndarray, lat: np.ndarray, lon: np.ndarray, band: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_lo_, x_hi_ = band
        x = X[:, 0]
        pts_xy, pts_lat, pts_lon = [], [], []
        for j in range(len(X) - 1):
            x0, x1 = x[j], x[j + 1]
            P0, P1 = X[j], X[j + 1]
            lat0, lat1 = lat[j], lat[j + 1]
            lon0, lon1 = lon[j], lon[j + 1]
            seg_min, seg_max = (x0, x1) if x0 <= x1 else (x1, x0)
            if seg_max < x_lo_ or seg_min > x_hi_:
                continue
            if x_lo_ <= x0 <= x_hi_:
                pts_xy.append(P0); pts_lat.append(lat0); pts_lon.append(lon0)
            for xb in (x_lo_, x_hi_):
                denom = (x1 - x0)
                if denom != 0.0 and (x0 - xb) * (x1 - xb) < 0.0:
                    t = (xb - x0) / denom
                    pts_xy.append(P0 + t * (P1 - P0))
                    pts_lat.append(lat0 + t * (lat1 - lat0))
                    pts_lon.append(lon0 + t * (lon1 - lon0))
            if j == len(X) - 2 and (x_lo_ <= x1 <= x_hi_):
                pts_xy.append(P1); pts_lat.append(lat1); pts_lon.append(lon1)
        if not pts_xy:
            return np.empty((0, 2)), np.empty((0,)), np.empty((0,))
        XYc = np.vstack(pts_xy)
        latc = np.asarray(pts_lat, dtype=float)
        lonc = np.asarray(pts_lon, dtype=float)
        if len(XYc) > 1:
            dup = np.all(np.isclose(np.diff(XYc, axis=0), 0.0, atol=1e-9), axis=1)
            keep = np.ones(len(XYc), dtype=bool); keep[1:] = ~dup
            XYc, latc, lonc = XYc[keep], latc[keep], lonc[keep]
        return XYc, latc, lonc

    def crossline_stats_from_overlap(
        Xi_seg_: np.ndarray, Xk_seg_: np.ndarray, step: int
    ) -> Tuple[float, float, float, float, int, np.ndarray, np.ndarray, np.ndarray]:
        if Xi_seg_.size == 0 or Xk_seg_.size == 0:
            return (np.nan, np.nan, np.nan, np.nan, 0,
                    np.empty((0,)), np.empty((0,)), np.empty((0,)))
        xi_min, xi_max = float(Xi_seg_[:, 0].min()), float(Xi_seg_[:, 0].max())
        xk_min, xk_max = float(Xk_seg_[:, 0].min()), float(Xk_seg_[:, 0].max())
        x_lo_, x_hi_ = max(xi_min, xk_min), min(xi_max, xk_max)
        if not (x_hi_ > x_lo_):
            return (np.nan, np.nan, np.nan, np.nan, 0,
                    np.empty((0,)), np.empty((0,)), np.empty((0,)))
        step = max(int(step), 1)
        N = max(int(np.floor((x_hi_ - x_lo_) / step)) + 1, 2)
        xgrid = np.linspace(x_lo_, x_hi_, N)
        yi = interp_y_of_x(Xi_seg_, xgrid)
        yk = interp_y_of_x(Xk_seg_, xgrid)
        dy = np.abs(yk - yi)
        return float(dy.mean()), float(np.median(dy)), float(np.percentile(dy, 5)), float(np.percentile(dy, 95)), int(N), xgrid, yi, yk

    def local_to_utm(X_local: np.ndarray, origin_: np.ndarray, ex_: np.ndarray, ey_: np.ndarray) -> np.ndarray:
        return origin_[None, :] + X_local[:, [0]] * ex_[None, :] + X_local[:, [1]] * ey_[None, :]

    def bin4_from_geod_az(az_deg_from_north: np.ndarray) -> np.ndarray:
        labels = np.array(["N", "E", "S", "W"], dtype=object)
        a = az_deg_from_north % 360.0
        idx = ((a + 45.0) // 90.0).astype(int) % 4
        return labels[idx]

    def vec_nearest_to_polyline(P: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = X[:-1]; B = X[1:]; AB = B - A
        AP = P[:, None, :] - A[None, :, :]
        denom = (AB[None, :, 0]**2 + AB[None, :, 1]**2)
        denom = np.where(denom == 0.0, 1e-12, denom)
        t = (AP[...,0]*AB[None,:,0] + AP[...,1]*AB[None,:,1]) / denom
        t = np.clip(t, 0.0, 1.0)
        Qall = A[None, :, :] + t[..., None] * AB[None, :, :]
        diff = Qall - P[:, None, :]
        d2 = diff[...,0]**2 + diff[...,1]**2
        j = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(P.shape[0]), j])
        t_sel = t[np.arange(P.shape[0]), j]
        Q = Qall[np.arange(P.shape[0]), j]
        return d, j, t_sel, Q

    # ---------- frames & angle ----------
    origin, ex_i, ey_i = build_local_frame_from_i(df_i, use_pca_axis)
    Xi = project_xy_to_frame(df_i, origin, ex_i, ey_i)
    Xk = project_xy_to_frame(df_k, origin, ex_i, ey_i)

    ex_k_global = build_ex_from_df(df_k, use_pca_axis)
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(ex_i, ex_k_global)), -1.0, 1.0))))
    alignment = "parallel_like" if angle_deg <= theta_parallel_deg else ("perpendicular" if angle_deg >= theta_perp_deg else "oblique")

    # ---------- always save map + i-frame ----------
    paths: Dict[str, str] = {}
    def save_fig(fig, key):
        if save_dir is not None:
            p = os.path.join(save_dir, f"{figure_prefix}_{key}.png")
            fig.savefig(p, dpi=150, bbox_inches="tight"); paths[key] = p
        plt.close(fig)

    fig_map = plt.figure(figsize=(8, 5)); ax = fig_map.add_subplot(111)
    ax.plot(df_i["x"], df_i["y"], label=f"Well {uwi_i}")
    ax.plot(df_k["x"], df_k["y"], label=f"Well {uwi_k}")
    ax.axis("equal"); ax.legend()
    ax.set_title("Original UTM map view")
    ax.set_xlabel("Easting (ft)"); ax.set_ylabel("Northing (ft)")
    save_fig(fig_map, "map_view")

    fig_if = plt.figure(figsize=(8, 5)); ax = fig_if.add_subplot(111)
    ax.plot(Xi[:, 0], Xi[:, 1], label="i in i-frame")
    ax.plot(Xk[:, 0], Xk[:, 1], label="k in i-frame")
    ax.axis("equal"); ax.legend()
    ax.set_title("Projected into i-frame")
    ax.set_xlabel("x (along i)"); ax.set_ylabel("y (crossline)")
    save_fig(fig_if, "projected_iframe")

    # we’ll collect panels with kind info: ("title","cart/polar", drawer OR theta)
    panels: List[Tuple[str, str, Any]] = []
    panels.append(("Original UTM map view", "cart",
                   lambda ax: (ax.plot(df_i["x"], df_i["y"], label=f"Well {uwi_i}"),
                               ax.plot(df_k["x"], df_k["y"], label=f"Well {uwi_k}"),
                               ax.axis("equal"), ax.legend(),
                               ax.set_xlabel("Easting (ft)"), ax.set_ylabel("Northing (ft)"))))
    panels.append(("Projected into i-frame", "cart",
                   lambda ax: (ax.plot(Xi[:, 0], Xi[:, 1], label="i in i-frame"),
                               ax.plot(Xk[:, 0], Xk[:, 1], label="k in i-frame"),
                               ax.axis("equal"), ax.legend(),
                               ax.set_xlabel("x (along i)"), ax.set_ylabel("y (crossline)"))))

    metrics: Dict[str, Any] = {"pair_alignment": alignment, "angle_deg": angle_deg, "paths": paths}

    # ---------- PARALLEL-LIKE ----------
    if alignment == "parallel_like":
        band = x_overlap_interval(Xi, Xk)

        fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
        ax.plot(Xi[:, 0], Xi[:, 1], label="i (full)")
        ax.plot(Xk[:, 0], Xk[:, 1], label="k (full)")
        if band is not None:
            x_lo, x_hi = band
            ax.axvline(x_lo, linestyle="--"); ax.axvline(x_hi, linestyle="--")
        ax.axis("equal"); ax.legend()
        ax.set_title("Overlap band (x in i-frame)")
        ax.set_xlabel("x (along i)"); ax.set_ylabel("y (crossline)")
        save_fig(fig, "overlap_band")

        panels.append(("Overlap band (x in i-frame)", "cart",
                      lambda ax: (ax.plot(Xi[:, 0], Xi[:, 1], label="i (full)"),
                                  ax.plot(Xk[:, 0], Xk[:, 1], label="k (full)"),
                                  (band is not None and (ax.axvline(x_lo, ls="--"), ax.axvline(x_hi, ls="--"))),
                                  ax.axis("equal"), ax.legend(),
                                  ax.set_xlabel("x (along i)"), ax.set_ylabel("y (crossline)"))))

        if band is None:
            metrics.update({
                "overlap_x_band": None,
                "crossline_mean_ft": np.nan, "crossline_median_ft": np.nan,
                "crossline_p05_ft": np.nan, "crossline_p95_ft": np.nan,
                "crossline_n_stations": 0,
                "direction_mode_4way": None, "direction_confidence": np.nan,
                "direction_distribution": {},
            })
        else:
            x_lo, x_hi = band
            Xi_seg = clip_by_x_band(Xi, band)
            Xk_seg = clip_by_x_band(Xk, band)

            # clipped segments figure
            fig = plt.figure(figsize=(8, 5)); ax = fig.add_subplot(111)
            if Xi_seg.size: ax.plot(Xi_seg[:, 0], Xi_seg[:, 1], label="i (clipped)")
            if Xk_seg.size: ax.plot(Xk_seg[:, 0], Xk_seg[:, 1], label="k (clipped)")
            ax.axis("equal"); ax.legend()
            ax.set_title("Clipped segments used for spacing")
            ax.set_xlabel("x (along i)"); ax.set_ylabel("y (crossline)")
            save_fig(fig, "clipped_segments")

            panels.append(("Clipped segments for spacing", "cart",
                           lambda ax: ((Xi_seg.size and ax.plot(Xi_seg[:, 0], Xi_seg[:, 1], label="i (clipped)")),
                                       (Xk_seg.size and ax.plot(Xk_seg[:, 0], Xk_seg[:, 1], label="k (clipped)")),
                                       ax.axis("equal"), ax.legend(),
                                       ax.set_xlabel("x (along i)"), ax.set_ylabel("y (crossline)"))))

            # crossline stats
            cross_mean, cross_median, cross_p05, cross_p95, N, xgrid, yi, yk = crossline_stats_from_overlap(
                Xi_seg, Xk_seg, step=step_ft
            )
            if N > 0:
                dy = np.abs(yk - yi)
                fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
                ax.plot(xgrid - x_lo, dy)
                ax.set_title(f"Crossline |Δy(x)|; mean={cross_mean:.1f} ft, median={cross_median:.1f} ft")
                ax.set_xlabel("Along-overlap x (ft from band start)"); ax.set_ylabel("|Δy| (ft)")
                save_fig(fig, "crossline_dy_series")

                panels.append((f"Crossline |Δy(x)|; mean={cross_mean:.1f}, median={cross_median:.1f}", "cart",
                               lambda ax: (ax.plot(xgrid - x_lo, np.abs(yk - yi)),
                                           ax.set_xlabel("Along-overlap x (ft)"),
                                           ax.set_ylabel("|Δy| (ft)"))))

                # UTM arrows
                Xi_utml = local_to_utm(np.column_stack([xgrid, yi]), origin, ex_i, ey_i)
                Xk_utml = local_to_utm(np.column_stack([xgrid, yk]), origin, ex_i, ey_i)
                U = Xk_utml[:, 0] - Xi_utml[:, 0]
                V = Xk_utml[:, 1] - Xi_utml[:, 1]
                fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
                ax.plot(df_i["x"], df_i["y"], label=f"Well {uwi_i}")
                ax.plot(df_k["x"], df_k["y"], label=f"Well {uwi_k}")
                ax.quiver(Xi_utml[::arrow_stride, 0], Xi_utml[::arrow_stride, 1],
                          U[::arrow_stride], V[::arrow_stride],
                          angles='xy', scale_units='xy', scale=1, width=0.002)
                ax.axis("equal"); ax.legend()
                ax.set_title("Crossline sampling pairs in UTM with direction arrows (i → k)")
                ax.set_xlabel("Easting (ft)"); ax.set_ylabel("Northing (ft)")
                save_fig(fig, "pairs_utm_crossline_dir")

                panels.append(("Crossline pairs in UTM (i→k arrows)", "cart",
                               lambda ax: (ax.plot(df_i["x"], df_i["y"], label=f"Well {uwi_i}"),
                                           ax.plot(df_k["x"], df_k["y"], label=f"Well {uwi_k}"),
                                           ax.quiver(Xi_utml[::arrow_stride, 0], Xi_utml[::arrow_stride, 1],
                                                     U[::arrow_stride], V[::arrow_stride],
                                                     angles='xy', scale_units='xy', scale=1, width=0.002),
                                           ax.axis("equal"), ax.legend(),
                                           ax.set_xlabel("Easting (ft)"), ax.set_ylabel("Northing (ft)"))))

                # directions
                if has_ll:
                    lat_i = df_i["latitude"].to_numpy(float); lon_i = df_i["longitude"].to_numpy(float)
                    lat_k = df_k["latitude"].to_numpy(float); lon_k = df_k["longitude"].to_numpy(float)
                    Xi_seg_ll, lat_i_seg, lon_i_seg = clip_with_latlon_by_x_band(Xi, lat_i, lon_i, band)
                    Xk_seg_ll, lat_k_seg, lon_k_seg = clip_with_latlon_by_x_band(Xk, lat_k, lon_k, band)
                    if (lat_i_seg.size > 0) and (lat_k_seg.size > 0) and geod is not None:
                        lat_i_x = np.interp(xgrid, Xi_seg_ll[:, 0], lat_i_seg)
                        lon_i_x = np.interp(xgrid, Xi_seg_ll[:, 0], lon_i_seg)
                        lat_k_x = np.interp(xgrid, Xk_seg_ll[:, 0], lat_k_seg)
                        lon_k_x = np.interp(xgrid, Xk_seg_ll[:, 0], lon_k_seg)
                        az12, _, _ = geod.inv(lon_i_x, lat_i_x, lon_k_x, lat_k_x)
                        labels = bin4_from_geod_az(az12)
                        theta_rose = np.deg2rad((90.0 - az12) % 360.0)
                    else:
                        dv = Xk_utml - Xi_utml
                        ang_east = np.degrees(np.arctan2(dv[:, 1], dv[:, 0]))
                        az_from_n = (90.0 - ang_east) % 360.0
                        labels = bin4_from_geod_az(az_from_n)
                        theta_rose = np.deg2rad((90.0 - az_from_n) % 360.0)
                else:
                    dv = Xk_utml - Xi_utml
                    ang_east = np.degrees(np.arctan2(dv[:, 1], dv[:, 0]))
                    az_from_n = (90.0 - ang_east) % 360.0
                    labels = bin4_from_geod_az(az_from_n)
                    theta_rose = np.deg2rad((90.0 - az_from_n) % 360.0)

                uniq, counts = np.unique(labels, return_counts=True)
                best = int(np.argmax(counts))
                dir_mode = str(uniq[best]); dir_conf = counts[best] / float(labels.size)
                dist_map = {str(u): float(c) / float(labels.size) for u, c in zip(uniq, counts)}

                # save standalone polar
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='polar')
                bins = np.linspace(0.0, 2*np.pi, 5)
                counts_b, _ = np.histogram(theta_rose % (2*np.pi), bins=bins)
                widths = np.diff(bins); centers_b = bins[:-1] + widths/2
                ax.bar(centers_b, counts_b, width=widths, align='center')
                ax.set_title(f"Direction i→k (modal {dir_mode}, conf {dir_conf:.2f})")
                save_fig(fig, "direction_rose")

                # add to combined grid as POLAR
                panels.append((f"Direction i→k (modal {dir_mode}, conf {dir_conf:.2f})",
                               "polar", theta_rose))

                metrics.update({
                    "overlap_x_band": (float(x_lo), float(x_hi)),
                    "overlap_len_ft": float(x_hi - x_lo),
                    "crossline_mean_ft": float(cross_mean),
                    "crossline_median_ft": float(cross_median),
                    "crossline_p05_ft": float(cross_p05),
                    "crossline_p95_ft": float(cross_p95),
                    "crossline_n_stations": int(N),
                    "direction_mode_4way": dir_mode,
                    "direction_confidence": float(dir_conf),
                    "direction_distribution": dist_map,
                })
            else:
                metrics.update({
                    "overlap_x_band": (float(x_lo), float(x_hi)),
                    "overlap_len_ft": float(x_hi - x_lo),
                    "crossline_mean_ft": np.nan, "crossline_median_ft": np.nan,
                    "crossline_p05_ft": np.nan, "crossline_p95_ft": np.nan,
                    "crossline_n_stations": 0,
                    "direction_mode_4way": None, "direction_confidence": np.nan,
                    "direction_distribution": {},
                })

    # ---------- OBLIQUE / PERP ----------
    else:
        Xi_utm = df_i[["x", "y"]].to_numpy()
        Xk_utm = df_k[["x", "y"]].to_numpy()
        si = arclength(Xi_utm); Li = float(si[-1])
        step = max(int(step_ft), 1)
        n = max(int(np.floor(Li / step)) + 1, 2)
        s_targets = np.linspace(0.0, Li, n)
        Pi = interp_by_arclength(Xi_utm, s_targets)

        d, j_arr, t_arr, Q = vec_nearest_to_polyline(Pi, Xk_utm)
        mean_d = float(d.mean()); median_d = float(np.median(d)); min_d = float(d.min())

        # nearest series (save + panel)
        fig = plt.figure(figsize=(8, 4)); ax = fig.add_subplot(111)
        ax.plot(s_targets, d)
        ax.set_title(f"Nearest-projection distance (i → k); mean={mean_d:.1f} ft, median={median_d:.1f} ft")
        ax.set_xlabel("Distance along well i (ft)"); ax.set_ylabel("Nearest distance to k (ft)")
        save_fig(fig, "nearest_projection_series")

        panels.append((f"Nearest distance along i; mean={mean_d:.1f}, median={median_d:.1f}",
                       "cart", lambda ax: (ax.plot(s_targets, d),
                                           ax.set_xlabel("Distance along i (ft)"),
                                           ax.set_ylabel("Nearest dist to k (ft)"))))

        # UTM nearest segments (decimated)
        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
        ax.plot(df_i["x"], df_i["y"], label=f"Well {uwi_i}")
        ax.plot(df_k["x"], df_k["y"], label=f"Well {uwi_k}")
        stride = max(int(arrow_stride), 1)
        for idx in range(0, len(Pi), stride):
            ax.plot([Pi[idx, 0], Q[idx, 0]], [Pi[idx, 1], Q[idx, 1]])
        ax.axis("equal"); ax.legend()
        ax.set_title("Nearest-projection segments in UTM (i samples to nearest on k)")
        ax.set_xlabel("Easting (ft)"); ax.set_ylabel("Northing (ft)")
        save_fig(fig, "utm_nearest_segments")

        panels.append(("Nearest segments in UTM", "cart",
                       lambda ax: (ax.plot(df_i["x"], df_i["y"], label=f"Well {uwi_i}"),
                                   ax.plot(df_k["x"], df_k["y"], label=f"Well {uwi_k}"),
                                   [ax.plot([Pi[idx, 0], Q[idx, 0]], [Pi[idx, 1], Q[idx, 1]])
                                    for idx in range(0, len(Pi), stride)],
                                   ax.axis("equal"), ax.legend(),
                                   ax.set_xlabel("Easting (ft)"), ax.set_ylabel("Northing (ft)"))))

        # directions for nearest segments
        if has_ll and geod is not None:
            lat_i = df_i["latitude"].to_numpy(float); lon_i = df_i["longitude"].to_numpy(float)
            lat_k = df_k["latitude"].to_numpy(float); lon_k = df_k["longitude"].to_numpy(float)
            lat_i_s = np.interp(s_targets, si, lat_i)
            lon_i_s = np.interp(s_targets, si, lon_i)
            lat_k_s = lat_k[j_arr] + t_arr*(lat_k[j_arr+1] - lat_k[j_arr])
            lon_k_s = lon_k[j_arr] + t_arr*(lon_k[j_arr+1] - lon_k[j_arr])
            az12, _, _ = geod.inv(lon_i_s, lat_i_s, lon_k_s, lat_k_s)
            labels = bin4_from_geod_az(az12)
            theta_rose = np.deg2rad((90.0 - az12) % 360.0)
        else:
            dv = Q - Pi
            ang_east = np.degrees(np.arctan2(dv[:, 1], dv[:, 0]))
            az_from_n = (90.0 - ang_east) % 360.0
            labels = bin4_from_geod_az(az_from_n)
            theta_rose = np.deg2rad((90.0 - az_from_n) % 360.0)

        uniq, counts = np.unique(labels, return_counts=True)
        best = int(np.argmax(counts))
        dir_mode = str(uniq[best]); dir_conf = counts[best] / float(labels.size)
        dist_map = {str(u): float(c) / float(labels.size) for u, c in zip(uniq, counts)}

        # save standalone polar
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
        bins = np.linspace(0.0, 2*np.pi, 5)
        counts_b, _ = np.histogram(theta_rose % (2*np.pi), bins=bins)
        widths = np.diff(bins); centers_b = bins[:-1] + widths/2
        ax.bar(centers_b, counts_b, width=widths, align='center')
        ax.set_title(f"Direction i→k (modal {dir_mode}, conf {dir_conf:.2f})")
        save_fig(fig, "direction_rose")

        # add to combined grid as POLAR
        panels.append((f"Direction i→k (modal {dir_mode}, conf {dir_conf:.2f})",
                       "polar", theta_rose))

        metrics.update({
            "nearest_mean_ft": mean_d, "nearest_median_ft": median_d, "nearest_min_ft": min_d,
            "n_samples": int(n),
            "direction_mode_4way": dir_mode,
            "direction_confidence": float(dir_conf),
            "direction_distribution": dist_map,
        })

    # ---------- combined grid (with polar support) ----------
    if show and panels:
        n_cols = 2
        n_rows = int(np.ceil(len(panels) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), constrained_layout=True)
        axes = np.atleast_2d(axes)

        for idx, (title, kind, payload) in enumerate(panels):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            if kind == "cart":
                ax.set_title(title)
                payload(ax)  # drawer
            else:
                # replace this cell with a polar subplot
                fig.delaxes(ax)
                axp = fig.add_subplot(n_rows, n_cols, idx + 1, projection="polar")
                axp.set_title(title)
                theta = payload
                if theta.size > 0:
                    bins = np.linspace(0.0, 2*np.pi, 5)
                    counts_b, _ = np.histogram(theta % (2*np.pi), bins=bins)
                    widths = np.diff(bins); centers_b = bins[:-1] + widths/2
                    axp.bar(centers_b, counts_b, width=widths, align='center')

        # hide unused cells
        for j in range(len(panels), n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r, c].axis("off")
        plt.show()

    return metrics
#%%