# %%
from __future__ import annotations

import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np

from typing import Dict, Union, Optional, Any, Mapping, Set

import time
import pyproj

import logging

import os
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.database_manager import SQLAlchemyDBClient, DatabricksConfig
from src.utils import read_csv_with_mapper, read_excel_with_mapper
# %%
class WellDataLoader:
    """
    Load well header and directional survey data from:
      1) file (CSV/Excel) using *source → canonical* column mapping, or
      2) database via an injected db client, or
      3) an in-memory DataFrame.

    This class enforces a *single mapping convention* to minimize confusion:
    -----------------------------------------------------------------------
        column_map = { "Column Name In Source File": "canonical_name", ... }

    Why this matters
    ----------------
    - Your downstream pipeline becomes stable and predictable because it always
      sees canonical column names.
    - The mapping direction matches pd.DataFrame.rename and your helpers:
      `read_csv_with_mapper()` / `read_excel_with_mapper()`.

    Validation behavior
    -------------------
    After data is loaded (from file/DB/DataFrame), this class checks required
    canonical columns for that dataset. If required columns are missing, it raises
    a ValueError that lists:
      - full required canonical set
      - which columns are missing
      - a preview of columns that are present

    Examples
    --------
    (A) Header from CSV:

    >>> loader = WellDataLoader(db=None)
    >>> header_map = {
    ...     "API14": "uwi",
    ...     "LeaseName": "lease_name",
    ...     "WellName": "well_name",
    ...     "WellNumber": "well_num",
    ...     "Operator": "operator",
    ...     "RSV_CAT": "rsv_cat",
    ...     "Bench": "bench",
    ...     "FirstProdDate": "first_prod_date",
    ...     "CompletionStartDate": "comp_date",
    ...     "HoleDirection": "hole_direction",
    ...     "SurfaceLatitude": "surface_lat",
    ...     "SurfaceLongitude": "surface_lon",
    ... }
    >>> df_header = loader.get_header_data(
    ...     source="header.csv",
    ...     column_map=header_map,
    ...     dtype_map={"uwi": "string"},
    ... )

    (B) Directional from Excel (sheet "Survey"):

    >>> dir_map = {
    ...     "UWI": "uwi",
    ...     "MD": "md",
    ...     "TVD": "tvd",
    ...     "Incl": "inclination",
    ...     "Azim": "azimuth",
    ...     "Lat": "latitude",
    ...     "Lon": "longitude",
    ...     "X_Offset": "deviation_E/W",
    ...     "EW_Dir": "E/W",
    ...     "Y_Offset": "deviation_N/S",
    ...     "NS_Dir": "N/S",
    ...     "PointType": "point_type_name",
    ... }
    >>> df_dir = loader.get_directional_data(
    ...     source="dir.xlsx",
    ...     column_map=dir_map,
    ...     sheet_name="Survey",
    ... )

    (C) Header from DB:

    >>> loader = WellDataLoader(db=my_db_client)
    >>> df_header = loader.get_header_data(source=None, basin="MB", start_year=2019)
    """

    # ----------------------------
    # Canonical required columns
    # ----------------------------
    HEADER_REQUIRED_COLS: Set[str] = {
        "uwi",
        "lease_name",
        "well_name",
        "operator",
        "bench",
        "first_prod_date",
        "hole_direction",
        "well_status",
        "surface_lat",
        "surface_lon",
    }

    DIRECTIONAL_REQUIRED_COLS: Dict[str, Set[str]] = {
        "IHS": {
            "uwi",
            "md",
            "tvd",
            "inclination",
            "azimuth",
            "latitude",
            "longitude",
            "deviation_E/W",  # E_W in Enverus DS
            "E/W",
            "deviation_N/S",  # N_S in Enverus DS
            "N/S",
        },
        "enverus": {
            "uwi12",
            "md",
            "tvd",
            "inclination",
            "azimuth",
            "latitude",
            "longitude",
            "deviation_E/W",  # E_W in Enverus DS
            "deviation_N/S",  # N_S in Enverus DS
        },
    }

    def __init__(
        self,
        db_cfg: Optional[DatabricksConfig] = None,
        directional_source: str = "IHS", # "IHS" or "enverus"
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Parameters
        ----------
        db_cfg:
            Optional Databricks configuration for database client.
        db:
            Optional database client providing:
              - connect()
              - execute_query(sql: str) -> pd.DataFrame
              - close_connection()
        logger:
            Optional logger instance. Defaults to a class-named logger.
        """
        if db_cfg:
            self.db = SQLAlchemyDBClient(config=db_cfg, use_null_pool=True, logger=logger)
        else:
            self.db = None
        
        parent_logger = logger or logging.getLogger("well_data_loader")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.well_data_loader")
        self.logger.propagate = True
        
        self.header_df: pd.DataFrame = pd.DataFrame()
        self.directional_source = directional_source

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def get_header_data(
        self,
        source: Optional[Union[str, pd.DataFrame]] = None,
        column_map: Optional[Dict[str, str]] = None,
        dtype_map: Optional[Mapping[str, str | type]] = None,
        header_query: Optional[str] = None,
        header_params: Optional[Mapping[str, Any]] = None,
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load header data from a DataFrame, file path, or DB query.

        Parameters
        ----------
        source:
            - pd.DataFrame: use as-is (then validate)
            - str: file path (CSV/Excel)
            - None: load from DB using basin/start_year (requires self.db)
        column_map:
            REQUIRED when `source` is a file path. Must be **source → canonical**.
            Example:
                {"API14": "uwi", "LeaseName": "lease_name", ...}
        basin, start_year:
            Used only when source is None (DB query).
        dtype_map:
            Dtypes to apply *after renaming* (on canonical names).
            Example: {"uwi": "string", "surface_lat": "float64"}
        **read_kwargs:
            Extra args forwarded to pandas readers via your helper functions.
            Examples: parse_dates=..., sheet_name=..., skiprows=..., engine=...

        Returns
        -------
        pd.DataFrame
            Header data with canonical column names.

        Raises
        ------
        ValueError
            If required canonical columns are missing or inputs are invalid.
        """
        if isinstance(source, pd.DataFrame):
            self.logger.info("Using provided header DataFrame.")
            df = source.copy()

        elif isinstance(source, str) and os.path.exists(source):
            self._require_column_map_for_file(dataset="header", column_map=column_map)
            self.logger.info(f"Loading header data from file: {source}")
            df = self.load_data_from_file(
                file_path=source,
                dataset="header",
                column_map=column_map,  # type: ignore[arg-type]
                dtype_map=dtype_map,
                **read_kwargs,
            )

        elif source is None:
            if self.db is None:
                raise ValueError("source=None requires a db client (self.db is None).")
            self.logger.info("Loading header data from SQL.")
            df = self._query_header_from_db(header_query=header_query, params=header_params)

        else:
            raise ValueError("Invalid `source`. Provide a DataFrame, an existing file path, or None for DB.")

        self._validate_required_columns(
            df=df,
            required_cols=self.HEADER_REQUIRED_COLS,
            dataset_name="header",
            source_hint=str(source) if isinstance(source, str) else "dataframe/db",
        )

        self.header_df = df
        return df

    def get_directional_data(
        self,
        source: Optional[Union[str, pd.DataFrame]] = None,
        column_map: Optional[Dict[str, str]] = None,
        dtype_map: Optional[Mapping[str, str | type]] = None,
        directional_source: Optional[str] = None,
        directional_query: Optional[str] = None,
        directional_params: Optional[Mapping[str, Any]] = None,
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        """
        ...
        directional_source:
            Optional override for this call ("IHS" or "enverus"). If None, uses self.directional_source.
        """
        if isinstance(source, pd.DataFrame):
            self.logger.info("Using provided directional DataFrame.")
            df = source.copy()

        elif isinstance(source, str) and os.path.exists(source):
            self._require_column_map_for_file(dataset="directional", column_map=column_map)
            self.logger.info(f"Loading directional data from file: {source}")
            df = self.load_data_from_file(
                file_path=source,
                dataset="directional",
                column_map=column_map,  # type: ignore[arg-type]
                dtype_map=dtype_map,
                directional_source=directional_source,  # <-- ADD
                **read_kwargs,
            )

        elif source is None:
            if self.db is None:
                raise ValueError("source=None requires a db client (self.db is None).")
            self.logger.info("Loading directional data from SQL.")
            df = self._query_directional_from_db(directional_query=directional_query, params=directional_params)

        else:
            raise ValueError("Invalid `source`. Provide a DataFrame, an existing file path, or None for DB.")

        # IMPORTANT: validate against the selected schema (Set[str]), not the dict
        req = self._required_cols_for_dataset("directional", directional_source=directional_source)  # <-- ADD

        self._validate_required_columns(
            df=df,
            required_cols=req,  # <-- CHANGED
            dataset_name=f"directional ({(directional_source or self.directional_source)})",
            source_hint=str(source) if isinstance(source, str) else "dataframe/db",
        )

        return df

    def load_data_from_file(
        self,
        file_path: str | Path,
        *,
        dataset: str,
        column_map: Mapping[str, str],
        dtype_map: Optional[Mapping[str, str | type]] = None,
        directional_source: Optional[str] = None,  # <-- ADD
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        """
        ...
        directional_source:
            Only used when dataset == "directional" to select the right required set.
        """
        path = Path(file_path)

        required_cols = self._required_cols_for_dataset(dataset, directional_source=directional_source)  # <-- CHANGED
        self._assert_source_to_canonical_map(column_map=column_map, required_cols=required_cols, dataset=dataset)

        if "usecols" not in read_kwargs:
            read_kwargs["usecols"] = list(column_map.keys())

        try:
            if path.suffix.lower() == ".csv":
                df = read_csv_with_mapper(path, col_map=column_map, dtype_map=dtype_map, **read_kwargs)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = read_excel_with_mapper(path, col_map=column_map, dtype_map=dtype_map, **read_kwargs)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}. Use CSV or Excel.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load {dataset} data from file {path}: {e}")
            raise

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _require_column_map_for_file(
        self,
        *,
        dataset: str,
        column_map: Optional[Mapping[str, str]],
    ) -> None:
        """
        Ensure column_map is provided for file reads.

        Raises
        ------
        ValueError if column_map is missing.
        """
        if not column_map:
            raise ValueError(
                f"column_map must be provided when reading {dataset} data from a file, "
                "and it must be in source → canonical direction."
            )

    def _assert_source_to_canonical_map(
        self,
        *,
        column_map: Mapping[str, str],
        required_cols: Set[str],
        dataset: str,
    ) -> None:
        """
        Enforce that `column_map` is in source → canonical direction.

        Allows identity pairs (e.g., "uwi" -> "uwi") because the source file may already
        use canonical names for some columns.
        """
        keys = set(column_map.keys())
        vals = set(column_map.values())

        # Identity pairs are allowed (source column already canonical)
        identity_keys = {k for k, v in column_map.items() if k == v}

        # Only treat canonical names in keys as suspicious if they are NOT identity pairs
        suspicious = sorted((keys - identity_keys) & required_cols)
        if suspicious:
            raise ValueError(
                f"{dataset.title()} column_map appears to be in the WRONG direction.\n"
                f"Detected canonical columns in mapping keys: {suspicious}\n\n"
                "Expected: source → canonical mapping, e.g.\n"
                "    {'API14': 'uwi', 'LeaseName': 'lease_name', ...}\n\n"
                "If you currently have canonical → source, invert it like:\n"
                "    new_map = {src: canon for canon, src in old_map.items()}\n"
            )

        # Soft sanity check: do we map to at least one required canonical?
        if len(vals & required_cols) == 0:
            self.logger.warning(
                f"{dataset.title()} column_map does not map to any required canonical columns. "
                "Validation may fail later if required columns are missing."
            )

    def _validate_required_columns(
        self,
        *,
        df: pd.DataFrame,
        required_cols: Set[str],
        dataset_name: str,
        source_hint: str,
    ) -> None:
        """
        Validate the DataFrame contains all required canonical columns.

        Raises
        ------
        ValueError
            Includes:
              - required set
              - missing set
              - preview of present columns
            plus a note that missing columns can occur due to:
              (a) missing columns in the source file, or
              (b) incorrect/incomplete column_map for file reads
        """
        present = set(df.columns)
        missing = sorted(required_cols - present)
        if not missing:
            return

        required_sorted = sorted(required_cols)
        present_preview = ", ".join(list(map(str, list(df.columns)[:40])))
        if len(df.columns) > 40:
            present_preview += ", ..."

        msg = (
            f"{dataset_name.title()} data is missing required canonical columns.\n"
            f"Source: {source_hint}\n\n"
            f"Required ({len(required_sorted)}): {required_sorted}\n"
            f"Missing  ({len(missing)}): {missing}\n\n"
            f"Present columns (preview): {present_preview}\n\n"
            "Note: For file-based loads, this typically means either:\n"
            "  (1) the source file does not contain those fields, or\n"
            "  (2) your column_map does not map the source columns to these canonical names.\n"
        )
        raise ValueError(msg)

    def _required_cols_for_dataset(
        self,
        dataset: str,
        directional_source: Optional[str] = None,  # <-- ADD
    ) -> Set[str]:
        """
        Return the canonical required column set for a dataset.

        For directional data, selects from DIRECTIONAL_REQUIRED_COLS based on
        directional_source (case-insensitive).
        """
        key = dataset.strip().lower()

        if key == "header":
            return self.HEADER_REQUIRED_COLS

        if key == "directional":
            src = (directional_source or self.directional_source or "IHS").strip().lower()
            # allow case-insensitive keys even if dict uses mixed case
            lookup = {k.strip().lower(): k for k in self.DIRECTIONAL_REQUIRED_COLS.keys()}
            if src not in lookup:
                raise ValueError(
                    f"Unknown directional_source='{directional_source}'. "
                    f"Valid options: {sorted(self.DIRECTIONAL_REQUIRED_COLS.keys())}"
                )
            return self.DIRECTIONAL_REQUIRED_COLS[lookup[src]]

        raise ValueError(f"Unknown dataset='{dataset}'. Expected 'header' or 'directional'.")

    # ------------------------------------------------------------------
    # DB methods (kept close to your original)
    # ------------------------------------------------------------------
    def _query_header_from_db(
        self,
        header_query: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Query header data from the database client `self.db`.

        This method now prefers `execute_query_auto()` (from your updated
        database_manager.SQLAlchemyDBClient) because it supports:
        - auto-expanding list parameters (IN :param)
        - optional chunking + retry logic (if large lists are present)

        Backward compatibility
        ----------------------
        If the injected db client does NOT have `execute_query_auto`, this falls back
        to `execute_query`.

        Parameters
        ----------
        header_query:
            SQL SELECT statement (should already alias columns to canonical names).
        params:
            Optional bind parameters dict.

        Returns
        -------
        pd.DataFrame
            Header DataFrame.

        Raises
        ------
        ValueError
            If self.db is not configured.
        Exception
            If the query fails.
        """
        if self.db is None:
            raise ValueError("Database client is not configured (self.db is None).")

        try:
            if self.db.test_connection():
                # Prefer new auto method if available
                if hasattr(self.db, "execute_query_auto") and callable(getattr(self.db, "execute_query_auto")):
                    return self.db.execute_query_auto(  # type: ignore[attr-defined]
                        header_query,
                        params=params,
                        # header queries usually don't need chunking; defaults are fine
                        use_retry=True,
                    )

                # Backward-compatible fallback
                return self.db.execute_query(header_query, params=params)

        except Exception as e:
            self.logger.error(f"An error occurred while executing header query: {e}")
            raise

        finally:
            if self.db is not None:
                try:
                    self.db.close()
                except Exception:
                    self.logger.exception("Failed to dispose SQLAlchemy Engine")

    def _query_directional_from_db(
        self,
        directional_query: str,
        params: Optional[Mapping[str, Any]] = None,
        chunk_size: int = 256,
        chunk_if_len_ge: int = 1000,
    ) -> pd.DataFrame:
        """
        Query directional survey data from the database client `self.db`.

        This method now prefers `execute_query_auto()` because directional pulls
        commonly use large IN-lists (e.g., uwis_12), which can:
        - exceed parameter limits
        - produce very large SQL payloads
        - intermittently fail mid-stream

        `execute_query_auto()` will:
        - auto-expand list params (IN :param)
        - auto-switch to chunking when the list is large (threshold-based)
        - optionally retry (especially useful for Databricks SQL Warehouse)

        IMPORTANT about LIMIT
        ---------------------
        If your SQL contains LIMIT and chunking is used, LIMIT applies *per chunk*,
        not globally. If you need a global limit, remove LIMIT from SQL and apply
        it after concatenation (or implement a global limit in the db layer).

        Backward compatibility
        ----------------------
        If the injected db client does NOT have `execute_query_auto`, this falls back
        to `execute_query`.

        Parameters
        ----------
        directional_query:
            SQL SELECT statement (should already alias columns).
        params:
            Optional bind parameters dict. Often includes a large list for IN-clause.

        Returns
        -------
        pd.DataFrame
            Directional survey DataFrame.

        Raises
        ------
        ValueError
            If self.db is not configured.
        Exception
            If the query fails.
        """
        if self.db is None:
            raise ValueError("Database client is not configured (self.db is None).")

        try:
            if self.db.test_connection():
                # Prefer new auto method if available
                if hasattr(self.db, "execute_query_auto") and callable(getattr(self.db, "execute_query_auto")):
                    # Try to auto-detect the IN-list param if there is exactly one list-like param
                    chunk_param_name: Optional[str] = None
                    if params:
                        list_like = [k for k, v in params.items() if isinstance(v, (list, tuple, set))]
                        if len(list_like) == 1:
                            chunk_param_name = list_like[0]

                    # Log a friendly warning if LIMIT exists and we might chunk
                    if params and chunk_param_name and " limit " in directional_query.lower():
                        self.logger.warning(
                            "Directional query contains LIMIT. If chunking is used, LIMIT applies per chunk (not globally)."
                        )

                    df = self.db.execute_query_auto(  # type: ignore[attr-defined]
                        directional_query,
                        params=params,
                        chunk_param_name=chunk_param_name,  # e.g., "uwis_12"
                        chunk_size=chunk_size,
                        chunk_if_len_ge=chunk_if_len_ge,
                        use_retry=True,
                        max_retries=3,
                        backoff_base_s=0.5,
                        backoff_cap_s=8.0,
                        # Don't force sort_by here (could error if aliases differ); we sort safely below.
                        sort_by=None,
                        drop_duplicates=False,
                    )

                    # Safe post-sort to restore global ordering when chunking was used
                    # (only sort if the expected columns exist)
                    if {"uwi12", "md"}.issubset(df.columns):
                        df = df.sort_values(["uwi12", "md"], kind="mergesort").reset_index(drop=True)
                    elif {"uwi", "md"}.issubset(df.columns):
                        df = df.sort_values(["uwi", "md"], kind="mergesort").reset_index(drop=True)

                    return df

                # Backward-compatible fallback
                return self.db.execute_query(directional_query, params=params)

        except Exception as e:
            self.logger.error(f"An error occurred while executing directional query: {e}")
            raise

        finally:
            if self.db is not None:
                try:
                    self.db.close()
                except Exception:
                    self.logger.exception("Failed to dispose SQLAlchemy Engine")

# %%
class GeoSurveyProcessor:
    """
    Process directional survey data and perform geospatial transformations.

    This utility converts lat/lon to UTM (and back), computes UTM coordinates for
    trajectories either from per-row lat/lon or from surface coordinates + lateral
    displacements, filters rows after the heel point, and extracts heel/toe/mid
    locations.

    Parameters
    ----------
    log_dir : str, default="./logs"
        Directory to write logs created by `CustomLogger`.

    Notes
    -----
    - UTM EPSG codes constructed in this class assume the **Northern Hemisphere**
      (`EPSG:326XX`). For Southern Hemisphere data, adapt EPSG construction to `EPSG:327XX`.
    - Distances are handled in **feet** for UTM (internally converted to/from meters
      when interacting with projection APIs).

    Examples
    --------
    >>> geo = GeoSurveyProcessor(log_dir="./logs")
    """
    def __init__(self, logger: Optional[logging.Logger] = None,):
        """
        Initialize the processor and its logger.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance to use for logging. If not provided, a default logger is created.

        Examples
        --------
        >>> geo = GeoSurveyProcessor()
        """
        parent_logger = logger or logging.getLogger("geo_survey_processor")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.geo_survey_processor")
        self.logger.propagate = True

    def determine_utm_zone(self, longitude: float) -> int:
        """
        Determine the UTM zone from a longitude (in degrees).

        Parameters
        ----------
        longitude : float
            Longitude in degrees (−180 to 180).

        Returns
        -------
        int
            UTM zone number in [1, 60].

        Examples
        --------
        >>> GeoSurveyProcessor().determine_utm_zone(-103.5)
        13
        """
        return int((longitude + 180) / 6) + 1
        
    def convert_utm_to_latlon(self, 
                              df: pd.DataFrame, x_col: str = "x", y_col: str = "y", 
                              zone_col: str = "utm_zone", epsg_col: str = "epsg_code", 
                              lat_col: str = "latitude", lon_col: str = "longitude",
                              round_output: bool = True) -> pd.DataFrame:
        """
        Convert UTM (x, y) coordinates (in feet) back to latitude/longitude.

        Uses either an existing EPSG column or constructs one from the UTM zone
        (assuming Northern Hemisphere: `EPSG:326{zone}`).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing UTM coordinates and zone/EPSG identifiers.
        x_col, y_col : str, default=("x", "y")
            Column names for UTM easting/northing in **feet**.
        zone_col : str, default="utm_zone"
            Column containing UTM zone numbers (integer 1–60) used if `epsg_col` is missing.
        epsg_col : str, default="epsg_code"
            Column containing EPSG strings (e.g., "EPSG:32613"). If missing, it is created.
        lat_col, lon_col : str, default=("latitude", "longitude")
            Output column names for latitude and longitude (in degrees).
        round_output : bool, default=True
            Round outputs to 8 decimal places for readability.

        Returns
        -------
        pandas.DataFrame
            A copy of `df` with `lat_col` and `lon_col` populated.

        Raises
        ------
        Exception
            Any projection or data error is logged and re-raised.

        Examples
        --------
        >>> import pandas as pd
        >>> geo = GeoSurveyProcessor()
        >>> data = pd.DataFrame({"x": [2210000.0], "y": [1200000.0], "utm_zone": [13]})
        >>> out = geo.convert_utm_to_latlon(data)
        >>> set(["latitude", "longitude"]).issubset(out.columns)
        True
        """

        df = df.copy()
        
        if epsg_col not in df.columns:
            df[epsg_col] = df[zone_col].apply(lambda z: f"EPSG:326{int(z)}")

        df[lat_col] = np.nan
        df[lon_col] = np.nan

        for epsg in df[epsg_col].unique():
            mask = df[epsg_col] == epsg

            # Convert feet to meters
            x_m = df.loc[mask, x_col] / 3.28084
            y_m = df.loc[mask, y_col] / 3.28084

            transformer = pyproj.Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x_m.values, y_m.values)

            if round_output:
                lon = np.round(lon, 8)
                lat = np.round(lat, 8)

            df.loc[mask, lat_col] = lat
            df.loc[mask, lon_col] = lon

        self.logger.info(f"✅ Back-converted UTM to lat/lon for {len(df)} rows.")
        return df
    
    def compute_utm_coordinates(
        self,
        df: pd.DataFrame,
        surface_df: Optional[pd.DataFrame] = None,
        force_utm_zone: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute UTM (x, y, z) in **feet** for survey points.

        Two operation modes:

        1) **Per-row lat/lon present in `df`**:
           - Requires `df` to contain: `["uwi", "md", "tvd", "latitude", "longitude"]`.
           - UTM zone is inferred per-row from `longitude` (unless `force_utm_zone` is set).

        2) **Lat/lon missing; use `surface_df` + displacements**:
           - Requires `surface_df` with `["uwi", "surface_lat", "surface_lon"]`.
           - Also expects in `df`: `["uwi", "md", "tvd", "E/W", "N/S", "deviation_E/W", "deviation_N/S"]`.
           - UTM of the surface is computed and displacements (signed by E/W, N/S)
             are added (all in **feet**).

        Adds:
        - `x`, `y` : UTM easting/northing (feet)
        - `z`      : vertical coordinate as `-tvd` (feet, depth positive down)
        - `utm_zone`, `epsg_code` : zone and EPSG used
        - If `surface_df` path used, back-computed `latitude`, `longitude` via `convert_utm_to_latlon`.

        Parameters
        ----------
        df : pandas.DataFrame
            Directional survey data, sorted internally by `["uwi", "md"]`.
        surface_df : Optional[pandas.DataFrame], default=None
            Surface locations keyed by `uwi` if per-row lat/lon are not available.
        force_utm_zone : Optional[int], default=None
            If provided, **all** points are projected to this single UTM zone, regardless
            of their actual longitude. This is **critical** for well spacing calculations
            when data spans multiple UTM zones (e.g., near zone boundaries like -102° longitude).

            When `force_utm_zone=None` (default), each point is projected to its own zone
            based on longitude. This can cause **catastrophic errors** in downstream spacing
            calculations if wells straddle a zone boundary, as UTM coordinates from different
            zones are not directly comparable (they can differ by ~500,000+ ft).

        Returns
        -------
        pandas.DataFrame
            Data with UTM coordinates and auxiliary metadata columns.

        Raises
        ------
        ValueError
            If neither per-row lat/lon nor a valid `surface_df` are provided.
            If `force_utm_zone` is not in the valid range [1, 60].
        Exception
            Any projection or merge errors are logged and re-raised.

        Warnings
        --------
        **Multi-Zone Data Warning**: If your data spans multiple UTM zones and
        `force_utm_zone` is not set, a warning will be logged. This is especially
        important for:

        - Well spacing calculations (`WellSpacingCalculator`)
        - Floating section analysis (`FloatingSectionWPS`)
        - Any distance-based computations between wells

        The distortion introduced by forcing a single zone is negligible (<0.1% at
        worst, typically <10 ft error on a 10,000 ft lateral) and far smaller than
        survey measurement uncertainties.

        Recommended UTM Zones by US Basin
        ---------------------------------
        Use this table to select the appropriate `force_utm_zone` for your study area:

        **Permian Basin (West Texas / SE New Mexico)**

        +-----------------------+------+------------------+---------------------------+
        | Sub-basin / Area      | Zone | Central Meridian | Counties / Region         |
        +=======================+======+==================+===========================+
        | Delaware Basin        | 13   | -105°            | Loving, Winkler, Ward,    |
        |                       |      |                  | Reeves, Culberson, Pecos  |
        +-----------------------+------+------------------+---------------------------+
        | Midland Basin         | 13   | -105°            | Midland, Martin, Howard,  |
        |                       |      |                  | Glasscock, Reagan, Upton  |
        +-----------------------+------+------------------+---------------------------+
        | Central Basin Platform| 13   | -105°            | Andrews, Ector, Crane     |
        +-----------------------+------+------------------+---------------------------+
        | Full Permian Basin    | 13   | -105°            | All of above combined     |
        +-----------------------+------+------------------+---------------------------+

        **Other Major US Basins**

        +-----------------------+------+------------------+---------------------------+
        | Basin                 | Zone | Central Meridian | States / Region           |
        +=======================+======+==================+===========================+
        | Eagle Ford Shale      | 14   | -99°             | South Texas (Karnes,      |
        |                       |      |                  | DeWitt, Gonzales, La Salle|
        +-----------------------+------+------------------+---------------------------+
        | Haynesville Shale     | 15   | -93°             | East Texas, NW Louisiana  |
        +-----------------------+------+------------------+---------------------------+
        | Anadarko Basin        | 14   | -99°             | Oklahoma (STACK, SCOOP),  |
        |                       |      |                  | Texas Panhandle           |
        +-----------------------+------+------------------+---------------------------+
        | Arkoma Basin          | 15   | -93°             | Oklahoma, Arkansas        |
        +-----------------------+------+------------------+---------------------------+
        | Williston / Bakken    | 13   | -105°            | North Dakota, Montana,    |
        |                       |      |                  | Saskatchewan              |
        +-----------------------+------+------------------+---------------------------+
        | DJ Basin / Niobrara   | 13   | -105°            | Colorado (Weld County),   |
        |                       |      |                  | Wyoming, Nebraska         |
        +-----------------------+------+------------------+---------------------------+
        | Powder River Basin    | 13   | -105°            | Wyoming, Montana          |
        +-----------------------+------+------------------+---------------------------+
        | Uinta Basin           | 12   | -111°            | Utah (Duchesne, Uintah)   |
        +-----------------------+------+------------------+---------------------------+
        | Piceance Basin        | 13   | -105°            | Colorado (Garfield, Mesa) |
        +-----------------------+------+------------------+---------------------------+
        | San Juan Basin        | 12   | -111°            | New Mexico, Colorado      |
        +-----------------------+------+------------------+---------------------------+
        | Green River Basin     | 12   | -111°            | Wyoming, Utah, Colorado   |
        +-----------------------+------+------------------+---------------------------+
        | Marcellus Shale       | 17   | -81°             | Pennsylvania, West        |
        |                       |      |                  | Virginia, Ohio            |
        +-----------------------+------+------------------+---------------------------+
        | Utica Shale           | 17   | -81°             | Ohio, Pennsylvania,       |
        |                       |      |                  | West Virginia             |
        +-----------------------+------+------------------+---------------------------+
        | Appalachian Basin     | 17   | -81°             | PA, WV, OH, NY, KY        |
        +-----------------------+------+------------------+---------------------------+
        | Michigan Basin        | 16   | -87°             | Michigan                  |
        +-----------------------+------+------------------+---------------------------+
        | Illinois Basin        | 16   | -87°             | Illinois, Indiana, KY     |
        +-----------------------+------+------------------+---------------------------+
        | Fort Worth / Barnett  | 14   | -99°             | North-Central Texas       |
        +-----------------------+------+------------------+---------------------------+
        | East Texas Basin      | 15   | -93°             | East Texas                |
        +-----------------------+------+------------------+---------------------------+
        | Gulf Coast Basin      | 15   | -93°             | Texas, Louisiana Coast    |
        +-----------------------+------+------------------+---------------------------+
        | Austin Chalk          | 14   | -99°             | South-Central Texas       |
        +-----------------------+------+------------------+---------------------------+
        | Cook Inlet Basin      | 6    | -147°            | Alaska                    |
        +-----------------------+------+------------------+---------------------------+
        | North Slope / Prudhoe | 6    | -147°            | Alaska                    |
        +-----------------------+------+------------------+---------------------------+
        | San Joaquin Basin     | 10   | -123°            | California (Kern County)  |
        +-----------------------+------+------------------+---------------------------+
        | Los Angeles Basin     | 11   | -117°            | Southern California       |
        +-----------------------+------+------------------+---------------------------+
        | Ventura Basin         | 11   | -117°            | Southern California       |
        +-----------------------+------+------------------+---------------------------+

        **UTM Zone Quick Reference (Longitude Boundaries)**

        +------+------------------+------------------+------------------+
        | Zone | Western Boundary | Eastern Boundary | Central Meridian |
        +======+==================+==================+==================+
        | 10   | -126°            | -120°            | -123°            |
        +------+------------------+------------------+------------------+
        | 11   | -120°            | -114°            | -117°            |
        +------+------------------+------------------+------------------+
        | 12   | -114°            | -108°            | -111°            |
        +------+------------------+------------------+------------------+
        | 13   | -108°            | -102°            | -105°            |
        +------+------------------+------------------+------------------+
        | 14   | -102°            | -96°             | -99°             |
        +------+------------------+------------------+------------------+
        | 15   | -96°             | -90°             | -93°             |
        +------+------------------+------------------+------------------+
        | 16   | -90°             | -84°             | -87°             |
        +------+------------------+------------------+------------------+
        | 17   | -84°             | -78°             | -81°             |
        +------+------------------+------------------+------------------+
        | 18   | -78°             | -72°             | -75°             |
        +------+------------------+------------------+------------------+
        | 19   | -72°             | -66°             | -69°             |
        +------+------------------+------------------+------------------+

        Examples
        --------
        **Basic usage with per-row lat/lon (auto zone detection)**:

        >>> data = pd.DataFrame({
        ...     "uwi": ["A","A","A"],
        ...     "md": [1000, 1100, 1200],
        ...     "tvd": [900, 1000, 1100],
        ...     "latitude": [31.4, 31.4001, 31.4002],
        ...     "longitude": [-103.3, -103.3001, -103.3002],
        ... })
        >>> out = GeoSurveyProcessor().compute_utm_coordinates(data)
        >>> set(["x","y","z","utm_zone","epsg_code"]).issubset(out.columns)
        True

        **Forcing a single UTM zone for Permian Basin data** (RECOMMENDED):

        >>> # Midland Basin wells - force zone 13 to avoid cross-zone issues
        >>> out = GeoSurveyProcessor().compute_utm_coordinates(
        ...     data,
        ...     force_utm_zone=13
        ... )
        >>> out["utm_zone"].unique()  # All wells in zone 13
        array([13])

        **Eagle Ford Shale example**:

        >>> # South Texas data - use zone 14
        >>> eagle_ford_data = pd.DataFrame({
        ...     "uwi": ["EF1","EF1","EF1"],
        ...     "md": [8000, 9000, 10000],
        ...     "tvd": [7500, 8500, 9500],
        ...     "latitude": [28.5, 28.5001, 28.5002],
        ...     "longitude": [-98.2, -98.2001, -98.2002],
        ... })
        >>> out = GeoSurveyProcessor().compute_utm_coordinates(
        ...     eagle_ford_data,
        ...     force_utm_zone=14
        ... )

        **Bakken / Williston Basin example**:

        >>> # North Dakota data - use zone 13
        >>> bakken_data = pd.DataFrame({
        ...     "uwi": ["BK1","BK1","BK1"],
        ...     "md": [10000, 11000, 12000],
        ...     "tvd": [9500, 10500, 11500],
        ...     "latitude": [48.1, 48.1001, 48.1002],
        ...     "longitude": [-103.5, -103.5001, -103.5002],
        ... })
        >>> out = GeoSurveyProcessor().compute_utm_coordinates(
        ...     bakken_data,
        ...     force_utm_zone=13
        ... )

        **Using surface_df + displacements**:

        >>> df = pd.DataFrame({
        ...     "uwi": ["B","B"],
        ...     "md": [1000, 1100],
        ...     "tvd": [900, 1000],
        ...     "E/W": ["E","E"],
        ...     "N/S": ["N","N"],
        ...     "deviation_E/W": [50.0, 75.0],
        ...     "deviation_N/S": [25.0, 40.0],
        ... })
        >>> surf = pd.DataFrame({
        ...     "uwi": ["B"], "surface_lat": [31.45], "surface_lon": [-103.35]
        ... })
        >>> out = GeoSurveyProcessor().compute_utm_coordinates(
        ...     df,
        ...     surface_df=surf,
        ...     force_utm_zone=13
        ... )
        >>> set(["x","y","latitude","longitude"]).issubset(out.columns)
        True

        See Also
        --------
        determine_utm_zone : Compute UTM zone from longitude.
        convert_utm_to_latlon : Convert UTM coordinates back to lat/lon.
        WellSpacingCalculator : Spacing calculations that require consistent UTM zones.
        FloatingSectionWPS : Floating section analysis that requires consistent UTM zones.
        """
        # Validate force_utm_zone if provided
        if force_utm_zone is not None:
            if not isinstance(force_utm_zone, int) or not (1 <= force_utm_zone <= 60):
                raise ValueError(
                    f"force_utm_zone must be an integer between 1 and 60, got {force_utm_zone}"
                )

        start_time = time.time()
        df = df.sort_values(by=["uwi", "md"]).copy()

        df["x"], df["y"] = np.zeros(len(df)), np.zeros(len(df))

        if "latitude" in df.columns and "longitude" in df.columns:
            self.logger.info("✅ Using lat/lon from input DataFrame.")

            # Determine zones - either forced or per-row
            if force_utm_zone is not None:
                df["utm_zone"] = force_utm_zone
                self.logger.info(f"🌍 Forcing all data to UTM zone {force_utm_zone}.")
            else:
                df["utm_zone"] = df["longitude"].apply(self.determine_utm_zone)

                # Warn if multiple zones detected
                unique_zones = df["utm_zone"].unique()
                if len(unique_zones) > 1:
                    self.logger.warning(
                        f"⚠️ Data spans multiple UTM zones: {sorted(unique_zones)}. "
                        f"This may cause coordinate inconsistencies in downstream spacing "
                        f"calculations. Consider using force_utm_zone parameter to project "
                        f"all data to a single zone. See docstring for recommended zones by basin."
                    )

            for zone in df["utm_zone"].unique():
                epsg_code = f"EPSG:326{zone:02d}" if zone < 10 else f"EPSG:326{zone}"
                mask = df["utm_zone"] == zone

                transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
                lon = df.loc[mask, "longitude"].values
                lat = df.loc[mask, "latitude"].values
                easting_m, northing_m = transformer.transform(lon, lat)

                df.loc[mask, "x"] = easting_m * 3.28084
                df.loc[mask, "y"] = northing_m * 3.28084
                df.loc[mask, "epsg_code"] = epsg_code

        elif surface_df is not None:
            self.logger.info("🧭 Lat/Lon not available — using surface_df and displacements.")

            required_cols = {"uwi", "surface_lat", "surface_lon"}

            if not required_cols.issubset(surface_df.columns):
                raise ValueError(f"surface_df must contain {required_cols}")

            df = df.merge(surface_df, on="uwi", how="left")

            # Determine zones - either forced or per-row
            if force_utm_zone is not None:
                df["utm_zone"] = force_utm_zone
                self.logger.info(f"🌍 Forcing all data to UTM zone {force_utm_zone}.")
            else:
                df["utm_zone"] = df["surface_lon"].apply(self.determine_utm_zone)

                # Warn if multiple zones detected
                unique_zones = df["utm_zone"].unique()
                if len(unique_zones) > 1:
                    self.logger.warning(
                        f"⚠️ Data spans multiple UTM zones: {sorted(unique_zones)}. "
                        f"This may cause coordinate inconsistencies in downstream spacing "
                        f"calculations. Consider using force_utm_zone parameter to project "
                        f"all data to a single zone. See docstring for recommended zones by basin."
                    )

            for zone in df["utm_zone"].unique():
                epsg_code = f"EPSG:326{zone:02d}" if zone < 10 else f"EPSG:326{zone}"
                mask = df["utm_zone"] == zone

                transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
                lon = df.loc[mask, "surface_lon"].values
                lat = df.loc[mask, "surface_lat"].values
                easting_m, northing_m = transformer.transform(lon, lat)

                # Convert to feet
                easting_ft = easting_m * 3.28084
                northing_ft = northing_m * 3.28084

                ew_sign = df.loc[mask, "E/W"].map({"E": 1, "W": -1}).fillna(0)
                ns_sign = df.loc[mask, "N/S"].map({"N": 1, "S": -1}).fillna(0)

                df.loc[mask, "x"] = easting_ft + df.loc[mask, "deviation_E/W"] * ew_sign
                df.loc[mask, "y"] = northing_ft + df.loc[mask, "deviation_N/S"] * ns_sign
                df.loc[mask, "epsg_code"] = epsg_code

            # Back-convert to lat/lon using x/y and epsg_code
            df = self.convert_utm_to_latlon(df)

        else:
            raise ValueError("Either lat/lon must be present in df, or surface_df must be provided.")

        df["z"] = -df["tvd"]

        self.logger.info(f"✅ UTM coordinate computation complete in {time.time() - start_time:.2f} sec.")
        return df
    
    def filter_after_heel_point(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep rows at/after the first high-inclination point per UWI.

        For each `uwi`, this method identifies the **first** survey station where
        `inclination >= 80` degrees (the "heel trigger") and then retains that row
        and all *subsequent* rows for that `uwi`. Wells that never reach
        `inclination >= 80` are dropped entirely.

        Implementation details
        ----------------------
        1. The input DataFrame is first sorted by ['uwi', 'md'] in ascending order
           and its index is reset to a simple RangeIndex (0..N-1). This ensures
           that index labels and row positions coincide.

        2. Among rows where `inclination >= 80`, the method finds, for each `uwi`,
           the index (row position) of the *first* such row.

        3. For every row, it then compares its position to the corresponding
           start index for its `uwi`. Rows whose position is **greater than or
           equal to** the heel start index are kept; rows before that point are
           dropped.

        Parameters
        ----------
        df : pandas.DataFrame
            Directional survey data with at least the following columns:
            - 'uwi'         : well identifier
            - 'md'          : measured depth (used for sorting within each UWI)
            - 'inclination' : inclination in degrees

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame containing only rows at/after the heel point for
            each `uwi` that reaches `inclination >= 80`. The result is:
              - sorted by ['uwi', 'md'] ascending, and
              - index reset to a simple RangeIndex.

        Notes
        -----
        - If a given `uwi` has **no** row with `inclination >= 80`, all of its
          rows are dropped from the output.
        - Because the index is explicitly reset before computing the heel
          positions, the logic is robust to any original index (e.g., API/uwi
          stored as the index).

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "uwi": ["A","A","A","B","B"],
        ...     "md": [1000,1100,1200,900,1000],
        ...     "inclination": [10, 82, 90, 79, 85],
        ... })
        >>> geo = GeoSurveyProcessor()
        >>> out = geo.filter_after_heel_point(df)
        >>> out
          uwi    md  inclination
        0   A  1100         82.0
        1   A  1200         90.0
        2   B  1000         85.0

        >>> out.groupby("uwi").size().to_dict()
        {'A': 2, 'B': 1}
        """
        # Sort and reset index so that labels == positions
        df = df.sort_values(by=["uwi", "md"], ascending=True).reset_index(drop=True)

        mask = (df["inclination"] >= 80)

        # If no heel trigger at all, drop everything
        if not mask.any():
            self.logger.warning("No inclination >= 80 found for any UWI; returning empty DataFrame.")
            return df.iloc[0:0].copy()

        idx_start = df[mask].groupby('uwi', sort=False).head(1).index
        start_idx_map = dict(zip(df.loc[idx_start, 'uwi'], idx_start))

        uwis = df['uwi'].values
        indices = np.arange(len(df))

        start_indices = np.vectorize(start_idx_map.get, otypes=[float])(uwis)
        valid_rows = indices >= start_indices

        out = df[valid_rows].reset_index(drop=True)
        return out
    
    def get_heel_toe_midpoints_latlon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract heel, toe, and geometric mid-point (lat/lon) per UWI.

        The input should represent the lateral section (or at least include the heel
        indicator). Internally, this method first filters rows to at/after the heel
        indicator via `filter_after_heel_point()`, then computes:
          - heel : first row's (lat, lon)
          - toe  : last row's (lat, lon)
          - mid  : arithmetic mean of heel and toe lat/lon (geometric midpoint)

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain: ['uwi', 'md', 'latitude', 'longitude', 'point_type_name'].

        Returns
        -------
        pandas.DataFrame
            Columns:
            ['uwi', 'heel_lat', 'heel_lon', 'toe_lat', 'toe_lon', 'mid_Lat', 'mid_Lon'].

        Examples
        --------
        >>> data = {
        ...     "uwi": [1001,1001,1001,1002,1002],
        ...     "md": [5000,5100,5200,6000,6100],
        ...     "point_type_name": ["heel","lateral","lateral","heel","lateral"],
        ...     "latitude": [31.388,31.389,31.387,31.400,31.401],
        ...     "longitude": [-103.314,-103.315,-103.316,-103.318,-103.319],
        ... }
        >>> df = pd.DataFrame(data)
        >>> out = GeoSurveyProcessor().get_heel_toe_midpoints_latlon(df)
        >>> set(["heel_lat","heel_lon","toe_lat","toe_lon","mid_Lat","mid_Lon"]).issubset(out.columns)
        True
        """
        # Getting DataFrame with only the rows after the heel point
        df = self.filter_after_heel_point(df)

        # Group by 'uwi' and extract heel/toe lat/lon
        heel_toe_df = (
            df.groupby("uwi")
            .agg(
                heel_lat=("latitude", "first"),
                heel_lon=("longitude", "first"),
                toe_lat=("latitude", "last"),
                toe_lon=("longitude", "last"),
            )
            .reset_index()
        )

        # Calculate midpoints
        heel_toe_df["mid_Lat"] = (heel_toe_df["heel_lat"] + heel_toe_df["toe_lat"]) / 2
        heel_toe_df["mid_Lon"] = (heel_toe_df["heel_lon"] + heel_toe_df["toe_lon"]) / 2

        return heel_toe_df
    
    def plot_utm_trajectory(
        self,
        df: pd.DataFrame,
        plot_3d: bool = True,
        uwis: Optional[Union[list, str]] = None
    ) -> None:
        """
        Visualize well trajectories in UTM coordinates (2D plan view or 3D with depth).

        The figure shows one line per well (grouped by `uwi`). For 3D plots, the Z-axis
        uses `z = -tvd` (feet), so deeper points appear with more negative values.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing at least:
            - 'uwi' : well identifier (str/int)
            - 'x'   : UTM easting in feet
            - 'y'   : UTM northing in feet
            - 'z'   : vertical coordinate in feet (only required when `plot_3d=True`)
        plot_3d : bool, default=True
            If True, render a 3D plot (x, y, z). If False, render a 2D plan view (x, y).
        uwis : Optional[Union[str, int, List[Union[str, int]]]], default=None
            One well ID (str/int) or a list of well IDs to filter before plotting.
            If None, all wells in `df` are plotted.

        Returns
        -------
        None
            Displays a matplotlib figure and returns None.

        Raises
        ------
        ValueError
            If required columns are missing, or if no rows remain after filtering.

        Examples
        --------
        Plot all wells in 3D:

        >>> geo = GeoSurveyProcessor()
        >>> geo.plot_utm_trajectory(df, plot_3d=True)

        Plot only two wells in 2D:

        >>> geo.plot_utm_trajectory(df, plot_3d=False, uwis=["3505123519", "3505123520"])
        """
        if uwis is not None:
            if isinstance(uwis, str):
                uwis = [uwis]
            df = df[df["uwi"].isin(uwis)]

        fig = plt.figure(figsize=(10, 10))
        title = "3D Well Trajectory (UTM ft)" if plot_3d else "2D Well Plan View (x-y, UTM ft)"
        fig.suptitle(title, fontsize=14)

        if plot_3d:
            ax = fig.add_subplot(111, projection='3d')
            for uwi, group in df.groupby("uwi"):
                ax.plot(group["x"], group["y"], group["z"], label=str(uwi))
            ax.set_zlabel("Z (ft, -TVD)")
        else:
            ax = fig.add_subplot(111)
            for uwi, group in df.groupby("uwi"):
                ax.plot(group["x"], group["y"], label=str(uwi))
        
        ax.set_xlabel("X (Easting, ft)")
        ax.set_ylabel("Y (Northing, ft)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()