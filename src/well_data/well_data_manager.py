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
        Initialize WellDataLoader with optional database connection and logging.

        This constructor sets up the loader for well data ingestion from multiple
        sources (files, database, or in-memory DataFrames). Database connectivity
        is optional - if no db_cfg is provided, only file-based and DataFrame-based
        loading will be available.

        The directional_source parameter determines which schema validation will be
        applied to directional survey data (IHS vs Enverus have different required
        column sets).

        Parameters
        ----------
        db_cfg : Optional[DatabricksConfig], default=None
            Databricks configuration for database client. If provided, creates an
            SQLAlchemyDBClient with connection pooling disabled (use_null_pool=True)
            for compatibility with Databricks SQL Warehouse. If None, database
            query methods will raise ValueError when called.
        directional_source : str, default="IHS"
            Data source identifier for directional survey validation. Valid options:
            - "IHS": Uses IHS schema (requires deviation_E/W, E/W, deviation_N/S, N/S)
            - "enverus": Uses Enverus schema (requires deviation_E/W, deviation_N/S only)
            Case-insensitive when used in validation methods.
        logger : Optional[logging.Logger], default=None
            Parent logger instance for logging operations. If None, creates a default
            logger named "well_data_loader". If provided, creates a child logger with
            name "{parent_logger.name}.well_data_loader" to maintain hierarchical
            logging structure. Logger propagation is enabled for all loggers.

        Attributes
        ----------
        db : Optional[SQLAlchemyDBClient]
            Database client instance (None if db_cfg not provided).
        logger : logging.Logger
            Logger instance for this loader.
        header_df : pd.DataFrame
            Placeholder for header data (initialized as empty DataFrame).
        directional_source : str
            Stored directional source identifier.

        Examples
        --------
        Initialize with database connection for Databricks:

        >>> from src.utils.database_manager import DatabricksConfig
        >>> db_config = DatabricksConfig(
        ...     server_hostname="your-workspace.databricks.com",
        ...     http_path="/sql/1.0/warehouses/abc123",
        ...     access_token="dapi..."
        ... )
        >>> loader = WellDataLoader(db_cfg=db_config, directional_source="enverus")

        Initialize for file-only loading (no database):

        >>> loader = WellDataLoader()  # Uses default directional_source="IHS"

        Initialize with custom logger:

        >>> import logging
        >>> parent_log = logging.getLogger("my_app")
        >>> loader = WellDataLoader(logger=parent_log)
        >>> # Creates child logger: "my_app.well_data_loader"

        Notes
        -----
        - The SQLAlchemyDBClient is created with use_null_pool=True to avoid
          connection pooling issues with Databricks SQL Warehouse.
        - Logger propagation is always enabled, allowing log messages to bubble
          up to parent handlers.
        - The directional_source can be changed after initialization by modifying
          the instance attribute, but it's recommended to set it during construction.
        """
        # Initialize database client if configuration provided
        if db_cfg:
            self.db = SQLAlchemyDBClient(config=db_cfg, use_null_pool=True, logger=logger)
            if logger:
                logger.info("WellDataLoader initialized with database connection")
            else:
                logging.getLogger("well_data_loader").info("WellDataLoader initialized with database connection")
        else:
            self.db = None
            if logger:
                logger.info("WellDataLoader initialized without database connection (file/DataFrame mode only)")
            else:
                logging.getLogger("well_data_loader").info("WellDataLoader initialized without database connection")

        # Set up hierarchical logger
        parent_logger = logger or logging.getLogger("well_data_loader")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.well_data_loader")
        self.logger.propagate = True

        # Initialize instance attributes
        self.header_df: pd.DataFrame = pd.DataFrame()
        self.directional_source = directional_source

        self.logger.debug(f"Directional source set to: {directional_source}")
        self.logger.debug(f"Logger hierarchy: {self.logger.name}")

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
        self.logger.info("=" * 60)
        self.logger.info("LOADING HEADER DATA")
        self.logger.info("=" * 60)

        if isinstance(source, pd.DataFrame):
            self.logger.info("Source: Provided DataFrame (in-memory)")
            self.logger.debug(f"DataFrame shape: {source.shape} (rows × columns)")
            self.logger.debug(f"DataFrame columns: {list(source.columns)}")
            df = source.copy()

        elif isinstance(source, str) and os.path.exists(source):
            self.logger.info(f"Source: File - {source}")
            self.logger.debug(f"Column mappings: {len(column_map or {})} mappings provided")
            self._require_column_map_for_file(dataset="header", column_map=column_map)
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
            self.logger.info("Source: Database query")
            has_query = header_query is not None
            has_params = header_params is not None
            self.logger.debug(f"Query provided: {has_query}, Parameters provided: {has_params}")
            df = self._query_header_from_db(header_query=header_query, params=header_params)

        else:
            raise ValueError("Invalid `source`. Provide a DataFrame, an existing file path, or None for DB.")

        # Log data dimensions before validation
        self.logger.debug(f"Pre-validation shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Validate required columns
        self._validate_required_columns(
            df=df,
            required_cols=self.HEADER_REQUIRED_COLS,
            dataset_name="header",
            source_hint=str(source) if isinstance(source, str) else "dataframe/db",
        )

        # Store and log success
        self.header_df = df
        self.logger.info(f"✓ Header data loaded successfully: {len(df):,} wells")
        self.logger.debug(f"Final columns: {list(df.columns)}")
        self.logger.info("=" * 60)

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
        Load directional survey data from a DataFrame, file path, or DB query.

        Parameters
        ----------
        source:
            - pd.DataFrame: use as-is (then validate)
            - str: file path (CSV/Excel)
            - None: load from DB query (requires self.db)
        column_map:
            REQUIRED when `source` is a file path. Must be **source → canonical**.
            Example:
                {"UWI": "uwi", "MD": "md", "TVD": "tvd", ...}
        dtype_map:
            Dtypes to apply *after renaming* (on canonical names).
            Example: {"uwi": "string", "md": "float64"}
        directional_source:
            Optional override for this call ("IHS" or "enverus"). If None, uses self.directional_source.
        directional_query:
            SQL query string (used only when source is None).
        directional_params:
            Query parameters dict (used only when source is None).
        **read_kwargs:
            Extra args forwarded to pandas readers via your helper functions.
            Examples: parse_dates=..., sheet_name=..., skiprows=..., engine=...

        Returns
        -------
        pd.DataFrame
            Directional survey data with canonical column names.

        Raises
        ------
        ValueError
            If required canonical columns are missing or inputs are invalid.
        """
        # Determine which directional schema to use
        active_source = directional_source or self.directional_source

        self.logger.info("=" * 60)
        self.logger.info("LOADING DIRECTIONAL SURVEY DATA")
        self.logger.info("=" * 60)
        self.logger.debug(f"Directional source schema: {active_source}")

        if isinstance(source, pd.DataFrame):
            self.logger.info("Source: Provided DataFrame (in-memory)")
            self.logger.debug(f"DataFrame shape: {source.shape} (rows × columns)")
            self.logger.debug(f"DataFrame columns: {list(source.columns)}")
            df = source.copy()

        elif isinstance(source, str) and os.path.exists(source):
            self.logger.info(f"Source: File - {source}")
            self.logger.debug(f"Column mappings: {len(column_map or {})} mappings provided")
            self._require_column_map_for_file(dataset="directional", column_map=column_map)
            df = self.load_data_from_file(
                file_path=source,
                dataset="directional",
                column_map=column_map,  # type: ignore[arg-type]
                dtype_map=dtype_map,
                directional_source=directional_source,
                **read_kwargs,
            )

        elif source is None:
            if self.db is None:
                raise ValueError("source=None requires a db client (self.db is None).")
            self.logger.info("Source: Database query")
            has_query = directional_query is not None
            has_params = directional_params is not None
            self.logger.debug(f"Query provided: {has_query}, Parameters provided: {has_params}")
            df = self._query_directional_from_db(directional_query=directional_query, params=directional_params)

        else:
            raise ValueError("Invalid `source`. Provide a DataFrame, an existing file path, or None for DB.")

        # Log data dimensions before validation
        self.logger.debug(f"Pre-validation shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # IMPORTANT: validate against the selected schema (Set[str]), not the dict
        req = self._required_cols_for_dataset("directional", directional_source=directional_source)

        self._validate_required_columns(
            df=df,
            required_cols=req,
            dataset_name=f"directional ({active_source})",
            source_hint=str(source) if isinstance(source, str) else "dataframe/db",
        )

        # Log success with survey point count
        unique_wells = df["uwi"].nunique() if "uwi" in df.columns else (
            df["uwi12"].nunique() if "uwi12" in df.columns else "unknown"
        )
        self.logger.info(
            f"✓ Directional data loaded successfully: {len(df):,} survey points "
            f"across {unique_wells} wells"
        )
        self.logger.debug(f"Final columns: {list(df.columns)}")
        self.logger.info("=" * 60)

        return df

    def load_data_from_file(
        self,
        file_path: str | Path,
        *,
        dataset: str,
        column_map: Mapping[str, str],
        dtype_map: Optional[Mapping[str, str | type]] = None,
        directional_source: Optional[str] = None,
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load well data from a CSV or Excel file using source-to-canonical column mapping.

        This method handles the complete file loading pipeline:
        1. Validates column mapping direction (source → canonical)
        2. Reads file using appropriate reader (CSV/Excel)
        3. Applies column renaming and dtype conversions
        4. Returns DataFrame with canonical column names

        Parameters
        ----------
        file_path : str or Path
            Path to the data file (CSV or Excel format).
        dataset : str
            Dataset type identifier: "header" or "directional".
            Used for validation and error messages.
        column_map : Mapping[str, str]
            Mapping from source column names to canonical names.
            Example: {"API14": "uwi", "LeaseName": "lease_name"}
            Direction: source_column → canonical_column
        dtype_map : Mapping[str, str or type], optional
            Data types to apply AFTER renaming (uses canonical names).
            Example: {"uwi": "string", "surface_lat": "float64"}
        directional_source : str, optional
            Data source identifier for directional data ("IHS" or "enverus").
            Only used when dataset=="directional" to select appropriate required columns.
        **read_kwargs : Any
            Additional arguments passed to pandas readers (e.g., sheet_name, skiprows).

        Returns
        -------
        pd.DataFrame
            Loaded data with canonical column names applied.

        Raises
        ------
        ValueError
            - If file type is not CSV or Excel
            - If column mapping validation fails
        FileNotFoundError
            If the specified file does not exist
        Exception
            Any error during file reading or processing

        Examples
        --------
        Load header data from CSV:

        >>> loader = WellDataLoader()
        >>> column_map = {"API14": "uwi", "LeaseName": "lease_name"}
        >>> df = loader.load_data_from_file(
        ...     "data/header.csv",
        ...     dataset="header",
        ...     column_map=column_map
        ... )

        Load directional data from Excel with sheet specification:

        >>> dir_map = {"UWI": "uwi", "MD": "md", "TVD": "tvd"}
        >>> df = loader.load_data_from_file(
        ...     "data/surveys.xlsx",
        ...     dataset="directional",
        ...     column_map=dir_map,
        ...     directional_source="IHS",
        ...     sheet_name="Survey Data"
        ... )

        See Also
        --------
        get_header_data : High-level method for loading header data
        get_directional_data : High-level method for loading directional data
        """
        path = Path(file_path)

        self.logger.info(f"Starting file load: {path.name} ({dataset} dataset)")
        self.logger.debug(f"  Full path: {path.absolute()}")
        self.logger.debug(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        self.logger.debug(f"  Mapping {len(column_map)} source columns to canonical names")

        # Validate column mapping direction
        required_cols = self._required_cols_for_dataset(dataset, directional_source=directional_source)
        self._assert_source_to_canonical_map(column_map=column_map, required_cols=required_cols, dataset=dataset)

        # Optimize read by specifying columns upfront
        if "usecols" not in read_kwargs:
            read_kwargs["usecols"] = list(column_map.keys())
            self.logger.debug(f"  Reading only {len(column_map)} specified columns")

        start_time = time.time()

        try:
            # Read file based on extension
            if path.suffix.lower() == ".csv":
                self.logger.debug("  Using CSV reader...")
                df = read_csv_with_mapper(path, col_map=column_map, dtype_map=dtype_map, **read_kwargs)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                sheet_name = read_kwargs.get("sheet_name", 0)
                self.logger.debug(f"  Using Excel reader (sheet: {sheet_name})...")
                df = read_excel_with_mapper(path, col_map=column_map, dtype_map=dtype_map, **read_kwargs)
            else:
                error_msg = f"Unsupported file type: {path.suffix}. Only CSV (.csv) and Excel (.xlsx, .xls) are supported."
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Log success metrics
            elapsed = time.time() - start_time
            self.logger.info(f"✅ Successfully loaded {len(df):,} rows × {len(df.columns)} columns in {elapsed:.2f}s")
            self.logger.debug(f"  Canonical columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")

            return df

        except FileNotFoundError as e:
            self.logger.error(f"❌ File not found: {path}")
            self.logger.error(f"   Searched at: {path.absolute()}")
            raise

        except pd.errors.ParserError as e:
            self.logger.error(f"❌ Failed to parse {dataset} file: {path.name}")
            self.logger.error(f"   Parser error: {e}")
            self.logger.error(f"   Hint: Check file format, encoding, or delimiters")
            raise

        except KeyError as e:
            self.logger.error(f"❌ Column mapping error for {dataset} file: {path.name}")
            self.logger.error(f"   Missing source column: {e}")
            self.logger.error(f"   Hint: Verify column_map keys match actual column names in the file")
            raise

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"❌ Failed to load {dataset} data from {path.name} after {elapsed:.2f}s")
            self.logger.error(f"   Error type: {type(e).__name__}")
            self.logger.error(f"   Error message: {e}")
            self.logger.exception("Full traceback:")
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
        Validate that column_map is provided for file-based data loading.

        This check enforces that users provide explicit column mappings when loading
        from files, preventing accidental use of files with non-canonical column names.
        DataFrame and database sources can optionally skip this check if they already
        use canonical names.

        Parameters
        ----------
        dataset : str
            Dataset identifier ("header" or "directional") for error messages.
        column_map : Optional[Mapping[str, str]]
            Column mapping dictionary (source → canonical). If None or empty, raises ValueError.

        Raises
        ------
        ValueError
            If column_map is None or empty. Error message reminds user that mapping
            must be in source → canonical direction.

        Examples
        --------
        Valid call (column_map provided):

        >>> loader._require_column_map_for_file(
        ...     dataset="header",
        ...     column_map={"API14": "uwi", "LeaseName": "lease_name"}
        ... )
        # No error

        Invalid call (no column_map):

        >>> loader._require_column_map_for_file(
        ...     dataset="header",
        ...     column_map=None
        ... )
        # Raises ValueError

        Notes
        -----
        This method is called internally by get_header_data() and get_directional_data()
        before attempting to load from file sources.
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
        Enforce that column_map follows source → canonical direction convention.

        This validation prevents a common error where users accidentally provide mappings
        in the wrong direction (canonical → source instead of source → canonical).
        The check examines mapping keys for canonical column names - if found (and not
        identity pairs), it assumes the mapping is backwards and raises an error with
        correction guidance.

        Identity pairs like {"uwi": "uwi"} are explicitly allowed, as source files may
        already use canonical names for some columns.

        Parameters
        ----------
        column_map : Mapping[str, str]
            Column mapping dictionary to validate. Expected format:
            {"SourceColumnName": "canonical_name", ...}
        required_cols : Set[str]
            Set of canonical column names for this dataset type. Used to detect
            if canonical names appear in mapping keys (wrong direction).
        dataset : str
            Dataset identifier ("header" or "directional") for error messages.

        Raises
        ------
        ValueError
            If canonical column names are detected in mapping keys (excluding identity
            pairs). Error message includes:
            - List of suspicious canonical names found in keys
            - Example of correct mapping direction
            - Python code to invert the mapping

        Warnings
        --------
        Logs WARNING if column_map doesn't map to ANY required canonical columns,
        which suggests potential mapping errors or incomplete coverage.

        Examples
        --------
        Valid mapping (source → canonical):

        >>> loader._assert_source_to_canonical_map(
        ...     column_map={"API14": "uwi", "LeaseName": "lease_name"},
        ...     required_cols={"uwi", "lease_name", "bench"},
        ...     dataset="header"
        ... )
        # No error

        Valid identity mapping (source already canonical):

        >>> loader._assert_source_to_canonical_map(
        ...     column_map={"uwi": "uwi", "LeaseName": "lease_name"},
        ...     required_cols={"uwi", "lease_name"},
        ...     dataset="header"
        ... )
        # No error - identity pairs allowed

        Invalid mapping (canonical → source, wrong direction):

        >>> loader._assert_source_to_canonical_map(
        ...     column_map={"uwi": "API14", "lease_name": "LeaseName"},
        ...     required_cols={"uwi", "lease_name", "bench"},
        ...     dataset="header"
        ... )
        # Raises ValueError: "Header column_map appears to be in the WRONG direction..."

        Notes
        -----
        - This check helps maintain consistency with pd.DataFrame.rename() semantics
        - Identity pairs {"col": "col"} are explicitly allowed for flexibility
        - The WARNING about zero required columns mapped is a soft check and doesn't
          raise an error, as it might be intentional for optional columns
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
        Validate that DataFrame contains all required canonical columns.

        This method performs a critical validation step after data loading to ensure
        downstream pipelines receive data with the expected schema. Validation failures
        typically indicate either missing source data or incomplete column mapping.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate (should have canonical column names).
        required_cols : Set[str]
            Set of canonical column names that MUST be present.
        dataset_name : str
            Dataset identifier for error messages ("header", "directional").
        source_hint : str
            Description of data source for error messages (file path, "dataframe", "db").

        Raises
        ------
        ValueError
            If any required canonical columns are missing. Error message includes:
            - Full required column set
            - Specific missing columns
            - Preview of columns that ARE present
            - Troubleshooting guidance with examples

        Notes
        -----
        This validation occurs AFTER column renaming (source → canonical), so all
        required_cols should be in canonical form.

        Common causes of validation failure:
        1. Source file/query missing expected fields
        2. Incomplete column_map (missing mappings for required fields)
        3. Typos in canonical names in column_map values
        4. Wrong directional_source specified (IHS vs Enverus have different schemas)

        Examples
        --------
        Successful validation (no error):

        >>> df = pd.DataFrame({"uwi": ["123"], "lease_name": ["A"], "bench": ["B"]})
        >>> required = {"uwi", "lease_name", "bench"}
        >>> loader._validate_required_columns(
        ...     df=df,
        ...     required_cols=required,
        ...     dataset_name="header",
        ...     source_hint="test.csv"
        ... )
        # No error raised

        Failed validation (raises ValueError):

        >>> df = pd.DataFrame({"uwi": ["123"], "lease_name": ["A"]})  # missing 'bench'
        >>> required = {"uwi", "lease_name", "bench"}
        >>> loader._validate_required_columns(
        ...     df=df,
        ...     required_cols=required,
        ...     dataset_name="header",
        ...     source_hint="test.csv"
        ... )
        # Raises ValueError with detailed guidance

        See Also
        --------
        _required_cols_for_dataset : Returns required columns for dataset type
        _assert_source_to_canonical_map : Validates column_map direction
        """
        present = set(df.columns)
        missing = sorted(required_cols - present)

        # Validation passed
        if not missing:
            self.logger.debug(f"✓ {dataset_name.title()} data has all {len(required_cols)} required columns")
            return

        # Validation failed - log detailed error and raise
        required_sorted = sorted(required_cols)
        present_preview = ", ".join(list(map(str, list(df.columns)[:40])))
        if len(df.columns) > 40:
            present_preview += ", ..."

        # Generate example mappings for missing columns
        example_mappings = []
        for col in missing[:5]:  # Show examples for first 5 missing columns
            # Suggest common source column patterns
            if col == "uwi":
                example_mappings.append('  "API14": "uwi"  or  "UWI": "uwi"  or  "API_NUMBER": "uwi"')
            elif col == "lease_name":
                example_mappings.append('  "LeaseName": "lease_name"  or  "Lease": "lease_name"')
            elif col == "well_name":
                example_mappings.append('  "WellName": "well_name"  or  "Well_Name": "well_name"')
            elif col == "md":
                example_mappings.append('  "MD": "md"  or  "MeasuredDepth": "md"  or  "MD_FT": "md"')
            elif col == "tvd":
                example_mappings.append('  "TVD": "tvd"  or  "TrueVerticalDepth": "tvd"  or  "TVD_FT": "tvd"')
            else:
                # Generic example
                example_mappings.append(f'  "<YourSourceColumn>": "{col}"')

        example_section = "\n".join(example_mappings) if example_mappings else "  (Add mappings for missing columns)"

        # Build comprehensive error message
        error_msg = (
            f"{'='*80}\n"
            f"❌ VALIDATION ERROR: Missing Required Columns\n"
            f"{'='*80}\n"
            f"Dataset: {dataset_name.title()}\n"
            f"Source:  {source_hint}\n\n"
            f"Required columns ({len(required_sorted)}): {required_sorted}\n"
            f"Missing  columns ({len(missing)}):  {missing}\n\n"
            f"Present columns ({len(present)}):\n  {present_preview}\n\n"
            f"{'─'*80}\n"
            f"TROUBLESHOOTING GUIDE\n"
            f"{'─'*80}\n\n"
            f"For file-based loads, this error typically means:\n\n"
            f"1. SOURCE FILE MISSING FIELDS\n"
            f"   → Check that your source file contains data for these columns\n"
            f"   → Open the file and verify column headers\n\n"
            f"2. INCOMPLETE column_map\n"
            f"   → Your column_map needs to map source columns to these canonical names\n"
            f"   → Add mappings like:\n\n"
            f"{example_section}\n\n"
            f"3. WRONG DIRECTION in column_map\n"
            f"   → Ensure mapping is: source_column → canonical_column\n"
            f"   → NOT: canonical_column → source_column\n\n"
            f"4. TYPOS in canonical names\n"
            f"   → Verify canonical column names exactly match: {missing}\n\n"
            f"For directional data with wrong source:\n"
            f"   → If you specified directional_source='IHS' but data is from Enverus (or vice versa),\n"
            f"     the required column set will mismatch\n\n"
            f"{'='*80}\n"
        )

        # Log error before raising
        self.logger.error(error_msg)

        raise ValueError(error_msg)

    def _required_cols_for_dataset(
        self,
        dataset: str,
        directional_source: Optional[str] = None,
    ) -> Set[str]:
        """
        Retrieve the required canonical column set for a dataset type.

        This method provides schema information for validation by returning the
        appropriate set of required columns based on dataset type. For header data,
        returns a fixed column set. For directional data, returns a schema-specific
        column set based on the data source (IHS vs Enverus).

        Parameters
        ----------
        dataset : str
            Dataset type identifier. Valid options: "header" or "directional"
            (case-insensitive after stripping whitespace).
        directional_source : Optional[str], default=None
            Data source identifier for directional datasets. Valid options:
            - "IHS": IHS Markit schema (requires E/W and N/S direction columns)
            - "enverus": Enverus schema (deviation columns only, no direction columns)
            Case-insensitive matching supported. If None, uses self.directional_source
            for directional datasets. Ignored for header datasets.

        Returns
        -------
        Set[str]
            Set of canonical column names that must be present in the dataset.
            For header: self.HEADER_REQUIRED_COLS
            For directional: schema-specific set from self.DIRECTIONAL_REQUIRED_COLS

        Raises
        ------
        ValueError
            - If dataset is not "header" or "directional"
            - If directional_source is not a valid key in DIRECTIONAL_REQUIRED_COLS

        Examples
        --------
        Get required columns for header data:

        >>> loader = WellDataLoader()
        >>> cols = loader._required_cols_for_dataset("header")
        >>> print(cols)
        {'uwi', 'lease_name', 'well_name', 'operator', 'bench', ...}

        Get required columns for IHS directional data:

        >>> cols = loader._required_cols_for_dataset("directional", directional_source="IHS")
        >>> print(cols)
        {'uwi', 'md', 'tvd', 'inclination', 'azimuth', 'E/W', 'N/S', ...}

        Get required columns for Enverus directional data:

        >>> cols = loader._required_cols_for_dataset("directional", directional_source="enverus")
        >>> print(cols)
        {'uwi12', 'md', 'tvd', 'inclination', 'azimuth', 'deviation_E/W', ...}

        Case-insensitive dataset matching:

        >>> cols = loader._required_cols_for_dataset("  HEADER  ")
        # Returns header columns (whitespace stripped, case-insensitive)

        Notes
        -----
        - Case-insensitive matching applies to both dataset and directional_source
        - For directional data without specified source, falls back to:
          1. directional_source parameter (if provided)
          2. self.directional_source (instance attribute)
          3. "IHS" (default)
        - The method performs lookup with case-insensitive key matching even if
          DIRECTIONAL_REQUIRED_COLS uses mixed-case keys
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

    def _sanitize_params_for_logging(self, params: Mapping[str, Any]) -> str:
        """
        Create a sanitized summary of query parameters for logging.

        Prevents log bloat by truncating large lists and avoiding exposure of
        potentially sensitive data in logs.

        Parameters
        ----------
        params : Mapping[str, Any]
            Dictionary of query parameters (typically bind parameters for SQL).

        Returns
        -------
        str
            Human-readable summary string suitable for logging.

        Examples
        --------
        >>> params = {"uwi": "12345", "uwis_12": [1, 2, 3, ..., 1000]}
        >>> _sanitize_params_for_logging(params)
        "{'uwi': '12345', 'uwis_12': <list with 1000 items>}"
        """
        sanitized = {}
        for key, val in params.items():
            if isinstance(val, (list, tuple, set)):
                sanitized[key] = f"<{type(val).__name__} with {len(val)} items>"
            elif isinstance(val, str) and len(val) > 100:
                sanitized[key] = f"{val[:100]}... <truncated, total length: {len(val)}>"
            else:
                sanitized[key] = val
        return str(sanitized)

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
        import time

        if self.db is None:
            raise ValueError("Database client is not configured (self.db is None).")

        # Log query initiation with sanitized parameter info
        param_summary = self._sanitize_params_for_logging(params) if params else "None"
        query_preview = header_query[:200].replace("\n", " ") + ("..." if len(header_query) > 200 else "")

        self.logger.info(f"Querying header data from database")
        self.logger.debug(f"Query preview: {query_preview}")
        self.logger.debug(f"Parameters: {param_summary}")

        start_time = time.time()

        try:
            # Test database connection
            self.logger.debug("Testing database connection...")
            if self.db.test_connection():
                self.logger.debug("Database connection successful")

                # Prefer new auto method if available
                if hasattr(self.db, "execute_query_auto") and callable(getattr(self.db, "execute_query_auto")):
                    self.logger.debug("Using execute_query_auto() method with retry enabled")
                    df = self.db.execute_query_auto(  # type: ignore[attr-defined]
                        header_query,
                        params=params,
                        # header queries usually don't need chunking; defaults are fine
                        use_retry=True,
                    )
                else:
                    # Backward-compatible fallback
                    self.logger.debug("Falling back to execute_query() method (execute_query_auto not available)")
                    df = self.db.execute_query(header_query, params=params)

                elapsed = time.time() - start_time
                self.logger.info(f"Header query completed successfully: {len(df):,} rows retrieved in {elapsed:.2f}s")

                return df

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"Header query failed after {elapsed:.2f}s: {type(e).__name__}: {e}\n"
                f"Query preview: {query_preview}\n"
                f"Parameters: {param_summary}"
            )
            raise

        finally:
            if self.db is not None:
                try:
                    self.logger.debug("Closing database connection...")
                    self.db.close()
                    self.logger.debug("Database connection closed successfully")
                except Exception as ex:
                    self.logger.exception(f"Failed to dispose SQLAlchemy Engine: {ex}")

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
        import time

        if self.db is None:
            raise ValueError("Database client is not configured (self.db is None).")

        # Log query initiation with sanitized parameter info
        param_summary = self._sanitize_params_for_logging(params) if params else "None"
        query_preview = directional_query[:200].replace("\n", " ") + ("..." if len(directional_query) > 200 else "")

        self.logger.info(f"Querying directional survey data from database")
        self.logger.debug(f"Query preview: {query_preview}")
        self.logger.debug(f"Parameters: {param_summary}")
        self.logger.debug(f"Chunking config: chunk_size={chunk_size}, chunk_if_len_ge={chunk_if_len_ge}")

        start_time = time.time()

        try:
            # Test database connection
            self.logger.debug("Testing database connection...")
            if self.db.test_connection():
                self.logger.debug("Database connection successful")

                # Prefer new auto method if available
                if hasattr(self.db, "execute_query_auto") and callable(getattr(self.db, "execute_query_auto")):
                    self.logger.debug("Using execute_query_auto() method with retry enabled")

                    # Try to auto-detect the IN-list param if there is exactly one list-like param
                    chunk_param_name: Optional[str] = None
                    if params:
                        list_like = [k for k, v in params.items() if isinstance(v, (list, tuple, set))]
                        if len(list_like) == 1:
                            chunk_param_name = list_like[0]
                            list_len = len(params[chunk_param_name])  # type: ignore[arg-type]
                            self.logger.debug(
                                f"Detected list parameter '{chunk_param_name}' with {list_len:,} items"
                            )

                            # Determine if chunking will be triggered
                            if list_len >= chunk_if_len_ge:
                                num_chunks = (list_len + chunk_size - 1) // chunk_size
                                self.logger.info(
                                    f"Large parameter list detected ({list_len:,} items). "
                                    f"Chunking will be used: {num_chunks} chunks of size {chunk_size}"
                                )
                            else:
                                self.logger.debug(
                                    f"List size ({list_len:,}) below chunking threshold ({chunk_if_len_ge:,}). "
                                    "Single query will be executed"
                                )

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

                    self.logger.debug(f"Query executed: retrieved {len(df):,} rows")

                    # Safe post-sort to restore global ordering when chunking was used
                    # (only sort if the expected columns exist)
                    if {"uwi12", "md"}.issubset(df.columns):
                        self.logger.debug("Post-sorting by ['uwi12', 'md'] to restore global ordering")
                        df = df.sort_values(["uwi12", "md"], kind="mergesort").reset_index(drop=True)
                    elif {"uwi", "md"}.issubset(df.columns):
                        self.logger.debug("Post-sorting by ['uwi', 'md'] to restore global ordering")
                        df = df.sort_values(["uwi", "md"], kind="mergesort").reset_index(drop=True)
                    else:
                        self.logger.debug("Skipping post-sort (uwi/uwi12 and md columns not found)")

                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"Directional query completed successfully: {len(df):,} rows retrieved in {elapsed:.2f}s"
                    )

                    return df

                # Backward-compatible fallback
                self.logger.debug("Falling back to execute_query() method (execute_query_auto not available)")
                df = self.db.execute_query(directional_query, params=params)

                elapsed = time.time() - start_time
                self.logger.info(
                    f"Directional query completed successfully: {len(df):,} rows retrieved in {elapsed:.2f}s"
                )

                return df

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"Directional query failed after {elapsed:.2f}s: {type(e).__name__}: {e}\n"
                f"Query preview: {query_preview}\n"
                f"Parameters: {param_summary}"
            )
            raise

        finally:
            if self.db is not None:
                try:
                    self.logger.debug("Closing database connection...")
                    self.db.close()
                    self.logger.debug("Database connection closed successfully")
                except Exception as ex:
                    self.logger.exception(f"Failed to dispose SQLAlchemy Engine: {ex}")

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
    def __init__(self, header_df: pd.DataFrame, directional_df: pd.DataFrame, directional_source: str = "enverus", logger: Optional[logging.Logger] = None,):
        """
        Initialize GeoSurveyProcessor with well header and directional survey data.

        This constructor sets up the processor for geospatial transformations by copying
        the input DataFrames (to avoid mutations) and configuring hierarchical logging.
        The directional_source parameter determines which column schema is expected in
        directional_df (e.g., "enverus" uses uwi12, "IHS" uses uwi).

        The processor provides methods for:
        - Converting between lat/lon and UTM coordinate systems
        - Computing UTM coordinates from surface locations + lateral displacements
        - Filtering survey data to lateral section only (after heel point)
        - Extracting heel/toe/midpoint locations

        Parameters
        ----------
        header_df : pd.DataFrame
            Well header data with at minimum: uwi, surface_lat, surface_lon.
            Expected to contain one row per well with surface location information.
            DataFrame is copied internally to prevent external mutations.
        directional_df : pd.DataFrame
            Directional survey data with wellbore trajectory information.
            Required columns depend on directional_source:
            - For "enverus": uwi12, md, tvd, latitude, longitude, deviation_E/W, deviation_N/S
            - For "IHS": uwi, md, tvd, latitude, longitude, deviation_E/W, E/W, deviation_N/S, N/S
            DataFrame is copied internally to prevent external mutations.
        directional_source : str, default="enverus"
            Data source identifier for directional survey schema. Valid options:
            - "enverus": Enverus/DrillingInfo schema (uses uwi12, deviation columns)
            - "IHS": IHS Markit schema (uses uwi, includes E/W and N/S direction columns)
            Case-sensitive; typically lowercase. Used by prepare_lateral_trajectory_data()
            to determine column access patterns.
        logger : Optional[logging.Logger], default=None
            Parent logger instance for hierarchical logging. If None, creates a default
            logger named "geo_survey_processor". If provided, creates a child logger with
            name "{parent_logger.name}.geo_survey_processor" to maintain log hierarchy.
            Logger propagation is always enabled.

        Attributes
        ----------
        logger : logging.Logger
            Logger instance for this processor.
        df_header : pd.DataFrame
            Copy of input header_df.
        df_directional : pd.DataFrame
            Copy of input directional_df.
        directional_source : str
            Stored directional source identifier.

        Examples
        --------
        Initialize with Enverus directional data (default):

        >>> header_df = pd.DataFrame({
        ...     "uwi": ["12345"],
        ...     "surface_lat": [31.5],
        ...     "surface_lon": [-102.3]
        ... })
        >>> directional_df = pd.DataFrame({
        ...     "uwi12": ["12345"],
        ...     "md": [0, 1000, 2000],
        ...     "tvd": [0, 995, 1990],
        ...     "latitude": [31.5, 31.501, 31.502],
        ...     "longitude": [-102.3, -102.299, -102.298],
        ...     "deviation_E/W": [0, 10, 20],
        ...     "deviation_N/S": [0, 5, 10]
        ... })
        >>> geo = GeoSurveyProcessor(
        ...     header_df=header_df,
        ...     directional_df=directional_df,
        ...     directional_source="enverus"
        ... )

        Initialize with IHS directional data:

        >>> header_df = pd.DataFrame({
        ...     "uwi": ["42-123-45678"],
        ...     "surface_lat": [31.5],
        ...     "surface_lon": [-102.3]
        ... })
        >>> directional_df = pd.DataFrame({
        ...     "uwi": ["42-123-45678"],
        ...     "md": [0, 1000],
        ...     "tvd": [0, 995],
        ...     "latitude": [31.5, 31.501],
        ...     "longitude": [-102.3, -102.299],
        ...     "deviation_E/W": [0, 10],
        ...     "E/W": ["E", "E"],
        ...     "deviation_N/S": [0, 5],
        ...     "N/S": ["N", "N"]
        ... })
        >>> geo = GeoSurveyProcessor(
        ...     header_df=header_df,
        ...     directional_df=directional_df,
        ...     directional_source="IHS"
        ... )

        Initialize with custom logger:

        >>> import logging
        >>> parent_log = logging.getLogger("my_pipeline")
        >>> geo = GeoSurveyProcessor(
        ...     header_df=header_df,
        ...     directional_df=directional_df,
        ...     logger=parent_log
        ... )
        >>> # Creates child logger: "my_pipeline.geo_survey_processor"

        Notes
        -----
        - Both input DataFrames are copied to prevent mutations affecting caller's data
        - Logger hierarchy enables structured logging with parent-child relationships
        - Logger propagation is always enabled, allowing messages to bubble up to handlers
        - The directional_source determines which columns are accessed during processing
        - UTM transformations assume Northern Hemisphere (EPSG:326XX codes)
        """
        # Set up hierarchical logger
        parent_logger = logger or logging.getLogger("geo_survey_processor")
        if logger is None:
            self.logger = parent_logger
        else:
            self.logger = logging.getLogger(f"{parent_logger.name}.geo_survey_processor")
        self.logger.propagate = True

        # Copy DataFrames to prevent mutations
        self.logger.debug(f"Initializing GeoSurveyProcessor with directional_source={directional_source}")
        self.logger.debug(f"Header shape: {header_df.shape}, Directional shape: {directional_df.shape}")

        self.df_header: pd.DataFrame = header_df.copy()
        self.df_directional: pd.DataFrame = directional_df.copy()
        self.directional_source: str = directional_source

        self.logger.info(
            f"GeoSurveyProcessor initialized: {len(self.df_header)} wells, "
            f"{len(self.df_directional):,} survey points (source: {directional_source})"
        )

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
        import time

        self.logger.debug(f"Converting UTM coordinates to lat/lon for {len(df):,} rows")
        self.logger.debug(f"UTM columns: x={x_col}, y={y_col} (feet)")
        self.logger.debug(f"Output columns: {lat_col}, {lon_col}")
        self.logger.debug(f"Rounding: {'enabled (8 decimals)' if round_output else 'disabled'}")

        start_time = time.time()

        df = df.copy()

        # Create EPSG codes if not present
        if epsg_col not in df.columns:
            self.logger.debug(f"EPSG column '{epsg_col}' not found - generating from {zone_col}")
            df[epsg_col] = df[zone_col].apply(lambda z: f"EPSG:326{int(z)}")
        else:
            self.logger.debug(f"Using existing EPSG column '{epsg_col}'")

        # Initialize output columns
        df[lat_col] = np.nan
        df[lon_col] = np.nan

        unique_epsgs = df[epsg_col].unique()
        num_zones = len(unique_epsgs)
        self.logger.debug(f"Processing {num_zones} unique EPSG code(s): {sorted(unique_epsgs)}")

        # Transform per EPSG zone
        for idx, epsg in enumerate(unique_epsgs, 1):
            mask = df[epsg_col] == epsg
            zone_rows = mask.sum()

            self.logger.debug(f"  [{idx}/{num_zones}] {epsg}: transforming {zone_rows:,} rows")

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

        elapsed = time.time() - start_time
        rate = f" ({len(df)/elapsed:.0f} rows/sec)" if elapsed > 0 else ""
        self.logger.info(
            f"✓ UTM → lat/lon conversion complete: {len(df):,} rows in {elapsed:.3f}s{rate}"
        )

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
        self.logger.info("Computing UTM coordinates from survey data")

        initial_rows = len(df)
        initial_wells = df['uwi'].nunique() if 'uwi' in df.columns else 0
        self.logger.debug(f"Input: {initial_rows:,} survey points across {initial_wells} wells")

        # Validate force_utm_zone if provided
        if force_utm_zone is not None:
            if not isinstance(force_utm_zone, int) or not (1 <= force_utm_zone <= 60):
                raise ValueError(
                    f"force_utm_zone must be an integer between 1 and 60, got {force_utm_zone}"
                )
            self.logger.debug(f"UTM zone forced to: {force_utm_zone}")
        else:
            self.logger.debug("UTM zones will be auto-detected from longitude")

        start_time = time.time()

        # Sort and initialize coordinate columns
        self.logger.debug("Sorting by [uwi, md] and initializing coordinate columns")
        df = df.sort_values(by=["uwi", "md"]).copy()

        df["x"], df["y"] = np.zeros(len(df)), np.zeros(len(df))

        # Determine operation mode
        has_latlon = "latitude" in df.columns and "longitude" in df.columns
        has_surface = surface_df is not None

        if has_latlon:
            self.logger.info("✓ Mode: Using lat/lon from input DataFrame (per-row coordinates)")
            self.logger.debug("Required columns present: latitude, longitude")

            # Determine zones - either forced or per-row
            if force_utm_zone is not None:
                df["utm_zone"] = force_utm_zone
                self.logger.info(f"🌍 Forcing all data to UTM zone {force_utm_zone}")
            else:
                self.logger.debug("Computing UTM zones from longitude values...")
                df["utm_zone"] = df["longitude"].apply(self.determine_utm_zone)

                # Warn if multiple zones detected
                unique_zones = df["utm_zone"].unique()
                if len(unique_zones) > 1:
                    self.logger.warning(
                        f"⚠️ Data spans {len(unique_zones)} UTM zones: {sorted(unique_zones)}. "
                        f"This may cause coordinate inconsistencies in downstream spacing "
                        f"calculations. Consider using force_utm_zone parameter to project "
                        f"all data to a single zone. See docstring for recommended zones by basin."
                    )
                else:
                    self.logger.debug(f"All data in single UTM zone: {unique_zones[0]}")

            # Transform coordinates per zone
            unique_zones_list = df["utm_zone"].unique()
            num_zones = len(unique_zones_list)
            self.logger.debug(f"Transforming coordinates for {num_zones} zone(s)...")

            for idx, zone in enumerate(unique_zones_list, 1):
                epsg_code = f"EPSG:326{zone:02d}" if zone < 10 else f"EPSG:326{zone}"
                mask = df["utm_zone"] == zone
                zone_rows = mask.sum()

                self.logger.debug(f"  [{idx}/{num_zones}] Zone {zone} ({epsg_code}): {zone_rows:,} rows")

                transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
                lon = df.loc[mask, "longitude"].values
                lat = df.loc[mask, "latitude"].values
                easting_m, northing_m = transformer.transform(lon, lat)

                df.loc[mask, "x"] = easting_m * 3.28084
                df.loc[mask, "y"] = northing_m * 3.28084
                df.loc[mask, "epsg_code"] = epsg_code

        elif surface_df is not None:
            self.logger.info("✓ Mode: Using surface_df + displacements (lat/lon not in survey data)")
            self.logger.debug(f"Surface DataFrame: {len(surface_df)} wells")

            required_cols = {"uwi", "surface_lat", "surface_lon"}

            if not required_cols.issubset(surface_df.columns):
                raise ValueError(f"surface_df must contain {required_cols}")

            df = df.merge(surface_df, on="uwi", how="left")

            # Determine zones - either forced or per-row
            if force_utm_zone is not None:
                df["utm_zone"] = force_utm_zone
                self.logger.info(f"🌍 Forcing all data to UTM zone {force_utm_zone}")
            else:
                self.logger.debug("Computing UTM zones from surface longitude values...")
                df["utm_zone"] = df["surface_lon"].apply(self.determine_utm_zone)

                # Warn if multiple zones detected
                unique_zones = df["utm_zone"].unique()
                if len(unique_zones) > 1:
                    self.logger.warning(
                        f"⚠️ Data spans {len(unique_zones)} UTM zones: {sorted(unique_zones)}. "
                        f"This may cause coordinate inconsistencies in downstream spacing "
                        f"calculations. Consider using force_utm_zone parameter to project "
                        f"all data to a single zone. See docstring for recommended zones by basin."
                    )
                else:
                    self.logger.debug(f"All data in single UTM zone: {unique_zones[0]}")

            # Transform coordinates per zone with displacement application
            unique_zones_list = df["utm_zone"].unique()
            num_zones = len(unique_zones_list)
            self.logger.debug(f"Transforming {num_zones} zone(s) with surface + displacement method...")

            for idx, zone in enumerate(unique_zones_list, 1):
                epsg_code = f"EPSG:326{zone:02d}" if zone < 10 else f"EPSG:326{zone}"
                mask = df["utm_zone"] == zone
                zone_rows = mask.sum()

                self.logger.debug(f"  [{idx}/{num_zones}] Zone {zone} ({epsg_code}): {zone_rows:,} rows")

                transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
                lon = df.loc[mask, "surface_lon"].values
                lat = df.loc[mask, "surface_lat"].values
                easting_m, northing_m = transformer.transform(lon, lat)

                # Convert to feet
                easting_ft = easting_m * 3.28084
                northing_ft = northing_m * 3.28084

                # Apply signed displacements
                ew_sign = df.loc[mask, "E/W"].map({"E": 1, "W": -1}).fillna(0)
                ns_sign = df.loc[mask, "N/S"].map({"N": 1, "S": -1}).fillna(0)

                df.loc[mask, "x"] = easting_ft + df.loc[mask, "deviation_E/W"] * ew_sign
                df.loc[mask, "y"] = northing_ft + df.loc[mask, "deviation_N/S"] * ns_sign
                df.loc[mask, "epsg_code"] = epsg_code

            # Back-convert to lat/lon using x/y and epsg_code
            self.logger.debug("Back-converting UTM coordinates to lat/lon...")
            df = self.convert_utm_to_latlon(df)

        else:
            raise ValueError("Either lat/lon must be present in df, or surface_df must be provided.")

        # Compute z-coordinate (depth positive down)
        self.logger.debug("Computing z-coordinates from TVD (z = -tvd)")
        df["z"] = -df["tvd"]

        elapsed = time.time() - start_time
        rate = f" ({len(df)/elapsed:.0f} rows/sec)" if elapsed > 0 else ""
        self.logger.info(
            f"✓ UTM coordinate computation complete: {len(df):,} rows in {elapsed:.2f}s{rate}"
        )

        return df
    
    def filter_after_heel_point(self, df: pd.DataFrame, inclination_filter: float) -> pd.DataFrame:
        """
        Keep rows at/after the first high-inclination point per UWI.

        For each `uwi`, this method identifies the **first** survey station where
        `inclination >= inclination_filter` degrees (the "heel trigger") and then
        retains that row and all *subsequent* rows for that `uwi`. Wells that never
        reach `inclination >= inclination_filter` are dropped entirely.

        Implementation details
        ----------------------
        1. The input DataFrame is first sorted by ['uwi', 'md'] in ascending order
           and its index is reset to a simple RangeIndex (0..N-1). This ensures
           that index labels and row positions coincide.

        2. Among rows where `inclination >= inclination_filter`, the method finds, for each `uwi`,
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
        inclination_filter : float
            Minimum inclination angle (in degrees) used to identify the heel
            point. Survey rows with ``inclination >= inclination_filter`` are
            considered lateral; the first such row per well marks the start of
            the lateral section.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame containing only rows at/after the heel point for
            each `uwi` that reaches `inclination >= inclination_filter`. The result is:
              - sorted by ['uwi', 'md'] ascending, and
              - index reset to a simple RangeIndex.

        Notes
        -----
        - If a given `uwi` has **no** row with `inclination >= inclination_filter`, all of its
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
        >>> out = geo.filter_after_heel_point(df, inclination_filter=80)
        >>> out
          uwi    md  inclination
        0   A  1100         82.0
        1   A  1200         90.0
        2   B  1000         85.0

        >>> out.groupby("uwi").size().to_dict()
        {'A': 2, 'B': 1}
        """
        self.logger.info("Filtering directional data to lateral section (post-heel)")

        initial_rows = len(df)
        initial_wells = df['uwi'].nunique() if 'uwi' in df.columns else 0

        self.logger.debug(f"Input: {initial_rows:,} survey points across {initial_wells} wells")

        # Sort and reset index so that labels == positions
        df = df.sort_values(by=["uwi", "md"], ascending=True).reset_index(drop=True)

        mask = (df["inclination"] >= inclination_filter)
        heel_count = mask.sum()

        self.logger.debug(f"Found {heel_count:,} survey points with inclination >= {inclination_filter}°")

        # If no heel trigger at all, drop everything
        if not mask.any():
            self.logger.warning(
                f"⚠️ No inclination >= {inclination_filter}° found for any well. "
                "All data represents vertical/build sections - returning empty DataFrame."
            )
            return df.iloc[0:0].copy()

        # Identify first heel point per well
        idx_start = df[mask].groupby('uwi', sort=False).head(1).index
        start_idx_map = dict(zip(df.loc[idx_start, 'uwi'], idx_start))

        wells_with_heel = len(start_idx_map)
        wells_without_heel = initial_wells - wells_with_heel

        if wells_without_heel > 0:
            all_uwis = set(df['uwi'].unique())
            wells_missing_heel = sorted(all_uwis - set(start_idx_map.keys()))
            uwi_lines = "\n".join(
                ", ".join(wells_missing_heel[i:i + 5])
                for i in range(0, len(wells_missing_heel), 5)
            )
            self.logger.debug(
                f"{wells_without_heel} wells do not reach {inclination_filter}° inclination "
                f"(will be excluded from output):\n{uwi_lines}"
            )

        # Vectorized filtering: keep rows at/after heel point
        uwis = df['uwi'].values
        indices = np.arange(len(df))

        start_indices = np.vectorize(start_idx_map.get, otypes=[float])(uwis)
        valid_rows = indices >= start_indices

        out = df[valid_rows].reset_index(drop=True)

        # Log results
        output_rows = len(out)
        output_wells = out['uwi'].nunique()
        retention_rate = (output_rows / initial_rows * 100) if initial_rows > 0 else 0

        self.logger.info(
            f"✓ Heel filtering complete: {output_rows:,} lateral points retained "
            f"({retention_rate:.1f}% of input) across {output_wells} wells"
        )
        self.logger.debug(
            f"Dropped {initial_rows - output_rows:,} points from vertical/build sections"
        )

        return out
    
    def get_heel_toe_midpoints_latlon(self, df: pd.DataFrame, inclination_filter: float) -> pd.DataFrame:
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
        inclination_filter : float
            Minimum inclination angle (in degrees) passed to
            ``filter_after_heel_point()`` to identify the heel point.

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
        >>> out = GeoSurveyProcessor().get_heel_toe_midpoints_latlon(df, inclination_filter=80)
        >>> set(["heel_lat","heel_lon","toe_lat","toe_lon","mid_Lat","mid_Lon"]).issubset(out.columns)
        True
        """
        self.logger.info("Extracting heel/toe/midpoint coordinates for lateral sections")

        initial_wells = df['uwi'].nunique() if 'uwi' in df.columns else 0
        self.logger.debug(f"Input: {len(df):,} survey points across {initial_wells} wells")

        # Getting DataFrame with only the rows after the heel point
        df = self.filter_after_heel_point(df, inclination_filter=inclination_filter)

        if df.empty:
            self.logger.warning("No lateral data after heel filtering - returning empty DataFrame")
            return pd.DataFrame(columns=['uwi', 'heel_lat', 'heel_lon', 'toe_lat', 'toe_lon', 'mid_Lat', 'mid_Lon'])

        # Group by 'uwi' and extract heel/toe lat/lon
        self.logger.debug(f"Aggregating heel/toe coordinates for {df['uwi'].nunique()} wells")

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
        self.logger.debug("Computing geometric midpoints from heel/toe coordinates")
        heel_toe_df["mid_Lat"] = (heel_toe_df["heel_lat"] + heel_toe_df["toe_lat"]) / 2
        heel_toe_df["mid_Lon"] = (heel_toe_df["heel_lon"] + heel_toe_df["toe_lon"]) / 2

        self.logger.info(
            f"✓ Extracted heel/toe/midpoint locations for {len(heel_toe_df)} wells"
        )

        return heel_toe_df

    def prepare_lateral_trajectory_data(self, inclination_filter: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare and synchronize directional survey data with well header data.

        This method performs a complete data preparation pipeline:
        1. Computes UTM coordinates from lat/lon in directional survey data
        2. Filters trajectories to keep only lateral section (after heel point)
        3. Cross-filters directional and header dataframes to ensure alignment:
           - Keeps only wells that exist in BOTH datasets
           - Removes orphan wells (directional without header, or vice versa)

        The synchronization ensures downstream analysis (e.g., spacing calculations,
        neighbor identification) has consistent well lists across both data sources.

        Parameters
        ----------
        inclination_filter : float
            Minimum inclination angle (in degrees) passed to
            ``filter_after_heel_point()`` to identify the heel point.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            - **directional_lateral** : pd.DataFrame
                Directional survey data with UTM coordinates, filtered to lateral
                section only, and limited to wells that have matching header records.
                Contains columns: uwi, md, tvd, x, y, z, latitude, longitude, azimuth, etc.

            - **header_aligned** : pd.DataFrame
                Header data filtered to wells that have directional survey data.
                Contains columns: uwi, api, operator, well_name, bench, surface_lat, etc.

        Notes
        -----
        - **Data Source Handling**: The method automatically handles different data
          sources (Enverus vs IHS) with appropriate join keys:
          * Enverus: Uses 'uwi12' for cross-filtering (12-digit identifier)
          * IHS: Uses 'uwi' for cross-filtering (full UWI)

        - **Lateral Filtering**: Only survey points after the heel (kickoff) are retained.
          This removes vertical/curve sections, keeping only the horizontal lateral.

        - **Data Quality**: Wells with incomplete data (missing header or directional)
          are automatically excluded from both output datasets.

        Logging
        -------
        INFO: Logs initial row counts for both datasets
        INFO: Logs UTM coordinate computation completion
        INFO: Logs lateral section filtering results
        INFO: Logs final synchronized dataset statistics (wells kept/removed)

        Examples
        --------
        >>> geo = GeoSurveyProcessor(header_df, directional_df, directional_source="enverus")
        >>> lateral_traj, header_aligned = geo.prepare_lateral_trajectory_data(inclination_filter=80)
        >>> print(f"Prepared {lateral_traj['uwi'].nunique()} wells with {len(lateral_traj)} survey points")

        See Also
        --------
        compute_utm_coordinates : Converts lat/lon to UTM coordinates
        filter_after_heel_point : Filters survey data to lateral section only
        """
        import time

        self.logger.info("=" * 80)
        self.logger.info("Starting lateral trajectory data preparation pipeline")
        self.logger.info(f"Data source: {self.directional_source}")
        self.logger.info(f"Initial directional rows: {len(self.df_directional):,}")
        self.logger.info(f"Initial header wells: {len(self.df_header):,}")

        pipeline_start = time.time()

        # Step 1: Compute UTM coordinates from lat/lon
        self.logger.info("Step 1/3: Computing UTM coordinates from directional survey data...")
        step1_start = time.time()
        df_utm = self.compute_utm_coordinates(self.df_directional)
        step1_elapsed = time.time() - step1_start
        self.logger.info(f"✓ Step 1 complete: UTM coordinates computed for {len(df_utm):,} survey points ({step1_elapsed:.2f}s)")

        # Step 2: Filter to lateral section (after heel point)
        self.logger.info("Step 2/3: Filtering to lateral section (after heel point)...")
        step2_start = time.time()
        wells_before_filter = df_utm["uwi"].nunique()
        df_utm_lateral = self.filter_after_heel_point(df_utm, inclination_filter=inclination_filter)
        wells_after_filter = df_utm_lateral["uwi"].nunique()
        step2_elapsed = time.time() - step2_start
        self.logger.info(
            f"✓ Step 2 complete: Lateral section extracted - {len(df_utm_lateral):,} points "
            f"from {wells_after_filter} wells ({step2_elapsed:.2f}s)"
        )

        if wells_after_filter < wells_before_filter:
            removed_wells = wells_before_filter - wells_after_filter
            self.logger.warning(f"⚠️  {removed_wells} wells removed (no lateral section found)")

        # Step 3: Cross-filter directional and header data for synchronization
        self.logger.info("Step 3/3: Synchronizing directional and header datasets...")
        step3_start = time.time()

        if self.directional_source == "enverus":
            # Enverus uses uwi12 as the join key
            self.logger.debug("Using 'uwi12' as join key for Enverus data")

            # Keep directional surveys for wells that have header records
            directional_lateral = df_utm_lateral[
                df_utm_lateral["uwi12"].isin(self.df_header["uwi12"].unique())
            ].reset_index(drop=True)

            # Keep header records for wells that have directional surveys
            header_aligned = self.df_header[
                self.df_header["uwi"].isin(directional_lateral["uwi"].unique())
            ].reset_index(drop=True)

        elif self.directional_source == "ihs":
            # IHS uses uwi as the join key
            self.logger.debug("Using 'uwi' as join key for IHS data")

            # Keep directional surveys for wells that have header records
            directional_lateral = df_utm_lateral[
                df_utm_lateral["uwi"].isin(self.df_header["uwi"].unique())
            ].reset_index(drop=True)

            # Keep header records for wells that have directional surveys
            header_aligned = self.df_header[
                self.df_header["uwi"].isin(directional_lateral["uwi"].unique())
            ].reset_index(drop=True)

        else:
            # Unsupported data source
            error_msg = f"Unsupported directional_source: '{self.directional_source}'. Must be 'enverus' or 'ihs'."
            self.logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Log synchronization results
        step3_elapsed = time.time() - step3_start
        final_wells = directional_lateral["uwi"].nunique()
        final_survey_points = len(directional_lateral)
        final_header_wells = len(header_aligned)

        self.logger.info(f"✓ Step 3 complete: Synchronization finished ({step3_elapsed:.2f}s)")
        self.logger.info(f"   • Aligned wells: {final_wells}")
        self.logger.info(f"   • Directional survey points: {final_survey_points:,}")
        self.logger.info(f"   • Header records: {final_header_wells:,}")

        # Check for data quality issues
        wells_lost_directional = wells_after_filter - final_wells
        wells_lost_header = len(self.df_header) - final_header_wells

        if wells_lost_directional > 0:
            self.logger.warning(f"⚠️  {wells_lost_directional} wells in directional data had no matching header records (excluded)")

        if wells_lost_header > 0:
            self.logger.warning(f"⚠️  {wells_lost_header} wells in header data had no matching directional surveys (excluded)")

        if final_wells == 0:
            self.logger.error("❌ No wells remain after synchronization! Check data compatibility.")
            raise ValueError("No wells remain after synchronization. Verify that directional and header data have overlapping well identifiers.")

        # Log pipeline completion with total timing
        pipeline_elapsed = time.time() - pipeline_start
        self.logger.info("─" * 80)
        self.logger.info(
            f"✓ Pipeline complete: {final_survey_points:,} survey points across {final_wells} wells "
            f"in {pipeline_elapsed:.2f}s"
        )
        self.logger.debug(f"  Step breakdown: UTM={step1_elapsed:.2f}s, Filter={step2_elapsed:.2f}s, Sync={step3_elapsed:.2f}s")
        self.logger.info("=" * 80)

        return directional_lateral, header_aligned

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