# %%
# Importing pandas package for data manipulation and analysis
import pandas as pd
pd.set_option('display.max_columns', None) # Set the maximum number of columns to display to None

import numpy as np # Importing numpy package for numerical operations

from typing import Dict, Union, Optional # Importing specific types from typing module

import time # Importing Time Module

import pyproj # Importing pyproj package

from src.utils import CustomLogger # Importing CustomLogger class from custom

import os # Importing os module for operating system dependent functionality

# Importing necessary modules for plotting and data manipulation
import matplotlib.pyplot as plt # Importing matplotlib.pyplot for plotting

# Setting matplotlib to inline mode for Jupyter notebooks
#%matplotlib inline

#%config InlineBackend.figure_format = 'svg' # Configuring inline backend to use SVG format for figures

# %%
class WellDataLoader:
    """
    Load well header and directional survey data from files (CSV/Excel) or a database.

    This utility centralizes I/O and column normalization so downstream code can rely on a
    stable schema. For file-based inputs, you provide a `column_map` that maps *canonical*
    column names (what your code expects) to the *actual* column names present in the file.
    The loader will select only those columns and rename them to your canonical names.

    Parameters
    ----------
    db : Optional[object], default=None
        A database client-like object that exposes:
        - `connect() -> None`
        - `execute_query(sql: str) -> pandas.DataFrame`
        - `close_connection() -> None`
    log_dir : str, default="./logs"
        Directory to write logs created by `CustomLogger`.

    Examples
    --------
    Basic setup:

    >>> loader = WellDataLoader(db=my_db_client, log_dir="./logs")

    Reading a header file with custom column names:

    >>> column_map = {
    ...     "uwi": "API14",
    ...     "well_name": "WellName",
    ...     "operator": "CurrentOperator"
    ... }
    >>> header = loader.get_header_data(
    ...     source="headers.xlsx",
    ...     column_map=column_map,
    ...     dtype={"API14": "string", "WellName": "string", "CurrentOperator": "string"}
    ... )
    >>> list(header.columns)
    ['uwi', 'well_name', 'operator']

    Pulling from the database (no file path provided):

    >>> header = loader.get_header_data(basin="MB", start_year=2019)
    """

    def __init__(
        self,
        db: Optional[object] = None,
        log_dir: str = "./logs"
    ):
        """
        Initialize the loader with an optional database client and logging directory.

        Parameters
        ----------
        db : Optional[object], default=None
            Database client-like object with `connect()`, `execute_query()`, `close_connection()`.
        log_dir : str, default="./logs"
            Directory for the `CustomLogger` to write logs.

        Examples
        --------
        >>> loader = WellDataLoader(db=my_db_client, log_dir="./logs")
        """
        self.db = db
        self.logger = CustomLogger("well_data_loader", "WellDataLoaderLogger", log_dir).get_logger()
        self.header_df = pd.DataFrame()

    def load_data_from_file(
        self,
        file_path: str,
        required_columns: Dict[str, str],
        dtype: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        """
        Load a CSV/Excel file, select required columns, and rename them to canonical names.

        This is a low-level helper used by `get_header_data()` and `get_directional_data()`.
        Provide `required_columns` as a mapping from your *canonical* names to the *actual*
        column names found in the file. Only those columns are read and then renamed.

        Parameters
        ----------
        file_path : str
            Path to a `.csv`, `.xlsx`, or `.xls` file.
        required_columns : Dict[str, str]
            Mapping of `{canonical_name: file_column_name}`. The values (file column names)
            must exist in the file.
        dtype : Optional[Dict[str, type]], default=None
            Pandas dtype mapping passed directly to `pd.read_csv` / `pd.read_excel`.
            **Important:** keys here must be the *file* column names (i.e., the values of
            `required_columns`), because dtypes are applied before renaming.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing only the requested columns, renamed to canonical names.

        Raises
        ------
        ValueError
            If the file type is unsupported or any required column is missing.
        Exception
            Any other error encountered during file read is logged and re-raised.

        Examples
        --------
        Reading CSV:

        >>> colmap = {"uwi": "API14", "well_name": "WellName"}
        >>> df = loader.load_data_from_file(
        ...     "wells.csv",
        ...     required_columns=colmap,
        ...     dtype={"API14": str, "WellName": str}
        ... )
        >>> list(df.columns)
        ['uwi', 'well_name']

        Reading Excel:

        >>> df = loader.load_data_from_file(
        ...     "wells.xlsx",
        ...     required_columns=colmap
        ... )
        """
        try:
            usecols = list(required_columns.values())
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, dtype=dtype, usecols=usecols)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, dtype=dtype, usecols=usecols)
            else:
                raise ValueError("Unsupported file type. Use CSV or Excel.")

            missing = [val for val in required_columns.values() if val not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in file: {missing}")

            return df.rename(columns={v: k for k, v in required_columns.items()})

        except Exception as e:
            self.logger.error(f"Failed to load data from file {file_path}: {e}")
            raise

    def get_header_data(
        self,
        source: Optional[Union[str, pd.DataFrame]] = None,
        column_map: Optional[Dict[str, str]] = None,
        basin: str = "MB",
        start_year: int = 2019,
        dtype: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        """
        Return header data either from a DataFrame, a file path, or the database.

        Behavior is determined by `source`:
        - If `source` is a `pandas.DataFrame`, it is returned as-is (and cached to `self.header_df`).
        - If `source` is a file path, `column_map` is REQUIRED and used to read & rename columns.
        - If `source` is `None`, data are queried from `self.db` via `_query_header_from_db()`.

        Parameters
        ----------
        source : Optional[Union[str, pandas.DataFrame]], default=None
            - `DataFrame`: use this directly.
            - `str`: path to CSV/Excel file.
            - `None`: query the database.
        column_map : Optional[Dict[str, str]], default=None
            Required when `source` is a file path. Mapping of `{canonical_name: file_column_name}`.
        basin : str, default="MB"
            Basin code used in the database query when `source is None`.
        start_year : int, default=2019
            Minimum first production year for the database query when `source is None`.
        dtype : Optional[Dict[str, type]], default=None
            Dtype mapping for file-based reads (keys must be *file* column names).

        Returns
        -------
        pandas.DataFrame
            Header data with canonical column names. Also stored in `self.header_df`.

        Raises
        ------
        ValueError
            If `column_map` is missing for file-based input or `source` is invalid.
        Exception
            Any database errors are logged and re-raised.

        Examples
        --------
        Using a provided DataFrame:

        >>> header_df = pd.DataFrame({"uwi": ["123"], "well_name": ["A-1"]})
        >>> out = loader.get_header_data(source=header_df)
        >>> out is header_df
        True

        From a file:

        >>> colmap = {"uwi": "API14", "well_name": "WellName", "operator": "CurrentOperator"}
        >>> out = loader.get_header_data(
        ...     source="headers.xlsx",
        ...     column_map=colmap,
        ...     dtype={"API14": "string", "WellName": "string", "CurrentOperator": "string"}
        ... )

        From the database:

        >>> out = loader.get_header_data(source=None, basin="MB", start_year=2020)
        """
        if isinstance(source, pd.DataFrame):
            self.logger.info("Using provided header DataFrame.")
            df = source
        elif isinstance(source, str) and os.path.exists(source):
            if not column_map:
                raise ValueError("Column map must be provided when reading from file.")
            self.logger.info(f"Loading header data from file: {source}")
            df = self.load_data_from_file(source, column_map, dtype=dtype)
        elif source is None:
            self.logger.info("Loading header data from SQL.")
            df = self._query_header_from_db(basin, start_year)
        else:
            raise ValueError("Invalid input: provide either a file path, DataFrame, or SQL query.")

        self.header_df = df
        return df

    def get_directional_data(
        self,
        source: Optional[str] = None,
        column_map: Optional[Dict[str, str]] = None,
        dtype: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        """
        Return directional survey data either from a file path or the database.

        Parameters
        ----------
        source : Optional[str], default=None
            - `str`: path to CSV/Excel file.
            - `None`: query the database via `_query_directional_from_db()`.
        column_map : Optional[Dict[str, str]], default=None
            Required when `source` is a file path. Mapping of `{canonical_name: file_column_name}`.
        dtype : Optional[Dict[str, type]], default=None
            Dtype mapping for file-based reads (keys must be *file* column names).

        Returns
        -------
        pandas.DataFrame
            Directional survey data with canonical column names (if file-based).

        Raises
        ------
        ValueError
            If `column_map` is missing for file-based input or `source` is invalid.
        Exception
            Any database errors are logged and re-raised.

        Examples
        --------
        From a file:

        >>> colmap = {
        ...     "uwi": "API14",
        ...     "md_ft": "MD_FT",
        ...     "tvd_ft": "TVD_FT",
        ...     "x": "UTM_X",
        ...     "y": "UTM_Y"
        ... }
        >>> dir_df = loader.get_directional_data(
        ...     source="directional.csv",
        ...     column_map=colmap,
        ...     dtype={"API14": "string", "MD_FT": float, "TVD_FT": float}
        ... )

        From the database:

        >>> dir_df = loader.get_directional_data(source=None)
        """
        if source and os.path.exists(source):
            if not column_map:
                raise ValueError("Column map must be provided when reading from file.")
            self.logger.info(f"Loading directional data from file: {source}")
            return self.load_data_from_file(source, column_map, dtype=dtype)
        elif source is None:
            self.logger.info("Loading directional data from SQL.")
            return self._query_directional_from_db()
        else:
            raise ValueError("Provide either a file path or SQL query for directional data.")

    def _query_header_from_db(self, basin: str, start_year: int) -> pd.DataFrame:
        """
        Query header records from the configured database client.

        The SQL selects a standardized set of columns and filters by basin, horizontal
        hole direction, allowed reserve categories, and minimum first production year.

        Parameters
        ----------
        basin : str
            Basin code to filter on (e.g., "MB").
        start_year : int
            Minimum first production year (inclusive).

        Returns
        -------
        pandas.DataFrame
            Query result as a DataFrame. Expected columns include:
            ['uwi', 'lease_name', 'well_name', 'well_num', 'operator', 'rsv_cat',
             'bench', 'first_prod_date', 'comp_date', 'hole_direction',
             'surface_lat', 'surface_lon'].

        Raises
        ------
        Exception
            Any connection or query error is logged and re-raised.

        Examples
        --------
        >>> df = loader._query_header_from_db(basin="MB", start_year=2020)
        >>> set(["uwi", "well_name"]).issubset(df.columns)
        True
        """
        query = f"""
        SELECT
            api14 AS uwi, 
            leaseName AS lease_name,
            wellName AS well_name,
            wellNumber AS well_num,
            currentOperator AS operator,
            customString2 AS rsv_cat,
            customString0 AS bench,
            DATE(firstProdDate) AS first_prod_date,
            DATE(completionStartDate) AS comp_date,
            holeDirection AS hole_direction,
            surfaceLatitude AS surface_lat,
            surfaceLongitude AS surface_lon
        FROM Combocurve.export.wells
        WHERE basin = '{basin}'
          AND customString2 in ("01PDP", "02PDNP", "02PA") 
          AND holeDirection = 'H' 
          AND YEAR(DATE(firstProdDate)) >= {start_year}
        """
        try:
            self.db.connect()
            df = self.db.execute_query(query)
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving header data from databricks: {e}")
            raise
        finally:
            self.db.close_connection()

    def _query_directional_from_db(self) -> pd.DataFrame:
        """
        Query directional survey records from the configured database client.

        This private helper encapsulates your environment-specific SQL. Implement the
        SQL to return a DataFrame that includes, at minimum, keys needed downstream
        (e.g., `uwi`, `md_ft`, `tvd_ft`, and optionally spatial columns like `x`, `y`,
        `lat`, `lon`, or any columns required by your spacing/trajectory pipeline).

        Returns
        -------
        pandas.DataFrame
            Directional survey data.

        Raises
        ------
        Exception
            Any connection or query error is logged and re-raised.

        Examples
        --------
        >>> dir_df = loader._query_directional_from_db()
        >>> isinstance(dir_df, pd.DataFrame)
        True
        """
        if self.header_df.empty or 'uwi' not in self.header_df.columns:
            raise ValueError("Header data must be loaded before querying directional data, and must contain a 'uwi' column.")

        uwis = ", ".join(f"'{id}'" for id in self.header_df['uwi'].unique())
        query = f"""
        SELECT
            uwi, 
            station_md_uscust AS md, 
            station_tvd_uscust AS tvd,
            inclination, 
            azimuth, 
            latitude, 
            longitude, 
            x_offset_uscust AS `deviation_E/W`,
            ew_direction as `E/W`,
            y_offset_uscust AS `deviation_N/S`,
            ns_direction  as `N/S`,
            point_type as point_type_name
        FROM ihs_sp.well.well_directional_survey_station
        WHERE uwi IN ({uwis})
        ORDER BY uwi, md;
        """
        try:
            self.db.connect()
            df = self.db.execute_query(query)
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving directional data from databricks: {e}")
            raise
        finally:
            self.db.close_connection()


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
    def __init__(self, log_dir: str = "./logs",):
        """
        Initialize the processor and its logger.

        Parameters
        ----------
        log_dir : str, default="./logs"
            Directory for logging via `CustomLogger`.

        Examples
        --------
        >>> geo = GeoSurveyProcessor()
        """
        self.logger = CustomLogger("geo_processor", "GeoLogger", log_dir).get_logger()  # Custom logger
        self.logger.info("GeoSurveyProcessor initialized.")

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
    
    def compute_utm_coordinates(self, df: pd.DataFrame, surface_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute UTM (x, y, z) in **feet** for survey points.

        Two operation modes:

        1) **Per-row lat/lon present in `df`**:
           - Requires `df` to contain: `["uwi", "md", "tvd", "latitude", "longitude"]`.
           - UTM zone is inferred per-row from `longitude`.

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

        Returns
        -------
        pandas.DataFrame
            Data with UTM coordinates and auxiliary metadata columns.

        Raises
        ------
        ValueError
            If neither per-row lat/lon nor a valid `surface_df` are provided.
        Exception
            Any projection or merge errors are logged and re-raised.

        Examples
        --------
        Using per-row lat/lon:
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

        Using surface_df + displacements:
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
        >>> out = GeoSurveyProcessor().compute_utm_coordinates(df, surface_df=surf)
        >>> set(["x","y","latitude","longitude"]).issubset(out.columns)
        True
        """
        
        start_time = time.time()
        df = df.sort_values(by=["uwi", "md"]).copy()

        df["x"], df["y"] = np.zeros(len(df)), np.zeros(len(df))

        if "latitude" in df.columns and "longitude" in df.columns:
            self.logger.info("✅ Using lat/lon from input DataFrame.")
            df["utm_zone"] = df["longitude"].apply(self.determine_utm_zone)

            for zone in df["utm_zone"].unique():
                epsg_code = f"EPSG:326{zone}"
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
            df["utm_zone"] = df["surface_lon"].apply(self.determine_utm_zone)

            for zone in df["utm_zone"].unique():
                epsg_code = f"EPSG:326{zone}"
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
            Keep rows at/after the first high‑inclination point per UWI.

            For each `uwi`, finds the first row where `inclination >= 80` (degrees),
            then retains that row and all subsequent rows for that `uwi`.

            Parameters
            ----------
            df : pandas.DataFrame
                Directional survey data with at least:
                - 'uwi'         : well identifier
                - 'md'          : measured depth (used for sorting)
                - 'inclination' : inclination in degrees

            Returns
            -------
            pandas.DataFrame
                Filtered DataFrame, index reset, sorted by ['uwi', 'md'] ascending.

            Notes
            -----
            - If a given `uwi` has no row with `inclination >= 80`, all its rows are dropped.

            Examples
            --------
            >>> df = pd.DataFrame({
            ...     "uwi": ["A","A","A","B","B"],
            ...     "md": [1000,1100,1200,900,1000],
            ...     "inclination": [10, 82, 90, 79, 85],
            ... })
            >>> out = GeoSurveyProcessor().filter_after_heel_point(df)
            >>> out.groupby("uwi").size().to_dict()  # rows retained per UWI
            {'A': 2, 'B': 1}
            """
        # Ensure the data is sorted by MD in ascending order
        df = df.sort_values(by=["uwi", "md"], ascending=True).copy()

        # Numeric heel trigger (first row with inclination >= 80 deg)
        mask = (df["inclination"] >= 80)

        # Identify the first occurrence for each uwi
        idx_start = df[mask].groupby('uwi', sort=False).head(1).index

        # Create a mapping of uwi to the starting index
        start_idx_map = dict(zip(df.loc[idx_start, 'uwi'], idx_start))

        # Create a boolean mask using NumPy to filter rows
        uwis = df['uwi'].values
        indices = np.arange(len(df))

        # Get the minimum start index for each row's uwi
        start_indices = np.vectorize(start_idx_map.get, otypes=[float])(uwis)

        # Mask rows where index is greater than or equal to the start index
        valid_rows = indices >= start_indices

        return df[valid_rows].reset_index(drop=True)
    
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