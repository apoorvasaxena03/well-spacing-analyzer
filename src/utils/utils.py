from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import List, Mapping, Any, Union, Optional, Sequence

import re

#%% Defining Functions

def reorder_columns(df: pd.DataFrame, columns_to_move: List[str], reference_column: str) -> pd.DataFrame:
    """
    Reorders the columns of a dataframe by moving specified columns next to a reference column.

    Parameters:
    df (pd.DataFrame): The dataframe whose columns need to be reordered.
    columns_to_move (List[str]): The names of the columns to move.
    reference_column (str): The name of the column next to which the specified columns should be placed.

    Returns:
    pd.DataFrame: The dataframe with reordered columns.
    """
    columns_order: List[str] = df.columns.tolist()  # Get current column order as a list
    if not all(col in columns_order for col in columns_to_move) or reference_column not in columns_order:
        raise ValueError("Specified columns must exist in the dataframe")
    
    # Find the index of the reference column
    ref_idx: int = columns_order.index(reference_column)
    
    # Remove the columns to move from their current positions
    for col in columns_to_move:
        columns_order.remove(col)
    
    # Insert the columns to move next to the reference column
    for col in reversed(columns_to_move):
        columns_order.insert(ref_idx + 1, col)
    
    # Reorder the dataframe columns
    return df[columns_order].copy()

def clean_column_names(df):
    """ 
    Cleans the column names of a DataFrame by removing special characters,
    replacing spaces with underscores, and converting to lowercase.
    Args:
        df (pd.DataFrame): The DataFrame whose column names need to be cleaned.
    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = [
        re.sub(r'[^a-zA-Z0-9 ]', '', col)  # Remove special characters except space
        .replace(' ', '_')                 # Replace space with underscore
        .lower()                           # Convert to lowercase
        for col in df.columns
    ]
    return df

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes DataFrame column names by:
    - Removing leading/trailing spaces
    - Treating leading 'ENV' as a single token (ENV* → env_*)
    - Converting CamelCase/PascalCase (including acronyms) to snake_case
    - Replacing spaces with underscores
    - Converting to lowercase
    - Removing duplicate underscores
    """

    def convert_to_snake_case(name: str) -> str:
        # Remove leading/trailing spaces
        name = name.strip()

        # 1) Special handling: treat leading 'ENV' as one word
        #    ENVOperator, ENVBasin, ENV_Peer_Group → EnvOperator, EnvBasin, Env_Peer_Group
        name = re.sub(r"^ENV", "Env", name)

        # 2) Split between ALLCAPS and CapitalizedWord:
        #    GALPerFT -> GAL_PerFT, LBSPerGAL -> LBS_PerGAL
        name = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', name)

        # 3) Split between lower/digit and Uppercase:
        #    StateProvince -> State_Province, LeaseName -> Lease_Name
        name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)

        # Replace spaces with underscores
        name = name.replace(" ", "_")

        # Convert to lowercase
        name = name.lower()

        # Remove duplicate underscores
        name = re.sub(r"_+", "_", name)

        return name

    df = df.copy()
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    return df

def read_csv_with_mapper(
    path: str | Path,
    *,
    col_map: Mapping[str, str] | None = None,
    dtype_map: Mapping[str, str | type] | None = None,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """
    Read a CSV and optionally:
      - rename some columns using col_map
      - cast dtypes using dtype_map (after renaming)

    Behavior
    --------
    - All columns in the CSV are read.
    - Only columns present in col_map are renamed; others keep their original name.
    - dtype_map is applied on the *final* column names
      (after renaming; or original if not renamed).
    - If both col_map and dtype_map are None, this is effectively:
        pd.read_csv(path, **read_kwargs)

    Parameters
    ----------
    path :
        Path to the CSV file.
    col_map :
        Mapping of original column names -> new names.
        Example: {"API_UWI": "api", "Prod Date": "prod_dt"}.
    dtype_map :
        Mapping of final column names -> dtypes.
        Example: {"api": "string", "prod_dt": "datetime64[ns]", "gas_mcf": "float64"}.
    **read_kwargs :
        Extra args forwarded to pd.read_csv (e.g. sep, parse_dates, dtype, etc.).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    df = pd.read_csv(path, **read_kwargs)

    # Rename only the specified columns; everything else stays as-is
    if col_map:
        df = df.rename(columns=col_map)

    # Cast dtypes using the *current* column names
    if dtype_map:
        cast_map = {
            col: dtype for col, dtype in dtype_map.items()
            if col in df.columns
        }
        if cast_map:
            df = df.astype(cast_map)

    return df

def read_excel_with_mapper(
    path: str | Path,
    *,
    sheet_name: str | int | None = 0,
    col_map: Mapping[str, str] | None = None,
    dtype_map: Mapping[str, str | type] | None = None,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """
    Read an Excel sheet and optionally:
      - rename some columns using col_map
      - cast dtypes using dtype_map (after renaming)

    Behavior
    --------
    - All columns in the sheet are read (unless restricted via read_kwargs).
    - Only columns present in col_map are renamed; others keep their original name.
    - dtype_map is applied on the *final* column names
      (after renaming; or original if not renamed).
    - If both col_map and dtype_map are None, this is effectively:
        pd.read_excel(path, sheet_name=sheet_name, **read_kwargs)

    Parameters
    ----------
    path :
        Path to the Excel file.
    sheet_name :
        Sheet to read, passed through to pd.read_excel.
        - int     -> 0-based sheet index
        - str     -> sheet name
        - None    -> all sheets (but this helper assumes a single sheet,
                    so keep this as int/str in normal use).
    col_map :
        Mapping of original column names -> new names.
        Example: {"API_UWI": "api", "Prod Date": "prod_dt"}.
    dtype_map :
        Mapping of final column names -> dtypes.
        Example: {"api": "string", "prod_dt": "datetime64[ns]", "gas_mcf": "float64"}.
    **read_kwargs :
        Extra args forwarded to pd.read_excel (e.g. skiprows, header, engine, etc.).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)

    df = pd.read_excel(path, sheet_name=sheet_name, **read_kwargs)

    # If user accidentally passed sheet_name=None and got dict-of-DataFrames,
    # fail loudly so behavior is explicit.
    if isinstance(df, dict):
        raise ValueError(
            "read_excel_with_mapper expects a single sheet. "
            "Got a dict of DataFrames (sheet_name=None). "
            "Pass a specific sheet_name (int or str)."
        )

    # Rename only the specified columns; everything else stays as-is
    if col_map:
        df = df.rename(columns=col_map)

    # Cast dtypes using the *current* column names
    if dtype_map:
        cast_map = {
            col: dtype for col, dtype in dtype_map.items()
            if col in df.columns
        }
        if cast_map:
            df = df.astype(cast_map)

    return df

def drop_duplicates_keep_max_last_prod(
    header_df,
    uwi_col: str = "uwi",
    last_prod_col: str = "last_prod_date",
) -> "pd.DataFrame":
    """
    For duplicated UWIs in the header dataframe, keep exactly one row per UWI:
      - If at least one row has a non-null last_prod_date, keep one row with the maximum last_prod_date.
      - If all rows for a UWI have null last_prod_date, keep the first occurrence for that UWI.

    The function is implemented using groupby/transform and vectorized filtering so it is efficient
    for large dataframes.

    Parameters
    ----------
    header_df : pd.DataFrame
        Header dataframe containing UWI and last production date columns.
    uwi_col : str, default "uwi"
        Column name for UWI.
    last_prod_col : str, default "last_prod_date"
        Column name for the last production date (datetime-like). Non-datetime values will be coerced.

    Returns
    -------
    pd.DataFrame
        A new dataframe with duplicates by UWI removed, keeping one row per UWI per rules above.
        Original dataframe index is preserved (rows returned in ascending index order).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "uwi": ["A","A","B","B","C"],
    ...     "last_prod_date": [pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"), pd.NaT, pd.Timestamp("2019-06-01"), pd.NaT],
    ...     "value": [1,2,3,4,5]
    ... })
    >>> drop_duplicates_keep_max_last_prod(df)
         uwi last_prod_date  value
    1    A    2022-01-01       2
    3    B    2019-06-01       4
    4    C           NaT       5

    Notes
    -----
    - Ties on last_prod_date (multiple rows with same max date) keep the first occurrence by original index.
    - The function coerces last_prod_col to datetime (errors -> NaT) to ensure robust comparisons.
    """
    if uwi_col not in header_df.columns:
        raise KeyError(f"UWI column '{uwi_col}' not found in dataframe")
    if last_prod_col not in header_df.columns:
        raise KeyError(f"last production column '{last_prod_col}' not found in dataframe")

    # Work on a view; preserve original index for ordering
    df = header_df

    # Ensure datetime-like (coerce invalid values to NaT)
    last_prod = pd.to_datetime(df[last_prod_col], errors="coerce")

    # group-wise maximum (Timestamp or NaT)
    group_max = last_prod.groupby(df[uwi_col]).transform("max")

    # Rows that equal the group max (covers groups with at least one non-null value)
    mask_is_max = last_prod == group_max
    max_candidates = df[mask_is_max].drop_duplicates(subset=uwi_col, keep="first")

    # Groups where the group's max is NaT (all rows in that group have NaT)
    groups_all_na_mask = group_max.isna()
    first_of_all_na = df[groups_all_na_mask].drop_duplicates(subset=uwi_col, keep="first")

    # Combine indices to keep and return rows in original index order
    keep_idx = pd.Index(list(max_candidates.index) + list(first_of_all_na.index)).sort_values()
    return df.loc[keep_idx].copy()

def add_bench_columns_to_spacing(
    df_spacing: pd.DataFrame,
    df_header: pd.DataFrame,
    reorder_columns_func,
) -> pd.DataFrame:
    """
    Adds 'bench_i' and 'bench_k' columns to the spacing DataFrame by mapping well IDs to their benches,
    and reorders columns for better readability.

    Parameters
    ----------
    df_spacing : pd.DataFrame
        DataFrame containing well spacing results with columns 'well_i' and 'well_k'.
    df_header : pd.DataFrame
        DataFrame containing well header information with columns 'uwi' and 'bench'.
    reorder_columns_func : callable
        Function to reorder columns in a DataFrame. Should accept arguments:
        - df: DataFrame to reorder
        - columns_to_move: list of columns to move
        - reference_column: column after which to insert the moved columns

    Returns
    -------
    pd.DataFrame
        The input spacing DataFrame with 'bench_i' and 'bench_k' columns added and columns reordered.

    Examples
    --------
    >>> df_header = pd.DataFrame({'uwi': ['A', 'B'], 'bench': ['Bench1', 'Bench2']})
    >>> df_spacing = pd.DataFrame({'well_i': ['A', 'B'], 'well_k': ['B', 'A'], '3D_dist': [100, 200]})
    >>> def reorder_columns(df, columns_to_move, reference_column):
    ...     cols = list(df.columns)
    ...     for col in columns_to_move:
    ...         cols.remove(col)
    ...     ref_idx = cols.index(reference_column) + 1
    ...     for col in reversed(columns_to_move):
    ...         cols.insert(ref_idx, col)
    ...     return df[cols]
    >>> result = add_bench_columns_to_spacing(df_spacing, df_header, reorder_columns)
    >>> result[['well_i', 'bench_i', 'well_k', 'bench_k']]
      well_i  bench_i well_k  bench_k
    0      A  Bench1      B   Bench2
    1      B  Bench2      A   Bench1
    """
    bench_map = df_header.set_index("uwi")["bench"]
    df_spacing = df_spacing.copy()
    df_spacing["bench_i"] = df_spacing["well_i"].map(bench_map)
    df_spacing["bench_k"] = df_spacing["well_k"].map(bench_map)
    df_spacing = reorder_columns_func(df=df_spacing, columns_to_move=['bench_i', 'bench_k'], reference_column='well_k')
    df_spacing = reorder_columns_func(
        df=df_spacing,
        columns_to_move=['direction_to_k_from_i_axis', 'overlap_pct_i', 'overlap_pct_k', 'overlap_len_common_ft', 'LL_i', 'LL_k'],
        reference_column='3D_dist'
    )
    return df_spacing

#%% BG_RCAT Computation

def compute_bg_rcat(
    df: pd.DataFrame,
    *,
    col_map: Mapping[str, str] | None = None,
    prod_cutoff: Union[str, pd.Timestamp] = "2025-01-01",
    spud_cutoff: Union[str, pd.Timestamp] = "2023-01-01",
    utc: bool = False,
) -> pd.Series:
    """
    Compute BG_RCAT classification for a well list.

    This function implements the BG_RCAT logic based on well status and
    key dates (spud, completion, last production).

    Column mapping
    --------------
    The function works with four *logical* columns:

        - 'status'    -> well status (e.g. 'WellStatus_Env')
        - 'last_prod' -> last production date
        - 'spud'      -> spud date
        - 'comp'      -> completion date

    By default, it assumes the following actual column names in `df`:

        status    : 'WellStatus_Env'
        last_prod : 'LastProdDt'
        spud      : 'SpudDt'
        comp      : 'CompDt'

    You can override these via the `col_map` parameter, e.g.:

        col_map = {
            "status": "Status",
            "last_prod": "Last_Prod_Date",
            "spud": "Spud_Date",
            "comp": "Completion_Date",
        }

    Parameters
    ----------
    df :
        DataFrame containing at least the four required columns
        (directly or via col_map).

    col_map :
        Optional mapping from logical names -> actual column names in `df`.

        Valid keys in col_map:
            - "status"
            - "last_prod"
            - "spud"
            - "comp"

        Any missing keys will fall back to the defaults:
            status    -> 'WellStatus_Env'
            last_prod -> 'LastProdDt'
            spud      -> 'SpudDt'
            comp      -> 'CompDt'

    prod_cutoff :
        Production recency cutoff. Wells with LastProdDt >= prod_cutoff
        and status PRODUCING/INACTIVE PRODUCER/TA are treated as current PDP.
        Default is '2025-01-01'.

    spud_cutoff :
        Spud recency cutoff. DUC-like wells with SpudDt >= spud_cutoff
        are '2DUC'; older DUC-like wells are '9XDUC'. Default '2023-01-01'.

    Returns
    -------
    pd.Series
        Series of BG_RCAT codes:
        '1PDP', '1PDSI', '1WOP', '2DUC', '3PRMT',
        '9PA', '9XDUC', '9XPMT'.

    Raises
    ------
    KeyError
        If any of the required logical columns ("status", "last_prod",
        "spud", "comp") is not found in the DataFrame after applying col_map.

    BG_RCAT code meanings
    ---------------------
    The function returns a series of compact reserve/status classification
    codes. They are derived from a well's status and key dates using the
    `prod_cutoff` and `spud_cutoff` thresholds.

    High-level structure
    ^^^^^^^^^^^^^^^^^^^^
    - Leading digit:
      - ``1***`` – current PDP or PDP-adjacent (producing / shut-in / WOP).
      - ``2***`` – near-term inventory (DUCs likely to be completed).
      - ``3***`` – future locations (permits).
      - ``9***`` – non-working buckets (dead permits, aged DUCs, P&A / abandoned).

    - Suffix:
      - ``PDP``  – proved developed producing (current).
      - ``PDSI`` – PDP with stale or shut-in production.
      - ``WOP``  – waiting on production (completed, no sales yet).
      - ``DUC``  – drilled but uncompleted.
      - ``PRMT`` – permitted location.
      - ``PA``   – plugged / abandoned or long-term non-producing.
      - ``XDUC`` – aged-out / failed DUC.
      - ``XPMT`` – dead permit (cancelled / expired).

    Code-by-code definitions
    ^^^^^^^^^^^^^^^^^^^^^^^^

    ``1PDP`` – Current Producing PDP
        Wells that are effectively *active PDP*.

        Assigned when:

        - ``LastProdDt`` is not null, and
        - Either:
          - ``WellStatus_Env`` is ``"PRODUCING"`` or ``"INACTIVE PRODUCER"`` with
            ``LastProdDt >= prod_cutoff``, or
          - ``WellStatus_Env`` is ``"TA"`` with ``LastProdDt >= prod_cutoff``.

        Interpretation: these wells anchor the current PDP base and near-term
        cash flow. The ``prod_cutoff`` threshold controls what is considered
        "recent" production.

    ``1PDSI`` – PDP with Stale Production / Shut-in PDP
        Wells that *have* a production history and are classified as producing
        or inactive, but whose last production is **older** than ``prod_cutoff``.

        Assigned when:

        - ``LastProdDt`` is not null,
        - ``WellStatus_Env`` is ``"PRODUCING"`` or ``"INACTIVE PRODUCER"``, and
        - ``LastProdDt < prod_cutoff``.

        Interpretation: structurally PDP, but functionally shut-in or idle. This
        bucket separates older non-flowing PDP from the active ``1PDP`` wells.

    ``1WOP`` – Completed, Waiting on Production
        Wells that are mechanically ready but not yet selling volumes.

        Assigned when:

        - ``LastProdDt`` is null (no recorded production),
        - ``CompDt`` is not null (well is completed), and
        - ``WellStatus_Env`` is ``"COMPLETED"`` or ``"PRODUCING"``.

        Interpretation: very short-term future PDP; these wells are expected to
        roll into ``1PDP`` once first sales hit the database. This is distinct
        from DUC inventory which still requires completion capital.

    ``2DUC`` – Fresh DUC Inventory
        Active, near-term DUC inventory: wells that have been spud, may be
        drilled or partially completed, but have not yet produced and are
        "recent" enough to be considered viable completions.

        Assigned when **all** of the following hold:

        - ``LastProdDt`` is null,
        - ``WellStatus_Env`` is one of:
          ``"DUC"``, ``"DRILLED"``, ``"DRILLING"``, ``"SPUD DATE ONLY"``,
          ``"COMPLETED"``, ``"PRODUCING"``,
        - The well has not already been assigned another BG_RCAT code (e.g.
          not captured by ``1WOP``, permits, etc.),
        - ``SpudDt >= spud_cutoff``.

        Interpretation: near-term growth inventory with capital already sunk
        into drilling. ``spud_cutoff`` filters out very old, zombie DUCs.

    ``3PRMT`` – Active Permits
        Future locations that have a **valid** drilling permit but have not
        yet been spud.

        Assigned when:

        - ``WellStatus_Env == "PERMITTED"``.

        This is applied after cancelled/expired permits are classified as
        ``9XPMT``.

        Interpretation: undeveloped locations with regulatory approval but no
        drilling commitment. Typically treated as longer-dated inventory than
        ``2DUC``, but more concrete than conceptual locations.

    ``9PA`` – Plugged / Abandoned / Long-Term Non-Producing
        Wells that are effectively dead from an inventory standpoint: plugged,
        abandoned, or long-term non-producing TA wells.

        Assigned in several situations, for example:

        - ``TA`` with production **older** than ``prod_cutoff``.
        - ``P & A`` or ``ABANDONED`` with any production history.
        - ``TA`` with no production, not already coded.
        - ``P & A`` with no production but a *recent* spud (``SpudDt >= spud_cutoff``).
        - ``ABANDONED`` with no production (safety catch).

        Interpretation: wells you do not plan to re-activate and do not treat
        as live inventory. Keeping them separate from ``9XDUC`` / ``9XPMT``
        distinguishes truly abandoned wells from abandoned inventory.

    ``9XDUC`` – Aged-Out / Failed DUCs
        Washed-out DUCs and similar cases: wells that look like DUCs
        (or early-life wells) on paper but are so old that they are no longer
        considered viable completions.

        Assigned when:

        - ``LastProdDt`` is null, and
        - Either:
          - ``WellStatus_Env == "P & A"`` with ``SpudDt < spud_cutoff``, or
          - ``WellStatus_Env`` is in
            ``"DUC"``, ``"DRILLED"``, ``"DRILLING"``, ``"SPUD DATE ONLY"``,
            ``"COMPLETED"``, ``"PRODUCING"`` with ``SpudDt < spud_cutoff``,
            and not already assigned another code.

        Interpretation: the main difference between ``2DUC`` and ``9XDUC`` is
        the age of the spud date. ``9XDUC`` is used to write off stale DUC
        inventory that has sat idle beyond a realistic completion window.

    ``9XPMT`` – Dead Permits (Cancelled / Expired)
        Dead permits: originally permitted locations that have either expired or
        been cancelled and are no longer considered drillable under the current
        permit.

        Assigned when:

        - ``WellStatus_Env`` is ``"PERMIT CANCELLED"`` or ``"PERMIT EXPIRED"``.

        This classification is applied at the start, so these records are always
        treated as dead permits rather than future inventory.

        Interpretation: historical permit records that no longer tie directly
        to drillable inventory. Separating them from ``3PRMT`` avoids over-
        counting future locations when rolling up inventories.
    """
    # Resolve column names (logical -> actual DataFrame columns)
    defaults = {
        "status": "WellStatus_Env",
        "last_prod": "LastProdDt",
        "spud": "SpudDt",
        "comp": "CompDt",
    }
    if col_map:
        defaults.update(col_map)

    status_col = defaults["status"]
    last_prod_col = defaults["last_prod"]
    spud_col = defaults["spud"]
    comp_col = defaults["comp"]

    missing = [c for c in [status_col, last_prod_col, spud_col, comp_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"compute_bg_rcat: required columns not found in DataFrame: {missing}. "
            f"Resolved from logical names using col_map={col_map!r}."
        )

    if utc:
        prod_cutoff_ts = pd.to_datetime(prod_cutoff, utc=True)
        spud_cutoff_ts = pd.to_datetime(spud_cutoff, utc=True)
    else:
        prod_cutoff_ts = pd.to_datetime(prod_cutoff)
        spud_cutoff_ts = pd.to_datetime(spud_cutoff)

    work = df.copy()

    # Coerce dates on the resolved columns
    work[last_prod_col] = pd.to_datetime(work[last_prod_col], errors="coerce")
    work[spud_col] = pd.to_datetime(work[spud_col], errors="coerce")
    work[comp_col] = pd.to_datetime(work[comp_col], errors="coerce")

    status = work[status_col].fillna("")
    last_prod = work[last_prod_col]
    spud = work[spud_col]
    comp = work[comp_col]

    # Start with blank codes
    result = pd.Series("", index=work.index, dtype="object")

    # 1) Permits
    is_cancelled = status.isin(["PERMIT CANCELLED", "PERMIT EXPIRED"])
    result[is_cancelled] = "9XPMT"

    is_permitted = status.eq("PERMITTED")
    result[is_permitted] = "3PRMT"

    has_last_prod = last_prod.notna()
    no_last_prod = ~has_last_prod

    # 2) Wells WITH production history

    # 2a) PDP / PDSI for PRODUCING & INACTIVE PRODUCER
    m_prod_like = has_last_prod & status.isin(["PRODUCING", "INACTIVE PRODUCER"])
    m_pdp = m_prod_like & (last_prod >= prod_cutoff_ts)
    m_pdsi = m_prod_like & (last_prod < prod_cutoff_ts)
    result[m_pdp] = "1PDP"
    result[m_pdsi] = "1PDSI"

    # 2b) TA with production
    m_ta_prod = has_last_prod & status.eq("TA")
    result[m_ta_prod & (last_prod >= prod_cutoff_ts)] = "1PDP"
    result[m_ta_prod & (last_prod < prod_cutoff_ts)] = "9PA"

    # 2c) P&A and ABANDONED with production
    m_pa_prod = has_last_prod & status.isin(["P & A", "ABANDONED"])
    result[m_pa_prod] = "9PA"

    # 3) Wells WITHOUT production history

    has_comp = comp.notna()

    # 3a) 1WOP: completed, waiting on production
    m_wop = no_last_prod & has_comp & status.isin(["COMPLETED", "PRODUCING"])
    result[m_wop] = "1WOP"

    # 3b) TA without production -> 9PA
    m_ta_no_prod = no_last_prod & status.eq("TA") & (result == "")
    result[m_ta_no_prod] = "9PA"

    # 3c) P&A without production: old spuds = 9XDUC, recent = 9PA
    m_pa_no_prod = no_last_prod & status.eq("P & A")
    m_pa_old_duc = m_pa_no_prod & (spud < spud_cutoff_ts)
    m_pa_recent = m_pa_no_prod & (spud >= spud_cutoff_ts)
    result[m_pa_old_duc] = "9XDUC"
    result[m_pa_recent] = "9PA"

    # 3d) DUC-like inventory (no prod, no completion, still in the drilling/DUC bucket)
    m_duc_status = status.isin(
        ["DUC", "DRILLED", "DRILLING", "SPUD DATE ONLY", "COMPLETED", "PRODUCING"]
    )
    m_duc_like = no_last_prod & m_duc_status & (result == "")
    m_duc_recent = m_duc_like & (spud >= spud_cutoff_ts)
    m_duc_old = m_duc_like & (spud < spud_cutoff_ts)
    result[m_duc_recent] = "2DUC"
    result[m_duc_old] = "9XDUC"

    # 4) ABANDONED with no production (safety catch)
    m_abandoned_no_prod = no_last_prod & status.eq("ABANDONED") & (result == "")
    result[m_abandoned_no_prod] = "9PA"

    return result


#%% RSV_CAT Classification
# Status normalization mappings for IHS and Enverus sources

IHS_STATUS_MAP: dict[str, str] = {
    # Producers
    "OIL PRODUCER": "PRODUCER",
    "OIL PRODUCER-OLD WEL": "PRODUCER",
    "GAS PRODUCER": "PRODUCER",
    "GAS PRODUCER-OLD WEL": "PRODUCER",
    "CARBON DIOXIDE INJEC": "INJECTOR",
    "WATER INJECTION WELL": "INJECTOR",
    "WATER INJECTOR - ENH": "INJECTOR",
    "SALT WATER DISPOSAL": "INJECTOR",
    # Abandoned producers
    "ABANDONED OIL PRODUC": "ABANDONED_PRODUCER",
    "ABANDONED GAS PRODUC": "ABANDONED_PRODUCER",
    "ABANDONED INJECTION": "ABANDONED_PRODUCER",
    "ABANDONED WELLBORE O": "ABANDONED_PRODUCER",
    # TA
    "TEMPORARILY ABANDONE": "TA",
    # Dry/Abandoned
    "DRY & ABANDONED": "ABANDONED",
    "DRY & ABANDONED-OLD": "ABANDONED",
    "ABANDONED": "ABANDONED",
    # Permits
    "APPROVED PERMIT": "PERMITTED",
    "PERMIT APPROVED": "PERMITTED",
    "PERMITTED": "PERMITTED",
    "UNAPPROVED PERMIT": "UNAPPROVED_PERMIT",
    "PERMIT PENDING": "UNAPPROVED_PERMIT",
    "PERMIT CANCELLED": "PERMIT_CANCELLED",
    "PERMIT EXPIRED": "PERMIT_EXPIRED",
    # Spud/Drilled
    "DRILLED": "DRILLED",
    "DRILLING": "DRILLING",
    "DUC": "DUC",
    "SPUD DATE ONLY": "DRILLED",
    "TREATED PENDING COMP": "DRILLED",
    # P&A
    "P & A": "PA",
    "P&A": "PA",
    "PLUGGED": "PA",
}

ENVERUS_STATUS_MAP: dict[str, str] = {
    "ACTIVE": "PRODUCER",
    "PRODUCING": "PRODUCER",
    "INACTIVE": "INACTIVE_PRODUCER",
    "INACTIVE PRODUCER": "INACTIVE_PRODUCER",
    "DRILLED": "DRILLED",
    "DRILLING": "DRILLING",
    "DUC": "DUC",
    "SPUD DATE ONLY": "DRILLED",
    "COMPLETED": "COMPLETED",
    "PERMITTED": "PERMITTED",
    "PERMIT APPROVED": "PERMITTED",
    "PERMIT CANCELLED": "PERMIT_CANCELLED",
    "PERMIT EXPIRED": "PERMIT_EXPIRED",
    "ABANDONED": "ABANDONED",
    "P&A": "PA",
    "P & A": "PA",
    "TA": "TA",
    "TEMPORARILY ABANDONED": "TA",
}


def _normalize_status_for_rsv_cat(
    status_series: pd.Series,
    source_type: str = "auto",
) -> pd.Series:
    """
    Normalize status values from IHS or Enverus to internal codes.

    Parameters
    ----------
    status_series : pd.Series
        Raw status values from the source.
    source_type : str
        "ihs", "enverus", or "auto" (detect from values).

    Returns
    -------
    pd.Series
        Normalized status codes.
    """
    status_upper = status_series.fillna("").astype(str).str.upper().str.strip()

    # Auto-detect source type based on typical values
    if source_type == "auto":
        sample = status_upper.head(200)
        # IHS typically has detailed status like "OIL PRODUCER", "GAS PRODUCER"
        if sample.str.contains("OIL PRODUCER|GAS PRODUCER|ABANDONED OIL|ABANDONED GAS").any():
            source_type = "ihs"
        # Enverus uses simpler "ACTIVE", "INACTIVE"
        elif sample.str.contains("^ACTIVE$|^INACTIVE$", regex=True).any():
            source_type = "enverus"
        else:
            source_type = "ihs"  # default fallback

    # Apply appropriate mapping
    if source_type == "ihs":
        return status_upper.map(IHS_STATUS_MAP).fillna("UNKNOWN")
    else:
        return status_upper.map(ENVERUS_STATUS_MAP).fillna("UNKNOWN")


def compute_rsv_cat(
    df: pd.DataFrame,
    *,
    col_map: Mapping[str, str] | None = None,
    prod_cutoff_months: int = 6,
    duc_age_years: int = 3,
    permit_window_years: int = 2,
    reference_date: Union[str, pd.Timestamp] | None = None,
    source_type: str = "auto",
    utc: bool = False,
) -> pd.Series:
    """
    Compute RSV_CAT classification for well header data.

    This function implements RSV_CAT (Reserve Category) logic based on well
    status and key dates (spud, completion, first/last production, permit).
    Supports both IHS and Enverus data sources via flexible column mapping.

    Column mapping
    --------------
    The function works with these *logical* columns:

        - 'status'      -> well status (e.g. 'Final Status' for IHS)
        - 'prod_status' -> current production status (IHS: A/P, optional)
        - 'last_prod'   -> last production/activity date
        - 'first_prod'  -> first production date
        - 'spud'        -> spud date
        - 'comp'        -> completion date
        - 'permit_date' -> permit/license date

    By default, it assumes the following actual column names in `df` (IHS format):

        status      : 'Final Status'
        prod_status : 'Current Production Status'
        last_prod   : 'Last Activity Date'
        first_prod  : 'First Prod Date'
        spud        : 'Spud Date'
        comp        : 'Comp Date'
        permit_date : 'Permit License Date'

    You can override these via the `col_map` parameter, e.g.:

        col_map = {
            "status": "Well Status",
            "last_prod": "Last Prod Date",
            "first_prod": "First Prod Date",
            "spud": "Spud Date",
            "comp": "Completion Date",
        }

    Parameters
    ----------
    df :
        DataFrame containing at least the required columns
        (directly or via col_map). Only 'status' and 'spud' are strictly required;
        other columns are used if available.

    col_map :
        Optional mapping from logical names -> actual column names in `df`.

        Valid keys in col_map:
            - "status"
            - "prod_status"
            - "last_prod"
            - "first_prod"
            - "spud"
            - "comp"
            - "permit_date"

        Any missing keys will fall back to the defaults (IHS column names).

    prod_cutoff_months :
        Months without production to classify as 02PDNP.
        Default is 6 months.

    duc_age_years :
        Years after spud to classify as "Old DUC" instead of 03PUD.
        Default is 3 years.

    permit_window_years :
        Years for permit development window. Permits older than this
        become "Expired Perm". Default is 2 years.

    reference_date :
        Reference date for calculations. Defaults to today if not specified.

    source_type :
        Data source type for status normalization: "ihs", "enverus", or "auto".
        "auto" will attempt to detect the source from status values.
        Default is "auto".

    utc :
        Whether to parse dates as UTC-aware. Default is False.

    Returns
    -------
    pd.Series
        Series of RSV_CAT codes:
        '01PDP', '02PA', '02PDNP', '03PA', '03PUD', '05PA',
        '05PBUD', '07PBUD', 'Old DUC', 'Expired Perm'.

    Raises
    ------
    KeyError
        If any of the required logical columns ("status", "spud") is not found
        in the DataFrame after applying col_map.

    RSV_CAT code meanings
    ---------------------
    The function returns a series of reserve/status classification codes.
    They are derived from a well's status and key dates using configurable
    thresholds.

    Code-by-code definitions
    ^^^^^^^^^^^^^^^^^^^^^^^^

    ``01PDP`` – Proved Developed Producing
        Current producer per IHS or check data. Wells that are actively
        producing with recent production within `prod_cutoff_months`.

    ``02PA`` – P&A Producer
        Abandoned or former producer per IHS or check data. Wells that
        have production history but are now abandoned or P&A.

    ``02PDNP`` – Proved Developed Non-Producing
        Producing wells with no reported production or check data for
        >= `prod_cutoff_months`. Still classified as producing status but
        stale production.

    ``03PA`` – Abandoned Spud Well
        Abandoned spud wells per IHS. No production, no check data.
        Wells that were spud but abandoned without producing.

    ``03PUD`` – Proved Undeveloped
        Locations with a confirmed spud date per IHS. Recent DUCs
        (spud within `duc_age_years`) that are expected to be completed.

    ``05PA`` – Abandoned Permit
        Abandoned/cancelled permit per IHS. Permits that were cancelled
        before drilling commenced.

    ``05PBUD`` – Approved Permit
        Approved permits that are within the development window by regulators.
        Valid permits that can still be drilled.

    ``07PBUD`` – Unapproved Permit
        Unapproved permits per IHS. Permits pending regulatory approval.

    ``Old DUC`` – Old Spud
        Spud wells not online >= `duc_age_years` years after spud date.
        Stale DUC inventory unlikely to be completed.

    ``Expired Perm`` – Expired Permit
        Permits that have exceeded the development window by regulators.
        Permits older than `permit_window_years` that haven't been drilled.

    Examples
    --------
    Using IHS data (default column names):

    >>> df_ihs = pd.read_excel("IHS Well Header List.xlsx")
    >>> df_ihs["rsv_cat"] = compute_rsv_cat(df_ihs)

    Using Enverus data with custom column mapping:

    >>> df_env = pd.read_csv("ENV_test_Well Headers.CSV")
    >>> enverus_map = {
    ...     "status": "Well Status",
    ...     "last_prod": "Last Prod Date",
    ...     "first_prod": "First Prod Date",
    ...     "spud": "Spud Date",
    ...     "comp": "Completion Date",
    ... }
    >>> df_env["rsv_cat"] = compute_rsv_cat(df_env, col_map=enverus_map)

    With configurable thresholds:

    >>> rsv_cat = compute_rsv_cat(
    ...     df,
    ...     prod_cutoff_months=12,  # 12 months instead of 6
    ...     duc_age_years=5,        # 5 years instead of 3
    ...     permit_window_years=3,  # 3 years instead of 2
    ... )
    """
    # Resolve column names (logical -> actual DataFrame columns)
    defaults = {
        "status": "Final Status",
        "prod_status": "Current Production Status",
        "last_prod": "Last Activity Date",
        "first_prod": "First Prod Date",
        "spud": "Spud Date",
        "comp": "Comp Date",
        "permit_date": "Permit License Date",
    }
    if col_map:
        defaults.update(col_map)

    status_col = defaults["status"]
    prod_status_col = defaults.get("prod_status")
    last_prod_col = defaults["last_prod"]
    first_prod_col = defaults.get("first_prod")
    spud_col = defaults["spud"]
    comp_col = defaults.get("comp")
    permit_date_col = defaults.get("permit_date")

    # Required columns
    required = [status_col, spud_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"compute_rsv_cat: required columns not found in DataFrame: {missing}. "
            f"Resolved from logical names using col_map={col_map!r}."
        )

    # Reference date for calculations
    if reference_date is None:
        ref_date = pd.Timestamp.now(tz="UTC" if utc else None)
    else:
        ref_date = pd.to_datetime(reference_date, utc=utc) if utc else pd.to_datetime(reference_date)

    # Calculate cutoff timestamps
    prod_cutoff_ts = ref_date - pd.DateOffset(months=prod_cutoff_months)
    duc_cutoff_ts = ref_date - pd.DateOffset(years=duc_age_years)
    permit_cutoff_ts = ref_date - pd.DateOffset(years=permit_window_years)

    work = df.copy()

    # Coerce dates on resolved columns (only if they exist)
    if spud_col in work.columns:
        work[spud_col] = pd.to_datetime(work[spud_col], errors="coerce")
    if last_prod_col in work.columns:
        work[last_prod_col] = pd.to_datetime(work[last_prod_col], errors="coerce")
    if first_prod_col and first_prod_col in work.columns:
        work[first_prod_col] = pd.to_datetime(work[first_prod_col], errors="coerce")
    if comp_col and comp_col in work.columns:
        work[comp_col] = pd.to_datetime(work[comp_col], errors="coerce")
    if permit_date_col and permit_date_col in work.columns:
        work[permit_date_col] = pd.to_datetime(work[permit_date_col], errors="coerce")

    # Extract series for convenience
    status_raw = work[status_col].fillna("")
    status = _normalize_status_for_rsv_cat(status_raw, source_type=source_type)

    spud = work[spud_col] if spud_col in work.columns else pd.Series(pd.NaT, index=work.index)
    last_prod = work[last_prod_col] if last_prod_col in work.columns else pd.Series(pd.NaT, index=work.index)
    first_prod = work[first_prod_col] if first_prod_col and first_prod_col in work.columns else pd.Series(pd.NaT, index=work.index)
    comp = work[comp_col] if comp_col and comp_col in work.columns else pd.Series(pd.NaT, index=work.index)
    permit_date = work[permit_date_col] if permit_date_col and permit_date_col in work.columns else pd.Series(pd.NaT, index=work.index)

    # Production status (IHS A/P) if available
    if prod_status_col and prod_status_col in work.columns:
        prod_status = work[prod_status_col].fillna("").astype(str).str.upper().str.strip()
    else:
        prod_status = pd.Series("", index=work.index)

    # Start with blank codes
    result = pd.Series("", index=work.index, dtype="object")

    # Helper masks
    # For production status, we need to be careful:
    # - first_prod indicates definite production
    # - comp indicates a completed well (ready for production)
    # - last_prod (Last Activity Date in IHS) only indicates production for PRODUCER status
    #   For non-producer statuses (DRILLED, DUC, etc.), last_prod is just activity, not production
    m_producer_status = status.isin(["PRODUCER", "INACTIVE_PRODUCER", "ABANDONED_PRODUCER", "INJECTOR"])
    has_prod = first_prod.notna() | (last_prod.notna() & m_producer_status) | (comp.notna() & m_producer_status)
    no_prod = ~has_prod
    has_spud = spud.notna()

    # ==========================================================================
    # Classification Logic (Priority Order)
    # ==========================================================================

    # --- Rule 1: 01PDP - Proved Developed Producing ---
    # Current producer with recent production/activity
    m_producer = status.isin(["PRODUCER", "INACTIVE_PRODUCER", "COMPLETED", "INJECTOR"])
    m_recent_prod = last_prod.notna() & (last_prod >= prod_cutoff_ts)
    m_pdp = m_producer & has_prod & m_recent_prod
    result[m_pdp] = "01PDP"

    # --- Rule 2: 02PDNP - Proved Developed Non-Producing ---
    # Producing wells with stale production (> prod_cutoff_months)
    m_stale_prod = last_prod.notna() & (last_prod < prod_cutoff_ts)
    m_pdnp = m_producer & has_prod & m_stale_prod & (result == "")
    result[m_pdnp] = "02PDNP"

    # Also handle producers with no last_prod date but has first_prod/comp
    m_producer_no_recent = m_producer & has_prod & last_prod.isna() & (result == "")
    result[m_producer_no_recent] = "02PDNP"

    # --- Rule 3: 02PA - P&A Producer ---
    # Abandoned or former producer with production history
    m_abandoned_prod = status.isin(["ABANDONED_PRODUCER", "PA"]) & has_prod
    result[m_abandoned_prod & (result == "")] = "02PA"

    # Also handle TA (temporarily abandoned) with production
    m_ta_prod = status.eq("TA") & has_prod
    result[m_ta_prod & (result == "")] = "02PA"

    # Also handle prod_status = "P" (IHS plugged) with production
    m_plugged_prod = prod_status.eq("P") & has_prod & (result == "")
    result[m_plugged_prod] = "02PA"

    # --- Rule 4: 03PA - Abandoned Spud Well ---
    # Abandoned spud wells - no production
    m_spud_abandoned = status.isin(["ABANDONED", "PA", "TA", "ABANDONED_PRODUCER"]) & no_prod & has_spud
    result[m_spud_abandoned & (result == "")] = "03PA"

    # --- Rule 5: Old DUC - Old Spud ---
    # Spud wells not online >= duc_age_years after spud date
    m_duc_status = status.isin(["DRILLED", "DUC", "DRILLING", "COMPLETED", "UNKNOWN"])
    m_old_duc = no_prod & has_spud & (spud < duc_cutoff_ts) & m_duc_status
    result[m_old_duc & (result == "")] = "Old DUC"

    # --- Rule 6: 03PUD - Proved Undeveloped ---
    # Recent DUCs/drilled wells (spud within duc_age_years)
    m_recent_duc = no_prod & has_spud & (spud >= duc_cutoff_ts) & m_duc_status
    result[m_recent_duc & (result == "")] = "03PUD"

    # --- Rule 7: Expired Perm - Expired Permit ---
    # Permits that have explicitly expired OR permits older than permit_window_years
    m_permit_expired_status = status.eq("PERMIT_EXPIRED")
    m_permit_old = status.eq("PERMITTED") & permit_date.notna() & (permit_date < permit_cutoff_ts)
    result[(m_permit_expired_status | m_permit_old) & (result == "")] = "Expired Perm"

    # --- Rule 8: 05PA - Abandoned Permit ---
    # Cancelled permits
    m_permit_cancelled = status.eq("PERMIT_CANCELLED")
    result[m_permit_cancelled & (result == "")] = "05PA"

    # --- Rule 9: 07PBUD - Unapproved Permit ---
    # Unapproved/pending permits
    m_unapproved = status.eq("UNAPPROVED_PERMIT")
    result[m_unapproved & (result == "")] = "07PBUD"

    # --- Rule 10: 05PBUD - Approved Permit ---
    # Approved permits within development window
    m_approved_permit = status.eq("PERMITTED") & (result == "")
    result[m_approved_permit] = "05PBUD"

    return result


#%% Helper to compute producing-month numbers (SQL ROW_NUMBER() OVER (PARTITION BY uwi ORDER BY prod_date)).

def add_producing_months(
    df: pd.DataFrame,
    uwi_col: str = "uwi",
    date_col: str = "prod_date",
    out_col: str = "producing_month_number",
    sort_ascending: bool = True,
    inplace: bool = False,
    dropna_dates: bool = False,
    secondary_sort: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with a producing-month counter per UWI (like ROW_NUMBER partitioned by uwi ordered by date).

    Parameters
    - df: input DataFrame
    - uwi_col: column name containing UWI identifiers
    - date_col: column name with production date (will be converted to datetime if not already)
    - out_col: name of the output column to create
    - sort_ascending: sort order for dates (True -> oldest first)
    - inplace: if True, modify input df; otherwise operate on a copy
    - dropna_dates: if True, drop rows where date_col is NA before numbering
    - secondary_sort: optional list/tuple of additional column names to include in sort keys after date_col

    Returns
    - DataFrame with out_col added (int, starting at 1 for each uwi)
    """
    if not inplace:
        df = df.copy()

    # ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if dropna_dates:
        df = df[df[date_col].notna()]

    # build sort keys: uwi first, then date, then any secondary sorts
    sort_keys = [uwi_col, date_col]
    if secondary_sort:
        sort_keys.extend(list(secondary_sort))

    # determine ascending flags: uwi always ascending, date ascending per sort_ascending, secondary assumed ascending
    ascending = [True, sort_ascending] + [True] * (len(sort_keys) - 2)

    # sort and assign row number per uwi
    df = df.sort_values(sort_keys, ascending=ascending)
    df[out_col] = df.groupby(uwi_col).cumcount() + 1

    return df

#%%

def calculate_cumulative_volumes_by_period(
    df: pd.DataFrame,
    uwi_col: str = "uwi",
    prod_date_col: str = "prod_date",
    lateral_length_col: str = "di_lateral_length",
    oil_col: str = "monthly_oil",
    gas_col: str = "monthly_gas",
    water_col: str = "monthly_water",
) -> pd.DataFrame:
    """
    For each UWI, calculate cumulative oil/gas/water volumes for the first
    180 and 365 days of production (based on actual calendar days in each
    production month), and normalize those volumes by lateral length.

    Returns one row per UWI with:
      - cum_oil_180d, cum_gas_180d, cum_water_180d
      - cum_oil_365d, cum_gas_365d, cum_water_365d
      - *_per_ft versions normalized by `di_lateral_length`.
    """
    df_work = df.copy()

    # Ensure prod_date is datetime
    df_work[prod_date_col] = pd.to_datetime(df_work[prod_date_col])

    # Sort by UWI + prod_date
    df_work = df_work.sort_values([uwi_col, prod_date_col])

    # 1) Exact days in each production month for that prod_date
    df_work["days_in_month"] = df_work[prod_date_col].dt.daysinmonth

    # 2) Cumulative days on production by end of each month
    df_work["cum_days"] = df_work.groupby(uwi_col)["days_in_month"].cumsum()

    # 3) Masks for 180- and 365-day windows
    mask_180 = df_work["cum_days"] <= 180
    mask_365 = df_work["cum_days"] <= 365

    # 4) Per-UWI cumulative volumes in each window
    agg_180 = (
        df_work.loc[mask_180]
        .groupby(uwi_col)[[oil_col, gas_col, water_col]]
        .sum()
        .rename(
            columns={
                oil_col: "cum_oil_180d",
                gas_col: "cum_gas_180d",
                water_col: "cum_water_180d",
            }
        )
    )

    agg_365 = (
        df_work.loc[mask_365]
        .groupby(uwi_col)[[oil_col, gas_col, water_col]]
        .sum()
        .rename(
            columns={
                oil_col: "cum_oil_365d",
                gas_col: "cum_gas_365d",
                water_col: "cum_water_365d",
            }
        )
    )

    # 5) Lateral length per UWI (assumed constant within a well)
    lateral = (
        df_work.groupby(uwi_col)[lateral_length_col]
        .first()
        .rename("lateral_length_ft")
        .to_frame()
    )

    # 6) Combine into a single per-UWI DataFrame
    result = lateral.join(agg_180, how="left").join(agg_365, how="left")

    # 7) Per-foot versions
    LL = result["lateral_length_ft"]
    for period in ("180d", "365d"):
        for phase in ("oil", "gas", "water"):
            col = f"cum_{phase}_{period}"
            result[f"{col}_per_ft"] = result[col] / LL

    return result.reset_index()
