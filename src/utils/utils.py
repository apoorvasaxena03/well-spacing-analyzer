from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import List, Mapping, Any, Union

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

#%% BG_RCAT Computation

def compute_bg_rcat(
    df: pd.DataFrame,
    *,
    col_map: Mapping[str, str] | None = None,
    prod_cutoff: Union[str, pd.Timestamp] = "2025-01-01",
    spud_cutoff: Union[str, pd.Timestamp] = "2023-01-01",
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