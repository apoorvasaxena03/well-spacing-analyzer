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
    prod_cutoff: Union[str, pd.Timestamp] = "2025-01-01",
    spud_cutoff: Union[str, pd.Timestamp] = "2023-01-01",
) -> pd.Series:
    """
    Compute BG_RCAT classification for a well list.

    Parameters
    ----------
    df :
        DataFrame containing at least:
        - 'WellStatus_Env'
        - 'LastProdDt'
        - 'SpudDt'
        - 'CompDt'

        Date columns may be strings or datetimes; they will be
        coerced via `pd.to_datetime(..., errors="coerce")`.

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
    """
    prod_cutoff_ts = pd.to_datetime(prod_cutoff)
    spud_cutoff_ts = pd.to_datetime(spud_cutoff)

    work = df.copy()

    # Coerce dates
    work["LastProdDt"] = pd.to_datetime(work["LastProdDt"], errors="coerce")
    work["SpudDt"] = pd.to_datetime(work["SpudDt"], errors="coerce")
    work["CompDt"] = pd.to_datetime(work["CompDt"], errors="coerce")

    status = work["WellStatus_Env"].fillna("")
    last_prod = work["LastProdDt"]
    spud = work["SpudDt"]
    comp = work["CompDt"]

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

    # 4) ABANDONED with no production (none in current file but safe to handle)
    m_abandoned_no_prod = no_last_prod & status.eq("ABANDONED") & (result == "")
    result[m_abandoned_no_prod] = "9PA"

    return result
