import pandas as pd
from typing import List # Importing specific types from typing module
import re  # Regular expression module for cleaning column names

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