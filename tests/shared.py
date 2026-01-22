"""Minimal shared utilities for DataFrame testing."""

from typing import Optional

import pandas as pd


def assert_dataframes_equal_unordered(
    df1: pd.DataFrame, df2: pd.DataFrame, ignore_columns: Optional[list[str]] = None
) -> None:
    """Assert DataFrames are equal ignoring row order and specified columns.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame to compare
    df2 : pd.DataFrame
        Second DataFrame to compare
    ignore_columns : Optional[List[str]]
        List of column names to ignore during comparison

    Raises
    ------
    AssertionError
        If DataFrames are not equal after sorting and column filtering
    """
    if ignore_columns is None:
        ignore_columns = []

    # Create copies to avoid modifying originals
    df1_filtered = df1.drop(columns=ignore_columns, errors="ignore").copy()
    df2_filtered = df2.drop(columns=ignore_columns, errors="ignore").copy()

    # Sort by all columns to enable comparison regardless of row order
    if not df1_filtered.empty:
        df1_sorted = df1_filtered.sort_values(by=list(df1_filtered.columns)).reset_index(drop=True)
    else:
        df1_sorted = df1_filtered.reset_index(drop=True)

    if not df2_filtered.empty:
        df2_sorted = df2_filtered.sort_values(by=list(df2_filtered.columns)).reset_index(drop=True)
    else:
        df2_sorted = df2_filtered.reset_index(drop=True)

    # Use pandas built-in testing function
    pd.testing.assert_frame_equal(df1_sorted, df2_sorted)


def assert_dataframe_matches_any(
    df: pd.DataFrame,
    df_list: list[pd.DataFrame],
    ignore_columns: Optional[list[str]] = None,
) -> None:
    """Assert DataFrame matches at least one from a list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for matches
    df_list : List[pd.DataFrame]
        List of DataFrames to compare against
    ignore_columns : Optional[List[str]]
        List of column names to ignore during comparison

    Raises
    ------
    AssertionError
        If DataFrame doesn't match any in the list
    """
    if not df_list:
        raise AssertionError("Empty DataFrame list provided")

    for candidate_df in df_list:
        try:
            assert_dataframes_equal_unordered(df, candidate_df, ignore_columns)
            return  # Found a match
        except AssertionError:
            continue  # Try next candidate

    # No match found
    raise AssertionError(f"DataFrame does not match any of the {len(df_list)} candidates")
