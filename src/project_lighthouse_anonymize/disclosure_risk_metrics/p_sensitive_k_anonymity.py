"""
P-sensitive k-anonymity disclosure risk validation for Project Lighthouse.

This module provides tools to measure disclosure risk through privacy model validation.
P-sensitive k-anonymity validation directly quantifies re-identification and attribute
disclosure risks by ensuring that anonymized datasets meet specified privacy requirements.

The main functionality includes:
- Calculating k-anonymity and p-sensitive k-anonymity compliance metrics
- Measuring disclosure risk through privacy model validation
"""

from typing import Optional, cast

import pandas as pd


def calculate_p_k(
    input_df: pd.DataFrame,
    qids: Optional[list[str]] = None,
    sens_attr: Optional[str] = None,
) -> tuple[Optional[int], Optional[int]]:
    """
    Calculate the p and k values for p-sensitive k-anonymity disclosure risk assessment.

    This function analyzes a dataframe to determine the level of p-sensitive
    k-anonymity it provides, which directly measures disclosure risk. K-anonymity
    ensures that each combination of quasi-identifier values appears at least k times
    (measuring re-identification risk), while p-sensitive k-anonymity additionally
    ensures that each such group contains at least p distinct values for the sensitive
    attribute (measuring attribute disclosure risk).

    Parameters
    ----------
    input_df : pd.DataFrame
        The dataframe to analyze for disclosure risk.
    qids : Optional[List[str]], default=None
        List of column names to treat as quasi-identifiers. If None, all columns
        except sens_attr are treated as QIDs.
    sens_attr : Optional[str], default=None
        Name of the sensitive attribute column. If None, only k-anonymity is
        calculated and p will be returned as None.

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        A tuple containing (p, k) values that measure disclosure risk:

        - p: The minimum number of distinct sensitive values in any equivalence class.
             Lower p values indicate higher attribute disclosure risk.
             None if sens_attr is not provided.
        - k: The minimum number of records in any equivalence class. Lower k values
             indicate higher re-identification disclosure risk. Typed Optional for
             API compatibility, but always an int for the non-empty input this
             function requires.

    Raises
    ------
    ValueError
        If input_df is empty, if any provided column names don't exist in input_df,
        or if any QID or sensitive attribute column has a categorical dtype.
        Use project_lighthouse_anonymize.wrappers.dtype_conversion.
        convert_categorical_to_object() to convert categorical columns first.

    Notes
    -----

    - If there are no QIDs provided, the entire dataframe is treated as a single
      equivalence class.
    - This function directly measures disclosure risk: lower p,k values indicate
      higher disclosure risk, while higher values indicate better privacy protection.

    Examples
    --------

    >>> df = pd.DataFrame({
    ...     'zipcode': [12345, 12345, 54321, 54321],
    ...     'age': [30, 30, 40, 40],
    ...     'hobby': ['hiking', 'reading', 'hiking', 'reading']
    ... })
    >>> p, k = calculate_p_k(df, qids=['zipcode', 'age'], sens_attr='hobby')
    >>> print(f"p={p}, k={k}")
    p=2, k=2
    """
    if len(input_df) == 0:
        raise ValueError("Input dataframe has no rows")
    cols = [str(col_name) for col_name in input_df.columns]
    qids = _normalize_qids(cols, qids)
    qids, sens_attr = _validate_sens_attr(cols, qids, sens_attr)
    _reject_categorical_columns(input_df, [*qids, *([sens_attr] if sens_attr is not None else [])])

    input_df = input_df.copy(deep=True)
    cols_to_drop = set(cols) - (set(qids) | set([sens_attr]))
    input_df.drop(list(cols_to_drop), axis="columns", inplace=True)

    if len(qids) > 0:
        actual_p, actual_k = _compute_p_k_with_qids(input_df, qids, sens_attr)
    else:
        actual_p, actual_k = _compute_p_k_without_qids(input_df, sens_attr)

    return actual_p, cast(int, actual_k)


def _normalize_qids(cols: list[str], qids: Optional[list[str]]) -> list[str]:
    normalized = cols.copy() if qids is None else list(qids)
    for qid_col in normalized:
        if qid_col not in cols:
            raise ValueError(f"QID col ({qid_col}) is not a column in the input dataframe")
    return normalized


def _validate_sens_attr(
    cols: list[str], qids: list[str], sens_attr: Optional[str]
) -> tuple[list[str], Optional[str]]:
    if sens_attr is None:
        return qids, None
    if sens_attr not in cols:
        raise ValueError(
            f"Sensitive attribute col ({sens_attr}) is not a column in the input dataframe"
        )
    if sens_attr in qids:
        qids.remove(sens_attr)
    return qids, sens_attr


def _compute_p_k_with_qids(
    df: pd.DataFrame,
    qids: list[str],
    sens_attr: Optional[str],
) -> tuple[Optional[int], int]:
    grouped = df.groupby(qids, dropna=False, observed=True)
    actual_k = int(grouped.size().min())
    if sens_attr is None:
        return None, actual_k
    return int(cast(int, grouped[sens_attr].nunique().min())), actual_k


def _compute_p_k_without_qids(
    df: pd.DataFrame,
    sens_attr: Optional[str],
) -> tuple[Optional[int], int]:
    actual_k = len(df)
    if sens_attr is None:
        return None, actual_k
    return int(df[sens_attr].nunique()), actual_k


def _reject_categorical_columns(input_df: pd.DataFrame, cols: list[str]) -> None:
    """
    Raise ValueError if any of the named columns has a categorical dtype.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame to check.
    cols : List[str]
        Column names to check.

    Raises
    ------
    ValueError
        If any column has a categorical dtype.
    """
    categorical_cols = [col for col in cols if isinstance(input_df.dtypes[col], pd.CategoricalDtype)]
    if categorical_cols:
        raise ValueError(
            f"Column(s) {categorical_cols} have a categorical dtype, which is not supported; use "
            "project_lighthouse_anonymize.wrappers.dtype_conversion."
            "convert_categorical_to_object() before and "
            "convert_object_to_categorical() after"
        )
