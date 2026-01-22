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
from first import first  # type: ignore[import-untyped]

from project_lighthouse_anonymize.pandas_utils import make_temp_id_col


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
             indicate higher re-identification disclosure risk.
             Can be None in edge cases handled by callers.

    Raises
    ------
    ValueError
        If input_df is empty or if any provided column names don't exist in input_df.

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
    qids = (
        cols.copy() if qids is None else list(qids)
    )  # need a list object for pandas.DataFrame.groupby below; it might be a tuple
    for qid_col in qids:
        if qid_col not in cols:
            raise ValueError(f"QID col ({qid_col}) is not a column in the input dataframe")
    if sens_attr is not None:
        if sens_attr not in cols:
            raise ValueError(
                f"Sensitive attribute col ({sens_attr}) is not a column in the input dataframe"
            )
        if sens_attr in qids:
            qids.remove(sens_attr)

    input_df = input_df.copy(deep=True)
    cols_to_drop = set(cols) - (set(qids) | set([sens_attr]))
    input_df.drop(list(cols_to_drop), axis="columns", inplace=True)
    make_temp_id_col(input_df)

    if len(qids) > 0:
        actual_k = first(input_df.groupby(qids, dropna=False).count().min())
        if sens_attr is not None:
            actual_p = int(
                cast(int, input_df.groupby(qids, dropna=False)[sens_attr].nunique().min())
            )
        else:
            actual_p = None
    else:
        actual_k = len(input_df)
        if sens_attr is not None:
            actual_p = int(input_df[sens_attr].nunique())
        else:
            actual_p = None

    return actual_p, cast(int, actual_k)
