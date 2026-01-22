"""
P-sensitive k-anonymity wrapper functions for anonymizing data.

This module provides p-sensitive k-anonymity functionality and includes
shared utilities needed for p-sensitive processing.
"""

import logging
from typing import Optional

import pandas as pd

from project_lighthouse_anonymize.disclosure_risk_metrics import (
    calculate_p_k,
    compute_entropy_log_l_diversity,
)
from project_lighthouse_anonymize.p_sensitize import p_sensitize as _impl_p_sensitize
from project_lighthouse_anonymize.pandas_utils import get_temp_col


def p_sensitize(
    logger: logging.Logger,
    input_df: pd.DataFrame,
    qids: list[str],
    sens_attr_col: str,
    target_p: int,
    target_k: int,
    sens_attr_value_to_prob: dict[str, float],
    seed: Optional[int] = None,
    parallelism: Optional[int] = 10,
    id_col: Optional[str] = None,
    rnd_metrics: bool = False,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Apply p-sensitive k-anonymity to an input dataframe.

    This function transforms an already k-anonymous dataframe to satisfy
    p-sensitive k-anonymity by ensuring each equivalence class contains at
    least p distinct sensitive attribute values. It employs a perturbation
    approach that modifies sensitive values while preserving the k-anonymity
    property of the dataset.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording the sensitization process
    input_df : pd.DataFrame
        Input dataframe to p-sensitize (should already be k-anonymous)
    qids : List[str]
        List of column names representing quasi-identifiers
    sens_attr_col : str
        Sensitive attribute column name (currently only a single sensitive
        attribute is supported)
    target_p : int
        Target p value (minimum number of distinct sensitive values in any
        equivalence class)
    target_k : int
        Target k value (minimum number of records in any equivalence class)
    sens_attr_value_to_prob : Dict[str, float]
        Probability distribution for sampling sensitive attribute values
        during perturbation
    seed : Optional[int], default=None
        Seed for random number generator; None makes the process non-deterministic
    parallelism : Optional[int], default=10
        Number of concurrent processes to use for computing metrics
    id_col : Optional[str], default=None
        Optional unique identifier column for computing metrics
    rnd_metrics : bool, default=False
        If True, computes additional research-oriented metrics

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]
        A tuple containing:
        1. A p-sensitive k-anonymous dataframe
        2. A dictionary of data quality metrics including perturbation statistics
        3. A dictionary of disclosure risk metrics including actual p and k values

    Raises
    ------
    RuntimeError
        If the resulting p-sensitive k-anonymous dataframe does not meet the
        target_p or target_k requirements (actual_p < target_p or actual_k < target_k)

    Notes
    -----
    The function works by modifying sensitive attribute values in equivalence
    classes that don't meet the p requirement, using a controlled perturbation
    approach that maintains statistical properties based on the provided
    probability distribution.
    """
    # Anonymize
    sensitized_df, num_rows_perturbated = _impl_p_sensitize(
        logger,
        input_df,
        qids,
        sens_attr_col,
        target_p,
        target_k,
        sens_attr_value_to_prob,
        seed=seed,
    )
    actual_p: Optional[int]
    actual_k: Optional[int]
    if len(sensitized_df) > 0:
        actual_p, actual_k = calculate_p_k(sensitized_df, qids, sens_attr=sens_attr_col)
        if actual_p is None or actual_p < target_p:
            raise RuntimeError(f"Result p ({actual_p}) not >= target p ({target_p})")
        if actual_k is None or actual_k < target_k:
            raise RuntimeError(f"Result k ({actual_k}) not >= target k ({target_k})")
    else:
        actual_p, actual_k = None, None
    # Compute metrics
    cols_to_remove = []
    if id_col is None:
        id_col = get_temp_col(sensitized_df, col_prefix="placeholder_id_")
        sensitized_df[id_col] = list(range(len(sensitized_df)))
        cols_to_remove.append(id_col)
    try:
        dq_metrics: dict[str, float] = {}
        dq_metrics["num_rows_perturbated"] = float(num_rows_perturbated)
        dq_metrics["pct_perturbated"] = float(100.0 * (num_rows_perturbated / len(input_df)))
        disclosure_metrics: dict[str, float] = {}
        disclosure_metrics["actual_p"] = float(actual_p) if actual_p is not None else float("nan")
        disclosure_metrics["actual_k"] = float(actual_k) if actual_k is not None else float("nan")
        if rnd_metrics:
            l_avg, l_min, l_max = compute_entropy_log_l_diversity(
                sensitized_df,
                qids,
                sens_attr_col,
            )
            disclosure_metrics["entropy_l_diversity_avg"] = l_avg
            disclosure_metrics["entropy_l_diversity_min"] = l_min
            disclosure_metrics["entropy_l_diversity_max"] = l_max
        return sensitized_df, dq_metrics, disclosure_metrics
    finally:
        if len(cols_to_remove) > 0:
            sensitized_df.drop(columns=cols_to_remove, inplace=True)
