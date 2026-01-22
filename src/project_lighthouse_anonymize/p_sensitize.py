"""
Functions for making a dataset p-sensitive k-anonymous.

This module provides tools to transform a k-anonymous dataset into one that
also satisfies p-sensitive k-anonymity. P-sensitive k-anonymity extends
traditional k-anonymity by ensuring that within each equivalence class,
there are at least p different values for the sensitive attribute.

The implementation assumes the input dataset is already k-anonymous and
focuses on adding the p-sensitivity property through controlled data
modification techniques that preserve privacy while minimizing information loss.
"""

from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.disclosure_risk_metrics import calculate_p_k


def p_sensitize(
    logger: Any,
    input_df: pd.DataFrame,
    qids: list[str],
    sens_attr_col: str,
    target_p: int,
    target_k: int,
    sens_attr_value_to_prob: dict[str, float],
    seed: Optional[int] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Transform a k-anonymous dataset into a p-sensitive k-anonymous dataset.

    This function modifies a k-anonymous dataset to ensure each equivalence class
    contains at least p distinct values for the sensitive attribute, creating
    a p-sensitive k-anonymous dataset. It achieves this through controlled
    perturbation of sensitive attribute values.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording operation information and warnings.
    input_df : pd.DataFrame
        Input dataframe to be p-sensitized. Must already be k-anonymous.
    qids : List[str]
        List of quasi-identifier column names. May be empty if there are no
        quasi-identifiers.
    sens_attr_col : str
        Column name of the sensitive attribute. Currently only a single
        sensitive attribute is supported.
    target_p : int
        The minimum number of distinct sensitive attribute values required
        in each equivalence class.
    target_k : int
        The minimum number of records required in each equivalence class.
        Must be >= target_p.
    sens_attr_value_to_prob : Dict[str, float]
        Dictionary mapping allowable sensitive attribute values to their
        selection probabilities during perturbation.
    seed : Optional[int], default=None
        Random seed for deterministic perturbation.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        A tuple containing:
        1. A p-sensitive k-anonymous version of the input dataframe
        2. The number of records that were perturbated

    Raises
    ------
    ValueError
        If input requirements are not met (e.g., empty dataframe, invalid column
        names, target_k < target_p, etc.)

    Notes
    -----
    This is an internal implementation function. For external use, please use
    project_lighthouse_anonymize.wrappers.p_sensitize instead.

    The function first checks if the dataset is already p-sensitive k-anonymous.
    If not, it processes each equivalence class to ensure it has at least p
    distinct sensitive values by replacing some values with alternatives from
    the sens_attr_value_to_prob dictionary.
    """
    if len(input_df) == 0:
        raise ValueError("Input dataframe has no rows")
    cols = [str(col_name) for col_name in input_df.columns]
    qids = list(qids)  # need a list object for pandas.DataFrame.groupby below; it might be a tuple
    for qid_col in qids:
        if qid_col not in cols:
            raise ValueError(f"QID col ({qid_col}) is not a column in the input dataframe")
    if sens_attr_col not in cols:
        raise ValueError(
            f"Sensitive attribute col ({sens_attr_col}) is not a column in the input dataframe"
        )
    if any(input_df[sens_attr_col].isna()):
        raise ValueError(
            "Sensitive attribute value na is not allowed due to a bug in pandas.groupby.count"
        )
    if target_p < 1:
        raise ValueError("target_p must be >= 1")
    if target_k < target_p:
        raise ValueError(f"Target k ({target_k}) not >= target p ({target_p}); p must be <= k")
    if target_p > len(sens_attr_value_to_prob.keys()):
        raise ValueError(
            f"Target p ({target_p}) cannot be > number of sensitive attributes for which we have priors"
        )
    actual_p, actual_k = calculate_p_k(input_df, qids, sens_attr=sens_attr_col)
    if actual_k is not None and actual_k < target_k:
        raise ValueError(
            f"Actual k ({actual_k}) not >= target k ({target_k}); P-Sensitize does not support increasing the actual k, due to a potential privacy attack vector"
        )

    output_df = input_df.copy()

    # break early if the dataset is already p-sensitive k-anonymouse
    if actual_p is not None and actual_p >= target_p:
        return (output_df, 0)

    _impl_p_sensitize_equivalence_class_func = (
        _impl_p_sensitize_equivalence_class_peq2
        if target_p == 2
        else _impl_p_sensitize_equivalence_class
    )

    rng = np.random.default_rng(seed)
    all_indices_perturbated: list[Any] = []
    all_indices_new_sens_attr_values: list[str] = []
    if len(qids) > 0:
        input_df.groupby(qids, dropna=False, group_keys=False).apply(
            lambda df: _impl_p_sensitize_equivalence_class_func(
                df,
                sens_attr_col,
                target_p,
                sens_attr_value_to_prob,
                rng,
                all_indices_perturbated,
                all_indices_new_sens_attr_values,
            )
        )
    else:
        _impl_p_sensitize_equivalence_class_func(
            input_df,
            sens_attr_col,
            target_p,
            sens_attr_value_to_prob,
            rng,
            all_indices_perturbated,
            all_indices_new_sens_attr_values,
        )
    output_df.loc[all_indices_perturbated, sens_attr_col] = all_indices_new_sens_attr_values
    return (output_df, len(all_indices_perturbated))


def _impl_p_sensitize_equivalence_class(
    equivalence_class_df: pd.DataFrame,
    sens_attr_col: str,
    target_p: int,
    sens_attr_value_to_prob: dict[str, float],
    rng: np.random.Generator,
    all_indices_perturbated: list[Any],
    all_indices_new_sens_attr_values: list[str],
) -> pd.DataFrame:
    """
    Process a single equivalence class to ensure p-sensitivity.

    This internal implementation function handles the modification of sensitive
    attribute values within a single equivalence class to ensure it contains at least
    p distinct sensitive values. It accumulates indices and new values for perturbation
    in the provided lists.

    Parameters
    ----------
    equivalence_class_df : pd.DataFrame
        DataFrame containing records from a single equivalence class.
    sens_attr_col : str
        Column name of the sensitive attribute.
    target_p : int
        The minimum number of distinct sensitive attribute values required (p).
    sens_attr_value_to_prob : Dict[str, float]
        Dictionary mapping allowable sensitive attribute values to their probabilities.
    rng : np.random._generator.Generator
        Random number generator for deterministic sampling.
    all_indices_perturbated : List[Any]
        List to collect indices of records selected for perturbation.
    all_indices_new_sens_attr_values : List[str]
        List to collect new sensitive attribute values for perturbated records.

    Returns
    -------
    pd.DataFrame
        The input equivalence class DataFrame (unchanged).

    Notes
    -----
    The function:
    1. Checks if the equivalence class already has p or more distinct sensitive values
    2. If not, selects records to modify ensuring no sensitive value is completely removed
    3. Assigns new sensitive values from the allowed set with probabilities proportional
       to their given distribution
    4. Accumulates the changes in the provided lists for later application
    """
    actual_p = equivalence_class_df[sens_attr_col].nunique()
    if actual_p >= target_p:
        return equivalence_class_df
    sens_values_to_perturbate = equivalence_class_df[sens_attr_col].value_counts()
    sens_attr_value_to_prob = {
        sens_attr_value: prob
        for sens_attr_value, prob in sens_attr_value_to_prob.items()
        if sens_attr_value not in sens_values_to_perturbate.index
    }
    sens_values_to_perturbate = cast(
        pd.Series, sens_values_to_perturbate[sens_values_to_perturbate > 1]
    )
    mask = cast(pd.Series, equivalence_class_df[sens_attr_col]).isin(
        list(sens_values_to_perturbate.index)
    )
    indices_to_consider = cast(np.ndarray, equivalence_class_df.index[mask])
    indices_to_perturbate = rng.choice(indices_to_consider, target_p - actual_p, replace=False)
    all_indices_perturbated.extend(indices_to_perturbate)
    potential_new_sens_attr_values = list(sens_attr_value_to_prob.keys())
    potential_new_sens_attr_probs = np.array(
        [
            sens_attr_value_to_prob[sens_attr_value]
            for sens_attr_value in potential_new_sens_attr_values
        ]
    )
    potential_new_sens_attr_probs /= potential_new_sens_attr_probs.sum()
    new_sens_attr_values = rng.choice(
        potential_new_sens_attr_values,
        size=target_p - actual_p,
        replace=False,
        p=potential_new_sens_attr_probs,
    )
    all_indices_new_sens_attr_values.extend(new_sens_attr_values)
    return equivalence_class_df


def _impl_p_sensitize_equivalence_class_peq2(
    equivalence_class_df: pd.DataFrame,
    sens_attr_col: str,
    target_p: int,
    sens_attr_value_to_prob: dict[str, float],
    rng: np.random.Generator,
    all_indices_perturbated: list[Any],
    all_indices_new_sens_attr_values: list[str],
) -> pd.DataFrame:
    """
    Process a single equivalence class to ensure 2-sensitivity (optimized for p=2).

    This is a specialized implementation function for the common case where p=2,
    optimizing performance by using a simpler approach than the general case.
    It handles the modification of a single sensitive attribute value within an
    equivalence class to ensure it contains at least 2 distinct sensitive values.

    Parameters
    ----------
    equivalence_class_df : pd.DataFrame
        DataFrame containing records from a single equivalence class.
    sens_attr_col : str
        Column name of the sensitive attribute.
    target_p : int
        Should be 2 for this function, but kept for API consistency.
    sens_attr_value_to_prob : Dict[str, float]
        Dictionary mapping allowable sensitive attribute values to their probabilities.
    rng : np.random._generator.Generator
        Random number generator for deterministic sampling.
    all_indices_perturbated : List[Any]
        List to collect indices of records selected for perturbation.
    all_indices_new_sens_attr_values : List[str]
        List to collect new sensitive attribute values for perturbated records.

    Returns
    -------
    pd.DataFrame
        The input equivalence class DataFrame (unchanged).

    Notes
    -----
    This implementation is specialized for p=2, which is a common case:
    1. For p=2, we only need to modify a single record when the class is homogeneous
    2. The function randomly selects one record to modify
    3. It chooses a new value from the allowed set (excluding the current value)
       with probabilities proportional to the given distribution
    4. It accumulates the changes in the provided lists for later application

    This specialized implementation is more efficient than the general case because:
    - It handles exactly one record modification
    - It doesn't need to preserve any of the original sensitive values
    - It's guaranteed that all records have the same value initially
    """
    actual_p = equivalence_class_df[sens_attr_col].nunique()
    if actual_p >= 2:
        return equivalence_class_df
    index_to_perturbate = rng.choice(cast(np.ndarray, equivalence_class_df.index))
    all_indices_perturbated.append(index_to_perturbate)
    homogenous_sens_attr_value = equivalence_class_df[sens_attr_col].iat[0]
    potential_new_sens_attr_values = [
        sens_attr_value
        for sens_attr_value in sens_attr_value_to_prob.keys()
        if sens_attr_value != homogenous_sens_attr_value
    ]
    potential_new_sens_attr_probs = np.array(
        [
            sens_attr_value_to_prob[sens_attr_value]
            for sens_attr_value in potential_new_sens_attr_values
        ]
    )
    potential_new_sens_attr_probs /= potential_new_sens_attr_probs.sum()
    new_sens_attr_value = rng.choice(
        potential_new_sens_attr_values, p=potential_new_sens_attr_probs
    )
    all_indices_new_sens_attr_values.append(new_sens_attr_value)
    return equivalence_class_df
