"""
L-diversity disclosure risk metrics for Project Lighthouse anonymization evaluation.

This module contains disclosure risk metrics based on l-diversity analysis.
L-diversity measures attribute disclosure risk by evaluating whether sensitive
attribute values within each equivalence class have sufficient diversity to
prevent attribute inference attacks.

Functions:
- compute_entropy_log_l_diversity: Calculate entropy l-diversity disclosure risk across equivalence classes
"""

from typing import Iterator, Optional, cast

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA


def compute_entropy_log_l_diversity(
    sensitive_df: pd.DataFrame,
    qids: list[str],
    sens_attr_col: str,
) -> tuple[float, float, float]:
    """
    Compute entropy l-diversity disclosure risk across all equivalence classes.

    This function calculates the entropy l-diversity measure based on Definition 4.1 from
    Machanavajjhala et al. Entropy l-diversity measures attribute disclosure risk by
    computing the entropy of sensitive attribute distributions within each equivalence class.
    Lower l-diversity values indicate higher disclosure risk.

    Parameters
    ----------
    sensitive_df : pd.DataFrame
        DataFrame containing the data with sensitive attributes to analyze.
    qids : List[str]
        List of quasi-identifier column names that define the equivalence classes.
    sens_attr_col : str
        Column name of the sensitive attribute to measure diversity for.
        Currently only a single sensitive attribute is supported.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing:
        - Average l-diversity across all equivalence classes
        - Minimum l-diversity (lowest value across all equivalence classes)
        - Maximum l-diversity (highest value across all equivalence classes)

    Notes
    -----
    Unlike other metrics in the project, the l-diversity value isn't constrained
    to [0, 1]. Higher values indicate greater diversity of sensitive values within
    equivalence classes, which provides stronger privacy protection and lower
    disclosure risk.

    The value computed for an equivalence class is the entropy of its sensitive
    attribute distribution:
    -sum(p_i * log(p_i))
    where p_i is the proportion of records having the ith value of the sensitive
    attribute, summed over the values present in the class. Per Definition 4.1
    a table is entropy l-diverse when this entropy is >= log(l) for every
    equivalence class, so the returned quantity is log(l), not l.

    If no QIDs are provided, the entire dataframe is treated as a single
    equivalence class, consistent with calculate_p_k.

    Records with a NaN sensitive value are excluded from the proportions. An
    equivalence class with no non-NaN sensitive values has undefined entropy and
    is excluded from the aggregation; if every class is excluded, all three
    return values are NaN.

    References
    ----------
    A. Machanavajjhala, J. Gehrke, D. Kifer, and M. Venkitasubramaniam,
    "L-diversity: privacy beyond k-anonymity," in 22nd International Conference
    on Data Engineering (ICDE'06), Atlanta, GA, USA: IEEE, 2006, pp. 24-24.
    doi: 10.1109/ICDE.2006.1.
    """
    if len(sensitive_df) == 0:
        return (NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA)

    l_values = []
    for q_star_block_df in _iter_q_star_blocks(sensitive_df, qids):
        l_value = _q_star_block_entropy(q_star_block_df, sens_attr_col)
        if l_value is not None:
            l_values.append(l_value)

    if len(l_values) == 0:
        return (NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA)

    return float(np.mean(l_values)), float(np.min(l_values)), float(np.max(l_values))


def _iter_q_star_blocks(sensitive_df: pd.DataFrame, qids: list[str]) -> Iterator[pd.DataFrame]:
    """
    Yield the q*-block (equivalence class) dataframes defined by the QIDs.

    Parameters
    ----------
    sensitive_df : pd.DataFrame
        DataFrame containing the data with sensitive attributes to analyze.
    qids : List[str]
        List of quasi-identifier column names that define the equivalence classes.
        If empty, the entire dataframe is one equivalence class.

    Yields
    ------
    pd.DataFrame
        One dataframe per equivalence class.
    """
    if len(qids) == 0:
        yield sensitive_df
        return
    # observed=True so that unobserved categorical combinations do not create
    # phantom empty equivalence classes with worst-case 0.0 entropy
    q_star_block_dfs = sensitive_df.groupby(
        qids if len(qids) > 1 else qids[0], dropna=False, observed=True
    )
    for _, q_star_block_df in q_star_block_dfs:
        yield q_star_block_df


def _q_star_block_entropy(q_star_block_df: pd.DataFrame, sens_attr_col: str) -> Optional[float]:
    """
    Compute the sensitive attribute entropy of a single q*-block.

    Parameters
    ----------
    q_star_block_df : pd.DataFrame
        DataFrame containing the records of one equivalence class.
    sens_attr_col : str
        Column name of the sensitive attribute.

    Returns
    -------
    Optional[float]
        The entropy -sum(p_i * log(p_i)) over the sensitive values present in
        the class, or None when the class has no non-NaN sensitive values
        (undefined entropy).
    """
    # value_counts drops NaN sensitive values; unobserved categories of a
    # categorical sensitive column appear with count 0 and are excluded so
    # that 0 * log(0) does not poison the entropy with NaN
    s_counts = cast(pd.Series, q_star_block_df[sens_attr_col].value_counts())
    s_counts = cast(pd.Series, s_counts[s_counts > 0])
    total_count = s_counts.sum()
    if total_count == 0:
        return None
    p_q_star = s_counts.to_numpy(dtype="float64") / total_count
    return float(-np.sum(p_q_star * np.log(p_q_star)))
