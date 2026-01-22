"""
L-diversity disclosure risk metrics for Project Lighthouse anonymization evaluation.

This module contains disclosure risk metrics based on l-diversity analysis.
L-diversity measures attribute disclosure risk by evaluating whether sensitive
attribute values within each equivalence class have sufficient diversity to
prevent attribute inference attacks.

Functions:
- compute_entropy_log_l_diversity: Calculate entropy l-diversity disclosure risk across equivalence classes
"""

from typing import cast

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

    The l-diversity value for an equivalence class is computed as:
    l = -sum(p_i * log(p_i))
    where p_i is the proportion of records having the ith value of the sensitive attribute.

    References
    ----------
    A. Machanavajjhala, J. Gehrke, D. Kifer, and M. Venkitasubramaniam,
    "L-diversity: privacy beyond k-anonymity," in 22nd International Conference
    on Data Engineering (ICDE'06), Atlanta, GA, USA: IEEE, 2006, pp. 24-24.
    doi: 10.1109/ICDE.2006.1.
    """
    if len(sensitive_df) == 0 or len(qids) == 0:
        return (NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA)

    q_star_block_dfs = sensitive_df.groupby(qids if len(qids) > 1 else qids[0], dropna=False)
    l_values = []
    for _, q_star_block_df in q_star_block_dfs:
        q_star_block_df = q_star_block_df[[sens_attr_col]].copy()
        q_star_block_df["count"] = 1
        s_to_count = cast(
            pd.Series, q_star_block_df.groupby(sens_attr_col).sum()["count"]
        ).to_dict()
        l_value = 0.0
        total_count = sum(s_to_count.values())
        for s, count in s_to_count.items():  # type: ignore[reportUnusedVariable]  # s is used below to construct variable name
            p_q_star_s = count / total_count
            l_value += p_q_star_s * np.log(p_q_star_s)
        l_value *= -1.0
        l_values.append(l_value)

    if len(l_values) == 0:
        return (NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA)

    return float(np.mean(l_values)), float(np.min(l_values)), float(np.max(l_values))
