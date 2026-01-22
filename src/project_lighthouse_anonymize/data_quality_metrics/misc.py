"""
Miscellaneous data quality metrics for Project Lighthouse anonymization evaluation.

This module contains additional data quality metrics that support the analysis
of anonymized data but don't fit into the other specialized metric modules.

Functions:
- compute_suppression_metrics: Record suppression counts and percentages
- compute_average_equivalence_class_metric: Average equivalence class size
- compute_discernibility_metric: Discernibility penalty metric
"""

from typing import cast

import numpy as np
import pandas as pd
from first import first  # type: ignore[import-untyped]

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA


def compute_suppression_metrics(input_df: pd.DataFrame, anon_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute suppression metrics for records after anonymization.

    This function calculates counts and percentages of records that were preserved
    and suppressed during the anonymization process. These metrics serve as
    secondary data quality indicators that measure record preservation.

    Parameters
    ----------
    input_df : pd.DataFrame
        Original input dataframe before anonymization.
    anon_df : pd.DataFrame
        Anonymized output dataframe.

    Returns
    -------
    Dict[str, float]
        Dictionary containing four metrics:
        - "n_non_suppressed": Count of records preserved in the anonymized data
        - "pct_non_suppressed": Percentage of original records preserved
        - "n_suppressed": Count of records suppressed during anonymization
        - "pct_suppressed": Percentage of original records suppressed

    Notes
    -----
    The percentage metrics will sum to 1.0 when the input dataframe is not empty.
    If the input dataframe is empty, percentage metrics will be np.nan.

    See Also
    --------
    get_suppressed_records : Function to identify which specific records were suppressed
    """
    n_total = len(input_df)
    n_non_suppressed = len(anon_df)
    n_suppressed = n_total - n_non_suppressed

    pct_non_suppressed = (n_non_suppressed / n_total) if n_total > 0 else np.nan
    pct_suppressed = (n_suppressed / n_total) if n_total > 0 else np.nan

    return {
        "n_non_suppressed": n_non_suppressed,
        "pct_non_suppressed": pct_non_suppressed,
        "n_suppressed": n_suppressed,
        "pct_suppressed": pct_suppressed,
    }


def compute_average_equivalence_class_metric(
    non_suppressed_records: pd.DataFrame, qids: list[str], join_suffix_anon: str
) -> float:
    """
    Compute the average equivalence class metric for anonymized data.

    This metric measures the average size of equivalence classes in the anonymized data,
    which is defined as the ratio of the total number of records to the number of
    equivalence classes. Higher values indicate larger equivalence classes and greater
    information loss.

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        Output of get_non_suppressed_records containing non-suppressed records with
        both original and anonymized values.
    qids : List[str]
        List of quasi-identifier column names that define the equivalence classes.
    join_suffix_anon : str
        Suffix that was added to anonymized column names during the join operation.

    Returns
    -------
    float
        The average equivalence class size. Note that unlike most metrics in this
        module, this metric doesn't follow the convention of being in [0, 1] with
        higher values meaning higher data quality. Also note it isn't normalized by k.

    Notes
    -----
    The average equivalence class metric is calculated as:
    :math:`|D*|`/:math:`|E*|` where :math:`|D*|` is the number of non-suppressed records and :math:`|E*|` is the
    number of equivalence classes in the anonymized dataset.

    A higher value indicates larger equivalence classes and thus more information loss.
    In an ideally anonymized dataset with minimal information loss, each equivalence
    class would have exactly k records.

    References
    ----------
    LeFevre, K., Dewitt, D. J. & Ramakrishnan, R.
    Mondrian Multidimensional K-Anonymity. 22nd Int Conf Data Eng Icde'06 1-11 (2006)
    doi:10.1109/icde.2006.101.
    """
    if len(non_suppressed_records) == 0 or len(qids) == 0:
        return NOT_DEFINED_NA
    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]
    qid_records_df = non_suppressed_records[anon_qids]
    return float(len(qid_records_df) / qid_records_df.groupby(anon_qids, dropna=False).ngroups)


def compute_discernibility_metric(
    non_suppressed_records: pd.DataFrame,
    suppressed_records: pd.DataFrame,
    qids: list[str],
    id_col: str,
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> float:
    """
    Compute the discernibility metric that penalizes generalization and suppression.

    This metric assigns a penalty to each record in the dataset based on how many other
    records are indistinguishable from it after anonymization. It captures both the cost
    of generalization (by squaring the size of each equivalence class) and the cost of
    suppression (by giving suppressed records a maximum penalty).

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        Output of get_non_suppressed_records containing non-suppressed records with
        both original and anonymized values.
    suppressed_records : pd.DataFrame
        Output of get_suppressed_records containing records that were suppressed
        during the anonymization process.
    qids : List[str]
        List of quasi-identifier column names that define the equivalence classes.
    id_col : str
        Column name of the unique identifier for records.
    join_suffix_orig : str
        Suffix that was added to original column names during the join operation.
    join_suffix_anon : str
        Suffix that was added to anonymized column names during the join operation.

    Returns
    -------
    float
        The discernibility metric value. Note that unlike most metrics in this module,
        this metric doesn't follow the convention of being in [0, 1] with higher values
        meaning higher data quality. Lower values indicate better data quality.

    Notes
    -----
    The discernibility metric (DM) penalizes each record r as follows. If r is in an
    equivalence class E of size ≥ k, the penalty is :math:`|E|` (the number of records in
    the equivalence class). If r is suppressed, the penalty is :math:`|D|` (the total number
    of records in the dataset).

    The total DM is the sum of penalties for all records. The smaller the DM value,
    the better the quality of the anonymized data.

    References
    ----------
    Bayardo, R. J. & Agrawal, R. Data Privacy through Optimal k-Anonymization.
    21st Int Conf Data Eng Icde'05 217-228 (2005) doi:10.1109/icde.2005.42.
    """
    orig_qids = [f"{qid}{join_suffix_orig}" for qid in qids]
    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]
    total_records = len(non_suppressed_records) + len(suppressed_records)
    discernibility_metric = 0.0

    if total_records == 0 or len(qids) == 0:
        return NOT_DEFINED_NA

    # We don't need k for below because it's implicit in the split between
    # non_suppressed_records and suppressed_records

    # If the size of the equivalence class E is no less than k,
    # then each tuple in E gets a penalty of |E|(the number of tuples in E)
    discernibility_metric += np.sum(
        np.power(
            cast(
                pd.DataFrame,
                non_suppressed_records[[id_col] + anon_qids]
                .groupby(anon_qids, dropna=False)
                .agg(["count"]),
            ),
            2,
        )
    )  # we need one additional column for agg
    # Otherwise each tuple is assigned a penalty of
    # |D|(the total number of tuples in the dataset)
    discernibility_metric += np.sum(
        cast(
            pd.DataFrame,
            suppressed_records[[id_col] + orig_qids]
            .groupby(orig_qids, dropna=False)
            .agg(["count"]),
        )
        * total_records
    )  # we need one additional column for agg

    # convert from pd.Series to single value
    discernibility_metric = first(discernibility_metric, 0.0)

    return float(discernibility_metric)
