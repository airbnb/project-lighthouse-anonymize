"""
Pearson's correlation coefficient metrics for Project Lighthouse data quality evaluation.

This module implements Pearson's correlation coefficient computation for measuring
the preservation of linear relationships between original and anonymized numerical
quasi-identifiers.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA


def compute_pearsons_correlation_coefficients(
    non_suppressed_records: pd.DataFrame,
    qids: list[str],
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> dict[str, float]:
    """
    Compute Pearson's correlation coefficient for numerical quasi-identifiers.

    This function calculates the Pearson's correlation coefficient between original and
    anonymized values for each numerical QID, floored to be in [0, 1]. This metric
    measures the preservation of linear relationships under anonymization.

    As described in the Project Lighthouse paper, a value of 1.0 indicates a perfect
    positive linear relationship between original and anonymized data, while 0.0 indicates
    either no linear relationship or a negative relationship. Negative correlations are
    floored to 0 because they represent a complete loss of the original relationship.

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        Output of get_non_suppressed_records containing non-suppressed records with
        both original and anonymized values.
    qids : List[str]
        List of quasi-identifier column names to analyze. Only numerical QIDs will be
        processed; categorical QIDs are ignored.
    join_suffix_orig : str
        Suffix that was added to original column names during the join operation.
    join_suffix_anon : str
        Suffix that was added to anonymized column names during the join operation.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping QID names to their Pearson's correlation coefficients (max(0, \\rho)).

    Notes
    -----
    This implementation handles multiple edge cases not explicitly addressed in the literature:

    - When all values are identical and unchanged: Returns 1.0 (perfect preservation)
    - When all values are identical but changed: Returns 0.0 (complete information loss)
    - When original values vary but all anonymized values are identical: Returns 0.0
    - When there are insufficient unique values: Returns NOT_DEFINED_NA

    A preprocessing step replaces locally suppressed values with the mean of
    non-NaN values, where a value is "locally suppressed" if its original value
    is non-NaN but its anonymized value is NaN.

    References
    ----------
    Originally used in:
    Kim, Y., Ngai, E. C.-H. & Srivastava, M. B.
    Cooperative state estimation for preserving privacy of user behaviors in smart grid.
    2011 Ieee Int Conf Smart Grid Commun Smartgridcomm 178-183 (2011)
    doi:10.1109/smartgridcomm.2011.6102313.
    """
    qids = [
        qid
        for qid in qids
        if non_suppressed_records.dtypes[f"{qid}{join_suffix_orig}"] != np.dtype("object")
    ]
    if len(non_suppressed_records) == 0 or len(qids) == 0:
        return {qid: NOT_DEFINED_NA for qid in qids}
    orig_qids = [f"{qid}{join_suffix_orig}" for qid in qids]
    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]

    pearsons = {}
    for qid, orig_qid, anon_qid in zip(qids, orig_qids, anon_qids):
        x, y = non_suppressed_records[orig_qid], non_suppressed_records[anon_qid]
        # Drop entries where both x is nan and y is nan--this should only
        # happen when the original attribute value is nan, and thus so is the
        # associated anonymized attribute value.
        x_and_y_nan = (np.isnan(x)) & (np.isnan(y))
        x, y = x[~x_and_y_nan], y[~x_and_y_nan]
        # Replace locally suppressed anonymized attribute values
        if len(y) > 0 and any(np.isnan(y)) and not all(np.isnan(y)):
            # We know no more than what is present in anonymized records
            # so presume the missing records are just the average of those present
            y[np.isnan(y)] = np.nanmean(y)

        if len(set(x)) == 1 and len(set(y)) == 1 and set(x) == set(y):
            # Edge case: a single value that is unchanged yields a Pearson's of 1.0
            # This represents perfect information preservation when the original data
            # had no variance to begin with
            corr = 1.0
        elif len(set(x)) == 1 and len(set(y)) == 1 and set(x) != set(y):
            # Edge case: a single value that is changed yields a Pearson's of 0.0
            # This represents complete information loss - the anonymized value
            # bears no relation to the original
            corr = 0.0
        elif len(set(x)) > 1 and len(set(y)) == 1:
            # Edge case: multiple values are anonymized to a single value
            # This represents complete information loss through generalization
            # since all original variance has been removed
            corr = 0.0
        elif len(set(x)) <= 1 or len(set(y)) <= 1:
            # Final edge case: insufficient unique values to compute Pearson's
            # Variance is zero for at least one column, so correlation is undefined
            corr = NOT_DEFINED_NA
        else:
            corr, _ = pearsonr(x, y)
            corr = max(0.0, corr)  # values < 0 are considered 0.0
        pearsons[qid] = corr
    return pearsons
