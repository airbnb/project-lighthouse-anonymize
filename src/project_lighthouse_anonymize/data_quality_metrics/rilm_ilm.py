"""
Revised Information Loss Metric (RILM) and Information Loss Metric (ILM) implementations.

This module contains the RILM and ILM metrics extracted from the main data_quality_metrics module
for better organization and modularity. RILM is a revision and extension of ILM adapted specifically
for Project Lighthouse.

RILM measures the preservation of "geometric size" under anonymization:
- For numerical QIDs: Based on how much of the original data's range (max-min) is preserved
- For categorical QIDs: Based on how far up the generalization hierarchy values are moved

A value of 1.0 indicates perfect information preservation (no loss), while 0.0 indicates
maximum information loss.

Main functions:
- compute_revised_information_loss_metric(): Main RILM entry point
- compute_average_information_loss_metric(): ILM implementation

Helper functions:
- _compute_il(): Helper for ILM calculation
- _compute_revised_information_loss_metric_numericals(): RILM for numerical QIDs
- compute_revised_information_loss_metric_categoricals(): RILM for categorical QIDs
- compute_revised_information_loss_metric_value_for_numerical(): Single RILM value for numerical
- compute_revised_information_loss_metric_value_for_categorical(): Single RILM value for categorical
"""

from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA
from project_lighthouse_anonymize.gtrees import GTree
from project_lighthouse_anonymize.utils import max_minus_min


def _compute_il(
    equivalence_class_df: pd.DataFrame,
    orig_qids: Optional[list[str]] = None,
    orig_domain_sizes: Optional[dict[str, float]] = None,
) -> float:
    """
    Calculate information loss (IL) for a single equivalence class.

    Parameters
    ----------
    equivalence_class_df : pd.DataFrame
        DataFrame containing all records in a single equivalence class.
    orig_qids : List[str], optional
        List of original QID column names to analyze.
    orig_domain_sizes : Dict[str, float], optional
        Dictionary mapping QID column names to their domain sizes (range) in the
        original dataset.

    Returns
    -------
    float
        The information loss metric value for this equivalence class, calculated as
        the sum of attribute-level information loss across all QIDs, multiplied by
        the size of the equivalence class.

    Notes
    -----
    This is a helper function for compute_average_information_loss_metric that computes
    the IL for a single equivalence class based on Byun et al. 2006. It calculates
    the ratio of each attribute's range in the equivalence class to its range in
    the entire dataset, and handles special cases like zero-width domains and NaN values.
    """
    ilm = 0.0
    assert orig_qids is not None, "orig_qids cannot be None"
    assert orig_domain_sizes is not None, "orig_domain_sizes cannot be None"
    for orig_qid in orig_qids:
        orig_domain_size = orig_domain_sizes[orig_qid]
        if not np.isnan(orig_domain_size) and orig_domain_size != 0:
            ec_domain_size = max_minus_min(equivalence_class_df[orig_qid].to_numpy())
            if not np.isnan(ec_domain_size):
                ilm += (
                    # Because we are microaggregating numerical attributes, we take the
                    # span of the original attribute values as the width of the equivalence
                    # class rather than using the span of the associated node in the
                    # interval-based generalization hierarchy
                    ec_domain_size / orig_domain_size
                )
            else:
                # edge case where qid values are nan, skipna
                pass
        else:
            # [Byun et al 2006] doesn't sketch out this edge case in detail.
            #
            # In this edge case, the domain for this QID has 0 width, thus the
            # region for this QID in the equivalence class also has 0 width. In
            # other words, there is only a single value for this QID pre-generalization.
            # Because of this, we'd expect that generalization shouldn't have an impact.
            #
            # We choose to provide an ILM of 0.0 for this edge case to reflect that
            # there should be no information loss.
            pass
    ilm *= len(equivalence_class_df)
    return ilm


def compute_average_information_loss_metric(
    non_suppressed_records: pd.DataFrame,
    qids: list[str],
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> float:
    """
    Compute the average Information Loss Metric (ILM) for numerical attributes.

    This metric measures the information loss due to generalization by comparing the
    ranges of attribute values in each equivalence class to the ranges of those
    attributes in the original data.

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
    float
        The average ILM value, referred to as "the average cost of IL metric" in the
        original paper. Note that this metric doesn't follow the convention of being
        in [0, 1] with higher values meaning higher data quality. Lower values indicate
        better data quality.

    Notes
    -----
    For each numerical QID and each equivalence class, ILM computes the ratio of the
    attribute's range in the equivalence class to its range in the entire dataset.
    These ratios are summed across all QIDs for each equivalence class, multiplied
    by the size of the equivalence class, and then averaged across all records.

    Edge cases:

    - If an attribute has zero width in the original data, no information loss is
      assigned for that attribute.
    - If a QID value is NaN in an equivalence class, that attribute is skipped.

    References
    ----------
    Byun, J.-W., Sohn, Y., Bertino, E. & Li, N.
    Secure Anonymization for Incremental Datasets. in Workshop on secure data management (2006).
    """
    qids = [
        qid
        for qid in qids
        if non_suppressed_records.dtypes[f"{qid}{join_suffix_orig}"] != np.dtype("object")
    ]
    if len(non_suppressed_records) == 0 or len(qids) == 0:
        return NOT_DEFINED_NA
    orig_qids = [f"{qid}{join_suffix_orig}" for qid in qids]
    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]
    orig_records = non_suppressed_records[orig_qids]
    # Because we are microaggregating numerical attributes, we take the
    # span of the original attribute values as the width of the domain
    # rather than using the span of the interval-based generalization
    # hierarchy
    orig_domain_sizes = {
        orig_qid: max_minus_min(cast(pd.Series, orig_records[orig_qid]).to_numpy())
        for orig_qid in orig_qids
    }
    equivalence_classes_df = non_suppressed_records.groupby(anon_qids, dropna=False)
    ilm_df = pd.DataFrame(
        {
            "ilm": equivalence_classes_df.apply(
                _compute_il, orig_qids=orig_qids, orig_domain_sizes=orig_domain_sizes
            )
        }
    )
    return float(ilm_df["ilm"].sum() / len(non_suppressed_records))


def compute_revised_information_loss_metric(
    non_suppressed_records: pd.DataFrame,
    qids: list[str],
    qid_to_gtree: dict[str, GTree],
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute the Revised Information Loss Metric (RILM).

    RILM is a revision and extension of the Information Loss Metric (ILM) from Byun et al. 2006,
    adapted specifically for Project Lighthouse. This metric measures the preservation of
    "geometric size" under anonymization:

    - For numerical QIDs: Based on how much of the original data's range (max-min) is preserved
    - For categorical QIDs: Based on how far up the generalization hierarchy values are moved

    A value of 1.0 indicates perfect information preservation (no loss), while 0.0 indicates
    maximum information loss. RILM is modified from ILM to be a column-level metric with values
    constrained to [0,1], making it more interpretable and comparable across columns.

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        Output of get_non_suppressed_records containing non-suppressed records with
        both original and anonymized values.
    qids : List[str]
        List of quasi-identifier column names to analyze.
    qid_to_gtree : Dict[str, GTree]
        Mapping from QID names to their associated generalization trees (GTrees)
        for categorical QIDs.
    join_suffix_orig : str
        Suffix that was added to original column names during the join operation.
    join_suffix_anon : str
        Suffix that was added to anonymized column names during the join operation.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        A tuple of two dictionaries mapping QIDs to their RILM scores:
        1. Dictionary for numerical QIDs
        2. Dictionary for categorical QIDs

    See Also
    --------
    compute_revised_information_loss_metric_categoricals : Compute RILM for categorical QIDs
    compute_revised_information_loss_metric_value_for_numerical : Compute RILM for numerical values
    """
    numerical_qids = [
        qid
        for qid in qids
        if non_suppressed_records.dtypes[f"{qid}{join_suffix_orig}"] != np.dtype("object")
    ]
    categorical_qids = [
        qid
        for qid in qids
        if non_suppressed_records.dtypes[f"{qid}{join_suffix_orig}"] == np.dtype("object")
    ]

    rilm_numericals = _compute_revised_information_loss_metric_numericals(
        non_suppressed_records,
        qids,
        numerical_qids,
        join_suffix_orig,
        join_suffix_anon,
    )
    rilm_categoricals = compute_revised_information_loss_metric_categoricals(
        non_suppressed_records,
        qids,
        categorical_qids,
        qid_to_gtree,
        join_suffix_anon,
    )

    return rilm_numericals, rilm_categoricals


def _compute_revised_information_loss_metric_numericals(
    non_suppressed_records: pd.DataFrame,
    qids: list[str],
    numerical_qids: list[str],
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> dict[str, float]:
    """
    Compute the Revised Information Loss Metric (RILM) for numerical QIDs.

    This function calculates RILM for numerical columns by measuring how much of the
    original data's range (perimeter) is preserved after anonymization. Higher RILM
    values indicate better preservation of the original data distributions.

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        Output of get_non_suppressed_records containing non-suppressed records with
        both original and anonymized values.
    qids : List[str]
        List of all quasi-identifier column names.
    numerical_qids : List[str]
        List of numerical quasi-identifier column names (filtered subset of qids).
    join_suffix_orig : str
        Suffix that was added to original column names during the join operation.
    join_suffix_anon : str
        Suffix that was added to anonymized column names during the join operation.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping numerical QID names to their RILM scores. Values range
        from 0.0 (maximum information loss) to 1.0 (no information loss).

    Notes
    -----
    The function groups records by equivalence class and calculates a weighted average
    of the RILM scores for each numerical attribute across all equivalence classes.

    See Also
    --------
    compute_revised_information_loss_metric_value_for_numerical : Calculates the RILM
        value for a single numerical attribute in an equivalence class.
    compute_revised_information_loss_metric_categoricals : Computes RILM for
        categorical QIDs.
    """
    if len(non_suppressed_records) == 0 or len(numerical_qids) == 0:
        return {qid: NOT_DEFINED_NA for qid in numerical_qids}

    orig_numerical_qids = [f"{qid}{join_suffix_orig}" for qid in numerical_qids]
    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]

    perim_ois = {
        orig_qid: max_minus_min(non_suppressed_records[orig_qid].to_numpy())
        for orig_qid in orig_numerical_qids
    }
    equivalence_classes_df = non_suppressed_records.groupby(anon_qids, dropna=False)

    rilm = {}
    for qid, orig_qid in zip(numerical_qids, orig_numerical_qids):
        perim_oi = perim_ois[orig_qid]
        if not np.isnan(perim_oi):
            rilm_df = pd.DataFrame(
                {
                    "ril_ei": equivalence_classes_df.apply(
                        lambda ec_df, orig_qid=orig_qid, perim_oi=perim_oi: (
                            compute_revised_information_loss_metric_value_for_numerical(
                                perim_oi, max_minus_min(ec_df[orig_qid].to_numpy())
                            )
                        )
                    ),
                    "e_len": equivalence_classes_df.apply(len),
                }
            )
            # weighted average, ignoring nan ril_ei values
            rilm[qid] = ((rilm_df["e_len"] * rilm_df["ril_ei"]).sum()) / (
                (rilm_df["e_len"] * rilm_df["ril_ei"].isna().apply(lambda x: int(not x))).sum()
            )
    return rilm


def compute_revised_information_loss_metric_value_for_numerical(
    perim_oi: float,
    perim_ec: float,
) -> float:
    """
    Compute a single RILM value for a numerical QID in an equivalence class.

    For numerical QIDs, RILM measures how much of the original data's range (perimeter)
    is preserved after anonymization. The formula is:

    RILM = 1.0 - (perimeter_of_equivalence_class / perimeter_of_original_data)

    Parameters
    ----------
    perim_oi : float
        The "domain size" from the original ILM paper, calculated as max - min
        of the QID's original values for the entire dataset. For QIDs with
        heavy tails, callers may modify this definition dynamically to better
        accommodate those tails.
    perim_ec : float
        The max - min range for the QID's original values within the
        equivalence class in question.

    Returns
    -------
    float
        RILM score for this numerical QID in this equivalence class. Higher values
        (closer to 1.0) indicate better information preservation.

    Notes
    -----
    The RILM calculation inverts the ILM approach to create a metric where:
    - 1.0 means perfect information preservation (no generalization)
    - 0.0 means maximum information loss (complete generalization)
    - NaN is returned when perimeter values are NaN

    The formula is designed so that smaller equivalence classes (with less range/generalization)
    receive higher scores, reflecting better preservation of the original data distribution.
    """
    if not np.isnan(perim_ec) and not np.isnan(perim_oi):
        return 1.0 - ((perim_ec / perim_oi) if perim_oi > 0 else 0.0)
    else:
        return np.nan


def compute_revised_information_loss_metric_categoricals(
    non_suppressed_records: pd.DataFrame,
    qids: list[str],
    categorical_qids: list[str],
    qid_to_gtree: dict[str, GTree],
    join_suffix_anon: str,
) -> dict[str, float]:
    """
    Compute the Revised Information Loss Metric (RILM) for categorical QIDs.

    This function calculates RILM for categorical columns by measuring how far up the
    generalization hierarchy (g-tree) values have been moved during anonymization.
    Each node in the g-tree has a "geometric size" that increases as you move up the tree.

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        Output of get_non_suppressed_records containing non-suppressed records with
        both original and anonymized values.
    qids : List[str]
        List of all quasi-identifier column names.
    categorical_qids : List[str]
        List of categorical quasi-identifier column names (filtered subset of qids).
    qid_to_gtree : Dict[str, GTree]
        Mapping from categorical QID names to their associated generalization trees (GTrees).
        These trees define the generalization hierarchy for each categorical attribute.
    join_suffix_anon : str
        Suffix that was added to anonymized column names during the join operation.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping categorical QID names to their RILM scores. Values range
        from 0.0 (maximum information loss) to 1.0 (no information loss).

    Notes
    -----
    For each categorical QID, this function:

    1. Groups records by equivalence class
    2. Calculates the RILM score for each equivalence class based on how far up the
       generalization hierarchy the values have been moved and the "geometric size"
    3. Computes a weighted average of RILM scores across all equivalence classes

    The calculation handles special cases such as:

    - Empty dataframes or no categorical QIDs
    - Missing generalization trees for certain QIDs
    - QID values that are NaN (Not-a-Number)

    See Also
    --------
    compute_revised_information_loss_metric_value_for_categorical : Calculates the RILM
        value for a single categorical attribute value.
    _compute_revised_information_loss_metric_numericals : Computes RILM for numerical QIDs.
    """
    if len(non_suppressed_records) == 0 or len(categorical_qids) == 0:
        return {qid: NOT_DEFINED_NA for qid in categorical_qids}

    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]
    anon_categorical_qids = [f"{qid}{join_suffix_anon}" for qid in categorical_qids]

    equivalence_classes_df = non_suppressed_records.groupby(anon_qids, dropna=False)

    rilm = {}
    for qid, anon_qid in zip(categorical_qids, anon_categorical_qids):
        if qid in qid_to_gtree:
            gtree = qid_to_gtree[qid]
            # -1 handles the edge case where ec_df.name will return a str rather than tuple(str)
            anon_qid_value_idx = anon_qids.index(anon_qid) if len(anon_qids) > 1 else -1
            rilm_df = pd.DataFrame(
                {
                    "ril_ei": equivalence_classes_df.apply(
                        lambda ec_df, qid=qid, gtree=gtree, anon_qid_value_idx=anon_qid_value_idx: (
                            compute_revised_information_loss_metric_value_for_categorical(
                                qid,
                                gtree,
                                (
                                    ec_df.name[anon_qid_value_idx]
                                    if anon_qid_value_idx != -1
                                    else ec_df.name
                                ),
                            )
                        )
                    ),
                    "e_len": equivalence_classes_df.apply(len),
                }
            )
            # weighted average, ignoring nan ril_ei values
            rilm[qid] = ((rilm_df["e_len"] * rilm_df["ril_ei"]).sum()) / (
                (rilm_df["e_len"] * rilm_df["ril_ei"].isna().apply(lambda x: int(not x))).sum()
            )
        else:
            rilm[qid] = NOT_DEFINED_NA
    return rilm


def compute_revised_information_loss_metric_value_for_categorical(
    qid: str, gtree: GTree, qid_value: Any
) -> float:
    """
    Compute a single RILM value for a categorical QID value.

    RILM for categoricals measures information loss based on the generalization hierarchy (g-tree).
    Each node in the g-tree has an associated "geometric size" which increases as you move up the tree.
    Leaf nodes have size 0, and the root has the maximum size.

    The RILM formula is: 1.0 - (geometric_size_of_node/geometric_size_of_root)

    Parameters
    ----------
    qid : str
        QID name (used only for error messages).
    gtree : GTree
        Generalization tree for this QID, defining the hierarchy of generalization
        levels from specific values (leaves) to the most general value (root).
    qid_value : Any
        Anonymized QID value to compute RILM for.

    Returns
    -------
    float
        RILM score for this categorical value. Higher values (closer to 1.0)
        indicate better information preservation.

    Notes
    -----
    The RILM score interpretations for categorical values:

    - For leaf nodes (original values): RILM = 1.0 (no information loss)
    - For root node (*): RILM = 0.0 (maximum information loss)
    - For intermediate nodes: RILM between 0 and 1 based on the proportion of
      geometric size of the node relative to the root's geometric size

    Special cases handled:

    - If the QID value is NaN: Returns NOT_DEFINED_NA
    - If the root geometric size is NaN: Returns NOT_DEFINED_NA
    - If the root geometric size is 0: Returns 0.0 (avoids division by zero)

    The function raises a ValueError if the QID value cannot be found in the
    generalization tree.
    """
    if gtree.root is not None:
        root_node = gtree.get_node(gtree.root)
        assert root_node is not None, "Root node not found in gtree"
        perim_oi = gtree.get_geometric_size(root_node)
    else:
        perim_oi = NOT_DEFINED_NA
    if np.isnan(perim_oi):
        return NOT_DEFINED_NA
    if qid_value != qid_value:  # shortcut to check for nan
        return NOT_DEFINED_NA
    gtree_node = gtree.get_highest_node_with_value(qid_value)
    if gtree_node is None:
        raise ValueError(f"{qid} missing node with value {qid_value}")
    return 1.0 - ((gtree.get_geometric_size(gtree_node) / perim_oi) if perim_oi != 0 else 0.0)
