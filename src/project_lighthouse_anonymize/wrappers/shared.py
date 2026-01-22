"""
Shared utility functions for anonymization wrappers.

This module contains utility functions that are shared across different anonymization
types and do not depend on specific anonymization implementations.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.gtrees import GTree, make_flat_default_gtree

# This maps from an input dataframe column's dtype to the expected anonymized output dataframe dtypes
# and also determines what input dtypes are allowed for anonymization.
# Only columns with dtypes present as keys in this mapping are supported.
_INPUT_TO_ALLOWED_OUTPUT_DTYPE = {
    np.dtype("object"): {np.dtype("object"), np.dtype("float64")},
    np.dtype("int64"): {np.dtype("int64"), np.dtype("float64")},
    np.dtype("float64"): {np.dtype("float64")},
}

# Error message for unsupported input dtypes
_INPUT_TO_ALLOWED_OUTPUT_DTYPE_MESSAGE = """
Unsupported column dtype detected. The following dtypes are supported: {supported_dtypes}.

For unsupported dtypes, please convert them before and after anonymization using the
conversion utilities in project_lighthouse_anonymize.wrappers.dtype_conversion:

- bool: Use convert_bool_to_float() before anonymization, convert_float_to_bool() after
- datetime64: Use convert_datetime_to_float() before, convert_float_to_datetime() after
- categorical: Use convert_categorical_to_object() before, convert_object_to_categorical() after

Example usage:
  from project_lighthouse_anonymize.wrappers.dtype_conversion import (
      convert_bool_to_float, convert_float_to_bool
  )

  # Before anonymization
  df_converted, metadata = convert_bool_to_float(df, 'bool_col')

  # Perform anonymization on df_converted...

  # After anonymization
  final_df = convert_float_to_bool(anon_df, 'bool_col', metadata)

- Int64 (pandas nullable integer): Use float64 instead (numpy/scipy don't support nullable integers)

See project_lighthouse_anonymize.wrappers.dtype_conversion.DTYPE_CONVERSION_MAP for all supported conversions.
""".strip()


def default_dq_metric_to_minimum_dq() -> dict[str, float]:
    """
    Provide default recommended thresholds for data quality metrics.

    This function returns a dictionary mapping data quality metric names to their
    recommended minimum threshold values. These thresholds represent baseline
    quality standards that anonymized data should meet to maintain utility.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to threshold values
    """
    return {
        "pct_non_suppressed": 0.99,
        "rilm_categorical__minimum": 0.90,
        "pearsons__minimum": 0.90,
        "nmi_sampled_scaled_v1__minimum": 0.80,
    }


def check_dq_meets_minimum_thresholds(
    dq_metrics: dict[str, float],
    dq_metric_to_minimum_dq: Optional[dict[str, float]] = None,
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Check if data quality metrics meet minimum threshold requirements.

    This function determines whether a single set of data quality metrics meets
    all specified minimum thresholds. It uses the same logic as select_best_run
    but focuses only on threshold validation for a single metric set.

    Parameters
    ----------
    dq_metrics : Dict[str, float]
        Dictionary of data quality metrics to validate
    dq_metric_to_minimum_dq : Optional[Dict[str, float]], default=None
        Optional mapping from data quality metric names to minimum thresholds;
        if not provided, uses the output from default_dq_metric_to_minimum_dq()

    Returns
    -------
    Tuple[bool, List[Tuple[str, str]]]
        A tuple containing two elements. The first element is a boolean indicating
        if all thresholds are met. The second element is a list of tuples for failing
        metrics, where each tuple contains the failing metric name and a descriptive
        error message.

    Notes
    -----
    Missing data quality metrics are considered as passing the threshold check.
    NaN values in metrics fail if the corresponding threshold is not NaN.
    When a threshold is NaN, that check is effectively disabled.

    TODO(Later): Improve error messages to show original metric names instead of
    aggregated names like "rilm_categorical__minimum" when possible.
    """
    if dq_metric_to_minimum_dq is None:
        dq_metric_to_minimum_dq = default_dq_metric_to_minimum_dq()

    failure_details = []

    for dq_metric, minimum_dq in dq_metric_to_minimum_dq.items():
        # allow missing data quality metrics
        if dq_metric in dq_metrics:
            dq_value = dq_metrics[dq_metric]
            if np.isnan(dq_value) and not np.isnan(minimum_dq):
                failure_details.append(
                    (
                        dq_metric,
                        f"{dq_metric} (nan) fails non-nan threshold {minimum_dq}",
                    )
                )
            elif not np.isnan(minimum_dq) and dq_value < minimum_dq:
                failure_details.append((dq_metric, f"{dq_metric} ({dq_value}) < {minimum_dq}"))

    return len(failure_details) == 0, failure_details


def select_best_run(
    dq_metrics_arr: list[dict[str, float]],
    dq_metric_to_minimum_dq: Optional[dict[str, float]] = None,
) -> tuple[int, bool]:
    """
    Select the best anonymization run from multiple candidates based on data quality.

    This function compares multiple anonymization results and selects the one that
    best preserves data utility while meeting quality thresholds. It first filters
    runs that meet all minimum thresholds, then scores the remaining candidates to
    find the optimal one.

    Parameters
    ----------
    dq_metrics_arr : List[Dict[str, float]]
        List of data quality metric dictionaries, each representing the metrics
        from one anonymization run (the second output from k_anonymize calls)
    dq_metric_to_minimum_dq : Optional[Dict[str, float]], default=None
        Optional mapping from data quality metric names to minimum thresholds;
        if not provided, uses the output from default_dq_metric_to_minimum_dq()

    Returns
    -------
    Tuple[int, bool]
        A tuple containing two elements. The first element is the index of the best
        run in the input array (0-based). The second element is a boolean indicating
        whether the selected run meets all minimum data quality thresholds (True) or
        not (False).

    Notes
    -----
    The scoring function uses a piecewise sigmoid approach to prioritize runs
    that exceed the minimum thresholds by appropriate margins. When no runs meet
    all thresholds, the function will select the best available option and
    return False for minimum_dq_met.
    """
    dq_metrics_arr = list(
        dq_metrics_arr
    )  # in case dq_metrics_arr was a generator, which we don't want
    if len(dq_metrics_arr) == 0:
        raise ValueError("dq_metrics_arr must have at-least one element")
    if dq_metric_to_minimum_dq is None:
        dq_metric_to_minimum_dq = default_dq_metric_to_minimum_dq()

    idx_and_dq_metrics_arr = list(enumerate(dq_metrics_arr))  # preserve index for return value

    def filter_pass_1(
        dq_metrics: dict[str, float],
        dq_metric_to_minimum_dq: Optional[dict[str, float]] = dq_metric_to_minimum_dq,
    ) -> bool:
        passes, _ = check_dq_meets_minimum_thresholds(dq_metrics, dq_metric_to_minimum_dq)
        return passes

    idx_and_dq_metrics_arr_filtered = [
        (idx, dq_metrics) for idx, dq_metrics in idx_and_dq_metrics_arr if filter_pass_1(dq_metrics)
    ]
    if len(idx_and_dq_metrics_arr_filtered) == 0:
        idx_and_dq_metrics_arr_filtered = idx_and_dq_metrics_arr
        minimum_dq_met = False
    else:
        minimum_dq_met = True

    if not minimum_dq_met:
        # If a data quality metric threshold is non-nan and all instances of that data quality metric are nan, then ignore that data quality metric
        # threshold and re-check to see if minimum data quality is met for the remaining data quality metric values.
        dq_metric_to_minimum_dq_modified = dq_metric_to_minimum_dq.copy()
        for dq_metric, minimum_dq in dq_metric_to_minimum_dq.items():
            dq_values = [
                dq_metric_dict[dq_metric]
                for dq_metric_dict in dq_metrics_arr
                if dq_metric in dq_metric_dict
            ]
            if np.isnan(dq_values).all() and not np.isnan(minimum_dq):
                dq_metric_to_minimum_dq_modified[dq_metric] = np.nan

        idx_and_dq_metrics_arr_filtered = [
            (idx, dq_metrics)
            for idx, dq_metrics in idx_and_dq_metrics_arr
            if filter_pass_1(dq_metrics, dq_metric_to_minimum_dq_modified)
        ]
        if len(idx_and_dq_metrics_arr_filtered) == 0:
            idx_and_dq_metrics_arr_filtered = idx_and_dq_metrics_arr
            minimum_dq_met = False
        else:
            minimum_dq_met = True

    def _compute_score(
        dq_metrics: dict[str, float],
        dq_metric_to_minimum_dq: Optional[dict[str, float]] = dq_metric_to_minimum_dq,
    ) -> float:
        return compute_score(dq_metrics, dq_metric_to_minimum_dq)

    idx, dq_metrics = max(  # type: ignore[reportUnusedVariable]  # dq_metrics needed for tuple unpacking but only idx is used
        idx_and_dq_metrics_arr_filtered,
        key=lambda tup: (_compute_score(tup[1])),  # (idx, dq_metrics)
    )
    return idx, minimum_dq_met


def compute_score(
    dq_metrics: dict[str, float],
    dq_metric_to_minimum_dq: Optional[dict[str, float]] = None,
) -> float:
    """
    Calculate a quality score for an anonymization result.

    This function evaluates the quality of a candidate anonymization solution by
    comparing its data quality metrics against threshold values. It produces a
    composite score that represents how well the solution preserves data utility.

    Parameters
    ----------
    dq_metrics : Dict[str, float]
        Dictionary of data quality metrics for the candidate solution
    dq_metric_to_minimum_dq : Optional[Dict[str, float]], default=None
        Target data quality metric thresholds; if None, defaults from
        default_dq_metric_to_minimum_dq() will be used

    Returns
    -------
    float
        A composite score where higher values indicate better data quality

    Notes
    -----
    The scoring uses a piecewise sigmoid function to:
    1. Penalize solutions that don't meet minimum thresholds
    2. Reward solutions that exceed thresholds, with diminishing returns
    3. Balance the importance of different metrics in the final score

    The sigmoid parameters have been tuned to provide appropriate sensitivity
    at different threshold levels.
    """
    if dq_metric_to_minimum_dq is None:
        dq_metric_to_minimum_dq = default_dq_metric_to_minimum_dq()
    total_score = 0.0
    for dq_metric, minimum_dq in dq_metric_to_minimum_dq.items():
        if dq_metric in dq_metrics:
            # allow missing data quality metrics
            dq_value = dq_metrics[dq_metric]
            # the score function is a piecwise sigmoid function, where we are capturing an
            # appropriately stretched (sig_l) of portion of the standard logistic function >= 0,
            # ie. the portion where the derivative is decreasing
            diff = dq_value - minimum_dq
            if diff <= 0:
                sig_l = (
                    -3.45
                    + 158 * minimum_dq
                    + -989 * math.pow(minimum_dq, 2)
                    + 2948 * math.pow(minimum_dq, 3)
                    + -4426 * math.pow(minimum_dq, 4)
                    + 3229 * math.pow(minimum_dq, 5)
                    + -906 * math.pow(minimum_dq, 6)
                )
                score = minimum_dq * (
                    ((1 / (1 + math.exp(-sig_l * (diff / minimum_dq + 1)))) / 0.5) - 2
                )
            else:
                sig_l = (
                    11.2
                    + -54.2 * minimum_dq
                    + 293 * math.pow(minimum_dq, 2)
                    + -546 * math.pow(minimum_dq, 3)
                    + 369 * math.pow(minimum_dq, 4)
                )
                score = (((1 / (1 + math.exp(-sig_l * diff))) - 0.5) / 0.5) * (1 - minimum_dq)
            total_score += score
    return total_score


def prepare_gtrees(
    logger: logging.Logger,
    input_df: pd.DataFrame,
    qids: list[str],
    qid_to_gtree: dict[str, GTree],
) -> dict[str, GTree]:
    """
    Prepare and validate generalization trees for anonymization.

    This function processes the input gtrees by dropping unneeded ones, adding
    default gtrees for object columns that don't have explicit gtrees, ensuring
    geometric sizes are set, and updating internal maps for efficient gtree use.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording gtree preparation process
    input_df : pd.DataFrame
        Input dataframe to analyze for gtree creation
    qids : List[str]
        List of quasi-identifier column names
    qid_to_gtree : Dict[str, GTree]
        Mapping from column names to generalization trees

    Returns
    -------
    Dict[str, GTree]
        Processed and validated generalization trees ready for anonymization
    """
    # drop unneeded gtrees
    qid_to_gtree = dict(qid_to_gtree)
    qid_to_gtree = {qid: gtree for qid, gtree in qid_to_gtree.items() if qid in input_df.columns}
    # add default gtrees
    for qid in qids:
        if (
            input_df.dtypes[qid] == np.dtype("object")
            and qid not in qid_to_gtree
            and any(~input_df[qid].isna())
        ):
            gtree = make_flat_default_gtree(set(input_df[~input_df[qid].isna()][qid]))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Creating default flat gtree for %s\n%s\n%s",
                    qid,
                    gtree.pprint(),
                    gtree.pprint_geometric_sizes(),
                )
            qid_to_gtree[qid] = gtree
    # geometric sizes
    for qid, gtree in qid_to_gtree.items():
        if gtree.root is not None:
            root_node = gtree.get_node(gtree.root)
            assert root_node is not None, f"Root node {gtree.root} not found in gtree for {qid}"
            if np.isnan(gtree.get_geometric_size(root_node)):
                logger.warning(f"GTree for {qid} is missing geometric sizes, adding defaults")
                gtree.add_default_geometric_sizes()
    # internal maps for efficient gtree use
    for qid, gtree in qid_to_gtree.items():
        gtree.update_highest_node_with_value_if()
        gtree.update_descendant_leaf_values_if()
        gtree.update_lowest_node_with_descendant_leaves_if()

    return qid_to_gtree


def validate_input_dtypes(input_df: pd.DataFrame, qids: list[str]) -> None:
    """
    Validate that QID columns use supported dtypes for anonymization.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe to validate
    qids : List[str]
        Quasi-identifier column names to validate

    Raises
    ------
    ValueError
        If any QID column has an unsupported dtype, with detailed error message
        including conversion instructions
    """
    unsupported_dtypes = []

    for col in qids:
        dtype = input_df[col].dtype
        if dtype not in _INPUT_TO_ALLOWED_OUTPUT_DTYPE:
            unsupported_dtypes.append((col, dtype))

    if unsupported_dtypes:
        supported_dtypes = list(_INPUT_TO_ALLOWED_OUTPUT_DTYPE.keys())
        error_msg = _INPUT_TO_ALLOWED_OUTPUT_DTYPE_MESSAGE.format(supported_dtypes=supported_dtypes)
        unsupported_info = "\n".join(
            [f"Column '{col}': {dtype}" for col, dtype in unsupported_dtypes]
        )
        raise ValueError(f"Unsupported dtypes found:\n{unsupported_info}\n\n{error_msg}")


def validate_anonymization_output(
    input_df: pd.DataFrame, anon_df: pd.DataFrame, qids: list[str]
) -> None:
    """
    Validate anonymization output dtype consistency for QID columns only.

    Parameters
    ----------
    input_df : pd.DataFrame
        Original input dataframe
    anon_df : pd.DataFrame
        Anonymized output dataframe
    qids : List[str]
        QID column names to validate dtypes for

    Raises
    ------
    AssertionError
        If column subsets don't match or QID dtypes are not as expected
    """
    assert set(input_df.columns) <= set(anon_df.columns), (
        f"input_df.columns ({' '.join(input_df.columns)}) not a subset of anon_df.columns ({' '.join(anon_df.columns)})"
    )

    for col in qids:
        input_dtype, output_dtype = input_df.dtypes[col], anon_df.dtypes[col]
        allowed_dtypes = _INPUT_TO_ALLOWED_OUTPUT_DTYPE[input_dtype]
        assert output_dtype in allowed_dtypes, (  # type: ignore[operator]
            f"QID column ({col}) dtype ({input_dtype}) has associated output dtype "
            f"({output_dtype}) not one of expected ({allowed_dtypes})"
        )
