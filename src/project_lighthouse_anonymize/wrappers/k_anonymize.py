"""
K-anonymity wrapper functions for anonymizing data.

This module provides k-anonymity anonymization capabilities. It contains the core
anonymization function and all shared utilities needed for k-anonymity processing.
"""

import concurrent.futures as cf
import logging
import multiprocessing as mp
from pprint import pformat
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.data_quality_metrics.misc import (
    compute_average_equivalence_class_metric,
    compute_discernibility_metric,
    compute_suppression_metrics,
)
from project_lighthouse_anonymize.data_quality_metrics.nmi import (
    compute_normalized_mutual_information_sampled_scaled,
)
from project_lighthouse_anonymize.data_quality_metrics.pearson import (
    compute_pearsons_correlation_coefficients,
)
from project_lighthouse_anonymize.data_quality_metrics.rilm_ilm import (
    compute_average_information_loss_metric,
    compute_revised_information_loss_metric,
)
from project_lighthouse_anonymize.disclosure_risk_metrics import (
    calculate_p_k,
)
from project_lighthouse_anonymize.futures import make_future
from project_lighthouse_anonymize.gtrees import GTree
from project_lighthouse_anonymize.mondrian.core import CoreMondrian
from project_lighthouse_anonymize.mondrian.implementation import (
    Implementation_Base,
    NumericalCutPointsMode,
)
from project_lighthouse_anonymize.mondrian.k_anonymity import Implementation_KAnonymity
from project_lighthouse_anonymize.mondrian.original import OriginalMondrian
from project_lighthouse_anonymize.pandas_utils import get_temp_col
from project_lighthouse_anonymize.utils import (
    AnonymizeCallbacks,
    get_generalized_records,
    get_non_suppressed_records,
    get_suppressed_records,
)

from .shared import (
    default_dq_metric_to_minimum_dq,
    prepare_gtrees,
    validate_anonymization_output,
    validate_input_dtypes,
)


def compute_metrics_with_callbacks(
    logger: logging.Logger,
    input_df: pd.DataFrame,
    anon_df: pd.DataFrame,
    qids: list[str],
    qid_to_gtree: dict[str, GTree],
    id_col: str,
    exclude_qids: list[str],
    parallelism: Optional[int],
    anonymize_callbacks: Optional[AnonymizeCallbacks],
    rnd_metrics: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute metrics with appropriate callbacks.

    This function wraps the metrics computation with the appropriate callbacks.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording the metrics computation process
    input_df : pd.DataFrame
        Original input dataframe before anonymization
    anon_df : pd.DataFrame
        Anonymized output dataframe
    qids : List[str]
        List of column names representing quasi-identifiers
    qid_to_gtree : Dict[str, GTree]
        Mapping from column name to generalization tree
    id_col : str
        Unique identifier column
    exclude_qids : List[str]
        QIDs to exclude from computing overall minimum metric scores
    parallelism : Optional[int]
        Number of concurrent processes to use
    anonymize_callbacks : Optional[AnonymizeCallbacks]
        Optional callback object for tracking performance metrics
    rnd_metrics : bool, default=False
        If True, computes additional research-oriented metrics

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        A tuple containing:
        1. Data quality metrics
        2. Disclosure risk metrics
    """
    if anonymize_callbacks is not None:
        anonymize_callbacks.compute_metrics_bm()

    dq_metrics, disclosure_metrics = compute_anonymization_data_quality_metrics(
        logger,
        input_df,
        anon_df,
        qids,
        qid_to_gtree,
        id_col,
        exclude_qids,
        parallelism,
        rnd_metrics=rnd_metrics,
    )

    if anonymize_callbacks is not None:
        anonymize_callbacks.compute_metrics_am()

    return dq_metrics, disclosure_metrics


def anonymize(
    logger: logging.Logger,
    input_df: pd.DataFrame,
    qids: list[str],
    qid_to_gtree: dict[str, GTree],
    id_col: str,
    implementation: Implementation_Base,
    dq_metric_to_minimum_dq: dict[str, float],
    exclude_qids: Optional[list[str]] = None,
    rnd_mode: bool = False,
    parallelism: Optional[int] = 10,
    anonymize_callbacks: Optional[AnonymizeCallbacks] = None,
    timeout_s: Optional[int] = None,
    recursive_partition_size_cutoff: int = 1_000,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Anonymize an input dataframe using the specified implementation.

    This function is the core anonymization gateway that applies a privacy model
    implementation to transform data while maintaining privacy guarantees and
    maximizing data utility. It handles validation, generalization tree setup,
    anonymization, and metrics computation.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording the anonymization process
    input_df : pd.DataFrame
        Input dataframe to anonymize; the index may be discarded
    qids : List[str]
        Quasi-identifier column names (attributes that, in combination,
        might identify individuals)
    qid_to_gtree : Dict[str, GTree]
        Mapping from column name to generalization tree; note that the
        GTrees may be modified during processing
    id_col : str
        Unique identifier column; the index from input_df is ignored
        and may be discarded
    implementation : Implementation_Base
        Anonymization implementation (e.g., Implementation_KAnonymity)
        that defines the privacy model to enforce
    dq_metric_to_minimum_dq : Dict[str, float]
        Target data quality metric thresholds
    exclude_qids : Optional[List[str]], default=None
        Optional list of QIDs to exclude from computing overall data
        quality metric minimum scores (X__minimum)
    rnd_mode : bool, default=False
        If True, extra debugging information for R&D is enabled
    parallelism : Optional[int], default=10
        Number of concurrent processes to use for anonymizing and
        computing metrics; None means no parallelism
    anonymize_callbacks : Optional[AnonymizeCallbacks], default=None
        Optional callback object for tracking performance metrics
    timeout_s : Optional[int], default=None
        Optional timeout in seconds; when hit, no further cuts will be
        pursued by the Core Mondrian algorithm (wrapping up quickly at
        the cost of data quality)
    recursive_partition_size_cutoff : int, default=1_000
        Size cutoff for recursive partitioning in the Core Mondrian algorithm.
        This parameter should only be modified for research and development purposes.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]
        A tuple containing:
        1. An anonymized dataframe
        2. A dictionary of data quality metrics
        3. A dictionary of disclosure risk metrics

    Raises
    ------
    Exception
        If an error occurs or the implementation cannot meet the
        target technical privacy model requirements
    """
    exclude_qids = [] if exclude_qids is None else list(exclude_qids)
    #
    # Prepare
    #
    input_df = input_df.copy(deep=True)  # input dataframe is mutated below
    if pd.isna(input_df[id_col]).any():  # type: ignore[reportGeneralTypeIssues]  # .any() returns bool, not Series
        raise ValueError("id_col must not contain NaN values")
    logger.debug("input_df columns: %s", input_df.columns)
    logger.debug("input_df dtypes:\n%s", input_df.dtypes)
    qid_to_gtree = prepare_gtrees(logger, input_df, qids, qid_to_gtree)

    validate_input_dtypes(input_df, qids)

    # support no QIDs
    if len(qids) == 0:
        placeholder_qid = get_temp_col(input_df, col_prefix="placeholder_qid_")
        logger.debug("introducing placeholder qid: %s", placeholder_qid)
        input_df[placeholder_qid] = 1
        qids = qids + [
            placeholder_qid,
        ]
    else:
        placeholder_qid = None

    if implementation.could_be_a_final_cut(input_df, qids, qid_to_gtree):
        if len(input_df) == 0:
            logger.debug(
                "Not modifying empty input dataset",
            )
            anon_df = input_df.copy()
        elif len(qids) == 0:
            logger.debug(
                "Not modifying input dataset with no qids",
            )
            anon_df = input_df.copy()
        else:
            if implementation.validate(input_df, qids):
                logger.debug(
                    "Skipping Anonymize for input dataset size = %d that already meets technical privacy model",
                    len(input_df),
                )
                anon_df = input_df.copy()
            else:
                logger.debug(
                    "Anonymizing input dataset size = %d",
                    len(input_df),
                )
                core = CoreMondrian(
                    logger,
                    implementation,
                    recursive_partition_size_cutoff,
                    parallelism,
                    deterministic_identifiers=rnd_mode,
                    track_stats=rnd_mode,
                )
                node_identifier_col = "node_identifier" if rnd_mode else None
                if anonymize_callbacks is not None:
                    anonymize_callbacks.anonymize_bm()
                anon_df = core.anonymize(
                    input_df,
                    qids,
                    qid_to_gtree,
                    exclude_qids,
                    dq_metric_to_minimum_dq,
                    timeout_s=timeout_s,
                    track_qid_cuts=rnd_mode,
                    node_identifier_col=node_identifier_col,
                )
                if anonymize_callbacks is not None:
                    anonymize_callbacks.anonymize_am()
    else:
        if len(input_df) > 0:
            logger.debug(
                "Suppressing input dataset size = %d that couldn't be a final cut",
                len(input_df),
            )
        anon_df = input_df.sample(0)

    dq_metrics, disclosure_metrics = compute_metrics_with_callbacks(
        logger,
        input_df,
        anon_df,
        qids,
        qid_to_gtree,
        id_col,
        exclude_qids,
        parallelism,
        anonymize_callbacks,
        rnd_metrics=rnd_mode,
    )

    #
    # Validations and final cleanup
    #
    if len(anon_df) > 0:
        assert implementation.validate(anon_df, qids), (
            "Anonymized dataframe doesn't meet technical privacy model"
        )

    validate_anonymization_output(input_df, anon_df, qids)
    assert len(anon_df) <= len(input_df), (
        f"number of output rows ({len(anon_df)}) not <= number of input rows ({len(input_df)})"
    )

    if placeholder_qid is not None:
        logger.debug("Removing placeholder qid: %s", placeholder_qid)
        anon_df.drop([placeholder_qid], axis="columns", inplace=True)
        for dq_k in list(dq_metrics.keys()):
            if placeholder_qid in dq_k:
                del dq_metrics[dq_k]
        qids = [qid for qid in qids if qid != placeholder_qid]
    return anon_df, dq_metrics, disclosure_metrics


def k_anonymize(
    logger: logging.Logger,
    input_df: pd.DataFrame,
    qids: list[str],
    k: int,
    qid_to_gtree: dict[str, GTree],
    id_col: str,
    exclude_qids: Optional[list[str]] = None,
    rnd_mode: bool = False,
    dq_metric_to_minimum_dq: Optional[dict[str, float]] = None,
    rilm_score_epsilon: float = 0.05,
    rilm_score_epsilon_partition_size_cutoff: float = 0.90,
    complex_numerical_cut_points_modes: bool = False,
    dynamic_breakout_rilm_multiplier: Optional[float] = None,
    parallelism: Optional[int] = 10,
    anonymize_callbacks: Optional[AnonymizeCallbacks] = None,
    timeout_s: Optional[int] = None,
    use_original_mondrian: bool = False,
    recursive_partition_size_cutoff: int = 1_000,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Apply k-anonymity to an input dataframe.

    This function creates a k-anonymity implementation and applies it to the data.
    It handles configuration of the anonymization process and validation of the results.

    IMPORTANT: When use_original_mondrian=True, the OriginalMondrian implementation
    is used, which is intended for research and development purposes only. For
    production use cases, please use the default CoreMondrian implementation.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording the anonymization process
    input_df : pd.DataFrame
        Input dataframe to anonymize; the index may be discarded
    qids : List[str]
        List of column names representing quasi-identifiers
    k : int
        Target k value (minimum number of records per equivalence class)
    qid_to_gtree : Dict[str, GTree]
        Mapping from column name to generalization tree; note that the
        GTrees may be modified during processing
    id_col : str
        Unique identifier column; the index from input_df is ignored
        and may be discarded
    exclude_qids : Optional[List[str]], default=None
        Optional list of QIDs to exclude from computing overall data
        quality metric minimum scores
    rnd_mode : bool, default=False
        If True, extra debugging information for R&D is enabled
    dq_metric_to_minimum_dq : Optional[Dict[str, float]], default=None
        Target data quality metric thresholds; if None, defaults from
        default_dq_metric_to_minimum_dq() will be used
    rilm_score_epsilon : float, default=0.05
        Threshold for considering multiple cut choices in the Core Mondrian
        algorithm; -1.0 means only the first appropriate cut choice will
        be considered
    rilm_score_epsilon_partition_size_cutoff : float, default=0.90
        Multiple appropriate cut choices will only be considered if within
        rilm_score_epsilon AND the partition size is <= input size *
        this cutoff value
    complex_numerical_cut_points_modes : bool, default=False
        Whether to use a simple (single cut point) or complex (multiple cut
        points) approach for numerical cuts
    dynamic_breakout_rilm_multiplier : Optional[float], default=None
        RILM score multiplier for dynamic cut point selection; None disables
        this feature
    parallelism : Optional[int], default=10
        Number of concurrent processes to use; None means no parallelism
    anonymize_callbacks : Optional[AnonymizeCallbacks], default=None
        Optional callback object for tracking performance metrics
    timeout_s : Optional[int], default=None
        Optional timeout in seconds; when hit, no further cuts will be
        pursued by the Core Mondrian algorithm
    use_original_mondrian : bool, default=False
        Whether to use the OriginalMondrian implementation; if True, only
        numerical QIDs are supported and microaggregation is used instead
        of range-based generalization
    recursive_partition_size_cutoff : int, default=1_000
        Size cutoff for recursive partitioning in the Core Mondrian algorithm.
        This parameter should only be modified for research and development purposes.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]
        A tuple containing:
        1. A k-anonymous dataframe
        2. A dictionary of data quality metrics
        3. A dictionary of disclosure risk metrics

    Raises
    ------
    Exception
        If an error occurs or the implementation cannot meet the target k value

    Notes
    -----
    When use_original_mondrian=True, the OriginalMondrian implementation will be used
    instead of the standard Mondrian implementation. Note that OriginalMondrian only
    supports numerical QIDs and uses microaggregation (averaging) instead of range-based
    generalization.
    """
    if dq_metric_to_minimum_dq is None:
        dq_metric_to_minimum_dq = default_dq_metric_to_minimum_dq()
    implementation: Implementation_Base
    numerical_cut_points_modes = (
        (
            NumericalCutPointsMode.MEDIAN,
            NumericalCutPointsMode.BIN_EDGES,
        )
        if complex_numerical_cut_points_modes
        else (NumericalCutPointsMode.MEDIAN,)
    )
    if use_original_mondrian:
        # Handle OriginalMondrian separately - it's a standalone implementation not integrated with CoreMondrian
        # Check that all QIDs are numerical for OriginalMondrian
        for qid in qids:
            if not np.issubdtype(input_df[qid].dtype, np.number):  # type: ignore[reportArgumentType]  # dtype is valid for issubdtype but pyright stubs are overly strict
                raise ValueError(
                    f"QID {qid} is not numerical. OriginalMondrian only supports numerical QIDs."
                )

            # Check that no QIDs contain NaN values
            if np.any(input_df[qid].isna()):
                raise ValueError(
                    f"QID {qid} contains NaN values. OriginalMondrian does not support NaN values."
                )

        logger.info(
            "Anonymizing using Original Mondrian k-anonymity with k = %d",
            k,
        )

        if anonymize_callbacks is not None:
            anonymize_callbacks.anonymize_bm()

        # Use OriginalMondrian directly
        implementation = OriginalMondrian(logger, k)  # type: ignore[assignment]
        anon_df = implementation.anonymize(input_df, qids)  # type: ignore[attr-defined]

        if anonymize_callbacks is not None:
            anonymize_callbacks.anonymize_am()

        # Compute metrics with the same function used for regular anonymization
        dq_metrics, disclosure_metrics = compute_metrics_with_callbacks(
            logger,
            input_df,
            anon_df,
            qids,
            qid_to_gtree,
            id_col,
            exclude_qids if exclude_qids else [],
            parallelism,
            anonymize_callbacks,
            rnd_metrics=rnd_mode,
        )
    else:
        # Handle standard k-anonymity with CoreMondrian
        logger.info(
            "Anonymizing using k-anonymity with k = %d",
            k,
        )
        implementation = Implementation_KAnonymity(
            logger,
            k,
            rilm_score_epsilon,
            rilm_score_epsilon_partition_size_cutoff,
            numerical_cut_points_modes,
            dq_metric_to_minimum_dq,
            dynamic_breakout_rilm_multiplier,
        )

        # Use CoreMondrian implementation
        anon_df, dq_metrics, disclosure_metrics = anonymize(
            logger,
            input_df,
            qids,
            qid_to_gtree,
            id_col,
            implementation,
            dq_metric_to_minimum_dq,
            exclude_qids,
            rnd_mode,
            parallelism,
            anonymize_callbacks=anonymize_callbacks,
            timeout_s=timeout_s,
            recursive_partition_size_cutoff=recursive_partition_size_cutoff,
        )
    return (anon_df, dq_metrics, disclosure_metrics)


def compute_anonymization_data_quality_metrics(
    logger: logging.Logger,
    input_df: pd.DataFrame,
    anon_df: pd.DataFrame,
    qids: list[str],
    qid_to_gtree: dict[str, GTree],
    id_col: str,
    exclude_qids: list[str],
    parallelism: Optional[int],
    skip_logging: bool = False,
    rnd_metrics: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute comprehensive data quality and disclosure risk metrics for anonymized data.

    This function calculates multiple metrics to evaluate the quality and privacy
    protection of anonymized data by comparing the original and anonymized datasets.
    It can run calculations in parallel and selectively compute certain metrics
    based on settings.

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording the metrics computation process
    input_df : pd.DataFrame
        Original input dataframe before anonymization
    anon_df : pd.DataFrame
        Anonymized output dataframe
    qids : List[str]
        List of column names representing quasi-identifiers
    qid_to_gtree : Dict[str, GTree]
        Mapping from column name to generalization tree
    id_col : str
        Unique identifier column for linking records between original and
        anonymized datasets
    exclude_qids : List[str]
        List of QIDs to exclude from computing overall minimum metric scores
    parallelism : Optional[int]
        Number of concurrent processes to use for computing metrics;
        None means no parallelism
    skip_logging : bool, default=False
        If True, suppresses logging of results
    rnd_metrics : bool, default=False
        If True, computes additional research-oriented metrics that are
        more computationally intensive

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        A tuple containing:

        1. A dictionary of data quality metrics including:

           - Suppression metrics (count and percentage)
           - Generalization metrics (count and percentage)
           - Information loss metrics (RILM, Pearson correlation)
           - Normalized mutual information metrics

        2. A dictionary of disclosure risk metrics including:

           - Actual k achieved
           - Non-unique fingerprint percentage

    Notes
    -----
    The function uses parallel processing when available to compute metrics
    efficiently. For large datasets, this can significantly improve performance.
    """
    non_qid_cols = {str(col) for col in input_df.columns} - set(qids)
    join_suffix_orig, join_suffix_anon = "_orig", "_anon"

    executor = (
        cf.ProcessPoolExecutor(max_workers=parallelism, mp_context=mp.get_context("spawn"))
        if parallelism is not None
        else None
    )

    try:
        # first kick-off potentially parallel execution of joins on data; if executor is None this does nothing
        non_suppressed_records = make_future(
            executor,
            get_non_suppressed_records,
            input_df,
            anon_df,
            id_col,
            join_suffix_orig,
            join_suffix_anon,
        )
        generalized_records = make_future(
            executor,
            get_generalized_records,
            input_df,
            anon_df,
            id_col,
            non_qid_cols,
            join_suffix_orig,
            join_suffix_anon,
        )
        suppressed_records = make_future(
            executor,
            get_suppressed_records,
            input_df,
            anon_df,
            id_col,
            non_qid_cols,
            join_suffix_orig,
            join_suffix_anon,
        )
        # finally collect results from parallel execution of joins on data; if executor is None this computes data quality metrics one by one in process
        non_suppressed_records = non_suppressed_records.result()
        generalized_records = generalized_records.result()
        suppressed_records = suppressed_records.result()

        dq_metrics, disclosure_metrics = __compute_anonymization_data_quality_metrics_impl(
            qids,
            qid_to_gtree,
            id_col,
            join_suffix_orig,
            join_suffix_anon,
            input_df,
            anon_df,
            non_suppressed_records,
            generalized_records,
            suppressed_records,
            exclude_qids,
            rnd_metrics,
            executor,
        )
        if not skip_logging:
            logger.debug("dq_metrics =\n%s", pformat(dq_metrics))
        return (dq_metrics, disclosure_metrics)
    finally:
        if executor is not None:
            executor.shutdown()


def __compute_anonymization_data_quality_metrics_impl(
    qids: list[str],
    qid_to_gtree: dict[str, GTree],
    id_col: str,
    join_suffix_orig: str,
    join_suffix_anon: str,
    input_df: pd.DataFrame,
    anon_df: pd.DataFrame,
    non_suppressed_records: pd.DataFrame,
    generalized_records: pd.DataFrame,
    suppressed_records: pd.DataFrame,
    exclude_qids: list[str],
    rnd_metrics: bool,
    executor: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute data quality metrics for K-Anonymize. This is the implementation method used above.

    qids: quasi-identifier column names
    qid_to_gtree: mapping from column name to GTree; note that the GTrees may be modified!
    id_col: unique identifier column
    join_suffix_orig: suffix to add to original column values at join
    join_suffix_anon: suffix to add to anonymized column values at join
    input_df: original input pd.DataFrame
    anon_df: anonymized input pd.DataFrame
    non_suppressed_records: output of get_non_suppressed_records
    generalized_records: output of get_generalized_records
    suppressed_records: output of get_suppressed_records
    exclude_qids: QIDs to exclude from computing overall data quality metric minimum scores (X__minimum)
    executor: executor to pass to make_future; if None no parallel processing will be used
    rnd_metrics: if True, extra dq metrics (for R&D) are computed and returned.

    Returns a tuple of:
    (1) dq metrics
    (2) disclosure risk metrics
    """
    dq_metrics: dict[str, Any] = {}
    disclosure_metrics = {}

    # first kick-off potentially parallel execution of data quality metrics; if executor is None this does nothing
    if rnd_metrics:
        dq_metrics["average_information_loss_metric"] = make_future(
            executor,
            compute_average_information_loss_metric,
            non_suppressed_records,
            qids,
            join_suffix_orig,
            join_suffix_anon,
        )
        dq_metrics["discernibility_metric"] = make_future(
            executor,
            compute_discernibility_metric,
            non_suppressed_records,
            suppressed_records,
            qids,
            id_col,
            join_suffix_orig,
            join_suffix_anon,
        )
        dq_metrics["average_equivalence_class_metric"] = make_future(
            executor,
            compute_average_equivalence_class_metric,
            non_suppressed_records,
            qids,
            join_suffix_anon,
        )

    compute_revised_information_loss_metric_future = make_future(
        executor,
        compute_revised_information_loss_metric,
        non_suppressed_records,
        qids,
        qid_to_gtree,
        join_suffix_orig,
        join_suffix_anon,
    )
    compute_pearsons_correlation_coefficients_future = make_future(
        executor,
        compute_pearsons_correlation_coefficients,
        non_suppressed_records,
        qids,
        join_suffix_orig,
        join_suffix_anon,
    )

    compute_normalized_mutual_information_sampled_scaled_future = make_future(
        executor,
        compute_normalized_mutual_information_sampled_scaled,
        non_suppressed_records,
        qids,
        join_suffix_orig,
        join_suffix_anon,
    )

    if rnd_metrics:
        compute_normalized_mutual_information_sampled_unscaled_future = make_future(
            executor,
            compute_normalized_mutual_information_sampled_scaled,
            non_suppressed_records,
            qids,
            join_suffix_orig,
            join_suffix_anon,
            scale=False,
        )
    else:
        compute_normalized_mutual_information_sampled_unscaled_future = None

    suppression_metrics = compute_suppression_metrics(input_df, anon_df)
    dq_metrics.update(suppression_metrics)

    dq_metrics["n_generalized"] = float(len(generalized_records))
    dq_metrics["pct_generalized"] = float(
        (len(generalized_records) / len(input_df)) if len(input_df) > 0 else np.nan
    )

    # finally collect results from parallel execution of data quality metrics; if executor is None this computes data quality metrics one by one in process
    if rnd_metrics:
        dq_metrics["average_information_loss_metric"] = dq_metrics[
            "average_information_loss_metric"
        ].result()
        dq_metrics["discernibility_metric"] = dq_metrics["discernibility_metric"].result()
        dq_metrics["average_equivalence_class_metric"] = dq_metrics[
            "average_equivalence_class_metric"
        ].result()
    # pearsons
    metric_dict = compute_pearsons_correlation_coefficients_future.result()
    for qid, val in metric_dict.items():
        dq_metrics[f"pearsons__{qid}"] = val
    values = list(v for k, v in metric_dict.items() if k not in exclude_qids)
    if values:
        dq_metrics["pearsons__minimum"] = float(
            np.nanmin(values) if len(values) > 0 and any(~np.isnan(values)) else np.nan
        )
    rilm_numericals, rilm_categoricals = compute_revised_information_loss_metric_future.result()
    to_process = [("categorical", rilm_categoricals)]
    if rnd_metrics:
        to_process.append(("numerical", rilm_numericals))
    for metric_which, metric_dict in to_process:
        for qid, val in metric_dict.items():
            dq_metrics[f"rilm__{qid}"] = val
        values = list(v for k, v in metric_dict.items() if k not in exclude_qids)
        if values:
            dq_metrics[f"rilm_{metric_which}__minimum"] = float(
                np.nanmin(values) if len(values) > 0 and any(~np.isnan(values)) else np.nan
            )
    metric_dict_v1, metric_dict_v2 = (
        compute_normalized_mutual_information_sampled_scaled_future.result()
    )
    to_process = [("nmi_sampled_scaled_v1", metric_dict_v1)]
    if rnd_metrics:
        to_process.append(("nmi_sampled_scaled_v2", metric_dict_v2))
    for metric, metric_dict in to_process:
        for qid, val in metric_dict.items():
            dq_metrics[f"{metric}__{qid}"] = val
        values = list(v for k, v in metric_dict.items() if k not in exclude_qids)
        if values:
            dq_metrics[f"{metric}__minimum"] = float(
                np.nanmin(values) if len(values) > 0 and any(~np.isnan(values)) else np.nan
            )
    if rnd_metrics:
        assert compute_normalized_mutual_information_sampled_unscaled_future is not None
        metric_dict_v1, metric_dict_v2 = (
            compute_normalized_mutual_information_sampled_unscaled_future.result()
        )
        to_process = [
            ("nmi_sampled_unscaled_v1", metric_dict_v1),
            ("nmi_sampled_unscaled_v2", metric_dict_v2),
        ]
        for metric, metric_dict in to_process:
            for qid, val in metric_dict.items():
                dq_metrics[f"{metric}__{qid}"] = val
            values = list(v for k, v in metric_dict.items() if k not in exclude_qids)
            if values:
                dq_metrics[f"{metric}__minimum"] = float(
                    np.nanmin(values) if len(values) > 0 and any(~np.isnan(values)) else np.nan
                )

    if len(anon_df) > 0:
        _, actual_k_val = calculate_p_k(anon_df, qids)
        actual_k = float(actual_k_val) if actual_k_val is not None else float("nan")
    else:
        actual_k = float("nan")
    disclosure_metrics["actual_k"] = actual_k

    # Redefine dq_metrics with proper float typing by casting all values. This is needed because
    # dq_metrics is built up with mixed types (Futures/InProcessResult that get .result() called,
    # plus plain floats), and mypy can't track that all values ultimately become floats.
    # TODO(Later) Refactor metric collection to maintain proper types throughout instead of casting at end
    dq_metrics = {k: cast(float, v) for k, v in dq_metrics.items()}

    return (dq_metrics, disclosure_metrics)
