"""
Normalized Mutual Information (NMI) functions for measuring data quality preservation.

This module implements NMI-based data quality metrics for Project Lighthouse anonymization.
NMI measures the preservation of minimal entropy under anonymization, providing a more
general measure than Pearson's correlation that doesn't assume linear relationships.

The module implements two versions:
1. NMIv1 = MI(X; Y) / H(X) - measures how much of the original data's information is preserved
2. NMIv2 = MI(X; Y) / H(Y) - measures how much of the anonymized data's information comes from original data

Functions:
- compute_normalized_mutual_information_sampled_scaled: Main NMI computation function
- nmi_unscaled_to_scaled: Scaling transformation for entropy adjustment
- Helper functions for data preprocessing and utility calculations
"""

import math
from typing import cast

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from project_lighthouse_anonymize.constants import (
    EPSILON,
    MAX_RANDOM_STATE,
    NOT_DEFINED_NA,
)


def compute_normalized_mutual_information_sampled_scaled(
    non_suppressed_records: pd.DataFrame,
    qids: list[str],
    join_suffix_orig: str,
    join_suffix_anon: str,
    sample_size: int = 10_000,
    number_runs: int = 10,
    scale: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute Normalized Mutual Information v1 (NMIv1) and v2 (NMIv2), sampled and scaled.

    The function computes two versions:
    1. NMIv1 = MI(X; Y) / H(X) - measures how much of the original data's information is preserved
    2. NMIv2 = MI(X; Y) / H(Y) - measures how much of the anonymized data's information comes from original data

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
    sample_size : int, default=10_000
        Maximum number of records to use when computing NMI. For larger datasets,
        random sampling is used to improve efficiency.
    number_runs : int, default=10
        Number of sampling runs to average when sample_size is less than the
        total number of records, to ensure consistent measurements.
    scale : bool, default=True
        Whether to scale NMI values based on input entropy. When True, the metric
        applies a decreasing penalty to each successive unit of entropy, useful
        when high-entropy inputs don't require perfect information preservation.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        A tuple of two dictionaries:
        1. Dictionary mapping QIDs to their NMIv1 values
        2. Dictionary mapping QIDs to their NMIv2 values

    Notes
    -----
    NMIv1 measures the preservation of minimal entropy under anonymization. In contrast to
    Pearson's correlation, NMIv1 provides a more general measure that doesn't assume a linear
    relationship between variables.

    NMIv1 is preferred for Project Lighthouse as it better measures the impact of entropy suppression,
    which is more critical for statistical analysis than entropy injection.

    When the number of records exceeds the sample_size, we sample the data multiple times to ensure
    consistent measurements across datasets of different sizes.

    Scale parameter adjusts the NMI value based on input entropy - higher entropy inputs have
    relaxed thresholds since not all original entropy is needed for analysis.

    The function attempts to determine whether each QID is discrete or continuous
    based on its data type, which affects the mutual information computation method.

    See Also
    --------
    nmi_unscaled_to_scaled : Function that applies the scaling transformation
    """
    qids = [
        qid
        for qid in qids
        if non_suppressed_records.dtypes[f"{qid}{join_suffix_orig}"] != np.dtype("object")
    ]
    if len(non_suppressed_records) == 0 or len(qids) == 0:
        return {qid: NOT_DEFINED_NA for qid in qids}, {qid: NOT_DEFINED_NA for qid in qids}
    orig_qids = [f"{qid}{join_suffix_orig}" for qid in qids]
    anon_qids = [f"{qid}{join_suffix_anon}" for qid in qids]
    mis_1, mis_2 = {}, {}
    for qid, orig_qid, anon_qid in zip(qids, orig_qids, anon_qids):
        # The next few lines are a very crude way to ascertain if a qid is discrete or continuous.
        # TODO(Later) require caller to specify discrete/continuous
        # _attempt_convert_to_discrete_dtype() is a hack: If a column has dtype float32 or float64, has nans, and the
        # non-nan unique value count is less than 1000, try casting it to Int64. As a result of this hack, discrete
        # features can now have dtype int32, int64, or Int64.
        # TODO(Later) remove hack once we support integer dtypes w/ nan
        non_suppressed_records = _attempt_convert_to_discrete_dtype(
            non_suppressed_records, orig_qid, anon_qid
        )
        x_dtype, y_dtype = (
            non_suppressed_records.dtypes[orig_qid],
            non_suppressed_records.dtypes[anon_qid],
        )
        continuous_dtypes = {np.dtype("float32"), np.dtype("float64")}
        x_discrete, y_discrete = (
            x_dtype not in continuous_dtypes,
            y_dtype not in continuous_dtypes,
        )
        # We only consider x, y discrete if both are not continuous
        discrete = x_dtype not in continuous_dtypes and y_dtype not in continuous_dtypes
        mutual_info_f = mutual_info_classif if discrete else mutual_info_regression
        x, y = (
            cast(pd.Series, non_suppressed_records[orig_qid]),
            cast(pd.Series, non_suppressed_records[anon_qid]),
        )
        # Drop entries where both x is NA/nan and y is NA/nan--this should only
        # happen when the original attribute value is NA/nan, and thus so is the
        # associated anonymized attribute value.
        x, y = _remove_entries_where_both_missing(x, y)
        # There should never be any rows where orig is nan and anon is not nan.
        assert any(x.isna() & ~y.isna()) is False, (
            "This should never happen: there should never be any rows where the orig cell is nan, but the anon cell is not nan"
        )
        # Replace locally suppressed anonymized attribute values. In other words, if orig is not NA/nan but anon is
        # NA/nan, replace anon with a numerical value.
        if len(y) > 0 and any(y.isna()) and not all(y.isna()):
            # We know no more than what is present in anonymized records
            # so presume the missing records are just the average of those present
            y[y.isna()] = y.mean(skipna=True)
        # All NA/nan values are gone by this point, so convert from Int64 to int64 so that the rest of this function can
        # run successfully.
        if x.dtype == "Int64" or y.dtype == "Int64":
            x = x.astype("int64")
            y = y.astype("int64")
        # Compute random seed based only on x so that it is deterministic
        rng = __rng_from_numpy_array(x.to_numpy())
        # Pre-allocate arrays for efficient use below
        x, y = x.to_numpy().ravel(), y.to_numpy().ravel()
        x_idxs = list(range(len(x)))
        # We run multiple times to smooth out variance in mi computations,
        # only run once if leq sample size.
        if len(x_idxs) <= sample_size:
            number_runs = 1
            sample_size = len(x_idxs)
        mi_samples_1: list[float] = [0.0] * number_runs
        mi_samples_2: list[float] = [0.0] * number_runs
        for i in range(number_runs):
            idx_samples = rng.choice(x_idxs, sample_size, replace=False)
            tx, ty = x[idx_samples], y[idx_samples]
            random_state = rng.integers(MAX_RANDOM_STATE)
            x_elements_all_same = _elements_all_same(tx, x_discrete)
            y_elements_all_same = _elements_all_same(ty, y_discrete)
            if x_elements_all_same or y_elements_all_same:
                mi_x_y = 0.0
            else:
                try:
                    mi_x_y = mutual_info_f(
                        X=tx.reshape(len(tx), 1),
                        y=ty.reshape(
                            len(ty),
                        ),
                        discrete_features=discrete,  # type: ignore[arg-type]  # sklearn accepts bool but pyright stubs incorrectly say str only
                        random_state=random_state,
                    )[0]
                except ValueError:
                    # may occur if e.g. n_samples < n_neighbors
                    mi_x_y = NOT_DEFINED_NA
            if y_elements_all_same:
                mi_y_y = 0.0
            else:
                try:
                    mi_y_y = mutual_info_f(
                        X=ty.reshape(len(ty), 1),
                        y=ty.reshape(
                            len(ty),
                        ),
                        discrete_features=discrete,  # type: ignore[arg-type]  # sklearn accepts bool but pyright stubs incorrectly say str only
                        random_state=random_state,
                    )[0]
                except ValueError:
                    # may occur if e.g. n_samples < n_neighbors
                    mi_y_y = NOT_DEFINED_NA
            if x_elements_all_same:
                mi_x_x = 0.0
            else:
                try:
                    mi_x_x = mutual_info_f(
                        X=tx.reshape(len(tx), 1),
                        y=tx.reshape(
                            len(tx),
                        ),
                        discrete_features=discrete,  # type: ignore[arg-type]  # sklearn accepts bool but pyright stubs incorrectly say str only
                        random_state=random_state,
                    )[0]
                except ValueError:
                    # may occur if e.g. n_samples < n_neighbors
                    mi_x_x = NOT_DEFINED_NA
            input_entropy = mi_x_x
            for num, den, arr in zip(
                [mi_x_y, mi_x_y], [mi_x_x, mi_y_y], [mi_samples_1, mi_samples_2]
            ):
                if np.isclose(den, 0.0, rtol=1e-5, atol=EPSILON):
                    if np.isclose(num, 0.0, rtol=1e-5, atol=EPSILON):
                        # special case where if there is no information to encode, then we have achieved perfect data quality w.r.t. mutual information
                        arr[i] = 1.0
                    else:
                        # if there is no information to encode, but information has been encoded, then something is very odd so return nan
                        arr[i] = NOT_DEFINED_NA
                else:
                    # Scale computed NMI score
                    unscaled_value = min(max(0, num / den), 1.0)
                    arr[i] = (
                        nmi_unscaled_to_scaled(unscaled_value, input_entropy)
                        if scale
                        else unscaled_value
                    )
        mis_1[qid] = (
            float(np.nanmean(mi_samples_1)) if not np.isnan(mi_samples_1).all() else NOT_DEFINED_NA
        )
        mis_2[qid] = (
            float(np.nanmean(mi_samples_2)) if not np.isnan(mi_samples_2).all() else NOT_DEFINED_NA
        )
    return mis_1, mis_2


def nmi_unscaled_to_scaled(unscaled_value: float, input_entropy: float) -> float:
    """
    Convert from unscaled to scaled NMI by applying an exponential penalty.

    As the input entropy increases, less of the total entropy is necessary for successful analysis.
    This scaling function applies a decreasing penalty to each successive unit of entropy (nat),
    where the first nat has penalty of (1-n), the second nat has (1/2)(1-n), the third nat has
    (1/4)(1-n), and so on.

    Parameters
    ----------
    unscaled_value : float
        Raw NMI value before scaling, typically between 0 and 1.
    input_entropy : float
        Original entropy H(X) of the data, measured in nats.

    Returns
    -------
    float
        Scaled NMI value where higher input entropy has a relaxed threshold.
        The scaling ensures that data with higher entropy doesn't need to preserve
        as much of its original information to achieve a high score.

    Notes
    -----
    This implements the formula from the Project Lighthouse technical paper:
    NMIv1(O, A, i) = 1 - ∫(0->e)(1-n)2^(-x)dx

    Where:
    - n is the unscaled NMI value
    - e is the input entropy
    - x is the integration variable

    Edge cases handled:
    - If unscaled_value is effectively zero: Returns zero
    - If input_entropy is effectively zero: Returns NaN (undefined)

    The function applies a decreasing penalty to each successive unit of entropy,
    reflecting the principle that high-entropy data can tolerate more information
    loss while still remaining useful for analysis.

    See Also
    --------
    __scaled_indefinite_integral : Helper function for the integration in the formula
    compute_normalized_mutual_information_sampled_scaled : Function that uses this scaling
    """
    if np.isclose(unscaled_value, 0.0, rtol=1e-5, atol=EPSILON):
        # edge case, don't produce a non-zero NMI score if the unscaled score is zero!
        scaled_value = 0.0
    elif np.isclose(input_entropy, 0.0, rtol=1e-5, atol=EPSILON):
        # edge case: no input entropy, prevent divide by zero warning
        scaled_value = np.nan
    else:
        area = __scaled_indefinite_integral(
            unscaled_value, input_entropy
        ) - __scaled_indefinite_integral(unscaled_value, 0.0)
        scaled_value = 1.0 - ((1.0 / input_entropy) * area)
    return scaled_value


def __scaled_indefinite_integral(n: float, x: float) -> float:
    """
    Calculate the scaled indefinite integral for NMI scaling.

    Parameters
    ----------
    n : float
        The unscaled NMI value.
    x : float
        The point at which to evaluate the integral.

    Returns
    -------
    float
        The value of the indefinite integral at point x.

    Notes
    -----
    This is a helper function for the nmi_unscaled_to_scaled function.
    It implements the indefinite integral for the formula:
    (1-n)2^(-x)dx

    This represents the integral of the exponentially decreasing penalty
    function used in the NMI scaling calculation.
    """
    return ((n - 1.0) * math.pow(2, -x)) / (math.log(2.0, math.e))


def _elements_all_same(arr: np.ndarray, discrete: bool) -> bool:
    """
    Determine whether all elements in an array are identical.

    Parameters
    ----------
    arr : numpy.ndarray
        1-D numpy array of numerical values to check.
    discrete : bool
        If True, checks exact equality for discrete numerical values.
        If False, uses np.isclose() for continuous numerical values.

    Returns
    -------
    bool
        True if all elements in arr are identical, False otherwise.

    Notes
    -----
    For discrete values, the function uses exact comparison (==).
    For continuous values, the function uses np.isclose() to account for
    floating-point precision issues.
    """
    if discrete:
        return bool(np.all(arr == arr[0]))
    else:
        return bool(np.all(np.isclose(arr, arr[0])))


def _attempt_convert_to_discrete_dtype(
    non_suppressed_records: pd.DataFrame, orig_qid: str, anon_qid: str
) -> pd.DataFrame:
    """
    Convert floating-point QIDs to integer types with NA support when appropriate.

    Parameters
    ----------
    non_suppressed_records : pd.DataFrame
        DataFrame containing QID columns to check for conversion.
    orig_qid : str
        Column name of the original QID values.
    anon_qid : str
        Column name of the anonymized QID values.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with potentially modified dtypes for the specified QID columns.

    Notes
    -----
    This helper function tries to infer when a QID that contains NaN values was
    originally integral (i.e., originally had a dtype of int64 or int32). In those cases,
    it changes the QID's dtype to Int64 (with a capital "I"), which can contain
    pd.NA values as described in the pandas documentation:
    https://pandas.pydata.org/docs/user_guide/integer_na.html#integer-na

    The conversion is attempted when:
    1. The QID has float32 or float64 dtype
    2. The QID contains NaN values
    3. The QID has fewer than 1,000 unique non-NaN values

    If the data cannot be cast to Int64 (e.g., contains true floating-point values),
    the original data is returned unchanged.
    """
    x_dtype, y_dtype = (
        non_suppressed_records.dtypes[orig_qid],
        non_suppressed_records.dtypes[anon_qid],
    )
    x, y = non_suppressed_records[orig_qid], non_suppressed_records[anon_qid]
    continuous_dtypes = {np.dtype("float32"), np.dtype("float64")}
    if (
        (x_dtype in continuous_dtypes or y_dtype in continuous_dtypes)
        and (x.hasnans or y.hasnans)
        and (pd.concat([x, y]).nunique(dropna=True) < 1_000)
    ):
        new_dtype = "Int64"
        # If data cannot be cast to Int64, then the original data is returned.
        non_suppressed_records[orig_qid] = non_suppressed_records[orig_qid].astype(
            new_dtype, errors="ignore"
        )
        non_suppressed_records[anon_qid] = non_suppressed_records[anon_qid].astype(
            new_dtype, errors="ignore"
        )
    return non_suppressed_records


def _remove_entries_where_both_missing(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Remove entries where both original and anonymized values are missing.

    Parameters
    ----------
    x : pd.Series
        Series containing original attribute values.
    y : pd.Series
        Series containing anonymized attribute values.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        A tuple containing the filtered x and y series with rows removed where both values
        were NA/NaN, and with index reset on both series.

    Notes
    -----
    This helper function drops entries where both x (original) and y (anonymized) values
    are NA/NaN. This should only happen when the original attribute value is NA/NaN,
    and thus the associated anonymized attribute value is also NA/NaN.

    The function is used to clean data before computing metrics like correlation or
    mutual information, which cannot handle missing values appropriately.
    """
    x_and_y_nan = x.isna() & y.isna()
    x, y = (
        cast(pd.Series, x[~x_and_y_nan]).reset_index(drop=True),
        cast(pd.Series, y[~x_and_y_nan]).reset_index(drop=True),
    )
    return x, y


def __rng_from_numpy_array(x: np.ndarray) -> np.random.Generator:
    """
    Construct a deterministic random generator based on an input numpy array.

    Parameters
    ----------
    x : numpy.ndarray
        Input array to use for generating the seed value.

    Returns
    -------
    numpy.random.Generator
        A seeded random number generator.

    Notes
    -----
    This function creates a reproducible random number generator by:
    1. Computing a hash of the sorted array values
    2. Taking the modulo of the hash with MAX_RANDOM_STATE
    3. Using this value as a seed for numpy's default_rng

    This ensures that identical input arrays will always produce the same
    sequence of random numbers, which is important for consistency in
    sampling operations used by metrics computation.
    """
    seed = hash(tuple(sorted(x))) % MAX_RANDOM_STATE
    return np.random.default_rng(seed)
