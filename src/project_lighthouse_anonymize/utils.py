"""
Shared utility functions for anonymization and data processing.

This module provides utility functions that support the core anonymization
functionality in Project Lighthouse. It includes functions for:

- Statistical calculations (standard deviation, min/max, etc.)
- Record manipulation (generalization, suppression)
- Parallel processing helpers
- Data partitioning and preprocessing

These utilities are designed to work with pandas DataFrames and numpy arrays
to facilitate efficient data processing while handling special cases like
NaN values and categorical data.
"""

import logging
import logging.handlers
import time
from typing import Any, Generator, cast
from uuid import uuid4

import numba
import numpy as np
import pandas as pd

from project_lighthouse_anonymize.gtrees import ReadOnlyGTree


class AnonymizeCallbacks:
    """
    Callback mechanism for tracking and instrumentation of anonymization operations.

    This class provides a simple way to track timing and execution flow during the
    anonymization process. It enables performance measurement and debugging by
    recording timestamps at key points in the process.

    Users can extend this class to add custom tracking or instrumentation by
    overriding the callback methods.

    Attributes
    ----------
    timestamps : dict
        Dictionary storing timestamps for different stages of the anonymization process.
        Keys include 'anonymize_bm', 'anonymize_am', 'compute_metrics_bm', and
        'compute_metrics_am', where 'bm' stands for "before method" and 'am' for
        "after method".

    Notes
    -----
    This is primarily used by the anonymization functions in this package to
    provide timing information. Custom callbacks can be used for logging,
    monitoring, or gathering additional statistics.

    Examples
    --------
    >>> class CustomCallbacks(AnonymizeCallbacks):
    ...     def anonymize_bm(self):
    ...         super().anonymize_bm()
    ...         print("Starting anonymization...")
    ...
    ...     def anonymize_am(self):
    ...         super().anonymize_am()
    ...         duration = self.timestamps["anonymize_am"] - self.timestamps["anonymize_bm"]
    ...         print(f"Anonymization completed in {duration:.2f} seconds")
    """

    def __init__(self) -> None:
        self.timestamps: dict[str, float] = {}

    def anonymize_bm(self) -> None:
        self.timestamps["anonymize_bm"] = time.time()

    def anonymize_am(self) -> None:
        self.timestamps["anonymize_am"] = time.time()

    def compute_metrics_bm(self) -> None:
        self.timestamps["compute_metrics_bm"] = time.time()

    def compute_metrics_am(self) -> None:
        self.timestamps["compute_metrics_am"] = time.time()


class BlockingQueueHandler(logging.handlers.QueueHandler):
    """
    Custom QueueHandler that uses blocking put operations to prevent queue.Full exceptions.

    This handler extends the standard logging.handlers.QueueHandler but uses
    put(block=True, timeout=None) instead of put_nowait() to ensure log records
    are always delivered to the queue, even under high log volume.
    """

    def enqueue(self, record: logging.LogRecord) -> None:
        """
        Enqueue a record using blocking put to prevent queue.Full exceptions.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to enqueue.
        """
        self.queue.put(record, block=True, timeout=None)  # type: ignore[attr-defined]


@numba.jit(nopython=True)
def max_minus_min(x: np.ndarray) -> float:
    """
    Calculate the range (max-min) of an array, ignoring NaN values.

    This function computes the difference between the maximum and minimum values
    in an array by iterating over it only once, making it more efficient than
    separate max and min operations.

    Parameters
    ----------
    x : np.ndarray
        Input array of numerical values.

    Returns
    -------
    float
        The difference between maximum and minimum values (x.max() - x.min()).
        Returns np.nan if the array is empty or contains only NaN values.

    Notes
    -----
    Implementation based on https://stackoverflow.com/a/33919126 (CC BY-SA 3.0)

    Examples
    --------
    >>> max_minus_min(np.array([1, 2, 3, 4, 5]))
    4.0
    >>> max_minus_min(np.array([1, 2, np.nan, 4, 5]))
    4.0
    >>> max_minus_min(np.array([np.nan, np.nan]))
    nan
    """
    minimum, maximum = min_max(x)
    return float(maximum - minimum)


@numba.jit(nopython=True)
def min_max(x: np.ndarray) -> tuple[float, float]:
    """
    Find the minimum and maximum values of an array, ignoring NaN values.

    This optimized function computes both minimum and maximum values in a single pass
    through the array, which is more efficient than separate calculations.

    Parameters
    ----------
    x : np.ndarray
        Input array of numerical values.

    Returns
    -------
    Tuple[float, float]
        A tuple containing (minimum, maximum) values from the array.
        Returns (np.nan, np.nan) if the array is empty or contains only NaN values.

    Examples
    --------
    >>> min_max(np.array([1, 2, 3, 4, 5]))
    (1.0, 5.0)
    >>> min_max(np.array([1, 2, np.nan, 4, 5]))
    (1.0, 5.0)
    >>> min_max(np.array([np.nan, np.nan]))
    (nan, nan)
    """
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)


@numba.jit(nopython=True)
def median_max_second_max(x: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate the median, maximum, and second maximum values of an array.

    This function computes three statistical measures in a single operation:
    - The median value
    - The maximum value
    - The second largest value (if any)

    All calculations ignore NaN values.

    Parameters
    ----------
    x : np.ndarray
        Input array of numerical values.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing (median, maximum, second_maximum) values.

        Special cases:

        - If array has no non-NaN values, returns (np.nan, np.nan, np.nan)
        - If array has only one unique non-NaN value t, returns (t, t, np.nan)

    Examples
    --------
    >>> median_max_second_max(np.array([1, 2, 3, 4, 5]))
    (3.0, 5.0, 4.0)
    >>> median_max_second_max(np.array([1, 1, 1]))
    (1.0, 1.0, nan)
    >>> median_max_second_max(np.array([np.nan, np.nan]))
    (nan, nan, nan)
    """
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    # reversed sort
    x = x[np.argsort(-x)]
    n = len(x)
    if n == 1:
        return (x[0], x[0], np.nan)
    median = x[n // 2] if n % 2 == 1 else ((x[n // 2 - 1] + x[n // 2]) / 2)
    maximum = x[0]
    for i in x[1:]:
        if i != maximum:
            second_maximum = i
            return (median, maximum, second_maximum)
    second_maximum = np.nan
    return (median, maximum, second_maximum)


@numba.jit(nopython=True)
def atleast_nunique(x: np.ndarray, n: int) -> bool:
    """
    Check if an array contains at least n unique elements.

    This function efficiently determines whether an array has at least
    the specified number of unique values, without computing the full set
    of unique values. It ignores NaN values in the calculation.

    Parameters
    ----------
    x : np.ndarray
        Input array to check for unique elements.
    n : int
        Threshold number of unique elements to check for.

    Returns
    -------
    bool
        True if the array contains at least n unique elements, False otherwise.

    Notes
    -----
    This implementation is optimized to stop counting once n unique elements
    are found, which is more efficient than computing all unique values when
    only checking against a threshold.

    Examples
    --------
    >>> atleast_nunique(np.array([1, 2, 3, 4, 5]), 3)
    True
    >>> atleast_nunique(np.array([1, 1, 2, 2, np.nan]), 3)
    False
    """
    x = x[~np.isnan(x)]
    m = 0
    seen = set()
    for i in x:
        if i not in seen:
            m += 1
            seen.add(i)
            if m >= n:
                return True
    return False


def atleast_nunique_categorical(x: pd.Series, n: int) -> bool:
    """
    Check if a pandas Series contains at least n unique elements.

    This function is similar to atleast_nunique but works specifically with pandas Series
    objects, particularly for categorical data. It efficiently determines whether a Series
    has at least the specified number of unique values, without computing the full set
    of unique values. It ignores NaN values in the calculation.

    Parameters
    ----------
    x : pd.Series
        Input Series to check for unique elements.
    n : int
        Threshold number of unique elements to check for.

    Returns
    -------
    bool
        True if the Series contains at least n unique elements, False otherwise.

    Notes
    -----
    This implementation is optimized to stop counting once n unique elements
    are found, which can be more efficient than calling pd.Series.nunique()
    when only checking against a threshold.

    See Also
    --------
    atleast_nunique : Similar function for NumPy arrays

    Examples
    --------
    >>> atleast_nunique_categorical(pd.Series(['a', 'b', 'c', 'd']), 3)
    True
    >>> atleast_nunique_categorical(pd.Series(['a', 'a', 'b', None]), 3)
    False
    """
    x = cast(pd.Series, x[~pd.isna(x)])
    m = 0
    seen = set()
    for i in x:
        if i not in seen:
            m += 1
            seen.add(i)
            if m >= n:
                return True
    return False


@numba.jit(nopython=True)
def __standard_deviation_and_len(
    x: np.ndarray,
) -> tuple[float, int]:
    """
    Compute the standard deviation and length of an array, ignoring NaN values.

    This helper function calculates both the standard deviation and valid length
    (count of non-NaN elements) of an array.

    Parameters
    ----------
    x : np.ndarray
        Input array of numerical values.

    Returns
    -------
    Tuple[float, int]
        A tuple containing (standard_deviation, length) where:
        - standard_deviation: The standard deviation of non-NaN values
        - length: The count of non-NaN values

        If there are no non-NaN values in the array, returns (np.nan, 0).

    Notes
    -----
    This is a private helper function used by standard_deviation and related functions.
    The implementation first filters out NaN values and then calculates the standard deviation.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    return (np.std(x), n) if n > 0 else (np.nan, 0)  # type: ignore[reportReturnType]  # numba-compiled function, actual return type matches


def standard_deviation(
    *values_arr: np.ndarray,
) -> float:
    """
    Compute the weighted average standard deviation across multiple arrays.

    This function calculates the standard deviation for each provided array,
    ignoring NaN values, then returns the average standard deviation weighted
    by the number of non-NaN elements in each array.

    Parameters
    ----------
    *values_arr : np.ndarray
        One or more input arrays of numerical values.

    Returns
    -------
    float
        The weighted average standard deviation across all input arrays.
        Returns np.nan if all input arrays are empty or contain only NaN values.

    Notes
    -----
    This function delegates to specialized implementations based on the number
    of input arrays:
    - For fewer than 1000 arrays, uses __standard_deviation_small with Numba
    - For 1000+ arrays, uses __standard_deviation_large without Numba

    See Also
    --------
    standard_deviation_categorical : Similar function for categorical data

    Examples
    --------
    >>> standard_deviation(np.array([1, 2, 3]), np.array([4, 5, 6]))
    0.8164965809277259
    """
    if len(values_arr) < 1_000:
        return float(__standard_deviation_small(*values_arr))
    return float(__standard_deviation_large(*values_arr))


@numba.jit(nopython=True)
def __standard_deviation_small(
    *values_arr: np.ndarray,
) -> float:
    """
    Calculate weighted average standard deviation for a small number of arrays.

    This is an optimized implementation of standard_deviation for fewer than 1000
    input arrays that uses Numba for performance. It computes the weighted average
    of standard deviations where weights are the number of non-NaN elements in each array.

    Parameters
    ----------
    *values_arr : np.ndarray
        One or more input arrays of numerical values.

    Returns
    -------
    float
        The weighted average standard deviation across all input arrays.
        Returns np.nan if all input arrays are empty or contain only NaN values.

    Notes
    -----
    This is a private helper function used by the standard_deviation function.
    It uses Numba acceleration for improved performance.
    """
    num, den = 0.0, 0.0
    for values in values_arr:
        distance, length = __standard_deviation_and_len(values)
        if length > 0:
            num += distance * length
            den += length
    return (num / den) if den != 0 else np.nan


def __standard_deviation_large(
    *values_arr: np.ndarray,
) -> float:
    """
    Calculate weighted average standard deviation for a large number of arrays.

    This is an implementation of standard_deviation for 1000 or more input arrays
    that does not use Numba. It computes the weighted average of standard deviations
    where weights are the number of non-NaN elements in each array.

    Parameters
    ----------
    *values_arr : np.ndarray
        One or more input arrays of numerical values.

    Returns
    -------
    float
        The weighted average standard deviation across all input arrays.
        Returns np.nan if all input arrays are empty or contain only NaN values.

    Notes
    -----
    This is a private helper function used by the standard_deviation function.
    It does not use Numba acceleration due to compilation performance issues
    with large numbers of arrays.
    """
    num, den = 0.0, 0.0
    for values in values_arr:
        distance, length = __standard_deviation_and_len(values)
        if length > 0:
            num += distance * length
            den += length
    return (num / den) if den != 0 else np.nan


def standard_deviation_categorical(
    *values_arr: pd.Series,
) -> float:
    """
    Compute standard deviation across multiple series of categorical values.

    This function adapts the standard deviation concept to categorical data by
    transforming categorical values into numerical ones based on their frequency:
    - Values matching the mode (most common element) in each series are assigned 1.0
    - All other values are assigned 0.0

    After this transformation, the standard_deviation function is used to compute
    the weighted average standard deviation.

    Parameters
    ----------
    *values_arr : pd.Series
        One or more pandas Series containing categorical values.

    Returns
    -------
    float
        The weighted average standard deviation across all input series after
        categorical-to-numerical transformation.
        Returns np.nan if all input series are empty or contain only NaN values.

    Notes
    -----
    This approach allows measuring dispersion in categorical data by focusing on
    the concentration of values in the mode versus other categories.

    See Also
    --------
    standard_deviation : Standard deviation for numerical data

    Examples
    --------
    >>> standard_deviation_categorical(
    ...     pd.Series(['a', 'a', 'b']),
    ...     pd.Series(['c', 'd', 'd'])
    ... )
    0.4714045207910316
    """
    values_arr_2 = [cast(pd.Series, values[~pd.isna(values)]) for values in values_arr]
    mode_values = [
        values.value_counts().idxmax() if not values.empty else None for values in values_arr_2
    ]
    values_10_arr = [
        np.array([1.0 if x == mode_value else 0.0 for x in values])
        for values, mode_value in zip(values_arr_2, mode_values)
        if not values.empty
    ]
    if len(values_10_arr) == 0:
        return np.nan
    if len(values_10_arr) == 1:
        return standard_deviation(values_10_arr[0])
    return standard_deviation(*values_10_arr)


def get_non_suppressed_records(
    df_orig: pd.DataFrame,
    df_anon: pd.DataFrame,
    id_col: str,
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> pd.DataFrame:
    """
    Return the records not suppressed (generalized or unchanged) by anonymization.

    This function identifies records that were preserved (not suppressed) during
    the anonymization process by performing an inner join between the original and
    anonymized dataframes. The result includes both the original and anonymized
    values for each record.

    Parameters
    ----------
    df_orig : pd.DataFrame
        Original dataframe before anonymization.
    df_anon : pd.DataFrame
        Anonymized dataframe, potentially with some records suppressed.
    id_col : str
        Name of the unique identifier column present in both dataframes.
    join_suffix_orig : str
        Suffix to append to column names from the original dataframe.
    join_suffix_anon : str
        Suffix to append to column names from the anonymized dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all non-suppressed records with columns from both
        original and anonymized dataframes. Column names are suffixed with
        join_suffix_orig and join_suffix_anon respectively.

    Notes
    -----
    If a record is present in df_orig but not in df_anon, it is considered
    suppressed and will not appear in the output.

    See Also
    --------
    get_suppressed_records : Complementary function that returns only suppressed records
    get_generalized_records : Function to get only records that were generalized
    """
    assert not pd.isna(df_orig[id_col]).any(), "id_col must not contain NaN values"  # type: ignore[reportGeneralTypeIssues]  # .any() returns bool, not Series
    df_anon_comparison = (
        df_orig.set_index(id_col)
        .join(
            df_anon.set_index(id_col),
            how="inner",
            lsuffix=join_suffix_orig,
            rsuffix=join_suffix_anon,
        )
        .reset_index()
    )
    return df_anon_comparison


def get_generalized_records(
    df_orig: pd.DataFrame,
    df_anon: pd.DataFrame,
    id_col: str,
    non_qid_cols: set[str],
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> pd.DataFrame:
    """
    Return only the records that were generalized during anonymization.

    This function identifies records where the values of quasi-identifiers (QIDs) were
    modified but not suppressed during anonymization. These are records where at least
    one QID value in the anonymized dataframe differs from its original value.

    Parameters
    ----------
    df_orig : pd.DataFrame
        Original dataframe before anonymization.
    df_anon : pd.DataFrame
        Anonymized dataframe.
    id_col : str
        Name of the unique identifier column present in both dataframes.
    non_qid_cols : set
        Set of column names that are NOT quasi-identifiers and were passed
        through unchanged by the anonymization process.
    join_suffix_orig : str
        Suffix to append to column names from the original dataframe.
    join_suffix_anon : str
        Suffix to append to column names from the anonymized dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe containing only records that were generalized, with columns from
        both original and anonymized dataframes. Column names are suffixed with
        join_suffix_orig and join_suffix_anon respectively.

    Notes
    -----
    - QID columns are determined as all columns except those in non_qid_cols
    - A record is considered generalized if any of its QID values differ between
      df_orig and df_anon
    - If there are no QID columns, an empty dataframe is returned as no generalization
      could have occurred

    See Also
    --------
    get_non_suppressed_records : Function to get all non-suppressed records
    get_suppressed_records : Function to get only suppressed records
    """
    assert not pd.isna(df_orig[id_col]).any(), "id_col must not contain NaN values"  # type: ignore[reportGeneralTypeIssues]  # .any() returns bool, not Series
    qid_cols = set(df_orig.columns) - non_qid_cols
    df_anon_comparison = (
        df_orig.set_index(id_col)
        .join(
            df_anon.set_index(id_col),
            how="inner",
            lsuffix=join_suffix_orig,
            rsuffix=join_suffix_anon,
        )
        .reset_index()
    )
    if len(df_anon_comparison) == 0:
        return df_anon_comparison
    if len(qid_cols) == 0:
        # no qids means there was no generalization (there may have been suppression), so no rows should be returned
        return df_anon_comparison.sample(0)
    or_idxer_1 = np.array([False] * len(df_anon_comparison))
    for qid in qid_cols:
        orig_col = df_anon_comparison[qid + join_suffix_orig]
        anon_col = df_anon_comparison[qid + join_suffix_anon]

        # Handle NaN values properly: use pd.isna() to identify NaN values
        # Only flag as different if values actually differ (not both NaN)
        both_na = pd.isna(orig_col) & pd.isna(anon_col)
        or_idxer_2 = (orig_col != anon_col) & ~both_na

        or_idxer_1 = np.logical_or(or_idxer_1, or_idxer_2)
    return cast(pd.DataFrame, df_anon_comparison[or_idxer_1])


def get_suppressed_records(
    df_orig: pd.DataFrame,
    df_anon: pd.DataFrame,
    id_col: str,
    non_qid_cols: set[str],
    join_suffix_orig: str,
    join_suffix_anon: str,
) -> pd.DataFrame:
    """
    Return only the records that were suppressed during anonymization.

    This function identifies records that were completely removed (suppressed) during
    anonymization by finding records present in the original dataframe but missing
    from the anonymized dataframe.

    Parameters
    ----------
    df_orig : pd.DataFrame
        Original dataframe before anonymization.
    df_anon : pd.DataFrame
        Anonymized dataframe.
    id_col : str
        Name of the unique identifier column present in both dataframes.
    non_qid_cols : set
        Set of column names that are NOT quasi-identifiers.
        (This parameter is included for API consistency but not used in this function).
    join_suffix_orig : str
        Suffix to append to column names from the original dataframe.
    join_suffix_anon : str
        Suffix to append to column names from the anonymized dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe containing only suppressed records with columns from both original
        and anonymized dataframes. Column names are suffixed with join_suffix_orig
        and join_suffix_anon respectively. The anonymized columns (with join_suffix_anon)
        will contain NaN values since these records were suppressed.

    Notes
    -----
    Suppression is identified by comparing the set of record IDs in the original
    and anonymized dataframes. Records with IDs present in df_orig but not in df_anon
    are considered suppressed.

    See Also
    --------
    get_non_suppressed_records : Function to get all non-suppressed records
    get_generalized_records : Function to get only generalized records
    """
    assert not pd.isna(df_orig[id_col]).any(), "id_col must not contain NaN values"  # type: ignore[reportGeneralTypeIssues]  # .any() returns bool, not Series
    orig_ids = set(df_orig[id_col])
    anon_ids = set(df_anon[id_col])
    suppressed_ids = orig_ids - anon_ids
    df_anon_comparison = (
        df_orig.set_index(id_col)
        .join(
            df_anon.set_index(id_col),
            how="left",
            lsuffix=join_suffix_orig,
            rsuffix=join_suffix_anon,
        )
        .reset_index()
    )
    return cast(
        pd.DataFrame, df_anon_comparison[df_anon_comparison[id_col].isin(list(suppressed_ids))]
    )


def parallelism__setup_logging(
    ctx: Any,
) -> tuple[Any, logging.handlers.QueueListener, list[logging.Handler], int]:
    """
    Set up logging queue and listener for multiprocess logging.

    Creates a logging queue and QueueListener following the Python logging cookbook
    pattern for multiprocess logging. Configures the parent process to also use the
    queue for consistent logging across all processes.

    Parameters
    ----------
    ctx : multiprocessing.BaseContext
        Multiprocessing context (typically from mp.get_context("spawn"))

    Returns
    -------
    tuple
        A tuple containing (logging_queue, logging_listener, original_handlers, original_logging_level) where:
        - logging_queue: Queue for sending log records between processes
        - logging_listener: QueueListener instance (already started)
        - original_handlers: List of original logging handlers to restore later
        - original_logging_level: Original logging level before modification

    Notes
    -----
    The caller is responsible for:
    1. Stopping the listener in a finally block
    2. Restoring original handlers after stopping the listener
    3. Passing the logging_queue to worker processes
    """
    logging_queue = ctx.Queue(-1)

    # Save original handlers and logging level before setting up QueueListener
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Add a basic handler if none exists
        root_logger.addHandler(logging.StreamHandler())
    original_handlers = root_logger.handlers[:]
    original_logging_level = root_logger.getEffectiveLevel()

    # Set up QueueListener to handle log records from all processes
    logging_listener = logging.handlers.QueueListener(logging_queue, *original_handlers)
    logging_listener.start()

    # Configure parent process to also use the queue
    root_logger.handlers.clear()
    root_logger.addHandler(BlockingQueueHandler(logging_queue))

    # Reduce numba logging noise
    logging.getLogger("numba").setLevel(logging.ERROR)

    return logging_queue, logging_listener, original_handlers, original_logging_level


def parallelism__cleanup_logging(
    logging_listener: logging.handlers.QueueListener,
    original_handlers: list[logging.Handler],
    original_logging_level: int,
    logging_queue: Any,
) -> None:
    """
    Clean up logging configuration after multiprocess execution.

    Stops the QueueListener, properly closes the logging queue, and restores
    the original logging handlers and level.

    Parameters
    ----------
    logging_listener : logging.handlers.QueueListener
        The QueueListener to stop
    original_handlers : list
        List of original logging handlers to restore
    original_logging_level : int
        Original logging level to restore
    logging_queue : multiprocessing.Queue
        The logging queue to close and join
    """
    logging_listener.stop()
    logging_queue.close()
    logging_queue.join_thread()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(original_logging_level)
    for handler in original_handlers:
        root_logger.addHandler(handler)


def parallelism__make_initializer_and_initargs(
    qid_to_gtree: dict[str, Any], logging_queue: Any, logging_level: int
) -> tuple[Any, tuple[str, dict[str, Any], Any, int], tuple[str]]:
    """
    Create initialization components for efficient parallel processing with generalization trees.

    This function prepares the necessary components to efficiently share generalization trees
    (gtrees) between processes during parallel execution. It creates a unique global variable
    name for storing the gtrees and returns the initializer function and arguments needed for
    a ProcessPoolExecutor.

    Parameters
    ----------
    qid_to_gtree : dict
        Dictionary mapping quasi-identifier column names to their associated
        generalization trees (GTree objects).
    logging_queue : multiprocessing.Queue
        Queue for sending log records from all processes to the QueueListener.
    logging_level : int
        Logging level to set for all processes (obtained from parent before handler modification).

    Returns
    -------
    tuple
        A tuple containing (initializer, initargs, lookup_args) where:

        - initializer: Function to initialize the subprocess with gtrees and logging
        - initargs: Arguments to pass to the initializer function
        - lookup_args: Arguments to pass to parallelism__get_qid_to_gtree() in
                      the subprocess to retrieve the gtrees

    Notes
    -----
    This parallelism mechanism works by:

    1. Converting all gtrees to read-only versions for thread safety
    2. Generating a unique global variable name to avoid collisions
    3. Setting up the gtrees in the global namespace of each subprocess
    4. Optionally configuring logging in worker processes via QueueHandler
    5. Providing lookup arguments to retrieve the gtrees in the subprocess

    """
    # make gtrees read-only
    qid_to_gtree = {
        qid: ReadOnlyGTree(gtree) if not isinstance(gtree, ReadOnlyGTree) else gtree
        for qid, gtree in qid_to_gtree.items()
    }
    # construct a unique variable name to use in globals
    var_name = None
    while var_name is None or var_name in globals():
        var_name = f"PARALLELISM__GTREES__{str(uuid4()).upper().replace('-', '_')}"
    assert var_name not in globals()

    # the initialize takes as its arguments the global variable name, gtrees, and logging config
    init_args = (var_name, qid_to_gtree, logging_queue, logging_level)
    # call the initializer so that calls from within this process will work, ie. when not parallelizing
    parallelism__initializer(*init_args)
    # the lookup function will take the global variable name as its only argument
    lookup_args = (var_name,)
    return parallelism__initializer, init_args, lookup_args


def parallelism__initializer(
    var_name: str, qid_to_gtree: dict[str, Any], logging_queue: Any, logging_level: int
) -> None:
    """
    Initialize a subprocess with generalization trees and logging configuration.

    This function is called by ProcessPoolExecutor to set up global variables
    in each subprocess before executing tasks. It stores the generalization trees
    (gtrees) in the global namespace using the provided variable name and
    optionally configures logging to send records to the parent process.

    Parameters
    ----------
    var_name : str
        The unique variable name to use in the global namespace.
    qid_to_gtree : dict
        Dictionary mapping quasi-identifier column names to their associated
        generalization trees (GTree objects).
    logging_queue : multiprocessing.Queue
        Queue for sending log records from this process to the QueueListener.
    logging_level : int
        Logging level to set for the root logger in this worker process.

    Raises
    ------
    AssertionError
        If the variable name already exists in the global namespace.

    Notes
    -----
    This is an internal helper function used by the parallelism framework.
    It should not be called directly, but rather through ProcessPoolExecutor's
    initializer parameter after being set up by parallelism__make_initializer_and_initargs.
    """
    assert var_name not in globals(), f"{var_name} already exists in globals"
    globals()[var_name] = qid_to_gtree

    # Configure logging to send log records to parent process
    queue_handler = BlockingQueueHandler(logging_queue)
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logging
    root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging_level)

    # Reduce numba logging noise in worker processes
    logging.getLogger("numba").setLevel(logging.ERROR)


def parallelism__get_qid_to_gtree(varname: str) -> dict:
    """
    Retrieve generalization trees from the global namespace in a subprocess.

    This function retrieves the dictionary of generalization trees that was stored
    in the global namespace by parallelism__initializer. It should be called
    within the worker function that runs in the subprocess.

    Parameters
    ----------
    varname : str
        The unique variable name where generalization trees are stored in the
        global namespace.

    Returns
    -------
    dict
        The dictionary mapping quasi-identifier column names to their associated
        generalization trees (GTree objects).

    Notes
    -----
    This is an internal helper function used by the parallelism framework.
    The varname parameter should be part of the lookup_args returned by
    parallelism__make_initializer_and_initargs.

    """
    result: dict = globals()[varname]
    return result


def parallelism__initializer_cleanup(var_name: str) -> None:
    """
    Clean up global variables created by parallelism__initializer.

    This function removes the generalization trees that were stored in the global
    namespace by parallelism__initializer. It should be called when the parallel
    processing is complete to free up memory and avoid potential namespace conflicts.

    Parameters
    ----------
    var_name : str
        The unique variable name where generalization trees were stored in the
        global namespace.

    Raises
    ------
    AssertionError
        If the variable name doesn't exist in the global namespace.

    Notes
    -----
    This is an internal helper function used by the parallelism framework.
    It's good practice to call this function after parallel processing is complete
    to avoid memory leaks, especially when running multiple parallel operations
    in sequence.

    """
    assert var_name in globals(), f"{var_name} does not exist in globals"
    del globals()[var_name]


def nan_generator(
    df: pd.DataFrame,
    cols: list[str],
) -> Generator[tuple[pd.DataFrame, list[str]], None, None]:
    """
    Generate partitions of a dataframe based on NaN value patterns in specified columns.

    This function splits the input dataframe into multiple partitions based on whether each
    record has NaN values in the specified columns. Each unique pattern of NaN values
    forms a separate partition.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to partition.
    cols : List[str]
        List of column names to consider for NaN-based partitioning.

    Yields
    ------
    Tuple[pd.DataFrame, List[str]]
        For each partition, yields a tuple containing:
        1. A dataframe subset containing records with the same NaN pattern
        2. A list of column names that do not have NaN values in this partition

    Notes
    -----
    This function creates partitions by grouping rows according to which columns
    have NaN values. For example, if examining columns A and B, there could be
    partitions for:
    - Rows where both A and B are not NaN
    - Rows where A is NaN but B is not
    - Rows where A is not NaN but B is
    - Rows where both A and B are NaN

    This is useful for processing dataframes with incomplete data, where different
    operations might need to be applied to records with different missing value patterns.

    The partitions are yielded in the order that unique NaN patterns first appear
    in the DataFrame. This ordering is deterministic for any given DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4],
    ...     'B': [np.nan, 2, 3, 4],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>> for partition_df, valid_cols in nan_generator(df, ['A', 'B']):
    ...     print(f"Partition with valid columns {valid_cols}:")
    ...     print(partition_df)
    Partition with valid columns ['A']:
         A   B  C
    0  1.0 NaN  1
    Partition with valid columns ['A', 'B']:
         A    B  C
    1  2.0  2.0  2
    3  4.0  4.0  4
    Partition with valid columns ['B']:
        A    B  C
    2 NaN  3.0  3
    """
    if len(df) == 0 or len(cols) == 0:
        yield (df, cols)
    else:
        # Create NaN mask for all columns at once: O(N*K)
        # where N = number of rows, K = number of columns, P = unique NaN patterns
        nan_mask = df[cols].isna()

        # Convert boolean mask to hashable tuples for groupby: O(N*K)
        pattern_tuples = [tuple(row) for row in nan_mask.values]

        # Group DataFrame indices by NaN patterns in a single pass: O(N*K)
        # This achieves true O(N*K) complexity instead of O(N*P*K) from the previous approach
        pattern_to_indices: dict[tuple, list[int]] = {}
        for idx, pattern_tuple in enumerate(pattern_tuples):
            if pattern_tuple not in pattern_to_indices:
                pattern_to_indices[pattern_tuple] = []
            pattern_to_indices[pattern_tuple].append(idx)

        for pattern_tuple, group_indices in pattern_to_indices.items():
            # Extract subset DataFrame for this pattern: O(|group|)
            subset = df.iloc[group_indices]

            # Build list of non-NaN columns for this pattern: O(K)
            non_nan_cols = [col for col, is_nan in zip(cols, pattern_tuple) if not is_nan]
            yield subset, non_nan_cols
