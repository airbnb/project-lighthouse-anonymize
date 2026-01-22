"""
Tests for utils
"""

import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.constants import EPSILON
from project_lighthouse_anonymize.gtrees import make_flat_default_gtree
from project_lighthouse_anonymize.utils import (
    AnonymizeCallbacks,
    atleast_nunique,
    atleast_nunique_categorical,
    get_generalized_records,
    get_non_suppressed_records,
    get_suppressed_records,
    max_minus_min,
    median_max_second_max,
    min_max,
    nan_generator,
    parallelism__make_initializer_and_initargs,
    standard_deviation,
    standard_deviation_categorical,
)


# Tests for atleast_nunique function
def test_atleast_nunique_0():
    actual = atleast_nunique(np.array([]), 1)
    assert not actual
    actual = atleast_nunique(np.array([np.nan, np.nan, np.nan]), 2)
    assert not actual


def test_atleast_nunique_1():
    actual = atleast_nunique(np.array([np.nan, 4, 4]), 1)
    assert actual
    actual = atleast_nunique(np.array([np.nan, 4, 4]), 2)
    assert not actual
    actual = atleast_nunique(np.array([np.nan, 4, 4, 5]), 2)
    assert actual


# Tests for atleast_nunique_categorical function
def test_atleast_nunique_categorical_0():
    actual = atleast_nunique_categorical(pd.Series([]), 1)
    assert not actual
    actual = atleast_nunique_categorical(pd.Series([np.nan, np.nan, np.nan]), 2)
    assert not actual


def test_atleast_nunique_categorical_1():
    actual = atleast_nunique_categorical(pd.Series([np.nan, "A", "A"]), 1)
    assert actual
    actual = atleast_nunique_categorical(pd.Series([np.nan, "A", "A"]), 2)
    assert not actual
    actual = atleast_nunique_categorical(pd.Series([np.nan, "A", "A", "B"]), 2)
    assert actual


# Tests for standard_deviation function
def test_standard_deviation_0():
    actual = standard_deviation(np.array([]))
    assert np.isnan(actual)


def test_standard_deviation_1():
    expected = 0.50
    actual = standard_deviation(pd.Series([1.0, 0.0]).to_numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    actual = standard_deviation(pd.Series([1.0, np.nan, 0.0]).to_numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    actual = standard_deviation(pd.Series([1.0, np.nan, 0.0]).to_numpy(), np.array([np.nan]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_standard_deviation_large_arrays():
    """Test standard_deviation with arrays >= 1000 elements to trigger __standard_deviation_large"""
    # Create arrays with exactly 1000 elements to trigger the large path
    array_1000_a = np.array(range(1000))
    array_1000_b = np.array(range(1000, 2000))

    result = standard_deviation(array_1000_a, array_1000_b)
    assert not np.isnan(result)
    assert result >= 0


def test_standard_deviation_with_mixed_sizes():
    """Test standard_deviation function with mixed array sizes"""
    # Mix of small and large arrays to test different paths
    small_array = np.array(range(100))
    large_array = np.array(range(1500))  # Larger than 1000

    result = standard_deviation(small_array, large_array)
    assert not np.isnan(result)
    assert result >= 0


def test_standard_deviation_edge_cases():
    """Test standard_deviation edge cases including tiny arrays"""
    # Empty arrays
    empty_arrays = [np.array([]), np.array([])]
    result_empty = standard_deviation(*empty_arrays)
    assert np.isnan(result_empty)

    # Single element arrays - test multiple variations
    single_arrays_1 = [np.array([5]), np.array([10])]
    result_single_1 = standard_deviation(*single_arrays_1)
    assert not np.isnan(result_single_1)

    single_arrays_2 = [np.array([1]), np.array([2])]
    result_single_2 = standard_deviation(*single_arrays_2)
    assert not np.isnan(result_single_2)


def test_standard_deviation_zero_arrays():
    """Test standard_deviation with arrays containing only zeros"""
    # Test with arrays containing only zeros
    zero_arrays = [np.array([0, 0, 0]), np.array([0, 0])]
    result_zeros = standard_deviation(*zero_arrays)
    assert result_zeros == 0.0


# Tests for standard_deviation_categorical function
def test_average_distance_from_categorical_0():
    actual = standard_deviation_categorical(pd.Series([], dtype=np.dtype("object")))
    assert np.isnan(actual)


def test_standard_deviation_categorical_1():
    expected = 0.50
    actual = standard_deviation_categorical(pd.Series(["A", "B"]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    actual = standard_deviation_categorical(pd.Series(["A", np.nan, "B"]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    actual = standard_deviation_categorical(pd.Series(["A", np.nan, "B"]), pd.Series([np.nan]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_standard_deviation_categorical_2():
    expected = 0.4714045207910316
    actual = standard_deviation_categorical(pd.Series(["A", "B", "A"]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    actual = standard_deviation_categorical(pd.Series(["A", np.nan, "B", "A"]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    actual = standard_deviation_categorical(pd.Series(["A", np.nan, "B", "A"]), pd.Series([np.nan]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_standard_deviation_categorical():
    """Test standard_deviation_categorical function"""
    # Basic categorical test
    series1 = pd.Series(["A", "B", "C", "A", "B"])
    series2 = pd.Series(["X", "Y", "Z", "X", "Y"])
    result = standard_deviation_categorical(series1, series2)
    assert not np.isnan(result)
    assert result >= 0


def test_standard_deviation_categorical_edge_cases():
    """Test standard_deviation_categorical with edge cases"""
    # Test with identical series
    series_same = pd.Series(["A", "B", "C"])
    result_same = standard_deviation_categorical(series_same, series_same)
    assert result_same >= 0.0

    # Test with single category
    series_single = pd.Series(["A", "A", "A"])
    result_single = standard_deviation_categorical(series_single, series_single)
    assert result_single == 0.0


# Tests for max_minus_min function
def test_max_minus_min_0():
    actual = max_minus_min(np.array([]))
    assert np.isnan(actual)
    actual = max_minus_min(np.array([np.nan, np.nan]))
    assert np.isnan(actual)


def test_max_minus_min_1():
    actual = max_minus_min(np.array([1.0, np.nan, 1.0, 3.0, 4.0, np.nan]))
    expected = 4.0 - 1.0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_max_minus_min():
    """Test max_minus_min function"""
    # Basic test
    result = max_minus_min(np.array([1, 5, 3, 2, 4]))
    assert result == 4.0

    # With NaN values
    result_nan = max_minus_min(np.array([1, np.nan, 5, 3]))
    assert result_nan == 4.0


def test_max_minus_min_edge_cases():
    """Test max_minus_min edge cases"""
    # Test max_minus_min edge cases
    same_vals = max_minus_min(np.array([5, 5, 5]))
    assert same_vals == 0.0


# Tests for median_max_second_max function
def test_median_max_second_max_0():
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(np.array([]))
    assert np.isnan(actual_median)
    assert np.isnan(actual_maximum)
    assert np.isnan(actual_second_maximum)
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(
        np.array([np.nan, np.nan])
    )
    assert np.isnan(actual_median)
    assert np.isnan(actual_maximum)
    assert np.isnan(actual_second_maximum)


def test_median_max_second_max_1():
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(
        np.array([np.nan, 1.0])
    )
    np.testing.assert_allclose(actual_median, 1.0, atol=EPSILON)
    np.testing.assert_allclose(actual_maximum, 1.0, atol=EPSILON)
    assert np.isnan(actual_second_maximum)
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(
        np.array([np.nan, 1.0, 1.0])
    )
    np.testing.assert_allclose(actual_median, 1.0, atol=EPSILON)
    np.testing.assert_allclose(actual_maximum, 1.0, atol=EPSILON)
    assert np.isnan(actual_second_maximum)


def test_median_max_second_max_2():
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(
        np.array([np.nan, 1.0, 2.0])
    )
    np.testing.assert_allclose(actual_median, 1.5, atol=EPSILON)
    np.testing.assert_allclose(actual_maximum, 2.0, atol=EPSILON)
    np.testing.assert_allclose(actual_second_maximum, 1.0, atol=EPSILON)


def test_median_max_second_max_3():
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(
        np.array([0.0, np.nan, 1.0, 2.0])
    )
    np.testing.assert_allclose(actual_median, 1.0, atol=EPSILON)
    np.testing.assert_allclose(actual_maximum, 2.0, atol=EPSILON)
    np.testing.assert_allclose(actual_second_maximum, 1.0, atol=EPSILON)


def test_median_max_second_max_4():
    actual_median, actual_maximum, actual_second_maximum = median_max_second_max(
        np.array([1.0, np.nan, 1.0, 2.0])
    )
    np.testing.assert_allclose(actual_median, 1.0, atol=EPSILON)
    np.testing.assert_allclose(actual_maximum, 2.0, atol=EPSILON)
    np.testing.assert_allclose(actual_second_maximum, 1.0, atol=EPSILON)


def test_median_max_second_max():
    """Test median_max_second_max function"""
    # Basic test
    result = median_max_second_max(np.array([1, 2, 3, 4, 5]))
    assert result[0] == 3.0  # median
    assert result[1] == 5.0  # max
    assert result[2] == 4.0  # second max


def test_median_max_second_max_edge_cases():
    """Test median_max_second_max edge cases"""
    # Test median_max_second_max edge cases
    two_vals = median_max_second_max(np.array([1, 3]))
    assert two_vals[0] == 2.0  # median of two
    assert two_vals[1] == 3.0  # max
    assert two_vals[2] == 1.0  # second max


# Tests for min_max function
def test_min_max():
    """Test min_max function with comprehensive coverage"""
    # Normal case
    result = min_max(np.array([1, 5, 3, 2, 4]))
    assert result == (1.0, 5.0)

    # With NaN values
    result = min_max(np.array([1, np.nan, 5, 3]))
    assert result == (1.0, 5.0)

    # All NaN
    result = min_max(np.array([np.nan, np.nan]))
    assert np.isnan(result[0]) and np.isnan(result[1])

    # Single element arrays - test multiple variations
    result_single_1 = min_max(np.array([42.0]))
    assert result_single_1 == (42.0, 42.0)

    result_single_2 = min_max(np.array([42]))
    assert result_single_2 == (42.0, 42.0)


# Tests for get_non_suppressed_records function
def test_get_non_suppressed_records_with_nan_id():
    """
    Test that get_non_suppressed_records raises an AssertionError when id_col has NaN values.
    """
    df_orig = pd.DataFrame({"id": [1, 2, np.nan, 4], "value": [10, 20, 30, 40]})
    df_anon = pd.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})

    with pytest.raises(AssertionError):
        get_non_suppressed_records(df_orig, df_anon, "id", "_orig", "_anon")


# Tests for get_generalized_records function
def test_get_generalized_records_with_nan_id():
    """
    Test that get_generalized_records raises an AssertionError when id_col has NaN values.
    """
    df_orig = pd.DataFrame({"id": [1, 2, np.nan, 4], "value": [10, 20, 30, 40]})
    df_anon = pd.DataFrame({"id": [1, 2, 3, 4], "value": [15, 25, 30, 45]})

    with pytest.raises(AssertionError):
        get_generalized_records(df_orig, df_anon, "id", {"id"}, "_orig", "_anon")


def test_get_generalized_records_no_changes():
    """
    Test that get_generalized_records correctly handles NaN QID values
    and does not report them as changed when they remain unchanged.
    This test verifies the proper NaN-aware comparison logic.
    """
    # Create dataframes where QID columns contain NaN values that remain unchanged
    df_orig = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "qid_col": [np.nan, 20, np.nan],  # QID column with NaN values
            "non_qid": [100, 200, 300],  # Non-QID column
        }
    )
    df_anon = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "qid_col": [np.nan, 20, np.nan],  # Same NaN values, no actual changes
            "non_qid": [100, 200, 300],
        }
    )

    # Call get_generalized_records
    result = get_generalized_records(df_orig, df_anon, "id", {"id", "non_qid"}, "_orig", "_anon")

    # The function should return 0 rows since no values actually changed
    # NaN values that remain NaN should not be considered as generalized
    assert len(result) == 0
    assert len(result[result["id"] == 1]) == 0  # Row with id=1 should not be flagged
    assert len(result[result["id"] == 2]) == 0  # Row with id=2 should not be flagged
    assert len(result[result["id"] == 3]) == 0  # Row with id=3 should not be flagged


def test_get_suppressed_records_with_nan_id():
    """
    Test that get_suppressed_records raises an AssertionError when id_col has NaN values.
    """
    df_orig = pd.DataFrame({"id": [1, 2, np.nan, 4], "value": [10, 20, 30, 40]})
    df_anon = pd.DataFrame({"id": [1, 4], "value": [10, 40]})

    with pytest.raises(AssertionError):
        get_suppressed_records(df_orig, df_anon, "id", {"id"}, "_orig", "_anon")


def test_anonymize_callbacks():
    """Test AnonymizeCallbacks timing functionality"""
    callbacks = AnonymizeCallbacks()

    # Test callback methods store timestamps
    callbacks.anonymize_bm()
    assert "anonymize_bm" in callbacks.timestamps

    callbacks.anonymize_am()
    assert "anonymize_am" in callbacks.timestamps

    callbacks.compute_metrics_bm()
    assert "compute_metrics_bm" in callbacks.timestamps

    callbacks.compute_metrics_am()
    assert "compute_metrics_am" in callbacks.timestamps


def test_anonymize_callbacks_class():
    """Test AnonymizeCallbacks class"""
    callbacks = AnonymizeCallbacks()

    # Test each callback method
    callbacks.anonymize_bm()
    assert "anonymize_bm" in callbacks.timestamps

    callbacks.anonymize_am()
    assert "anonymize_am" in callbacks.timestamps

    callbacks.compute_metrics_bm()
    assert "compute_metrics_bm" in callbacks.timestamps

    callbacks.compute_metrics_am()
    assert "compute_metrics_am" in callbacks.timestamps


# Tests for nan_generator function
def test_nan_generator():
    """Test nan_generator function with various scenarios"""
    # Test empty DataFrame
    df = pd.DataFrame({})
    result = list(nan_generator(df, []))
    assert len(result) == 1
    pd.testing.assert_frame_equal(result[0][0], df)
    assert result[0][1] == []

    # Test DataFrame with no specified columns
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    result = list(nan_generator(df, []))
    assert len(result) == 1

    # Test DataFrame with mixed NaN patterns
    df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4, np.nan],
            "B": [np.nan, 2, 3, np.nan, 5],
            "C": [1, 2, 3, 4, 5],
        }
    )
    result = list(nan_generator(df, ["A", "B"]))
    # Should create multiple partitions based on NaN patterns
    assert len(result) >= 2


def test_nan_generator_patterns():
    """Test nan_generator function"""
    # Test with DataFrame containing NaN patterns
    df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, 2, 3]})

    result = list(nan_generator(df, ["col1", "col2"]))
    assert len(result) >= 1

    # Verify all rows are preserved across partitions
    total_rows = sum(len(partition[0]) for partition in result)
    assert total_rows == len(df)


def test_nan_generator_empty_columns():
    """Test nan_generator with empty columns list"""
    # Test nan_generator with empty columns list
    df_empty_cols = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    result_empty_cols = list(nan_generator(df_empty_cols, []))
    assert len(result_empty_cols) == 1
    pd.testing.assert_frame_equal(result_empty_cols[0][0], df_empty_cols)


# Tests for parallelism functions
def test_parallelism_make_initializer():
    """Test parallelism__make_initializer_and_initargs setup function"""
    gtree = make_flat_default_gtree(["A", "B", "C"])
    qid_to_gtree = {"test_qid": gtree}
    ctx = mp.get_context("spawn")
    logging_queue = ctx.Queue()

    try:
        result = parallelism__make_initializer_and_initargs(
            qid_to_gtree, logging_queue, logging.INFO
        )
        assert len(result) == 3
        initializer_func, init_args, lookup_args = result
        assert callable(initializer_func)
        assert isinstance(init_args, tuple)
        assert isinstance(lookup_args, tuple)
        assert len(init_args) == 4
    finally:
        logging_queue.close()
        logging_queue.cancel_join_thread()


def test_parallelism_setup():
    """Test parallelism setup function"""
    gtree = make_flat_default_gtree(["A", "B"])
    qid_to_gtree = {"test": gtree}
    ctx = mp.get_context("spawn")
    logging_queue = ctx.Queue()

    try:
        result = parallelism__make_initializer_and_initargs(
            qid_to_gtree, logging_queue, logging.INFO
        )
        assert len(result) == 3
        initializer_func, init_args, lookup_args = result
        assert callable(initializer_func)
    finally:
        logging_queue.close()
        logging_queue.cancel_join_thread()


def test_parallelism_logging_configuration():
    """Test that parallelism logging setup and cleanup functions work correctly"""
    import logging.handlers

    from project_lighthouse_anonymize.utils import (
        parallelism__cleanup_logging,
        parallelism__setup_logging,
    )

    ctx = mp.get_context("spawn")
    original_handlers = logging.getLogger().handlers[:]
    original_level = logging.getLogger().level

    try:
        logging_queue, logging_listener, saved_handlers, saved_level = parallelism__setup_logging(
            ctx
        )

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.handlers.QueueHandler)
        assert saved_level == original_level
        assert len(saved_handlers) == len(original_handlers)

        parallelism__cleanup_logging(logging_listener, saved_handlers, saved_level, logging_queue)

        root_logger = logging.getLogger()
        assert root_logger.level == saved_level
        assert len(root_logger.handlers) == len(saved_handlers)

    finally:
        # Ensure logging state is always restored regardless of test outcome
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(original_level)
        for handler in original_handlers:
            root_logger.addHandler(handler)
