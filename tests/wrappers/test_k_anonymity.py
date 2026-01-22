"""
Tests for k_anonymize wrapper function

# Test Architecture Overview

This file defines the base test classes for k-anonymization testing. The test structure
uses inheritance-based patterns to maximize code reuse and ensure consistent testing
across different anonymization modes.

## Base Test Classes

This file contains several base test classes that define shared test methods:

- `TestKAnonymitySharedNumericalWithoutNaN`: Tests for numerical data without NaN values
- `TestKAnonymitySharedNumerical`: Tests for numerical data including NaN handling
- `TestKAnonymitySharedCategorical`: Tests for categorical data and mixed data types

These base classes have `__test__ = False` to prevent pytest from collecting them directly.
They contain the actual test method implementations that are inherited by concrete test classes.

## Concrete Test Classes

Concrete test classes inherit from the base classes and have `__test__ = True`:

- `TestKAnonymityNumerical`: Inherits numerical tests for basic k-anonymization
- `TestKAnonymityCategorical`: Inherits categorical tests for basic k-anonymization

## Pytest Parametrization

The `pytest_generate_tests()` function automatically parametrizes all test methods that
have a `kwargs` parameter. This runs each test with multiple algorithm configurations
defined in the `KWARGS` list, ensuring comprehensive coverage of different algorithm
settings (parallel vs non-parallel, different scoring methods, etc.).

## Usage in Other Test Files

Other test files (test_wrappers_u_anonymity.py, test_wrappers_strict_s_u_anonymity.py, etc.)
import and inherit from these base classes to reuse test logic while applying different
anonymization parameters. See the docstrings in those files for specific inheritance patterns.

## Adding New Tests

To add new k-anonymization tests:
1. Add test methods to the appropriate base class if they should be shared
2. Create a new concrete class inheriting from base classes if you need different configurations
3. Set `__test__ = True` on concrete classes, `__test__ = False` on base classes
4. Ensure test methods accept `kwargs` parameter for automatic parametrization
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize import gtrees
from project_lighthouse_anonymize.constants import (
    EPSILON,
    MAXIMUM_PRECISION_DIGITS,
    NOT_DEFINED_NA,
)
from project_lighthouse_anonymize.utils import AnonymizeCallbacks
from project_lighthouse_anonymize.wrappers.k_anonymize import (
    default_dq_metric_to_minimum_dq,
    k_anonymize,
)
from tests.shared import assert_dataframes_equal_unordered

_LOGGER = logging.getLogger(__name__)
_ID_COL = "row_id"

KWARGS = [
    # Parallel K-Anonymize, examining multiple proposed cuts and cut points
    {
        "parallelism": 10,
        "rilm_score_epsilon": 0.05,
        "dynamic_breakout_rilm_multiplier": 0.75,
        "complex_numerical_cut_points_modes": None,
    },
    # Same as above but with R&D mode enabled
    {
        "parallelism": 10,
        "rilm_score_epsilon": 0.05,
        "dynamic_breakout_rilm_multiplier": 0.75,
        "complex_numerical_cut_points_modes": None,
        "rnd_mode": True,
    },
    # Not-parallel K-Anonymize, not examining multiple proposed cuts nor cut points
    {
        "parallelism": None,
        "rilm_score_epsilon": -1.0,
        "dynamic_breakout_rilm_multiplier": None,
        "complex_numerical_cut_points_modes": False,
    },
]


def pytest_generate_tests(metafunc):
    if "kwargs" in metafunc.fixturenames:
        metafunc.parametrize("kwargs", KWARGS)


class TestKAnonymitySharedNumericalWithoutNaN:
    """
    Base test class for k-anonymization with numerical data that do not contain NaN values.

    This class defines shared test methods for numerical data scenarios without NaN handling
    complexity. Test methods are inherited by concrete test classes that set specific
    anonymization parameters.

    Inheritance Pattern:
    - This is a base class with `__test__ = False` to prevent direct pytest collection
    - Concrete classes inherit from this and set `__test__ = True`
    - All test methods automatically receive different `kwargs` configurations via pytest parametrization

    Test Coverage:
    - Empty dataframe handling
    - Boundary conditions (k-1 records)
    - Basic numerical anonymization scenarios
    - Algorithm correctness verification
    """

    # This is a base class for inheritance - pytest should not collect it directly
    __test__ = False

    def _call_anonymize_function(self, logger, input_df, qids, k, qid_to_gtree, id_col, **kwargs):
        """
        Abstract method for calling the appropriate anonymization function.
        Base implementation calls k_anonymize. Subclasses override to call u_anonymize.
        """
        return k_anonymize(logger, input_df, qids, k, qid_to_gtree, id_col, **kwargs)

    def test_empty_dataframe_returns_empty_result(self, kwargs):
        """
        Test that k-anonymization of an empty dataframe returns empty result with correct metrics.
        """
        input_df = pd.DataFrame(data={_ID_COL: [], "col_1": [], "col_2": []})
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == 0
        assert dq_metrics["n_suppressed"] == 0
        assert np.isnan(dq_metrics["pearsons__minimum"])
        assert np.isnan(dq_metrics["nmi_sampled_scaled_v1__minimum"])

    def test_numerical_data_with_k_minus_one_records_suppresses_all(self, kwargs):
        """
        Test that numerical data with k-1 records suppresses all records (boundary condition).
        """
        input_df = pd.DataFrame(data={_ID_COL: [1, 2], "col_1": [1, 1], "col_2": [2, 2]})
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == 0
        assert dq_metrics["n_suppressed"] == 2
        assert np.isnan(dq_metrics["pearsons__minimum"])
        assert np.isnan(dq_metrics["nmi_sampled_scaled_v1__minimum"])

    def test_numerical_data_with_k_identical_records_unchanged(self, kwargs):
        """
        Test that k identical numerical records remain unchanged with perfect metrics.
        """
        input_df = pd.DataFrame(data={_ID_COL: [1, 2, 3], "col_1": [1, 1, 1], "col_2": [2, 2, 2]})
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        assert_dataframes_equal_unordered(
            input_df, output_df[input_df.columns], ignore_columns=["row_id"]
        )  # should be unchanged
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(dq_metrics["nmi_sampled_scaled_v1__minimum"], 1, atol=EPSILON)
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 1, atol=EPSILON)

    def test_numerical_two_ecs_totaling_k_records_with_idempotency(self, kwargs):
        """
        Test numerical data with two equivalence classes totaling k records, and verify k-anonymization is idempotent.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "col_2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["nmi_sampled_scaled_v1__minimum"], 0, atol=EPSILON
        )  # all entropy lost
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 0.0, atol=EPSILON)
        old_output_df = output_df
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            old_output_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert_dataframes_equal_unordered(
            old_output_df, output_df, ignore_columns=["row_id"]
        )  # should be unchanged
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(dq_metrics["nmi_sampled_scaled_v1__minimum"], 1, atol=EPSILON)
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 1, atol=EPSILON)

    def test_numerical_anonymize_callbacks_invoked_correctly(self, kwargs):
        """
        Test that anonymize_callbacks are invoked correctly during numerical data k-anonymization.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "col_2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            }
        )
        anonymize_callbacks = AnonymizeCallbacks()
        anonymize_callbacks.anonymize_bm = MagicMock()
        anonymize_callbacks.anonymize_am = MagicMock()
        anonymize_callbacks.compute_metrics_bm = MagicMock()
        anonymize_callbacks.compute_metrics_am = MagicMock()
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            anonymize_callbacks=anonymize_callbacks,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        anonymize_callbacks.anonymize_bm.assert_called_once()
        anonymize_callbacks.anonymize_am.assert_called_once()
        anonymize_callbacks.compute_metrics_bm.assert_called_once()
        anonymize_callbacks.compute_metrics_am.assert_called_once()

    def test_numerical_two_ecs_totaling_2k_records_unchanged(self, kwargs):
        """
        Test that numerical data with two ECs totaling 2*k records remains unchanged with perfect metrics.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "col_1": [1] * 10 + [2] * 10,
                "col_2": [2] * 10 + [2] * 10,
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["nmi_sampled_scaled_v1__minimum"], 1, atol=EPSILON
        )  # no changes needed
        np.testing.assert_allclose(
            dq_metrics["pearsons__minimum"], 1, atol=EPSILON
        )  # no changes needed

    def test_numerical_no_qids_specified_unchanged(self, kwargs):
        """
        Test that numerical data with no QIDs specified remains unchanged.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3],  # id col + not marked as a QID
                "col_1": [0.0, 0.0, 0.0],  # not marked as a QID
            }
        )
        expected_output_df = input_df.copy(deep=True)
        actual_output_df, actual_dq_metrics, actual_disclosure_metrics = (
            self._call_anonymize_function(
                _LOGGER,
                input_df,
                [],
                3,
                {},
                _ID_COL,
                **kwargs,
            )
        )

        assert_dataframes_equal_unordered(
            expected_output_df,
            actual_output_df[input_df.columns],
            ignore_columns=["row_id"],
        )  # should be unchanged

        # these exist in actual_dq_metrics as a result of how we handle no qids, but shouldn't be assumed to exist:
        #  - nmi_sampled_scaled_v1__minimum
        #  - pearsons__minimum
        np.testing.assert_allclose(actual_dq_metrics["pct_non_suppressed"], 1.0, atol=EPSILON)
        np.testing.assert_allclose(actual_dq_metrics["pct_suppressed"], 0, atol=EPSILON)
        np.testing.assert_allclose(actual_dq_metrics["pct_generalized"], 0, atol=EPSILON)

    def test_exclude_qids_parameter_affects_rilm_metrics(self, kwargs):
        """
        Test that exclude_qids parameter correctly affects RILM metric calculations.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6],
                "col_1": [1, 2, 3, 4, 5, 6],
                "col_2": [2, 2, 2, 2, 2, 2],
            }
        )
        _, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            exclude_qids=["col_1"],
            **kwargs,
        )
        np.testing.assert_allclose(dq_metrics["nmi_sampled_scaled_v1__minimum"], 1, atol=EPSILON)
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 1, atol=EPSILON)

    def test_ec_of_exactly_k_is_kept_numerical(self, kwargs):
        """
        Test 1: Boundary Condition - EC of Exactly k is Kept (Numerical)

        An equivalence class (EC) whose size is exactly equal to k must be retained.
        This catches off-by-one errors where ">" is mistakenly used instead of ">=".
        Uses numerical QID that gets generalized to satisfy k-anonymity.
        """
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "age": [25, 26, 27, 45],  # All will be generalized together
                "dx": ["flu", "cold", "covid", "cancer"],
            }
        )

        # With k=3, all records should be kept and generalized to the same age range
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["age"], 3, {}, _ID_COL, **kwargs
        )

        assert len(output_df) == 4  # All records kept
        assert dq_metrics["n_suppressed"] == 0  # No suppression with numerical generalization
        # All records should have the same generalized age value
        assert len(output_df["age"].unique()) == 1

    def test_ec_assignment_only_by_qids_numerical(self, kwargs):
        """
        Test 3: EC Assignment Only By QIDs, Not Other Fields (Numerical)

        Grouping must use only the columns given in `qid_cols`;
        non-QID columns are ignored even if they vary.
        Uses numerical QID with different non-QID numerical values.
        """
        # Input data with similar QID values but different non-QID values
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2],
                "age": [25, 26],  # Close ages that will be generalized together
                "score": [85, 92],  # Different non-QID numerical values
            }
        )

        # Run k-anonymization
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["age"], 2, {}, _ID_COL, **kwargs
        )

        # Verify that all records are kept despite different non-QID values
        assert len(output_df) == 2
        assert dq_metrics["n_suppressed"] == 0
        # All records should have the same generalized age value
        assert len(output_df["age"].unique()) == 1
        # Non-QID values should remain unchanged
        assert set(output_df["score"]) == {85, 92}

    def test_user_id_columns_ignored_numerical(self, kwargs):
        """
        Test 4: User-ID Columns Are Ignored for k-anonymity (Numerical)

        When a user-ID is present, it must be ignored;
        what matters is row count, not unique users.
        Uses numerical QID that gets generalized.
        """
        # Input data with user_id column - k-anonymity should count rows, not users
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "user_id": [1, 1, 2, 3],  # 3 unique users
                "salary": [50000, 52000, 51000, 80000],  # Different salary ranges
            }
        )

        # Run k-anonymization (don't pass user_id_col to ensure k-anonymity not u-anonymity)
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["salary"], 3, {}, _ID_COL, **kwargs
        )

        # With k=3, all records should be kept and generalized based on row count
        assert len(output_df) == 4  # All records kept
        assert dq_metrics["n_suppressed"] == 0  # No suppression with numerical generalization
        # All records should have the same generalized salary value
        assert len(output_df["salary"].unique()) == 1
        # User IDs should remain unchanged
        assert set(output_df["user_id"]) == {1, 2, 3}

    def test_grouping_uses_all_qid_columns_numerical(self, kwargs):
        """
        Test 5: Grouping Uses ALL QID Columns in Combination (Numerical)

        ECs are defined as tuples of all QID columns, not by each individually.
        Uses multiple numerical QID columns that get generalized together.
        """
        # Input data with multiple numerical QID columns
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "age": [25, 26, 27, 45],
                "salary": [50000, 52000, 51000, 80000],
            }
        )

        # Run k-anonymization with multiple QID columns
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["age", "salary"], 2, {}, _ID_COL, **kwargs
        )

        # All records should be kept and generalized together
        assert len(output_df) == 4
        assert dq_metrics["n_suppressed"] == 0
        # With k=2 and 4 records, we should have exactly 2 groups of 2 records each
        assert len(output_df["age"].unique()) == 2
        assert len(output_df["salary"].unique()) == 2
        # Each group should have exactly 2 records
        age_counts = output_df["age"].value_counts()
        salary_counts = output_df["salary"].value_counts()
        assert all(count == 2 for count in age_counts.values)
        assert all(count == 2 for count in salary_counts.values)

    def test_k_parameter_enforced_as_at_least_numerical(self, kwargs):
        """
        Test 6: k Parameter is Enforced as "At Least" (Numerical)

        Rows are kept only if their EC size is **≥ k**. If k is too large for any EC, all are suppressed.
        Even with numerical generalization, if k is larger than the total number of records, all are suppressed.
        """
        # Input data with 9 records, k=10 (more than total records)
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "age": [25, 26, 27, 35, 36, 37, 45, 46, 47],
            }
        )

        # Run k-anonymization with k larger than total records
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["age"], 10, {}, _ID_COL, **kwargs
        )

        # Verify that all records are suppressed since k=10 > 9 total records
        assert len(output_df) == 0
        assert dq_metrics["n_suppressed"] == 9

    def test_k_equals_one_identity_transformation_numerical(self, kwargs):
        """
        Test that k=1 preserves numerical input unchanged (identity transformation).

        This test verifies the edge case where k=1, which should result in
        no generalization or suppression since every record forms its own
        equivalence class of size 1, which satisfies k-anonymity.
        """
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3],
                "age": [25.5, 30.1, 45.8],
                "salary": [50000, 65000, 80000],
            }
        )

        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["age", "salary"], 1, {}, _ID_COL, **kwargs
        )

        # For k=1, input should equal output exactly
        assert_dataframes_equal_unordered(
            input_df, output_df[input_df.columns], ignore_columns=["row_id"]
        )
        assert dq_metrics["n_suppressed"] == 0


class TestKAnonymitySharedNumerical(TestKAnonymitySharedNumericalWithoutNaN):
    """
    Base test class for k-anonymization with numerical data including NaN handling.

    This class extends TestKAnonymitySharedNumericalWithoutNaN to add tests for more complex
    numerical scenarios involving NaN values and edge cases.

    Inheritance Pattern:
    - Inherits all basic numerical tests from TestKAnonymitySharedNumericalWithoutNaN
    - Adds NaN-specific test scenarios
    - Base class with `__test__ = False` - concrete classes set `__test__ = True`

    Test Coverage:
    - All tests from parent class (basic numerical scenarios)
    - NaN value handling in QID columns
    - Mixed ECs with NaN and non-NaN values
    - Edge cases with NaN data
    """

    # This is a base class for inheritance - pytest should not collect it directly
    __test__ = False

    def test_numerical_mixed_ecs_with_nan_qid_values_preserved(self, kwargs):
        """
        Test numerical data with mixed ECs including one with NaN QID values, all meeting k requirement.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "col_2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                + [
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["nmi_sampled_scaled_v1__minimum"], 0.358794, atol=EPSILON
        )
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 0.5773503, atol=EPSILON)

    def test_numerical_undersized_ec_with_nan_qid_suppressed(self, kwargs):
        """
        Test that numerical data with undersized EC containing NaN QID values gets suppressed.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13],
                "col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] + [2, 2, 2],
                "col_2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                + [NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert (
            len(output_df) == len(input_df) - 3
        )  # all 3 values in that additional equivalence class were suppressed
        assert (
            dq_metrics["n_suppressed"] == 3
        )  # all 3 values in that additional equivalence class were suppressed
        np.testing.assert_allclose(
            dq_metrics["nmi_sampled_scaled_v1__minimum"], 0, atol=EPSILON
        )  # no entropy left after merging two values for col_1 together
        np.testing.assert_allclose(
            dq_metrics["pearsons__minimum"], 0.0, atol=EPSILON
        )  # col_1 0.0, col_2 1.0

    def test_numerical_ec_with_all_qids_nan_preserved(self, kwargs):
        """
        Test numerical data with EC having all QIDs as NaN values, meeting k requirement.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
                + [
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
                "col_2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                + [
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["nmi_sampled_scaled_v1__minimum"], 0, atol=EPSILON
        )  # no entropy left after merging two values for col_1 together
        np.testing.assert_allclose(
            dq_metrics["pearsons__minimum"], 0.0, atol=EPSILON
        )  # col_1 0.0, col_2 1.0

    def test_numerical_mixed_nan_and_non_nan_qids_tree_handling(self, kwargs):
        """
        Test numerical data with mixed NaN and non-NaN QID values to verify tree structure handling.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6],
                "col_1": [1, 1, 1, NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA],
                "col_2": [10, 10, 10, NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        # All rows should be kept since each group meets k=3
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        # The non-NaN values should remain unchanged
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 1.0, atol=EPSILON)

    def test_nulls_in_qids_form_own_ec_numerical(self, kwargs):
        """
        Test 7: Nulls in QIDs Form Their Own EC (Numerical)

        If a QID value is None/NaN, those records form their own EC and are counted together.
        Uses numerical QID with NaN values.
        """
        # Input data with NaN values in QID columns
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "age": [np.nan, np.nan, 25.0, 26.0],
                "score": [12, 15, 23, 25],
            }
        )

        # Run k-anonymization
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["age"], 2, {}, _ID_COL, **kwargs
        )

        # Verify that both NaN and non-NaN ECs remain
        assert len(output_df) == 4
        assert dq_metrics["n_suppressed"] == 0
        # Should have exactly 2 unique age values: NaN and a generalized numerical value
        age_values = output_df["age"].dropna().unique()
        nan_count = output_df["age"].isna().sum()
        assert len(age_values) == 1  # One generalized value for non-NaN records
        assert nan_count == 2  # Two NaN records remain as NaN


class TestKAnonymityNumerical(TestKAnonymitySharedNumerical):
    """Concrete test class for k_anonymize with numerical data including NaN tests"""

    __test__ = True  # Override base class to allow test collection


class TestKAnonymitySharedCategorical:
    """
    Base test class for k-anonymization with categorical and mixed data types.

    This class defines shared test methods for categorical data scenarios and mixed
    numerical/categorical datasets. It's the foundation for testing complex data
    anonymization scenarios.

    Inheritance Pattern:
    - Independent base class (doesn't inherit from numerical classes)
    - Base class with `__test__ = False` - concrete classes set `__test__ = True`
    - Test methods automatically parametrized via pytest_generate_tests()

    Test Coverage:
    - Categorical data anonymization
    - Mixed numerical/categorical datasets
    - Boundary conditions with categorical data
    - Algorithm determinism verification
    - Data quality metrics validation
    """

    # This is a base class for inheritance - pytest should not collect it directly
    __test__ = False

    def _call_anonymize_function(self, logger, input_df, qids, k, qid_to_gtree, id_col, **kwargs):
        return k_anonymize(logger, input_df, qids, k, qid_to_gtree, id_col, **kwargs)

    def test_categorical_data_with_k_minus_one_records_suppresses_all(self, kwargs):
        """
        Test that categorical data with k-1 records suppresses all records (boundary condition).
        """
        input_df = pd.DataFrame(data={_ID_COL: [1, 2], "col_1": ["A", "A"], "col_2": ["B", "B"]})
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == 0
        assert dq_metrics["n_suppressed"] == 2
        # no non-suppressed records
        assert np.isnan(dq_metrics["rilm_categorical__minimum"])

    def test_categorical_data_with_k_identical_records_unchanged(self, kwargs):
        """
        Test that k identical categorical records remain unchanged with perfect metrics.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3],
                "col_1": ["A", "A", "A"],
                "col_2": ["B", "B", "B"],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        assert_dataframes_equal_unordered(
            input_df, output_df[input_df.columns], ignore_columns=["row_id"]
        )  # should be unchanged
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(dq_metrics["rilm_categorical__minimum"], 1, atol=EPSILON)

    def test_categorical_two_ecs_totaling_k_records_with_idempotency(self, kwargs):
        """
        Test categorical data with two equivalence classes totaling k records, and verify k-anonymization is idempotent.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col_1": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                "col_2": ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["rilm_categorical__minimum"], 0, atol=EPSILON
        )  # the two distinct values for col_1 were merged together
        old_output_df = output_df
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            old_output_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert_dataframes_equal_unordered(
            old_output_df, output_df, ignore_columns=["row_id"]
        )  # should be unchanged
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["rilm_categorical__minimum"], 1, atol=EPSILON
        )  # values are unchanged because the input was already k-anonymous

    def test_categorical_two_ecs_totaling_2k_records_unchanged(self, kwargs):
        """
        Test that categorical data with two ECs totaling 2*k records remains unchanged with perfect metrics.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "col_1": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
                + ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                "col_2": ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
                + ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["rilm_categorical__minimum"], 1, atol=EPSILON
        )  # no changes needed

    def test_categorical_mixed_ecs_with_nan_qid_values_preserved(self, kwargs):
        """
        Test categorical data with mixed ECs including one with NaN QID values, all meeting k requirement.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "col_1": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
                + ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
                "col_2": ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
                + [
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["rilm_categorical__minimum"], 0.5, atol=EPSILON
        )  # half of col_1 required complete merging (0.0) and half no changes (1.0)

    def test_categorical_undersized_ec_with_nan_qid_suppressed(self, kwargs):
        """
        Test that categorical data with undersized EC containing NaN QID values gets suppressed.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13],
                "col_1": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"] + ["B", "B", "B"],
                "col_2": ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
                + [NOT_DEFINED_NA, NOT_DEFINED_NA, NOT_DEFINED_NA],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert (
            len(output_df) == len(input_df) - 3
        )  # all 3 values in that additional equivalence class were suppressed
        assert (
            dq_metrics["n_suppressed"] == 3
        )  # all 3 values in that additional equivalence class were suppressed
        np.testing.assert_allclose(
            dq_metrics["rilm_categorical__minimum"], 0, atol=EPSILON
        )  # the two distinct values for col_1 were merged together

    def test_categorical_ec_with_all_qids_nan_preserved(self, kwargs):
        """
        Test categorical data with EC having all QIDs as NaN values, meeting k requirement.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "col_1": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
                + [
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
                "col_2": ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
                + [
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            10,
            {},
            _ID_COL,
            **kwargs,
        )
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(
            dq_metrics["rilm_categorical__minimum"], 0, atol=EPSILON
        )  # the two distinct values for col_1 were merged together

    def test_categorical_mixed_nan_and_non_nan_qids_tree_handling(self, kwargs):
        """
        Test categorical data with mixed NaN and non-NaN QID values to verify tree structure handling.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6],
                "col_1": [
                    "A",
                    "A",
                    "A",
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
                "col_2": [
                    "X",
                    "X",
                    "X",
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                    NOT_DEFINED_NA,
                ],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        # All rows should be kept since each group meets k=3
        assert len(output_df) == len(input_df)
        assert dq_metrics["n_suppressed"] == 0
        # The non-NaN values should remain unchanged
        np.testing.assert_allclose(dq_metrics["rilm_categorical__minimum"], 1.0, atol=EPSILON)

    def test_categorical_no_qids_specified_unchanged(self, kwargs):
        """
        Test that categorical data with no QIDs specified remains unchanged.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3],  # id col + not marked as a QID
                "col_1": ["A", "A", "A"],  # not marked as a QID
            }
        )
        expected_output_df = input_df.copy(deep=True)
        actual_output_df, actual_dq_metrics, actual_disclosure_metrics = (
            self._call_anonymize_function(
                _LOGGER,
                input_df,
                [],
                3,
                {},
                _ID_COL,
                **kwargs,
            )
        )
        assert_dataframes_equal_unordered(
            expected_output_df,
            actual_output_df[input_df.columns],
            ignore_columns=["row_id"],
        )  # should be unchanged
        # these exist in actual_dq_metrics as a result of how we handle no qids, but shouldn't be assumed to exist:
        #  - nmi_sampled_scaled_v1__minimum
        #  - pearsons__minimum
        np.testing.assert_allclose(actual_dq_metrics["pct_non_suppressed"], 1.0, atol=EPSILON)
        np.testing.assert_allclose(actual_dq_metrics["pct_suppressed"], 0, atol=EPSILON)
        np.testing.assert_allclose(actual_dq_metrics["pct_generalized"], 0, atol=EPSILON)

    def test_nan_in_id_column_raises_assertion_error(self, kwargs):
        """
        Test that k_anonymize raises an AssertionError when id_col has NaN values.
        """
        input_df = pd.DataFrame(data={_ID_COL: [1, 2, np.nan, 4], "col_1": ["A", "B", "C", "D"]})

        with pytest.raises(ValueError):
            self._call_anonymize_function(
                _LOGGER,
                input_df,
                ["col_1"],
                3,
                {},
                _ID_COL,
                **kwargs,
            )

    def test_categorical_two_ecs_size_k_without_gtree(self, kwargs):
        """
        Test categorical data with two ECs of size k when no generalization tree is provided.
        """
        input_df = pd.DataFrame(
            data={
                _ID_COL: [1, 2, 3, 4, 5, 6],
                "col_1": ["foo", "foo", "foo", "bar", "bar", "bar"],
                "col_2": [2, 2, 2, 2, 2, 2],
            }
        )
        output_df, dq_metrics, disclosure_metrics = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["col_1", "col_2"],
            3,
            {},
            _ID_COL,
            **kwargs,
        )
        assert_dataframes_equal_unordered(
            input_df, output_df[input_df.columns], ignore_columns=["row_id"]
        )  # should be unchanged
        assert dq_metrics["n_suppressed"] == 0
        np.testing.assert_allclose(dq_metrics["rilm_categorical__minimum"], 1, atol=EPSILON)
        np.testing.assert_allclose(dq_metrics["nmi_sampled_scaled_v1__minimum"], 1, atol=EPSILON)
        np.testing.assert_allclose(dq_metrics["pearsons__minimum"], 1.0)

    def test_deterministic_algorithm_behavior(self, kwargs):
        """
        Test that the k-anonymization algorithm produces deterministic results.

        Creates a dataset with both numerical and categorical QIDs and verifies
        that running k-anonymization multiple times on identical input produces
        identical output, ensuring algorithm determinism.
        """
        rng = np.random.default_rng(seed=1204890231)  # deterministic across test runs
        n_rows = 10_000
        input_df = pd.DataFrame(
            data={
                _ID_COL: list(range(n_rows)),
                "col_1": rng.random(n_rows).round(MAXIMUM_PRECISION_DIGITS),
                "col_2": rng.random(n_rows).round(MAXIMUM_PRECISION_DIGITS),
                "col_3": rng.choice(["a", "b", "c", "d"], size=n_rows, replace=True),
                "col_4": rng.choice(["a", "b", "c", "d"], size=n_rows, replace=True),
            }
        )
        qids = ["col_1", "col_2", "col_3", "col_4"]
        expected_output_df, _, _ = self._call_anonymize_function(
            _LOGGER,
            input_df,
            qids,
            5,
            {},
            _ID_COL,
            **kwargs,
        )
        for _ in range(5):
            actual_output_df, _, _ = self._call_anonymize_function(
                _LOGGER,
                input_df,
                qids,
                5,
                {},
                _ID_COL,
                **kwargs,
            )
            assert_dataframes_equal_unordered(
                expected_output_df, actual_output_df, ignore_columns=["row_id"]
            )

    def test_ec_of_exactly_k_is_kept(self, kwargs):
        """
        Test 1: Boundary Condition - EC of Exactly k is Kept

        An equivalence class (EC) whose size is exactly equal to k must be retained.
        This catches off-by-one errors where ">" is mistakenly used instead of ">=".
        """
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "zipcode": ["90210", "90210", "90210", "10001"],
                "dx": ["flu", "cold", "covid", "cancer"],
            }
        )

        expected_output = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3],
                "zipcode": ["90210", "90210", "90210"],
                "dx": ["flu", "cold", "covid"],
            }
        )

        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["zipcode"], 3, {}, _ID_COL, **kwargs
        )

        assert len(output_df) == 3
        assert dq_metrics["n_suppressed"] == 1
        assert_dataframes_equal_unordered(
            expected_output,
            output_df[expected_output.columns],
            ignore_columns=["row_id"],
        )

    def test_mixed_ecs_some_survive_some_suppressed(self, kwargs):
        """
        Test 2: Multiple ECs - Some Survive, Some Are Suppressed

        If a dataset contains several ECs and some are under-sized,
        only the compliant ECs are kept; non-compliant are fully suppressed.
        """
        input_df = pd.DataFrame(
            {_ID_COL: [1, 2, 3], "city": ["NY", "NY", "Boston"], "dx": ["A", "B", "C"]}
        )

        expected_output = pd.DataFrame({_ID_COL: [1, 2], "city": ["NY", "NY"], "dx": ["A", "B"]})

        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["city"], 2, {}, _ID_COL, **kwargs
        )

        assert len(output_df) == 2
        assert dq_metrics["n_suppressed"] == 1
        assert_dataframes_equal_unordered(
            expected_output,
            output_df[expected_output.columns],
            ignore_columns=["row_id"],
        )

    def test_ec_assignment_only_by_qids(self, kwargs):
        """
        Test 3: EC Assignment Only By QIDs, Not Other Fields

        Grouping must use only the columns given in `qid_cols`;
        non-QID columns are ignored even if they vary.
        """
        # Input data with the same QID value but different non-QID values
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2],
                "city": ["NY", "NY"],
                "dx": ["foo", "bar"],  # Different non-QID values
            }
        )

        # Expected output: all records should be kept since they have the same QID values
        expected_output = pd.DataFrame(
            {_ID_COL: [1, 2], "city": ["NY", "NY"], "dx": ["foo", "bar"]}
        )

        # Run k-anonymization
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["city"], 2, {}, _ID_COL, **kwargs
        )

        # Verify that all records are kept despite different non-QID values
        assert len(output_df) == 2
        assert dq_metrics["n_suppressed"] == 0
        assert_dataframes_equal_unordered(
            expected_output,
            output_df[expected_output.columns],
            ignore_columns=["row_id"],
        )

    def test_user_id_columns_ignored(self, kwargs):
        """
        Test 4: User-ID Columns Are Ignored for k-anonymity

        When a user-ID is present, it must be ignored;
        what matters is row count, not unique users.
        """
        # Input data with user_id column - k-anonymity should count rows, not users
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "user_id": [1, 1, 2, 3],  # 3 unique users
                "dept": ["Sales", "Sales", "Sales", "HR"],
            }
        )

        # Expected output: only Sales records remain (EC size = 3 ≥ k)
        expected_output = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3],
                "user_id": [1, 1, 2],
                "dept": ["Sales", "Sales", "Sales"],
            }
        )

        # Run k-anonymization (don't pass user_id_col to ensure k-anonymity not u-anonymity)
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["dept"], 3, {}, _ID_COL, **kwargs
        )

        # Verify that all Sales records remain based on row count, not user count
        assert len(output_df) == 3
        assert dq_metrics["n_suppressed"] == 1
        assert_dataframes_equal_unordered(
            expected_output,
            output_df[expected_output.columns],
            ignore_columns=["row_id"],
        )

    def test_grouping_uses_all_qid_columns(self, kwargs):
        """
        Test 5: Grouping Uses ALL QID Columns in Combination

        ECs are defined as tuples of all QID columns, not by each individually.
        """
        # Input data with multiple QID columns that must be considered together
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "city": ["NY", "NY", "NY", "Boston"],
                "year": ["2020", "2021", "2020", "2021"],
            }
        )

        # Expected output: only NY records remain (EC size = 3 ≥ k)
        expected_output = pd.DataFrame(
            {_ID_COL: [1, 2, 3], "city": ["NY", "NY", "NY"], "year": ["*", "*", "*"]}
        )

        # Run k-anonymization with multiple QID columns
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["city", "year"], 2, {}, _ID_COL, **kwargs
        )

        # Verify that only records with city=NY AND year=2020 remain
        assert len(output_df) == 3
        assert dq_metrics["n_suppressed"] == 1
        assert_dataframes_equal_unordered(
            expected_output,
            output_df[expected_output.columns],
            ignore_columns=["row_id"],
        )

    def test_k_parameter_enforced_as_at_least(self, kwargs):
        """
        Test 6: k Parameter is Enforced as "At Least"

        Rows are kept only if their EC size is **≥ k**. If k is too large for any EC, all are suppressed.
        """
        # Input data with multiple ECs, none meeting k=4 requirement
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "city": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            }
        )

        # Run k-anonymization with k too large for any EC
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["city"], 4, {}, _ID_COL, **kwargs
        )

        # Verify that all records are suppressed
        assert len(output_df) == 0
        assert dq_metrics["n_suppressed"] == 9

    def test_nulls_in_qids_form_own_ec(self, kwargs):
        """
        Test 7: Nulls in QIDs Form Their Own EC

        If a QID value is None/NaN, those records form their own EC and are counted together.
        """
        # Input data with null values in QID columns
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "city": [None, None, "NY", "NY"],
                "score": [12, 15, 23, 25],
            }
        )

        # Expected output: Both null and non-null ECs should meet k=2
        expected_output = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "city": [None, None, "NY", "NY"],
                "score": [12, 15, 23, 25],
            }
        )

        # Run k-anonymization
        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["city"], 2, {}, _ID_COL, **kwargs
        )

        # Verify that both null and non-null ECs remain
        assert len(output_df) == 4
        assert dq_metrics["n_suppressed"] == 0
        assert_dataframes_equal_unordered(
            expected_output,
            output_df[expected_output.columns],
            ignore_columns=["row_id"],
        )

    def test_k_equals_one_identity_transformation(self, kwargs):
        """
        Test that k=1 preserves input unchanged (identity transformation).

        This test verifies the edge case where k=1, which should result in
        no generalization or suppression since every record forms its own
        equivalence class of size 1, which satisfies k-anonymity.
        """
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3],
                "color": ["blue", "green", "red"],
                "city": ["NY", "LA", "SF"],
            }
        )

        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER, input_df, ["color", "city"], 1, {}, _ID_COL, **kwargs
        )

        # For k=1, input should equal output exactly
        assert_dataframes_equal_unordered(
            input_df, output_df[input_df.columns], ignore_columns=["row_id"]
        )
        assert dq_metrics["n_suppressed"] == 0

    def test_categorical_generalization_with_gtree_intermediate(self, kwargs):
        """
        Test categorical generalization to intermediate node using custom GTree.

        This test creates a hierarchy where "foo" and "bar" are children of "foobar",
        and verifies that when k-anonymization is applied with k=2, both values
        are correctly generalized to their common parent "foobar".
        """
        # Create the same GTree structure as in test_k_anon_implementations.py
        gtree = gtrees.GTree()
        root = gtree.create_node("*")
        gtree.create_node("test", root)
        foobar = gtree.create_node("foobar", root)
        gtree.create_node("foo", foobar)
        gtree.create_node("bar", foobar)
        gtree.update_highest_node_with_value_if()
        gtree.update_descendant_leaf_values_if()
        gtree.update_lowest_node_with_descendant_leaves_if()
        gtree.add_default_geometric_sizes()
        qid_to_gtree = {"gh_test_col": gtree}

        # Input data with foo and bar that should generalize to foobar
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2],
                "gh_test_col": ["foo", "bar"],
            }
        )

        # Expected output: both should be generalized to foobar
        expected_df = pd.DataFrame(
            {
                _ID_COL: [1, 2],
                "gh_test_col": ["foobar", "foobar"],
            }
        )

        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["gh_test_col"],
            2,
            qid_to_gtree,
            _ID_COL,
            dq_metric_to_minimum_dq=default_dq_metric_to_minimum_dq(),
            **kwargs,
        )

        # Verify generalization to intermediate node
        assert len(output_df) == 2
        assert dq_metrics["n_suppressed"] == 0
        assert_dataframes_equal_unordered(
            expected_df, output_df[expected_df.columns], ignore_columns=["row_id"]
        )

    def test_categorical_generalization_with_gtree_root(self, kwargs):
        """
        Test that all records are suppressed because when all qid values are generalized
        to root (for categorical that is *), the rows are suppressed to reflect total
        information loss.
        """
        # Create the same GTree structure as in test_k_anon_implementations.py
        gtree = gtrees.GTree()
        root = gtree.create_node("*")
        gtree.create_node("test", root)
        foobar = gtree.create_node("foobar", root)
        gtree.create_node("foo", foobar)
        gtree.create_node("bar", foobar)
        gtree.update_highest_node_with_value_if()
        gtree.update_descendant_leaf_values_if()
        gtree.update_lowest_node_with_descendant_leaves_if()
        gtree.add_default_geometric_sizes()
        qid_to_gtree = {"gh_test_col": gtree}

        # Input data with foo and test from different branches
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2],
                "gh_test_col": ["foo", "test"],
            }
        )

        output_df, dq_metrics, _ = self._call_anonymize_function(
            _LOGGER,
            input_df,
            ["gh_test_col"],
            2,
            qid_to_gtree,
            _ID_COL,
            dq_metric_to_minimum_dq=default_dq_metric_to_minimum_dq(),
            **kwargs,
        )

        # Verify that k_anonymize wrapper suppresses records when root generalization
        # would violate data quality thresholds (RILM ≈ 0.0 < 0.90 minimum)
        assert len(output_df) == 0
        assert dq_metrics["n_suppressed"] == 2


class TestKAnonymityCategorical(TestKAnonymitySharedCategorical):
    """Concrete test class for k_anonymize with categorical data"""

    __test__ = True  # Override base class to allow test collection
