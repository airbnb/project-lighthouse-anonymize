"""
Tests for Pearson's correlation coefficient
"""

import logging

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.constants import EPSILON
from project_lighthouse_anonymize.data_quality_metrics.pearson import (
    compute_pearsons_correlation_coefficients,
)

_ID_COL = "id_user"

_LOGGER = logging.getLogger(__name__)


def _assert_dicts_equal(actual_dict, expected_dict, atol=EPSILON):
    """
    Small helper function for comparing two dicts of floats.
    """
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{actual_dict.keys()} != {expected_dict.keys()}"
    )
    for k in actual_dict.keys():
        actual = actual_dict[k]
        expected = expected_dict[k]
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            err_msg=f"key = {k}: {actual} != {expected} (atol = {atol})",
        )


class TestPearsonsCorrelationCoefficient:
    """
    Tests for Pearson's correlation coefficient
    """

    # pylint: disable=no-self-use

    def test_pearsons_correlation_coefficients_0(self):
        """
        Tests for no records
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
                "qid1_anon": [],
                "qid2_anon": [],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": np.nan, "qid2": np.nan}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_pearsons_correlation_coefficients_1(self):
        """
        Tests for a single equivalence class with no changes in values
        Because there is only a single unchanged value for each QID, Pearson's is 1.0.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 1, 1, 1, 1],
                "qid2_orig": [1, 1, 1, 1, 1],
                "qid1_anon": [1, 1, 1, 1, 1],
                "qid2_anon": [1, 1, 1, 1, 1],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_pearsons_correlation_coefficients_2(self):
        """
        Tests for a two equivalence classes with no changes in values
        Because there is only a single unchanged value for qid1, Pearson's is 1.0.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 1, 1, 1, 1],
                "qid2_orig": [1, 1, 1, 2, 2],
                "qid1_anon": [1, 1, 1, 1, 1],
                "qid2_anon": [1, 1, 1, 2, 2],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_pearsons_correlation_coefficients_3(self):
        """
        Tests for a single equivalence class with both values changed.
        Because there is only a single changed value for each QID, Pearson's is 0.0.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 1, 1, 1, 1],
                "qid2_orig": [2, 2, 2, 2, 2],
                "qid1_anon": [-1, -1, -1, -1, -1],
                "qid2_anon": [-2, -2, -2, -2, -2],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 0.0, "qid2": 0.0}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_pearsons_correlation_coefficients_4(self):
        """
        Tests for a five equivalence classes with both values changed.
        Because there is a single value for anonymized qid2 but multiple original values, Pearson's is 0.0.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3, 4, 5],
                "qid2_orig": [1, 2, 3, 4, 5],
                "qid1_anon": [5, 4, 3, 2, 1],
                "qid2_anon": [0, 0, 0, 0, 0],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 0.0, "qid2": 0.0}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_pearsons_correlation_coefficients_6(self):
        """
        Tests for a five equivalence classes with all values for qid1 changed,
        and no values for qid2 changed.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3, 4, 5],
                "qid2_orig": [1, 2, 3, 4, 5],
                "qid1_anon": [5, 4, 3, 2, 1],
                "qid2_anon": [1, 2, 3, 4, 5],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 0.0, "qid2": 1.0}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_pearsons_correlation_coefficients_7(self):
        """
        Tests slight modification to test 6 above, where a cell for QID1 is nan
        in original and anonymized records. This tests that nan values are ignored.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3, 4, np.nan],
                "qid2_orig": [1, 2, 3, 4, 5],
                "qid1_anon": [5, 4, 3, 2, np.nan],
                "qid2_anon": [1, 2, 3, 4, 5],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 0.0, "qid2": 1.0}
        actual = compute_pearsons_correlation_coefficients(input_df, input_qids, "_orig", "_anon")
        _assert_dicts_equal(actual, expected)

    def test_nan_mean_imputation(self):
        """Test NaN imputation with mean when anonymized data has partial NaNs"""
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4, 5],
                "value_orig": [10, 20, 30, 40, 50],
                "value_anon": [10, np.nan, 30, np.nan, 50],  # Partial NaN
            }
        )
        result = compute_pearsons_correlation_coefficients(input_df, ["value"], "_orig", "_anon")
        # Should use mean imputation and compute correlation
        assert isinstance(result["value"], float)
        assert not np.isnan(result["value"])

    def test_insufficient_variance(self):
        """Test handling when one column has insufficient variance"""

        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4],
                "value_orig": [10, 20, 30, 40],  # Has variance
                "value_anon": [5, 5, 5, 5],  # No variance (single value)
            }
        )
        result = compute_pearsons_correlation_coefficients(input_df, ["value"], "_orig", "_anon")
        assert result["value"] == 0.0
