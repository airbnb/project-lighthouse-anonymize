"""
Tests for miscellaneous data quality metrics
"""

import logging

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.constants import EPSILON
from project_lighthouse_anonymize.data_quality_metrics.misc import (
    compute_average_equivalence_class_metric,
    compute_discernibility_metric,
    compute_suppression_metrics,
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


class TestAverageEquivalenceClassMetric:
    """
    Tests for the average equivalence class metric.
    """

    # pylint: disable=no-self-use

    def test_average_equivalence_class_metric_0(self):
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

        actual = compute_average_equivalence_class_metric(input_df, input_qids, "_anon")
        assert np.isnan(actual), f"{actual} is not {np.nan}"

    def test_average_equivalence_class_metric_1(self):
        """
        Tests for a single equivalence class
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

        expected = 5.0
        actual = compute_average_equivalence_class_metric(input_df, input_qids, "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    @pytest.mark.parametrize("has_nan_values", [True, False])
    def test_average_equivalence_class_metric_2(self, has_nan_values):
        """
        Tests for two equal size equivalence classes

        Also tests that nan values are properly handled when defining equivalence classes.
        """
        values_1 = (
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            if not has_nan_values
            else [1, 1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan]
        )
        input_df = pd.DataFrame(
            data={
                "qid1_orig": values_1,
                "qid2_orig": [1, 1, 1, 1, 1, 2, 2, 2, 2, 3],
                "qid1_anon": values_1,
                "qid2_anon": [1, 1, 1, 1, 1, 2.2, 2.2, 2.2, 2.2, 2.2],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = 5.0
        actual = compute_average_equivalence_class_metric(input_df, input_qids, "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_average_equivalence_class_metric_3(self):
        """
        Tests for two unequal size equivalence classes
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "qid2_orig": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3],
                "qid1_anon": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "qid2_anon": [1, 1, 1, 1, 1, 1, 2.2, 2.2, 2.2, 2.2, 2.2],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = 5.5
        actual = compute_average_equivalence_class_metric(input_df, input_qids, "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )


class TestDiscernibilityMetric:
    """
    Tests for the discernibility metric.
    """

    # pylint: disable=no-self-use

    def test_discernibility_metric_0(self):
        """
        Tests when there are no records.
        """
        gen_input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
                "qid1_anon": [],
                "qid2_anon": [],
            }
        )
        gen_input_df[_ID_COL] = pd.Series(list(range(len(gen_input_df))), dtype=np.dtype("int64"))
        sup_input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
            }
        )
        sup_input_df[_ID_COL] = pd.Series(list(range(len(sup_input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = np.nan
        actual = compute_discernibility_metric(
            gen_input_df, sup_input_df, input_qids, _ID_COL, "_orig", "_anon"
        )
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_discernibility_metric_1(self):
        """
        Tests when there is a single equivalence class for generalized records and no suppressed records.
        """
        gen_input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 4, 1],
                "qid2_orig": [2, 2, 2],
                "qid1_anon": [2, 2, 2],
                "qid2_anon": [2, 2, 2],
            }
        )
        gen_input_df[_ID_COL] = pd.Series(list(range(len(gen_input_df))), dtype=np.dtype("int64"))
        sup_input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
            }
        )
        sup_input_df[_ID_COL] = pd.Series(list(range(len(sup_input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = 9.0  # 3 records so 3^2
        actual = compute_discernibility_metric(
            gen_input_df, sup_input_df, input_qids, _ID_COL, "_orig", "_anon"
        )
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    @pytest.mark.parametrize("has_nan_values", [True, False])
    def test_discernibility_metric_2(self, has_nan_values):
        """
        Tests when there are two equivalence classes for generalized records and no suppressed records.
        Also tests that nan values are properly handled when defining equivalence classes.
        """
        qid1_value = 3 if not has_nan_values else np.nan
        gen_input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 4, 1, qid1_value, qid1_value],
                "qid2_orig": [2, 2, 2, 3, 3],
                "qid1_anon": [2, 2, 2, qid1_value, qid1_value],
                "qid2_anon": [2, 2, 2, 3, 3],
            }
        )
        gen_input_df[_ID_COL] = pd.Series(list(range(len(gen_input_df))), dtype=np.dtype("int64"))
        sup_input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
            }
        )
        sup_input_df[_ID_COL] = pd.Series(list(range(len(sup_input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        dm_equiv_class_1 = 9.0  # 3 records so 3^2
        dm_equiv_class_2 = 4.0  # 2 records so 2^2
        expected = dm_equiv_class_1 + dm_equiv_class_2
        actual = compute_discernibility_metric(
            gen_input_df, sup_input_df, input_qids, _ID_COL, "_orig", "_anon"
        )
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_discernibility_metric_3(self):
        """
        Tests when all records are suppressed
        """
        gen_input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
                "qid1_anon": [],
                "qid2_anon": [],
            }
        )
        gen_input_df[_ID_COL] = pd.Series(list(range(len(gen_input_df))), dtype=np.dtype("int64"))
        sup_input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3, 4],
                "qid2_orig": [1, 2, 3, 4],
            }
        )
        sup_input_df[_ID_COL] = pd.Series(list(range(len(sup_input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = 16.0  # 4 records * size of dataset of 4
        actual = compute_discernibility_metric(
            gen_input_df, sup_input_df, input_qids, _ID_COL, "_orig", "_anon"
        )
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_discernibility_metric_4(self):
        """
        Tests when there is a single equivalence class for generalized records and two suppressed records.
        """
        gen_input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 4, 1],
                "qid2_orig": [2, 2, 2],
                "qid1_anon": [2, 2, 2],
                "qid2_anon": [2, 2, 2],
            }
        )
        gen_input_df[_ID_COL] = pd.Series(list(range(len(gen_input_df))), dtype=np.dtype("int64"))
        sup_input_df = pd.DataFrame(
            data={
                "qid1_orig": [8, 9],
                "qid2_orig": [8, 9],
            }
        )
        sup_input_df[_ID_COL] = pd.Series(list(range(len(sup_input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        dm_generalized = 9.0  # 3 records so 3^2
        dm_suppressed = 10.0  # 2 records * dataset size of 5
        expected = dm_generalized + dm_suppressed
        actual = compute_discernibility_metric(
            gen_input_df, sup_input_df, input_qids, _ID_COL, "_orig", "_anon"
        )
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )


class TestSuppressionMetrics:
    """
    Tests for compute_suppression_metrics function
    """

    # pylint: disable=no-self-use

    def test_suppression_metrics_0(self):
        """
        Tests for no records in either input or output dataframe
        """
        input_df = pd.DataFrame(data={"col1": []})
        anon_df = pd.DataFrame(data={"col1": []})

        expected = {
            "n_non_suppressed": 0,
            "pct_non_suppressed": np.nan,
            "n_suppressed": 0,
            "pct_suppressed": np.nan,
        }
        actual = compute_suppression_metrics(input_df, anon_df)
        _assert_dicts_equal(actual, expected)

    def test_suppression_metrics_1(self):
        """
        Tests for no suppression - all records preserved
        """
        input_df = pd.DataFrame(data={"col1": [1, 2, 3, 4, 5]})
        anon_df = pd.DataFrame(data={"col1": [10, 20, 30, 40, 50]})

        expected = {
            "n_non_suppressed": 5,
            "pct_non_suppressed": 1.0,
            "n_suppressed": 0,
            "pct_suppressed": 0.0,
        }
        actual = compute_suppression_metrics(input_df, anon_df)
        _assert_dicts_equal(actual, expected)

    def test_suppression_metrics_2(self):
        """
        Tests for partial suppression - some records preserved
        """
        input_df = pd.DataFrame(data={"col1": [1, 2, 3, 4, 5]})
        anon_df = pd.DataFrame(data={"col1": [10, 20, 30]})

        expected = {
            "n_non_suppressed": 3,
            "pct_non_suppressed": 0.6,
            "n_suppressed": 2,
            "pct_suppressed": 0.4,
        }
        actual = compute_suppression_metrics(input_df, anon_df)
        _assert_dicts_equal(actual, expected)

    def test_suppression_metrics_3(self):
        """
        Tests for complete suppression - no records preserved
        """
        input_df = pd.DataFrame(data={"col1": [1, 2, 3, 4, 5]})
        anon_df = pd.DataFrame(data={"col1": []})

        expected = {
            "n_non_suppressed": 0,
            "pct_non_suppressed": 0.0,
            "n_suppressed": 5,
            "pct_suppressed": 1.0,
        }
        actual = compute_suppression_metrics(input_df, anon_df)
        _assert_dicts_equal(actual, expected)
