"""
Tests for p_sensitize
"""

import logging
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize import gtrees
from project_lighthouse_anonymize.constants import MAXIMUM_PRECISION_DIGITS
from project_lighthouse_anonymize.wrappers.p_sensitize import p_sensitize
from tests.shared import assert_dataframes_equal_unordered

_LOGGER = logging.getLogger(__name__)


class TestPSensitize:
    """
    Tests for p_sensitize.
    """

    # Not unittest.TestCase so that generators work with nosetests.
    # per "Please note that method generators are not supported in unittest.TestCase subclasses."
    # from https://nose.readthedocs.io/en/latest/writing_tests.html
    # pylint: disable=no-self-use

    def test_p_sensitize_1(self):
        """test case 1: no change needed"""
        input_df = pd.DataFrame(
            {
                "numerical_1": [2, 2],
                "numerical_2": [2, 2],
                "categorical_1": ["bar", "bar"],
                "sensitive_value": ["blue", "green"],
                "row_id": [0, 1],
            }
        )
        qid_cols = ["numerical_1", "numerical_2", "categorical_1"]
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.50,
            "orange": 0.00,
        }
        expected_output_df = pd.DataFrame(
            {
                "numerical_1": [2, 2],
                "numerical_2": [2, 2],
                "categorical_1": ["bar", "bar"],
                "sensitive_value": ["blue", "green"],
                "row_id": [0, 1],
            }
        )
        expected_num_records_perturbated = 0
        actual_output_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER, input_df, qid_cols, sens_attr_col, 2, 2, sens_attr_value_to_prob
        )
        actual_num_records_perturbated = dq_metrics["num_rows_perturbated"]
        assert_dataframes_equal_unordered(
            expected_output_df, actual_output_df, ignore_columns=["row_id"]
        )  # should be unchanged
        assert expected_num_records_perturbated == actual_num_records_perturbated

    def test_p_sensitize_2(self):
        """test case 2: flip color of one record in qid 1's equivalence class to green"""
        input_df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 2, 2],
                "numerical_2": [1, 1, 2, 2],
                "categorical_1": ["foo", "foo", "bar", "bar"],
                "sensitive_value": ["blue", "blue", "blue", "green"],
                "row_id": [0, 1, 2, 3],
            }
        )
        qid_cols = ["numerical_1", "numerical_2", "categorical_1"]
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.50,
            "orange": 0.00,
        }
        expected_output_df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 2, 2],
                "numerical_2": [1, 1, 2, 2],
                "categorical_1": ["foo", "foo", "bar", "bar"],
                "sensitive_value": ["blue", "green", "blue", "green"],
                "row_id": [0, 1, 2, 3],
            }
        )
        expected_num_records_perturbated = 1
        actual_output_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER, input_df, qid_cols, sens_attr_col, 2, 2, sens_attr_value_to_prob
        )
        actual_num_records_perturbated = dq_metrics["num_rows_perturbated"]
        assert_dataframes_equal_unordered(
            expected_output_df, actual_output_df, ignore_columns=["row_id"]
        )
        assert expected_num_records_perturbated == actual_num_records_perturbated

    def test_p_sensitize_3(self):
        """test case 3: flip color of two records in qid 1's equivalence class: one to blue and one to orange"""
        input_df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 1],
                "numerical_2": [1, 1, 1],
                "categorical_1": ["foo", "foo", "foo"],
                "sensitive_value": ["green", "green", "green"],
                "row_id": [0, 1, 2],
            }
        )
        qid_cols = ["numerical_1", "numerical_2", "categorical_1"]
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.20,
            "orange": 0.30,
        }
        expected_output_df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 1],
                "numerical_2": [1, 1, 1],
                "categorical_1": ["foo", "foo", "foo"],
                "sensitive_value": ["blue", "green", "orange"],
                "row_id": [0, 1, 2],
            }
        )
        expected_num_records_perturbated = 2
        actual_output_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER, input_df, qid_cols, sens_attr_col, 3, 3, sens_attr_value_to_prob
        )
        actual_num_records_perturbated = dq_metrics["num_rows_perturbated"]
        assert_dataframes_equal_unordered(
            expected_output_df, actual_output_df, ignore_columns=["row_id"]
        )
        assert expected_num_records_perturbated == actual_num_records_perturbated

    def test_p_sensitize_4(self):
        """test case 4: ensure that p_sensitize is deterministic when a seed is passed"""
        input_df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 1],
                "numerical_2": [1, 1, 1],
                "categorical_1": ["foo", "foo", "foo"],
                "sensitive_value": ["green", "green", "green"],
                "row_id": [0, 1, 2],
            }
        )
        qid_cols = ["numerical_1", "numerical_2", "categorical_1"]
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.00,
            "blue": 0.50,
            "orange": 0.50,
        }
        expected_output_df, expected_dq_metrics, expected_disclosure_metrics = p_sensitize(
            _LOGGER,
            input_df,
            qid_cols,
            sens_attr_col,
            2,
            3,
            sens_attr_value_to_prob,
            seed=41,
        )
        expected_num_records_perturbated = expected_dq_metrics["num_rows_perturbated"]
        for _ in range(100):
            actual_output_df, actual_dq_metrics, actual_disclosure_metrics = p_sensitize(
                _LOGGER,
                input_df,
                qid_cols,
                sens_attr_col,
                2,
                3,
                sens_attr_value_to_prob,
                seed=41,
            )
            actual_num_records_perturbated = actual_dq_metrics["num_rows_perturbated"]
            assert_dataframes_equal_unordered(
                expected_output_df, actual_output_df, ignore_columns=["row_id"]
            )
            assert expected_num_records_perturbated == actual_num_records_perturbated

    def test_p_sensitize_5(self):
        """test case 5: no change needed when there are no QIDs"""
        input_df = pd.DataFrame(
            data={
                "row_id": [0, 0, 0],
                "sensitive_value": ["green", "blue", "orange"],
            }
        )
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.50,
            "orange": 0.00,
        }
        expected_output_df = input_df.copy()
        expected_num_records_perturbated = 0
        actual_output_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER, input_df, [], sens_attr_col, 2, 3, sens_attr_value_to_prob
        )
        actual_num_records_perturbated = dq_metrics["num_rows_perturbated"]
        assert_dataframes_equal_unordered(
            expected_output_df, actual_output_df, ignore_columns=["row_id"]
        )  # should be unchanged
        assert expected_num_records_perturbated == actual_num_records_perturbated

    def test_p_sensitize_6(self):
        """test case 6: flip color of two records when there are no qids: one to blue and one to orange"""
        input_df = pd.DataFrame(
            data={
                "row_id": [0, 0, 0],
                "sensitive_value": ["green", "green", "green"],
            }
        )
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.20,
            "orange": 0.30,
        }
        expected_output_df = pd.DataFrame(
            data={
                "row_id": [0, 0, 0],
                "sensitive_value": ["green", "blue", "orange"],
            }
        )
        expected_num_records_perturbated = 2
        actual_output_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER, input_df, [], sens_attr_col, 3, 3, sens_attr_value_to_prob
        )
        actual_num_records_perturbated = dq_metrics["num_rows_perturbated"]
        assert_dataframes_equal_unordered(
            expected_output_df, actual_output_df, ignore_columns=["row_id"]
        )
        assert expected_num_records_perturbated == actual_num_records_perturbated

    def test_p_sensitize_7(self):
        """test case 7: two equivalence classes, one of which has nan values. two records in each equivalence class, as with test case 6, has to be flipped."""
        input_df = pd.DataFrame(
            data={
                "row_id": [0, 0, 0, 0, 0, 0],
                "col_1": [1, 1, 1, np.nan, np.nan, np.nan],
                "sensitive_value": [
                    "green",
                    "green",
                    "green",
                    "green",
                    "green",
                    "green",
                ],
            }
        )
        sens_attr_col = "sensitive_value"
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.20,
            "orange": 0.30,
        }
        expected_output_df = pd.DataFrame(
            data={
                "row_id": [0, 0, 0, 0, 0, 0],
                "col_1": [1, 1, 1, np.nan, np.nan, np.nan],
                "sensitive_value": [
                    "green",
                    "blue",
                    "orange",
                    "green",
                    "blue",
                    "orange",
                ],
            }
        )
        expected_num_records_perturbated = 4
        actual_output_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER, input_df, ["col_1"], sens_attr_col, 3, 3, sens_attr_value_to_prob
        )
        actual_num_records_perturbated = dq_metrics["num_rows_perturbated"]
        assert_dataframes_equal_unordered(
            expected_output_df, actual_output_df, ignore_columns=["row_id"]
        )
        assert expected_num_records_perturbated == actual_num_records_perturbated

    @pytest.mark.slow
    def test_p_sensitize_8(self):
        """test case 8: large random numerical+categorical dataset for performance testing; note that this test is not run by default!"""
        approx_n_rows = 1_000_000
        target_p = 3
        target_k = 3
        mult = 0.75
        n_equivalence_classes = int((approx_n_rows / target_k) * mult)
        n_cat_values = 2**13
        qid_to_gtree = {}
        for qid in ["col_3", "col_4"]:
            gtree = gtrees.GTree()
            gtree.create_node("*")
            while len(gtree.leaves()) < n_cat_values:
                leaves = gtree.leaves()
                for leaf in leaves:
                    gtree.create_node(str(uuid4()), leaf)
                    gtree.create_node(str(uuid4()), leaf)
            qid_to_gtree[qid] = gtree
        rng = np.random.default_rng(seed=1204890231)  # deterministic across test runs
        sens_attr_value_to_prob = {
            "green": 0.50,
            "blue": 0.20,
            "orange": 0.30,
        }
        input_dfs = []
        col_1_vals = rng.random(n_equivalence_classes).round(MAXIMUM_PRECISION_DIGITS)
        col_2_vals = rng.random(n_equivalence_classes).round(MAXIMUM_PRECISION_DIGITS)
        col_3_vals = rng.choice(
            [qid_to_gtree["col_3"].get_value(leaf) for leaf in qid_to_gtree["col_3"].leaves()],
            n_equivalence_classes,
        )
        col_4_vals = rng.choice(
            [qid_to_gtree["col_4"].get_value(leaf) for leaf in qid_to_gtree["col_4"].leaves()],
            n_equivalence_classes,
        )
        for i in range(n_equivalence_classes):
            min_rows, max_rows = target_k, int(target_k / mult)
            n_rows = rng.integers(min_rows, max_rows + 1)
            input_df = pd.DataFrame(
                data={
                    "col_1": [col_1_vals[i]] * n_rows,
                    "col_2": [col_2_vals[i]] * n_rows,
                    "col_3": [col_3_vals[i]] * n_rows,
                    "col_4": [col_4_vals[i]] * n_rows,
                }
            )
            input_dfs.append(input_df)
        input_df = pd.concat(input_dfs).reset_index(drop=True)
        n_rows = len(input_df)
        input_df["row_id"] = list(range(n_rows))
        input_df["sensitive_value"] = rng.choice(
            list(sens_attr_value_to_prob.keys()), n_rows, replace=True
        )
        result_df, dq_metrics, disclosure_metrics = p_sensitize(
            _LOGGER,
            input_df,
            ["col_1", "col_2", "col_3", "col_4"],
            "sensitive_value",
            target_p,
            target_k,
            sens_attr_value_to_prob,
        )
