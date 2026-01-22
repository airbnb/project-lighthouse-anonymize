"""
Tests for validate
"""

import logging

import pandas as pd

from project_lighthouse_anonymize.disclosure_risk_metrics import calculate_p_k

_LOGGER = logging.getLogger(__name__)


class TestK:
    """
    Tests for calculate_p_k for k.
    """

    # pylint: disable=missing-function-docstring,no-self-use

    @property
    def qid_cols(self):
        return ["numerical_1", "numerical_2", "categorical_1"]

    @property
    def sens_attr_col(self):
        return "sensitive_value"

    def test_k_1(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1],
                "numerical_2": [0],
                "categorical_1": ["foo"],
                "sensitive_value": ["blue"],
                "row_id": [1],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[1] == 1

    def test_k_2(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 2],
                "numerical_2": [0, 0],
                "categorical_1": ["foo", "bar"],
                "sensitive_value": ["blue", "blue"],
                "row_id": [1, 2],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[1] == 1

    def test_k_3(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 2],
                "numerical_2": [0, 0, 0],
                "categorical_1": ["foo", "foo", "bar"],
                "sensitive_value": ["blue", "blue", "blue"],
                "row_id": [1, 1, 2],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[1] == 1

    def test_k_4(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 1],
                "numerical_2": [0, 0],
                "categorical_1": ["foo", "foo"],
                "sensitive_value": ["blue", "blue"],
                "row_id": [1, 1],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[1] == 2

    def test_k_5(self):
        df = pd.DataFrame(
            {
                "numerical_1": [2, 1, 2, 1, 1],
                "numerical_2": [0, 0, 0, 0, 0],
                "categorical_1": ["bar", "foo", "bar", "foo", "foo"],
                "sensitive_value": ["blue", "blue", "blue", "blue", "blue"],
                "row_id": [2, 1, 2, 1, 1],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[1] == 2

    def test_k_6a(self):
        df = pd.DataFrame(
            {
                "numerical_1": [2, 1, 2, 1],
                "numerical_2": [0, 0, 0, 0],
                "categorical_1": ["bar", "foo", "bar", "foo"],
                "sensitive_value": ["blue", "blue", "blue", "blue"],
                "row_id": [2, 1, 2, 1],
            }
        )
        assert calculate_p_k(df, [], self.sens_attr_col)[1] == 4

    def test_k_6b(self):
        df = pd.DataFrame(
            {
                "numerical_1": [2, 1, 2, 1, 1],
                "numerical_2": [0, 0, 0, 0, 0],
                "categorical_1": ["bar", "foo", "bar", "foo", "foo"],
                "sensitive_value": ["blue", "blue", "blue", "blue", "blue"],
                "row_id": [2, 1, 2, 1, 1],
            }
        )
        assert calculate_p_k(df, [], self.sens_attr_col)[1] == 5

    def test_k_7a(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1],
                "numerical_2": [0],
                "categorical_1": ["foo"],
                "sensitive_value": ["blue"],
                "row_id": [1],
            }
        )
        assert calculate_p_k(df, [])[1] == 1

    def test_k_7b(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 1],
                "numerical_2": [0, 0, 0],
                "categorical_1": ["foo", "foo", "foo"],
                "sensitive_value": ["blue", "blue", "blue"],
                "row_id": [1, 1, 1],
            }
        )
        assert calculate_p_k(df, [])[1] == 3


class TestP:
    """
    Tests for calculate_p_k for p.
    """

    # pylint: disable=missing-function-docstring,no-self-use

    @property
    def qid_cols(self):
        return ["numerical_1", "numerical_2", "categorical_1"]

    @property
    def sens_attr_col(self):
        return "sensitive_value"

    def test_p_1(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1],
                "numerical_2": [0],
                "categorical_1": ["foo"],
                "sensitive_value": ["blue"],
                "row_id": [1],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[0] == 1

    def test_p_2(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 2],
                "numerical_2": [0, 0],
                "categorical_1": ["foo", "bar"],
                "sensitive_value": ["blue", "blue"],
                "row_id": [1, 2],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[0] == 1

    def test_p_3(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 2],
                "numerical_2": [0, 0, 0],
                "categorical_1": ["foo", "foo", "bar"],
                "sensitive_value": ["blue", "green", "blue"],
                "row_id": [1, 1, 2],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[0] == 1

    def test_p_4(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 1],
                "numerical_2": [0, 0],
                "categorical_1": ["foo", "foo"],
                "sensitive_value": ["blue", "green"],
                "row_id": [1, 1],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[0] == 2

    def test_p_5(self):
        df = pd.DataFrame(
            {
                "numerical_1": [2, 1, 2, 1, 1],
                "numerical_2": [0, 0, 0, 0, 0],
                "categorical_1": ["bar", "foo", "bar", "foo", "foo"],
                "sensitive_value": ["green", "blue", "blue", "green", "blue"],
                "row_id": [2, 1, 2, 1, 1],
            }
        )
        assert calculate_p_k(df, self.qid_cols, self.sens_attr_col)[0] == 2

    def test_p_6a(self):
        df = pd.DataFrame(
            {
                "numerical_1": [1, 1, 2],
                "numerical_2": [0, 0, 0],
                "categorical_1": ["foo", "foo", "bar"],
                "sensitive_value": ["blue", "green", "blue"],
                "row_id": [1, 1, 2],
            }
        )
        assert calculate_p_k(df, [], self.sens_attr_col)[0] == 2

    def test_p_6b(self):
        df = pd.DataFrame(
            {
                "numerical_1": [2, 1, 2, 1, 1],
                "numerical_2": [0, 0, 0, 0, 0],
                "categorical_1": ["bar", "foo", "bar", "foo", "foo"],
                "sensitive_value": ["green", "blue", "blue", "green", "blue"],
                "row_id": [2, 1, 2, 1, 1],
            }
        )
        assert calculate_p_k(df, [], self.sens_attr_col)[0] == 2

    def test_sens_attr_in_qids_removal(self):
        """Test removal of sens_attr from qids when it appears in both"""
        input_df = pd.DataFrame({"qid1": [1, 1, 2, 2], "sens": ["A", "B", "A", "B"]})
        # Include sens_attr in qids - should be automatically removed
        result_p, result_k = calculate_p_k(input_df, qids=["qid1", "sens"], sens_attr="sens")
        assert result_p == 2
        assert result_k == 2

    def test_no_qids_no_sens_attr(self):
        """Test handling when no qids and no sens_attr"""
        input_df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["A", "B", "C", "D"]})
        # No qids, no sens_attr - should set actual_p = None
        result_p, result_k = calculate_p_k(input_df, qids=[], sens_attr=None)
        assert result_p is None
        assert result_k == 4


class TestErrorHandling:
    """Test error handling for calculate_p_k function"""

    def test_empty_dataframe_raises_valueerror(self):
        """Test that empty DataFrame raises ValueError"""
        import pytest

        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input dataframe has no rows"):
            calculate_p_k(empty_df, qids=["col1"])

    def test_invalid_qid_column_raises_valueerror(self):
        """Test that invalid QID column raises ValueError"""
        import pytest

        df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        with pytest.raises(
            ValueError,
            match="QID col \\(nonexistent\\) is not a column in the input dataframe",
        ):
            calculate_p_k(df, qids=["nonexistent"])

    def test_invalid_sens_attr_column_raises_valueerror(self):
        """Test that invalid sensitive attribute column raises ValueError"""
        import pytest

        df = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        with pytest.raises(
            ValueError,
            match="Sensitive attribute col \\(nonexistent\\) is not a column in the input dataframe",
        ):
            calculate_p_k(df, qids=["col1"], sens_attr="nonexistent")
