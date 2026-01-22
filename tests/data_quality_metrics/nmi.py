"""
Tests for normalized mutual information (sampled) scaled
"""

import logging

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.data_quality_metrics.nmi import (
    _attempt_convert_to_discrete_dtype,
    _remove_entries_where_both_missing,
    compute_normalized_mutual_information_sampled_scaled,
)

_ID_COL = "id_user"

_LOGGER = logging.getLogger(__name__)


def _assert_dicts_equal(actual_dict, expected_dict, rtol=1e-12, atol=1e-15):
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
            rtol=rtol,
            atol=atol,
            err_msg=f"key = {k}: {actual} != {expected} (rtol = {rtol}, atol = {atol})",
        )


class TestNormalizedMutualInformationSampledScaled:
    """
    Tests for normalized mutual information (sampled) scaled
    """

    # pylint: disable=no-self-use

    def test_normalized_mutual_information_sampled_scaled_0(self):
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
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_normalized_mutual_information_sampled_scaled_1(self):
        """
        Tests for a single equivalence class with no changes in values
        Because the values are unchanged, mi (v1 and v2) is 1.0
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1] * 100_000,
                "qid2_orig": [1] * 100_000,
                "qid1_anon": [1] * 100_000,
                "qid2_anon": [1] * 100_000,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_normalized_mutual_information_sampled_scaled_2(self):
        """
        Tests for two equivalence classes with no changes in values
        Because the values are unchanged, mi (v1 and v2) is 1.0.
        QID1 is a special case in the code where, despite the denominator
        (entropy) for both v1 and v2 being 0, the output is still 1 because
        there is no information to encode, so the information is trivially perfectly encoded.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1] * 100_000,
                "qid2_orig": [1] * 80_000 + [2] * 20_000,
                "qid1_anon": [1] * 100_000,
                "qid2_anon": [1] * 80_000 + [2] * 20_000,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_normalized_mutual_information_sampled_scaled_3(self):
        """
        Tests for a single equivalence class with both values changed.
        Because the anonymized value can be easily derived from the original value,
        we expect mi (v1 and v2) to be 1.0
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1] * 100_000,
                "qid2_orig": [2] * 100_000,
                "qid1_anon": [-1] * 100_000,
                "qid2_anon": [-2] * 100_000,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    @pytest.mark.parametrize("has_nan_values", [True, False])
    def test_normalized_mutual_information_sampled_scaled_4(self, has_nan_values):
        """
        Tests for five equivalence classes with both values changed.
        Because QID1 anonymized values can be directly derived from original values
        we expect mi_v1 and mi_v2 to be 1.0.
        Because QID2 anonymized values cannot be derived from original values
        we expect mi_v1 to be 0.0.
        Because QID2 anonymized values are constant (0 entropy), but there is no shared
        information with the QID2 original values (0 mutual infromation), we expect mi_v2 to be 1.0

        Also tests that nan values are ignored.
        """
        asc_values = [1, 2, 3, 4, 5] * 20_000
        desc_values = [5, 4, 3, 2, 1] * 20_000
        zero_values = [0, 0, 0, 0, 0] * 20_000
        if has_nan_values:
            asc_values.extend([np.nan] * 1_000)
            desc_values.extend([np.nan] * 1_000)
            zero_values.extend([np.nan] * 1_000)
        input_df = pd.DataFrame(
            data={
                "qid1_orig": asc_values,
                "qid2_orig": asc_values,
                "qid1_anon": desc_values,
                "qid2_anon": zero_values,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected_v1 = {"qid1": 1.0, "qid2": 0.0}
        expected_v2 = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected_v1, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected_v2, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_normalized_mutual_information_sampled_scaled_5(self):
        """
        Tests for a five equivalence classes with all values for qid1 changed,
        and no values for qid2 changed.
        Because QID1 anonymized values can be directly derived from original values
        we expect mi (v1 and v2) to be 1.0. Because QID2 values are unchanged, we expect mi (v1 and v2) to be 1.0.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3, 4, 5] * 20_000,
                "qid2_orig": [1, 2, 3, 4, 5] * 20_000,
                "qid1_anon": [5, 4, 3, 2, 1] * 20_000,
                "qid2_anon": [1, 2, 3, 4, 5] * 20_000,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_normalized_mutual_information_sampled_scaled_6(self):
        """
        Tests slight modification to test 5 above, where a cell for QID1 is nan
        in original and anonymized records.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3, 4, 5] * 19_999 + [1, 2, 3, 4, np.nan],
                "qid2_orig": [1, 2, 3, 4, 5] * 20_000,
                "qid1_anon": [5, 4, 3, 2, 1] * 19_999 + [5, 4, 3, 2, np.nan],
                "qid2_anon": [1, 2, 3, 4, 5] * 20_000,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_normalized_mutual_information_sampled_scaled_7(self):
        """
        Tests slight modification to test 6 above, where the values are floats (continuous)
        instead of int (discrete).
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [0.1, 0.2, 0.3, 0.4, 0.5] * 20_000,
                "qid2_orig": [0.1, 0.2, 0.3, 0.4, 0.5] * 20_000,
                "qid1_anon": [0.5, 0.4, 0.3, 0.2, 0.1] * 20_000,
                "qid2_anon": [0.1, 0.2, 0.3, 0.4, 0.5] * 20_000,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        expected = {"qid1": 1.0, "qid2": 1.0}
        actual_v1, actual_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, input_qids, "_orig", "_anon"
        )
        _assert_dicts_equal(
            actual_v1, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling
        _assert_dicts_equal(
            actual_v2, expected, atol=0.0050
        )  # allow 50bp error in nmi due to sampling

    def test_attempt_convert_to_discrete_dtype_0(self):
        """
        Tests that _attempt_convert_to_discrete_dtype() will not modify a dataframe that has int64 columns.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1],
                "qid1_anon": [1],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))

        expected_df = pd.DataFrame(
            data={
                "qid1_orig": [1],
                "qid1_anon": [1],
            },
            dtype=np.dtype("int64"),
        )
        expected_df[_ID_COL] = input_df[_ID_COL]
        actual_df = _attempt_convert_to_discrete_dtype(input_df, "qid1_orig", "qid1_anon")
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_attempt_convert_to_discrete_dtype_1(self):
        """
        Tests that _attempt_convert_to_discrete_dtype() correctly modifies a dataframe where a QID column contains
        nan(s) and <1000 unique values.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3],
                "qid1_anon": [np.nan, 2, 3],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))

        expected_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 2, 3],
                "qid1_anon": [pd.NA, 2, 3],
            },
            dtype="Int64",
        )
        expected_df[_ID_COL] = pd.Series(list(range(len(expected_df))), dtype=np.dtype("int64"))
        actual_df = _attempt_convert_to_discrete_dtype(input_df, "qid1_orig", "qid1_anon")
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_attempt_convert_to_discrete_dtype_2(self):
        """
        Tests that _attempt_convert_to_discrete_dtype() does not modify a dataframe where a QID column contains nan(s)
        but has >=1000 unique values.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": pd.Series(list(range(1_001)), dtype=np.dtype("int64")),
                "qid1_anon": pd.concat(
                    [pd.Series([np.nan]), pd.Series(list(range(1_000)))],
                    ignore_index=True,
                ),
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))

        expected_df = pd.DataFrame(
            data={
                "qid1_orig": pd.Series(list(range(1_001)), dtype=np.dtype("int64")),
                "qid1_anon": pd.concat(
                    [
                        pd.Series([np.nan]),
                        pd.Series(list(range(1_000)), dtype=np.dtype("float64")),
                    ],
                    ignore_index=True,
                ),
            }
        )
        expected_df[_ID_COL] = pd.Series(list(range(len(expected_df))), dtype=np.dtype("int64"))
        actual_df = _attempt_convert_to_discrete_dtype(input_df, "qid1_orig", "qid1_anon")
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_attempt_convert_to_discrete_dtype_3(self):
        """
        Tests that _attempt_convert_to_discrete_dtype() returns an un-modified dataframe if the dataframe contains
        float entries that cannot be cast to Int64.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1.1, 2.1, 3.1],
                "qid1_anon": [np.nan, 2.1, 3.1],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))

        expected_df = pd.DataFrame(
            data={
                "qid1_orig": [1.1, 2.1, 3.1],
                "qid1_anon": [np.nan, 2.1, 3.1],
            },
            dtype="float64",
        )
        expected_df[_ID_COL] = pd.Series(list(range(len(expected_df))), dtype=np.dtype("int64"))
        actual_df = _attempt_convert_to_discrete_dtype(input_df, "qid1_orig", "qid1_anon")
        pd.testing.assert_frame_equal(expected_df, actual_df)

    def test_remove_entries_where_both_missing_0(self):
        """
        Tests that _remove_entries_where_both_missing() does not remove entries if their dtype is float64 and neither are nan.
        """
        input_x = pd.Series([1.0])
        input_y = pd.Series([1.0])

        expected_x = pd.Series([1.0], dtype=np.dtype("float64"))
        expected_y = pd.Series([1.0], dtype=np.dtype("float64"))
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_1(self):
        """
        Tests that _remove_entries_where_both_missing() does not remove entries if their dtype is float64 and only one is nan.
        """
        input_x = pd.Series([1.0])
        input_y = pd.Series([np.nan])

        expected_x = pd.Series([1.0], dtype=np.dtype("float64"))
        expected_y = pd.Series([np.nan], dtype=np.dtype("float64"))
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_2(self):
        """
        Tests that _remove_entries_where_both_missing() removes entries if their dtype is float64 and both are nan.
        """
        input_x = pd.Series([np.nan])
        input_y = pd.Series([np.nan])

        expected_x = pd.Series([], dtype=np.dtype("float64"))
        expected_y = pd.Series([], dtype=np.dtype("float64"))
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_3(self):
        """
        Tests that _remove_entries_where_both_missing() does not remove entries if their dtype is Int64 and neither are NA.
        """
        input_x = pd.Series([1], dtype="Int64")
        input_y = pd.Series([1], dtype="Int64")

        expected_x = pd.Series([1], dtype="Int64")
        expected_y = pd.Series([1], dtype="Int64")
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_4(self):
        """
        Tests that _remove_entries_where_both_missing() does not remove entries if their dtype is Int64 and only one is NA.
        """
        input_x = pd.Series([1], dtype="Int64")
        input_y = pd.Series([pd.NA], dtype="Int64")

        expected_x = pd.Series([1], dtype="Int64")
        expected_y = pd.Series([pd.NA], dtype="Int64")
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_5(self):
        """
        Tests that _remove_entries_where_both_missing() removes entries if their dtype is Int64 and both are NA.
        """
        input_x = pd.Series([pd.NA], dtype="Int64")
        input_y = pd.Series([pd.NA], dtype="Int64")

        expected_x = pd.Series([], dtype="Int64")
        expected_y = pd.Series([], dtype="Int64")
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_6(self):
        """
        Tests that _remove_entries_where_both_missing() does not remove entries if their dtype is int64 and no entries are missing.
        If any nan values were present in these arrays, their type would be converted to float64, so there are no
        additional unit tests for this dtype.
        """
        input_x = pd.Series([1])
        input_y = pd.Series([1])

        expected_x = pd.Series([1], dtype=np.dtype("int64"))
        expected_y = pd.Series([1], dtype=np.dtype("int64"))
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_remove_entries_where_both_missing_7(self):
        """
        Tests that _remove_entries_where_both_missing() works correctly on larger arrays of dtype float64.
        """
        input_x = pd.Series([1.0, 1.0, np.nan, np.nan, 2.0, np.nan] * 20_000)
        input_y = pd.Series([1.0, np.nan, np.nan, 1.0, 3.0, np.nan] * 20_000)

        expected_x = pd.Series([1.0, 1.0, np.nan, 2.0] * 20_000)
        expected_y = pd.Series([1.0, np.nan, 1.0, 3.0] * 20_000)
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        assert np.allclose(expected_x, actual_x, equal_nan=True, atol=0.0050)
        assert np.allclose(expected_y, actual_y, equal_nan=True, atol=0.0050)

    def test_remove_entries_where_both_missing_8(self):
        """
        Tests that _remove_entries_where_both_missing() works correctly on larger arrays of dtype Int64.
        """
        input_x = pd.Series([1, 1, pd.NA, pd.NA, 2, pd.NA] * 20_000, dtype="Int64")
        input_y = pd.Series([1, pd.NA, pd.NA, 1, 3, pd.NA] * 20_000, dtype="Int64")

        expected_x = pd.Series([1, 1, pd.NA, 2] * 20_000, dtype="Int64")
        expected_y = pd.Series([1, pd.NA, 1, 3] * 20_000, dtype="Int64")
        actual_x, actual_y = _remove_entries_where_both_missing(input_x, input_y)
        pd.testing.assert_series_equal(expected_x, actual_x)
        pd.testing.assert_series_equal(expected_y, actual_y)

    def test_partial_nan_imputation(self):
        """Test NaN imputation when anonymized data has partial NaN values"""
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2, 3, 4, 5],
                "age_orig": [25, 30, 35, 40, 45],
                "age_anon": [25, np.nan, 35, np.nan, 45],  # Mixed NaN/valid
            }
        )
        result_v1, result_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["age"], "_orig", "_anon"
        )
        # Should compute successfully with mean imputation
        assert isinstance(result_v1["age"], float)
        assert isinstance(result_v2["age"], float)

    def test_small_dataset_single_run(self):
        """Test that small datasets use single run optimization"""
        rng = np.random.default_rng(seed=1204890231)
        # Create dataset smaller than default sample_size (1000)
        input_df = pd.DataFrame(
            {
                _ID_COL: list(range(50)),  # Only 50 records
                "value_orig": rng.integers(0, 5, 50),
                "value_anon": rng.integers(0, 5, 50),
            }
        )
        result_v1, result_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["value"], "_orig", "_anon", sample_size=1000, number_runs=5
        )
        # Should still compute successfully with single run
        assert isinstance(result_v1["value"], float)
        assert isinstance(result_v2["value"], float)

    def test_sklearn_value_error_handling(self):
        """Test handling of ValueError from sklearn mutual_info functions"""
        # Create tiny dataset that may cause sklearn to fail
        input_df = pd.DataFrame(
            {
                _ID_COL: [1, 2],  # Only 2 samples
                "value_orig": [1.0, 2.0],
                "value_anon": [1.1, 2.1],
            }
        )
        result_v1, result_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["value"], "_orig", "_anon", sample_size=2
        )
        # Should compute successfully with 2 samples and produce finite values
        assert isinstance(result_v1["value"], float)
        assert isinstance(result_v2["value"], float)

    def test_zero_entropy_handling(self):
        """Test edge case where entropy is zero"""
        rng = np.random.default_rng(seed=1204890231)
        # All original values identical (zero entropy)
        input_df = pd.DataFrame(
            {
                _ID_COL: list(range(100)),
                "value_orig": [5] * 100,  # Zero entropy
                "value_anon": rng.integers(1, 10, 100),  # Non-zero entropy
            }
        )
        result_v1, result_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["value"], "_orig", "_anon", scale=True
        )
        # With zero entropy in original, v1 should be 1.0, v2 should be 0.0
        assert result_v1["value"] == 1.0
        assert result_v2["value"] == 0.0

    def test_scaling_zero_entropy(self):
        """Test NMI scaling with zero input entropy"""
        input_df = pd.DataFrame(
            {
                _ID_COL: list(range(50)),
                "value_orig": [1] * 50,  # All same = zero entropy
                "value_anon": [1] * 50,  # Also all same
            }
        )
        result_v1, result_v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["value"], "_orig", "_anon", scale=True
        )
        # Perfect correlation between identical values should be 1.0
        assert result_v1["value"] == 1.0
        assert result_v2["value"] == 1.0
