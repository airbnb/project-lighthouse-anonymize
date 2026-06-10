"""
Tests for NMI dtype handling around the Int64 discrete-conversion path.

The Int64 conversion hack and its downstream handling must not corrupt data
(truncating genuinely fractional anonymized values to int), crash on
realistic suppression patterns (fractional means imputed into Int64,
all-NaN QIDs), or mutate the caller's DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.data_quality_metrics.nmi import (
    compute_normalized_mutual_information_sampled_scaled,
)


class TestNMIInt64Truncation:
    """Tests that the Int64 conversion does not truncate fractional values"""

    def test_fractional_anon_values_not_truncated(self):
        """One suppressed cell must not collapse fractional anon values to zeros

        With an int64 orig column and a fractional float anon column containing
        a single locally suppressed (NaN) cell, casting both columns to int64
        truncates every anon value toward zero, reporting NMIv1 = 0.0 (complete
        information loss) for a nearly perfect anonymization.
        """
        orig = np.arange(30, dtype=np.int64)
        anon = orig.astype(float) / 30 * 0.9 + 0.05
        anon[3] = np.nan
        df = pd.DataFrame({"q_o": orig, "q_a": anon})
        mis_1, mis_2 = compute_normalized_mutual_information_sampled_scaled(
            df, ["q"], "_o", "_a", scale=False
        )
        assert mis_1["q"] > 0.5, f"NMIv1 = {mis_1['q']}"
        assert mis_2["q"] > 0.5, f"NMIv2 = {mis_2['q']}"


class TestNMIInt64Imputation:
    """Tests imputation of locally suppressed values under Int64 dtypes"""

    def test_fractional_mean_imputed_into_integer_valued_anon(self):
        """A fractional mean imputed into an integer-valued anon column must not crash"""
        df = pd.DataFrame(
            {
                "q_o": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
                "q_a": [1.0, 1.0, 4.0, 4.0, np.nan, np.nan],
            }
        )
        mis_1, mis_2 = compute_normalized_mutual_information_sampled_scaled(df, ["q"], "_o", "_a")
        assert "q" in mis_1 and "q" in mis_2


class TestNMIAllNaNColumns:
    """Tests for QIDs whose values are entirely NaN"""

    def test_all_nan_qid_does_not_crash_other_qids(self):
        """An all-NaN QID maps to NaN; other QIDs are still computed"""
        df = pd.DataFrame(
            {
                "a_o": [np.nan] * 5,
                "a_a": [np.nan] * 5,
                "b_o": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b_a": [1.5, 1.5, 3.5, 3.5, 5.0],
            }
        )
        mis_1, mis_2 = compute_normalized_mutual_information_sampled_scaled(
            df, ["a", "b"], "_o", "_a"
        )
        assert np.isnan(mis_1["a"]) and np.isnan(mis_2["a"])
        assert not np.isnan(mis_1["b"]) and not np.isnan(mis_2["b"])


class TestNMIFullySuppressedAnon:
    """Tests for anon columns whose values are entirely NaN (fully suppressed)"""

    def test_fully_suppressed_anon_reports_complete_loss(self):
        """A fully suppressed anon column reports NMIv1=0.0 and NMIv2=1.0

        The anonymized column carries zero information, so the constant-column
        convention applies: complete information loss (failing data quality
        thresholds) rather than NaN, which the wrapper's nanmin aggregation
        would silently drop, certifying the destroyed attribute as passing.
        """
        df = pd.DataFrame(
            {
                "q_o": np.arange(10, dtype="float64") + 0.5,
                "q_a": [np.nan] * 10,
            }
        )
        mis_1, mis_2 = compute_normalized_mutual_information_sampled_scaled(df, ["q"], "_o", "_a")
        assert mis_1["q"] == pytest.approx(0.0)
        assert mis_2["q"] == pytest.approx(1.0)

    def test_fully_suppressed_constant_orig_reports_no_information(self):
        """A fully suppressed anon column with a constant orig column reports 1.0

        With H(X) = 0 there is no information to encode, so the 0/0 special
        case applies to both NMI versions.
        """
        df = pd.DataFrame({"q_o": [2.5, 2.5, 2.5], "q_a": [np.nan] * 3})
        mis_1, mis_2 = compute_normalized_mutual_information_sampled_scaled(df, ["q"], "_o", "_a")
        assert mis_1["q"] == pytest.approx(1.0)
        assert mis_2["q"] == pytest.approx(1.0)


class TestNMICallerIsolation:
    """Tests that NMI computation does not mutate the caller's DataFrame"""

    def test_caller_dataframe_dtypes_unchanged(self):
        """Column dtypes of the input frame must be unchanged after the call"""
        df = pd.DataFrame(
            {
                "q_o": [1.0, 2.0, 3.0, 4.0, np.nan],
                "q_a": [1.0, 2.0, 3.0, 4.0, np.nan],
            }
        )
        dtypes_before = df.dtypes.copy()
        compute_normalized_mutual_information_sampled_scaled(df, ["q"], "_o", "_a")
        assert df.dtypes.equals(dtypes_before), f"{dtypes_before} -> {df.dtypes}"
