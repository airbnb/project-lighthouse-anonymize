"""
Tests that NMI computations for one QID are not affected by other QIDs.
"""

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.data_quality_metrics.nmi import (
    compute_normalized_mutual_information_sampled_scaled,
)


class TestNMIPerQidIsolation:
    """Tests that per-QID NMI results are independent of the other QIDs processed"""

    @staticmethod
    def make_input_df():
        """Two numerical QIDs: 'a' with only 5 non-missing rows, 'b' fully populated"""
        rng = np.random.default_rng(7)
        n = 60
        a_orig = np.full(n, np.nan)
        a_anon = np.full(n, np.nan)
        a_orig[:5] = rng.random(5) * 10
        a_anon[:5] = a_orig[:5] + rng.random(5) * 0.1
        b_orig = rng.random(n) * 100
        b_anon = b_orig + rng.random(n)
        return pd.DataFrame(
            {
                "a_orig": a_orig,
                "a_anon": a_anon,
                "b_orig": b_orig,
                "b_anon": b_anon,
            }
        )

    def test_qid_result_independent_of_preceding_qid(self):
        """A QID's NMI must be the same whether or not a smaller QID precedes it.

        QID 'a' has fewer non-missing rows than sample_size. Processing it must
        not reduce the sample_size/number_runs used for QID 'b'.
        """
        input_df = self.make_input_df()
        v1_both, v2_both = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["a", "b"], "_orig", "_anon", sample_size=20, number_runs=5
        )
        v1_b_only, v2_b_only = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["b"], "_orig", "_anon", sample_size=20, number_runs=5
        )
        assert v1_both["b"] == pytest.approx(v1_b_only["b"])
        assert v2_both["b"] == pytest.approx(v2_b_only["b"])

    def test_failed_mi_computation_yields_nan_not_zero(self):
        """When sklearn mutual information raises (n_samples < n_neighbors), the
        result must be NOT_DEFINED_NA (NaN), not silently coerced to 0.0."""
        input_df = pd.DataFrame(
            {
                # 2 continuous samples < default n_neighbors=3 used by
                # sklearn.feature_selection.mutual_info_regression
                "a_orig": [1.5, 2.5],
                "a_anon": [1.6, 2.7],
            }
        )
        v1, v2 = compute_normalized_mutual_information_sampled_scaled(
            input_df, ["a"], "_orig", "_anon"
        )
        assert np.isnan(v1["a"])
        assert np.isnan(v2["a"])
