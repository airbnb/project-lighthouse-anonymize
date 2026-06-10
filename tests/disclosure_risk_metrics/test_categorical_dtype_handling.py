"""
Tests for disclosure risk metrics on categorical dtype columns.

pandas groupby defaults to observed=False for categorical columns, producing
phantom empty equivalence classes for unobserved categories. Empty classes
contain no individuals and must not influence disclosure risk metrics:
calculate_p_k must not report p=0/k=None, and entropy l-diversity must not
report NaN or worst-case 0.0 entropy for classes that do not exist.
"""

import math

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.disclosure_risk_metrics.l_diversity import (
    compute_entropy_log_l_diversity,
)
from project_lighthouse_anonymize.disclosure_risk_metrics.p_sensitive_k_anonymity import (
    calculate_p_k,
)


class TestCalculatePKCategorical:
    """Tests for calculate_p_k with categorical QID columns"""

    @staticmethod
    def categorical_qid_df():
        """2-anonymous, 2-sensitive frame with an unobserved QID category"""
        return pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "b", "a", "b"],
            }
        )

    def test_categorical_qid_with_unobserved_category(self):
        """Unobserved categories must not produce phantom empty equivalence classes"""
        actual_p, actual_k = calculate_p_k(self.categorical_qid_df(), ["qid"], "sens")
        assert (actual_p, actual_k) == (2, 2), f"got ({actual_p}, {actual_k})"

    def test_categorical_qid_k_only(self):
        """k computation alone must also ignore phantom groups"""
        _, actual_k = calculate_p_k(self.categorical_qid_df(), ["qid"])
        assert actual_k == 2, f"got {actual_k}"


class TestEntropyLDiversityCategorical:
    """Tests for entropy l-diversity with categorical dtype columns"""

    def test_categorical_sensitive_column(self):
        """Unobserved sensitive categories (count 0) must not poison entropy with NaN"""
        df = pd.DataFrame(
            {
                "qid": ["x", "x", "y", "y"],
                "sens": pd.Categorical(["a", "b", "a", "a"], categories=["a", "b", "c"]),
            }
        )
        avg, minimum, maximum = compute_entropy_log_l_diversity(df, ["qid"], "sens")
        assert minimum == pytest.approx(0.0)
        assert maximum == pytest.approx(math.log(2))
        assert avg == pytest.approx(math.log(2) / 2)

    def test_categorical_qid_column(self):
        """Phantom empty q*-blocks must not contribute worst-case 0.0 entropy"""
        df = pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "b", "a", "b"],
            }
        )
        avg, minimum, maximum = compute_entropy_log_l_diversity(df, ["qid"], "sens")
        assert minimum == pytest.approx(math.log(2))
        assert maximum == pytest.approx(math.log(2))
        assert avg == pytest.approx(math.log(2))


class TestEntropyLDiversityEdgeCases:
    """Tests for entropy l-diversity edge cases"""

    def test_empty_qids_single_equivalence_class(self):
        """No QIDs means one whole-dataset equivalence class, like calculate_p_k"""
        df = pd.DataFrame({"sens": ["a", "b", "a", "b"]})
        avg, minimum, maximum = compute_entropy_log_l_diversity(df, [], "sens")
        assert avg == pytest.approx(math.log(2))
        assert minimum == pytest.approx(math.log(2))
        assert maximum == pytest.approx(math.log(2))

    def test_all_nan_sensitive_class_excluded(self):
        """A class with only NaN sensitive values has undefined entropy, not 0.0"""
        df = pd.DataFrame(
            {
                "qid": [1, 1, 2, 2],
                "sens": ["a", "b", np.nan, np.nan],
            }
        )
        avg, minimum, maximum = compute_entropy_log_l_diversity(df, ["qid"], "sens")
        assert minimum == pytest.approx(math.log(2))
        assert maximum == pytest.approx(math.log(2))
        assert avg == pytest.approx(math.log(2))
