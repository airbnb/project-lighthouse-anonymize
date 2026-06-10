"""
Tests for disclosure risk metrics on categorical dtype columns.

Categorical columns are outside the library's documented dtype contract
(object/int64/float64); calculate_p_k and compute_entropy_log_l_diversity
reject them up front with conversion guidance instead of silently producing
phantom empty equivalence classes for unobserved categories.
"""

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
    """Tests that categorical dtype columns are rejected with clear guidance"""

    @staticmethod
    def categorical_qid_df():
        """2-anonymous, 2-sensitive frame with an unobserved QID category"""
        return pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "b", "a", "b"],
            }
        )

    def test_categorical_qid_rejected(self):
        """Categorical QID columns fail validation with conversion guidance"""
        with pytest.raises(ValueError, match="categorical"):
            calculate_p_k(self.categorical_qid_df(), ["qid"], "sens")

    def test_categorical_qid_k_only_rejected(self):
        """Categorical QID columns are rejected even when sens_attr is not provided"""
        with pytest.raises(ValueError, match="categorical"):
            calculate_p_k(self.categorical_qid_df(), ["qid"])

    def test_categorical_sensitive_attr_rejected(self):
        """Categorical sensitive attribute columns fail validation with conversion guidance"""
        df = pd.DataFrame(
            {
                "qid": ["x", "x", "y", "y"],
                "sens": pd.Categorical(["a", "b", "a", "b"], categories=["a", "b", "c"]),
            }
        )
        with pytest.raises(ValueError, match="categorical"):
            calculate_p_k(df, ["qid"], "sens")


class TestEntropyLDiversityCategorical:
    """Tests that categorical dtype columns are rejected with clear guidance"""

    def test_categorical_sensitive_column_rejected(self):
        """Categorical sensitive attribute columns fail validation with conversion guidance"""
        df = pd.DataFrame(
            {
                "qid": ["x", "x", "y", "y"],
                "sens": pd.Categorical(["a", "b", "a", "a"], categories=["a", "b", "c"]),
            }
        )
        with pytest.raises(ValueError, match="categorical"):
            compute_entropy_log_l_diversity(df, ["qid"], "sens")

    def test_categorical_qid_column_rejected(self):
        """Categorical QID columns fail validation with conversion guidance"""
        df = pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "b", "a", "b"],
            }
        )
        with pytest.raises(ValueError, match="categorical"):
            compute_entropy_log_l_diversity(df, ["qid"], "sens")


class TestEntropyLDiversityEdgeCases:
    """Tests for entropy l-diversity edge cases"""

    def test_empty_qids_single_equivalence_class(self):
        """No QIDs means one whole-dataset equivalence class, like calculate_p_k"""
        import math

        df = pd.DataFrame({"sens": ["a", "b", "a", "b"]})
        avg, minimum, maximum = compute_entropy_log_l_diversity(df, [], "sens")
        assert avg == pytest.approx(math.log(2))
        assert minimum == pytest.approx(math.log(2))
        assert maximum == pytest.approx(math.log(2))

    def test_all_nan_sensitive_class_excluded(self):
        """A class with only NaN sensitive values has undefined entropy, not 0.0"""
        import math

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
