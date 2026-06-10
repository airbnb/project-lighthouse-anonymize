"""
Tests for p_sensitize with categorical QID columns and probability validation.

Categorical QIDs produce phantom empty groups under groupby(observed=False),
which crash perturbation (rng.choice on an empty index) and bypass the
documented k-anonymity guard (calculate_p_k returns k=None). Validation must
also count only sensitive values with non-zero probability toward target_p
feasibility, since zero-probability values can never be selected.
"""

import logging

import pandas as pd
import pytest

from project_lighthouse_anonymize.p_sensitize import p_sensitize

LOGGER = logging.getLogger(__name__)


class TestPSensitizeCategoricalQids:
    """Tests for p_sensitize with categorical QID columns"""

    def test_categorical_qid_with_unobserved_category(self):
        """Categorical QIDs with unobserved categories must p-sensitize successfully"""
        input_df = pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "a", "b", "b"],
            }
        )
        output_df, num_perturbated = p_sensitize(
            LOGGER,
            input_df,
            ["qid"],
            "sens",
            2,
            2,
            {"a": 0.4, "b": 0.4, "c": 0.2},
            seed=0,
        )
        assert num_perturbated == 2
        for _, equivalence_class_df in output_df.groupby("qid", observed=True):
            assert equivalence_class_df["sens"].nunique() >= 2

    def test_categorical_qid_k_guard_enforced(self):
        """The k guard must reject non-k-anonymous categorical input"""
        input_df = pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "a", "b"],
            }
        )
        with pytest.raises(ValueError, match="Actual k"):
            p_sensitize(
                LOGGER,
                input_df,
                ["qid"],
                "sens",
                2,
                2,
                {"a": 0.4, "b": 0.4, "c": 0.2},
                seed=0,
            )


class TestPSensitizeZeroProbabilityValidation:
    """Tests for validation of zero-probability sensitive values"""

    def test_zero_probability_values_not_counted_toward_target_p(self):
        """target_p exceeding the non-zero-probability value count fails validation

        Values with probability 0.0 can never be selected by perturbation
        (rng.choice with replace=False raises), so they cannot help reach
        target_p; validation must reject this up front with a clear error.
        """
        input_df = pd.DataFrame({"qid": [1, 1, 1], "sens": ["green"] * 3})
        with pytest.raises(ValueError, match="non-zero probability"):
            p_sensitize(
                LOGGER,
                input_df,
                ["qid"],
                "sens",
                3,
                3,
                {"green": 0.5, "blue": 0.5, "orange": 0.0},
                seed=1,
            )

    def test_achievable_target_p_with_zero_probability_value_present(self):
        """A zero-probability value in the priors is fine when target_p is achievable"""
        input_df = pd.DataFrame({"qid": [1, 1, 1], "sens": ["green"] * 3})
        output_df, num_perturbated = p_sensitize(
            LOGGER,
            input_df,
            ["qid"],
            "sens",
            2,
            3,
            {"green": 0.5, "blue": 0.5, "orange": 0.0},
            seed=1,
        )
        assert num_perturbated == 1
        assert output_df["sens"].nunique() == 2
        assert "orange" not in set(output_df["sens"])
