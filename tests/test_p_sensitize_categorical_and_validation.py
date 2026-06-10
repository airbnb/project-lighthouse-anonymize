"""
Tests for p_sensitize dtype validation and probability-prior feasibility.

Categorical columns are outside the library's documented dtype contract
(object/int64/float64); p_sensitize rejects them up front with conversion
guidance instead of crashing in perturbation with an opaque error. Sensitive
values with zero probability can never be selected by perturbation, but values
already present in an equivalence class count toward its p for free — so
feasibility is validated globally against all prior values and per class
against the values that must actually be added.
"""

import logging

import pandas as pd
import pytest

from project_lighthouse_anonymize.p_sensitize import p_sensitize

LOGGER = logging.getLogger(__name__)


class TestPSensitizeCategoricalRejection:
    """Tests that categorical dtype columns are rejected with clear guidance"""

    def test_categorical_qid_rejected(self):
        """Categorical QID columns fail validation with conversion guidance"""
        input_df = pd.DataFrame(
            {
                "qid": pd.Categorical(["x", "x", "y", "y"], categories=["x", "y", "z"]),
                "sens": ["a", "a", "b", "b"],
            }
        )
        with pytest.raises(ValueError, match="categorical"):
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

    def test_categorical_sensitive_column_rejected(self):
        """Categorical sensitive columns fail validation with conversion guidance

        Perturbation assigns replacement values that may not be existing
        categories, which raises TypeError seed-dependently; rejecting up
        front makes the failure deterministic and actionable.
        """
        input_df = pd.DataFrame(
            {
                "qid": [1, 1, 2, 2],
                "sens": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"]),
            }
        )
        with pytest.raises(ValueError, match="categorical"):
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


class TestPSensitizeZeroProbabilityFeasibility:
    """Tests for feasibility of target_p given zero-probability priors"""

    def test_infeasible_class_raises_clear_error(self):
        """A class that cannot reach target_p fails with a clear per-class error

        The homogeneous class needs two additional values, but only one value
        with non-zero probability is available to add; the error must say so
        instead of surfacing an opaque numpy sampling error.
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


class TestPSensitizeZeroProbabilityValuesInData:
    """Tests that zero-probability values already in the data count toward p"""

    def test_already_compliant_with_zero_probability_value(self):
        """A dataset already at target_p needs no perturbation, regardless of priors

        The class contains both 'a' and 'b'; 'b' having probability 0.0 is
        irrelevant because nothing needs to be drawn.
        """
        input_df = pd.DataFrame({"qid": [1, 1], "sens": ["a", "b"]})
        output_df, num_perturbated = p_sensitize(
            LOGGER,
            input_df,
            ["qid"],
            "sens",
            2,
            2,
            {"a": 1.0, "b": 0.0},
            seed=0,
        )
        assert num_perturbated == 0
        assert output_df["sens"].tolist() == ["a", "b"]

    def test_present_zero_probability_value_counts_toward_p(self):
        """A zero-probability value present in a class counts toward its p

        The class contains {a, c} (c has probability 0.0) and needs only one
        more value; 'b' is selectable, so target_p=3 is feasible.
        """
        input_df = pd.DataFrame({"qid": [1, 1, 1], "sens": ["a", "a", "c"]})
        output_df, num_perturbated = p_sensitize(
            LOGGER,
            input_df,
            ["qid"],
            "sens",
            3,
            3,
            {"a": 0.5, "b": 0.5, "c": 0.0},
            seed=0,
        )
        assert num_perturbated == 1
        assert output_df["sens"].nunique() == 3
        assert "c" in set(output_df["sens"])
