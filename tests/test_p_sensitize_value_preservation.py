"""
Tests that p_sensitize never eliminates an existing sensitive value while perturbating,
which would leave the equivalence class below the target p.
"""

import logging

import pandas as pd

from project_lighthouse_anonymize.p_sensitize import p_sensitize

_LOGGER = logging.getLogger(__name__)


class TestPSensitizeValuePreservation:
    """Tests for p_sensitize perturbation record selection"""

    @staticmethod
    def make_input_df():
        """Single equivalence class with sensitive value counts {A: 2, B: 2}"""
        return pd.DataFrame(
            {
                "qid": [1, 1, 1, 1],
                "sensitive_value": ["A", "A", "B", "B"],
                "row_id": [0, 1, 2, 3],
            }
        )

    @staticmethod
    def run_p_sensitize(seed):
        """Run p_sensitize on the canonical input with target_p=4, target_k=4"""
        sens_attr_value_to_prob = {
            "A": 0.2,
            "B": 0.2,
            "C": 0.2,
            "D": 0.2,
            "E": 0.1,
            "F": 0.1,
        }
        return p_sensitize(
            _LOGGER,
            TestPSensitizeValuePreservation.make_input_df(),
            ["qid"],
            "sensitive_value",
            4,
            4,
            sens_attr_value_to_prob,
            seed=seed,
        )

    def test_target_p_reached_without_eliminating_values(self):
        """With counts {A: 2, B: 2} and target_p=4, two records must be perturbated
        without exhausting either A or B; otherwise only 3 distinct values remain.
        Exercise many seeds since the selection is random."""
        for seed in range(30):
            output_df, n_perturbated = self.run_p_sensitize(seed)
            actual_p = output_df["sensitive_value"].nunique()
            output_values = sorted(output_df["sensitive_value"])
            assert n_perturbated == 2
            assert actual_p >= 4, (
                f"seed={seed}: p-sensitize produced p={actual_p} < target_p=4 "
                f"(sensitive values: {output_values})"
            )
            assert {"A", "B"} <= set(output_values), (
                f"seed={seed}: an original sensitive value was eliminated "
                f"(remaining: {output_values})"
            )
