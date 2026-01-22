"""
Tests for check_dq_meets_minimum_thresholds wrapper function
"""

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA
from project_lighthouse_anonymize.wrappers.shared import (
    check_dq_meets_minimum_thresholds,
)


class TestCheckDqMeetsMinimumThresholds:
    """
    Tests for check_dq_meets_minimum_thresholds.
    """

    # Not unittest.TestCase so that generators work with nosetests.
    # per "Please note that method generators are not supported in unittest.TestCase subclasses."
    # from https://nose.readthedocs.io/en/latest/writing_tests.html
    # pylint: disable=no-self-use

    def test_check_dq_meets_minimum_thresholds_passes_all(self):
        """
        Test case: all metrics meet minimum thresholds
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.95,
            "pearsons__minimum": 0.95,
            "nmi_sampled_scaled_v1__minimum": 0.85,
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is True
        assert failure_details == []

    def test_check_dq_meets_minimum_thresholds_fails_one_metric(self):
        """
        Test case: one metric fails to meet minimum threshold
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.85,  # Below threshold of 0.90
            "pearsons__minimum": 0.95,
            "nmi_sampled_scaled_v1__minimum": 0.85,
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is False
        assert len(failure_details) == 1
        assert failure_details[0][0] == "rilm_categorical__minimum"
        assert "0.85" in failure_details[0][1]
        assert "0.9" in failure_details[0][1]

    def test_check_dq_meets_minimum_thresholds_missing_metric_passes(self):
        """
        Test case: missing metrics should pass threshold check
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.95,
            # pearsons__minimum is missing
            "nmi_sampled_scaled_v1__minimum": 0.85,
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is True
        assert failure_details == []

    def test_check_dq_meets_minimum_thresholds_nan_metric_with_non_nan_threshold_fails(
        self,
    ):
        """
        Test case: NaN metric value with non-NaN threshold should fail
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.95,
            "pearsons__minimum": NOT_DEFINED_NA,  # NaN value
            "nmi_sampled_scaled_v1__minimum": 0.85,
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,  # Non-NaN threshold
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is False
        assert len(failure_details) == 1
        assert failure_details[0][0] == "pearsons__minimum"
        assert "(nan)" in failure_details[0][1]

    def test_check_dq_meets_minimum_thresholds_nan_metric_with_nan_threshold_passes(
        self,
    ):
        """
        Test case: NaN metric value with NaN threshold should pass (threshold disabled)
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.95,
            "pearsons__minimum": NOT_DEFINED_NA,  # NaN value
            "nmi_sampled_scaled_v1__minimum": 0.85,
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": NOT_DEFINED_NA,  # NaN threshold (disabled)
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is True
        assert failure_details == []

    def test_check_dq_meets_minimum_thresholds_default_thresholds(self):
        """
        Test case: using default thresholds (None parameter)
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.95,
            "pearsons__minimum": 0.95,
            "nmi_sampled_scaled_v1__minimum": 0.85,
            "pct_non_suppressed": 0.995,
        }
        # Using None to trigger default thresholds

        passes, failure_details = check_dq_meets_minimum_thresholds(dq_metrics, None)
        assert passes is True
        assert failure_details == []

    def test_check_dq_meets_minimum_thresholds_default_thresholds_fails(self):
        """
        Test case: failing with default thresholds
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.85,  # Below default threshold of 0.90
            "pearsons__minimum": 0.95,
            "nmi_sampled_scaled_v1__minimum": 0.85,
            "pct_non_suppressed": 0.995,
        }
        # Using None to trigger default thresholds

        passes, failure_details = check_dq_meets_minimum_thresholds(dq_metrics, None)
        assert passes is False
        assert len(failure_details) == 1
        assert failure_details[0][0] == "rilm_categorical__minimum"

    def test_check_dq_meets_minimum_thresholds_edge_case_exact_threshold(self):
        """
        Test case: metric exactly meets threshold (should pass)
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.90,  # Exactly meets threshold
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is True
        assert failure_details == []

    def test_check_dq_meets_minimum_thresholds_multiple_failures(self):
        """
        Test case: multiple metrics fail, all should be captured
        """
        dq_metrics = {
            "rilm_categorical__minimum": 0.75,  # Below threshold of 0.90
            "pearsons__minimum": 0.80,  # Below threshold of 0.90
            "nmi_sampled_scaled_v1__minimum": 0.85,  # Passes
        }
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.80,
        }

        passes, failure_details = check_dq_meets_minimum_thresholds(
            dq_metrics, dq_metric_to_minimum_dq
        )
        assert passes is False
        assert len(failure_details) == 2

        # Check that both failing metrics are captured
        failing_metrics = [detail[0] for detail in failure_details]
        assert "rilm_categorical__minimum" in failing_metrics
        assert "pearsons__minimum" in failing_metrics
