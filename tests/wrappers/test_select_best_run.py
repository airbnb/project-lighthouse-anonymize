"""
Tests for select_best_run wrapper function
"""

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA
from project_lighthouse_anonymize.wrappers.shared import select_best_run


class TestSelectBestRun:
    """
    Tests for select_best_run.
    """

    # Not unittest.TestCase so that generators work with nosetests.
    # per "Please note that method generators are not supported in unittest.TestCase subclasses."
    # from https://nose.readthedocs.io/en/latest/writing_tests.html
    # pylint: disable=no-self-use

    def test_select_best_run_1(self):
        """
        test case 1: select_best_run with a single run
        """
        dq_metrics_arr = [
            {
                "rilm_categorical__minimum": 0.10,
                "pearsons__minimum": 0.20,
                "nmi_sampled_scaled_v1__minimum": NOT_DEFINED_NA,
            }
        ]
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.50,
            "pearsons__minimum": 0.50,
            "nmi_sampled_scaled_v1__minimum": 0.50,
        }
        expected_idx, expected_minimum_dq_met = 0, False
        actual_idx, actual_minimum_dq_met = select_best_run(dq_metrics_arr, dq_metric_to_minimum_dq)
        assert actual_idx == expected_idx
        assert actual_minimum_dq_met == expected_minimum_dq_met

    def test_select_best_run_2(self):
        """
        test case 2: select_best_run with two runs, one of which meets minimum dq
        """
        dq_metrics_arr = [
            {
                "rilm_categorical__minimum": 0.99,
                "pearsons__minimum": 0.99,
                "nmi_sampled_scaled_v1__minimum": NOT_DEFINED_NA,
            },
            {
                "rilm_categorical__minimum": 0.50,
                "pearsons__minimum": 0.50,
                "nmi_sampled_scaled_v1__minimum": 0.50,
            },
        ]
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.50,
            "pearsons__minimum": 0.50,
            "nmi_sampled_scaled_v1__minimum": 0.50,
        }
        expected_idx, expected_minimum_dq_met = 1, True
        actual_idx, actual_minimum_dq_met = select_best_run(dq_metrics_arr, dq_metric_to_minimum_dq)
        assert actual_idx == expected_idx
        assert actual_minimum_dq_met == expected_minimum_dq_met

    def test_select_best_run_3(self):
        """
        test case 3: select_best_run with three runs, two of which meets minimum dq
        """
        dq_metrics_arr = [
            {
                "rilm_categorical__minimum": 0.99,
                "pearsons__minimum": 0.99,
                "nmi_sampled_scaled_v1__minimum": NOT_DEFINED_NA,
            },
            {
                "rilm_categorical__minimum": 0.50,
                "pearsons__minimum": 0.90,  # 0.40 improvement on minimum
                "nmi_sampled_scaled_v1__minimum": 0.50,
            },
            {
                "rilm_categorical__minimum": 0.50,
                "pearsons__minimum": 0.60,  # 0.10 improvement on minimum
                "nmi_sampled_scaled_v1__minimum": 0.60,  # 0.10 improvement on minimum
            },
        ]
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.50,
            "pearsons__minimum": 0.50,
            "nmi_sampled_scaled_v1__minimum": 0.50,
        }
        expected_idx, expected_minimum_dq_met = 2, True
        actual_idx, actual_minimum_dq_met = select_best_run(dq_metrics_arr, dq_metric_to_minimum_dq)
        assert actual_idx == expected_idx
        assert actual_minimum_dq_met == expected_minimum_dq_met

    def test_select_best_run_4(self):
        """
        test case 4: select_best_run with two runs, with some nan thresholds
        """
        dq_metrics_arr = [
            {
                "rilm_categorical__minimum": 0.99,
                "pearsons__minimum": 0.99,
                "nmi_sampled_scaled_v1__minimum": NOT_DEFINED_NA,  # threshold is nan so don't care
            },
            {
                "rilm_categorical__minimum": 0.50,
                "pearsons__minimum": 0.90,
                "nmi_sampled_scaled_v1__minimum": 0.50,
            },
        ]
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.50,
            "pearsons__minimum": NOT_DEFINED_NA,
            "nmi_sampled_scaled_v1__minimum": NOT_DEFINED_NA,
        }
        expected_idx, expected_minimum_dq_met = 0, True
        actual_idx, actual_minimum_dq_met = select_best_run(dq_metrics_arr, dq_metric_to_minimum_dq)
        assert actual_idx == expected_idx
        assert actual_minimum_dq_met == expected_minimum_dq_met

    def test_select_best_run_5(self):
        """
        test case 5: select_best_run with two runs, none of which meet minimum dq
        """
        dq_metrics_arr = [
            {
                "rilm_categorical__minimum": 0.90,
                "pearsons__minimum": 0.90,
                "nmi_sampled_scaled_v1__minimum": 0.50,
            },
            {
                "rilm_categorical__minimum": 0.90,
                "pearsons__minimum": 0.60,
                "nmi_sampled_scaled_v1__minimum": 0.60,
            },
        ]
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.90,
        }
        expected_idx, expected_minimum_dq_met = 1, False
        actual_idx, actual_minimum_dq_met = select_best_run(dq_metrics_arr, dq_metric_to_minimum_dq)
        assert actual_idx == expected_idx
        assert actual_minimum_dq_met == expected_minimum_dq_met

    def test_select_best_run_6(self):
        """
        test case 6: select_best_run with one run, which meets targets except for nan values; this reflects
                     the case where only categoricals exist, so that some data quality metrics are nan.
        """
        dq_metrics_arr = [
            {
                "rilm_categorical__minimum": 0.90,
                "pearsons__minimum": NOT_DEFINED_NA,
                "nmi_sampled_scaled_v1__minimum": NOT_DEFINED_NA,
            }
        ]
        dq_metric_to_minimum_dq = {
            "rilm_categorical__minimum": 0.90,
            "pearsons__minimum": 0.90,
            "nmi_sampled_scaled_v1__minimum": 0.90,
        }
        expected_idx, expected_minimum_dq_met = 0, True
        actual_idx, actual_minimum_dq_met = select_best_run(dq_metrics_arr, dq_metric_to_minimum_dq)
        assert actual_idx == expected_idx
        assert actual_minimum_dq_met == expected_minimum_dq_met
