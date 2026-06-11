"""
Tests for NaN handling in compute_score and select_best_run.

A NaN data quality metric value (or a NaN threshold) must not poison the
composite score: a NaN total score makes max() in select_best_run
order-dependent, degenerating "select the best of N runs" into "select the
first run" whenever any run scores NaN. Through the k_anonymize wrapper,
categorical-only datasets omit the pearsons/NMI keys rather than setting them
to NaN; explicit NaN metric values reach these public functions through direct
API callers and are the shape modeled by the legacy select_best_run tests.
"""

import math

import numpy as np
import pytest

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA
from project_lighthouse_anonymize.wrappers.shared import compute_score, select_best_run


class TestComputeScoreNaN:
    """Tests for compute_score NaN handling"""

    def test_nan_threshold_is_skipped(self):
        """A NaN threshold disables the metric instead of producing a NaN score"""
        score = compute_score(
            {"a": 0.95, "b": 0.95},
            {"a": 0.90, "b": NOT_DEFINED_NA},
        )
        assert not np.isnan(score)
        assert score == pytest.approx(compute_score({"a": 0.95}, {"a": 0.90}))

    def test_nan_value_scores_worst_not_nan(self):
        """A NaN metric value scores strictly below any real value, never NaN

        Strictness matters: if a NaN tied a measured value (e.g. 0.0), max()
        in select_best_run would break the tie by position, re-introducing
        order-dependent selection for that tie.
        """
        score_nan = compute_score({"a": NOT_DEFINED_NA}, {"a": 0.90})
        assert not np.isnan(score_nan)
        assert score_nan < compute_score({"a": 0.0}, {"a": 0.90})
        assert score_nan < compute_score({"a": 0.5}, {"a": 0.90})


class TestSelectBestRunNaNCategoricalOnly:
    """Tests for select_best_run when all runs have NaN values for a metric"""

    @staticmethod
    def categorical_only_runs():
        """Two categorical-only runs (NaN pearsons), the second clearly better"""
        return [
            {"rilm_categorical__minimum": 0.91, "pearsons__minimum": NOT_DEFINED_NA},
            {"rilm_categorical__minimum": 0.99, "pearsons__minimum": NOT_DEFINED_NA},
        ]

    @staticmethod
    def thresholds():
        return {"rilm_categorical__minimum": 0.90, "pearsons__minimum": 0.90}

    def test_categorical_only_selects_best_run(self):
        """With all-NaN pearsons, the higher-RILM run wins regardless of position"""
        runs = self.categorical_only_runs()
        idx, minimum_dq_met = select_best_run(runs, self.thresholds())
        assert (idx, minimum_dq_met) == (1, True), f"got ({idx}, {minimum_dq_met})"

    def test_categorical_only_selection_is_order_independent(self):
        """Reversing the run order must select the same (best) run"""
        runs = list(reversed(self.categorical_only_runs()))
        idx, minimum_dq_met = select_best_run(runs, self.thresholds())
        assert (idx, minimum_dq_met) == (0, True), f"got ({idx}, {minimum_dq_met})"


class TestSelectBestRunNaNFallback:
    """Tests for select_best_run fallback when no run passes thresholds"""

    @staticmethod
    def thresholds():
        return {"rilm_categorical__minimum": 0.90, "pearsons__minimum": 0.90}

    def test_fallback_prefers_near_miss_over_nan_garbage(self):
        """When no run passes, a near-miss run beats a garbage run with a NaN metric"""
        runs = [
            {"rilm_categorical__minimum": 0.10, "pearsons__minimum": NOT_DEFINED_NA},
            {"rilm_categorical__minimum": 0.89, "pearsons__minimum": 0.89},
        ]
        idx, minimum_dq_met = select_best_run(runs, self.thresholds())
        assert (idx, minimum_dq_met) == (1, False), f"got ({idx}, {minimum_dq_met})"

    def test_existing_non_nan_behavior_unchanged(self):
        """Validation check: selection among finite-scored runs is unaffected"""
        runs = [
            {"rilm_categorical__minimum": 0.92, "pearsons__minimum": 0.92},
            {"rilm_categorical__minimum": 0.99, "pearsons__minimum": 0.99},
        ]
        idx, minimum_dq_met = select_best_run(runs, self.thresholds())
        assert (idx, minimum_dq_met) == (1, True)
        assert math.isfinite(compute_score(runs[0], self.thresholds()))


class TestComputeScoreZeroThreshold:
    """Tests for compute_score with a zero minimum_dq threshold"""

    def test_zero_threshold_at_threshold(self):
        """dq_value=0, minimum_dq=0 -> score of 0.0 (threshold met exactly)"""
        assert compute_score({"metric": 0.0}, {"metric": 0.0}) == pytest.approx(0.0)

    def test_zero_threshold_above(self):
        """dq_value above threshold -> positive score"""
        assert compute_score({"metric": 0.5}, {"metric": 0.0}) > 0.0

    def test_zero_threshold_below_does_not_raise(self):
        """dq_value at zero threshold -> finite score, no ZeroDivisionError"""
        assert math.isfinite(compute_score({"metric": 0.0}, {"metric": 0.0}))


class TestComputeScoreOutOfContract:
    """Tests that out-of-contract inputs raise ValueError"""

    def test_dq_value_above_one_raises(self):
        with pytest.raises(ValueError):
            compute_score({"metric": 1.5}, {"metric": 0.5})

    def test_minimum_dq_above_one_raises(self):
        with pytest.raises(ValueError):
            compute_score({"metric": 0.5}, {"metric": 1.5})
