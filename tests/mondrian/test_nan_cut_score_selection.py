"""
Tests for cut selection when a proposed cut scores NaN.

score_proposed_cut returns (NaN, tiebreaker) when the (sampled) partition has
no variance. NaN never compares less-than, so a NaN-scored cut that becomes
the early best can never be displaced by a later finite-scored cut, and
sorting tuples containing NaN violates strict weak ordering. NaN scores must
order as worst, never blocking finite-scored cuts.
"""

import logging

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.mondrian.implementation import NumericalCutPointsMode
from project_lighthouse_anonymize.mondrian.k_anonymity import Implementation_KAnonymity
from project_lighthouse_anonymize.mondrian.tree import CutChoice

LOGGER = logging.getLogger(__name__)


def _make_implementation():
    return Implementation_KAnonymity(
        LOGGER,
        k=2,
        cut_choice_score_epsilon=None,
        cut_score_epsilon_partition_size_cutoff=0.9,
        numerical_cut_points_modes=(NumericalCutPointsMode.MEDIAN,),
        dq_metric_to_minimum_dq={},
        dynamic_breakout_rilm_multiplier=None,
    )


class TestProposeCutsNaNScore:
    """Tests that a finite-scored cut displaces an earlier NaN-scored cut"""

    def test_finite_score_beats_earlier_nan_score(self):
        """propose_cuts must yield the finite-scored candidate as best"""
        implementation = _make_implementation()
        input_df = pd.DataFrame({"q": [0.0, 0.0, 1.0, 1.0]})
        candidate_a = [input_df.iloc[:2], input_df.iloc[2:]]
        candidate_b = [input_df.iloc[:3], input_df.iloc[3:]]
        implementation.propose_cuts_numerical = lambda nid, df, cut_choice: iter(
            [candidate_a, candidate_b]
        )
        implementation.classify_and_merge_partitions = (
            lambda partition_dfs, qids, gtrees, force_output_df=False: ([], partition_dfs)
        )
        implementation.score_proposed_cut = lambda proposed_cut, df, qids, gtrees: (
            (np.nan, 1.0) if proposed_cut.further_cuts is candidate_a else (-0.5, 1.0)
        )

        cut_choice = CutChoice("q", 0.0, False)
        proposed = list(implementation.propose_cuts("nid", input_df, ["q"], {}, cut_choice, None))
        assert len(proposed) == 1
        assert proposed[0].further_cuts is candidate_b, (
            "NaN-scored cut blocked a finite-scored cut from becoming best"
        )


class TestNaNScoreSortKey:
    """Tests for the NaN-safe cut score sort key"""

    def test_nan_scores_sort_last(self):
        """Sorting cut scores with the NaN-safe key places NaN scores last"""
        from project_lighthouse_anonymize.utils import nan_safe_cut_score_sort_key

        scores = [(np.nan, 1.0), (0.5, 1.0), (np.nan, 0.5), (-0.5, 1.0)]
        ordered = sorted(scores, key=nan_safe_cut_score_sort_key)
        assert ordered[0] == (-0.5, 1.0)
        assert ordered[1] == (0.5, 1.0)
        assert all(np.isnan(score[0]) for score in ordered[2:])
