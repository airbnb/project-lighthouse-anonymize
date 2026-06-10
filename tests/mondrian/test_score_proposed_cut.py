"""
Tests for Implementation_Base.score_proposed_cut secondary (tiebreaker) score.
"""

import logging

import pandas as pd
import pytest

from project_lighthouse_anonymize.mondrian.implementation import NumericalCutPointsMode
from project_lighthouse_anonymize.mondrian.k_anonymity import Implementation_KAnonymity
from project_lighthouse_anonymize.mondrian.tree import CutChoice, ProposedCut

_LOGGER = logging.getLogger(__name__)


class TestScoreProposedCutTiebreaker:
    """Tests that score_2 reflects the fraction of records not suppressed by the cut"""

    @staticmethod
    def make_implementation():
        return Implementation_KAnonymity(
            _LOGGER,
            2,
            0.05,
            0.90,
            (NumericalCutPointsMode.MEDIAN,),
            {},
            None,
        )

    def test_score_2_reflects_suppression(self):
        """A cut whose output_dfs and further_cuts retain 4 of 5 records suppresses
        one record, so score_2 must be 0.8, preferring (per the documented
        tiebreaker) cuts that suppress earlier."""
        implementation = self.make_implementation()
        input_df = pd.DataFrame({"q": [1.0, 1.0, 2.0, 2.0, 9.0]})
        proposed_cut = ProposedCut(
            CutChoice("q", 0.0, False),
            input_df,
            [input_df.iloc[:2], input_df.iloc[2:4]],
            [],
        )
        _, score_2 = implementation.score_proposed_cut(proposed_cut, input_df, ["q"], {})
        assert score_2 == pytest.approx(0.8)

    def test_score_2_is_one_without_suppression(self):
        """A cut retaining all records must have score_2 = 1.0"""
        implementation = self.make_implementation()
        input_df = pd.DataFrame({"q": [1.0, 1.0, 2.0, 2.0]})
        proposed_cut = ProposedCut(
            CutChoice("q", 0.0, False),
            input_df,
            [input_df.iloc[:2]],
            [input_df.iloc[2:]],
        )
        _, score_2 = implementation.score_proposed_cut(proposed_cut, input_df, ["q"], {})
        assert score_2 == pytest.approx(1.0)
