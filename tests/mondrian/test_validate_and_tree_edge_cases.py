"""
Tests for k-anonymity validation of empty frames and tree extraction isolation.

An empty DataFrame is vacuously k-anonymous (documented contract). Extracting
a candidate solution with a node identifier column must not mutate the
DataFrames stored in the tree.
"""

import logging

import pandas as pd

from project_lighthouse_anonymize.mondrian.implementation import NumericalCutPointsMode
from project_lighthouse_anonymize.mondrian.k_anonymity import Implementation_KAnonymity
from project_lighthouse_anonymize.mondrian.tree import MondrianTree, Node_FinalCutData

LOGGER = logging.getLogger(__name__)


class TestValidateEmptyDataFrame:
    """Tests for Implementation_KAnonymity.validate on empty input"""

    def test_empty_dataframe_is_k_anonymous(self):
        """An empty DataFrame satisfies k-anonymity vacuously"""
        implementation = Implementation_KAnonymity(
            LOGGER,
            k=2,
            cut_choice_score_epsilon=None,
            cut_score_epsilon_partition_size_cutoff=0.9,
            numerical_cut_points_modes=(NumericalCutPointsMode.MEDIAN,),
            dq_metric_to_minimum_dq={},
            dynamic_breakout_rilm_multiplier=None,
        )
        empty_df = pd.DataFrame({"q": pd.Series([], dtype="float64")})
        assert implementation.validate(empty_df, ["q"]) is True


class TestCandidateSolutionIsolation:
    """Tests that candidate_solution does not mutate stored node DataFrames"""

    def test_node_identifier_col_does_not_mutate_stored_df(self):
        """Adding a node identifier column must not write into the tree's data"""
        stored_df = pd.DataFrame({"q": [1.0, 2.0]})
        tree = MondrianTree()
        tree.create_node(None, Node_FinalCutData(stored_df))

        solution = tree.candidate_solution(
            qids=["q"],
            all_work_should_be_done=False,
            track_cuts=False,
            node_identifier_col="source_nid",
        )
        assert "source_nid" in solution.columns
        assert "source_nid" not in stored_df.columns, (
            "candidate_solution mutated the DataFrame stored in the tree"
        )
