"""
Base implementation for privacy-preserving data anonymization.

This module defines the abstract base class and shared logic for implementing
different technical privacy models using the Core Mondrian algorithm. It provides
the foundation for various anonymization strategies like k-anonymity and u-anonymity
by defining the common functionality while allowing specific implementations to
override key methods.

The Implementation_Base class encapsulates:
1. Core data partitioning logic for both numerical and categorical attributes
2. Cut selection and scoring mechanics
3. Abstract methods that specific privacy models must implement
4. Generalization and suppression mechanisms

Key Architectural Differences from Original Mondrian:
-----------------------------------------------------
The original Mondrian algorithm was specifically designed for k-anonymity, whereas
this implementation uses a strategy pattern through the Implementation_Base class
that allows for multiple privacy models (k-anonymity, u-anonymity, etc.) with
different behaviors. This architecture enables:

1. Pluggable Privacy Models: Different technical privacy models can be implemented
   by extending the base class and overriding specific abstract methods.

2. Customizable Cut Selection: The implementation allows different strategies for
   selecting attributes to cut and evaluating those cuts based on data quality metrics.

3. Flexible Partition Validation: Each implementation can define its own criteria
   for when partitions can be further cut and when they should become final cuts.

4. Separation of Concerns: The base implementation handles common operations like
   generalization and suppression mechanics, while concrete implementations focus
   on the specific requirements of each privacy model.

Implementations derived from this base class can customize the anonymization behavior
by defining how cuts are chosen, validated, and scored according to the specific
technical privacy model being enforced.
"""

import functools as ft
import itertools as it
import logging
import warnings
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Generator, Optional, cast

import numpy as np
import pandas as pd
from treelib import (
    Node,  # type: ignore[reportPrivateImportUsage]  # Node is publicly exported from treelib
)

from project_lighthouse_anonymize.gtrees import GTree
from project_lighthouse_anonymize.mondrian.funnel_stats import FunnelStats
from project_lighthouse_anonymize.mondrian.tree import CutChoice, ProposedCut
from project_lighthouse_anonymize.utils import (
    atleast_nunique,
    median_max_second_max,
    standard_deviation,
    standard_deviation_categorical,
)

ALLOWED_DTYPES_NUMERICAL = {np.dtype("int64"), np.dtype("int64"), np.dtype("float64")}
# This anonymization algorithm allows numerical QIDs to be treated as categoricals w.r.t. generalization trees
# TODO(Later) micro-aggregate for QIDs where that's the case?
ALLOWED_DTYPES_CATEGORICAL = {
    np.dtype("int64"),
    np.dtype("int64"),
    np.dtype("float64"),
    np.dtype("object"),
}

NumericalCutPointsMode = Enum("NumericalCutPointsMode", ["MEDIAN", "BIN_EDGES"])

EMPTY_CUT_QID = "<N/A Empty>"  # TODO(Later) get rid of this


class Implementation_Base(metaclass=ABCMeta):
    """
    Abstract base class for Core Mondrian algorithm implementations.

    This class defines the interface and common functionality for different
    implementations of the Mondrian algorithm, each enforcing different technical
    privacy models such as k-anonymity or u-anonymity.

    Design Differences from Original Mondrian:
    -------------------------------------------
    The original Mondrian algorithm (LeFevre et al., 2006) implemented a monolithic
    approach specifically for k-anonymity with fixed cut selection mechanics. In contrast,
    this Implementation_Base class:

    1. Uses the Strategy Pattern: Separates the algorithm structure (in CoreMondrian)
       from the privacy model-specific behaviors defined in concrete implementations.

    2. Provides Extension Points:

       Defines abstract methods that subclasses must implement
       to specify their privacy model's unique requirements:

       - validate(): Checks if a partition meets the specific privacy model requirements
       - are_cuts_possible(): Determines if further splitting a partition is feasible
       - cut_choices(): Generates and ranks potential attributes to cut on
       - propose_cuts_numerical_impl() and propose_cuts_categorical_impl(): Implement
         type-specific cutting logic

    3. Shares Common Functionality:

       Implements reusable mechanisms applicable across
       different privacy models:

       - QID setup and validation
       - Cut proposal generation and scoring
       - Generalization and suppression logic
       - Partition classification and merging

    The base implementation provides mechanisms for:
    - Setting up and processing both categorical and numerical quasi-identifiers (QIDs)
    - Proposing and evaluating different partition cuts
    - Scoring proposed cuts to optimize for data utility
    - Generalizing and potentially suppressing values to meet privacy requirements

    Concrete implementations must override abstract methods to define specific behaviors
    related to their technical privacy model, such as determining when cuts are possible
    or when a partition meets the privacy requirements.
    """

    def __init__(
        self,
        logger: logging.Logger,
        cut_choice_score_epsilon: Optional[float],
        cut_score_epsilon_partition_size_cutoff: Optional[float],
        numerical_cut_points_modes: tuple[NumericalCutPointsMode, ...],
    ):
        """
        Initialize the base implementation for Core Mondrian algorithm.

        Parameters
        ----------
        logger : logging.Logger
            Logger for recording the anonymization process.
        cut_choice_score_epsilon : float or None
            Epsilon for determining whether two cut choices should be considered at the same time.
            - A higher value may increase data quality by evaluating multiple proposed cuts,
              at the cost of increased runtime.
            - A lower value may decrease data quality, with the benefit of reduced runtime.
            - If None, all cut choices will be considered regardless of their scores.
        cut_score_epsilon_partition_size_cutoff : float or None
            Relative partition size (as a fraction of input data) below which multiple cut
            choices will not be considered together, regardless of cut_choice_score_epsilon.
            - A higher value may increase data quality by evaluating multiple proposed cuts,
              at the cost of increased runtime.
            - A lower value may decrease data quality, with the benefit of reduced runtime.
            - If None, partition size won't affect consideration of multiple cut choices.
        numerical_cut_points_modes : Tuple[NumericalCutPointsMode, ...]
            One or more modes for cut points to consider for numerical QID cuts.
            Should be a tuple containing one or more of:
            - NumericalCutPointsMode.MEDIAN: Use the median value as a cut point, unless it
              is the maximum value in which case use the next lower value.
            - NumericalCutPointsMode.BIN_EDGES: Compute multiple cut points based on
              np.histogram_bin_edges with bins="auto".
        """
        self.logger = logger
        self.cut_choice_score_epsilon = cut_choice_score_epsilon
        self.cut_score_epsilon_partition_size_cutoff = cut_score_epsilon_partition_size_cutoff
        self.numerical_cut_points_modes = numerical_cut_points_modes
        self.multiple_proposed_cuts_partition_size_cutoff = 0
        self.complex_numerical_cut_points_size_cutoff = 0
        self.qids: list[str] = []

    def debug_logging_enabled(self) -> bool:
        return self.logger.isEnabledFor(logging.DEBUG)

    def setup(
        self,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
        exclude_qids: list[str],
    ) -> None:
        """
        Set up the implementation for running the Core Mondrian algorithm.

        This method performs initial setup for the anonymization process, including
        validating and preparing each QID for processing. It determines how to handle
        numerical versus categorical QIDs and calculates thresholds for optimization
        decisions during the anonymization process.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame to be anonymized.
        qids : List[str]
            List of quasi-identifier column names in the input DataFrame.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to their generalization trees.
            QIDs without entries in this dictionary are treated as numerical.
        exclude_qids : List[str]
            QIDs to exclude from computing overall data quality metric minimum scores.

        Notes
        -----
        Implementations may override this method to add columns to input_df to
        optimize processing, but they are responsible for removing these columns
        by also overriding the generalize_and_suppress method.

        Implementations should *not* save gtrees as instance attributes. The gtrees
        will be provided by core.py during processing in a manner that minimizes
        pickling overhead when parallel processing is enabled.
        """
        if EMPTY_CUT_QID in qids:
            raise ValueError(
                f"'{EMPTY_CUT_QID}' is reserved for internal use and cannot be used as a QID name"
            )
        self.qids = list(qids)
        for qid in qids:
            gtree_if = gtrees.get(qid)
            if gtree_if is None:
                self.setup_numerical_qid(input_df, qid)
            else:
                self.setup_categorical_qid(input_df, qid, gtree_if)
        self.multiple_proposed_cuts_partition_size_cutoff = (
            int(self.cut_score_epsilon_partition_size_cutoff * len(input_df))
            if self.cut_score_epsilon_partition_size_cutoff is not None
            else 0
        )
        self.complex_numerical_cut_points_size_cutoff = int(0.25 * len(input_df))
        self.logger.debug(
            "multiple_proposed_cuts_partition_size_cutoff = %d",
            self.multiple_proposed_cuts_partition_size_cutoff,
        )

    def setup_categorical_qid(self, input_df: pd.DataFrame, qid: str, gtree: GTree) -> None:
        """
        Set up a categorical QID for Core Mondrian anonymization.

        This method validates and prepares a categorical quasi-identifier column for
        processing. It ensures the column has a supported data type, validates that
        all values in the column are represented in the generalization tree as leaf nodes,
        and prepares the generalization tree for efficient processing.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame containing the data to be anonymized.
        qid : str
            The column name of the categorical quasi-identifier.
        gtree : GTree
            The generalization tree for this categorical QID.

        Raises
        ------
        ValueError
            If the column data type is not supported or if any value in the column
            is not represented as a leaf node in the generalization tree.

        Notes
        -----
        Implementations may override this method to add columns to input_df to
        optimize processing, but they are responsible for removing these columns
        by also overriding the generalize_and_suppress method.

        Implementations should *not* save gtree as an instance attribute. The gtrees
        will be provided by core.py during processing in a manner that minimizes
        pickling overhead when parallel processing is enabled.
        """
        if input_df.dtypes[qid] not in ALLOWED_DTYPES_CATEGORICAL:
            raise TypeError(f"categorical qid {qid} dtype ({input_df.dtypes[qid]}) not supported")
        uniq_orig_values = set(input_df[~pd.isna(input_df[qid])][qid])
        if not uniq_orig_values.issubset(gtree.get_value(node) for node in gtree.leaves()):
            leaf_values = set(gtree.get_value(node) for node in gtree.leaves())
            non_leaf_values = uniq_orig_values - leaf_values
            raise ValueError(
                f"Mondrian requires that all categorical values are leaves in the associated gtree, these aren't for qid {qid}: {non_leaf_values}"
            )
        # we update here so that if process-based parallelism is enabled, the gtrees pickled for
        # sub-processes are already ready for efficient calls to gtree.descendant_leaf_values
        if gtree.update_descendant_leaf_values_if():
            warnings.warn(
                f"Had to update descendant leaf values in GTree for {qid}; it's more efficient to provide the GTree with this pre-computed by calling gtree.update_descendant_leaf_values_if() prior to anonymizing.",
            )
        if gtree.update_lowest_node_with_descendant_leaves_if():
            warnings.warn(
                f"Had to update leaf node mappings in GTree for {qid}; it's more efficient to provide the GTree with this pre-computed by calling gtree.update_lowest_node_with_descendant_leaves_if() prior to anonymizing.",
            )

    def setup_numerical_qid(self, input_df: pd.DataFrame, qid: str) -> None:
        """
        Set up a numerical QID for Core Mondrian anonymization.

        This method validates and prepares a numerical quasi-identifier column for
        processing. It ensures the column has a supported numeric data type and performs
        any necessary preparations for anonymization.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame containing the data to be anonymized.
        qid : str
            The column name of the numerical quasi-identifier.

        Raises
        ------
        AssertionError
            If the column data type is not a supported numerical type.

        Notes
        -----
        Implementations may override this method to add columns to input_df to
        optimize processing, but they are responsible for removing these columns
        by also overriding the generalize_and_suppress method.

        Unlike categorical QIDs, numerical QIDs do not require a generalization tree.
        """
        if input_df.dtypes[qid] not in ALLOWED_DTYPES_NUMERICAL:
            raise TypeError(f"numerical qid {qid} dtype ({input_df.dtypes[qid]}) not supported")

    @abstractmethod
    def cut_choices(
        self,
        nid: str,
        input_df: pd.DataFrame,
        gtrees: dict[str, GTree],
        qids_to_consider: list[str],
    ) -> Generator[CutChoice, None, None]:
        """
        Generate potential cut choices for partitioning the data.

        This abstract method must be implemented by concrete classes to determine
        which QIDs should be considered for cutting and how they should be scored.
        It generates a sequence of CutChoice objects that represent possible ways
        to partition the data, in order of preference.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.
        qids_to_consider : List[str]
            List of QID column names that have at least two unique values and
            should be considered for cutting.

        Yields
        ------
        CutChoice
            Potential cut choices for partitioning the data, each specifying:
            - The QID column to cut on
            - A score to compare cuts (lower is better)
            - Whether the QID is categorical

        Notes
        -----
        Implementations should yield CutChoice objects in order of preference.
        The Core Mondrian algorithm will consider these choices in the yielded order,
        possibly grouping similar-scoring choices together based on the
        cut_choice_score_epsilon parameter.
        """
        # This is an abstract method that must be implemented by subclasses
        pass

    def keep_examining_cut_choices(
        self,
        nid: str,
        prior_cut_choice: Optional[CutChoice],
        cut_choice: CutChoice,
        input_df: pd.DataFrame,
    ) -> bool:
        """
        Determine whether to continue examining additional cut choices.

        This method decides whether the algorithm should continue examining more
        cut choices after already finding some viable options. It implements an
        optimization that stops examining cut choices when:
        1. We've already found a cut choice, AND
        2. Either:

           a. The new cut choice's score is not close to the prior one's score, OR
           b. The partition is small enough that we want to limit processing time

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        prior_cut_choice : CutChoice or None
            The previously examined cut choice that was found viable, if any.
        cut_choice : CutChoice
            The current cut choice being considered.
        input_df : pd.DataFrame
            The DataFrame partition being considered for cutting.

        Returns
        -------
        bool
            True if the algorithm should continue examining more cut choices,
            False if it should stop and use the cut choices already found.
        """
        if prior_cut_choice is not None and not self.is_cut_choice_close(
            prior_cut_choice, cut_choice
        ):
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - cut_choice %s !~ prior_cut_choice %s, pausing on further cut choices",
                    nid,
                    cut_choice,
                    prior_cut_choice,
                )
            return False
        if (
            prior_cut_choice is not None
            and len(input_df) < self.multiple_proposed_cuts_partition_size_cutoff
        ):
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - one cut_choice found, and len(input_df) %d < multiple_proposed_cuts_partition_size_cutoff %d, pausing on further cut choices",
                    nid,
                    len(input_df),
                    self.multiple_proposed_cuts_partition_size_cutoff,
                )
            return False
        return True

    def is_cut_choice_close(
        self,
        x: CutChoice,
        y: CutChoice,
    ) -> bool:
        """
        Check if two cut choices have scores close enough to be considered together.

        This method determines whether two cut choices have similar enough scores
        that both should be evaluated for their impact on data quality. It uses the
        cut_choice_score_epsilon parameter to establish a threshold for similarity.

        Parameters
        ----------
        x : CutChoice
            First cut choice to compare.
        y : CutChoice
            Second cut choice to compare.

        Returns
        -------
        bool
            True if the scores are close enough to consider both cut choices,
            False otherwise.

        Notes
        -----
        If cut_choice_score_epsilon is None, this method always returns True,
        indicating that all cut choices should be considered regardless of their
        score differences.
        """
        if self.cut_choice_score_epsilon is None:
            return True
        return bool(np.abs(x.score - y.score) <= self.cut_choice_score_epsilon)

    def make_base_case_proposed_cut(
        self,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> ProposedCut:
        """
        Create a base case proposed cut when no other cuts are possible.

        This method creates a special proposed cut for the base case in the Mondrian
        algorithm, which occurs when no further partitioning is possible or beneficial.
        The base case cut effectively marks the partition as a leaf node in the
        partitioning tree.

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame partition that cannot be further partitioned.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from QID column names to generalization trees.

        Returns
        -------
        ProposedCut
            A special ProposedCut object representing the base case, with:
            - A placeholder CutChoice with an empty QID identifier
            - The input_df as both input and the sole element of further_cuts
            - An empty list for output_dfs

        Notes
        -----
        The base case occurs when:
        1. No further cuts are possible (e.g., all QIDs have only one unique value)
        2. No suitable cut can be found that meets privacy requirements
        3. The partition is too small to be further divided
        """
        output_dfs, _ = self.classify_and_merge_partitions(
            [
                input_df,
            ],
            qids,
            gtrees,
            force_output_df=True,
        )
        return ProposedCut(
            CutChoice(EMPTY_CUT_QID, np.nan, False),
            input_df,
            output_dfs,
            [],
        )

    def propose_cuts(
        self,
        nid: str,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
        cut_choice: CutChoice,
        stats: Optional[FunnelStats],
    ) -> Generator[ProposedCut, None, None]:
        """
        Generate proposed cuts for a given cut choice and select the best one.

        This method evaluates different ways to implement a cut choice (e.g., different
        cut points for a numerical QID or different partition strategies for a categorical
        QID). It generates multiple candidate cuts, evaluates them, and yields only the
        best candidate based on scoring.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from QID column names to generalization trees.
        cut_choice : CutChoice
            The cut choice being evaluated, specifying which QID to cut on.
        stats : FunnelStats, optional
            If provided, tracks statistics about the cut process.

        Yields
        ------
        ProposedCut
            At most one ProposedCut representing the best way to implement the given
            cut_choice. If multiple cuts are possible, only the one with the best
            score according to score_proposed_cut is yielded.

        Notes
        -----
        The method delegates to propose_cuts_numerical or propose_cuts_categorical
        based on whether the QID has a generalization tree, then evaluates and scores
        all potential cuts to find the best one.
        """
        gtree_if = gtrees.get(cut_choice.qid)
        if gtree_if is None:
            partition_dfs_gen = self.propose_cuts_numerical(nid, input_df, cut_choice)
        else:
            partition_dfs_gen = self.propose_cuts_categorical(nid, input_df, cut_choice, gtree_if)
        best_proposed_cut: Optional[ProposedCut] = None
        best_proposed_cut_score: Optional[tuple[float, float]] = None
        for partition_dfs in partition_dfs_gen:
            output_dfs, further_cuts = self.classify_and_merge_partitions(
                partition_dfs, qids, gtrees
            )
            proposed_cut = ProposedCut(cut_choice, input_df, output_dfs, further_cuts)
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - proposed cut for %s under consideration: %s",
                    nid,
                    cut_choice,
                    proposed_cut,
                )
            if best_proposed_cut is None:
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "%s - proposed cut for %s now best proposed cut, since it is the first one considered: %s",
                        nid,
                        cut_choice,
                        proposed_cut,
                    )
                best_proposed_cut = proposed_cut
            else:
                if best_proposed_cut_score is None:
                    best_proposed_cut_score = self.score_proposed_cut(
                        best_proposed_cut,
                        input_df,
                        qids,
                        gtrees,
                    )
                proposed_cut_score = self.score_proposed_cut(
                    proposed_cut,
                    input_df,
                    qids,
                    gtrees,
                )
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "%s - comparing proposed cut scores for %s: %s %s vs. %s %s",
                        nid,
                        cut_choice,
                        best_proposed_cut,
                        best_proposed_cut_score,
                        proposed_cut,
                        proposed_cut_score,
                    )
                if proposed_cut_score < best_proposed_cut_score:
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "%s - proposed cut for %s now best proposed cut, due to superior score: %s %s",
                            nid,
                            cut_choice,
                            proposed_cut,
                            proposed_cut_score,
                        )
                    best_proposed_cut = proposed_cut
                    best_proposed_cut_score = proposed_cut_score
                else:
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "%s - proposed cut for %s ignored, due to non-superior score: %s %s",
                            nid,
                            cut_choice,
                            proposed_cut,
                            proposed_cut_score,
                        )
        if best_proposed_cut is not None:
            if stats is not None:
                stats.increment_under_consideration(cut_choice.qid, cut_choice.is_categorical)
            yield best_proposed_cut

    def classify_and_merge_partitions(
        self,
        partition_dfs: list[pd.DataFrame],
        qids: list[str],
        gtrees: dict[str, GTree],
        force_output_df: bool = False,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Classify partitions and merge intermediate ones that don't meet cut criteria.

        This function takes a list of partition DataFrames and classifies each as:
        1. Further cut candidates - partitions that can be further subdivided
        2. Final cut candidates - partitions that should become final cuts
        3. Intermediate partitions - partitions that don't clearly fit either category

        Parameters
        ----------
        partition_dfs : List[pd.DataFrame]
            List of partition DataFrames to classify
        qids : List[str]
            List of quasi-identifier column names
        gtrees : Dict[str, GTree]
            Dictionary mapping QID column names to their GTree objects
        force_output_df : bool, default=False
            If True, move all further_dfs to output_dfs, effectively preventing further cuts

        Notes
        -----
        Intermediate partitions are merged into a single DataFrame and then
        reclassified. This helps reduce the number of proposed cuts by treating
        related intermediate partitions as a unified group.

        Note: Some intermediate partitions may be discarded if, even after merging,
        they cannot meet the criteria for either further or final cuts.

        Returns
        -------
            A tuple of (output_dfs, further_dfs) where:
            - output_dfs: List of DataFrames that should become final cuts
            - further_dfs: List of DataFrames that can be further subdivided
        """
        output_dfs, further_dfs, intermediate_dfs = [], [], []
        for partition_df in partition_dfs:
            if self.is_a_further_cut(partition_df, qids, gtrees):
                further_dfs.append(partition_df)
            elif self.could_be_a_final_cut(partition_df, qids, gtrees):
                output_dfs.append(partition_df)
            elif not partition_df.empty:
                intermediate_dfs.append(partition_df)

        # Consolidate intermediate partitions that individually don't meet cut criteria
        # instead of processing them separately.
        if len(intermediate_dfs) > 0:
            partition_df = pd.concat(intermediate_dfs)
            if self.is_a_further_cut(partition_df, qids, gtrees):
                further_dfs.append(partition_df)
            elif self.could_be_a_final_cut(partition_df, qids, gtrees):
                output_dfs.append(partition_df)
            else:
                # partition_df will be suppressed. We have experimented with joining with one of the output_dfs/further_dfs, and found a reduction in suppression in exchange for lower data quality metrics.
                pass
        if force_output_df:
            output_dfs += further_dfs
            further_dfs = []
        return output_dfs, further_dfs

    def propose_cuts_numerical(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
    ) -> Generator[list[pd.DataFrame], None, None]:
        """
        Generate proposed cuts for a numerical QID.

        This method identifies potential cut points for a numerical quasi-identifier
        and creates partitions based on those cut points. It first calls a preparation
        method that can be overridden by implementations, then proposes cuts at various
        numerical values, and finally processes those cuts using an implementation-specific
        method.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        cut_choice : CutChoice
            The cut choice specifying which numerical QID to cut on.

        Yields
        ------
        List[pd.DataFrame]
            Lists of DataFrames representing the partitions created by each proposed cut,
            for cuts that were deemed valid by the implementation.

        Notes
        -----
        The method delegates to propose_cuts_numerical_cut_points to identify potential
        cut points, and then to propose_cuts_numerical_impl to apply those cuts to create
        partitions. It skips any cuts that produce None from the implementation method,
        which indicates an invalid cut.
        """
        context = self.propose_cuts_numerical_impl_before(
            nid,
            input_df,
            cut_choice,
        )
        for cut in self.propose_cuts_numerical_cut_points(nid, input_df, cut_choice):
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - attempting cut for %s at value %.3f",
                    nid,
                    cut_choice,
                    cut,
                )
            dfs_if = self.propose_cuts_numerical_impl(nid, input_df, cut_choice, cut, context)
            if dfs_if is not None:
                yield dfs_if

    def propose_cuts_numerical_cut_points(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
    ) -> Generator[float, None, None]:
        """
        Generate potential cut points for a numerical QID.

        This method identifies specific numerical values that could be used as cut points
        for partitioning the data based on a numerical quasi-identifier. It implements
        several strategies for finding cut points depending on the numerical_cut_points_modes
        configuration.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        cut_choice : CutChoice
            The cut choice specifying which numerical QID to cut on.

        Yields
        ------
        float
            Numerical values that could be used as cut points for partitioning the data.

        Notes
        -----
        The method implements two main strategies for finding cut points:

        1. MEDIAN: Uses the median value (or second maximum if median equals maximum)
        2. BIN_EDGES: For larger partitions, uses histogram bin edges to find multiple
           potential cut points that reflect the data distribution

        The BIN_EDGES approach is only used for larger partitions (based on
        complex_numerical_cut_points_size_cutoff) and when the data has enough
        unique values to make this approach meaningful.
        """
        if NumericalCutPointsMode.MEDIAN in self.numerical_cut_points_modes:
            median, maximum, second_maximum = median_max_second_max(
                input_df[cut_choice.qid].to_numpy()
            )
            cut = median
            # shift cut leftward if it is the last unique value; except if only one unique value
            if cut == maximum and not np.isnan(second_maximum):
                cut = second_maximum
            yield cut
        if (
            NumericalCutPointsMode.BIN_EDGES in self.numerical_cut_points_modes
            and len(input_df) >= self.complex_numerical_cut_points_size_cutoff
        ):
            values = input_df[cut_choice.qid].to_numpy()
            if atleast_nunique(values, 5):
                bin_edges = np.histogram_bin_edges(values, bins="auto")
                if len(bin_edges) > 2:
                    bin_edges = bin_edges[
                        :-1
                    ]  # throw out the rightmost interval as that is the maximum
                    # at most 10 cut points
                    if len(bin_edges) > 10:
                        idx = np.round(np.linspace(0, len(bin_edges) - 1, 10)).astype(int)
                        bin_edges = bin_edges[idx]
                    left_edge = -np.inf
                    for right_edge in bin_edges:
                        # only edges whose bin has values are reasonable cuts
                        # TODO(Optimize) benchmark this vs. just running np.histogram above
                        if any((values > left_edge) & (values <= right_edge)):
                            yield right_edge
                        left_edge = right_edge

    def propose_cuts_numerical_impl_before(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
    ) -> Optional[Any]:
        """
        Perform preparation work before proposing numerical cuts.

        This method is called once before any calls to propose_cuts_numerical_impl
        for a specific cut_choice. It allows implementations to perform setup or
        preprocessing that will be used by all proposed numerical cuts.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        cut_choice : CutChoice
            The cut choice specifying which numerical QID to cut on.

        Returns
        -------
        Any or None
            An optional context object that will be passed to each call of
            propose_cuts_numerical_impl. This can contain any data or state
            needed for efficient cut proposal processing.

        Notes
        -----
        Implementations may override this method to do preprocessing work before
        proposing numerical cuts. There is no equivalent _after method because
        we cannot easily guarantee it will always be called due to the generator
        pattern being employed in implementation.py and core.py.
        """
        return None

    @abstractmethod
    def propose_cuts_numerical_impl(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
        cut: float,
        context: Optional[Any],
    ) -> Optional[list[pd.DataFrame]]:
        """
        Create partitions using a numerical cut point.

        This abstract method must be implemented by concrete classes to create partitions
        based on a numerical cut point for a specific QID. It applies the cut to the data
        and creates partitions accordingly.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        cut_choice : CutChoice
            The cut choice specifying which numerical QID to cut on.
        cut : float
            The specific numerical value to use as the cut point.
        context : Any or None
            Optional context data from propose_cuts_numerical_impl_before.

        Returns
        -------
        List[pd.DataFrame] or None
            If the cut is valid and produces acceptable partitions, returns a list of
            the partitioned DataFrames. If the cut is invalid or doesn't meet the
            implementation's criteria, returns None.

        Notes
        -----
        Implementations of this method should:
        1. Apply the specified cut to partition the data
        2. Validate that the resulting partitions meet the privacy model requirements
        3. Return either the list of partitions or None if the cut is invalid

        The implementation should be specific to the privacy model being enforced.
        """
        pass

    def propose_cuts_categorical(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
        gtree: GTree,
    ) -> Generator[list[pd.DataFrame], None, None]:
        """
        Generate proposed cuts for a categorical QID.

        This method delegates to the implementation-specific propose_cuts_categorical_impl
        method to create partitions based on a categorical quasi-identifier using its
        generalization tree.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        cut_choice : CutChoice
            The cut choice specifying which categorical QID to cut on.
        gtree : GTree
            The generalization tree for this categorical QID.

        Yields
        ------
        List[pd.DataFrame]
            Lists of DataFrames representing the partitions created by each proposed cut,
            for cuts that were deemed valid by the implementation.

        Notes
        -----
        Unlike numerical QIDs which can have multiple possible cut points,
        categorical QIDs typically have one natural partitioning strategy based on
        the generalization tree structure. The method yields any valid partition
        returned by the implementation's propose_cuts_categorical_impl method.
        """
        dfs_if = self.propose_cuts_categorical_impl(nid, input_df, cut_choice, gtree)
        if dfs_if is not None:
            yield dfs_if

    @abstractmethod
    def propose_cuts_categorical_impl(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
        gtree: GTree,
    ) -> Optional[list[pd.DataFrame]]:
        """
        Create partitions using a categorical QID's generalization tree.

        This abstract method must be implemented by concrete classes to create partitions
        based on a categorical quasi-identifier using its generalization tree. It partitions
        the data according to the generalization hierarchy.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be considered for cutting.
        cut_choice : CutChoice
            The cut choice specifying which categorical QID to cut on.
        gtree : GTree
            The generalization tree for this categorical QID.

        Returns
        -------
        List[pd.DataFrame] or None
            If a valid partitioning can be created, returns a list of the partitioned
            DataFrames. If no valid partitioning can be created (e.g., because the
            resulting partitions would not meet privacy requirements), returns None.

        Notes
        -----
        Implementations of this method should:
        1. Use the generalization tree structure to create a logical partitioning
        2. Validate that the resulting partitions meet the privacy model requirements
        3. Return either the list of partitions or None if no valid partitioning is possible

        The implementation should be specific to the privacy model being enforced.
        """
        pass

    def is_proposed_cut_appropriate(
        self,
        proposed_cut: ProposedCut,
        input_df: pd.DataFrame,
        max_suppression_n: int,
        gtrees: dict[str, GTree],
    ) -> bool:
        """
        Determine if a proposed cut is appropriate for further consideration.

        This method evaluates whether a proposed cut is suitable for further processing.
        The base implementation simply checks if the cut produces more than one partition,
        but subclasses may override this to implement more complex validation logic.

        Parameters
        ----------
        proposed_cut : ProposedCut
            The proposed cut to evaluate.
        input_df : pd.DataFrame
            The original DataFrame partition that the cut would be applied to.
        max_suppression_n : int
            Maximum number of records that can be suppressed.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        bool
            True if the proposed cut is appropriate for further consideration,
            False otherwise.

        Notes
        -----
        The base implementation only checks if the cut produces multiple partitions
        (i.e., if it actually divides the data) OR produces one partition and
        suppresses some rows. Subclasses may implement more complex validation
        logic based on the specific privacy model requirements.
        """
        dfs_remaining = len(proposed_cut.further_cuts) + len(proposed_cut.output_dfs)
        # Accept cuts that either:
        # 1. Produce multiple partitions (meaningful data division)
        # 2. Produce one partition but suppress some records (privacy protection through suppression)
        return (dfs_remaining > 1) or (
            dfs_remaining == 1
            and (
                len(input_df)
                > (
                    # Count records remaining after cut (further_cuts + output_dfs)
                    (
                        sum(len(df) for df in proposed_cut.further_cuts)
                        if proposed_cut.further_cuts
                        else 0
                    )
                    + (
                        sum(len(df) for df in proposed_cut.output_dfs)
                        if proposed_cut.output_dfs
                        else 0
                    )
                )
            )
        )

    def score_proposed_cut(
        self,
        proposed_cut: ProposedCut,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> tuple[float, float]:
        """
        Score a proposed cut to evaluate its quality. Lower scores are better.

        This method calculates a quality score for a proposed cut to determine how well
        it preserves data utility while meeting privacy requirements. Scores are used to
        select the best cut among multiple possibilities.

        Parameters
        ----------
        proposed_cut : ProposedCut
            The proposed cut to score.
        input_df : pd.DataFrame
            The original DataFrame partition that the cut would be applied to.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        Tuple[float, float]
            A tuple of two scores:
            - score_1: Primary score based on data utility preservation (lower is better)
            - score_2: Secondary score/tiebreaker based on % records not suppressed (lower is oddly better)

        Notes
        -----
        The method performs these steps:
        1. Computes a tiebreaker score based on relative partition size
        2. Samples the data (if large) to efficiently compute utility metrics
        3. Delegates to specific scoring functions for numerical or categorical QIDs
        4. Returns the utility score and tiebreaker as a tuple

        For two cuts with the same utility score, we prefer to select cuts that will
        suppress data earlier in the Mondrian cut tree, as this makes better use of
        the suppression budget.
        """
        # the tiebreaker between two proposed cuts with the same score_1
        # is pct_non_suppressed: we *prefer* to suppress earlier in
        # the Mondrian cut tree because it should make better use of the suppression budget.
        score_2 = len(proposed_cut.input_df) / len(input_df)
        # sample down dfs to limit run-time complexity w.r.t. df lengths
        input_df = self.__score_proposed_cut_sample(proposed_cut.input_df, 42)
        output_dfs = [
            self.__score_proposed_cut_sample(output_df, 42) for output_df in proposed_cut.output_dfs
        ]
        further_cuts = [
            self.__score_proposed_cut_sample(further_cut, 42)
            for further_cut in proposed_cut.further_cuts
        ]
        gtree_if = gtrees.get(proposed_cut.cut_choice.qid, None)
        if gtree_if is None:
            score_1 = self.score_proposed_cut_numerical(
                proposed_cut.cut_choice.qid,
                input_df,
                output_dfs,
                further_cuts,
            )
        else:
            score_1 = self.score_proposed_cut_categorical(
                proposed_cut.cut_choice.qid,
                gtree_if,
                input_df,
                output_dfs,
                further_cuts,
            )
        return (score_1, score_2)

    def __score_proposed_cut_sample(
        self,
        df: pd.DataFrame,
        random_state: int,
        max_size: int = 1_000,
    ) -> pd.DataFrame:
        """
        Sample a DataFrame to limit computational complexity when scoring cuts.

        This helper method creates a deterministic sample of a DataFrame when it exceeds
        a specified size threshold. This helps maintain reasonable performance when
        scoring cuts on large datasets.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to potentially sample.
        random_state : int
            Seed for the random number generator to ensure deterministic sampling.
        max_size : int, default 1_000
            Maximum size threshold. When the DataFrame exceeds this size, it will be sampled.
            - A higher value leads to more accurate scoring at the cost of increased runtime
            - A lower value leads to faster scoring at the cost of reduced accuracy

        Returns
        -------
        pd.DataFrame
            Either the original DataFrame if it's small enough, or a sampled version
            if it exceeds the max_size threshold.

        Notes
        -----
        Using deterministic sampling ensures reproducible results while still
        maintaining reasonable performance characteristics for large datasets.
        """
        if len(df) <= max_size:
            return df
        return df.sample(n=max_size, random_state=random_state)

    def score_proposed_cut_categorical(
        self,
        qid: str,
        gtree: GTree,
        input_df: pd.DataFrame,
        output_dfs: list[pd.DataFrame],
        further_cuts: list[pd.DataFrame],
    ) -> float:
        """
        Score a proposed cut for a categorical QID.

        This method calculates a quality score for a proposed cut based on a categorical
        quasi-identifier. The score is derived from the standard deviation of the values
        before and after the cut, measuring how well the cut preserves data distribution.

        Parameters
        ----------
        qid : str
            The categorical QID column name that the cut is based on.
        gtree : GTree
            The generalization tree for this categorical QID.
        input_df : pd.DataFrame
            The original DataFrame partition before the cut.
        output_dfs : List[pd.DataFrame]
            List of final cut DataFrames resulting from the proposed cut.
        further_cuts : List[pd.DataFrame]
            List of DataFrames for further partitioning from the proposed cut.

        Returns
        -------
        float
            A score in the range [-1.0, 1.0], where lower values indicate better cuts.
            Returns np.nan if the original data has no variance (standard deviation = 0).

        Notes
        -----
        The score is calculated as:
        (new_distance / original_distance) - 1.0

        Where:
        - original_distance is the standard deviation of the QID values in the input DataFrame
        - new_distance is the standard deviation across all resulting partitions

        This produces a score where:
        - Negative values indicate the cut reduces variation within partitions (good)
        - Zero indicates no change in variation
        - Positive values indicate increased variation within partitions (bad)
        """
        original_distance = standard_deviation_categorical(cast(pd.Series, input_df[qid]))
        if original_distance == 0:
            return np.nan
        new_distance = standard_deviation_categorical(
            *it.chain(
                (cast(pd.Series, output_df[qid]) for output_df in output_dfs),
                (cast(pd.Series, further_cut[qid]) for further_cut in further_cuts),
            )
        )
        return (new_distance / original_distance) - 1.0  # [-1.0, 1.0]: lower is better

    def score_proposed_cut_numerical(
        self,
        qid: str,
        input_df: pd.DataFrame,
        output_dfs: list[pd.DataFrame],
        further_cuts: list[pd.DataFrame],
    ) -> float:
        """
        Score a proposed cut for a numerical QID.

        This method calculates a quality score for a proposed cut based on a numerical
        quasi-identifier. The score is derived from the standard deviation of the values
        before and after the cut, measuring how well the cut preserves data distribution.

        Parameters
        ----------
        qid : str
            The numerical QID column name that the cut is based on.
        input_df : pd.DataFrame
            The original DataFrame partition before the cut.
        output_dfs : List[pd.DataFrame]
            List of final cut DataFrames resulting from the proposed cut.
        further_cuts : List[pd.DataFrame]
            List of DataFrames for further partitioning from the proposed cut.

        Returns
        -------
        float
            A score in the range [-1.0, 1.0], where lower values indicate better cuts.
            Returns np.nan if the original data has no variance (standard deviation = 0).

        Notes
        -----
        The score is calculated as:
        (new_distance / original_distance) - 1.0

        Where:
        - original_distance is the standard deviation of the QID values in the input DataFrame
        - new_distance is the standard deviation across all resulting partitions

        This produces a score where:
        - Negative values indicate the cut reduces variation within partitions (good)
        - Zero indicates no change in variation
        - Positive values indicate increased variation within partitions (bad)

        For numerical QIDs, the standard deviation is calculated directly on the numeric values,
        unlike categorical QIDs which require special handling.
        """
        original_distance = standard_deviation(input_df[qid].to_numpy())
        if original_distance == 0:
            return np.nan
        new_distance = standard_deviation(
            *it.chain(
                (output_df[qid].to_numpy() for output_df in output_dfs),
                (further_cut[qid].to_numpy() for further_cut in further_cuts),
            )
        )
        return (new_distance / original_distance) - 1.0  # [-1.0, 1.0]: lower is better

    def validate(self, input_df: pd.DataFrame, qids: list[str]) -> bool:
        """
        Validate if a DataFrame meets the technical privacy model requirements.

        This method checks whether a DataFrame satisfies the privacy requirements
        of the technical model being implemented (e.g., k-anonymity, u-anonymity).
        Concrete implementation classes must override this method.

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame to validate against the privacy model.
        qids : List[str]
            List of quasi-identifier column names.

        Returns
        -------
        bool
            True if the DataFrame meets the technical privacy model requirements,
            False otherwise.

        Notes
        -----
        The base implementation always returns True, but concrete implementations
        should override this method to implement the specific checks required by
        their privacy model.
        """
        return True

    def are_cuts_possible(
        self, input_df: pd.DataFrame, qids: list[str], gtrees: dict[str, GTree]
    ) -> bool:
        """
        Check if further cuts are possible while maintaining privacy requirements.

        This method determines whether additional cuts can be made to the DataFrame
        while still satisfying the technical privacy model. It is a preliminary check
        before attempting to identify specific cut choices.

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame partition to evaluate for potential further cuts.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        bool
            True if further cuts are possible while maintaining privacy requirements,
            False otherwise.

        Notes
        -----
        The base implementation always returns True, but concrete implementations
        should override this method to reflect the specific constraints of their
        privacy model.

        This method is called before attempting to identify specific cut choices,
        allowing for an early exit if no valid cuts are possible.
        """
        return True

    def is_a_further_cut(
        self,
        partition_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> bool:
        """
        Check if a partition can be further cut while maintaining privacy.

        This method determines whether a specific partition could be cut further
        while still satisfying the technical privacy model requirements. It is used
        to classify partitions during the anonymization process.

        Parameters
        ----------
        partition_df : pd.DataFrame
            The DataFrame partition to evaluate.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        bool
            True if the partition can be further cut while maintaining privacy requirements,
            False otherwise.

        Notes
        -----
        The base implementation always returns False, but concrete implementations
        should override this method to reflect the specific constraints of their
        privacy model.

        This method is used by classify_and_merge_partitions to determine which
        partitions should be processed further versus which are ready for final
        generalization.
        """
        return False

    def could_be_a_final_cut(
        self,
        partition_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> bool:
        """
        Check if a partition could become a valid final cut.

        This method determines whether a partition could potentially satisfy the technical
        privacy model requirements after generalization and thus become a valid final cut.
        It is used to identify partitions that should be kept versus those that must be suppressed.

        Parameters
        ----------
        partition_df : pd.DataFrame
            The DataFrame partition to evaluate.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        bool
            True if the partition could become a valid final cut after generalization,
            False otherwise.

        Notes
        -----
        The default implementation simply delegates to is_a_further_cut, which is often
        a conservative but correct approach. However, implementations may override this
        method to implement more precise criteria specific to their privacy model.

        This method is used in both classify_and_merge_partitions and generalize_and_suppress
        to determine which partitions should be kept versus suppressed.
        """
        return self.is_a_further_cut(
            partition_df,
            qids,
            gtrees,
        )

    def generate_final_cuts(
        self,
        nid: str,
        proposed_cut: ProposedCut,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], int]:
        """
        Generate final cuts for a proposed cut and apply generalization.

        This method processes a proposed cut, applies generalization to all output
        partitions that should become final cuts, and calculates the number of
        records suppressed in the process.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        proposed_cut : ProposedCut
            The proposed cut to finalize.
        input_df : pd.DataFrame
            The original DataFrame partition that was cut.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        Tuple[List[pd.DataFrame], List[pd.DataFrame], int]
            A tuple containing:
            - further_cuts: List of DataFrames for further partitioning
            - final_cuts: List of generalized DataFrames that are final cuts
            - n_suppressed: Number of records that were suppressed

        Notes
        -----
        The method:
        1. Takes the proposed cut's further_cuts directly
        2. Processes each output_df through generalize_and_suppress
        3. Collects valid final cuts (those that meet privacy requirements)
        4. Calculates the number of records suppressed during the process

        Records are considered suppressed if they are not included in any further_cuts
        or final_cuts DataFrame.
        """
        further_cuts = proposed_cut.further_cuts
        final_cuts = []
        for output_df in proposed_cut.output_dfs:
            final_cut_if = self.generalize_and_suppress(nid, output_df, qids, gtrees)
            if final_cut_if is not None:
                final_cuts.append(final_cut_if)
        n_non_suppressed = 0
        for dfs in (final_cuts, further_cuts):
            if dfs:
                n_non_suppressed += sum(len(df) for df in dfs)
        n_suppressed = len(input_df) - n_non_suppressed
        return further_cuts, final_cuts, n_suppressed

    def generalize_and_suppress(
        self,
        nid: str,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> Optional[pd.DataFrame]:
        """
        Transform a partition into a final cut through generalization and suppression.

        This method applies generalization to QID values in a partition and potentially
        suppresses records that cannot be adequately anonymized. It converts a raw
        partition into an anonymized final cut that meets the privacy requirements.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to generalize and potentially suppress.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        pd.DataFrame or None
            The generalized DataFrame if it meets privacy requirements, or
            None if the partition cannot be adequately anonymized and should
            be entirely suppressed.

        Notes
        -----
        The method performs these steps:

        1. First checks if the partition could be a valid final cut
        2. For each categorical QID, generalizes values to the lowest common ancestor
           in the generalization tree
        3. For each numerical QID, applies micro-aggregation by replacing values with
           the mean if there are multiple distinct values
        4. Optionally suppresses records with excessive generalization
        5. Performs a final validation to ensure privacy requirements are met

        Implementations that add columns during setup should override this method
        to remove those columns before calling super().
        """
        if not self.could_be_a_final_cut(input_df, qids, gtrees):
            return None
        output_df = input_df.copy()
        all_qids_have_local_suppression = True
        for qid in qids:
            if any(pd.isna(output_df[qid])):
                # because in core.anonymize(..) we first partitioned by the existence of nan values, this means
                # that if there is any non-nan value for a qid then no values for that qid are nan--thus
                # if any is nan, then all must be nan and no change to values is needed
                qid_has_local_suppression = False  # but this isn't considered local suppression
            else:
                gtree_if = gtrees.get(qid, None)
                if gtree_if is None:
                    # rewrite to micro-aggregated value if there are multiple origin values
                    if atleast_nunique(output_df[qid].to_numpy(), 2):
                        output_df[qid] = output_df[qid].mean()
                    qid_has_local_suppression = False
                else:
                    root_node = gtree_if.get_node(gtree_if.root)
                    assert root_node is not None, f"Root node not found in gtree for {qid}"
                    root_value = gtree_if.get_value(root_node)
                    uniq_orig_values = set(output_df[qid].unique())
                    lowest_node = gtree_if.get_lowest_node_with_descendant_leaves(
                        uniq_orig_values,
                    )
                    assert lowest_node is not None, f"Lowest node not found for values in {qid}"
                    # rewrite to the lowest node that contains all values
                    lowest_value = gtree_if.get_value(lowest_node)
                    output_df[qid] = lowest_value
                    qid_has_local_suppression = (
                        lowest_value == root_value  # we've recoded to root
                        and len(uniq_orig_values - set([root_value]))
                        > 0  # at-least one of the values wasn't orginally root
                    )
            all_qids_have_local_suppression &= qid_has_local_suppression
        # if all QIDs had local suppression, then check and suppress records where all qid cells have been suppressed
        # this can also occur if output_df couldn't satisfy the implementation's technical privacy model
        if all_qids_have_local_suppression and qids:
            output_df = output_df[
                ft.reduce(
                    np.logical_or,
                    it.chain(
                        (
                            (
                                # only keep non-* values
                                (
                                    output_df[qid]
                                    != gtrees[qid].get_value(
                                        # gtree is guaranteed to have a root Node when gtree exists in dict
                                        cast(
                                            Node, gtrees[qid].get_node(cast(str, gtrees[qid].root))
                                        )
                                    )
                                )
                                &
                                # only keep non-nan values
                                (~pd.isna(output_df[qid]))
                            )
                            if qid in gtrees.keys()
                            # only keep non-nan values
                            else ~pd.isna(output_df[qid])
                        )
                        for qid in qids
                    ),
                )
            ]
        if not self.could_be_a_final_cut(cast(pd.DataFrame, output_df), qids, gtrees):
            return None
        return cast(pd.DataFrame, output_df)
