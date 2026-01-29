"""
Core Mondrian algorithm.

This module implements an extension of the Mondrian algorithm [LeFevre2006]_ for multidimensional
data anonymization. The algorithm recursively partitions the data space and applies
domain-specific generalization strategies to achieve anonymity. The implementation
is designed to be extensible through different implementation classes that define
behaviors for different technical privacy models [Bloomston2025a]_.

Key Enhancements Over Original Mondrian Algorithm:
---------------------------------------------------
1. Strategy Pattern Architecture: Uses an implementation interface that allows different
   privacy models to share core functionality while customizing key behaviors.

2. Hybrid Execution Model: Combines both recursive and queue-based processing with
   parallel execution capabilities for improved scalability.

3. Sophisticated Missing Value Handling: Creates separate initial partitions for records
   with missing values in different attribute combinations.

4. Dynamic Suppression Budget Allocation: Proportionally distributes suppression budget
   across partitions based on their sizes.

5. Configurable Cut Selection: Supports multiple methods for numerical cut point selection
   (median, histogram bins) and customizable scoring.

6. Timeout-Based Early Termination: Graceful handling of timeouts while still ensuring
   completed partitions meet privacy requirements.

7. Detailed Statistics Tracking: Optional collection of partition statistics for algorithm
   tuning and quality assessment.

References
----------
.. [LeFevre2006] LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006, April).
       Mondrian multidimensional k-anonymity. In 22nd International Conference on
       Data Engineering (ICDE'06) (pp. 25-25). IEEE.
.. [Bloomston2025a] Bloomston, A., Burke, E., Cacace, M., Diaz, A., Dougherty, W., Gonzalez, M.,
       Gregg, R., Güngör, Y., Hayes, B., Hsu, E., Israeli, O., Kim, H., Kwasnick, S.,
       Lacsina, J., Rodriguez, D. R., Schiller, A., Schumacher, W., Simon, J., Tang, M.,
       Wharton, S., & Wilcken, M. (2025). Core Mondrian: Basic Mondrian beyond k-anonymity.
       arXiv preprint arXiv:2510.09661. https://arxiv.org/abs/2510.09661
"""

import concurrent.futures as cf
import logging
import math
import multiprocessing as mp
import threading
import time
import warnings
from collections import deque
from typing import Any, Generator, Optional, cast

import pandas as pd
from first import first  # type: ignore[import-untyped]

from project_lighthouse_anonymize.gtrees import GTree
from project_lighthouse_anonymize.mondrian.funnel_stats import FunnelStats
from project_lighthouse_anonymize.mondrian.implementation import Implementation_Base
from project_lighthouse_anonymize.mondrian.tree import (
    MondrianTree,
    Node_Cut,
    Node_DeferredCutData,
    Node_DeferredCutData__PartitionRoot,
    Node_FinalCutData,
    ProposedCut,
)
from project_lighthouse_anonymize.utils import (
    atleast_nunique,
    atleast_nunique_categorical,
    nan_generator,
    parallelism__cleanup_logging,
    parallelism__get_qid_to_gtree,
    parallelism__initializer_cleanup,
    parallelism__make_initializer_and_initargs,
    parallelism__setup_logging,
)


class CoreMondrian:
    """
    Core Mondrian algorithm implementation for recursive data partitioning and anonymization.

    This class orchestrates the anonymization process through recursive partitioning
    of the data according to the specified implementation strategy. It manages both
    the core algorithm flow and optional parallelization of the partitioning process.

    Extensions Beyond Original Mondrian:
    -------------------------------------
    While the original Mondrian algorithm focused specifically on k-anonymity with a
    straightforward recursive implementation, CoreMondrian offers:

    1. Strategy-Based Design: Delegates privacy model-specific logic to implementation
       classes, allowing support for k-anonymity, u-anonymity, and other models.

    2. Performance Optimizations:

       - Hybrid recursive/queue-based processing for better memory efficiency
       - Optional parallelization for large datasets using process pools
       - Optimized handling of partitions based on size thresholds

    3. Enhanced Suppression Management:

       - Sophisticated suppression budget allocation across partitions
       - Proportional distribution of suppression budget based on partition size
       - Proper tracking and accounting of suppression across the partition hierarchy

    4. Practical Usability Features:

       - Configurable timeout with graceful termination
       - Optional statistics tracking for algorithm performance analysis
       - Support for deterministic execution for testing and validation

    The high-level process is as follows::

        # Create objects
        implementation = Implementation_Base(..)  # A concrete implementation
        mondrian = CoreMondrian(implementation, ..)

        # Main anonymization function
        anonymized_df = mondrian.anonymize(input_df, qids, gtrees, ..)
          self.implementation.setup(..)

          # Core recursive function that processes each partition
          self.make_cut(partition_data, node_id)

            # Try to find valid cuts for this partition
            for cut, further_cuts, final_cuts in self.make_cut_generate_final_cut():

              # Check if cuts are possible and find the best ones
              if self.implementation.are_cuts_possible():
                for cut_choice in self.implementation.cut_choices():
                  for proposed_cut in self.implementation.propose_cuts():
                    if self.implementation.is_proposed_cut_appropriate():
                      # Keep valid proposed cut

              # Process the best proposed cuts
              self.make_cut_generate_final_cut_from_proposed_cuts():
                self.implementation.score_proposed_cut()
                self.implementation.generate_final_cuts()

              # Recursive processing of smaller partitions
              for partition in recursive_cuts:
                self.make_cut()  # Recursive call

          return tree.candidate_solution()

    Parameters
    ----------
    logger : logging.Logger
        Logger for recording the anonymization process.
    implementation : Implementation_Base
        Concrete implementation of anonymization strategy.
    recursive_partition_size_cutoff : int
        Threshold for partition size to determine whether to process recursively.
    parallelism : int or None
        Number of parallel workers to use, None disables parallelism.
    deterministic_identifiers : bool, default False
        If True, generates deterministic node identifiers.
    track_stats : bool, default False
        If True, tracks detailed statistics about the anonymization process.

    Notes
    -----
    The algorithm uses a hybrid recursive/queue-based approach for processing data partitions,
    with optional parallelization for larger datasets. The implementation
    determines specific behaviors related to the technical privacy model being enforced.
    """

    def __init__(
        self,
        logger: logging.Logger,
        implementation: Implementation_Base,
        recursive_partition_size_cutoff: int,
        parallelism: Optional[int],
        deterministic_identifiers: bool = False,
        track_stats: bool = False,
    ):
        """
        Initialize the CoreMondrian algorithm instance.

        Parameters
        ----------
        logger : logging.Logger
            Logger for recording the anonymization process.
        implementation : Implementation_Base
            Concrete implementation of anonymization strategy.
        recursive_partition_size_cutoff : int
            Breakpoint for jobs running in the queue vs. recursively run.
            This parameter is particularly important when parallelism is enabled:
            - A higher value means more jobs are run recursively, which may reduce
              effective parallelism and, at the extremes, lead to maximum recursion depth.
            - A lower value means fewer jobs are run recursively, which may increase
              effective parallelism at the cost of increased parallelism overhead.
            - This not only affects performance, it ALSO affects suppression because
              when running recursively the suppression budget is shared among cut children!
        parallelism : int or None
            Number of parallel workers to use. None means run without parallelism.
        deterministic_identifiers : bool, default False
            If True, generates deterministic node identifiers for reproducible results.
        track_stats : bool, default False
            If True, tracks detailed statistics about the anonymization process.
        """
        self.logger = logger
        self.implementation = implementation
        self.recursive_partition_size_cutoff = recursive_partition_size_cutoff
        self.parallelism = parallelism
        self.qids: list[str] = []
        self.lookup_args: Any = None
        self.timeout_hit = False
        self.running = False
        self.deterministic_identifiers = deterministic_identifiers
        self.track_stats = track_stats

        if self.debug_logging_enabled():
            warnings.warn(
                "DEBUG logging is enabled for Core Mondrian, this WILL adversely impact performance."
            )

    def debug_logging_enabled(self) -> bool:
        """
        Check if debug logging is enabled.

        Returns
        -------
        bool
            True if the logger is configured for DEBUG level logging, False otherwise.

        Notes
        -----
        This is used throughout the code to conditionally execute debug logging
        statements which can significantly impact performance when enabled.
        """
        return self.logger.isEnabledFor(logging.DEBUG)

    def make_cut_generate_final_cut_from_proposed_cuts(
        self,
        input_df: pd.DataFrame,
        max_suppression_n: int,
        nid: str,
        gtrees: dict[str, GTree],
        proposed_cuts: list[ProposedCut],
        stats: Optional[FunnelStats],
    ) -> Generator[tuple[Node_Cut, list[pd.DataFrame], list[pd.DataFrame]], None, None]:
        """
        Generate final cuts from a list of proposed cuts.

        This method processes a list of proposed cuts, scores them (if there are multiple),
        and generates final cuts for each proposal. It yields those final cuts that stay
        within the suppression budget.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame to be partitioned.
        max_suppression_n : int
            Maximum number of records that can be suppressed.
        nid : str
            Node identifier for logging purposes.
        gtrees : Dict[str, GTree]
            Mapping from QID column names to generalization trees.
        proposed_cuts : List[ProposedCut]
            List of proposed cuts to consider.
        stats : FunnelStats, optional
            If provided, tracks statistics about the cut process.

        Yields
        ------
        Tuple[Node_Cut, List[pd.DataFrame], List[pd.DataFrame]]
            For each acceptable proposed cut:
            - The Node_Cut representing the cut
            - A list of DataFrames for further partitioning
            - A list of DataFrames that are final cuts, i.e., will not be further partitioned

        Notes
        -----
        This method first scores and sorts multiple proposed cuts if there are more than one.
        For each proposed cut, it generates final cuts and checks if they meet the suppression
        budget. Only acceptable cuts (those that don't exceed the suppression budget) are yielded.
        """
        if len(proposed_cuts) > 1:
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - multiple proposed cuts, scoring and sorting by proposed cut score (lower is better)",
                    nid,
                )
            proposed_cut_and_scores = [
                (
                    proposed_cut,
                    self.implementation.score_proposed_cut(
                        proposed_cut,
                        input_df,
                        self.qids,
                        gtrees,
                    ),
                )
                for proposed_cut in proposed_cuts
            ]
            proposed_cut_and_scores = sorted(
                proposed_cut_and_scores,
                key=lambda pcs: pcs[1],
            )
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - proposed cut and scores:\n%s",
                    nid,
                    "\n".join(
                        [
                            f"{proposed_cut} == ({score[0]:.3f}, {score[1]:.3f})"
                            for proposed_cut, score in proposed_cut_and_scores
                        ]
                    ),
                )
            proposed_cuts = [proposed_cut for proposed_cut, _ in proposed_cut_and_scores]
        for proposed_cut in proposed_cuts:
            further_cuts = proposed_cut.further_cuts
            further_cuts, final_cuts, n_suppressed = self.implementation.generate_final_cuts(
                nid,
                proposed_cut,
                input_df,
                self.qids,
                gtrees,
            )
            if stats is not None:
                stats.increment_final(
                    proposed_cut.cut_choice.qid, proposed_cut.cut_choice.is_categorical
                )
            cut = self.make_node_cut(
                proposed_cut,
                input_df,
                max_suppression_n,
                further_cuts,
                final_cuts,
                n_suppressed,
            )
            if n_suppressed <= max_suppression_n:
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "%s - accepted final cut %s with further_cuts = %s, final_cuts = %s",
                        nid,
                        cut,
                        [len(df) for df in further_cuts],
                        [len(df) for df in final_cuts],
                    )
                if stats is not None:
                    stats.increment_accepted(
                        proposed_cut.cut_choice.qid,
                        proposed_cut.cut_choice.is_categorical,
                    )
                yield cut, further_cuts, final_cuts
            else:
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "%s - rejected final cut %s with further_cuts = %s, final_cuts = %s",
                        nid,
                        cut,
                        [len(df) for df in further_cuts],
                        [len(df) for df in final_cuts],
                    )

    def make_cut_generate_final_cut(
        self,
        input_df: pd.DataFrame,
        max_suppression_n: int,
        nid: str,
        gtrees: dict[str, GTree],
        stats: Optional[FunnelStats],
    ) -> Generator[tuple[Node_Cut, list[pd.DataFrame], list[pd.DataFrame]], None, None]:
        """
        Generate a final cut for a partition by considering possible cut choices.

        This method examines a partition to determine if cuts are possible, evaluates
        potential cut choices, and proposes specific cuts. It ultimately generates
        final cuts for the best proposals, or returns a base case cut if no suitable
        cuts can be made.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame partition to be considered for cutting.
        max_suppression_n : int
            Maximum number of records that can be suppressed.
        nid : str
            Node identifier for logging purposes.
        gtrees : Dict[str, GTree]
            Mapping from QID column names to generalization trees.
        stats : FunnelStats, optional
            If provided, tracks statistics about the cut process.

        Yields
        ------
        Tuple[Node_Cut, List[pd.DataFrame], List[pd.DataFrame]]
            For each acceptable cut:
            - The Node_Cut representing the cut
            - A list of DataFrames for further partitioning
            - A list of DataFrames that are final cuts, i.e., will not be further partitioned

        Notes
        -----
        This method first determines if cuts are possible using the implementation's
        are_cuts_possible method. If cuts are possible, it considers QIDs with at least
        two unique values for cutting, examines cut choices, proposes specific cuts,
        and evaluates their appropriateness. If no suitable cuts are found, or if cuts
        are not possible, a base case cut is generated.
        """
        if self.debug_logging_enabled():
            self.logger.debug(
                "%s - attempting to make cut on partition with cardinality %d",
                nid,
                len(input_df),
            )
        if not self.timeout_hit and self.implementation.are_cuts_possible(
            input_df, self.qids, gtrees
        ):
            qids_to_consider = []
            # we consider a QID for cut choices only if it has at-least two unique values. because
            # in anonymize(..) below we first partitioned by the existence of nan values, this means
            # that if there is any non-nan value for a qid then no values for that qid are nan--thus
            # if we consider a cut below for a qid then we know all its values are not nan.
            for qid in self.qids:
                gtree_if = gtrees.get(qid, None)
                if gtree_if is None:
                    # numerical
                    if atleast_nunique(input_df[qid].to_numpy(), 2):
                        qids_to_consider.append(qid)
                else:
                    # categorical
                    if atleast_nunique_categorical(cast(pd.Series, input_df[qid]), 2):
                        qids_to_consider.append(qid)
            proposed_cuts: list[ProposedCut] = []
            all_proposed_cuts = []
            prior_cut_choice = None
            for cut_choice in self.implementation.cut_choices(
                nid, input_df, gtrees, qids_to_consider
            ):
                if not self.implementation.keep_examining_cut_choices(
                    nid, prior_cut_choice, cut_choice, input_df
                ):
                    yield from self.make_cut_generate_final_cut_from_proposed_cuts(
                        input_df,
                        max_suppression_n,
                        nid,
                        gtrees,
                        proposed_cuts,
                        stats,
                    )
                    all_proposed_cuts.extend(proposed_cuts)
                    proposed_cuts.clear()
                    prior_cut_choice = None

                if stats is not None:
                    stats.increment_examined(cut_choice.qid, cut_choice.is_categorical)
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "%s - examining cut_choice %s",
                        nid,
                        cut_choice,
                    )
                for proposed_cut in self.implementation.propose_cuts(
                    nid, input_df, self.qids, gtrees, cut_choice, stats
                ):
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "%s - cut_choice %s yielded proposed cut for consideration %s",
                            nid,
                            cut_choice,
                            proposed_cut,
                        )
                    if self.implementation.is_proposed_cut_appropriate(
                        proposed_cut, input_df, max_suppression_n, gtrees
                    ):
                        if self.debug_logging_enabled():
                            self.logger.debug(
                                "%s - proposed cut %s (max_suppression_n = %d) deemed appropriate for a potential final cut",
                                nid,
                                proposed_cut,
                                max_suppression_n,
                            )

                        if stats is not None:
                            stats.increment_proposed(
                                proposed_cut.cut_choice.qid,
                                proposed_cut.cut_choice.is_categorical,
                            )

                        proposed_cuts.append(proposed_cut)
                        prior_cut_choice = cut_choice
                    else:
                        if self.debug_logging_enabled():
                            self.logger.debug(
                                "%s - proposed cut %s (max_suppression_n = %d) deemed inappropriate for a potential final cut",
                                nid,
                                proposed_cut,
                                max_suppression_n,
                            )
            if proposed_cuts:
                yield from self.make_cut_generate_final_cut_from_proposed_cuts(
                    input_df,
                    max_suppression_n,
                    nid,
                    gtrees,
                    proposed_cuts,
                    stats,
                )
                all_proposed_cuts.extend(proposed_cuts)
                proposed_cuts.clear()
        else:
            if self.debug_logging_enabled():
                self.logger.debug(
                    "%s - no cuts are possible",
                    nid,
                )
        # base case, no further cuts to make
        if self.debug_logging_enabled():
            self.logger.debug("%s - no cut found for partition", nid)
        base_case_proposed_cut = self.implementation.make_base_case_proposed_cut(
            input_df,
            self.qids,
            gtrees,
        )
        further_cuts, final_cuts, n_suppressed = self.implementation.generate_final_cuts(
            nid,
            base_case_proposed_cut,
            input_df,
            self.qids,
            gtrees,
        )
        cut = self.make_node_cut(
            base_case_proposed_cut,
            input_df,
            max_suppression_n,
            further_cuts,
            final_cuts,
            n_suppressed,
        )
        if further_cuts:
            raise RuntimeError("There should be no further cuts at the base case")
        yield (cut, further_cuts, final_cuts)

    def make_cut(
        self,
        data: Node_DeferredCutData,
        nid: str,
    ) -> Optional[tuple[str, MondrianTree]]:
        """
        Process a partition node and create a subtree representing cuts for anonymization.

        This method is the core recursive function that processes each partition.
        It attempts to find valid cuts for the partition, creates a tree structure
        representing the anonymization decisions, and handles the suppression budget
        allocation for child partitions.

        Parameters
        ----------
        data : Node_DeferredCutData
            Data for the partition node to process, including the DataFrame and suppression budget.
        nid : str
            Node identifier for the current partition.

        Returns
        -------
        Tuple[str, MondrianTree]
            - The node identifier of the processed partition
            - A MondrianTree representing the anonymization structure for this partition

        Notes
        -----
        This method:
        1. Attempts to find valid cuts that stay within the suppression budget
        2. Creates a tree structure with the chosen cut as the root
        3. Handles final cuts (leaf nodes) directly
        4. Divides remaining partitions into deferred (large) and recursive (small)
        5. Processes recursive partitions immediately, allocating suppression budget
        6. Creates deferred cut nodes for larger partitions to be processed later

        The suppression budget is carefully managed across the partitioning to ensure
        that the total suppression doesn't exceed the maximum allowed.
        """
        gtrees = parallelism__get_qid_to_gtree(*self.lookup_args)
        stats = FunnelStats() if self.track_stats else None
        # Try cuts in sequence until we find one that stays within the suppression budget
        for cut, further_cuts, final_cuts in self.make_cut_generate_final_cut(
            data.df,
            data.max_suppression_n,
            nid,
            gtrees,
            stats,
        ):
            if self.debug_logging_enabled():
                self.logger.debug("%s - make_cut processing cut %s", nid, cut)
            try:
                tree = MondrianTree(deterministic_identifiers=self.deterministic_identifiers)
                root = tree.create_node(None, cut)
                if stats is not None:
                    cut.attach_stats(stats)
                for final_cut in final_cuts:
                    tree.create_node(root, self.make_node_final_cut(final_cut))
                total_n = sum(len(df) for df in further_cuts) if further_cuts else 0
                deferred_cuts, recursive_cuts = (
                    [
                        cut_df
                        for cut_df in further_cuts
                        if len(cut_df) >= self.recursive_partition_size_cutoff
                    ],
                    [
                        cut_df
                        for cut_df in further_cuts
                        if len(cut_df) < self.recursive_partition_size_cutoff
                    ],
                )
                max_suppression_n = max(0, data.max_suppression_n - cut.suppression_n)
                # Proportional budget allocation strategy: distribute suppression budget across cuts
                # based on their relative size to prevent any single cut from consuming the entire budget
                max_suppression_n_remaining = max_suppression_n
                for cut_df in deferred_cuts:
                    # Allocate suppression budget proportionally: (cut_size / total_size) * budget
                    # Use int() truncation to ensure budget is never exceeded due to rounding
                    sub_max_suppression_n = int((len(cut_df) / total_n) * max_suppression_n)
                    tree.create_node(
                        root,
                        self.make_node_deferred_cut(
                            cut_df.copy(),  # copy to get rid of views
                            sub_max_suppression_n,
                        ),
                    )
                    max_suppression_n_remaining -= sub_max_suppression_n
                # Greedy processing strategy: process larger recursive cuts first to make
                # best use of the suppression budget
                recursive_cuts = sorted(recursive_cuts, key=len, reverse=True)
                for recursive_cut_df in recursive_cuts:
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "%s - processing recursive cut with max_suppression_n_remaining = %d",
                            nid,
                            max_suppression_n_remaining,
                        )
                    parent_data = self.make_node_deferred_cut(
                        recursive_cut_df,
                        max_suppression_n_remaining,
                    )
                    parent_node = tree.create_node(root, parent_data)
                    result = self.make_cut(
                        parent_data,
                        parent_node.identifier,
                    )
                    if result is None:
                        raise RuntimeError(
                            f"make_cut unexpectedly returned None for recursive cut at node {nid}"
                        )
                    sub_nid, sub_tree = result
                    sub_total_suppressed = tree.stitch_in_subtree_recursive(sub_nid, sub_tree)
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "%s - recursive cut finished with sub_total_suppressed = %d",
                            nid,
                            sub_total_suppressed,
                        )
                    max_suppression_n_remaining -= sub_total_suppressed
                    if max_suppression_n_remaining < 0:
                        if self.debug_logging_enabled():
                            self.logger.debug(
                                "%s - make_cut bailing on processing cut %s because we exceeded our maximum suppression budget by %d",
                                nid,
                                cut,
                                abs(max_suppression_n_remaining),
                            )
                        raise StopIteration()
                if self.debug_logging_enabled():
                    self.logger.debug("%s - make_cut finished processing cut %s", nid, cut)
                return (nid, tree)
            except StopIteration:
                continue
        return None

    def signal_anonymize_timeout(self, timeout_s: int) -> None:
        """
        Monitor and enforce a timeout for the anonymization process.

        This method runs in a separate thread and sets the timeout_hit flag
        after the specified timeout period if the anonymization is still running.

        Parameters
        ----------
        timeout_s : int
            Timeout period in seconds.

        Notes
        -----
        This method is called in a separate thread from the anonymize method.
        When the timeout is reached, it sets the timeout_hit flag which causes
        the algorithm to finish its current processing and return the best
        anonymization achieved so far, rather than continuing to optimize.
        """
        running_s = 0
        while running_s < timeout_s and self.running:
            time.sleep(1)
            running_s += 1
        if self.running:
            self.logger.warning("Signaling anonymize timeout")
            self.timeout_hit = True
            # each time a job is serialized for parallel processing, self is serialized, including self.timeout_hit

    def anonymize(
        self,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
        exclude_qids: list[str],
        dq_metric_to_minimum_dq: dict[str, float],
        # TODO(Later) try with max_suppression_multiplier > 1.0 then, if
        # minimum data quality not met *only* for pct_non_suppressed,
        # try again with max_suppression_multiplier = 1.0?
        # Also relates to work with all_proposed_cuts in make_cut_generate_final_cut
        max_suppression_multiplier: float = 1.0,
        timeout_s: Optional[int] = None,
        track_qid_cuts: bool = False,
        node_identifier_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Anonymize a dataset using the Core Mondrian algorithm.

        This is the main entry point for anonymization. It orchestrates the complete
        anonymization process, including setting up the implementation, creating the initial
        partitioning, processing partitions (potentially in parallel), and assembling
        the final anonymized dataset.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame to anonymize.
        qids : List[str]
            List of quasi-identifier column names in the input DataFrame.
        gtrees : Dict[str, GTree]
            Mapping from QID column names to generalization trees.
        exclude_qids : List[str]
            QIDs to exclude from computing overall data quality metric minimum scores.
        dq_metric_to_minimum_dq : Dict[str, float]
            Mapping from data quality metric names to minimum acceptable values.
        max_suppression_multiplier : float, default 1.0
            Multiplier for the maximum suppression budget.
        timeout_s : int, optional
            Optional timeout in seconds. If specified, the algorithm will finish early
            if this time is reached.
        track_qid_cuts : bool, default False
            If True, tracks and includes QID cut information in the output.
        node_identifier_col : str, optional
            If provided, adds a column with this name to the output containing node identifiers.

        Returns
        -------
        pd.DataFrame
            The anonymized DataFrame.

        Notes
        -----
        The anonymization process:
        1. Sets up the implementation with the input data
        2. Calculates the suppression budget based on minimum data quality requirements
        3. Creates initial partitioning based on missing values in QIDs
        4. Processes partitions recursively and/or in parallel
        5. Assembles the final anonymized dataset

        The algorithm respects the suppression budget to ensure that the specified
        minimum data quality metrics can be achieved.
        """
        if len(qids) == 0:
            raise ValueError("At least one QID must be specified")
        qids = list(qids)  # ensure qids is a list for pandas column selection operations
        self.qids = qids
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive if specified")
        if len(input_df) < 100:
            # when the input data is small, use floor to prevent always requiring 0 suppression
            min_non_suppressed = math.floor(
                dq_metric_to_minimum_dq.get("pct_non_suppressed", 0.90) * len(input_df)
            )
        else:
            # otherwise use to ceil to ensure we don't exceed the budget by even 1 row
            min_non_suppressed = math.ceil(
                dq_metric_to_minimum_dq.get("pct_non_suppressed", 0.90) * len(input_df)
            )
        max_suppression_n = math.floor(
            max_suppression_multiplier * (len(input_df) - min_non_suppressed)
        )
        if self.debug_logging_enabled():
            self.logger.debug("Starting root max_suppression_n = %d", max_suppression_n)
        if node_identifier_col is not None:
            if node_identifier_col in input_df.columns:
                raise ValueError(
                    f"node_identifier_col '{node_identifier_col}' already exists in input DataFrame columns"
                )
        mp_ctx = mp.get_context("spawn")
        logging_queue, logging_listener, original_handlers, original_logging_level = (
            parallelism__setup_logging(mp_ctx)
        )
        initializer, initargs, lookup_args = parallelism__make_initializer_and_initargs(
            gtrees, logging_queue, original_logging_level
        )
        # each time a job is serialized for parallel processing, self is serialized, including self.timeout_hit
        self.lookup_args = lookup_args
        executor = (
            cf.ProcessPoolExecutor(
                max_workers=self.parallelism,
                mp_context=mp_ctx,
                initializer=initializer,
                initargs=initargs,
            )
            if self.parallelism is not None
            else None
        )
        self.timeout_hit = False
        self.running = True
        if timeout_s is not None:
            thread = threading.Thread(target=self.signal_anonymize_timeout, args=(timeout_s,))
            thread.start()
        else:
            thread = None
        try:
            self.implementation.setup(input_df, qids, gtrees, exclude_qids)
            tree = MondrianTree(deterministic_identifiers=self.deterministic_identifiers)
            # the root is a no-op final cut node
            # this makes processing below much easier because
            # we don't need special cases for the root
            # TODO(Later) find a more elegant approach
            root = tree.create_node(None, self.make_node_final_cut(input_df.sample(0)))
            if self.track_stats:
                root.data.attach_stats(FunnelStats())
            queue: deque[tuple[str, MondrianTree]] = deque([])
            if self.debug_logging_enabled():
                self.logger.debug(
                    "Splitting input data (size: %d) on NaN partitions for QIDs %s",
                    len(input_df),
                    qids,
                )
            # Note: This is one place where suppression may exceed budget, as these initial partition nodes
            # are always created regardless of whether they will meet their suppression budget
            # First, handle the all-NaN QIDs partition separately to determine actual suppression used
            # This allows reallocating any unused suppression budget to the remaining partitions
            i_input_df = input_df[input_df[qids].isna().all(axis=1)]
            max_suppression_n_remaining = max_suppression_n
            if len(i_input_df) > 0:
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "Processing special partition with all NaN QIDs (size: %d) directly with make_cut - handling separately from other partitions",
                        len(i_input_df),
                    )
                # Use floor division (int) instead of rounding to ensure we don't exceed the budget
                sub_max_suppression_n = int((len(i_input_df) / len(input_df)) * max_suppression_n)
                result = self.make_cut(
                    self.make_node_deferred_cut__partition_root(
                        cast(pd.DataFrame, i_input_df.copy()),  # get rid of views
                        sub_max_suppression_n,
                    ),
                    root.identifier,
                )
                if result is None:
                    raise RuntimeError(
                        "make_cut unexpectedly returned None for all-NaN QIDs partition"
                    )
                _, sub_tree = result
                nodes_to_process, suppression_n = tree.stitch_in_subtree(root.identifier, sub_tree)
                assert not nodes_to_process, (
                    "No work should remain when processing partition with all qids nan"
                )
                max_suppression_n_remaining = max(0, max_suppression_n_remaining - suppression_n)
                if self.debug_logging_enabled():
                    self.logger.debug(
                        "All NaN QIDs partition processed: suppression used = %d, remaining suppression budget = %d",
                        suppression_n,
                        max_suppression_n_remaining,
                    )
            # now, examine all the other nan partitions allocating the remaining suppression budget
            for i_input_df, i_cols in nan_generator(input_df, qids):
                # Skip the case where all qid values are nan (handled above)
                if len(i_cols) == 0:
                    continue
                # Use floor division (int) instead of rounding to ensure we don't exceed the budget
                sub_max_suppression_n = int(
                    (len(i_input_df) / len(input_df)) * max_suppression_n_remaining
                )
                sub_tree = MondrianTree(deterministic_identifiers=self.deterministic_identifiers)
                sub_tree.create_node(
                    None,
                    self.make_node_deferred_cut__partition_root(
                        i_input_df.copy(),  # get rid of views
                        sub_max_suppression_n,
                    ),
                )
                assert tree.root is not None, "Tree root should exist at this point"
                queue.append((tree.root, sub_tree))
            futures: list[cf.Future] = []
            futures_processed = 0
            loop_count = 0
            while futures or queue:
                if self.debug_logging_enabled():
                    self.logger.debug("mondrian loop # %d", loop_count)
                # 1. process entire in-process queue
                queue_count = 0
                while queue:
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "mondrian loop # %d queue loop # %d",
                            loop_count,
                            queue_count,
                        )
                    nid, sub_tree = queue.popleft()
                    # We don't use total_suppressed returned from stitch_in_subtree
                    # here because it's very difficult to do so in a way that maintains
                    # determinism when using parallelism
                    nodes_to_process, _ = tree.stitch_in_subtree(nid, sub_tree)
                    while nodes_to_process:
                        node = nodes_to_process.popleft()
                        # if parallelism is enabled, we parallelize all calls, leaving this sub-process
                        # open for processing the results from those calls.
                        if executor is not None:
                            future = executor.submit(
                                self.make_cut,
                                node.data,
                                node.identifier,
                            )
                            futures.append(future)
                        else:
                            result = self.make_cut(
                                node.data,
                                node.identifier,
                            )
                            if result is None:
                                raise RuntimeError(
                                    f"make_cut unexpectedly returned None for node {node.identifier}"
                                )
                            queue.append(result)
                    queue_count += 1
                # 2. wait for one finished element from futures
                future = first(cf.as_completed(futures), None)
                if future is not None:
                    futures.remove(future)
                    futures_processed += 1
                    result = future.result()
                    if result is None:
                        raise RuntimeError("make_cut future unexpectedly returned None")
                    queue.append(result)
                # 3. process any completed futures, without waiting
                try:
                    futures_to_remove = []
                    for future in cf.as_completed(futures, timeout=0):
                        futures_to_remove.append(future)
                        futures_processed += 1
                        result = future.result()
                        if result is not None:
                            queue.append(result)
                except cf.TimeoutError:
                    pass
                finally:
                    for future in futures_to_remove:
                        futures.remove(future)
                loop_count += 1
        finally:
            self.running = False
            if thread is not None:
                thread.join()
            parallelism__initializer_cleanup(*self.lookup_args)
            if executor is not None:
                executor.shutdown()
            parallelism__cleanup_logging(
                logging_listener,
                original_handlers,
                original_logging_level,
                logging_queue,
            )
        if self.debug_logging_enabled():
            if executor is not None:
                self.logger.debug("futures processed = %d", futures_processed)
            self.logger.debug("final mondrian tree nodes = %d", len(tree.nodes))
            self.logger.debug("final mondrian tree =\n%s", tree.pprint())
        if self.track_stats:
            root_node = tree.get_node(tree.root)
            assert root_node is not None, "Root node not found in tree"
            self.logger.info(
                "final mondrian tree cut stats =\n%s",
                root_node.data.get_stats(),
            )
        if track_qid_cuts:
            warnings.warn("track_qid_cuts = True WILL adversely impact performance.")
        return tree.candidate_solution(qids, True, track_qid_cuts, node_identifier_col)

    def make_node_cut(
        self,
        proposed_cut: ProposedCut,
        input_df: pd.DataFrame,
        max_suppression_n: int,
        further_cuts: list[pd.DataFrame],
        final_cuts: list[pd.DataFrame],
        n_suppressed: int,
    ) -> Node_Cut:
        """
        Create a Node_Cut object from a proposed cut.

        Parameters
        ----------
        proposed_cut : ProposedCut
            The proposed cut to convert to a Node_Cut.
        input_df : pd.DataFrame
            The input DataFrame being partitioned.
        max_suppression_n : int
            Maximum number of records that can be suppressed.
        further_cuts : List[pd.DataFrame]
            List of DataFrames for further partitioning.
        final_cuts : List[pd.DataFrame]
            List of DataFrames that are final cuts.
        n_suppressed : int
            Number of records suppressed by this cut.

        Returns
        -------
        Node_Cut
            A Node_Cut object representing this cut in the Mondrian tree.
        """
        return Node_Cut(
            proposed_cut.cut_choice,
            input_df,
            max_suppression_n,
            n_suppressed,
        )

    def make_node_deferred_cut__partition_root(
        self, input_df: pd.DataFrame, max_suppression_n: int
    ) -> Node_DeferredCutData:
        """
        Create a special deferred cut node for an initial partition root.

        This method creates a node that represents an initial partition in the Mondrian
        tree that will be processed later. It differs from regular deferred cut nodes
        in that it specifically tracks that it's a partition root.

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame for this partition.
        max_suppression_n : int
            Maximum number of records that can be suppressed in this partition.

        Returns
        -------
        Node_DeferredCutData
            A deferred cut node marked as a partition root.

        Notes
        -----
        If track_stats is enabled, this method also attaches a new FunnelStats object
        to the node to track statistics for this partition.
        """
        data = Node_DeferredCutData__PartitionRoot(input_df, max_suppression_n)
        if self.track_stats:
            data.attach_stats(FunnelStats())
        return data

    def make_node_deferred_cut(
        self, input_df: pd.DataFrame, max_suppression_n: int
    ) -> Node_DeferredCutData:
        """
        Create a standard deferred cut node.

        This method creates a node that represents a partition in the Mondrian tree
        that will be processed later (typically by the parallel processing machinery).

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame for this partition.
        max_suppression_n : int
            Maximum number of records that can be suppressed in this partition.

        Returns
        -------
        Node_DeferredCutData
            A deferred cut node representing this partition.
        """
        return Node_DeferredCutData(input_df, max_suppression_n)

    def make_node_final_cut(self, output_df: pd.DataFrame) -> Node_FinalCutData:
        """
        Make a final cut Data object.

        This method creates a final cut node containing a DataFrame that will not be
        further partitioned. It represents a leaf node in the Mondrian tree.

        Parameters
        ----------
        output_df : pd.DataFrame
            The DataFrame representing the final partition.

        Returns
        -------
        Node_FinalCutData
            A final cut node containing this partition.

        Notes
        -----
        Implementations that added columns by overriding setup* methods
        should remove them by overriding this method.
        """
        return Node_FinalCutData(output_df)
