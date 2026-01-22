"""
Statistics tracking for Mondrian funnel analysis.

This module provides tools for collecting and analyzing performance metrics during
the Mondrian algorithm's execution. It tracks the progression of cut choices through
various stages of the algorithm (examination, consideration, proposal, finalization,
and acceptance), breaking down statistics by quasi-identifier and data type
(categorical vs. numerical).

These statistics help with analyzing algorithm efficiency, identifying bottlenecks,
and understanding how different attributes contribute to the anonymization process.
"""

import copy
from typing import Optional


class FunnelStats:
    """
    Tracks statistics about the Mondrian cut processing funnel.

    This class collects and aggregates statistics about the Mondrian algorithm's
    execution, tracking how many cut choices progress through each stage of
    the processing pipeline:

    1. Examination: All potential cut choices considered
    2. Consideration: Cut choices selected for further evaluation
    3. Proposal: Cut choices that generate concrete proposed partitions
    4. Finalization: Proposed cuts that result in final partitions
    5. Acceptance: Final cuts that are actually accepted into the solution

    Statistics are tracked across three dimensions:
    - Overall totals across all QIDs
    - Per-QID breakdowns showing how each attribute performs
    - Categorical vs numerical QID comparisons

    The class provides methods to increment counters, merge statistics from
    different execution branches, and generate detailed reports.
    """

    def __init__(self) -> None:
        """
        Initialize a new FunnelStats instance with zero counts for all metrics.

        Creates counter structures for tracking overall statistics,
        per-QID breakdowns, and categorical/numerical comparisons.
        """
        # Overall counts
        self.cut_choices_examined = 0
        self.proposed_cuts_under_consideration = 0
        self.proposed_cuts_considered = 0
        self.final_cuts_generated = 0
        self.cuts_accepted = 0

        # Per-QID breakdowns
        self.qid_stats: dict[str, dict[str, int]] = {}

        # Categorical vs numerical breakdowns
        self.categorical_stats = {
            "examined": 0,
            "under_consideration": 0,
            "proposed": 0,
            "final": 0,
            "accepted": 0,
        }
        self.numerical_stats = {
            "examined": 0,
            "under_consideration": 0,
            "proposed": 0,
            "final": 0,
            "accepted": 0,
        }

    def _ensure_qid_stats(self, qid: str) -> None:
        """
        Ensure a QID entry exists in the qid_stats dictionary.

        This internal helper method creates an entry for tracking statistics
        for a specific quasi-identifier if one doesn't already exist.

        Args:
            qid: The quasi-identifier name to ensure exists in tracking
        """
        if qid not in self.qid_stats:
            self.qid_stats[qid] = {
                "examined": 0,
                "under_consideration": 0,
                "proposed": 0,
                "final": 0,
                "accepted": 0,
            }

    def increment_examined(self, qid: str, is_categorical: bool = False) -> None:
        """
        Increment counter for examined cut choices.

        Called when the algorithm examines a potential attribute to cut on.
        This is the first stage in the funnel process.

        Args:
            qid: The quasi-identifier name being examined
            is_categorical: Whether this QID is categorical (True) or numerical (False)
        """
        self.cut_choices_examined += 1
        self._ensure_qid_stats(qid)
        self.qid_stats[qid]["examined"] += 1

        if is_categorical:
            self.categorical_stats["examined"] += 1
        else:
            self.numerical_stats["examined"] += 1

    def increment_under_consideration(self, qid: str, is_categorical: bool = False) -> None:
        """
        Increment counter for proposed cuts under consideration.

        Called when a cut choice passes initial filtering and is selected
        for further evaluation. This is the second stage in the funnel process.

        Args:
            qid: The quasi-identifier name under consideration
            is_categorical: Whether this QID is categorical (True) or numerical (False)
        """
        self.proposed_cuts_under_consideration += 1
        self._ensure_qid_stats(qid)
        self.qid_stats[qid]["under_consideration"] += 1

        if is_categorical:
            self.categorical_stats["under_consideration"] += 1
        else:
            self.numerical_stats["under_consideration"] += 1

    def increment_proposed(self, qid: str, is_categorical: bool = False) -> None:
        """
        Increment counter for proposed cuts.

        Called when a cut under consideration is fully evaluated and
        proposed as a candidate partition. This is the third stage in the funnel.

        Args:
            qid: The quasi-identifier name being proposed for partitioning
            is_categorical: Whether this QID is categorical (True) or numerical (False)
        """
        self.proposed_cuts_considered += 1
        self._ensure_qid_stats(qid)
        self.qid_stats[qid]["proposed"] += 1

        if is_categorical:
            self.categorical_stats["proposed"] += 1
        else:
            self.numerical_stats["proposed"] += 1

    def increment_final(self, qid: str, is_categorical: bool = False) -> None:
        """
        Increment counter for final cuts generated.

        Called when a proposed cut has been processed and results in
        final partitions. This is the fourth stage in the funnel process.

        Args:
            qid: The quasi-identifier name used for the final cut
            is_categorical: Whether this QID is categorical (True) or numerical (False)
        """
        self.final_cuts_generated += 1
        self._ensure_qid_stats(qid)
        self.qid_stats[qid]["final"] += 1

        if is_categorical:
            self.categorical_stats["final"] += 1
        else:
            self.numerical_stats["final"] += 1

    def increment_accepted(self, qid: str, is_categorical: bool = False) -> None:
        """
        Increment counter for accepted cuts.

        Called when a final cut is accepted as part of the solution.
        This is the final stage in the funnel process.

        Args:
            qid: The quasi-identifier name used in the accepted cut
            is_categorical: Whether this QID is categorical (True) or numerical (False)
        """
        self.cuts_accepted += 1
        self._ensure_qid_stats(qid)
        self.qid_stats[qid]["accepted"] += 1

        if is_categorical:
            self.categorical_stats["accepted"] += 1
        else:
            self.numerical_stats["accepted"] += 1

    def merge(self, other: Optional["FunnelStats"]) -> None:
        """
        Merge another FunnelStats object into this one.

        This method combines the statistics from another FunnelStats instance
        with this one, which is useful when aggregating results from parallel
        processing or from subtrees in the Mondrian algorithm.

        Args:
            other: Another FunnelStats object to merge into this one.
                  If None, this method does nothing.
        """
        if other is None:
            return

        # Merge overall counts
        self.cut_choices_examined += other.cut_choices_examined
        self.proposed_cuts_under_consideration += other.proposed_cuts_under_consideration
        self.proposed_cuts_considered += other.proposed_cuts_considered
        self.final_cuts_generated += other.final_cuts_generated
        self.cuts_accepted += other.cuts_accepted

        # Merge per-QID stats
        for qid, stats in other.qid_stats.items():
            self._ensure_qid_stats(qid)
            for key, value in stats.items():
                self.qid_stats[qid][key] += value

        # Merge categorical/numerical stats
        for key, value in other.categorical_stats.items():
            self.categorical_stats[key] += value

        for key, value in other.numerical_stats.items():
            self.numerical_stats[key] += value

    def copy(self) -> "FunnelStats":
        """
        Create a deep copy of this FunnelStats object.

        Returns:
            A new FunnelStats instance with the same statistics as this one,
            but with independent counter objects that can be modified separately.
        """
        return copy.deepcopy(self)

    def _calculate_rates(
        self, stats_dict: dict[str, int]
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate transition rates between successive funnel stages.

        This internal helper method computes efficiency rates between the
        different stages of the processing funnel, helping identify where
        the algorithm is most selective or where bottlenecks may be occurring.

        Args:
            stats_dict: Dictionary with counts for each funnel stage

        Returns:
            A tuple containing five rate metrics (all in percentages):
            1. consideration_rate: % of examined cuts that enter consideration
            2. proposed_rate: % of considered cuts that become proposed cuts
            3. final_rate: % of proposed cuts that become final cuts
            4. acceptance_rate: % of final cuts that are accepted
            5. overall_yield: % of examined cuts that make it to acceptance
        """
        proposed_rate = 0.0
        final_rate = 0.0
        acceptance_rate = 0.0
        overall_yield = 0.0
        consideration_rate = 0.0

        if stats_dict["examined"] > 0:
            consideration_rate = stats_dict["under_consideration"] / stats_dict["examined"] * 100
            overall_yield = stats_dict["accepted"] / stats_dict["examined"] * 100

            if stats_dict["under_consideration"] > 0:
                proposed_rate = stats_dict["proposed"] / stats_dict["under_consideration"] * 100

                if stats_dict["proposed"] > 0:
                    final_rate = stats_dict["final"] / stats_dict["proposed"] * 100

                    if stats_dict["final"] > 0:
                        acceptance_rate = stats_dict["accepted"] / stats_dict["final"] * 100

        return (
            consideration_rate,
            proposed_rate,
            final_rate,
            acceptance_rate,
            overall_yield,
        )

    def _format_stats_section(self, title: str, stats_dict: dict[str, int]) -> list[str]:
        """
        Format a section of statistics with calculated rates for display.

        This internal helper method creates a formatted report section for a
        particular set of statistics (overall, by QID type, or by specific QID),
        including both raw counts and calculated transition rates.

        Args:
            title: The title for this statistics section
            stats_dict: Dictionary with counts for each funnel stage

        Returns:
            List of formatted strings representing the statistics section
        """
        result = [f"=== {title} ==="]
        result.append(f"Cut Choices:              {stats_dict['examined']}")
        result.append(f"→ Under Consideration:    {stats_dict['under_consideration']}")
        result.append(f"→ → Proposed Cuts:        {stats_dict['proposed']}")
        result.append(f"→ → → Final Cuts:         {stats_dict['final']}")
        result.append(f"→ → → → Accepted:         {stats_dict['accepted']}")

        (
            consideration_rate,
            proposed_rate,
            final_rate,
            acceptance_rate,
            overall_yield,
        ) = self._calculate_rates(stats_dict)

        result.append(f"Consideration Rate:  {consideration_rate:.1f}%")
        result.append(f"Proposed Rate:       {proposed_rate:.1f}%")
        result.append(f"Finalized Rate:      {final_rate:.1f}%")
        result.append(f"Acceptance Rate:     {acceptance_rate:.1f}%")
        result.append(f"Overall Yield:       {overall_yield:.1f}%")

        return result

    def summary(self) -> str:
        """
        Return a brief summary of statistics suitable for concise logging.

        This method provides a compact representation of the key statistics
        for use in log messages or when a full report would be too verbose.

        Returns:
            A concise string representation with the main funnel stage counts
        """
        return f"Stats({self.cut_choices_examined}, {self.proposed_cuts_under_consideration}, {self.proposed_cuts_considered}, {self.final_cuts_generated}, {self.cuts_accepted})"

    def __repr__(self) -> str:
        """
        Generate a comprehensive report of all funnel statistics.

        This method creates a detailed, multi-section report showing:
        1. Overall funnel statistics across all QIDs
        2. Categorical QID statistics
        3. Numerical QID statistics
        4. Per-QID breakdown of all statistics

        Each section includes both raw counts and calculated transition rates,
        formatted in a hierarchical structure that visually represents the
        funnel stages.

        Returns:
            A multi-line string containing the full statistical report
        """
        result = []

        # Overall stats
        overall_stats = {
            "examined": self.cut_choices_examined,
            "under_consideration": self.proposed_cuts_under_consideration,
            "proposed": self.proposed_cuts_considered,
            "final": self.final_cuts_generated,
            "accepted": self.cuts_accepted,
        }
        result.extend(self._format_stats_section("Mondrian Cut Funnel Analysis", overall_stats))

        # Categorical stats
        result.append("")
        result.extend(self._format_stats_section("Categorical QIDs", self.categorical_stats))

        # Numerical stats
        result.append("")
        result.extend(self._format_stats_section("Numerical QIDs", self.numerical_stats))

        # Per QID breakdown
        result.append("\n=== Per QID Breakdown ===")
        for qid, stats in sorted(self.qid_stats.items()):
            result.append(f"\nQID: {qid}")
            result.append(f"Cut Choices:              {stats['examined']}")
            result.append(f"→ Under Consideration:    {stats['under_consideration']}")
            result.append(f"→ → Proposed Cuts:        {stats['proposed']}")
            result.append(f"→ → → Final Cuts:         {stats['final']}")
            result.append(f"→ → → → Accepted:         {stats['accepted']}")

            (
                consideration_rate,
                proposed_rate,
                final_rate,
                acceptance_rate,
                overall_yield,
            ) = self._calculate_rates(stats)

            result.append(f"Consideration Rate:  {consideration_rate:.1f}%")
            result.append(f"Proposed Rate:       {proposed_rate:.1f}%")
            result.append(f"Finalized Rate:      {final_rate:.1f}%")
            result.append(f"Acceptance Rate:     {acceptance_rate:.1f}%")
            result.append(f"Overall Yield:       {overall_yield:.1f}%")

        return "\n".join(result)
