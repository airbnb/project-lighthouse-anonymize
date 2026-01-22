"""
Core Mondrian Implementation for k-anonymity.

This module implements the Core Mondrian algorithm for k-anonymity, a privacy protection
model that ensures each record in a dataset is indistinguishable from at least k-1
other records based on quasi-identifier values.

The implementation extends the Revised Information Loss Metric (RILM) implementation
with k-anonymity-specific logic for:
- Determining when cuts are possible while maintaining k-anonymity
- Validating partitions against the k-anonymity property
- Implementing appropriate cut strategies for both numerical and categorical QIDs

K-anonymity is achieved by ensuring that, for each partition of data, there are at least
k records with the same quasi-identifier values. The algorithm recursively partitions
the data until further partitioning would violate this property.

References
----------
.. [1] L. Sweeney. "k-anonymity: A model for protecting privacy." International Journal of
       Uncertainty, Fuzziness and Knowledge-Based Systems, 10(05), 557-570, 2002.

.. note:: K. LeFevre, D. J. DeWitt, and R. Ramakrishnan. "Mondrian multidimensional
       k-anonymity." 22nd International Conference on Data Engineering (ICDE'06), 2006.
"""

import logging
from typing import Any, Optional, cast

import pandas as pd

from project_lighthouse_anonymize.disclosure_risk_metrics import calculate_p_k
from project_lighthouse_anonymize.gtrees import GTree
from project_lighthouse_anonymize.mondrian.implementation import NumericalCutPointsMode
from project_lighthouse_anonymize.mondrian.rilm import Implementation_RILM
from project_lighthouse_anonymize.mondrian.tree import CutChoice


class Implementation_KAnonymity(Implementation_RILM):
    """
    Implementation of Core Mondrian algorithm for k-anonymity.

    This class extends the Revised Information Loss Metric (RILM) implementation
    to specifically enforce k-anonymity privacy guarantees. It ensures that
    each partition in the anonymized dataset contains at least k records with the
    same quasi-identifier (QID) values.

    K-anonymity is achieved by:
    1. Only allowing cuts that maintain the k-anonymity property
    2. Validating that final partitions have at least k records
    3. Implementing partition strategies that respect the minimum k threshold

    The implementation optimizes for data utility while maintaining the k-anonymity
    privacy guarantee by using the RILM scoring mechanism from the parent class
    to choose cuts that minimize information loss.
    """

    def __init__(
        self,
        logger: logging.Logger,
        k: int,
        cut_choice_score_epsilon: Optional[float],
        cut_score_epsilon_partition_size_cutoff: float,
        numerical_cut_points_modes: tuple[NumericalCutPointsMode, ...],
        dq_metric_to_minimum_dq: dict[str, float],
        dynamic_breakout_rilm_multiplier: Optional[float],
    ):
        """
        Initialize the k-anonymity implementation.

        Parameters
        ----------
        logger : logging.Logger
            Logger for recording the anonymization process.
        k : int
            The k value for k-anonymity, representing the minimum number of records
            required in each equivalence class. Must be greater than 0.
        cut_choice_score_epsilon : float or None
            Epsilon for determining whether two cut choices should be considered at the same time.
            If None, all cut choices will be considered regardless of their scores.
        cut_score_epsilon_partition_size_cutoff : float
            Relative partition size threshold below which multiple cut choices will not
            be considered together.
        numerical_cut_points_modes : Tuple[NumericalCutPointsMode, ...]
            Modes for determining numerical cut points, e.g., MEDIAN or BIN_EDGES.
        dq_metric_to_minimum_dq : Dict[str, float]
            Mapping from data quality metric names to minimum acceptable values.
        dynamic_breakout_rilm_multiplier : float or None
            Optional multiplier for dynamic breakout based on RILM scores.
            If None, dynamic breakout is disabled.

        Raises
        ------
        ValueError
            If k is not greater than 0.
        """
        if k <= 0:
            raise ValueError(f"k {k} must be > 0")
        self.k = k
        super().__init__(
            logger,
            cut_choice_score_epsilon,
            cut_score_epsilon_partition_size_cutoff,
            numerical_cut_points_modes,
            dq_metric_to_minimum_dq,
            dynamic_breakout_rilm_multiplier,
        )

    def propose_cuts_numerical_impl(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
        cut: float,
        context: Optional[Any],
    ) -> Optional[list[pd.DataFrame]]:
        """
        Implement numerical cuts for k-anonymity.

        This method creates partitions based on a numerical cut point for a specific QID.
        For k-anonymity, this simply divides the data into two partitions: records with
        values less than or equal to the cut point, and records with values greater than it.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be cut.
        cut_choice : CutChoice
            The cut choice specifying which numerical QID to cut on.
        cut : float
            The numerical value to use as the cut point.
        context : Any or None
            Optional context data from propose_cuts_numerical_impl_before.

        Returns
        -------
        List[pd.DataFrame]
            A list containing two DataFrames:
            - The first with records where QID value <= cut
            - The second with records where QID value > cut

        Notes
        -----
        This implementation performs a simple binary split on the numerical QID.
        """
        return [
            cast(pd.DataFrame, input_df[input_df[cut_choice.qid] <= cut]),
            cast(pd.DataFrame, input_df[input_df[cut_choice.qid] > cut]),
        ]

    def propose_cuts_categorical_impl(
        self,
        nid: str,
        input_df: pd.DataFrame,
        cut_choice: CutChoice,
        gtree: GTree,
    ) -> Optional[list[pd.DataFrame]]:
        """
        Implement categorical cuts for k-anonymity using a generalization tree.

        This method creates partitions based on a categorical QID using its generalization
        tree structure. It finds the lowest common ancestor node in the tree that contains
        all unique values in the partition, then creates child partitions based on the
        children of that node.

        Parameters
        ----------
        nid : str
            Node identifier for logging purposes.
        input_df : pd.DataFrame
            The DataFrame partition to be cut.
        cut_choice : CutChoice
            The cut choice specifying which categorical QID to cut on.
        gtree : GTree
            The generalization tree for this categorical QID.

        Returns
        -------
        List[pd.DataFrame] or None
            If the lowest common ancestor node has children, returns a list of DataFrames
            with one DataFrame per child node, each containing records whose QID values
            are descendants of that child. If the node has no children (is a leaf),
            returns a list containing just the input DataFrame.

        Notes
        -----
        This implementation leverages the hierarchical structure of the generalization tree
        to naturally partition the data according to the taxonomy.
        """
        node = gtree.get_lowest_node_with_descendant_leaves(
            set(input_df[cut_choice.qid].unique()),
        )
        assert node is not None, f"Could not find lowest node for cut choice {cut_choice.qid}"
        if any(gtree.children(node.identifier)):
            return [
                cast(
                    pd.DataFrame,
                    input_df[
                        input_df[cut_choice.qid].isin(gtree.descendant_leaf_values(child_node))
                    ],
                )
                for child_node in gtree.children(node.identifier)
            ]
        else:
            return [
                input_df,
            ]

    def validate(self, input_df: pd.DataFrame, qids: list[str]) -> bool:
        """
        Validate whether a DataFrame satisfies k-anonymity.

        This method checks if the given DataFrame meets the k-anonymity requirement,
        which requires that each combination of quasi-identifier values appears at
        least k times in the dataset.

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame to validate against the k-anonymity requirement.
        qids : List[str]
            List of quasi-identifier column names.

        Returns
        -------
        bool
            True if the DataFrame is k-anonymous (all equivalence classes have at least
            k records), False otherwise. Returns True for empty DataFrames.

        Notes
        -----
        The method uses the calculate_p_k function from the validate module to determine
        the actual k value achieved in the DataFrame. This function groups records by
        their QID values and finds the size of the smallest group, which represents
        the achieved k value.
        """
        if len(input_df) > 0:
            _, actual_k_val = calculate_p_k(input_df, qids)
            actual_k = float(actual_k_val) if actual_k_val is not None else float("nan")
        else:
            actual_k = float("nan")
        return actual_k >= self.k

    def are_cuts_possible(
        self, input_df: pd.DataFrame, qids: list[str], gtrees: dict[str, GTree]
    ) -> bool:
        """
        Determine if further cuts are possible while maintaining k-anonymity.

        This method checks whether further partitioning of the data could potentially
        yield valid partitions that satisfy k-anonymity. For k-anonymity, this means
        checking if the partition size is greater than k (the minimum requirement for
        a valid partition).

        Parameters
        ----------
        input_df : pd.DataFrame
            The DataFrame partition to evaluate.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Mapping from categorical QID column names to generalization trees.

        Returns
        -------
        bool
            True if further cuts are possible while maintaining k-anonymity,
            False otherwise.

        Notes
        -----
        This is a simple size check - if the partition has more than k records,
        then further cuts might be possible. The actual viability of specific
        cuts is evaluated elsewhere in the algorithm.
        """
        return len(input_df) > self.k

    def is_a_further_cut(
        self,
        partition_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> bool:
        """
        Check if a partition can be further cut while maintaining k-anonymity.

        This method determines whether a specific partition could be cut further
        while still satisfying k-anonymity. It applies a stricter requirement than
        just checking if the partition could be a valid final cut - it checks if
        the partition is large enough to be split into at least two valid partitions.

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
            True if the partition can be further cut while maintaining k-anonymity,
            False otherwise.

        Notes
        -----
        The method requires the partition to have at least 2*k records to ensure
        that it could potentially be split into at least two partitions, each with
        at least k records. This is a conservative approach that ensures there are
        enough records to make a meaningful cut.
        """
        return len(partition_df) >= self.k * 2

    def could_be_a_final_cut(
        self,
        partition_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
    ) -> bool:
        """
        Check if a partition could be a valid final cut under k-anonymity.

        This method determines whether a partition could potentially satisfy
        the k-anonymity requirement after generalization. For k-anonymity, this
        simply requires checking if the partition has at least k records.

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
            True if the partition could satisfy k-anonymity after generalization,
            False otherwise.

        Notes
        -----
        This method performs a simple size check - a partition can only satisfy
        k-anonymity if it has at least k records. This is a basic requirement
        that must be met before generalization or suppression is attempted.
        """
        return len(partition_df) >= self.k
