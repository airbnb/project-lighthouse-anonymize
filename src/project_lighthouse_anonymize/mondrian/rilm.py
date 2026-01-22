"""
Shared logic for Mondrian Implementations that utilize RILM for scoring cut choices and dynamic breakout.

This module encapsulates a balanced framework for scoring partitioning decisions across
both categorical and numerical attributes, based on RILM. It does so by:

1. Creating appropriate domains for numerical attributes using percentile-based boundaries
2. Using specialized approaches for categorical and numerical attributes for RILM calculation
3. Supporting dynamic breakout strategies that optimize the partitioning process

The RILM implementation serves as a base for specific privacy models like k-anonymity
and u-anonymity, providing shared scoring logic for data utility based decisions.
"""

import logging
from pprint import pformat
from typing import Generator, Optional, cast

import numpy as np
import pandas as pd
from first import first  # type: ignore[import-untyped]

from project_lighthouse_anonymize.constants import EPSILON
from project_lighthouse_anonymize.data_quality_metrics.rilm_ilm import (
    compute_revised_information_loss_metric_value_for_categorical,
    compute_revised_information_loss_metric_value_for_numerical,
)
from project_lighthouse_anonymize.gtrees import GTree
from project_lighthouse_anonymize.mondrian.implementation import (
    Implementation_Base,
    NumericalCutPointsMode,
)
from project_lighthouse_anonymize.mondrian.tree import CutChoice
from project_lighthouse_anonymize.utils import min_max


class NumericalRILM_Domain:
    """
    Represents a domain for numerical RILM (Revised Information Loss Metric) calculation.

    This class creates domains using percentiles from a distribution of numerical values.
    By using percentile-based domains, we can effectively handle fat-tailed distributions
    in numerical data (e.g., Pareto).

    The domains provide appropriate boundaries for evaluating data quality
    of numerical attributes. This approach helps normalize the assessment of
    information loss, ensuring the anonymization process makes appropriate
    tradeoffs.

    Attributes:
        start_percentile: Lower percentile boundary (epsilon away from 0)
        end_percentile: Upper percentile boundary (epsilon away from 100)
        start_value: Actual value at the lower percentile boundary
        end_value: Actual value at the upper percentile boundary
        domain_size: Range of the domain (end_value - start_value)
    """

    def __init__(
        self,
        numerical_values: list[float],
        epsilon: float,
    ):
        """
        Initialize a numerical RILM domain based on the distribution of numerical values.

        Args:
            numerical_values: List of numeric values to analyze for domain creation.
            epsilon: Small value to prevent including extreme outliers by setting
                    boundaries epsilon away from the 0th and 100th percentiles.

        Notes:
            The domain is established by calculating percentile-based boundaries
            that exclude extreme values. This helps create a robust domain that
            isn't overly influenced by outliers while still representing the
            majority of the data distribution.
        """
        start_percentile = 0.0 + epsilon
        end_percentile = 100.0 - epsilon
        start_value, end_value = np.percentile(numerical_values, [start_percentile, end_percentile])
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.start_value = start_value
        self.end_value = end_value
        self.domain_size = self.end_value - self.start_value

        # Precompute ULP-adjusted boundaries for fast boundary checking
        self._start_inclusive = np.nextafter(start_value, -np.inf)
        self._end_inclusive = np.nextafter(end_value, +np.inf)

    def contains_range(self, minimum: float, maximum: float) -> bool:
        """Fast boundary check using precomputed ULP bounds."""
        return bool(minimum >= self._start_inclusive and maximum <= self._end_inclusive)

    def __repr__(self) -> str:
        return f"< [{self.start_percentile}%, {self.end_percentile}%] => [{self.start_value}, {self.end_value}] >"


DEFAULT_NUMERICAL_RILM_DOMAIN_EPSILONS: tuple[float, ...] = (
    10.0,
    1.0,
    0.10,
    0.01,
    0.001,
    0.0,
)
"""
Default epsilon values used for creating NumericalRILM_Domain instances.

These values define the percentiles (0+epsilon and 100-epsilon) to establish
numerical domain boundaries. A list of domains is created using these epsilons,
allowing the selection of a domain that best fits a sub-partition's data range
while mitigating the influence of extreme outliers. The value 0.0 corresponds
to using the true 0th and 100th percentiles (i.e., min/max of the initial data).
"""


class NumericalRILM_CutChoice(CutChoice):
    """
    Extended cut choice class for numerical RILM-based Mondrian implementations.

    This class extends the base CutChoice by adding RILM-specific attributes
    related to the size of numerical cuts relative to their domains. These additional
    attributes help in evaluating and optimizing the data quality impact
    of different cut choices within the RILM framework.

    Attributes:
        cut_size: The size of this cut in the attribute's domain
        numerical_rilm_domain: Reference to the domain for this attribute
        Plus inherited attributes from CutChoice (qid, score, is_categorical)
    """

    def __init__(
        self,
        qid: str,
        score: float,
        cut_size: float,
        numerical_rilm_domain: NumericalRILM_Domain,
    ):
        """
        Initialize a numerical RILM-specific cut choice.

        Args:
            qid: The quasi-identifier column name
            score: Quality score for this cut (lower is better)
            cut_size: Size of the cut within its domain
            numerical_rilm_domain: Reference to the domain for this attribute
        """
        self.cut_size = cut_size
        self.numerical_rilm_domain = numerical_rilm_domain
        super().__init__(qid, score, False)

    def __repr__(self) -> str:
        """
        Create a string representation including RILM-specific details.

        Returns:
            String showing cut details with domain information
        """
        return f"{super().__repr__()} ({self.cut_size} / {self.numerical_rilm_domain})"


class Implementation_RILM(Implementation_Base):
    """
    Mondrian implementation that uses RILM (Revised Information Loss Metric) for scoring.

    This implementation creates appropriate domains for numerical
    variables to balance information loss. The key features:

    1. For numerical variables: Creates domains based on percentile ranges to handle
       fat-tailed distributions effectively

    2. Specialized approaches: Uses dedicated methods for calculating RILM scores
       for numerical and categorical variables

    3. Dynamic breakout: Optionally supports dynamic breakout strategies that allow
       early termination of the partitioning process based on RILM score calculations

    4. Data quality metrics: Supports configurable minimum thresholds for various
       data quality metrics to guide the anonymization process

    Note: We experimented with extending the domain-based approach to categoricals
    (using a unified RILM_Domain for both numerical and categorical variables). While
    this produced more balanced quality scores between variable types, it led to
    significant differences in anonymization results when comparing mixed datasets
    (e.g., 3 complex categoricals + 2 heavy-tailed numericals) versus separate
    anonymization calls for each type. The current implementation strikes a better
    balance for practical applications.

    This approach helps achieve good information loss metrics for both numerical
    and categorical variables, improving overall anonymization quality.
    It serves as a foundation for specific privacy models like k-anonymity and
    u-anonymity with enhanced utility preservation.

    Attributes:
        dq_metric_to_minimum_dq (Dict[str, float]): Mapping of data quality metric
            names to their minimum acceptable values, used for dynamic breakout.
        dynamic_breakout_rilm_multiplier (Optional[float]): Multiplier used to
            calculate dynamic breakout RILM score thresholds. If None, dynamic
            breakout is disabled.
        numerical_rilm_domains (Dict[str, List[NumericalRILM_Domain]]): Stores
            pre-calculated RILM domains for each numerical QID, generated during setup.
        numerical_rilm_domain_epsilons (Tuple[float, ...]): Epsilon values used
            to create varied NumericalRILM_Domain instances for numerical QIDs.
        qid_to_breakout_score (Optional[Dict[str, float]]): Pre-calculated RILM
            score thresholds per QID for dynamic breakout. If a cut's score
            exceeds its QID's threshold, it may be skipped.
    """

    def __init__(
        self,
        logger: logging.Logger,
        cut_choice_score_epsilon: Optional[float],
        cut_score_epsilon_partition_size_cutoff: float,
        numerical_cut_points_modes: tuple[NumericalCutPointsMode, ...],
        dq_metric_to_minimum_dq: dict[str, float],
        dynamic_breakout_rilm_multiplier: Optional[float],
        numerical_rilm_domain_epsilons: tuple[float, ...] = DEFAULT_NUMERICAL_RILM_DOMAIN_EPSILONS,
    ):
        """
        Initialize the RILM implementation for Core Mondrian algorithm.

        Args:
            logger: Logger for recording the anonymization process
            cut_choice_score_epsilon: Epsilon for determining whether two cut
                choices should be considered at the same time
            cut_score_epsilon_partition_size_cutoff: Relative partition size threshold
                below which multiple cut choices will not be considered together
            numerical_cut_points_modes: Modes for determining numerical cut points
            dq_metric_to_minimum_dq: Dictionary mapping data quality metric names
                to minimum acceptable values
            dynamic_breakout_rilm_multiplier: Optional multiplier for the dynamic
                breakout threshold based on RILM scores. If provided, enables early
                termination of the partitioning process when cuts would reduce
                data quality below acceptable thresholds.
            numerical_rilm_domain_epsilons: Tuple of epsilon values to try when creating
                RILM domains to avoid extreme outliers
        """
        self.dq_metric_to_minimum_dq = dq_metric_to_minimum_dq
        self.dynamic_breakout_rilm_multiplier = dynamic_breakout_rilm_multiplier
        self.numerical_rilm_domains: dict[str, list[NumericalRILM_Domain]] = {}
        self.numerical_rilm_domain_epsilons = numerical_rilm_domain_epsilons
        self.qid_to_breakout_score: Optional[dict[str, float]] = None
        super().__init__(
            logger,
            cut_choice_score_epsilon,
            cut_score_epsilon_partition_size_cutoff,
            numerical_cut_points_modes,
        )

    def setup(
        self,
        input_df: pd.DataFrame,
        qids: list[str],
        gtrees: dict[str, GTree],
        exclude_qids: list[str],
    ) -> None:
        """
        Initializes RILM-specific settings and prepares for the anonymization process.

        This method clears any existing numerical RILM domains. If dynamic
        breakout is enabled (i.e., `self.dynamic_breakout_rilm_multiplier` is set),
        it calculates the `qid_to_breakout_score` thresholds. These thresholds are
        derived from `self.dq_metric_to_minimum_dq` and the multiplier. QIDs
        listed in `exclude_qids` use specific `rilm__<qid>` entries from
        `dq_metric_to_minimum_dq` if available; otherwise, general categorical or
        numerical minimums are used.

        It then calls the parent class's `setup` method to complete
        the general setup, which in turn calls `self.setup_numerical_qid`
        for each numerical QID to populate `self.numerical_rilm_domains`.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input DataFrame to be anonymized.
        qids : List[str]
            List of quasi-identifier column names.
        gtrees : Dict[str, GTree]
            Dictionary mapping categorical QID names to their GTree hierarchies.
        exclude_qids : List[str]
            List of QIDs that should use specific RILM threshold
            configurations (e.g., `rilm__<qid>`) from
            `dq_metric_to_minimum_dq` for dynamic breakout. Other
            QIDs will use general categorical/numerical minimums.
        """
        self.numerical_rilm_domains.clear()
        if self.dynamic_breakout_rilm_multiplier is not None:
            qid_to_breakout_score = {
                qid: (
                    self.dq_metric_to_minimum_dq.get(f"rilm__{qid}", 1.0)
                    if (qid in exclude_qids)
                    else (
                        self.dq_metric_to_minimum_dq.get("rilm_categorical__minimum", 1.0)
                        if qid in gtrees
                        else self.dq_metric_to_minimum_dq.get(
                            "rilm_numerical__minimum",
                            self.dq_metric_to_minimum_dq.get("pearsons__minimum", 1.0),
                        )
                    )
                )
                for qid in qids
            }
            self.qid_to_breakout_score = {
                qid: (
                    breakout_score
                    + ((1.0 - breakout_score) * self.dynamic_breakout_rilm_multiplier)
                )
                for qid, breakout_score in qid_to_breakout_score.items()
            }
            self.logger.debug(
                "[dynamic breakout] qid_to_breakout_score = \n%s",
                pformat(self.qid_to_breakout_score),
            )
        else:
            self.qid_to_breakout_score = None
            self.logger.debug("[dynamic breakout] disabled")
        super().setup(input_df, qids, gtrees, exclude_qids)
        self.logger.info(
            "numerical_rilm_domains = %s",
            pformat(self.numerical_rilm_domains),
        )

    def setup_numerical_qid(self, input_df: pd.DataFrame, qid: str) -> None:
        """
        Sets up RILM domains for a given numerical quasi-identifier.

        For the specified numerical QID, this method extracts its non-null values
        from the `input_df`. It then computes a list of `NumericalRILM_Domain`
        objects, where each domain is created with a different epsilon value from
        `self.numerical_rilm_domain_epsilons`. This allows for different
        percentile-based boundaries to be considered for RILM calculations.
        These domains are stored in `self.numerical_rilm_domains[qid]`, sorted by
        epsilon in descending order.

        Finally, it calls the parent class's `setup_numerical_qid` method.

        Args
        ----
        input_df: The input DataFrame containing the data for all QIDs.
        qid: The numerical quasi-identifier column name for which to set up domains.
        """
        numerical_values = cast(pd.Series, input_df[~pd.isna(input_df[qid])][qid]).to_numpy()

        # Handle empty datasets gracefully - when there are no numerical values,
        # we cannot create meaningful RILM domains, so we use an empty list
        if len(numerical_values) == 0:
            self.numerical_rilm_domains[qid] = []
        else:
            self.numerical_rilm_domains[qid] = [
                NumericalRILM_Domain(cast(list[float], numerical_values), epsilon)  # type: ignore[reportArgumentType]  # numpy accepts ndarray but stubs expect list
                for epsilon in sorted(self.numerical_rilm_domain_epsilons, reverse=True)
            ]
        super().setup_numerical_qid(input_df, qid)

    def cut_choices__categorical_impl(
        self,
        categorical_values: pd.Series,
        gtree: GTree,
    ) -> str:
        """
        Determines the generalized categorical value for a set of input values.

        Given a pandas Series of categorical values (typically representing the
        unique values for a QID within a specific data partition) and
        its corresponding generalization hierarchy (GTree), this method finds
        the lowest (i.e., most specific) node in the GTree that serves as an
        ancestor to all unique values present in the input series.

        Args
        ----
        categorical_values: A pandas Series containing the categorical values
                            from the current partition for a specific QID.
        gtree: The GTree (generalization hierarchy) for this categorical attribute.

        Returns
        -------
        str
            The string representation of the generalized value from the GTree that
            covers all unique input `categorical_values`.
        """
        lowest_node = gtree.get_lowest_node_with_descendant_leaves(set(categorical_values.unique()))
        assert lowest_node is not None, "Could not find lowest node for categorical values"
        return str(gtree.get_value(lowest_node))

    def cut_choices(
        self,
        nid: str,
        input_df: pd.DataFrame,
        gtrees: dict[str, GTree],
        qids_to_consider: list[str],
    ) -> Generator[CutChoice, None, None]:
        """
        Generates and scores potential cut choices for the current partition using RILM.

        For each quasi-identifier in `qids_to_consider`, this method calculates
        the Revised Information Loss Metric (RILM) score for the current data
        in `input_df` (representing a partition).

        - For numerical QIDs: It determines the current range (min, max) of the data.
          It then selects the smallest pre-calculated `NumericalRILM_Domain`
          (from `self.numerical_rilm_domains`) that encompasses this range.
          The RILM score is computed based on this domain and the current range.
          A `NumericalRILM_CutChoice` object is created.

        - For categorical QIDs: It determines the current generalized value using
          `self.cut_choices__categorical_impl` and computes RILM using the GTree.
          A `CutChoice` object is created.

        All generated cut choices are then sorted by their RILM score (lower is better).

        If dynamic breakout is enabled (i.e., `self.qid_to_breakout_score` is not None),
        the sorted cut choices are further filtered. Each `CutChoice`'s score is
        compared against a breakout threshold.

        Note: In the current implementation, the breakout score used for comparison is
        based on the last QID processed in the outer loop, not on each cut choice's QID.
        This may lead to unexpected behavior when yielding cut choices.

        Args:
            nid: Node identifier, primarily used for logging purposes.
            input_df: The pandas DataFrame representing the data within the current partition.
            gtrees: A dictionary mapping categorical QID names to their GTree objects.
            qids_to_consider: A list of QID names to evaluate for potential cuts.

        Yields:
            CutChoice (or NumericalRILM_CutChoice): Potential cut choices, ordered by
            their RILM score, and filtered by dynamic breakout if enabled.
        """
        cut_choices: list[CutChoice] = []
        for qid in qids_to_consider:
            gtree_if = gtrees.get(qid, None)
            if gtree_if is None:
                # numerical
                minimum, maximum = min_max(input_df[qid].to_numpy())
                # find the smallest NumericalRILM_Domain that encompasses this sub-problem
                # and use that to define the domain size; we do this to better manage numericals
                # with fat tails
                numerical_rilm_domain = first(
                    self.numerical_rilm_domains[qid],
                    key=lambda numerical_rilm_domain: numerical_rilm_domain.contains_range(
                        minimum, maximum
                    ),
                )
                assert numerical_rilm_domain is not None
                cut_size = maximum - minimum
                rilm = compute_revised_information_loss_metric_value_for_numerical(
                    numerical_rilm_domain.domain_size, cut_size
                )
                cut_choice: CutChoice = NumericalRILM_CutChoice(
                    qid, rilm, cut_size, numerical_rilm_domain
                )
            else:
                # categorical
                rilm = compute_revised_information_loss_metric_value_for_categorical(
                    qid,
                    gtree_if,
                    self.cut_choices__categorical_impl(
                        cast(pd.Series, input_df[~pd.isna(input_df[qid])][qid]), gtree_if
                    ),
                )
                cut_choice = CutChoice(qid, rilm, True)
            cut_choices.append(cut_choice)
        cut_choices = sorted(cut_choices, key=lambda cut_choice: cut_choice.score)
        if self.debug_logging_enabled():
            self.logger.debug("%s - cut choices = %s", nid, cut_choices)
        for cut_choice in cut_choices:
            if self.qid_to_breakout_score is not None:
                breakout_score = self.qid_to_breakout_score.get(cut_choice.qid, np.inf)

                is_greater_than_breakout = cut_choice.score > breakout_score
                is_close_to_breakout = np.isclose(
                    cut_choice.score, breakout_score, rtol=1e-5, atol=EPSILON
                )

                if is_greater_than_breakout or is_close_to_breakout:
                    if self.debug_logging_enabled():
                        self.logger.debug(
                            "%s - [dynamic breakout] Skipping cut choice (%s) >= breakout score (%.3f)",
                            nid,
                            cut_choice,
                            breakout_score,
                        )
                    continue

            yield cut_choice
