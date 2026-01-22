"""
Implementation of the Original Mondrian algorithm for k-anonymity.

IMPORTANT: This implementation is intended for research and development purposes only.
           For production use cases, please use Core Mondrian from the mondrian/ module.

This implementation differs from the algorithm described in MONDRIAN.md in the following key ways:

1.  **Processing Strategy (Iterative vs. Recursive)**: This implementation uses an iterative,
    queue-based approach to manage partitions, which is functionally identical to the recursive
    partitioning strategy detailed in MONDRIAN.md. The iterative approach was chosen to avoid
    Python's recursion depth limitations, making the implementation more robust for large datasets
    and preventing potential stack overflow errors.

2.  **Quasi-Identifier (QID) Support (Numerical Only)**: This implementation exclusively
    supports numerical QIDs. MONDRIAN.md, on the other hand, describes handling for
    both numerical and categorical QIDs, including specific generalization techniques
    (e.g., using value generalization hierarchies or taxonomies) for categorical attributes.
    Attempting to use categorical QIDs with this implementation will result in a `ValueError`.

3.  **Generalization Method for Numerical QIDs (Microaggregation vs. Range)**: For numerical
    QIDs, values within a final partition are generalized by replacing them with their
    arithmetic mean (a form of microaggregation). In contrast, MONDRIAN.md specifies
    generalizing numerical QIDs to the `[min, max]` range of values observed within
    that partition.

4.  **Input Data Pre-condition (No NaN Values in QIDs)**: This implementation strictly
    requires that QID columns in the input DataFrame contain no NaN (Not a Number) values.
    The algorithm described in MONDRIAN.md does not explicitly detail procedures for
    handling missing or NaN values, often assuming pre-cleaned data. Processing data
    containing NaNs in QID columns with this implementation will likely lead to errors
    during numerical calculations or data processing steps.

Standard Mondrian principles that this implementation *does* follow include:

-   **Dimension Selection**: Choosing the dimension (QID) with the widest normalized range for splitting.
-   **Split Point**: Splitting partitions at the median value of the selected dimension.

API::

    class OriginalMondrian:
        __init__(self, logger: logging.Logger, k: int)
        anonymize(self, input_df: pd.DataFrame, qids: List[str]) -> pd.DataFrame
        validate(self, df: pd.DataFrame, qids: List[str]) -> bool

Integration:

- Use through the k_anonymize wrapper function with use_original_mondrian=True
"""

import logging
from collections import deque
from typing import cast

import numpy as np
import pandas as pd


class OriginalMondrian:
    """
    Implements the original Mondrian algorithm for multidimensional k-anonymity (numerical QIDs only).
    """

    def __init__(self, logger: logging.Logger, k: int):
        """
        Parameters
        ----------
        logger : logging.Logger
            Logger for reporting process/info/warnings/errors.
        k : int
            The minimal number of records in each equivalence class (k-anonymity).
        """
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.logger = logger

    # setup method removed as it was only needed for CoreMondrian compatibility

    def anonymize(
        self, input_df: pd.DataFrame, qids: list[str], *args: object, **kwargs: object
    ) -> pd.DataFrame:
        """
        Anonymize the input dataframe using the Mondrian algorithm on the specified QIDs.
        Only numerical QIDs are supported.

        Additional parameters are ignored to maintain compatibility with CoreMondrian interface.

        Parameters
        ----------
        input_df : pd.DataFrame
            The input data to anonymize.
        qids : List[str]
            The columns to use as quasi-identifiers (must be numerical).
        *args, **kwargs :
            Additional arguments are ignored (for compatibility with CoreMondrian)

        Returns
        -------
        pd.DataFrame
            The k-anonymized, generalized dataframe (same columns as input).
            In each final group, QID values are replaced with the average value of that QI.
        """
        if len(input_df) == 0 or len(qids) == 0:
            self.logger.warning("Input DataFrame empty or QID set empty. Returning copy.")
            return input_df.copy()
        if len(input_df) < self.k:
            self.logger.warning(
                f"Input dataframe has fewer than k={self.k} records; cannot anonymize"
            )
            return pd.DataFrame(columns=input_df.columns)

        # Ensure QIDs are numerical
        for qid in qids:
            if not np.issubdtype(input_df[qid].dtype, np.number):  # type: ignore[reportArgumentType]  # dtype is valid for issubdtype but pyright stubs are overly strict
                self.logger.error(
                    f"QID {qid} is not numerical. Original Mondrian only supports numerical QIDs."
                )
                raise ValueError(
                    f"QID {qid} is not numerical. Original Mondrian only supports numerical QIDs."
                )

        # Compute global min/max for QID normalization (ranges)
        qid_domains = {}
        for qid in qids:
            colvals = input_df[qid]  # No longer need to drop NaNs since we assume there are none
            dom_min, dom_max = colvals.min(), colvals.max()
            qid_domains[qid] = (dom_min, dom_max)

        self.logger.info(
            f"Starting Mondrian partitioning with k={self.k} on {len(input_df)} records with QIDs={qids}"
        )

        # Non-recursive queue-based partitioning
        equivalence_classes = []
        partition_queue = deque([input_df])

        while partition_queue:
            # Get the next partition to process
            current_partition = partition_queue.popleft()

            # Check stopping conditions
            if len(current_partition) < 2 * self.k:
                # Not enough records to split both parts with >=k
                equivalence_classes.append(current_partition)
                continue

            if self._is_partition_homogeneous(current_partition, qids):
                # All QID values identical (partition is pure)
                equivalence_classes.append(current_partition)
                continue

            # Select the QID to split on (with largest normalized range)
            norm_ranges = {}
            selected_qid = None
            for qid in qids:
                colvals = current_partition[qid]  # No longer need to drop NaNs

                attr_min, attr_max = colvals.min(), colvals.max()
                if attr_min == attr_max:
                    # Can't split: all values the same
                    norm_ranges[qid] = -np.inf
                    continue

                dom_min, dom_max = qid_domains[qid]
                denominator = dom_max - dom_min if dom_max != dom_min else 1.0  # type: ignore[reportOperatorIssue]  # numeric dtypes support subtraction
                norm_range = (attr_max - attr_min) / denominator if denominator != 0 else 0.0  # type: ignore[reportOperatorIssue]  # numeric dtypes support subtraction
                norm_ranges[qid] = norm_range

            if not norm_ranges:
                # No valid QID to split on
                equivalence_classes.append(current_partition)
                continue

            selected_qid = max(norm_ranges, key=lambda q: norm_ranges[q])
            if norm_ranges[selected_qid] <= 0 or norm_ranges[selected_qid] == -np.inf:
                # No valid split dimension found
                equivalence_classes.append(current_partition)
                continue

            # Get values for selected QID (no need to check for NaNs since we assume there aren't any)
            colvals = current_partition[selected_qid].values

            # Split at median value
            median = np.median(colvals)  # type: ignore[reportCallIssue,reportArgumentType]  # colvals is numeric ndarray, np.median accepts it
            left_mask = current_partition[selected_qid] <= median
            right_mask = ~left_mask

            d_left = current_partition[left_mask]
            d_right = current_partition[right_mask]

            # If either split violates k-anonymity, do not split further
            if len(d_left) < self.k or len(d_right) < self.k:
                equivalence_classes.append(current_partition)
                continue

            # Add both partitions to the processing queue
            partition_queue.append(cast(pd.DataFrame, d_left))
            partition_queue.append(cast(pd.DataFrame, d_right))

        # Generalize each partition and assemble output
        generalized_rows = []
        for part in equivalence_classes:
            generalized = self._generalize_partition(part, qids)
            generalized_rows.append(generalized)

        if generalized_rows:
            anon_df = pd.concat(generalized_rows, ignore_index=True)
        else:
            anon_df = pd.DataFrame(columns=input_df.columns)

        self.logger.info(f"Completed Mondrian anonymization: {len(anon_df)} records in output")
        return anon_df

    # Remove the recursive _mondrian_split method since we've replaced it with queue-based approach

    def _is_partition_homogeneous(self, df: pd.DataFrame, qids: list[str]) -> bool:
        """
        Check if all records in the partition have identical values for all QIDs.
        """
        for qid in qids:
            vals = df[qid].unique()  # No longer need to dropna
            if len(vals) > 1:
                return False
        return True

    def _generalize_partition(self, df: pd.DataFrame, qids: list[str]) -> pd.DataFrame:
        """
        Generalize each QID by replacing values with the average value in the partition.
        Uses microaggregation approach (averaging) for numerical values.

        If all values in a column are identical, preserves the original data type by using
        the first value instead of computing the mean.

        Returns
            pd.DataFrame: Copy of df with each QID replaced by the group average.
        """
        result = df.copy()
        for qid in qids:
            col = result[qid]

            # Check if all values are the same to preserve data type
            unique_values = col.unique()  # No longer need to dropna
            if len(unique_values) == 1:
                # If all values are the same, use the original value to preserve data type
                result[qid] = unique_values[0]
            else:
                # Calculate the average for this QI in the partition
                avg_value = np.mean(col)
                # Assign the average to all rows in the partition
                result[qid] = avg_value
        return result

    def validate(self, df: pd.DataFrame, qids: list[str]) -> bool:
        """
        Validate that the dataframe satisfies k-anonymity with respect to given QID columns.

        Returns
        -------
        bool
            True if the dataframe satisfies k-anonymity, False otherwise
        """
        if len(df) == 0:
            return True
        # Count group sizes in terms of QIDs (as string ranges)
        group_counts = df.groupby(qids).size()
        return group_counts.min() >= self.k if not group_counts.empty else True  # type: ignore[reportOperatorIssue]  # .min() returns int, >= comparison is valid

    # could_be_a_final_cut method removed as it was only needed for CoreMondrian compatibility
