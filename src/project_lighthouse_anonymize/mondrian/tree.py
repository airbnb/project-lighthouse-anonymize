"""
Tree and related classes for the Mondrian algorithm.

This module implements the tree data structure used by the Core Mondrian algorithm.
The tree represents the hierarchical partitioning of data based on quasi-identifiers (QIDs).

The key classes are:
- CutChoice: Represents a selected attribute and score for partitioning
- ProposedCut: Describes a potential partition with its input and output data
- MondrianTree: The main tree structure tracking partitioning decisions
- Various NodeData classes: Store data and metadata at different tree nodes
"""

import os
import uuid
from collections import deque
from hashlib import sha256
from io import StringIO
from typing import Any, Optional

import pandas as pd
from treelib import (
    Node,  # type: ignore[reportPrivateImportUsage]  # Node is publicly exported from treelib
    Tree,  # type: ignore[reportPrivateImportUsage]  # Tree is publicly exported from treelib
)

from project_lighthouse_anonymize.mondrian.funnel_stats import FunnelStats
from project_lighthouse_anonymize.pandas_utils import hash_df


class CutChoice:
    """
    Represents a chosen quasi-identifier (QID) attribute for partitioning data.

    This class stores information about which attribute was selected for a cut,
    its numerical score (representing the cut's quality), and whether the attribute
    is categorical or numerical.
    """

    def __init__(
        self,
        qid: str,
        score: float,
        is_categorical: bool,
    ):
        """
        Initialize a CutChoice instance.

        Args:
            qid: The name of the quasi-identifier attribute being cut
            score: Numerical score representing the quality of this cut
            is_categorical: True if this attribute is categorical, False if numerical
        """
        self.qid = qid
        self.score = score
        self.is_categorical = is_categorical

    def __repr__(self) -> str:
        return f"{self.qid} ({'c' if self.is_categorical else 'n'}) (score = {self.score:.3f})"


class ProposedCut:
    """
    Represents a potential data partitioning based on a specific CutChoice.

    This class contains the input dataframe, resulting output dataframes after
    the cut, and dataframes requiring further cuts to meet anonymity constraints.
    """

    def __init__(
        self,
        cut_choice: CutChoice,
        input_df: pd.DataFrame,
        output_dfs: list[pd.DataFrame],
        further_cuts: list[pd.DataFrame],
    ):
        """
        Initialize a ProposedCut instance.

        Args:
            cut_choice: The CutChoice determining how data will be partitioned
            input_df: Original dataframe before partitioning
            output_dfs: List of resulting dataframes for which no additional partitioning may occur
            further_cuts: List of dataframes that need additional partitioning
        """
        self.cut_choice = cut_choice
        self.input_df = input_df
        self.output_dfs = output_dfs
        self.further_cuts = further_cuts

    def __repr__(self) -> str:
        output_dfs_str = [len(df) for df in self.output_dfs]
        further_cuts_str = [len(df) for df in self.further_cuts]
        return f"{self.cut_choice} len(input_df) = {len(self.input_df)}, len(output_dfs) = {output_dfs_str}, len(further_cuts) = {further_cuts_str}"


class NodeDataBase:
    """
    Base class for all node data objects in the Mondrian tree.

    This abstract base class provides common functionality shared by all node
    types in the tree, including statistics tracking and string representation.
    All concrete node classes should inherit from this class.
    """

    def __init__(self) -> None:
        """
        Initialize a NodeDataBase instance with no attached statistics.
        """
        self.stats: Optional[FunnelStats] = None

    def attach_stats(self, stats: FunnelStats) -> None:
        """
        Attach funnel statistics to this node.

        Args:
            stats: The funnel statistics to attach

        Raises:
            AssertionError: If statistics are already attached to this node
        """
        assert self.stats is None
        self.stats = stats

    def get_stats(self) -> Optional[FunnelStats]:
        """
        Get funnel statistics attached to this node.

        Returns:
            The attached funnel statistics or None if no statistics are attached
        """
        return self.stats

    def __getattr__(self, name: str) -> str:
        """
        Custom attribute access to support the 'pprint' property for tree display.

        Args:
            name: The attribute name being accessed

        Returns:
            String representation if name is 'pprint'

        Raises:
            AttributeError: For any other unknown attribute
        """
        if name == "pprint":
            return str(self)
        # For other attributes, raise AttributeError (default behavior)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def to_str(self, base_repr: str) -> str:
        """
        Create a string representation of the node, optionally including statistics.

        Args:
            base_repr: The base string representation without statistics

        Returns:
            String representation with statistics appended if available
        """
        if self.stats:
            return str(f"{base_repr}{self.stats.summary()} ")
        return str(base_repr)


class Node_Cut(NodeDataBase):
    """
    Data associated with a Tree.Node that represents a Mondrian cut.

    This class represents an internal node in the Mondrian tree where a specific
    attribute (quasi-identifier) was chosen to partition the data. It tracks
    the cut choice, original data, and suppression statistics.
    """

    def __init__(
        self,
        cut_choice: CutChoice,
        df: pd.DataFrame,
        max_suppression_n: int,
        suppression_n: int,
    ):
        """
        Initialize an instance of Node_Cut.

        Args:
            cut_choice: The chosen attribute and split information for this cut
            df: DataFrame of data under this cut, will not be saved beyond MondrianTree.create_node
            max_suppression_n: The maximum allowed suppression at this cut
            suppression_n: The actual number of records suppressed at this cut
        """
        self.cut_choice = cut_choice
        self.df = df
        self.original_df_len = len(self.df)
        self.max_suppression_n = max_suppression_n
        self.suppression_n = suppression_n
        super().__init__()

    def __repr__(self) -> str:
        """
        Create a string representation of this cut node.

        The representation includes cut details, dataframe size, and suppression statistics.
        A '***' marker is added if suppression exceeds the maximum allowed suppression.

        Returns:
            String representation of this cut node
        """
        suppression_flag = " ***" if self.suppression_n > self.max_suppression_n else ""
        return self.to_str(
            f"<Node_Cut> cut_choice = {self.cut_choice}, len(df) = {len(self.df) if self.df is not None else 0} (originally = {self.original_df_len}) max_suppression_n = {self.max_suppression_n} suppression_n = {self.suppression_n}{suppression_flag} "
        )


class Node_DeferredCutData(NodeDataBase):
    """
    Data associated with a Tree.Node that represents data to be processed.

    This class represents data that needs further partitioning.
    It tracks the dataframe waiting to be processed and suppression constraints.
    These nodes are typically expanded further as the algorithm progresses.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_suppression_n: int,
    ):
        """
        Initialize an instance of Node_DeferredCutData.

        Args:
            df: The dataframe waiting to be processed by the Mondrian algorithm
            max_suppression_n: The maximum number of records that can be suppressed
                               at or below this node in the tree
        """
        self.df = df
        self.original_df_len = len(self.df)
        self.max_suppression_n = max_suppression_n
        self.suppression_n = 0  # Initially no suppression
        super().__init__()

    def __repr__(self) -> str:
        """
        Create a string representation of this deferred cut data node.

        The representation includes dataframe size and suppression statistics.
        A '***' marker is added if suppression exceeds the maximum allowed.

        Returns:
            String representation of this deferred cut data node
        """
        suppression_flag = " ***" if self.suppression_n > self.max_suppression_n else ""
        return self.to_str(
            f"<Node_DeferredCutData> len(df) = {len(self.df)} (originally = {self.original_df_len}), max_suppression_n = {self.max_suppression_n}, suppresion_n = {self.suppression_n}{suppression_flag} "
        )


class Node_DeferredCutData__PartitionRoot(Node_DeferredCutData):
    """
    Data associated with a Tree.Node that represents data to be processed.

    This is a special case of Node_DeferredCutData which represents one of the original
    input data partitions on QID nan values. These nodes are created during the initial
    partitioning of data with missing QID values. See core.py's use of nan_generator
    for more context on how these partition roots are created.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_suppression_n: int,
    ):
        """
        Initialize a Node_DeferredCutData__PartitionRoot instance.

        Args:
            df: The dataframe for this partition root
            max_suppression_n: The maximum number of records that can be suppressed

        See initializer for Node_DeferredCutData for more details.
        """
        super().__init__(df, max_suppression_n)

    def __repr__(self) -> str:
        """
        Create a string representation of this partition root node.

        Returns:
            String representation of this partition root node
        """
        suppression_flag = " ***" if self.suppression_n > self.max_suppression_n else ""
        return self.to_str(
            f"<Node_DeferredCutData__PartitionRoot> len(df) = {len(self.df)} (originally = {self.original_df_len}), max_suppression_n = {self.max_suppression_n}, suppresion_n = {self.suppression_n}{suppression_flag} "
        )


class Node_FinalCutData(NodeDataBase):
    """
    Data associated with a Tree.Node that represents processed data.

    This class represents a leaf node in the Mondrian tree where data has been
    fully processed (generalized and suppressed as needed). These nodes contain
    the final anonymized data partitions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
    ):
        """
        Initialize an instance of Node_FinalCutData.

        Args:
            df: Dataframe containing data that has been fully processed
                (generalized and suppressed as needed)
        """
        self.df = df
        self.original_df_len = len(self.df)
        super().__init__()

    def __repr__(self) -> str:
        """
        Create a string representation of this final cut data node.

        Returns:
            String representation of this final cut data node
        """
        return self.to_str(
            f"<Node_FinalCutData> len(df) = {len(self.df)} (originally = {self.original_df_len}) "
        )


class MondrianTree(Tree):
    """
    Tree tracking the Mondrian algorithm state.

    This class extends treelib.Tree to represent the hierarchical partitioning
    structure created by the Mondrian algorithm. It tracks the partitioning process
    from the initial data through a series of cuts to the final anonymized partitions.

    Each Node's data object is a subclass of NodeDataBase, with different node types
    representing different stages of processing:
    - Node_DeferredCutData__PartitionRoot: Initial partitions by NaN values
    - Node_DeferredCutData: Data awaiting further processing
    - Node_Cut: Internal nodes where cuts were made
    - Node_FinalCutData: Leaf nodes with fully anonymized data
    """

    def __init__(
        self,
        deterministic_identifiers: bool = False,
    ):
        """
        Initialize a MondrianTree instance.

        Args:
            deterministic_identifiers: When True, uses content-based hashing to
                                      generate deterministic node identifiers
                                      instead of random UUIDs
        """
        self.deterministic_identifiers = deterministic_identifiers
        self.output_io: Optional[StringIO] = None  # Used for capturing tree visualization output
        super().__init__()

    # Signature intentionally differs from Tree.create_node for Mondrian-specific API
    def create_node(  # type: ignore[override]
        self,
        parent: Optional[Node],
        data: NodeDataBase,
    ) -> Node:
        """
        Create a Node in this MondrianTree and return it.

        Args:
            parent: The parent node, or None if creating the root node
            data: The node data object (a subclass of NodeDataBase)

        Returns:
            The newly created Node object

        Notes:
            After creation, if data is a Node_Cut, its dataframe is cleared to reduce
            memory usage since it's only needed for node ID generation.
        """
        try:
            return super().create_node(
                identifier=self.generate_node_id(parent, data),
                parent=parent,
                data=data,
            )
        finally:
            # TODO(Later) I hacked in Node_Cut.df to help with generate_node_id below; this should be refactored to be more elegant
            if isinstance(data, Node_Cut):
                data.df = None  # type: ignore[reportAttributeAccessIssue]  # Intentional hack to free memory, df should be Optional[DataFrame]

    def generate_node_id(
        self,
        parent: Optional[Node],
        data: NodeDataBase,
    ) -> str:
        """
        Generate an identifier for a new MondrianTree.Node.

        Args:
            parent: The parent node, or None if creating the root node
            data: The node data object (a subclass of NodeDataBase)

        Returns:
            A string identifier for the node, either deterministic (based on content hash)
            or non-deterministic (based on process ID and UUID)
        """
        if self.deterministic_identifiers:
            # Generate a deterministic ID based on the data content
            hashcode = hash(
                (
                    int.from_bytes(sha256(str(type(data)).encode()).digest(), "big"),
                    hash_df(data.df),  # type: ignore[reportArgumentType]  # data.df is DataFrame when generate_node_id is called, None assignment happens after
                )
            )
            identifier = hex(hashcode)[2:]
        else:
            # Generate a unique ID based on process ID and UUID
            identifier = f"{os.getpid()}-{str(uuid.uuid4())}"
        return identifier

    def __stitch_in_subtree_update_funnel_stats(self, nid: str, sub_tree: Tree) -> None:
        """
        Helper function to propagate funnel statistics up the tree.

        When a subtree is integrated into the main tree, this method ensures that
        funnel statistics from the subtree's root are merged into all ancestor nodes,
        maintaining accurate aggregated statistics throughout the tree hierarchy.

        Args:
            nid: Node identifier where subtree is being attached
            sub_tree: The subtree being attached to the main tree
        """
        root_node = sub_tree.get_node(sub_tree.root)
        assert root_node is not None
        stats = root_node.data.get_stats()
        if stats is not None:
            for nid_to_adjust in self.rsearch(nid):
                node_to_adjust = self.get_node(nid_to_adjust)
                assert node_to_adjust is not None
                stats_to_adjust = node_to_adjust.data.get_stats()
                if stats_to_adjust is not None:
                    stats_to_adjust.merge(stats)

    def __stitch_in_subtree_cleanup(self, nid: str) -> None:
        """
        Cleanup helper function for stitch_in_subtree to clear dataframes and reduce memory usage.

        After a subtree is integrated and all deferred work is processed, this method
        clears dataframes from intermediate nodes that are no longer needed, helping
        to reduce the overall memory footprint of the Mondrian algorithm.

        Args:
            nid: Node identifier to potentially clear dataframe for
        """
        node = self.get_node(nid)
        assert node is not None
        if not self.is_deferred_work_definitely_remaining(node.identifier):
            if isinstance(node.data, Node_DeferredCutData):
                # We can clear out this intermediate dataframe at this point because
                # if the work was deferred then, once a result arrives for some of that
                # deferred work then all of the deferred work must have already been enqueued.
                # This reduces memory usage for Mondrian.
                node.data.df = node.data.df.sample(0).copy()

    def stitch_in_subtree_recursive(self, nid: str, sub_tree: "MondrianTree") -> int:
        """
        Integrate a subtree into the main tree with optimized performance.

        This method pastes a subtree and calculates the total suppression count, but
        without tracking deferred work or updating ancestor node suppression counts.
        It's an optimized version used specifically for recursive calls from make_cut
        to avoid expensive tracking operations when building the tree.

        Args:
            nid: Node identifier whose node should be parent to the sub-tree
            sub_tree: The MondrianTree to stitch in

        Returns:
            Total records suppressed in this sub-tree
        """
        self.paste(nid, sub_tree)

        # Calculate total_suppressed without updating ancestors
        total_suppressed = 0
        for cnid in self.expand_tree(nid=sub_tree.root, mode=self.WIDTH):
            cnode = self.get_node(cnid)
            assert cnode is not None
            if isinstance(cnode.data, Node_Cut):
                total_suppressed += cnode.data.suppression_n

        self.__stitch_in_subtree_update_funnel_stats(nid, sub_tree)
        self.__stitch_in_subtree_cleanup(nid)
        return total_suppressed

    def stitch_in_subtree(
        self,
        nid: str,
        sub_tree: "MondrianTree",
    ) -> tuple[deque[Node], int]:
        """
        Integrate a subtree into the main tree and track work remaining.

        This is the main method for integrating results from the Core Mondrian algorithm
        into the tree. It attaches a subtree to the specified node, tracks any remaining
        deferred work, updates suppression counts throughout the ancestry chain, and
        cleans up intermediate dataframes to reduce memory usage.

        Args:
            nid: Node identifier whose node should be parent to the sub-tree
            sub_tree: The MondrianTree to stitch in

        Returns:
            A tuple containing:
            1. A deque of nodes within the subtree that have deferred work remaining
            2. Total records suppressed in this subtree (a lower bound if deferred work remains)
        """
        self.paste(nid, sub_tree)
        try:
            _nodes_with_definite_deferred_work_remaining: deque[Node] = deque([])
            total_suppressed = 0

            # Find nodes with deferred work and calculate total suppression
            for cnid in self.expand_tree(nid=sub_tree.root, mode=self.WIDTH):
                cnode = self.get_node(cnid)
                assert cnode is not None
                if self.is_deferred_work_definitely_remaining(cnid):
                    _nodes_with_definite_deferred_work_remaining.append(cnode)
                if isinstance(cnode.data, Node_Cut):
                    total_suppressed += cnode.data.suppression_n

            # Update suppression counts in ancestor nodes
            if total_suppressed > 0:
                for nid_to_adjust in self.rsearch(nid):
                    node_to_adjust = self.get_node(nid_to_adjust)
                    assert node_to_adjust is not None
                    if isinstance(node_to_adjust.data, Node_FinalCutData):
                        # Stop before the root for the partition where all QIDs are NaN
                        break
                    node_to_adjust.data.suppression_n += total_suppressed
                    if isinstance(node_to_adjust.data, Node_DeferredCutData__PartitionRoot):
                        # Stop after the roots for partitions where some or no QIDs are NaN
                        break

            # Update funnel statistics
            self.__stitch_in_subtree_update_funnel_stats(nid, sub_tree)

            return (
                _nodes_with_definite_deferred_work_remaining,
                total_suppressed,
            )
        finally:
            # Clean up intermediate dataframes to reduce memory usage
            self.__stitch_in_subtree_cleanup(nid)

    def is_deferred_work_definitely_remaining(self, nid: str) -> bool:
        """
        Check if a node definitely has deferred work remaining to be processed.

        This method identifies nodes that contain data waiting to be processed by the
        Core Mondrian algorithm. A node has deferred work if it's a DeferredCutData node
        and has no children (indicating it hasn't been partitioned yet).

        Args:
            nid: Node identifier to check

        Returns:
            True if the node definitely has deferred work, False otherwise

        Notes:
            This is a conservative check - it may return False even when there is
            still some work remaining, but will never return True when no work remains.
        """
        node = self.get_node(nid)
        assert node is not None
        return isinstance(node.data, Node_DeferredCutData) and not self.children(nid)

    def is_deferred_cut_non_empty(self, nid: str) -> bool:
        """
        Check if a node is a deferred work node and has a non-empty dataframe.

        This method checks both that a node is marked for deferred processing
        and that it contains actual data to process (non-empty dataframe).

        Args:
            nid: Node identifier to check

        Returns:
            True if the node is a deferred cut node with a non-empty dataframe,
            False otherwise
        """
        node = self.get_node(nid)
        assert node is not None
        return isinstance(node.data, Node_DeferredCutData) and len(node.data.df) > 0

    def candidate_solution(
        self,
        qids: list[str],
        all_work_should_be_done: bool,
        track_cuts: bool,
        node_identifier_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Extract the final anonymized data from the Mondrian tree.

        This method traverses the tree, collects all Node_FinalCutData leaf nodes
        (which contain the fully anonymized data partitions), and combines them
        into a single dataframe that represents the complete anonymized dataset.

        When track_cuts is True, the method also adds metadata columns showing how
        many cuts were made on each quasi-identifier to reach each partition.

        Args
        ----
        qids: List of all quasi-identifier attributes used to build this tree
        all_work_should_be_done: If True, verify that all processing is complete
                                 and fail with an assertion error if not
        track_cuts: If True, add columns showing the number of cuts on each QID
                    in the format "{qid}_cuts"
        node_identifier_col: If not None, add a column with this name containing
                            the node identifier from which each row came

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the complete anonymized dataset

        Raises
        ------
        AssertionError
            If all_work_should_be_done is True and there is still
            deferred work remaining in the tree
        """
        output_dfs = []
        assert self.root is not None
        queue = deque([self.root])

        while queue:
            nid = queue.popleft()
            assert nid is not None

            # Verify all work is complete if requested
            if all_work_should_be_done:
                assert not self.is_deferred_work_definitely_remaining(nid), (
                    f"Deferred work remaining in node {nid}"
                )
                assert not self.is_deferred_cut_non_empty(nid), (
                    f"Deferred work node w/ non-empty dataframe remaining for node {nid}"
                )

            node = self.get_node(nid)
            assert node is not None

            # Collect data from final cut nodes (leaves with anonymized data)
            if isinstance(node.data, Node_FinalCutData):
                df = self.__add_cut_counts(qids, nid, node.data.df) if track_cuts else node.data.df
                if node_identifier_col is not None:
                    df[node_identifier_col] = node.identifier
                output_dfs.append(df)

            # Continue traversal
            for child in self.children(nid):
                queue.append(child.identifier)

        # Combine all partitions into the final solution
        return pd.concat(output_dfs)

    def __add_cut_counts(self, qids: list[str], nid: Any, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function to annotate a final cut dataframe with cut counts.

        This method traces the path from a leaf node back to the root, counting
        how many times each quasi-identifier was used for partitioning. It adds
        these counts as new columns to the dataframe, which can be useful for
        analyzing the anonymization process and the resulting data quality.

        Args:
            qids: List of all quasi-identifier attributes used to build this tree
            nid: Node identifier associated with this final cut (leaf node)
            df: DataFrame associated with this final cut

        Returns:
            A copy of the input dataframe with additional columns in the format
            "{qid}_cuts" showing the number of cuts made on each QID
        """
        df = df.copy()
        qid_to_count: dict[str, int] = {}

        # Traverse from this node back to the root, counting cuts by QID
        for cnid in self.rsearch(nid):
            cnode = self.get_node(cnid)
            assert cnode is not None
            if isinstance(cnode.data, Node_Cut):
                qid = cnode.data.cut_choice.qid
                if qid in qids:  # ignore e.g. implementation.EMPTY_CUT_QID
                    qid_to_count[qid] = qid_to_count.get(qid, 0) + 1

        # Add cut count columns for each QID that was used
        # Note: We intentionally don't add 0 values for QIDs with no cuts
        # because having an NaN value in the final output_df makes analysis easier
        for qid, count in qid_to_count.items():
            df[f"{qid}_cuts"] = count

        return df

    def pprint(self) -> str:
        """
        Return a pretty-printed string representation of the tree.

        This is a customized version of Tree.show() that captures the output into
        a string instead of printing it directly. This is especially useful for
        logging the tree structure or for debugging purposes.

        Returns:
            A string containing the tree visualization with all nodes and their data
        """
        self.output_io = StringIO()
        try:

            def collect(line: bytes) -> None:
                assert self.output_io is not None
                self.output_io.write(line.decode("utf-8") + "\n")

            # Use internal treelib method to generate the tree visualization
            self._Tree__print_backend(  # type: ignore[attr-defined]
                nid=None,
                level=Tree.ROOT,
                idhidden=False,
                filter=None,
                key=None,
                reverse=False,
                line_type="ascii-ex",
                data_property="pprint",  # Use the pprint property of each node's data
                func=collect,
            )
            result = self.output_io.getvalue()
            return str(result)
        finally:
            # Clean up the StringIO object to prevent memory leaks
            self.output_io = None
