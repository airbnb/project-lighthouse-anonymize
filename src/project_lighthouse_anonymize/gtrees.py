"""
Classes and functions to represent and use generalization trees.

This module provides tools for creating, manipulating, and using generalization
trees, which are hierarchical structures used for data anonymization and
generalization. These trees define how specific values can be generalized into
broader categories.
"""

import itertools as it
import json
import math
import tempfile
from collections import deque, namedtuple
from typing import Any, Optional

import numpy as np
from treelib import (
    Node,  # type: ignore[reportPrivateImportUsage]  # Node is publicly exported from treelib
    Tree,  # type: ignore[reportPrivateImportUsage]  # Tree is publicly exported from treelib
)

from project_lighthouse_anonymize.constants import NOT_DEFINED_NA

GTreeData = namedtuple("GTreeData", ["geometric_size"])


class GTree(Tree):
    """
    Generalization tree for hierarchical data representation.

    A generalization tree represents hierarchical relationships between values,
    with more specific values positioned as leaf nodes and increasingly generalized
    values positioned higher in the tree. This structure supports operations for
    privacy-preserving data transformations by enabling the replacement of specific
    values with more general ones.

    This class extends the treelib.Tree class with additional functionality for
    efficient lookups and manipulation of generalization hierarchies.

    Attributes
    ----------
    value_to_highest_node_nid : dict
        Maps values to the highest node ID containing that value
    nid_to_descendant_leaf_values : dict
        Maps node IDs to their descendant leaf values
    value_to_leaf_node_nid : dict
        Maps values to leaf node IDs
    """

    def __init__(
        self,
        tree: Optional["GTree"] = None,
        json_obj: Optional[dict[str, Any]] = None,
        identifier: Optional[str] = None,
        deep: bool = True,
    ) -> None:
        """
        Initialize a generalization tree.

        Parameters
        ----------
        tree : GTree, optional
            An existing GTree to copy, by default None
        json_obj : Dict[str, Any], optional
            A JSON object representation of a GTree to load, by default None
        identifier : str, optional
            A string identifier for the tree, by default None
        deep : bool, optional
            Whether to perform a deep copy when copying from an existing tree, by default True

        Notes
        -----
        Either `tree` or `json_obj` can be provided, but not both. If `tree` is provided,
        the new GTree will be a copy of the existing one. If `json_obj` is provided,
        the tree will be built from that configuration.
        """
        super().__init__(tree=tree, deep=deep, identifier=identifier)
        assert not (tree is not None and json_obj is not None)
        self.value_to_highest_node_nid: dict[Any, str] = {}
        self.nid_to_descendant_leaf_values: dict[str, tuple[Any, ...]] = {}
        self.value_to_leaf_node_nid: dict[Any, str] = {}
        if tree is not None:
            self.value_to_highest_node_nid = dict(tree.value_to_highest_node_nid)
            self.nid_to_descendant_leaf_values = dict(tree.nid_to_descendant_leaf_values)
            self.value_to_leaf_node_nid = dict(tree.value_to_leaf_node_nid)
        elif json_obj is not None:
            self.from_config_json(json_obj)

    def pprint(self) -> str:
        """
        Return a pretty (multi-line) string representation of the generalization tree.

        Returns
        -------
        str
            String representation of the tree structure
        """
        result = self.show(stdout=False)
        return str(result) if result is not None else ""

    def pprint_geometric_sizes(self) -> str:
        """
        Return a pretty (multi-line) string showing the tree's geometric sizes.

        Returns
        -------
        str
            String representation of the tree with geometric size values
        """
        result = self.show(data_property="geometric_size", stdout=False)
        return str(result) if result is not None else ""

    def create_node(  # type: ignore[override]
        self,
        value: Any,
        parent: Optional[Node] = None,
        geometric_size: float = NOT_DEFINED_NA,
        identifier: Optional[str] = None,
    ) -> Node:  # pylint: disable=arguments-differ,arguments-renamed
        """
        Create a new node in the generalization tree.

        Parameters
        ----------
        value : Any
            Value of the node, typically a string. This is what is present in the
            input/output dataframes to/from anonymization. Must be hashable since
            it's used as a dictionary key.
        parent : Node, optional
            Parent node to which this node will be attached, by default None
        geometric_size : float, optional
            Geometric size, used for computing RILM (Relative Information Loss Measure),
            by default NOT_DEFINED_NA
        identifier : str, optional
            Unique identifier for the node, by default None

        Returns
        -------
        Node
            The newly created Node object

        Notes
        -----
        This method resets the internal mapping dictionaries since the tree structure
        has changed. These mappings will be rebuilt when needed.
        """
        self.value_to_highest_node_nid = {}
        self.nid_to_descendant_leaf_values = {}
        self.value_to_leaf_node_nid = {}
        return super().create_node(
            tag=value,
            identifier=identifier,
            parent=parent,
            data=GTreeData(geometric_size),
        )

    def remove_node(self, identifier: str) -> None:  # type: ignore[override]
        """
        Remove a node from the tree.

        Parameters
        ----------
        identifier : str
            The identifier of the node to remove

        Notes
        -----
        This method resets the internal mapping dictionaries since the tree
        structure has changed. These mappings will be rebuilt when needed.
        """
        self.value_to_highest_node_nid = {}
        self.nid_to_descendant_leaf_values = {}
        self.value_to_leaf_node_nid = {}
        super().remove_node(identifier)

    def remove_subtree(self, nid: str, identifier: Optional[str] = None) -> None:
        """
        Remove the subtree starting at node with given ID.

        Parameters
        ----------
        nid : str
            The identifier of the root node of the subtree to remove
        identifier : str, optional
            If provided, only remove the subtree if the node has this identifier,
            by default None

        Notes
        -----
        This method resets the internal mapping dictionaries since the tree
        structure has changed. These mappings will be rebuilt when needed.
        """
        self.value_to_highest_node_nid = {}
        self.nid_to_descendant_leaf_values = {}
        self.value_to_leaf_node_nid = {}
        super().remove_subtree(nid, identifier=identifier)

    def update_highest_node_with_value_if(self) -> bool:
        """
        Update the internal mapping for efficient value-to-node lookups.

        This method builds the mapping of values to their highest nodes in the tree
        if the mapping doesn't already exist. This optimizes subsequent calls to
        get_highest_node_with_value().

        Returns
        -------
        bool
            True if the mapping was updated, False if it was already populated
        """
        if not self.value_to_highest_node_nid:
            queue = deque([self.root] if self.root is not None else [])
            while queue:
                nid = queue.popleft()
                # process nid
                node = self.get_node(nid)
                assert node is not None
                node_value = self.get_value(node)
                if node_value not in self.value_to_highest_node_nid:
                    self.value_to_highest_node_nid[node_value] = node.identifier
                # add children (bfs)
                queue.extend([child.identifier for child in self.children(nid)])
            return True
        return False

    def get_highest_node_with_value(self, value: Any) -> Optional[Node]:
        """
        Get the highest node in the tree with a given value.

        Finds the highest node (closest to the root) in the generalization tree
        that has the specified value. If multiple nodes at the same level have
        the same value, the one returned is arbitrary.

        Parameters
        ----------
        value : Any
            The value to search for

        Returns
        -------
        Node or None
            The highest node with the specified value, or None if not found
        """
        self.update_highest_node_with_value_if()
        nid = self.value_to_highest_node_nid.get(value, None)
        return self.get_node(nid) if nid is not None else None

    def update_descendant_leaf_values_if(self) -> bool:
        """
        Update the internal mapping of nodes to their descendant leaf values.

        This method builds the mapping of node IDs to their descendant leaf values
        if the mapping doesn't already exist. This optimizes subsequent calls to
        descendant_leaf_values().

        Returns
        -------
        bool
            True if the mapping was updated, False if it was already populated

        Notes
        -----
        The method works bottom-up, first processing leaves and then moving up
        the tree, ensuring that a node is only processed after all its children
        have been processed.
        """
        if not self.nid_to_descendant_leaf_values:
            # We work our way up the tree, starting with leaves
            queue = deque(self.leaves())
            while queue:
                node = queue.popleft()
                if node.identifier not in self.nid_to_descendant_leaf_values:
                    children = list(self.children(node.identifier))
                    if all(
                        child.identifier in self.nid_to_descendant_leaf_values for child in children
                    ):
                        descendant_leaf_values_tups = []
                        # base case: node is a leaf
                        if node.is_leaf():
                            descendant_leaf_values_tups.append(tuple([self.get_value(node)]))
                        for child in children:
                            descendant_leaf_values_tups.append(
                                self.nid_to_descendant_leaf_values[child.identifier]
                            )
                        self.nid_to_descendant_leaf_values[node.identifier] = tuple(
                            it.chain(*descendant_leaf_values_tups)
                        )
                        parent = self.parent(node.identifier)
                        if parent is not None:
                            queue.append(parent)
                    else:
                        # push to end of queue, we still need to compute its children
                        queue.append(node)
                else:
                    # node already processed--this can happen because each child adds its parent
                    pass
            return True
        return False

    def descendant_leaf_values(self, node: Node) -> tuple[Any, ...]:
        """
        Get all values from leaf nodes that are descendants of the given node.

        Parameters
        ----------
        node : Node
            The node whose descendant leaf values should be returned

        Returns
        -------
        Tuple[Any, ...]
            A tuple containing all values from the leaf nodes that are
            descendants of the given node

        Notes
        -----
        This method uses a cached mapping of node IDs to descendant leaf values
        for efficiency. The mapping is built if it doesn't exist.
        """
        self.update_descendant_leaf_values_if()
        return self.nid_to_descendant_leaf_values[node.identifier]

    def update_lowest_node_with_descendant_leaves_if(self) -> bool:
        """
        Update the internal mapping of leaf values to their corresponding nodes.

        This method builds the mapping of leaf values to their node IDs if the
        mapping doesn't already exist. This optimizes subsequent calls to
        get_lowest_node_with_descendant_leaves().

        Returns
        -------
        bool
            True if the mapping was updated, False if it was already populated
        """
        if not self.value_to_leaf_node_nid:
            for leaf in self.leaves():
                self.value_to_leaf_node_nid[self.get_value(leaf)] = leaf.identifier
            return True
        return False

    def get_lowest_node_with_descendant_leaves(
        self, values: set[Any], start_value: Optional[Any] = None
    ) -> Optional[Node]:
        """
        Find the lowest common ancestor node for a set of leaf values.

        Locates the lowest node in the generalization tree where all the provided
        leaf values are descendants of that node. This is essentially finding the
        lowest common ancestor in the tree for the specified leaf values.

        Parameters
        ----------
        values : Set[Any]
            Set of leaf values whose common ancestor is being sought
        start_value : Any, optional
            Value to find node to start the search from; if None, starts from root,
            by default None

        Returns
        -------
        Node or None
            The lowest node that has all specified leaf values as descendants,
            or None if no such node exists

        Raises
        ------
        ValueError
            If the values parameter is empty
        """
        if not values:
            raise ValueError("values param must not be empty")
        self.update_lowest_node_with_descendant_leaves_if()
        root_node = self.get_node(self.root)
        highest_node = (
            self.get_highest_node_with_value(start_value) if start_value is not None else root_node
        )
        assert highest_node is not None
        highest_node_parent = self.parent(highest_node.identifier)
        leaf_nodes_list = []
        for value in values:
            node = self.get_node(self.value_to_leaf_node_nid[value])
            assert node is not None
            leaf_nodes_list.append(node)
        leaf_nodes = set(leaf_nodes_list)

        walk_from_highest_node_to_leaf = []
        for leaf in leaf_nodes:
            nodes_in_path = []
            for nid in self.rsearch(leaf.identifier):
                node = self.get_node(nid)
                assert node is not None
                nodes_in_path.append(node)
            walk_from_highest_node_to_leaf.append(
                reversed(
                    list(
                        # reverse so that it's a walk from highest_node until leaf
                        it.takewhile(
                            # stop at highest_node, ie. don't walk all the way to root if we don't have to
                            lambda node: node != highest_node_parent,
                            nodes_in_path,
                        )
                    )
                )
            )
        # walk along the pathes from highest down and stop at the node before they diverge
        prev_walk_node = None
        for walk_nodes in zip(*walk_from_highest_node_to_leaf):
            first_node = walk_nodes[0]
            if all(walk_node == first_node for walk_node in walk_nodes):
                prev_walk_node = first_node
            else:
                return prev_walk_node
        return prev_walk_node

    def get_value(self, node: Node) -> Any:
        """
        Get the value stored in a node.

        Parameters
        ----------
        node : Node
            The node from which to retrieve the value

        Returns
        -------
        Any
            The value stored in the node (accessed via the tag attribute)
        """
        return node.tag

    def get_geometric_size(self, node: Node) -> float:
        """
        Get the geometric size of a node.

        Parameters
        ----------
        node : Node
            The node from which to retrieve the geometric size

        Returns
        -------
        float
            The geometric size of the node, which may be NaN if undefined

        Notes
        -----
        Geometric size is used for computing Relative Information Loss Measure (RILM)
        during anonymization operations.
        """
        return float(node.data.geometric_size)

    def add_default_geometric_sizes(self) -> None:
        """
        Compute and set default geometric sizes for all nodes in the tree.

        For each node, the geometric size is computed as 10^i - 1, where i is the
        maximum number of unique values from the node to all of its leaves.
        This reflects the information loss when generalizing from specific values
        to more general categories.

        Notes
        -----
        This method modifies the tree in place by updating the geometric_size
        attribute of each node's data.

        TODO(Later) no longer mutate in place
        """
        #
        # First we compute the unique level of every node
        #

        node_to_unique_level = {node: 0 for node in self.leaves()}

        # We work our way up the tree.
        # The queue starts with all parents of leaves
        # TODO(Later) merge above into the queue-based algorithm
        # and start the queue with all leaves.
        queue = deque([self.parent(node.identifier) for node in self.leaves()])
        while queue:
            node = queue.popleft()
            assert node is not None
            if node not in node_to_unique_level:
                children = list(self.children(node.identifier))
                if all(child in node_to_unique_level for child in children):
                    unique_level = max(
                        [
                            node_to_unique_level[child]
                            + (0 if self.get_value(child) == self.get_value(node) else 1)
                            for child in children
                        ]
                    )
                    node_to_unique_level[node] = unique_level
                    parent = self.parent(node.identifier)
                    if parent is not None:
                        queue.append(parent)
                else:
                    # push to end of queue, we still need to compute its children
                    queue.append(node)
            else:
                # node already processed--this can happen because each child adds its parent
                pass

        #
        # Now we use unique level to compute and set geometric size.
        #

        for node, unique_level in node_to_unique_level.items():
            geometric_size = math.pow(10, node_to_unique_level[node]) - 1
            node.data = GTreeData(geometric_size=geometric_size)

    def trim_to_values_needed(self, values: set[Any]) -> "GTree":
        """
        Create a new tree containing only nodes needed for the specified values.

        Produces a pruned copy of the original tree that contains only the paths
        and nodes necessary to represent the provided set of values. This is useful
        for optimizing tree size when only a subset of values are relevant.

        Parameters
        ----------
        values : Set[Any]
            Set of values that need to be represented in the trimmed tree

        Returns
        -------
        GTree
            A new GTree instance containing only the nodes needed to represent
            the specified values

        Notes
        -----
        The method performs a breadth-first search of the tree and removes subtrees
        that don't contain any of the specified values.
        """
        gtree = GTree(tree=self)
        queue = deque([gtree.root] if gtree.root is not None else [])
        while queue:
            nid = queue.popleft()
            if all(
                gtree.get_value(node) not in values
                for cnid in gtree.expand_tree(
                    nid, mode=Tree.DEPTH
                )  # go dfs because values are likely leaves
                if (node := gtree.get_node(cnid)) is not None
            ):
                # we don't need this node nor its successors
                gtree.remove_node(nid)
            else:
                # we need something in this path
                queue.extend(
                    [child.identifier for child in gtree.children(nid)]
                )  # overall search in this function is bfs
        return gtree

    def to_config_json(self) -> dict[str, Any]:
        """
        Convert the tree to a JSON serializable dictionary.

        Creates a JSON-compatible dictionary representation of the GTree, including
        its structure, node values, geometric sizes, and internal mappings. This can
        be used to persist the tree or transmit it across services.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the tree suitable for JSON serialization

        Notes
        -----
        This method also updates all internal mappings to ensure the serialized
        tree includes the most up-to-date information.
        """
        # update internal maps before serializing, for efficient use later
        self.update_highest_node_with_value_if()
        self.update_lowest_node_with_descendant_leaves_if()
        self.update_descendant_leaf_values_if()
        # construct and return json object
        json_obj: dict[str, Any] = {}
        json_obj["value_to_highest_node_nid"] = self.value_to_highest_node_nid
        json_obj["nid_to_descendant_leaf_values"] = {
            k: list(v) for k, v in self.nid_to_descendant_leaf_values.items()
        }
        json_obj["value_to_leaf_node_nid"] = self.value_to_leaf_node_nid
        json_obj["root_nid"] = self.root
        json_obj["nodes"] = {}
        for node in self.all_nodes_itr():
            parent = self.parent(node.identifier)
            children_ids = [child.identifier for child in self.children(node.identifier)]
            json_obj["nodes"][node.identifier] = [
                self.get_value(node),
                parent.identifier if parent is not None else None,
                self.get_geometric_size(node),
                children_ids,
            ]
        return json_obj

    def from_config_json(self, json_obj: dict[str, Any]) -> None:
        """
        Load the tree structure from a JSON dictionary representation.

        Populates this GTree instance using the structure and data from a dictionary
        that was previously generated by to_config_json(). This enables tree
        deserialization from saved configurations.

        Parameters
        ----------
        json_obj : Dict[str, Any]
            Dictionary containing the GTree representation

        Notes
        -----
        This method completely replaces the current tree structure with the one
        defined in the input dictionary. It also loads all precomputed mappings
        for efficiency.
        """
        queue = deque([json_obj["root_nid"]] if json_obj["nodes"] else [])
        while queue:
            nid = queue.pop()
            node_json_obj = json_obj["nodes"].pop(nid)
            value, parent_nid, geometric_size, children = node_json_obj
            self.create_node(
                value,
                identifier=nid,
                parent=self.get_node(parent_nid) if parent_nid is not None else None,
                geometric_size=geometric_size,
            )
            queue.extend(children)
        assert not json_obj["nodes"]
        self.value_to_highest_node_nid = dict(json_obj["value_to_highest_node_nid"])
        self.nid_to_descendant_leaf_values = {
            # json doesn't distinguish between tuples and lists
            # we store these internally as tuples (more memory efficient and immutable)
            nid: tuple(descendant_leaf_values)
            for nid, descendant_leaf_values in json_obj["nid_to_descendant_leaf_values"].items()
        }
        self.value_to_leaf_node_nid = dict(json_obj["value_to_leaf_node_nid"])

    def __eq__(self, other: Any) -> bool:
        """
        Two gtrees are equal if they have the same node values + geometric sizes
        in the same hierarchy.
        """
        if not isinstance(other, GTree):
            return NotImplemented
        if self.root is None and other.root is None:
            # both empty gtrees
            return True
        if (self.root is None) != (other.root is None):
            # one is empty and the other is not
            return False
        self_root_node = self.get_node(self.root)
        other_root_node = other.get_node(other.root)
        assert self_root_node is not None and other_root_node is not None
        return self.__eq__recursive(other, self_root_node, other_root_node)

    def __eq__recursive(self, other: "GTree", self_node: Node, other_node: Node) -> bool:
        if self.get_value(self_node) != other.get_value(other_node):
            return False
        if np.isnan(self.get_geometric_size(self_node)) != np.isnan(
            other.get_geometric_size(other_node)
        ):
            return False
        if not np.isnan(self.get_geometric_size(self_node)) and self.get_geometric_size(
            self_node
        ) != other.get_geometric_size(other_node):
            return False
        self_children_to_examine = list(self.children(self_node.identifier))
        other_children_to_examine = list(other.children(other_node.identifier))
        if len(self_children_to_examine) != len(other_children_to_examine):
            return False
        while self_children_to_examine:
            self_child = self_children_to_examine.pop()
            other_child = self.__eq__recursive__match(other, self_child, other_children_to_examine)
            if other_child is None:
                return False
            other_children_to_examine.remove(other_child)
        return True

    def __eq__recursive__match(
        self, other: "GTree", self_child: Node, other_children_to_examine: list[Node]
    ) -> Optional[Node]:
        for other_child in other_children_to_examine:
            if self.__eq__recursive(other, self_child, other_child):
                return other_child
        return None


class ReadOnlyGTree(GTree):
    """
    A read-only version of the generalization tree.

    This class extends GTree but prevents modifications to the tree structure
    after initialization. It raises assertions if any methods that would modify
    the tree are called after the tree is locked.

    Attributes
    ----------
    locked : bool
        Flag indicating whether the tree is in read-only mode

    Notes
    -----
    The tree starts in an unlocked state during initialization to allow for
    setup, then becomes locked when initialization is complete.

    TODO: Replace assertions with a specific error type for read-only violations
    """

    def __init__(
        self, tree: Optional["GTree"] = None, json_obj: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize a read-only generalization tree.

        Parameters
        ----------
        tree : GTree, optional
            An existing GTree to copy, by default None
        json_obj : Dict[str, Any], optional
            A JSON object representation of a GTree to load, by default None

        Notes
        -----
        The tree is temporarily unlocked during initialization to allow setup,
        then locked immediately after to prevent further modifications.
        """
        self.locked = False
        super().__init__(tree=tree, json_obj=json_obj)
        self.locked = True

    def create_node(  # type: ignore[override]
        self,
        value: Any,
        parent: Optional[Node] = None,
        geometric_size: float = NOT_DEFINED_NA,
        identifier: Optional[str] = None,
    ) -> Node:  # pylint: disable=arguments-differ
        assert not self.locked, "GTree is read-only"
        return super().create_node(
            value, parent=parent, geometric_size=geometric_size, identifier=identifier
        )

    def remove_node(self, identifier: str) -> None:  # type: ignore[override]
        assert not self.locked, "GTree is read-only"
        super().remove_node(identifier)

    def remove_subtree(self, nid: str, identifier: Optional[str] = None) -> None:
        assert not self.locked, "GTree is read-only"
        super().remove_subtree(nid, identifier=identifier)

    def update_highest_node_with_value_if(self) -> bool:
        updated = super().update_highest_node_with_value_if()
        assert not updated or not self.locked, "GTree is read-only"
        return updated

    def update_descendant_leaf_values_if(self) -> bool:
        updated = super().update_descendant_leaf_values_if()
        assert not updated or not self.locked, "GTree is read-only"
        return updated

    def update_lowest_node_with_descendant_leaves_if(self) -> bool:
        updated = super().update_lowest_node_with_descendant_leaves_if()
        assert not updated or not self.locked, "GTree is read-only"
        return updated

    def add_default_geometric_sizes(self) -> None:
        assert not self.locked, "GTree is read-only"
        return super().add_default_geometric_sizes()

    def from_config_json(self, json_obj: dict[str, Any]) -> None:
        assert not self.locked, "GTree is read-only"
        return super().from_config_json(json_obj)


def make_flat_default_gtree(uniq_values: set[Any]) -> GTree:
    """
    Create a simple two-level generalization tree with default geometric sizes.

    Constructs a flat generalization tree with "*" as the root node and all unique
    values from the input set as direct children (leaf nodes). Geometric sizes are
    automatically calculated using the default algorithm.

    Parameters
    ----------
    uniq_values : Set[Any]
        Set of unique values to include as leaves in the tree

    Returns
    -------
    GTree
        A new generalization tree with a flat structure
    """
    gtree = GTree()
    root = gtree.create_node("*")  # the root should always be '*'
    for value in uniq_values:
        gtree.create_node(value, parent=root)
    gtree.add_default_geometric_sizes()
    # update internal maps before serializing, for efficient use later
    gtree.update_highest_node_with_value_if()
    gtree.update_lowest_node_with_descendant_leaves_if()
    gtree.update_descendant_leaf_values_if()
    return gtree


def load_from_config_file(filename: str) -> GTree:
    """
    Load a generalization tree from a JSON configuration file.

    Parameters
    ----------
    filename : str
        Path to the JSON configuration file that was created by generate_config_file

    Returns
    -------
    GTree
        A generalization tree reconstructed from the configuration file

    Notes
    -----
    The configuration file should have been created using the generate_config_file
    function to ensure proper format.
    """
    with open(filename) as config_file:
        json_obj = json.load(config_file)
    return GTree(json_obj=json_obj)


def generate_config_file(gtree: GTree, filename: Optional[str] = None) -> str:
    """
    Serialize a generalization tree to a JSON configuration file.

    Parameters
    ----------
    gtree : GTree
        The generalization tree to serialize
    filename : str, optional
        Path where the configuration should be written. If None, a temporary
        file will be created, by default None

    Returns
    -------
    str
        The path to the written configuration file

    Notes
    -----
    The generated file can be loaded back using the load_from_config_file function.
    If no filename is provided, a temporary file with a .json extension will be created.
    """
    if filename is None:
        _, filename = tempfile.mkstemp(suffix=".json")
    with open(filename, "w") as config_file:
        json.dump(gtree.to_config_json(), config_file)
    return filename
