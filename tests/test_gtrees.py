"""
Tests for gtrees
"""

import pickle

import numpy as np

from project_lighthouse_anonymize.gtrees import (
    GTree,
    ReadOnlyGTree,
    generate_config_file,
    load_from_config_file,
    make_flat_default_gtree,
)


class TestGTree:
    """
    Tests for generalization trees.
    """

    # pylint: disable=unused-variable
    # TODO consider adding a setup for gtree creation

    def test_gtree_equals(self):
        """Tests that GTree == works correctly."""
        gtree = GTree()
        gtree.create_node("*")

        gtree_2 = GTree()
        assert gtree != gtree_2

        gtree_2.create_node("*")
        assert gtree == gtree_2

        gtree_3 = GTree()
        gtree_3.create_node("*", geometric_size=0.1)
        assert gtree != gtree_3

        gtree_4 = GTree()
        gtree_4.create_node("*", geometric_size=0.12)
        assert gtree_3 != gtree_4

        gtree_5 = GTree()
        gtree_5.create_node("*", geometric_size=0.10)
        assert gtree_3 == gtree_5

        gtree_6 = GTree()
        root_6 = gtree_6.create_node("*")
        a_6 = gtree_6.create_node("a", parent=root_6)
        gtree_6.create_node("b", parent=root_6)
        c_6 = gtree_6.create_node("c", parent=a_6)

        gtree_7 = GTree()
        root_7 = gtree_7.create_node("*")
        a_7 = gtree_7.create_node("a", parent=root_7)
        gtree_7.create_node("c", parent=a_7)
        assert gtree_6 != gtree_7

        gtree_7.create_node("b", parent=root_7)
        assert gtree_6 == gtree_7

        gtree_6.create_node("d", parent=c_6)
        assert gtree_6 != gtree_7

    def test_gtree_pickle(self):
        """Tests that GTree is pickleable."""
        gtree = GTree()
        gtree.create_node("*")

        gtree = pickle.loads(pickle.dumps(gtree))

        actual = gtree.get_value(gtree.get_node(gtree.root))
        expected = "*"
        assert actual == expected, f"{actual} != {expected}"

    def test_gtree_get_value(self):
        """Tests get_value."""
        gtree = GTree()
        node = gtree.create_node("*")

        actual = gtree.get_value(node)
        expected = "*"
        assert actual == expected, f"{actual} != {expected}"

    def test_gtree_get_value_read_only(self):
        """Tests get_value on a read-only gtree."""
        gtree = GTree()
        node = gtree.create_node("*")

        read_only_gtree = ReadOnlyGTree(gtree)
        actual = read_only_gtree.get_value(node)
        expected = "*"
        assert actual == expected, f"{actual} != {expected}"

    def test_gtree_depth_1(self):
        """Tests depth of tree with 1 level."""
        gtree = GTree()
        gtree.create_node("*")

        actual = gtree.depth()
        expected = 1 - 1  # 0-indexed
        assert actual == expected, f"{actual} != {expected}"

    def test_gtree_depth_2(self):
        """Tests depth of tree with 2 levels."""
        gtree = GTree()
        root = gtree.create_node("*")
        gtree.create_node("90210", parent=root)

        actual = gtree.depth()
        expected = 2 - 1  # 0-indexed
        assert actual == expected, f"{actual} != {expected}"

    def test_gtree_pprint(self):
        """Tests that pprint function doesn't throw and returns a string."""
        gtree = GTree()
        actual_val = gtree.pprint()
        assert isinstance(actual_val, str), f"{type(actual_val)} not instance of {str}"

        root = gtree.create_node("*")
        actual_val = gtree.pprint()
        assert isinstance(actual_val, str), f"{type(actual_val)} not instance of {str}"

        gtree.create_node("90210", parent=root)
        actual_val = gtree.pprint()
        assert isinstance(actual_val, str), f"{type(actual_val)} not instance of {str}"

    def test_gtree_pprint_geometric_sizes(self):
        """Tests that pprint_geometric_sizes function doesn't throw and returns a string."""
        gtree = GTree()
        actual_val = gtree.pprint_geometric_sizes()
        assert isinstance(actual_val, str), f"{type(actual_val)} not instance of {str}"

        root = gtree.create_node("*")
        actual_val = gtree.pprint_geometric_sizes()
        assert isinstance(actual_val, str), f"{type(actual_val)} not instance of {str}"

        gtree.create_node("90210", parent=root, geometric_size=1)
        actual_val = gtree.pprint_geometric_sizes()
        assert isinstance(actual_val, str), f"{type(actual_val)} not instance of {str}"

    def test_get_highest_node_with_value(self):
        """Tests get_highest_node_with_value."""
        gtree = GTree()
        root = gtree.create_node("*")
        x1 = gtree.create_node("x", parent=root)
        y = gtree.create_node("y", parent=root)
        gtree.create_node("x", parent=y)

        actual = gtree.get_highest_node_with_value("x").identifier
        expected = x1.identifier
        assert actual == expected, f"{actual} != {expected}"

    def test_get_lowest_node_with_descendant_leaves(self):
        """Tests get_lowest_node_with_descendant_leaves"""
        gtree = GTree()
        root = gtree.create_node("*")

        foobar_parent = gtree.create_node("foobar_parent", parent=root)
        foobar = gtree.create_node("foobar", parent=foobar_parent)
        gtree.create_node("test", parent=root)

        foo = gtree.create_node("foo", parent=foobar)
        gtree.create_node("bar", parent=foobar)
        assert foo == gtree.get_lowest_node_with_descendant_leaves({"foo"})
        assert foo == gtree.get_lowest_node_with_descendant_leaves({"foo"}, start_value="foobar")
        assert foo == gtree.get_lowest_node_with_descendant_leaves(
            {"foo"}, start_value="foobar_parent"
        )
        assert foo == gtree.get_lowest_node_with_descendant_leaves({"foo"}, start_value="*")
        assert foobar == gtree.get_lowest_node_with_descendant_leaves({"foo", "bar"})
        assert foobar == gtree.get_lowest_node_with_descendant_leaves(
            {"foo", "bar"}, start_value="foobar"
        )
        assert foobar == gtree.get_lowest_node_with_descendant_leaves(
            {"foo", "bar"}, start_value="foobar_parent"
        )
        assert foobar == gtree.get_lowest_node_with_descendant_leaves(
            {"foo", "bar"}, start_value="*"
        )
        assert root == gtree.get_lowest_node_with_descendant_leaves({"foo", "bar", "test"})
        assert root == gtree.get_lowest_node_with_descendant_leaves(
            {"foo", "bar", "test"}, start_value="*"
        )

    def test_gtree_add_default_geometric_sizes(self):
        """Tests add_default_geometric_size and get_geometric_size."""
        gtree = GTree()
        root = gtree.create_node("*")

        foobar = gtree.create_node("foobar", parent=root)
        test = gtree.create_node("test", parent=root)

        foo = gtree.create_node("foo", parent=foobar)
        bar = gtree.create_node("bar", parent=foobar)

        geometric_sizes = [gtree.get_geometric_size(node) for node in gtree.all_nodes_itr()]
        assert all(np.isnan(geometric_size) for geometric_size in geometric_sizes), (
            f"{geometric_sizes} not all nan"
        )

        gtree.add_default_geometric_sizes()
        geometric_sizes = [gtree.get_geometric_size(node) for node in gtree.all_nodes_itr()]
        assert all(not np.isnan(geometric_size) for geometric_size in geometric_sizes), (
            f"Some {geometric_sizes} are nan"
        )

        expected = 0
        for node in [foo, bar, test]:
            actual = gtree.get_geometric_size(node)
            assert actual == expected, f"{node} geometric_size {actual} != {expected}"

        expected = 9
        actual = gtree.get_geometric_size(foobar)
        assert actual == expected, f"{foobar} geometric_size {actual} != {expected}"

        expected = 99
        actual = gtree.get_geometric_size(root)
        assert actual == expected, f"{root} geometric_size {actual} != {expected}"

    def test_gtree_add_default_geometric_sizes_read_only(self):
        """Tests add_default_geometric_size on a read-only gtree"""
        gtree = GTree()
        root = gtree.create_node("*")

        foobar = gtree.create_node("foobar", parent=root)
        gtree.create_node("test", parent=root)

        gtree.create_node("foo", parent=foobar)
        gtree.create_node("bar", parent=foobar)

        read_only_gtree = ReadOnlyGTree(gtree)

        try:
            read_only_gtree.add_default_geometric_sizes()
        except AssertionError:
            pass
        else:
            assert False, "An AssertionError was not raised"

    def test_make_flat_default_gtree(self):
        """Tests make_flat_default_gtree"""
        gtree = make_flat_default_gtree({"foo", "bar"})
        node = gtree.get_highest_node_with_value("*")
        assert node is not None
        assert gtree.get_geometric_size(node) > 0.0
        node = gtree.get_highest_node_with_value("foo")
        assert node is not None
        assert gtree.get_geometric_size(node) == 0.0
        node = gtree.get_highest_node_with_value("bar")
        assert node is not None
        assert gtree.get_geometric_size(node) == 0.0

    def test_gtree_remove_node(self):
        """Tests gtree remove_node removes node and descendents while invalidating cache,
        unless given a read-only gtree"""
        gtree = GTree()
        root = gtree.create_node("*", identifier="*-nid")

        foobar_parent = gtree.create_node(
            "foobar_parent", identifier="foobar_parent-nid", parent=root
        )
        gtree.create_node("foobar", identifier="foobar-nid", parent=foobar_parent)
        gtree.create_node("test", identifier="test-nid", parent=root)

        gtree.update_highest_node_with_value_if()
        gtree.update_lowest_node_with_descendant_leaves_if()
        gtree.update_descendant_leaf_values_if()

        read_only_gtree = ReadOnlyGTree(gtree)

        try:
            read_only_gtree.remove_node("foobar_parent-nid")
        except AssertionError:
            pass
        else:
            assert False, "An AssertionError was not raised"

        assert gtree.get_node("foobar_parent-nid") is not None
        assert gtree.get_node("foobar-nid") is not None

        gtree.remove_node("foobar_parent-nid")

        assert gtree.get_node("foobar_parent-nid") is None
        assert gtree.get_node("foobar-nid") is None

        # TODO find a more robust method of checking that the cache is invalidated
        assert gtree.value_to_highest_node_nid == {}
        assert gtree.nid_to_descendant_leaf_values == {}
        assert gtree.value_to_leaf_node_nid == {}

    def test_gtree_remove_subtree(self):
        """Tests gtree remove_subtree removes node and descendents while invalidating cache,
        unless given a read-only gtree"""
        gtree = GTree()
        root = gtree.create_node("*", identifier="*-nid")

        foobar_parent = gtree.create_node(
            "foobar_parent", identifier="foobar_parent-nid", parent=root
        )
        gtree.create_node("foobar", identifier="foobar-nid", parent=foobar_parent)
        gtree.create_node("test", identifier="test-nid", parent=root)

        gtree.update_highest_node_with_value_if()
        gtree.update_lowest_node_with_descendant_leaves_if()
        gtree.update_descendant_leaf_values_if()

        read_only_gtree = ReadOnlyGTree(gtree)

        try:
            read_only_gtree.remove_subtree("foobar_parent-nid")
        except AssertionError:
            pass
        else:
            assert False, "An AssertionError was not raised"

        foobar_parent_node = gtree.get_node("foobar_parent-nid")
        assert foobar_parent_node is not None
        foobar_children = list(gtree.children("foobar_parent-nid"))
        assert len(foobar_children) == 1
        assert gtree.get_value(foobar_children[0]) == "foobar"

        gtree.remove_subtree("foobar_parent-nid")

        assert gtree.get_node("foobar_parent-nid") is None
        assert gtree.get_node("foobar-nid") is None

        assert gtree.value_to_highest_node_nid == {}
        assert gtree.nid_to_descendant_leaf_values == {}
        assert gtree.value_to_leaf_node_nid == {}


class TestGTreeConfigs:
    """
    Tests for generalization tree configs.
    """

    # pylint: disable=unused-variable

    def _assert_gtrees_equal(self, gtree, new_gtree):
        """Helper function to assert two gtrees are equal."""
        actual = new_gtree.size()
        expected = gtree.size()
        assert actual == expected, f"{actual} != {expected}"

        leaves = set(
            [
                (gtree.get_value(leaf), gtree.get_geometric_size(leaf), leaf.identifier)
                for leaf in gtree.leaves()
            ]
        )
        new_leaves = set(
            [
                (
                    new_gtree.get_value(leaf),
                    new_gtree.get_geometric_size(leaf),
                    leaf.identifier,
                )
                for leaf in new_gtree.leaves()
            ]
        )

        actual = len(leaves)
        expected = len(new_leaves)
        assert actual == expected, f"{actual} != {expected}"

        for leaf_tuple, new_leaf_tuple in zip(
            sorted(
                leaves, key=lambda leaf_tuple: (leaf_tuple[0], leaf_tuple[1])
            ),  # identifier may vary
            sorted(
                new_leaves, key=lambda leaf_tuple: (leaf_tuple[0], leaf_tuple[1])
            ),  # identifier may vary
        ):
            # identifier may vary
            actual = new_leaf_tuple[0]
            expected = leaf_tuple[0]
            assert actual == expected, f"{actual} != {expected}"

            actual = new_leaf_tuple[1]
            expected = leaf_tuple[1]
            assert all([np.isnan(actual), np.isnan(expected)]) or actual == expected, (
                f"{actual} != {expected}"
            )

            # walk to root should have same value and geometric_size
            # note that this assumes that the test gtree had no two leaves had the same value and geometric_size
            leaf_walk = list(gtree.rsearch(leaf_tuple[2]))
            new_leaf_walk = list(new_gtree.rsearch(new_leaf_tuple[2]))

            actual = len(leaf_walk)
            expected = len(new_leaf_walk)
            assert actual == expected, f"{actual} != {expected}"

            for nid, new_nid in zip(leaf_walk, new_leaf_walk):
                node, new_node = gtree.get_node(nid), new_gtree.get_node(new_nid)
                actual = new_gtree.get_value(new_node)
                expected = gtree.get_value(node)
                assert actual == expected, f"{actual} != {expected}"
                actual = new_gtree.get_geometric_size(new_node)
                expected = gtree.get_geometric_size(node)
                assert all([np.isnan(actual), np.isnan(expected)]) or actual == expected, (
                    f"{actual} != {expected}"
                )

    def test_generate_and_load_config_file(self):
        """Tests generate_config_file and load_from_config_file."""
        gtree = GTree()
        root = gtree.create_node("*")

        foobar = gtree.create_node("foobar", parent=root)
        gtree.create_node("test", parent=root, geometric_size=3)

        gtree.create_node("foo", parent=foobar)
        gtree.create_node("bar", parent=foobar, geometric_size=1)

        filename = generate_config_file(gtree)
        new_gtree = load_from_config_file(filename)

        self._assert_gtrees_equal(gtree, new_gtree)

    def test_trim_to_values_needed(self):
        """Tests trim_to_values_needed."""
        gtree = GTree()
        root = gtree.create_node("*")

        foobar = gtree.create_node("foobar", parent=root)
        gtree.create_node("test", parent=root, geometric_size=3)

        gtree.create_node("foo", parent=foobar)
        gtree.create_node("bar", parent=foobar, geometric_size=1)

        actual_gtree = gtree.trim_to_values_needed(set(["test"]))

        expected_gtree = GTree()
        expected_root = expected_gtree.create_node("*")

        expected_gtree.create_node("test", parent=expected_root, geometric_size=3)

        self._assert_gtrees_equal(expected_gtree, actual_gtree)
