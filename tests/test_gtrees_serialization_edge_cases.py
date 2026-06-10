"""
Tests for GTree serialization round trips and structural edge cases.

from_config_json must not destructively consume its input; JSON round trips
must preserve lookup semantics for non-string node values; duplicate leaf
values silently corrupt the value->leaf map (producing wrong lowest-common-
ancestor results and overlapping partitions downstream) and must be rejected;
and a root-only tree must support default geometric sizes.
"""

import os

import pytest

from project_lighthouse_anonymize.gtrees import (
    GTree,
    generate_config_file,
    load_from_config_file,
    make_flat_default_gtree,
)


class TestFromConfigJsonInputPreservation:
    """Tests that loading from a JSON object does not consume it"""

    def test_second_load_from_same_json_obj(self):
        """Loading twice from the same dict must produce identical trees"""
        gtree = GTree()
        root = gtree.create_node("*")
        gtree.create_node("a", parent=root)
        gtree.create_node("b", parent=root)
        json_obj = gtree.to_config_json()

        first_load = GTree(json_obj=json_obj)
        second_load = GTree(json_obj=json_obj)
        assert first_load.size() == 3
        assert second_load.size() == 3
        assert second_load.get_highest_node_with_value("a") is not None
        assert first_load == second_load


class TestRoundTripNonStringValues:
    """Tests that JSON round trips preserve lookups for non-string node values"""

    def test_integer_values_survive_config_file_round_trip(self):
        """Integer node values must still be found after a file round trip"""
        gtree = GTree()
        root = gtree.create_node("*")
        gtree.create_node(90210, parent=root)
        gtree.create_node(10001, parent=root)
        gtree.add_default_geometric_sizes()

        filename = generate_config_file(gtree)
        try:
            loaded = load_from_config_file(filename)
        finally:
            os.unlink(filename)

        node = loaded.get_highest_node_with_value(90210)
        assert node is not None
        assert loaded.get_value(node) == 90210
        lca = loaded.get_lowest_node_with_descendant_leaves({90210})
        assert lca is not None
        assert loaded.get_value(lca) == 90210


class TestDuplicateLeafValues:
    """Tests that duplicate leaf values are rejected instead of silently colliding"""

    def test_duplicate_leaf_values_raise(self):
        """Duplicate leaf values corrupt the value->leaf map and must raise

        With leaves "x" under both "US" and "EU", the last-one-wins map makes
        get_lowest_node_with_descendant_leaves({"x", "y"}) return the root
        instead of "US", over-generalizing and (worse) producing overlapping
        partitions in the Mondrian categorical split.
        """
        gtree = GTree()
        root = gtree.create_node("*")
        us = gtree.create_node("US", parent=root)
        eu = gtree.create_node("EU", parent=root)
        gtree.create_node("x", parent=us)
        gtree.create_node("y", parent=us)
        gtree.create_node("x", parent=eu)
        gtree.create_node("z", parent=eu)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            gtree.get_lowest_node_with_descendant_leaves({"x", "y"})

    def test_duplicate_non_leaf_values_still_allowed(self):
        """Duplicate values are fine when at most one of them is a leaf"""
        gtree = GTree()
        root = gtree.create_node("*")
        x1 = gtree.create_node("x", parent=root)
        gtree.create_node("x_child", parent=x1)
        y = gtree.create_node("y", parent=root)
        gtree.create_node("x", parent=y)
        node = gtree.get_lowest_node_with_descendant_leaves({"x_child"})
        assert node is not None
        assert gtree.get_value(node) == "x_child"


class TestRootOnlyTreeGeometricSizes:
    """Tests for default geometric sizes on degenerate trees"""

    def test_root_only_tree(self):
        """A root-only tree gets geometric size 0 instead of crashing"""
        gtree = GTree()
        root = gtree.create_node("*")
        gtree.add_default_geometric_sizes()
        assert gtree.get_geometric_size(root) == pytest.approx(0.0)

    def test_make_flat_default_gtree_empty_values(self):
        """make_flat_default_gtree with no values produces a root-only tree"""
        gtree = make_flat_default_gtree(set())
        assert gtree.size() == 1
        root_node = gtree.get_node(gtree.root)
        assert root_node is not None
        assert gtree.get_geometric_size(root_node) == pytest.approx(0.0)
