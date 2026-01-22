"""
Tests for RILM and ILM metrics
"""

import logging

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize import gtrees
from project_lighthouse_anonymize.constants import EPSILON, NOT_DEFINED_NA
from project_lighthouse_anonymize.data_quality_metrics.rilm_ilm import (
    compute_average_information_loss_metric,
    compute_revised_information_loss_metric,
)

_ID_COL = "id_user"

_LOGGER = logging.getLogger(__name__)


def _assert_dicts_equal(actual_dict, expected_dict, atol=EPSILON):
    """
    Small helper function for comparing two dicts of floats.
    """
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{actual_dict.keys()} != {expected_dict.keys()}"
    )
    for k in actual_dict.keys():
        actual = actual_dict[k]
        expected = expected_dict[k]
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            err_msg=f"key = {k}: {actual} != {expected} (atol = {atol})",
        )


class TestAverageInformationLossMetric:
    """
    Tests for the average Information Loss Metric (ILM).
    """

    # pylint: disable=no-self-use

    def test_average_ilm_0(self):
        """
        Tests for no records
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
                "qid1_anon": [],
                "qid2_anon": [],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        actual = compute_average_information_loss_metric(input_df, input_qids, "_orig", "_anon")
        assert np.isnan(actual), f"{actual} is not {np.nan}"

    def test_average_ilm_1(self):
        """
        Tests where there is a single equivalence class of identical records.

        Following the spatial representation in [Byun et al 2006], this would
        look like two identical points on a 2-dimensional space with 0 width
        in each dimension.

        In this case, there is no information loss from generalization.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 1],
                "qid2_orig": [1, 1],
                "qid1_anon": [1, 1],
                "qid2_anon": [1, 1],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        ilm_equiv_class_1 = 0.0  # both qid1 and qid2 regions have 0 width so ILM is 0
        expected = ilm_equiv_class_1 / len(input_df)  # average per record
        actual = compute_average_information_loss_metric(input_df, input_qids, "_orig", "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_average_ilm_2(self):
        """
        Tests where there are two equivalence classes each of identical records.

        Following the spatial representation in [Byun et al 2006], this would
        look like two distinct sets of two identical points on a 2-dimensional
        space.

        In this case, there is no information loss from generalization.
        """
        values_1 = [1, 1, 2, 2]
        values_2 = [1, 1, 2, 2]
        input_df = pd.DataFrame(
            data={
                "qid1_orig": values_1,
                "qid2_orig": values_2,
                "qid1_anon": values_1,
                "qid2_anon": values_2,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        ilm_equiv_class_1 = 0.0  # both qid1 and qid2 regions have 0 width so ILM is 0
        ilm_equiv_class_2 = 0.0  # both qid1 and qid2 regions have 0 width so ILM is 0
        expected = (ilm_equiv_class_1 + ilm_equiv_class_2) / len(input_df)  # average per record
        actual = compute_average_information_loss_metric(input_df, input_qids, "_orig", "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_average_ilm_3(self):
        """
        Tests where there is a single equivalence class with two distinct records
        that, after generalization, are merged. These records vary in only one
        quasi-identifier.

        Following the spatial representation in [Byun et al 2006], this would
        look like two distinct points on a 2-dimensional space with different x
        coordinates but identical y coordinates.

        In this case, there is half of the maximum information loss possible in
        a 2-QID dataframe: 2.0/2 = 1.0.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 3],
                "qid2_orig": [1, 1],
                "qid1_anon": [2, 2],
                "qid2_anon": [1, 1],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        # qid1 region has non-zero width and the region covers the entire domain ~ 1.0
        # qid2 region has 0 width ~ 0.0
        ilm_equiv_class_1 = 2.0  # 2 records * (1.0 + 0.0) = 2.0
        expected = ilm_equiv_class_1 / len(input_df)  # average per record
        actual = compute_average_information_loss_metric(input_df, input_qids, "_orig", "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_average_ilm_4(self):
        """
        Tests where there is a single equivalence class with two distinct records
        that, after generalization, are merged. These records vary in both
        quasi-identifiers. Also tests that a qid with all nan values is properly ignored.

        Following the spatial representation in [Byun et al 2006], this would
        look like two distinct points on a 2-dimensional space with different x
        and y coordinates.

        In this case, there is the maximum information loss possible in
        an effectively 2-QID dataframe: 2.0
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 3],
                "qid2_orig": [1, 3],
                "qid3_orig": [np.nan, np.nan],
                "qid1_anon": [2, 2],
                "qid2_anon": [2, 2],
                "qid3_anon": [np.nan, np.nan],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2", "qid3"]

        # qid1 region has non-zero width and the region covers the entire domain ~ 1.0
        # qid2 region similarly ~ 1.0
        # qid3 is all nan so ignored
        ilm_equiv_class_1 = 4.0  # 2 records * (1.0 + 1.0) = 4.0
        expected = ilm_equiv_class_1 / len(input_df)  # average per record
        actual = compute_average_information_loss_metric(input_df, input_qids, "_orig", "_anon")
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )


class TestRevisedInformationLossMetric:
    """
    Tests for the Revised Information Loss Metric (RILM).

    The numerical test cases (test_rilm_Na) parallel the tests for ILM,
    while the categorical test cases (test_rilm_Nb) test categorical RILM-specific behavior.
    """

    # pylint: disable=no-self-use

    @pytest.fixture(autouse=True)
    def _setup(self):
        """
        Setup test fixture to create GTrees for categorical tests
        """
        # Create a simple generalization tree for 'qid1_cat'
        qid1_tree = gtrees.GTree()
        qid1_root = qid1_tree.create_node("*")
        qid1_group1 = qid1_tree.create_node("group_ab", qid1_root)
        qid1_tree.create_node("a", qid1_group1)
        qid1_tree.create_node("b", qid1_group1)
        qid1_group2 = qid1_tree.create_node("group_cd", qid1_root)
        qid1_tree.create_node("c", qid1_group2)
        qid1_tree.create_node("d", qid1_group2)

        # Update GTree internal indices and add default geometric sizes
        qid1_tree.update_highest_node_with_value_if()
        qid1_tree.update_descendant_leaf_values_if()
        qid1_tree.update_lowest_node_with_descendant_leaves_if()
        qid1_tree.add_default_geometric_sizes()

        # Create a simple generalization tree for 'qid2_cat'
        qid2_tree = gtrees.GTree()
        qid2_root = qid2_tree.create_node("*", geometric_size=4.0)
        qid2_group1 = qid2_tree.create_node("group_xy", qid2_root, geometric_size=3.0)
        qid2_tree.create_node("x", qid2_group1, geometric_size=0.0)
        qid2_tree.create_node("y", qid2_group1, geometric_size=0.0)
        qid2_group2 = qid2_tree.create_node("group_zw", qid2_root, geometric_size=1.0)
        qid2_tree.create_node("z", qid2_group2, geometric_size=0.0)
        qid2_tree.create_node("w", qid2_group2, geometric_size=0.0)

        # Update GTree internal indices
        qid2_tree.update_highest_node_with_value_if()
        qid2_tree.update_descendant_leaf_values_if()
        qid2_tree.update_lowest_node_with_descendant_leaves_if()

        self.qid_to_gtree = {"qid1_cat": qid1_tree, "qid2_cat": qid2_tree}

    def test_rilm_0(self):
        """
        Tests for no records
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [],
                "qid2_orig": [],
                "qid1_anon": [],
                "qid2_anon": [],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        # Overall RILM score
        expected = {"qid1": np.nan, "qid2": np.nan}
        actual, _ = compute_revised_information_loss_metric(
            input_df, input_qids, {}, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected)

    def test_rilm_1a(self):
        """
        Tests where there is a single equivalence class of identical records with numerical QIDs.

        Following the spatial representation in [Byun et al 2006], this would
        look like two identical points on a 2-dimensional space with 0 width
        in each dimension.

        In this case, there is no information loss from generalization.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 1],
                "qid2_orig": [1, 1],
                "qid1_anon": [1, 1],
                "qid2_anon": [1, 1],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        # Overall RILM score
        expected = {"qid1": 1.0, "qid2": 1.0}
        actual, _ = compute_revised_information_loss_metric(
            input_df, input_qids, {}, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected)

    def test_rilm_1b(self):
        """
        Tests where there is a single equivalence class of identical records with categorical QIDs.

        In this case, there is no information loss from generalization since the original and
        anonymized values are identical.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "a"],
                "qid2_cat_orig": ["x", "x"],
                "qid1_cat_anon": ["a", "a"],
                "qid2_cat_anon": ["x", "x"],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        # When no generalization occurs (leaf nodes), RILM = 1.0 - (0/root_size) = 1.0 (perfect information preservation)
        # For qid1_cat: All records use leaf node values (a), so geometric_size = 0, giving RILM = 1.0
        # For qid2_cat: All records use leaf node values (x), so geometric_size = 0, giving RILM = 1.0
        expected = {"qid1_cat": 1.0, "qid2_cat": 1.0}
        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected, atol=0.001)

    @pytest.mark.parametrize("has_nan_values", [True, False])
    def test_rilm_2a(self, has_nan_values):
        """
        Tests where there are two equivalence classes each of identical records with numerical QIDs.
        Also tests that nan values are properly handled when defining equivalence classes.

        Following the spatial representation in [Byun et al 2006], this would
        look like two distinct sets of two identical points on a 2-dimensional
        space.

        In this case, there is no information loss from generalization.
        """
        values_1 = [1, 1, 2, 2] if not has_nan_values else [1, 1, np.nan, np.nan]
        values_2 = [1, 1, 2, 2]
        input_df = pd.DataFrame(
            data={
                "qid1_orig": values_1,
                "qid2_orig": values_2,
                "qid1_anon": values_1,
                "qid2_anon": values_2,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        # Overall RILM score
        expected = {"qid1": 1.0, "qid2": 1.0}
        actual, _ = compute_revised_information_loss_metric(
            input_df, input_qids, {}, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected)

    @pytest.mark.parametrize("has_nan_values", [True, False])
    def test_rilm_2b(self, has_nan_values):
        """
        Tests where there are two equivalence classes each of identical records with categorical QIDs.
        Also tests that nan values are properly handled when defining equivalence classes.

        In this case, there is no information loss from generalization since for each equivalence class,
        the original and anonymized values are identical.
        """
        values_1 = ["a", "a", "c", "c"] if not has_nan_values else ["a", "a", np.nan, np.nan]
        values_2 = ["x", "x", "z", "z"]
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": values_1,
                "qid2_cat_orig": values_2,
                "qid1_cat_anon": values_1,
                "qid2_cat_anon": values_2,
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        # When no generalization occurs (leaf nodes), RILM = 1.0 - (0/root_size) = 1.0 (perfect information preservation)
        # For qid1_cat: All records use leaf node values (a, c or NaN), so geometric_size = 0, giving RILM = 1.0
        # For qid2_cat: All records use leaf node values (x, z), so geometric_size = 0, giving RILM = 1.0
        expected = {"qid1_cat": 1.0, "qid2_cat": 1.0}
        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected, atol=0.001)

    @pytest.mark.parametrize("has_nan_values", [True, False])
    def test_rilm_3a(self, has_nan_values):
        """
        Tests where there is a single equivalence class with two distinct records
        that, after generalization, are merged. These records vary in only one
        numerical quasi-identifier.

        Also tests that a qid with all nan values is ignored.

        Following the spatial representation in [Byun et al 2006], this would
        look like two distinct points on a 2-dimensional space with different x
        coordinates but identical y coordinates.

        In this case, there is no information loss in qid1,
        but maximum information loss in qid2. qid3 is nan so has no rilm.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 3],
                "qid2_orig": [1, 1],
                "qid3_orig": [np.nan, np.nan],
                "qid1_anon": [2, 2],
                "qid2_anon": [1, 1],
                "qid3_anon": [np.nan, np.nan],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2", "qid3"]

        # Overall RILM score
        expected = {"qid1": 0.0, "qid2": 1.0}
        actual, _ = compute_revised_information_loss_metric(
            input_df, input_qids, {}, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected)

    def test_rilm_3b(self):
        """
        Tests where there is a single equivalence class with two distinct records
        that, after generalization, are merged. These records vary in only one
        categorical quasi-identifier.

        Note: This test focuses solely on qid1_cat and qid2_cat. Unlike test_rilm_3a, we don't include
        qid3_cat in this test since without explicitly setting dtype="object", columns with all NaN values
        would be treated as numeric rather than categorical QIDs.

        In this case, there is partial information loss in qid1_cat since the anonymized value
        is the parent node in the generalization hierarchy, but no information loss in qid2_cat
        since the values remain unchanged.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "b"],
                "qid2_cat_orig": ["x", "x"],
                "qid1_cat_anon": ["group_ab", "group_ab"],
                "qid2_cat_anon": ["x", "x"],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        # For qid1_cat: group_ab generalization (one level up from leaves) gives:
        # RILM = 1.0 - (geometric_size_of_node/geometric_size_of_root) = 1.0 - (9/99) = 0.909
        # For qid2_cat: no generalization (leaf node x with geometric_size=0) gives RILM = 1.0 (perfect preservation)
        expected = {"qid1_cat": 0.909, "qid2_cat": 1.0}
        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected, atol=0.001)

    def test_rilm_4a(self):
        """
        Tests where there is a single equivalence class with two distinct records
        that, after generalization, are merged. These records vary in both
        numerical quasi-identifiers.

        Following the spatial representation in [Byun et al 2006], this would
        look like two distinct points on a 2-dimensional space with different x
        and y coordinates.

        In this case, there is the maximum information loss in both
        qid1 and qid2.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_orig": [1, 3],
                "qid2_orig": [1, 3],
                "qid1_anon": [2, 2],
                "qid2_anon": [2, 2],
            }
        )
        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1", "qid2"]

        # Overall RILM score
        expected = {"qid1": 0.0, "qid2": 0.0}
        actual, _ = compute_revised_information_loss_metric(
            input_df, input_qids, {}, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected)

    def test_rilm_4b(self):
        """
        Tests where there is a single equivalence class with two distinct records
        that, after generalization, are merged. These records vary in both
        categorical quasi-identifiers.

        In this case, there is substantial information loss in both categorical QIDs
        since both are generalized to higher-level nodes in the hierarchy.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "c"],
                "qid2_cat_orig": ["x", "z"],
                "qid1_cat_anon": ["*", "*"],
                "qid2_cat_anon": ["*", "*"],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        # Generalization to root node "*" gives maximum information loss:
        # RILM = 1.0 - (geometric_size_of_node/geometric_size_of_root) = 1.0 - (root_size/root_size) = 1.0 - 1.0 = 0.0
        # For qid1_cat: All records generalized to root node "*" gives RILM = 0.0
        # For qid2_cat: All records generalized to root node "*" gives RILM = 0.0
        expected = {"qid1_cat": 0.0, "qid2_cat": 0.0}
        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )
        _assert_dicts_equal(actual, expected, atol=0.001)

    def test_rilm_5b(self):
        """
        Tests a more complex scenario with mixed levels of generalization across multiple
        categorical QIDs.

        This test creates multiple equivalence classes with different generalization levels
        to test the weighted scoring algorithm for categorical QIDs. The different RILM scores
        for qid1_cat and qid2_cat are due to different geometric size configurations:
        - qid1_cat uses default geometric sizes (10^i - 1, where i is the height)
        - qid2_cat uses custom geometric sizes explicitly set in the fixture
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "b", "c", "d", "a", "b"],
                "qid2_cat_orig": ["x", "y", "z", "w", "x", "y"],
                "qid1_cat_anon": [
                    "group_ab",
                    "group_ab",
                    "group_cd",
                    "group_cd",
                    "a",
                    "b",
                ],
                "qid2_cat_anon": [
                    "group_xy",
                    "group_xy",
                    "group_zw",
                    "group_zw",
                    "x",
                    "y",
                ],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )

        # Expected values calculated using the RILM formula: 1.0 - (geometric_size_of_node/geometric_size_of_root)
        # For qid1_cat: Weighted average across 4 records with group-level generalization (0.909) and 2 with leaf nodes (1.0)
        #               = (4*0.909 + 2*1.0)/6 = 0.939
        # For qid2_cat: Weighted average using custom geometric sizes:
        #               = (4*(1.0-(3.0/4.0)) + 2*(1.0-(0.0/4.0)))/6 = (4*0.25 + 2*1.0)/6 = 0.667
        expected = {"qid1_cat": 0.939, "qid2_cat": 0.667}
        _assert_dicts_equal(actual, expected, atol=0.001)

    def test_rilm_6b(self):
        """
        Tests categorical QIDs with different geometric sizes at various hierarchy levels.

        This test verifies that the RILM calculation correctly accounts for geometric sizes
        when calculating information loss. The RILM formula is:
        RILM = 1.0 - (geometric_size_of_node / geometric_size_of_root)

        The test shows different RILM scores between:
        - qid1_cat (using default geometric sizes: 10^i - 1, where i is the tree level)
        - qid2_cat (using explicitly defined geometric sizes: root=4.0, group_xy=3.0, group_zw=1.0)
        for the same generalization pattern.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "b", "a", "b"],
                "qid2_cat_orig": ["x", "y", "x", "y"],
                "qid1_cat_anon": ["group_ab", "group_ab", "a", "b"],
                "qid2_cat_anon": ["group_xy", "group_xy", "x", "y"],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )

        # Expected values calculated using the RILM formula: 1.0 - (geometric_size_of_node/geometric_size_of_root)
        # For qid1_cat: Weighted average with default geometric sizes:
        #               = (2*(1.0-(9/99)) + 2*1.0)/4 = (2*0.909 + 2*1.0)/4 = 0.955
        # For qid2_cat: Weighted average with custom geometric sizes:
        #               = (2*(1.0-(3.0/4.0)) + 2*1.0)/4 = (2*0.25 + 2*1.0)/4 = 0.625
        expected = {"qid1_cat": 0.955, "qid2_cat": 0.625}
        _assert_dicts_equal(actual, expected, atol=0.001)

    def test_rilm_7b(self):
        """
        Tests categorical QIDs with NaN values and partial generalization.

        This test verifies that the RILM calculation correctly handles NaN values and
        properly weighs different levels of generalization for categorical QIDs.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "b", "c", np.nan, "a"],
                "qid2_cat_orig": ["x", "y", "z", "w", np.nan],
                "qid1_cat_anon": ["group_ab", "group_ab", "*", np.nan, "a"],
                "qid2_cat_anon": ["x", "group_xy", "group_zw", np.nan, np.nan],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )

        # Expected values calculated using the RILM formula: 1.0 - (geometric_size_of_node/geometric_size_of_root)
        # For qid1_cat: Complex weighted average across different generalization levels:
        #               - 2 records with group_ab level (RILM ≈ 0.909)
        #               - 1 record with root node "*" (RILM = 0.0)
        #               - 1 NaN record (ignored)
        #               - 1 record with leaf node (RILM = 1.0)
        #               = (2*0.909 + 1*0.0 + 1*1.0)/4 = 0.705
        # For qid2_cat: Weighted average with custom geometric sizes:
        #               - 1 record with leaf node x (RILM = 1.0)
        #               - 1 record with group_xy node (RILM = 0.25)
        #               - 1 record with group_zw node (RILM = 0.75)
        #               - 2 NaN records (ignored)
        #               = (1*1.0 + 1*0.25 + 1*0.75)/3 = 0.667
        expected = {"qid1_cat": 0.705, "qid2_cat": 0.667}
        _assert_dicts_equal(actual, expected, atol=0.001)

    def test_rilm_8b(self):
        """
        Tests when a QID is not in the qid_to_gtree mapping.

        This test verifies that when a categorical QID is not found in the qid_to_gtree mapping,
        its RILM score is set to NOT_DEFINED_NA.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "b", "c", "d"],
                "qid3_cat_orig": ["p", "q", "r", "s"],
                "qid1_cat_anon": ["group_ab", "group_ab", "c", "d"],
                "qid3_cat_anon": ["p_q", "p_q", "r_s", "r_s"],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid3_cat"]

        _, actual = compute_revised_information_loss_metric(
            input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
        )

        # Expected values calculated using the RILM formula: 1.0 - (geometric_size_of_node/geometric_size_of_root)
        # For qid1_cat: Weighted average with default geometric sizes:
        #               = (2*(1.0-(9/99)) + 2*1.0)/4 = (2*0.909 + 2*1.0)/4 = 0.955
        # For qid3_cat: Not present in qid_to_gtree mapping so set to NOT_DEFINED_NA
        expected = {"qid1_cat": 0.955, "qid3_cat": NOT_DEFINED_NA}
        _assert_dicts_equal(actual, expected, atol=0.001)

    def test_rilm_9b(self):
        """
        Tests when values in anonymized data are not found in the GTree.

        This test verifies the behavior when encountering QID values that don't exist
        in the corresponding GTree.
        """
        input_df = pd.DataFrame(
            data={
                "qid1_cat_orig": ["a", "b", "c", "unknown"],
                "qid2_cat_orig": ["x", "y", "z", "unknown"],
                "qid1_cat_anon": ["a", "b", "unknown_group", "unknown"],
                "qid2_cat_anon": ["x", "y", "unknown_group", "unknown"],
            }
        )

        input_df[_ID_COL] = pd.Series(list(range(len(input_df))), dtype=np.dtype("int64"))
        input_qids = ["qid1_cat", "qid2_cat"]

        # This test should raise an exception because "unknown_group" is not in the GTree
        with pytest.raises(Exception):
            _, actual = compute_revised_information_loss_metric(
                input_df, input_qids, self.qid_to_gtree, "_orig", "_anon"
            )
