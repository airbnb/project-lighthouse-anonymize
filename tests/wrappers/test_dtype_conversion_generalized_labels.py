"""
Tests for convert_object_to_categorical handling of generalized gtree labels.

Anonymization generalizes object QID values to gtree node labels. Intermediate
node labels (e.g., "north_america") are not in the original categories, and
pd.Categorical silently converts out-of-category values to NaN, corrupting
generalized data into missing data.
"""

import pandas as pd

from project_lighthouse_anonymize.wrappers.dtype_conversion import (
    convert_categorical_to_object,
    convert_object_to_categorical,
)


class TestGtreeGeneralizedLabels:
    """Tests that generalized gtree node labels survive conversions"""

    def test_intermediate_gtree_label_preserved(self):
        """Intermediate gtree node labels must not be nulled out"""
        anon_df = pd.DataFrame({"col": ["a", "north_america", "*", "b"]})
        result = convert_object_to_categorical(anon_df, "col", pd.Index(["a", "b"]))
        assert result["col"].tolist() == ["a", "north_america", "*", "b"]
        assert not result["col"].isna().any()

    def test_roundtrip_with_generalization(self):
        """Full categorical -> object -> generalize -> categorical round trip"""
        input_df = pd.DataFrame(
            {"col": pd.Categorical(["us", "ca", "fr", "de"], categories=["us", "ca", "fr", "de"])}
        )
        object_df, metadata = convert_categorical_to_object(input_df, "col")
        # simulate anonymization generalizing to intermediate gtree node labels
        object_df["col"] = ["north_america", "north_america", "europe", "europe"]
        result = convert_object_to_categorical(object_df, "col", metadata)
        assert result["col"].tolist() == ["north_america", "north_america", "europe", "europe"]
        assert not result["col"].isna().any()


class TestNanPreservation:
    """Tests that NaN values remain NaN after conversion"""

    def test_nan_values_remain_nan(self):
        """Suppressed (NaN) cells remain NaN, not converted to a category"""
        anon_df = pd.DataFrame({"col": ["a", None, "*"]})
        result = convert_object_to_categorical(anon_df, "col", pd.Index(["a", "b"]))
        assert result["col"].tolist()[0] == "a"
        assert pd.isna(result["col"].iloc[1])
        assert result["col"].tolist()[2] == "*"
