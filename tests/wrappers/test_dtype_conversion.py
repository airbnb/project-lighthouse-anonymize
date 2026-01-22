"""
Unit tests for dtype conversion utilities.

This module tests the conversion functions that transform unsupported pandas dtypes
to supported ones for anonymization and back again.
"""

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.constants import GTREE_ROOT_TAG
from project_lighthouse_anonymize.wrappers.dtype_conversion import (
    DTYPE_CONVERSION_MAP,
    convert_bool_to_float,
    convert_categorical_to_object,
    convert_datetime_to_float,
    convert_float_to_bool,
    convert_float_to_datetime,
    convert_object_to_categorical,
)


class TestBoolConversion:
    """Test boolean to float conversion and back."""

    def test_convert_bool_to_float_basic(self):
        """Test basic boolean to float conversion."""
        df = pd.DataFrame({"col": [True, False, True]})
        result_df, metadata = convert_bool_to_float(df, "col")

        assert result_df["col"].dtype == "float64"
        assert list(result_df["col"]) == [1.0, 0.0, 1.0]
        assert metadata is None

    def test_convert_bool_to_float_with_nan(self):
        """Test boolean to float conversion with NaN values."""
        df = pd.DataFrame({"col": pd.array([True, False, None], dtype="boolean")})
        result_df, metadata = convert_bool_to_float(df, "col")

        assert result_df["col"].dtype == "float64"
        assert result_df["col"].iloc[0] == 1.0
        assert result_df["col"].iloc[1] == 0.0
        assert pd.isna(result_df["col"].iloc[2])

    @pytest.mark.parametrize("rng_state", [None, 42])
    def test_convert_float_to_bool_basic(self, rng_state):
        """Test float to boolean conversion using weighted coin flip."""
        # Create a simple dataframe where we can predict the outcome
        anon_df = pd.DataFrame({"col": [0.8, 0.2, 0.9]})
        result_df = convert_float_to_bool(anon_df, "col", None, rng_state)

        assert result_df["col"].dtype == "boolean"
        # Values should be boolean or NA
        assert all(pd.isna(val) or isinstance(val, (bool, np.bool_)) for val in result_df["col"])

    def test_convert_float_to_bool_with_nan(self):
        """Test float to boolean conversion preserving NaN values."""
        anon_df = pd.DataFrame({"col": [0.8, np.nan, 0.2]})
        result_df = convert_float_to_bool(anon_df, "col", None)

        assert result_df["col"].dtype == "boolean"
        assert pd.isna(result_df["col"].iloc[1])

    def test_bool_roundtrip(self):
        """Test that bool -> float -> bool preserves boolean nature and is deterministic."""
        df = pd.DataFrame({"col": [True, False, True, False]})

        # Forward conversion
        float_df, metadata = convert_bool_to_float(df, "col")

        # Back conversion - run twice to test determinism
        result_df1 = convert_float_to_bool(float_df, "col", metadata)
        result_df2 = convert_float_to_bool(float_df, "col", metadata)

        assert result_df1["col"].dtype == "boolean"
        assert result_df2["col"].dtype == "boolean"
        # Should produce identical results (deterministic)
        pd.testing.assert_series_equal(result_df1["col"], result_df2["col"])


class TestDatetimeConversion:
    """Test datetime to float conversion and back."""

    def test_convert_datetime_to_float_basic(self):
        """Test basic datetime to float conversion."""
        df = pd.DataFrame({"col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])})
        result_df, metadata = convert_datetime_to_float(df, "col")

        assert result_df["col"].dtype == "float64"
        assert metadata is None
        # Should be nanoseconds since epoch
        assert all(val > 0 for val in result_df["col"])

    def test_convert_datetime_to_float_with_nat(self):
        """Test datetime to float conversion with NaT values."""
        df = pd.DataFrame({"col": pd.to_datetime(["2023-01-01", None, "2023-01-03"])})
        result_df, metadata = convert_datetime_to_float(df, "col")

        assert result_df["col"].dtype == "float64"
        assert result_df["col"].iloc[0] > 0
        assert pd.isna(result_df["col"].iloc[1])
        assert result_df["col"].iloc[2] > 0

    def test_convert_float_to_datetime_basic(self):
        """Test float to datetime conversion."""
        # Use known timestamp
        timestamp_ns = pd.Timestamp("2023-01-01").value  # nanoseconds
        anon_df = pd.DataFrame({"col": [float(timestamp_ns), float(timestamp_ns + 86400000000000)]})

        result_df = convert_float_to_datetime(anon_df, "col", None)

        assert pd.api.types.is_datetime64_any_dtype(result_df["col"])
        assert result_df["col"].iloc[0] == pd.Timestamp("2023-01-01")

    def test_convert_float_to_datetime_with_nan(self):
        """Test float to datetime conversion preserving NaN values."""
        timestamp_ns = pd.Timestamp("2023-01-01").value
        anon_df = pd.DataFrame({"col": [float(timestamp_ns), np.nan]})

        result_df = convert_float_to_datetime(anon_df, "col", None)

        assert pd.api.types.is_datetime64_any_dtype(result_df["col"])
        assert result_df["col"].iloc[0] == pd.Timestamp("2023-01-01")
        assert pd.isna(result_df["col"].iloc[1])

    def test_datetime_roundtrip(self):
        """Test that datetime -> float -> datetime preserves values."""
        original_dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        df = pd.DataFrame({"col": original_dates})

        # Forward conversion
        float_df, metadata = convert_datetime_to_float(df, "col")

        # Back conversion
        result_df = convert_float_to_datetime(float_df, "col", metadata)

        assert pd.api.types.is_datetime64_any_dtype(result_df["col"])
        pd.testing.assert_series_equal(result_df["col"], df["col"])


class TestCategoricalConversion:
    """Test categorical to object conversion and back."""

    def test_convert_categorical_to_object_basic(self):
        """Test basic categorical to object conversion."""
        df = pd.DataFrame({"col": pd.Categorical(["a", "b", "c", "a"], categories=["a", "b", "c"])})
        result_df, metadata = convert_categorical_to_object(df, "col")

        assert result_df["col"].dtype == "object"
        assert list(result_df["col"]) == ["a", "b", "c", "a"]
        assert list(metadata) == ["a", "b", "c"]

    def test_convert_categorical_to_object_with_nan(self):
        """Test categorical to object conversion with NaN values."""
        df = pd.DataFrame({"col": pd.Categorical(["a", None, "b"], categories=["a", "b"])})
        result_df, metadata = convert_categorical_to_object(df, "col")

        assert result_df["col"].dtype == "object"
        assert result_df["col"].iloc[0] == "a"
        assert pd.isna(result_df["col"].iloc[1])  # Should preserve NaN, not become 'nan' string
        assert result_df["col"].iloc[2] == "b"

    def test_categorical_nan_preservation_bug(self):
        """Test that NaN values are preserved as actual NaN, not 'nan' string."""
        df = pd.DataFrame({"col": pd.Categorical(["x", None, "y"], categories=["x", "y"])})
        result_df, metadata = convert_categorical_to_object(df, "col")

        # This test verifies the bug is fixed - NaN should stay NaN, not become string 'nan'
        assert pd.isna(result_df["col"].iloc[1])
        assert result_df["col"].iloc[1] != "nan"  # Should not be string 'nan'

    def test_convert_object_to_categorical_basic(self):
        """Test object to categorical conversion."""
        anon_df = pd.DataFrame({"col": ["a", "b", "a", "c"]})
        categories = pd.Index(["a", "b", "c"])

        result_df = convert_object_to_categorical(anon_df, "col", categories)

        assert pd.api.types.is_categorical_dtype(result_df["col"])
        # Should include original categories only (no GTREE_ROOT_TAG since it's not in data)
        expected_categories = ["a", "b", "c"]
        assert list(result_df["col"].cat.categories) == expected_categories

    def test_convert_object_to_categorical_with_gtree_root(self):
        """Test object to categorical conversion with GTREE_ROOT_TAG introduced."""
        anon_df = pd.DataFrame({"col": ["a", GTREE_ROOT_TAG, "b"]})
        categories = pd.Index(["a", "b"])

        result_df = convert_object_to_categorical(anon_df, "col", categories)

        assert pd.api.types.is_categorical_dtype(result_df["col"])
        assert result_df["col"].iloc[1] == GTREE_ROOT_TAG
        assert GTREE_ROOT_TAG in result_df["col"].cat.categories

    def test_categorical_roundtrip(self):
        """Test that categorical -> object -> categorical preserves structure."""
        original_categories = ["a", "b", "c"]
        df = pd.DataFrame(
            {"col": pd.Categorical(["a", "b", "c", "a"], categories=original_categories)}
        )

        # Forward conversion
        object_df, metadata = convert_categorical_to_object(df, "col")

        # Back conversion
        result_df = convert_object_to_categorical(object_df, "col", metadata)

        assert pd.api.types.is_categorical_dtype(result_df["col"])
        # Original categories should be preserved (plus GTREE_ROOT_TAG)
        assert all(cat in result_df["col"].cat.categories for cat in original_categories)


class TestDtypeConversionMap:
    """Test the DTYPE_CONVERSION_MAP structure."""

    def test_conversion_map_structure(self):
        """Test that DTYPE_CONVERSION_MAP has expected structure."""
        expected_dtypes = ["bool", "datetime64[ns]", "category"]

        assert set(DTYPE_CONVERSION_MAP.keys()) == set(expected_dtypes)

        for dtype_name, conversion_info in DTYPE_CONVERSION_MAP.items():
            assert "to_supported" in conversion_info
            assert "from_supported" in conversion_info
            assert "target_dtype" in conversion_info

            # Check that functions are callable
            assert callable(conversion_info["to_supported"])
            assert callable(conversion_info["from_supported"])

            # Check target dtypes are supported
            assert conversion_info["target_dtype"] in ["float64", "object"]

    def test_conversion_map_functions_work(self):
        """Test that functions in conversion map actually work."""
        # Test bool conversion
        df_bool = pd.DataFrame({"col": [True, False]})
        to_func = DTYPE_CONVERSION_MAP["bool"]["to_supported"]
        from_func = DTYPE_CONVERSION_MAP["bool"]["from_supported"]

        converted_df, metadata = to_func(df_bool, "col")
        assert converted_df["col"].dtype == "float64"

        result_df = from_func(converted_df, "col", metadata)
        assert result_df["col"].dtype == "boolean"
