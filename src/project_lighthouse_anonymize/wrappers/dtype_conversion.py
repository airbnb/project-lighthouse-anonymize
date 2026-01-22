"""
Data type conversion utilities for anonymization preprocessing and postprocessing.

This module provides functions to convert unsupported pandas dtypes to supported ones
before anonymization, and convert them back after anonymization while preserving
the original data semantics and handling missing values appropriately.
"""

from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.constants import GTREE_ROOT_TAG, MAX_RANDOM_STATE
from project_lighthouse_anonymize.pandas_utils import hash_df


def convert_bool_to_float(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, Any]:
    """
    Convert boolean column to float64 for anonymization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name to convert

    Returns
    -------
    Tuple[pd.DataFrame, Any]
        Modified dataframe and metadata for reconstruction
    """
    df_copy = df.copy()
    df_copy[col] = df_copy[col].astype(float)
    return df_copy, None


def convert_float_to_bool(
    anon_df: pd.DataFrame, col: str, metadata: Any, rng_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Convert anonymized float column back to boolean using weighted coin flip.

    Parameters
    ----------
    anon_df : pd.DataFrame
        Anonymized dataframe
    col : str
        Column name to convert back
    metadata : Any
        Metadata from the forward conversion (unused for bool)
    rng_state : Optional[int], default=None
        Optional RNG state for reproducible conversion; if None, a deterministic
        state is derived from the column data

    Returns
    -------
    pd.DataFrame
        Dataframe with column converted back to boolean
    """
    df_copy = anon_df.copy()
    if rng_state is None:
        rng_state = hash_df(cast(pd.DataFrame, anon_df[[col]])) % MAX_RANDOM_STATE
    rng = np.random.default_rng(seed=rng_state)
    mask = ~df_copy[col].isna()
    df_copy.loc[mask, col] = df_copy.loc[mask, col].clip(0.0, 1.0) > rng.random(mask.sum())
    df_copy[col] = df_copy[col].astype("boolean")
    return df_copy


def convert_datetime_to_float(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, Any]:
    """
    Convert datetime column to float64 nanoseconds since epoch for anonymization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name to convert

    Returns
    -------
    Tuple[pd.DataFrame, Any]
        Modified dataframe and metadata for reconstruction
    """
    df_copy = df.copy()
    # Use pandas to_numeric to convert datetime to float, handling NaT properly
    df_copy[col] = df_copy[col].astype("datetime64[ns]").view("int64").astype("float64")
    # Convert NaT sentinel value to proper NaN
    df_copy.loc[df_copy[col] == np.datetime64("NaT").view("int64").astype("float64"), col] = np.nan
    return df_copy, None


def convert_float_to_datetime(anon_df: pd.DataFrame, col: str, metadata: Any) -> pd.DataFrame:
    """
    Convert anonymized float column back to datetime.

    Parameters
    ----------
    anon_df : pd.DataFrame
        Anonymized dataframe
    col : str
        Column name to convert back
    metadata : Any
        Metadata from the forward conversion (unused for datetime)

    Returns
    -------
    pd.DataFrame
        Dataframe with column converted back to datetime
    """
    df_copy = anon_df.copy()
    df_copy[col] = pd.to_datetime(df_copy[col], unit="ns")
    return df_copy


def convert_categorical_to_object(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, Any]:
    """
    Convert categorical column to object (string) for anonymization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name to convert

    Returns
    -------
    Tuple[pd.DataFrame, Any]
        Modified dataframe and original categories for reconstruction
    """
    df_copy = df.copy()
    categories = df_copy[col].cat.categories
    df_copy[col] = df_copy[col].astype("object")
    return df_copy, categories


def convert_object_to_categorical(anon_df: pd.DataFrame, col: str, metadata: Any) -> pd.DataFrame:
    """
    Convert anonymized object column back to categorical.

    Parameters
    ----------
    anon_df : pd.DataFrame
        Anonymized dataframe
    col : str
        Column name to convert back
    metadata : Any
        Original categories from the forward conversion

    Returns
    -------
    pd.DataFrame
        Dataframe with column converted back to categorical
    """
    df_copy = anon_df.copy()
    categories = metadata
    # Only add GTREE_ROOT_TAG if it's present in data and not already in categories
    new_categories = list(categories)
    if GTREE_ROOT_TAG in df_copy[col].values and GTREE_ROOT_TAG not in categories:
        new_categories.append(GTREE_ROOT_TAG)
    df_copy[col] = pd.Categorical(df_copy[col], categories=new_categories)
    return df_copy


# Mapping of conversion functions for supported dtype conversions
DTYPE_CONVERSION_MAP = {
    "bool": {
        "to_supported": convert_bool_to_float,
        "from_supported": convert_float_to_bool,
        "target_dtype": "float64",
    },
    "datetime64[ns]": {
        "to_supported": convert_datetime_to_float,
        "from_supported": convert_float_to_datetime,
        "target_dtype": "float64",
    },
    "category": {
        "to_supported": convert_categorical_to_object,
        "from_supported": convert_object_to_categorical,
        "target_dtype": "object",
    },
}
