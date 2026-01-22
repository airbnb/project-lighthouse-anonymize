"""
Pandas utility functions.

We call this pandas_utils instead of pandas to avoid mistakes in import statements.
"""

from hashlib import sha256

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.constants import MAX_RANDOM_STATE


def get_temp_col(
    input_df: pd.DataFrame,
    col_prefix: str = "id_col_",
    random_seed: int = 42,
    max_attempts: int = 10_000,
) -> str:
    """
    Get a unique temporary column name not in the given dataframe.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame to check for column name conflicts.
    col_prefix : str, optional
        Prefix for column name, defaults to ``id_col_``.
    random_seed : int, optional
        Random seed for reproducible column names, defaults to 42.
    max_attempts : int, optional
        Maximum number of attempts to find a unique name, defaults to 10,000.

    Returns
    -------
    str
        Unique column name not present in input_df.

    Raises
    ------
    RuntimeError
        If unable to generate a unique column name after max_attempts.
    """
    cols = set(str(col_name) for col_name in input_df.columns)
    rng = np.random.default_rng(seed=random_seed)
    for attempt in range(max_attempts):  # type: ignore[reportUnusedVariable]  # attempt is loop counter, value not needed
        id_col = f"{col_prefix}_{rng.integers(0, MAX_RANDOM_STATE)}"
        if id_col not in cols:
            return id_col
    raise RuntimeError(
        f"Unable to generate unique column name after {max_attempts} attempts. "
        f"DataFrame may have too many existing columns with prefix '{col_prefix}_'."
    )


def make_temp_id_col(input_df: pd.DataFrame) -> str:
    """
    Insert a temporary column with row numbers (0-indexed) to the dataframe.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame to modify in-place.

    Returns
    -------
    str
        Name of the inserted column.
    """
    id_col = get_temp_col(input_df)
    input_df.insert(0, id_col, list(range(len(input_df))))  # type: ignore[reportArgumentType]  # pandas accepts list[int] but stubs are overly restrictive
    return id_col


def hash_df(df: pd.DataFrame) -> int:
    """
    Generate a deterministic hash value for a DataFrame.

    This approach using SHA256 is deterministic across Python runs, unlike hash()
    which uses random seeding for security purposes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash

    Returns
    -------
    int
        Deterministic hash value for the DataFrame
    """
    hash_value = sha256(pd.util.hash_pandas_object(df, index=True).values)  # type: ignore[reportAttributeAccessIssue]  # hash_pandas_object is a real pandas.util function
    return int.from_bytes(hash_value.digest(), "big")
