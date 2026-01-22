"""
Tests for pd
"""

import logging

import pandas as pd

from project_lighthouse_anonymize.pandas_utils import (
    get_temp_col,
    hash_df,
    make_temp_id_col,
)

_LOGGER = logging.getLogger(__name__)
_TEST_NULL_STR_VALUE = "__NULL_STR_VALUE"
_TEST_NULL_NON_STR_VALUE = "__NULL_NONSTR_VALUE"


def test__get_temp_col():
    # TODO not deterministic
    test_df = pd.DataFrame(
        {
            "StringColumn": ["row1", "row2", "row3", "row4"],
            "IntColumn": [1, 2, 3, 4],
            "FloatColumn": [1.1, 2.1, 3.1, 4.1],
        }
    )
    actual = get_temp_col(test_df)
    assert isinstance(actual, str)
    assert len(actual) > 0
    assert actual not in test_df.columns


def test__make_temp_id_col():
    # TODO not deterministic
    test_df = pd.DataFrame(
        {
            "StringColumn": ["row1", "row2", "row3", "row4"],
            "IntColumn": [1, 2, 3, 4],
            "FloatColumn": [1.1, 2.1, 3.1, 4.1],
        }
    )
    id_col = make_temp_id_col(test_df)
    assert id_col in test_df.columns
    assert list(test_df[id_col]) == list(range(len(test_df)))


def test__hashable_dataframe():
    test_df_1 = pd.DataFrame(
        {
            "StringColumn": ["row1", "row2", "row3", "row4"],
            "IntColumn": [1, 2, 3, 4],
            "FloatColumn": [1.1, 2.1, 3.1, 4.1],
        }
    )
    test_df_2 = pd.DataFrame(
        {
            "StringColumn": ["row1", "row2", "row3", "row4"],
            "IntColumn": [1, 2, 3, 4],
            "FloatColumn": [1.1, 2.1, 3.1, 4.1],
        }
    )
    hash_1 = hash_df(test_df_1)
    hash_2 = hash_df(test_df_2)
    assert hash_1 == hash_2
