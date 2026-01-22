"""
Tests using the Adults dataset for k-anonymity and p-sensitive k-anonymity.

This file contains profiling tests that use k-anonymity and p-sensitize functionality
with the UCI Adults dataset.
"""

import logging
import os
import tempfile
import time
from pprint import pformat

import numpy as np
import pandas as pd
import pytest

from project_lighthouse_anonymize.wrappers.k_anonymize import k_anonymize
from project_lighthouse_anonymize.wrappers.p_sensitize import p_sensitize
from project_lighthouse_anonymize.wrappers.shared import (
    check_dq_meets_minimum_thresholds,
    compute_score,
    select_best_run,
)

_LOGGER = logging.getLogger(__name__)

_QIDS = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
_QIDS_NUMERICAL = ["age", "capital-gain", "capital-loss", "hours-per-week"]
_QIDS_CATEGORICAL = [qid for qid in _QIDS if qid not in _QIDS_NUMERICAL]
_ID_COL = "row_id"
_DEMOGRAPHIC_COL = "race"
_ALL_COLS = _QIDS + [
    _DEMOGRAPHIC_COL,
]

_ADULTS_FILE = os.getenv("ADULTS_DATA_PATH") or os.path.join(tempfile.gettempdir(), "adults.pqt")

_ALL_DEMOGRAPHICS = [
    "Amer-Indian-Eskimo",
    "Asian-Pac-Islander",
    "Black",
    "Other",
    "White",
]
_SENS_ATTR_VALUE_TO_PROB = {
    demographic: 1.0 / len(_ALL_DEMOGRAPHICS) for demographic in _ALL_DEMOGRAPHICS
}


@pytest.fixture(scope="session")
def download_and_preprocess_adults():
    """
    Download and preprocess the Adults dataset for testing.

    This fixture downloads the dataset directly from UCI ML Repository and performs
    preprocessing, saving it as a parquet file for test use. Runs once per session.
    """
    from ucimlrepo import fetch_ucirepo

    if os.path.exists(_ADULTS_FILE):
        return

    adult = fetch_ucirepo(id=2)

    features = adult.data.features
    y = adult.data.targets

    assert features is not None and y is not None, "Failed to download Adult dataset"

    adult_df = pd.concat([features, y], axis=1)
    assert len(adult_df) == len(features), (
        f"DataFrame lengths don't match: {len(adult_df)} vs {len(features)}"
    )

    adult_df["income"].replace("<=50K.", "<=50K", inplace=True)
    adult_df["income"].replace(">50K.", ">50K", inplace=True)

    input_df = adult_df[_ALL_COLS].copy()
    for qid in _QIDS_CATEGORICAL:
        input_df.loc[input_df[qid] == "?", qid] = np.nan
    input_df = input_df.reset_index(drop=True)
    input_df[_ID_COL] = list(range(len(input_df)))

    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(_ADULTS_FILE), suffix=".pqt", delete=False
    ) as temp_file:
        temp_filename = temp_file.name
    input_df.to_parquet(temp_filename, index=False)
    os.replace(temp_filename, _ADULTS_FILE)


def __run_adults_k_anonymity_and_p_sensitive(
    qids,
    parallelism,
    dynamic_breakout_rilm_multiplier,
    rilm_score_epsilon,
):
    """Run k-anonymity and p-sensitive k-anonymity tests"""
    run_details = f"""
    qids = {qids}
    dynamic_breakout_rilm_multiplier = {dynamic_breakout_rilm_multiplier}, rilm_score_epsilon = {rilm_score_epsilon}
    parallelism = {parallelism}
    """
    input_df = pd.read_parquet(_ADULTS_FILE)[
        list(qids)
        + [
            _ID_COL,
            _DEMOGRAPHIC_COL,
        ]
    ]
    t0 = time.perf_counter()

    anon_df, dq_metrics, disclosure_metrics = k_anonymize(
        _LOGGER,
        input_df,
        qids,
        5,
        {},
        _ID_COL,
        rilm_score_epsilon=rilm_score_epsilon,
        rilm_score_epsilon_partition_size_cutoff=0.90,
        dynamic_breakout_rilm_multiplier=dynamic_breakout_rilm_multiplier,
        parallelism=parallelism,
        rnd_mode=True,
    )

    t1 = time.perf_counter()
    _, minimum_dq_met = select_best_run([dq_metrics])
    _, failure_details = check_dq_meets_minimum_thresholds(dq_metrics)
    score = compute_score(dq_metrics)
    _LOGGER.info(
        f"""
    {run_details}
    K-Anonymize
    dt = {t1 - t0}
    minimum_dq_met = {minimum_dq_met}
    failure_details = {failure_details}
    score = {score}
    dq_metrics =
    {pformat(dq_metrics)}
    disclosure_metrics =
    {pformat(disclosure_metrics)}
    """
    )

    _LOGGER.info("Running P-Sensitize")

    t0 = time.perf_counter()

    sensitized_df, dq_metrics, disclosure_metrics = p_sensitize(
        _LOGGER,
        anon_df,
        qids,
        _DEMOGRAPHIC_COL,
        2,
        5,
        _SENS_ATTR_VALUE_TO_PROB,
        seed=42,
        parallelism=parallelism,
        id_col=_ID_COL,
    )

    t1 = time.perf_counter()
    _LOGGER.info(
        f"""
    {run_details}
    P-Sensitize
    dt = {t1 - t0}
    dq_metrics =
    {pformat(dq_metrics)}
    disclosure_metrics =
    {pformat(disclosure_metrics)}
    """
    )


@pytest.mark.slow
def test_profile_k(download_and_preprocess_adults):
    __run_adults_k_anonymity_and_p_sensitive(list(_QIDS), None, None, -1.0)
