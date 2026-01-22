"""
Tests for l-diversity disclosure risk metrics
"""

import logging
import math

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.constants import EPSILON
from project_lighthouse_anonymize.disclosure_risk_metrics import (
    compute_entropy_log_l_diversity,
)

_ID_COL = "id_user"

_LOGGER = logging.getLogger(__name__)


class TestEntropLogLDiversity:
    """
    Tests for compute_entropy_log_l_diversity
    """

    # pylint: disable=no-self-use

    def test_empty_equivalence_classes(self):
        """Test handling when no equivalence classes exist"""
        # Create empty dataframe that results in no equivalence classes
        input_df = pd.DataFrame({"qid1": [], "sens": []})
        result = compute_entropy_log_l_diversity(input_df, [], "sens")
        assert np.isnan(result[0]) and np.isnan(result[1]) and np.isnan(result[2])

    def test_entropy_log_l_diversity_0(self):
        """
        Tests when there are no records.
        """
        sensitive_df = pd.DataFrame(
            data={
                "qid1": [],
                "qid2": [],
                "sens_value": [],
            }
        )
        sensitive_df[_ID_COL] = pd.Series(list(range(len(sensitive_df))), dtype=np.dtype("int64"))
        qids = ["qid1", "qid2"]
        sens_attr_col = "sens_value"

        expected = np.nan, np.nan, np.nan
        actual = compute_entropy_log_l_diversity(sensitive_df, qids, sens_attr_col)
        np.testing.assert_allclose(
            actual, expected, atol=EPSILON, err_msg=f"{actual} != {expected}"
        )

    def test_entropy_log_l_diversity_1(self):
        """
        Tests Figure 3.3 from:

        A. Machanavajjhala, J. Gehrke, D. Kifer, and M. Venkitasubramaniam, "L-diversity: privacy beyond k-anonymity,"
        in 22nd International Conference on Data Engineering (ICDE'06), Atlanta, GA, USA: IEEE, 2006, pp. 24–24. doi: 10.1109/ICDE.2006.1.
        """
        sensitive_df = pd.DataFrame(
            data={
                "zip_code": (
                    [
                        "1305*",
                    ]
                    * 4
                    + [
                        "1485*",
                    ]
                    * 4
                    + [
                        "1306*",
                    ]
                    * 4
                ),
                "age": (
                    [
                        "<= 40",
                    ]
                    * 4
                    + [
                        "> 40",
                    ]
                    * 4
                    + [
                        "<= 40",
                    ]
                    * 4
                ),
                "nationality": (
                    [
                        "*",
                    ]
                    * 4
                    + [
                        "*",
                    ]
                    * 4
                    + [
                        "*",
                    ]
                    * 4
                ),
                "condition": [
                    "Heart Disease",
                    "Viral Infection",
                    "Cancer",
                    "Cancer",
                    "Cancer",
                    "Heart Disease",
                    "Viral Infection",
                    "Viral Infection",
                    "Heart Disease",
                    "Viral Infection",
                    "Cancer",
                    "Cancer",
                ],
            }
        )
        assert len(set(sensitive_df["condition"])) == 3
        sensitive_df[_ID_COL] = pd.Series(list(range(len(sensitive_df))), dtype=np.dtype("int64"))
        qids = ["zip_code", "age", "nationality"]
        sens_attr_col = "condition"

        actual_avg, actual_min, actual_max = compute_entropy_log_l_diversity(
            sensitive_df, qids, sens_attr_col
        )
        np.testing.assert_allclose(
            actual_min, math.log(2.8), atol=0.015, err_msg=f"{actual_min} != log({2.8})"
        )

    @staticmethod
    def kartal_et_al__figure2a() -> tuple[pd.DataFrame, list[str], str, str]:
        """
        Get testing data for Figure 2a from:

        University of Illinois at Springfield, USA, H. Kartal, X.-B. Li, and University of Massachusetts Lowell, USA,
        "Protecting Privacy When Sharing and Releasing Data with Multiple Records per Person," JAIS, vol. 21, pp. 1461–1485, Nov. 2020, doi: 10.17705/1jais.00643.

        Returns (sensitive_df, qids, user_id_col, sens_attr_col)
        """
        sensitive_df = pd.DataFrame(
            data={
                "name": [
                    "A",
                    "A",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                    "C",
                    "D",
                    "D",
                    "E",
                    "E",
                    "F",
                    "G",
                    "H",
                    "H",
                    "H",
                    "H",
                    "H",
                ],
                "age": [
                    86,
                    86,
                    85,
                    69,
                    70,
                    71,
                    71,
                    71,
                    84,
                    84,
                    84,
                    84,
                    78,
                    78,
                    74,
                    75,
                    76,
                    74,
                    76,
                ],
                "gender": [
                    "Female",
                    "Female",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Female",
                    "Female",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                    "Male",
                ],
                "zip": [
                    "20375",
                    "20375",
                    "20375",
                    "20048",
                    "20048",
                    "20048",
                    "20048",
                    "20048",
                    "20090",
                    "20090",
                    "20090",
                    "20090",
                    "20400",
                    "20420",
                    "20400",
                    "20400",
                    "20400",
                    "20400",
                    "20400",
                ],
                "disease": [
                    "Asthma",
                    "Reflux",
                    "Reflux",
                    "Pneumonia",
                    "Pneumonia",
                    "Pneumonia",
                    "Gastritis",
                    "Gastritis",
                    "Ulcer",
                    "Gastritis",
                    "Pneumonia",
                    "Gastritis",
                    "Pneumonia",
                    "Ulcer",
                    "Asthma",
                    "Bronchitis",
                    "Asthma",
                    "Ulcer",
                    "Ulcer",
                ],
            }
        )
        qids = ["age", "gender", "zip"]
        user_id_col = "name"
        sens_attr_col = "disease"
        return sensitive_df, qids, user_id_col, sens_attr_col
