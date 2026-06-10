"""
Tests RILM numerical computation for QIDs whose original values are all NaN.
"""

import numpy as np
import pandas as pd

from project_lighthouse_anonymize.data_quality_metrics.rilm_ilm import (
    _compute_revised_information_loss_metric_numericals,
)


class TestRILMNanPerimeter:
    """Tests for RILM numericals when a QID's original perimeter is NaN"""

    def test_all_nan_qid_present_in_result_as_nan(self):
        """A numerical QID whose original values are all NaN has an undefined
        perimeter; it must still appear in the result dict, mapped to NaN,
        rather than being silently omitted."""
        input_df = pd.DataFrame(
            {
                "a_orig": [np.nan, np.nan, np.nan, np.nan],
                "a_anon": [np.nan, np.nan, np.nan, np.nan],
                "b_orig": [1.0, 2.0, 3.0, 4.0],
                "b_anon": [1.5, 1.5, 3.5, 3.5],
            }
        )
        rilm = _compute_revised_information_loss_metric_numericals(
            input_df, ["a", "b"], ["a", "b"], "_orig", "_anon"
        )
        assert set(rilm.keys()) == {"a", "b"}
        assert np.isnan(rilm["a"])
        assert not np.isnan(rilm["b"])
