"""
Project Lighthouse Anonymize - Privacy-preserving data anonymization.

This package implements k-anonymity, p-sensitive k-anonymity, and related
privacy-preserving algorithms for tabular data.
"""

from project_lighthouse_anonymize._version import __version__
from project_lighthouse_anonymize.wrappers.k_anonymize import k_anonymize
from project_lighthouse_anonymize.wrappers.p_sensitize import p_sensitize as p_sensitize
from project_lighthouse_anonymize.wrappers.shared import (
    check_dq_meets_minimum_thresholds,
    compute_score,
    default_dq_metric_to_minimum_dq,
    prepare_gtrees,
    select_best_run,
)

__all__ = [
    "__version__",
    "k_anonymize",
    "p_sensitize",
    "default_dq_metric_to_minimum_dq",
    "check_dq_meets_minimum_thresholds",
    "select_best_run",
    "compute_score",
    "prepare_gtrees",
]
