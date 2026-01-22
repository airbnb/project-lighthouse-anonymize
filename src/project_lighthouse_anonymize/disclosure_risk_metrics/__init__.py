"""
Disclosure risk metrics for evaluating privacy protection in anonymized datasets.

This module provides metrics that quantify disclosure risk by validating privacy
model compliance and analyzing re-identification and attribute disclosure risks.
All functions in this module measure various aspects of disclosure risk - the
likelihood that sensitive information can be inferred or individuals can be
re-identified from anonymized data.

The module is organized into two main categories:

1. **L-diversity based disclosure risk**: Measures attribute disclosure risk through
   diversity analysis of sensitive values within equivalence classes.

2. **P-sensitive k-anonymity disclosure risk**: Measures both re-identification risk
   (through k-anonymity) and attribute disclosure risk (through p-sensitivity) by
   validating privacy model compliance.

Key Insight: Privacy model validation IS disclosure risk calculation. Lower compliance
with privacy models (lower k, p, l-diversity values) indicates higher disclosure risk.

Functions
---------
compute_entropy_log_l_diversity : Tuple[float, float, float]
    Calculate entropy l-diversity disclosure risk across equivalence classes.
    Higher l-diversity values indicate lower attribute disclosure risk.

calculate_p_k : Tuple[Optional[int], int]
    Calculate p-sensitive k-anonymity compliance for disclosure risk assessment.
    Lower p,k values indicate higher re-identification and attribute disclosure risk.

"""

from .l_diversity import compute_entropy_log_l_diversity
from .p_sensitive_k_anonymity import calculate_p_k

__all__ = [
    "compute_entropy_log_l_diversity",
    "calculate_p_k",
]
