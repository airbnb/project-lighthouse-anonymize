"""
Data quality metrics package for Project Lighthouse anonymization evaluation.

This package provides modular implementation of data quality metrics [Bloomston2025b]_ used to
evaluate the impact of anonymization on data utility. The metrics are organized into focused
modules while maintaining backward compatibility with existing imports.

Modules:
- pearson: Pearson's correlation coefficient for linear relationship preservation
- rilm_ilm: Revised Information Loss Metric (RILM) and Information Loss Metric (ILM)
- nmi: Normalized Mutual Information for entropy preservation measurement
- misc: Miscellaneous metrics including generalization/suppression metrics

All functions can be imported directly from this package for backward compatibility:
    from project_lighthouse_anonymize.data_quality_metrics import compute_pearsons_correlation_coefficients

References
----------
.. [Bloomston2025b] Bloomston, A., Burke, E., Cacace, M., Diaz, A., Dougherty, W., Gonzalez, M.,
       Gregg, R., Güngör, Y., Hayes, B., Hsu, E., Israeli, O., Kim, H., Kwasnick, S.,
       Lacsina, J., Rodriguez, D. R., Schiller, A., Schumacher, W., Simon, J., Tang, M.,
       Wharton, S., & Wilcken, M. (2025). Measuring Data Quality for Project Lighthouse.
       arXiv preprint arXiv:2510.06121. https://arxiv.org/abs/2510.06121
"""

from .misc import (
    compute_average_equivalence_class_metric,
    compute_discernibility_metric,
    compute_suppression_metrics,
)
from .nmi import (
    compute_normalized_mutual_information_sampled_scaled,
    nmi_unscaled_to_scaled,
)

# Import all functions from submodules to maintain backward compatibility
from .pearson import compute_pearsons_correlation_coefficients
from .rilm_ilm import (
    compute_average_information_loss_metric,
    compute_revised_information_loss_metric,
    compute_revised_information_loss_metric_categoricals,
    compute_revised_information_loss_metric_value_for_categorical,
    compute_revised_information_loss_metric_value_for_numerical,
)

# Define __all__ to control what gets exported with "from package import *"
__all__ = [
    # Pearson correlation
    "compute_pearsons_correlation_coefficients",
    # RILM/ILM functions
    "compute_revised_information_loss_metric",
    "compute_average_information_loss_metric",
    "compute_revised_information_loss_metric_categoricals",
    "compute_revised_information_loss_metric_value_for_numerical",
    "compute_revised_information_loss_metric_value_for_categorical",
    # NMI functions
    "compute_normalized_mutual_information_sampled_scaled",
    "nmi_unscaled_to_scaled",
    # Generalization/suppression metrics
    "compute_suppression_metrics",
    "compute_average_equivalence_class_metric",
    "compute_discernibility_metric",
]
