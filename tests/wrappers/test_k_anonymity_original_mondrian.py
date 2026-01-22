"""
Tests for k_anonymize wrapper function using Original Mondrian implementation

This file contains tests specifically for the Original Mondrian algorithm implementation,
which predates the current optimized Mondrian algorithm. These tests ensure backward
compatibility and validate the original algorithm behavior.

# Test Architecture

This file follows the inheritance-based test architecture defined in test_wrappers_k_anonymity.py.
See that file's docstring for detailed explanation of:
- Base test class patterns and inheritance hierarchy
- Pytest parametrization via pytest_generate_tests()
- The __test__ = True/False pattern for test collection
- How to add new test classes and methods

# Original Mondrian Specific Implementation

## Test Classes

- `TestKAnonymityOriginalMondrian`: Inherits from TestKAnonymitySharedNumericalWithoutNaN
  - Uses KWARGS_ORIGINAL_MONDRIAN configuration instead of the standard KWARGS
  - Tests the original Mondrian algorithm implementation
  - Ensures legacy algorithm still produces correct anonymization results

## Parametrization Logic

The pytest_generate_tests() function uses KWARGS_ORIGINAL_MONDRIAN instead of the
standard KWARGS, providing algorithm-specific configuration for the original
Mondrian implementation.
"""

import logging

# Import the base test classes
from .test_k_anonymity import TestKAnonymitySharedNumericalWithoutNaN

_LOGGER = logging.getLogger(__name__)
_ID_COL = "row_id"

# Original Mondrian configuration only
KWARGS_ORIGINAL_MONDRIAN = [
    {
        "parallelism": 10,
        "rilm_score_epsilon": 0.05,
        "dynamic_breakout_rilm_multiplier": 0.75,
        "complex_numerical_cut_points_modes": None,
        "use_original_mondrian": True,
    }
]


def pytest_generate_tests(metafunc):
    if "kwargs" in metafunc.fixturenames:
        metafunc.parametrize("kwargs", KWARGS_ORIGINAL_MONDRIAN)


class TestKAnonymityOriginalMondrianNumerical(TestKAnonymitySharedNumericalWithoutNaN):
    """Tests for k_anonymize using Original Mondrian with numerical data"""

    # Inherits all NaN-free numerical test methods from TestKAnonymitySharedNumericalWithoutNaN
    # but uses KWARGS_ORIGINAL_MONDRIAN parameters via pytest_generate_tests
    __test__ = True  # Override the base class to allow test collection

    def test_k_equals_one_identity_transformation_numerical(self, kwargs):
        """
        Override for Original Mondrian: k=1 not supported (requires k >= 2).

        Original Mondrian implementation has an assertion that k >= 2, so this test
        is not applicable and is skipped.
        """
        pass
