"""
Shared constants for Project Lighthouse anonymization utilities.

This module defines constants used across Project Lighthouse anonymization components
for consistency in operations like equality comparisons, random number generation,
and data processing.
"""

import math

import numpy as np

# Used to determine equality of floats in Lighthouse
MAXIMUM_PRECISION_DIGITS: int = 8
EPSILON: float = math.pow(10, -MAXIMUM_PRECISION_DIGITS)

MAX_RANDOM_STATE: int = 2**31 - 1
NOT_DEFINED_NA: float = np.nan

GTREE_ROOT_TAG: str = "*"
