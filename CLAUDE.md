# Project Lighthouse Anonymize - AI Assistant Instructions

This file contains instructions for AI assistants (Claude, ChatGPT, etc.) working on this codebase.

## Project Overview

Privacy-preserving data anonymization library implementing k-anonymity and related algorithms.

## Development Workflow

### Setup
```bash
# Clone and install
git clone https://github.com/airbnb/project-lighthouse-anonymize.git
cd project-lighthouse-anonymize
pip install -e ".[dev]"
```

### Testing
```bash
# Run all fast tests (run with 20 minute timeout)
pytest -n auto -m "not slow" --doctest-modules

# Profile K-Anonymize performance (run with 10 minute timeout)
pytest --profile-svg -k "test_profile_k" --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)" --log-cli-date-format="%Y-%m-%d %H:%M:%S"

# Profile P-Sensitive performance (run with 10 minute timeout)
pytest --profile-svg -k "test_p_sensitize_8" --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)" --log-cli-date-format="%Y-%m-%d %H:%M:%S"

# Run with coverage (includes multiprocessing coverage)
pytest -n auto -m "not slow" --cov=project_lighthouse_anonymize --cov-report=html --cov-report=term-missing --doctest-modules

# Run specific test file
pytest tests/wrappers/test_k_anonymize.py --doctest-modules
```

### Code Quality
```bash
# Format code
ruff format src/ tests/
ruff check src/ tests/

# Type checking
pyright src/

# All checks
ruff format src/ tests/ && ruff check src/ tests/
```

### Pre-Commit Requirements

**IMPORTANT**: Before committing any changes, you MUST:

1. **Lint the code**: Run `ruff format src/ tests/ && ruff check src/ tests/`
2. **Run tests**: Run `pytest -n auto -m "not slow" --doctest-modules`

All linting and tests must pass before creating a commit.

## Code Style Guidelines

<edit_guidelines>

### Docstrings
Use **NumPy style conventions**:

```python
def example_function(param1, param2):
    """Brief description.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description

    Returns
    -------
    type
        Description
    """
```

### Citations in Docstrings

When adding academic references to module or function docstrings:

1. **Use globally unique citation names** - Sphinx combines all docstrings, so `[1]` in different modules will collide. Use `[AuthorYear]` style (e.g., `[Sweeney2002]`, `[LeFevre2006]`).

2. **Always reference citations in the text** - Sphinx requires citations to be referenced with `[CitationName]_` syntax:

```python
"""
This implements k-anonymity [Sweeney2002]_, using Mondrian partitioning [LeFevre2006]_.

References
----------
.. [Sweeney2002] L. Sweeney. "k-anonymity: A model for protecting privacy." ...
.. [LeFevre2006] K. LeFevre et al. "Mondrian multidimensional k-anonymity." ...
"""
```

3. **For multiple papers by same author/year**, add suffixes: `[Bloomston2025a]`, `[Bloomston2025b]`

### Comments
- ONLY add comments for: complex algorithms, non-obvious privacy model assumptions, performance optimizations, subtle bugs/workarounds
- DON'T comment: obvious code, standard library usage, basic flow
- Focus on **why**, not **what**

### Imports
```python
# Standard library
import os
from typing import List, Optional

# Third-party
import numpy as np
import pandas as pd

# Local
from project_lighthouse_anonymize.utils import foo
```

### Test Style
Use **class-based tests** (pytest class style) for consistency:

```python
class TestKAnonymize:
    """Tests for k_anonymize function"""

    def test_basic_functionality(self):
        """Test basic k-anonymization"""
        # Test implementation
        pass

    @staticmethod
    def helper_data():
        """Helper method for test data"""
        return {"key": "value"}
```

Guidelines:
- Group related tests in test classes (e.g., `TestKAnonymize`)
- Use `@staticmethod` for test helper methods that don't need instance state
- Use `@property` or class attributes for shared test data when appropriate
- Keep test data generation methods within the test class

</edit_guidelines>

## Documentation

### Building Documentation

The project uses Sphinx for documentation generation with NumPy-style docstrings.

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
sphinx-build -b html docs docs/_build/html

# Open documentation in browser
open docs/_build/html/index.html  # macOS
```

### API Reference Structure (docs/api.rst)

**IMPORTANT**: The API reference in `docs/api.rst` must be kept in sync with the source code structure.

**When to Update**: You MUST update `docs/api.rst` whenever:
- New Python modules are added to `src/project_lighthouse_anonymize/`
- Python modules are removed or renamed
- New packages/subdirectories are added
- The package structure changes

**Structure Format**: The API reference follows a DFS (depth-first search) traversal of the source code:

1. **Main Functions** - All functions exported by top-level `__init__.py`
2. **Source modules in alphabetical order**, following the directory tree structure

Example structure:
```rst
Main Functions
--------------
.. autofunction:: project_lighthouse_anonymize.k_anonymize
.. autofunction:: project_lighthouse_anonymize.p_sensitize

Constants
---------
.. automodule:: project_lighthouse_anonymize.constants

Data Quality Metrics
--------------------
Miscellaneous Metrics
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: project_lighthouse_anonymize.data_quality_metrics.misc

[Continue DFS through all modules...]
```

When adding new modules, insert them in the correct alphabetical/hierarchical position following the DFS pattern.

## Project Structure

```
src/project_lighthouse_anonymize/
├── __init__.py              # Public API exports
├── wrappers/                # Main user-facing functions
│   ├── k_anonymize.py      # K-anonymity wrapper
│   ├── p_sensitize.py      # P-sensitive wrapper
│   └── shared.py           # Shared utilities
├── mondrian/               # Core Mondrian algorithm
├── data_quality_metrics/   # DQ metric implementations
└── utils.py                # Utility functions

tests/
├── wrappers/               # Wrapper tests
├── data_quality_metrics/   # DQ metric tests
└── integration/            # Integration tests

docs/
├── conf.py                 # Sphinx configuration
├── index.rst               # Documentation home
├── api.rst                 # API reference (KEEP IN SYNC!)
├── getting_started.md      # Getting started guide
└── requirements.txt        # Doc build dependencies
```
