# Project Lighthouse Anonymize

[![PyPI version](https://badge.fury.io/py/project-lighthouse-anonymize.svg)](https://pypi.org/project/project-lighthouse-anonymize/)
[![Python versions](https://img.shields.io/pypi/pyversions/project-lighthouse-anonymize.svg)](https://pypi.org/project/project-lighthouse-anonymize/)
[![Documentation Status](https://readthedocs.org/projects/project-lighthouse-anonymize/badge/?version=latest)](https://project-lighthouse-anonymize.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Privacy-preserving data anonymization using k-anonymity and related algorithms.

## Installation

```bash
pip install project-lighthouse-anonymize
```

## Documentation

Full documentation: https://project-lighthouse-anonymize.readthedocs.io

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](https://project-lighthouse-anonymize.readthedocs.io/en/latest/api/)

## Publications

This work builds on research into privacy-preserving data analysis:

- [Measuring Discrepancies in Airbnb Guest Acceptance Rates Using Anonymized Demographic Data](https://arxiv.org/abs/2204.12001) - The foundational paper for Project Lighthouse
- [Core Mondrian: Scalable Mondrian for Partition-Based Anonymization](https://arxiv.org/abs/2510.09661) - Covers the anonymization algorithm
- [Measuring data quality for Project Lighthouse](https://arxiv.org/abs/2510.06121) - Covers the way we measure the impact of anonymization

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Releasing

To create a new release, use the `/release` command in Claude Code. This will:
1. Analyze changes since the last release
2. Propose an appropriate version bump following semantic versioning
3. Update CHANGELOG.md
4. Create and push a git tag
5. Trigger automated PyPI publishing via GitHub Actions

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed by the Airbnb Anti-Discrimination & Equity team.

## Citation

If you use this software in your research, please cite:

```
Bloomston, A., & Airbnb Anti-Discrimination & Equity Engineering Team. (2025).
Project Lighthouse Anonymize. https://github.com/airbnb/project-lighthouse-anonymize
```

BibTeX:
```bibtex
@software{bloomston2025lighthouse,
  author = {Bloomston, Adam and {Airbnb Anti-Discrimination \& Equity Engineering Team}},
  title = {Project Lighthouse Anonymize},
  year = {2026},
  url = {https://github.com/airbnb/project-lighthouse-anonymize},
  license = {MIT}
}
```
