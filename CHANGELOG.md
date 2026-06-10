# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

- Documented that RILM omits numerical QIDs whose original values are all NaN
  from its result dictionary rather than mapping them to NaN; downstream
  threshold checks rely on the distinction between a missing metric (passing)
  and a NaN metric (failing)
  ([#16](https://github.com/airbnb/project-lighthouse-anonymize/pull/16))

### Deprecated

### Removed

### Fixed

- NMI values are now computed with the requested `sample_size` and
  `number_runs` for every QID; previously a QID with fewer non-missing rows
  than `sample_size` shrank the sample size and forced a single run for all
  QIDs processed after it, degrading and destabilizing their scores
  ([#16](https://github.com/airbnb/project-lighthouse-anonymize/pull/16))
- NMI now reports NaN when a mutual information computation fails (e.g. fewer
  samples than `n_neighbors`); previously the undefined result was silently
  coerced to 0.0, reporting complete information loss and wrongly failing
  data quality thresholds
  ([#16](https://github.com/airbnb/project-lighthouse-anonymize/pull/16))
- `p_sensitize` no longer perturbs every record of an existing sensitive
  value, which could eliminate that value from an equivalence class and leave
  the output below the target p, causing the wrapper to raise `RuntimeError`
  ([#16](https://github.com/airbnb/project-lighthouse-anonymize/pull/16))
- The Mondrian cut-scoring suppression tiebreaker now reflects the fraction
  of records a proposed cut retains; previously it always evaluated to 1.0,
  so the documented preference for suppressing earlier in the cut tree never
  took effect
  ([#16](https://github.com/airbnb/project-lighthouse-anonymize/pull/16))

### Security

## [1.0.0] - 2026-01-29

### Added

- Initial release of Project Lighthouse Anonymize

[unreleased]: https://github.com/airbnb/project-lighthouse-anonymize/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/airbnb/project-lighthouse-anonymize/releases/tag/v1.0.0
