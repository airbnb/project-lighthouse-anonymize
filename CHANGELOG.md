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
- `GTree` now raises `ValueError` when two leaves share the same value;
  previously the value-to-leaf lookup silently kept the last leaf, producing
  wrong lowest-common-ancestor generalizations and overlapping (record
  duplicating) partitions when cutting on such a gtree
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `compute_entropy_log_l_diversity` now treats empty `qids` as a single
  whole-dataset equivalence class, consistent with `calculate_p_k`, and
  documents that the computed quantity is log(l) (the entropy), not l
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))

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
- `compute_score` no longer returns NaN when a metric value or threshold is
  NaN: NaN thresholds disable their metric and NaN values score worst-possible
  for their metric. Previously a single NaN poisoned the total score, making
  `select_best_run` order-dependent — for categorical-only datasets (all-NaN
  pearsons/NMI metrics) it always returned the first run regardless of
  quality. `select_best_run` also now scores with the same thresholds used
  for its final filtering pass
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `calculate_p_k` and `compute_entropy_log_l_diversity` no longer count
  phantom empty equivalence classes created by unobserved categories of
  categorical columns; previously categorical QIDs yielded p=0/k=None and
  NaN or worst-case 0.0 entropy for classes containing no individuals
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `p_sensitize` now supports categorical QID columns: phantom empty groups no
  longer crash perturbation or bypass the documented k-anonymity guard, and
  input validation counts only sensitive values with non-zero probability
  toward `target_p` feasibility
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- NMI no longer truncates genuinely fractional anonymized values to integers
  when the original column is integer-valued and the anonymized column has
  locally suppressed cells; previously a single suppressed cell could flip
  NMIv1 from ~1.0 to 0.0. NMI also no longer crashes on fractional means
  imputed into integer-converted columns, all-NaN QIDs, or fully suppressed
  integer-convertible columns (these now report NaN), and no longer mutates
  the caller's DataFrame dtypes
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `convert_object_to_categorical` now preserves generalized gtree node
  labels; previously any label other than the gtree root tag was silently
  converted to NaN, corrupting generalized values into missing data
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `GTree.from_config_json` no longer destructively consumes its input
  dictionary (a second construction from the same dictionary silently
  produced an empty tree with stale lookup maps), and JSON round trips now
  preserve lookups for non-string node values (previously stringified by
  JSON serialization, breaking value lookups after reload)
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `GTree.add_default_geometric_sizes` no longer crashes on a root-only tree
  (e.g. `make_flat_default_gtree(set())`); the root receives geometric size 0
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `InProcessResult.result()` now executes its function exactly once and
  caches the result, matching `concurrent.futures.Future`; previously every
  call re-executed against the current (possibly mutated) state of shared
  arguments, so serial and parallel runs could diverge
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- Mondrian cut selection now orders NaN proposed-cut scores as worst;
  previously a NaN-scored cut (zero variance in the scoring sample) could
  never be displaced by a later finite-scored cut, and sorting scores
  containing NaN violated strict weak ordering, selecting arbitrary cuts
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `Implementation_KAnonymity.validate` now returns True for empty DataFrames
  per its documented vacuous-truth contract
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `MondrianTree.candidate_solution` no longer mutates the DataFrames stored
  in the tree when `node_identifier_col` is set without `track_cuts`
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))

### Security

## [1.0.0] - 2026-01-29

### Added

- Initial release of Project Lighthouse Anonymize

[unreleased]: https://github.com/airbnb/project-lighthouse-anonymize/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/airbnb/project-lighthouse-anonymize/releases/tag/v1.0.0
