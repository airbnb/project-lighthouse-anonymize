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
- `GTree` now raises `ValueError` when two leaves share the same value,
  including at config save and load time; previously the value-to-leaf
  lookup silently kept the last leaf, producing wrong
  lowest-common-ancestor results and overlapping partition filters, and
  anonymization runs on such gtrees failed pathologically (infinite
  recursion or empty output) when the duplicated value was present in the
  data. Gtrees with duplicate leaf values are rejected even when the
  duplicated value does not occur in the data
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `compute_entropy_log_l_diversity` now treats empty `qids` as a single
  whole-dataset equivalence class, consistent with `calculate_p_k`, and
  documents that the computed quantity is log(l) (the entropy), not l
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- NMI values may shift for QID pairs containing locally suppressed cells:
  a column whose missing values are imputed with the (generally fractional)
  mean is now consistently treated as continuous, where previously the
  estimator choice flipped between the discrete and continuous mutual
  information estimators depending on whether the imputed mean happened to
  be integral
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `p_sensitize` now rejects categorical dtype columns (QIDs and the
  sensitive attribute) up front with guidance to use the dtype conversion
  utilities, consistent with the documented dtype contract; previously
  categorical QIDs crashed mid-perturbation with an opaque error and
  categorical sensitive columns crashed seed-dependently
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
  NaN: NaN thresholds disable their metric and NaN values score strictly
  below any measurable value for their metric. Previously a single NaN
  poisoned the total score, making `select_best_run` order-dependent and
  able to return an arbitrary run regardless of quality.
  `select_best_run` also now scores with the same thresholds used for its
  final filtering pass
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `calculate_p_k` and `compute_entropy_log_l_diversity` no longer count
  phantom empty equivalence classes created by unobserved categories of
  categorical columns; previously categorical QIDs yielded p=0/k=None and
  NaN or worst-case 0.0 entropy for classes containing no individuals
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `p_sensitize` raises a clear per-class error when an equivalence class
  cannot reach `target_p` because too few values with non-zero probability
  are available to add; previously the failure surfaced as an opaque numpy
  sampling error. Zero-probability values already present in a class still
  count toward its p, so previously feasible configurations remain accepted
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- NMI no longer truncates genuinely fractional anonymized values to integers
  when one column of a QID pair converts to a discrete dtype and the other is
  genuinely fractional (e.g. 0/1-valued float QIDs with suppressed cells).
  NMI also no longer crashes on fractional means imputed into
  integer-converted columns, all-NaN QIDs (these report NaN — there is no
  data), or fully suppressed anonymized columns (these report NMIv1=0.0 and
  NMIv2=1.0 per the constant-column convention, so the destroyed attribute
  fails data quality thresholds instead of being dropped by the NaN-skipping
  aggregation), and no longer mutates the caller's DataFrame dtypes
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `convert_object_to_categorical` now preserves generalized gtree node
  labels; previously any label other than the gtree root tag was silently
  converted to NaN, corrupting generalized values into missing data
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `GTree.from_config_json` no longer destructively consumes its input
  dictionary (a second construction from the same dictionary silently
  produced an empty tree with stale lookup maps), and all lookup maps are
  now rebuilt from the loaded nodes instead of trusted from the file: JSON
  round trips preserve lookups for non-string node values (previously
  stringified by JSON serialization), and stale or hand-edited
  configurations can no longer produce internally inconsistent trees
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `GTree.add_default_geometric_sizes` no longer crashes on a root-only tree
  (e.g. `make_flat_default_gtree(set())`); the root receives geometric size 0
  ([#17](https://github.com/airbnb/project-lighthouse-anonymize/pull/17))
- `InProcessResult.result()` now executes its function exactly once, caching
  the result and re-raising a cached exception on subsequent calls, matching
  `concurrent.futures.Future`; previously every call re-executed against the
  current (possibly mutated) state of shared arguments, so serial and
  parallel runs could diverge
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
