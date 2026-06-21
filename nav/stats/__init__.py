"""Statistical analysis of benchmark results.

The rebuttal stats live in three flavors, all sharing the same primitives
in this package:

- **Full** — scene-clustered bootstrap + paired permutation + Spearman on
  per-(scene, point, seed) cells from the grid tree (or any
  scene-labeled per_run.csv).
- **Partial** — episode-level bootstrap + unpaired permutation on the
  legacy ``experiment_results_v6.csv``, which lacks scene labels.
  Spearman is deferred entirely there (the proper rank-correlation test
  cannot be constructed without scene metadata).
- **Merged** — supersession-merge of grid + xlsx per_run.csv into a
  combined source for the headline leaderboard.

Submodules
----------

- :mod:`nav.stats.load` — adapters from raw data sources to a normalized
  per-run row format used by everything downstream.
- :mod:`nav.stats.bootstrap` — both ``clustered_bootstrap_ci`` (the
  scene-cluster resampler) and ``episode_bootstrap_ci`` (the simple
  resampler used when scene labels are missing).
- :mod:`nav.stats.permutation` — both ``paired_permutation_pvalue``
  (sign-flip on cell-paired differences) and ``unpaired_permutation_pvalue``
  (label-shuffle for two-sample tests).
- :mod:`nav.stats.spearman` — Spearman rank correlation + tie-aware
  ``rankdata`` helper, numpy-only.
- :mod:`nav.stats.merge` — supersession merge for combining grid + xlsx.
- :mod:`nav.stats.pipeline` — high-level orchestrators: ``run_full_analysis``,
  ``run_partial_analysis``, ``run_merge_pipeline``. The CLI driver in
  :mod:`nav.scripts.compile_stats` calls into these.
"""
