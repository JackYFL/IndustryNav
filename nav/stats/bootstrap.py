"""Bootstrap confidence intervals over benchmark metrics.

Two flavors:

- :func:`clustered_bootstrap_ci` — resample at the *scene* level (with
  replacement) and recompute the global metric. Used when per-cell
  observations within a scene are positively correlated, which they are
  for navigation runs (same map geometry, same target). The CI it
  produces is wider than naive episode-level resampling and is the
  honest one to report.

- :func:`episode_bootstrap_ci` — naive resampling of individual
  observations. Used only as a fallback when scene labels are missing
  (the legacy ``experiment_results_v6.csv`` path). The CI it produces is
  artificially tight and should be flagged as a lower bound.

Both use add-1 smoothing-style percentile CIs (no parametric assumption)
and the caller-supplied ``rng`` so seeded reproducibility is preserved
end-to-end.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def clustered_bootstrap_ci(
    per_scene_values: Dict[str, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float, float]:
    """Scene-clustered bootstrap CI for the mean of a metric.

    ``per_scene_values[scene]`` is the array of per-cell observations in
    that scene. Each bootstrap iteration resamples *scenes* with
    replacement (not cells) and pools the chosen scenes' cells before
    computing the mean.

    Returns ``(point_estimate, ci_lo, ci_hi)`` where ``point_estimate`` is
    the mean across all observations in all scenes (unconditioned on the
    bootstrap), and ``ci_lo / ci_hi`` are the percentile bounds.
    """
    scenes = list(per_scene_values.keys())
    if not scenes:
        return float("nan"), float("nan"), float("nan")
    pe = float(np.mean(np.concatenate(list(per_scene_values.values()))))
    samples = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(len(scenes), size=len(scenes), replace=True)
        pooled = np.concatenate([per_scene_values[scenes[j]] for j in chosen])
        samples[i] = pooled.mean()
    lo, hi = np.percentile(samples, ci)
    return pe, float(lo), float(hi)


def episode_bootstrap_ci(
    arr: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    stat: Callable[[np.ndarray], float] = np.mean,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float, float]:
    """Naive episode-level bootstrap CI (no clustering).

    Resamples individual observations with replacement and applies
    ``stat`` to each resample. Use only when no cluster label is
    available; otherwise prefer :func:`clustered_bootstrap_ci`.

    Returns ``(point_estimate, ci_lo, ci_hi)``.
    """
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    pe = float(stat(arr))
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = stat(arr[idx])
    lo, hi = np.percentile(samples, ci)
    return pe, float(lo), float(hi)
