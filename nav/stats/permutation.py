"""Permutation tests for benchmark metric comparisons.

Both implementations share the same conventions:

- Two-sided alternative hypothesis.
- Add-1 smoothing on both numerator and denominator (``(count + 1) /
  (n_perm + 1)``) so reported p-values never hit 0 from a finite sample
  of permutations.
- Caller-supplied ``rng`` for reproducibility.

Pick the right test for the data you have:

- :func:`paired_permutation_pvalue` — when each observation in A has a
  natural match in B (e.g. (scene, point, seed) pairs across two models).
  Operates on per-pair differences via sign-flipping. Stronger than the
  unpaired test when the pairing is real, because it controls for shared
  per-cell noise.
- :func:`unpaired_permutation_pvalue` — when the two groups are
  independent samples with no cell-level correspondence (e.g. the legacy
  archive CSV across model pairs, where cells aren't scene-labeled).
  Operates on raw values via label shuffling.
"""

from __future__ import annotations

import numpy as np


def paired_permutation_pvalue(
    diffs: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided sign-flip permutation test on per-pair differences.

    ``diffs[i] = a[i] - b[i]`` for the ``i``-th matched pair. Each
    iteration multiplies the diffs by independent random ±1 signs and
    asks how often the resulting mean is at least as extreme as the
    observed mean. Returns NaN if there are zero diffs.
    """
    if len(diffs) == 0:
        return float("nan")
    obs = float(diffs.mean())
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((signs * diffs).mean()) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def unpaired_permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided two-sample label-shuffle permutation test on the difference of means.

    Pools ``x`` and ``y``, repeatedly partitions the pool back into two
    groups of the original sizes, and asks how often the resulting
    difference of means is at least as extreme as the observed
    ``mean(x) - mean(y)``.
    """
    obs = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    nx = len(x)
    n_total = len(pooled)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        diff = pooled[perm[:nx]].mean() - pooled[perm[nx:]].mean()
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)
