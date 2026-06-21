"""Spearman rank correlation, numpy-only (no scipy dependency).

Used for leaderboard correlation across two independent point subsets —
how well a per-model ranking on points {1,2} predicts the ranking on
points {3,4}. Degenerate at N=2 models (only ±1 possible); callers
should return NaN with a note rather than report a misleading rank-1
correlation.
"""

from __future__ import annotations

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank assignment with tie handling.

    Equivalent to ``scipy.stats.rankdata(a, method="average")``. Used as
    the building block of :func:`spearman_corr` but exported so callers
    that want the ranks themselves don't need to reimplement them.
    """
    a = np.asarray(a)
    sorter = np.argsort(a, kind="stable")
    sorted_a = a[sorter]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1  # 1-based mid-rank
        ranks[sorter[i:j]] = avg
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two arrays.

    Returns NaN when ``len(x) < 2``, when the lengths don't match, or
    when either rank vector has zero variance (degenerate ranking).
    """
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    rx = rankdata(np.asarray(x))
    ry = rankdata(np.asarray(y))
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    denom = np.sqrt((rxc ** 2).sum() * (ryc ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rxc * ryc).sum() / denom)
