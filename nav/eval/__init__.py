"""Post-experiment evaluation utilities.

Submodules
----------

- :mod:`nav.eval.io` — shared loaders for per-run output directories.
- :mod:`nav.eval.base` — reusable metric lifecycle and streaming accumulators.
- :mod:`nav.eval.warning` — warning-rate aggregation over run artifacts.
- :mod:`nav.eval.collision` — collision-rate aggregation over action CSVs.
- :mod:`nav.eval.metrics` — per-run orchestrator combining the above.
- :mod:`nav.eval.aggregate` — batch aggregation across many runs into a summary.

Each function reads from the per-run output directory (``outputs/<scene>/
<point>/<model>/seed<k>/``) produced by ``run_headless_benchmark.py``.

Online detector primitives live in :mod:`nav.safety` so environments and
visualization code can use the same definitions without importing evaluation.
"""

from nav.eval.base import BaseEvaluator, BaseMetric, BinaryRateMetric, RateResult

__all__ = ["BaseEvaluator", "BaseMetric", "BinaryRateMetric", "RateResult"]
