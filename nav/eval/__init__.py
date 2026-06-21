"""Post-experiment evaluation utilities.

Submodules
----------

- :mod:`nav.eval.io` — shared loaders for per-run output directories.
- :mod:`nav.eval.warning` — depth-based forward-collision warning detection.
- :mod:`nav.eval.collision` — pixel-displacement collision rate from action CSVs.
- :mod:`nav.eval.metrics` — per-run orchestrator combining the above.
- :mod:`nav.eval.aggregate` — batch aggregation across many runs into a summary.

Each function reads from the per-run output directory (``outputs/<scene>/
<point>/<model>/seed<k>/``) produced by ``run_headless_benchmark.py``.
"""
