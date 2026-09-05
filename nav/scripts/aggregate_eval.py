"""CLI: aggregate per-run evaluations into a single xlsx + summary.

Walks a glob of run output directories, calls :func:`nav.eval.aggregate
.aggregate_runs` on the set, and writes detailed + per-scene-summary
sheets to an xlsx (or a CSV if openpyxl isn't installed).

Default output lives under ``analysis/aggregate_eval.xlsx`` — per
``docs/cleanup.md``, aggregate stats are git-tracked under ``analysis/``,
not under ``outputs/``. Pass ``--out`` to override.

Example::

    .venv/bin/python -m nav.scripts.aggregate_eval \\
        --input-glob "outputs/*/point*/*/seed*" \\
        --out analysis/grid_eval_summary.xlsx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nav.config import EVAL_AGGREGATE_DEFAULT_OUT
from nav.eval.aggregate import (
    aggregate_runs,
    write_aggregate_csv,
    write_aggregate_xlsx,
)
from nav.eval.metrics import EvaluateOptions
from nav.utils import logger_config


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-run evaluations into a summary xlsx."
    )
    p.add_argument(
        "--input-glob",
        type=str,
        required=True,
        help='Glob of run dirs (e.g. "outputs/*/point*/*/seed*").',
    )
    p.add_argument(
        "--out",
        type=str,
        default=EVAL_AGGREGATE_DEFAULT_OUT,
        help=f"Output xlsx path (default: {EVAL_AGGREGATE_DEFAULT_OUT}).",
    )
    p.add_argument(
        "--summary-axis",
        type=str,
        default="scene_name",
        help="Column to group the summary sheet by (default: scene_name).",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Directory for run log file (default: sibling 'logs/' next to --out).",
    )
    p.add_argument(
        "--warning-threshold",
        type=float,
        default=None,
        help=(
            "Base warning clearance in meters; positive move distance is added "
            "per frame (default: 0.4)."
        ),
    )
    p.add_argument(
        "--warning-min-pixel-ratio",
        type=float,
        default=None,
        help="Minimum near-depth fraction of the ROI (default: 0.005).",
    )
    p.add_argument(
        "--warning-eval-width",
        type=int,
        default=None,
        help=(
            "Optional compatibility resize width; native resolution is used "
            "by default."
        ),
    )
    p.add_argument(
        "--warning-eval-height",
        type=int,
        default=None,
        help=(
            "Optional compatibility resize height; native resolution is used "
            "by default."
        ),
    )
    p.add_argument(
        "--collision-min-forward-ratio",
        type=float,
        default=None,
        help=(
            "Count a positive move as a collision when observed/theoretical "
            "world displacement is below this ratio (default: 0.95)."
        ),
    )
    p.add_argument(
        "--success-dist-m",
        type=float,
        default=None,
        help="Success threshold in Unity world meters (default: 2.0).",
    )
    p.add_argument(
        "--no-use-actions",
        action="store_true",
        help="Evaluate every depth .npy, not just those that match an action step.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out)
    log_dir = args.log_dir or str(out_path.parent / "logs")
    logger_config(log_dir)

    opts = EvaluateOptions()
    if args.warning_threshold is not None:
        opts.warning_threshold_m = args.warning_threshold
    if args.warning_min_pixel_ratio is not None:
        opts.warning_min_pixel_ratio = args.warning_min_pixel_ratio
    if args.warning_eval_width is not None:
        opts.warning_image_width = args.warning_eval_width
    if args.warning_eval_height is not None:
        opts.warning_image_height = args.warning_eval_height
    if args.collision_min_forward_ratio is not None:
        opts.collision_min_forward_ratio = args.collision_min_forward_ratio
    if args.success_dist_m is not None:
        opts.success_dist_m = args.success_dist_m
    if args.no_use_actions:
        opts.use_actions = False

    input_dirs = sorted(Path(".").glob(args.input_glob))
    logger.info(f"Aggregating {len(input_dirs)} run dirs (glob: {args.input_glob!r})")
    rows = aggregate_runs(input_dirs, opts=opts)

    if write_aggregate_xlsx(out_path, rows, summary_axis=args.summary_axis):
        logger.info(f"Wrote xlsx with {len(rows)} rows -> {out_path.resolve()}")
    else:
        csv_path = out_path.with_suffix(".csv")
        write_aggregate_csv(csv_path, rows)
        logger.info(
            f"openpyxl not installed — wrote CSV with {len(rows)} rows -> "
            f"{csv_path.resolve()}"
        )


if __name__ == "__main__":
    main()
