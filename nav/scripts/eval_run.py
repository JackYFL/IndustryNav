"""CLI: evaluate one run (or a glob of runs) and print/write metrics.

Single-dir mode prints the metric bundle to stdout. Glob mode walks the
matching directories and writes a flat summary CSV (one row per run).
Aggregation across runs into xlsx + per-axis summary lives in
:mod:`nav.scripts.aggregate_eval` — keep the two separate.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from nav.config import EVAL_DEFAULT_LOG_DIR
from nav.eval.aggregate import aggregate_runs, write_aggregate_csv
from nav.eval.metrics import SUCCESS_THRESHOLD_FIELDS, EvaluateOptions, evaluate_run
from nav.utils import logger_config


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate warning + collision + success metrics for a run."
    )
    p.add_argument("--input-dir", type=str, default="",
                   help="Single run output dir to evaluate.")
    p.add_argument("--input-glob", type=str, default="",
                   help="Glob of run dirs (e.g. \"outputs/*/point*/Astar\").")
    p.add_argument("--summary-csv", type=str, default="",
                   help="Glob mode: where to write the per-run summary CSV.")
    p.add_argument(
        "--log-dir",
        type=str,
        default="",
        help=(
            "Directory for the run log file. Default: input_dir for single-dir "
            f"mode, '{EVAL_DEFAULT_LOG_DIR}' for glob mode."
        ),
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
    p.add_argument("--bottom-margin", type=float, default=None)
    p.add_argument("--top-margin", type=float, default=None)
    p.add_argument("--bottom-pad", type=float, default=None)
    p.add_argument("--top-pad", type=float, default=None)
    p.add_argument(
        "--no-use-actions", action="store_true",
        help="Evaluate every depth .npy, not just those that match an action step.",
    )
    args = p.parse_args()
    if not args.input_dir and not args.input_glob:
        p.error("--input-dir or --input-glob is required")
    return args


def _opts_from_args(args: argparse.Namespace) -> EvaluateOptions:
    opts = EvaluateOptions()
    for arg_name, attr_name in [
        ("warning_threshold", "warning_threshold_m"),
        ("warning_min_pixel_ratio", "warning_min_pixel_ratio"),
        ("warning_eval_width", "warning_image_width"),
        ("warning_eval_height", "warning_image_height"),
        ("collision_min_forward_ratio", "collision_min_forward_ratio"),
        ("success_dist_m", "success_dist_m"),
        ("bottom_margin", "bottom_margin"),
        ("top_margin", "top_margin"),
        ("bottom_pad", "bottom_pad"),
        ("top_pad", "top_pad"),
    ]:
        val = getattr(args, arg_name)
        if val is not None:
            setattr(opts, attr_name, val)
    if args.no_use_actions:
        opts.use_actions = False
    return opts


def _log_metrics(metrics: dict) -> None:
    logger.info(f"Total steps: {metrics['total_steps']}")
    logger.info(f"Success ratio: {metrics['success_ratio']}")
    for threshold_m, field in SUCCESS_THRESHOLD_FIELDS:
        logger.info(f"Success@{threshold_m:g}m: {metrics[field]}")
    logger.info(f"Efficiency (total steps): {metrics['efficiency_steps']}")
    logger.info(f"Distance ratio: {metrics['distance_ratio']:.4f}")
    if metrics["final_distance_world"] is not None:
        logger.info(f"Final distance world: {metrics['final_distance_world']:.2f} m")
    if metrics["stop_reason"]:
        logger.info(f"Stop reason: {metrics['stop_reason']}")
    logger.info(f"Warning steps: {metrics['warning_steps']}")
    logger.info(f"Warning rate: {metrics['warning_rate']:.4f}")
    logger.info(f"Forward steps: {metrics['forward_steps']}")
    logger.info(f"Collision steps: {metrics['collision_steps']}")
    logger.info(f"Collision rate: {metrics['collision_rate']:.4f}")


def main() -> None:
    args = _parse_args()
    opts = _opts_from_args(args)

    log_dir = args.log_dir or (args.input_dir if args.input_dir else EVAL_DEFAULT_LOG_DIR)
    logger_config(log_dir)

    if args.input_glob:
        input_dirs = sorted(Path(".").glob(args.input_glob))
        rows = aggregate_runs(input_dirs, opts=opts)
        if args.summary_csv:
            write_aggregate_csv(Path(args.summary_csv), rows)

        ok_rows = [r for r in rows if not r.get("error")]
        error_rows = [r for r in rows if r.get("error")]
        logger.info(f"Discovered runs: {len(rows)}")
        logger.info(f"Evaluated runs: {len(ok_rows)}")
        if error_rows:
            logger.warning(f"Errored runs: {len(error_rows)}")
        if rows:
            success = sum(int(r["success_ratio"]) for r in ok_rows)
            denom = len(ok_rows) if ok_rows else 1
            logger.info(f"Success: {success}/{len(ok_rows)} ({success / denom:.4f})")
            for threshold_m, field in SUCCESS_THRESHOLD_FIELDS:
                successes = sum(int(r.get(field, 0)) for r in ok_rows)
                logger.info(
                    f"Success@{threshold_m:g}m: {successes}/{len(ok_rows)} "
                    f"({successes / denom:.4f})"
                )
        if ok_rows:
            logger.info(
                f"Average steps: "
                f"{np.mean([float(r['efficiency_steps']) for r in ok_rows]):.2f}"
            )
            world_dists = [
                float(r['final_distance_world'])
                for r in ok_rows if r['final_distance_world'] is not None
            ]
            if world_dists:
                logger.info(f"Average final distance world: {np.mean(world_dists):.2f} m")
            logger.info(
                f"Average warning rate: "
                f"{np.mean([float(r['warning_rate']) for r in ok_rows]):.4f}"
            )
            logger.info(
                f"Average collision rate: "
                f"{np.mean([float(r['collision_rate']) for r in ok_rows]):.4f}"
            )
        return

    metrics = evaluate_run(Path(args.input_dir), opts=opts)
    _log_metrics(metrics)


if __name__ == "__main__":
    main()
