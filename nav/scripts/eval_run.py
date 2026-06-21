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
from nav.eval.metrics import EvaluateOptions, evaluate_run
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
    p.add_argument("--warning-threshold", type=float, default=None)
    p.add_argument("--collision-threshold", type=int, default=None)
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
        ("collision_threshold", "collision_threshold_px"),
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
    logger.info(f"Efficiency (total steps): {metrics['efficiency_steps']}")
    logger.info(f"Distance ratio: {metrics['distance_ratio']:.4f}")
    if metrics["final_distance_px"] is not None:
        logger.info(f"Final distance px: {metrics['final_distance_px']:.2f}")
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
        if ok_rows:
            logger.info(
                f"Average steps: "
                f"{np.mean([float(r['efficiency_steps']) for r in ok_rows]):.2f}"
            )
            dists = [
                float(r['final_distance_px'])
                for r in ok_rows if r['final_distance_px'] is not None
            ]
            if dists:
                logger.info(f"Average final distance px: {np.mean(dists):.2f}")
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
