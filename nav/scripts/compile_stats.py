"""CLI: drive the four-source rebuttal stats pipeline.

Subcommands map 1:1 to what the legacy four-script incantation in
``.agents/skills/compile_stats/SKILL.md`` did, but go through the
deduped library code in :mod:`nav.stats`:

- ``grid``    — grid output tree → full analysis report.
- ``xlsx``    — xlsx single-run sweep → per_run.csv (no stats here, just adaptation).
- ``per-run`` — pre-built per_run.csv → full analysis report (used after
                ``xlsx`` or ``merge`` to re-run the stats on a different source).
- ``merge``   — supersession-merge a grid + xlsx pair of per_run.csv files.
- ``partial`` — legacy experiment_results_v6.csv → partial analysis report.
- ``all``     — run grid → xlsx → per-run(xlsx) → merge → per-run(merged) in
                sequence, mirroring the old SKILL.md's six-step workflow.

Outputs land under ``analysis/<subdir>/`` (default subdir from
:data:`nav.config.DEFAULT_ANALYSIS_SUBDIR`). The merge writes its
provenance trace to ``analysis/<subdir>/merged/per_run_provenance.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from nav.config import (
    ANALYSIS_ROOT,
    DEFAULT_ANALYSIS_SUBDIR,
)
from nav.stats import (
    full as stats_full,
    load as stats_load,
    merge as stats_merge,
    partial as stats_partial,
)
from nav.utils import logger_config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand argument parsers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drive the rebuttal stats pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--log-dir", type=str, default="",
        help="Directory for the run log file (default: analysis/<subdir>/logs).",
    )
    subs = p.add_subparsers(dest="cmd", required=True)

    common_full = dict(
        n_perm=10_000, n_boot=10_000, seed=0, spearman_split="halves",
    )

    def _add_common_full(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--analysis-subdir", default=DEFAULT_ANALYSIS_SUBDIR)
        sp.add_argument("--n-perm", type=int, default=common_full["n_perm"])
        sp.add_argument("--n-boot", type=int, default=common_full["n_boot"])
        sp.add_argument("--seed", type=int, default=common_full["seed"])
        sp.add_argument("--spearman-split", default=common_full["spearman_split"],
                        choices=["halves", "odd_even"])
        sp.add_argument(
            "--vision-input", choices=["on", "off", "both"], default="both",
        )
        sp.add_argument("--models", nargs="+", default=None,
                        help="Restrict to these model ids (default: all).")

    # grid: walk outputs/ tree → full report
    sp = subs.add_parser("grid", help="Run full analysis on the grid output tree.")
    sp.add_argument("--outputs-root", default="outputs",
                    help="Grid output tree root (default: outputs).")
    _add_common_full(sp)

    # xlsx: parse xlsx → per_run.csv (no stats here, just the adapter)
    sp = subs.add_parser("xlsx", help="Convert the xlsx sweep to per_run.csv.")
    sp.add_argument("--xlsx", required=True, help="Path to the xlsx sweep.")
    sp.add_argument("--out", required=True, help="Output per_run.csv path.")

    # per-run: full analysis from a pre-built per_run.csv
    sp = subs.add_parser("per-run", help="Run full analysis on a per_run.csv.")
    sp.add_argument("--per-run-csv", required=True,
                    help="Pre-built per_run.csv (e.g. from xlsx or merge).")
    _add_common_full(sp)

    # merge: supersession of grid + xlsx
    sp = subs.add_parser("merge", help="Supersession-merge a grid per_run.csv with xlsx per_run.csv.")
    sp.add_argument("--grid-per-run", required=True)
    sp.add_argument("--xlsx-per-run", required=True)
    sp.add_argument("--out", required=True, help="Output merged per_run.csv path.")
    sp.add_argument("--grid-vision", choices=["on", "off"], default="on")
    sp.add_argument("--prefer", choices=["xlsx", "grid"], default="xlsx")

    # partial: experiment_results_v6.csv → partial report
    sp = subs.add_parser("partial", help="Run partial analysis on a label-less archive CSV.")
    sp.add_argument("--csv", default="experiment_results_v6.csv")
    sp.add_argument("--analysis-subdir", default=DEFAULT_ANALYSIS_SUBDIR)
    sp.add_argument("--min-n", type=int, default=10)
    sp.add_argument("--n-perm", type=int, default=common_full["n_perm"])
    sp.add_argument("--n-boot", type=int, default=common_full["n_boot"])
    sp.add_argument("--seed", type=int, default=common_full["seed"])
    sp.add_argument("--success-stop-reason", default="reached_vicinity")

    # all: chain grid → xlsx → per-run(xlsx) → merge → per-run(merged)
    sp = subs.add_parser("all", help="Run the six-step compile_stats pipeline end to end.")
    sp.add_argument("--outputs-root", default="outputs")
    sp.add_argument("--xlsx", required=True,
                    help="Path to IndustryNav_extra_agents_results.xlsx.")
    _add_common_full(sp)
    sp.add_argument("--prefer", choices=["xlsx", "grid"], default="xlsx")
    sp.add_argument("--grid-vision", choices=["on", "off"], default="on")

    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _vision_set(name: str) -> set:
    return {"on": {"on"}, "off": {"off"}, "both": {"on", "off"}}[name]


def _filter_rows(rows, models_filter, vision_input):
    """Apply --models / --vision-input filters to a row list."""
    vf = _vision_set(vision_input)
    if models_filter:
        rows = [r for r in rows if r["model"] in models_filter]
    return [
        r for r in rows
        if (r["vision_input"] and "on" in vf) or (not r["vision_input"] and "off" in vf)
    ]


def _cmd_grid(args: argparse.Namespace) -> None:
    out_dir = ANALYSIS_ROOT / args.analysis_subdir
    models_filter = set(args.models) if args.models else None
    rows = stats_load.discover_grid_runs(
        Path(args.outputs_root),
        models_filter=models_filter,
        vision_filter=_vision_set(args.vision_input),
    )
    if not rows:
        raise SystemExit(
            f"No runs found under {args.outputs_root} matching the filters. "
            "Has the grid runner produced any cells yet?"
        )
    stats_full.run_full_analysis(
        rows, out_dir,
        n_perm=args.n_perm, n_boot=args.n_boot,
        seed=args.seed, spearman_split=args.spearman_split,
        source_label=f"{args.outputs_root}/",
    )


def _cmd_xlsx(args: argparse.Namespace) -> None:
    rows = stats_load.xlsx_to_per_run_rows(Path(args.xlsx))
    if not rows:
        raise SystemExit(f"No usable rows parsed from {args.xlsx}")
    stats_load.write_per_run_csv(rows, Path(args.out))
    logger.info(f"[xlsx] wrote {len(rows)} rows to {args.out}")
    logger.info(stats_load.summarize_coverage(rows))


def _cmd_per_run(args: argparse.Namespace) -> None:
    out_dir = ANALYSIS_ROOT / args.analysis_subdir
    rows = stats_load.load_per_run_csv(Path(args.per_run_csv))
    models_filter = set(args.models) if args.models else None
    rows = _filter_rows(rows, models_filter, args.vision_input)
    if not rows:
        raise SystemExit("Filters yielded zero rows; check --models / --vision-input.")
    stats_full.run_full_analysis(
        rows, out_dir,
        n_perm=args.n_perm, n_boot=args.n_boot,
        seed=args.seed, spearman_split=args.spearman_split,
        source_label=args.per_run_csv,
    )


def _cmd_merge(args: argparse.Namespace) -> None:
    grid_rows = stats_load.load_per_run_csv(Path(args.grid_per_run))
    xlsx_rows = stats_load.load_per_run_csv(Path(args.xlsx_per_run))
    grid_collapsed = stats_merge.collapse_grid_to_cells(
        grid_rows, vision_filter=args.grid_vision
    )
    xlsx_indexed = stats_merge.index_by_cell(xlsx_rows)
    merged = stats_merge.merge(grid_collapsed, xlsx_indexed, prefer=args.prefer)
    out_path, prov_path = stats_merge.write_merged_outputs(merged, Path(args.out))
    logger.info(f"[merge] wrote {len(merged)} merged cells to {out_path}")
    logger.info(f"[merge] provenance trace: {prov_path}")


def _cmd_partial(args: argparse.Namespace) -> None:
    out_dir = ANALYSIS_ROOT / args.analysis_subdir / "partial"
    by_model = stats_load.load_experiment_v6_by_model(
        Path(args.csv), success_stop_reason=args.success_stop_reason
    )
    if not by_model:
        raise SystemExit(f"No usable rows parsed from {args.csv}")
    stats_partial.run_partial_analysis(
        by_model, out_dir,
        min_n=args.min_n, n_perm=args.n_perm, n_boot=args.n_boot,
        seed=args.seed, source_label=args.csv,
    )


def _cmd_all(args: argparse.Namespace) -> None:
    subdir_root = ANALYSIS_ROOT / args.analysis_subdir
    out_grid = subdir_root
    out_xlsx = subdir_root / "before_rebuttal"
    out_merged = subdir_root / "merged"

    # Stage 1: grid → full report
    logger.info("[all] stage 1/5: grid → full analysis")
    _cmd_grid(args)

    # Stage 2: xlsx → per_run.csv
    logger.info("[all] stage 2/5: xlsx → per_run.csv")
    xlsx_per_run = out_xlsx / "per_run.csv"
    xlsx_rows = stats_load.xlsx_to_per_run_rows(Path(args.xlsx))
    stats_load.write_per_run_csv(xlsx_rows, xlsx_per_run)
    logger.info(f"[all] wrote {len(xlsx_rows)} rows to {xlsx_per_run}")

    # Stage 3: xlsx per_run → full report (pre-supersession reference)
    logger.info("[all] stage 3/5: xlsx per_run → full analysis (before_rebuttal)")
    stats_full.run_full_analysis(
        stats_load.load_per_run_csv(xlsx_per_run),
        out_xlsx,
        n_perm=args.n_perm, n_boot=args.n_boot,
        seed=args.seed, spearman_split=args.spearman_split,
        source_label=str(xlsx_per_run),
    )

    # Stage 4: merge grid + xlsx → merged per_run.csv
    logger.info("[all] stage 4/5: merge grid + xlsx")
    grid_per_run = subdir_root / "per_run.csv"
    merge_args = argparse.Namespace(
        grid_per_run=str(grid_per_run),
        xlsx_per_run=str(xlsx_per_run),
        out=str(out_merged / "per_run.csv"),
        grid_vision=args.grid_vision,
        prefer=args.prefer,
    )
    _cmd_merge(merge_args)

    # Stage 5: merged per_run → full report (HEADLINE)
    logger.info("[all] stage 5/5: merged per_run → full analysis (HEADLINE)")
    stats_full.run_full_analysis(
        stats_load.load_per_run_csv(out_merged / "per_run.csv"),
        out_merged,
        n_perm=args.n_perm, n_boot=args.n_boot,
        seed=args.seed, spearman_split=args.spearman_split,
        source_label=str(out_merged / "per_run.csv"),
    )

    logger.info(f"[all] done. Headline report: {out_merged / 'report.md'}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_HANDLERS = {
    "grid": _cmd_grid,
    "xlsx": _cmd_xlsx,
    "per-run": _cmd_per_run,
    "merge": _cmd_merge,
    "partial": _cmd_partial,
    "all": _cmd_all,
}


def main() -> None:
    args = _build_parser().parse_args()

    # Resolve a sensible default log_dir if not provided. For "xlsx" and "merge"
    # we don't have an analysis_subdir argument, so fall back to a top-level
    # logs/stats directory.
    subdir = getattr(args, "analysis_subdir", None)
    log_dir = args.log_dir or (
        str(ANALYSIS_ROOT / subdir / "logs") if subdir else "logs/stats"
    )
    logger_config(log_dir)

    _HANDLERS[args.cmd](args)


if __name__ == "__main__":
    main()
