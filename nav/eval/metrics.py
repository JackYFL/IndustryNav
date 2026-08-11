"""Per-run evaluation orchestrator.

:func:`evaluate_run` reads one finished run's output directory and returns
the full bundle of metrics (success / efficiency / distance ratio, warning
rate, collision rate, plus context from ``results.csv``). It composes
:mod:`nav.eval.warning` and :mod:`nav.eval.collision` and is the canonical
entry point that ``stats_analysis_full`` and the rebuttal pipeline rely on.

The two helpers :func:`compute_success_efficiency_distance` and
:func:`compute_collision_rate` (re-exported from :mod:`nav.eval.collision`)
are kept at module scope without the leading-underscore convention used
in the legacy ``eval_metrics.py`` — they are de-facto public API and the
underscore was misleading.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from nav.config import (
    EVAL_COLLISION_PX_THRESH,
    EVAL_ROI_PARAMS,
    EVAL_SUCCESS_DIST_M,
    EVAL_WARNING_THRESHOLD_M,
)
from nav.eval.collision import compute_collision_rate
from nav.eval.io import (
    find_actions_csv,
    find_depth_dir,
    read_latest_results_row,
)
from nav.eval.warning import WarningDetector, compute_warning_rate


@dataclass
class EvaluateOptions:
    """Per-run evaluation knobs. Defaults pulled from :mod:`nav.config`."""

    warning_threshold_m: float = EVAL_WARNING_THRESHOLD_M
    collision_threshold_px: int = EVAL_COLLISION_PX_THRESH
    success_dist_m: float = EVAL_SUCCESS_DIST_M
    bottom_margin: float = EVAL_ROI_PARAMS["bottom_margin"]
    top_margin: float = EVAL_ROI_PARAMS["top_margin"]
    bottom_pad: float = EVAL_ROI_PARAMS["bottom_pad"]
    top_pad: float = EVAL_ROI_PARAMS["top_pad"]
    use_actions: bool = True

    def roi_params(self) -> dict:
        return {
            "bottom_margin": self.bottom_margin,
            "top_margin": self.top_margin,
            "bottom_pad": self.bottom_pad,
            "top_pad": self.top_pad,
        }


def compute_success_efficiency_distance(
    actions_csv: Path,
    success_dist_m: float = EVAL_SUCCESS_DIST_M,
) -> Tuple[int, int, float]:
    """Compute (success, efficiency_steps, distance_ratio) from an actions CSV.

    - ``success`` ∈ {0, 1}: 1 if the final world-distance is at or below
      ``success_dist_m``. Rows without a valid world distance are ignored.
    - ``efficiency_steps``: the number of step rows in the CSV (a proxy for
      time-to-target; lower is better when ``success == 1``).
    - ``distance_ratio``: ``|start_dist - final_dist| / start_dist``,
      clamped to ``1.0`` on success so the leaderboard isn't punished for
      tiny terminal jitter; ``+inf`` if the agent started on top of the
      target (degenerate run).

    Empty/unreadable CSV → ``(0, 0, 0.0)``.
    """
    rows = []
    with open(actions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step_num = int(float(row["step"]))
                distance_world = float(row["distance_world"])
            except (ValueError, KeyError, TypeError):
                distance_world = None
            if distance_world is None or not math.isfinite(distance_world):
                continue
            rows.append({
                "step": step_num,
                "distance_world": distance_world,
            })

    if not rows:
        return 0, 0, 0.0

    rows.sort(key=lambda r: r["step"])
    start_dist = float(rows[0]["distance_world"])
    final_dist = float(rows[-1]["distance_world"])
    efficiency_steps = len(rows)

    success = 1 if final_dist <= success_dist_m else 0

    if start_dist > 0:
        distance_ratio = math.fabs(start_dist - final_dist) / start_dist
    else:
        distance_ratio = 1.0 if final_dist == 0 else float("inf")
    if success == 1:
        distance_ratio = 1.0

    return success, efficiency_steps, distance_ratio


def evaluate_run(
    input_dir: Path,
    opts: Optional[EvaluateOptions] = None,
) -> dict:
    """Compute the full metric bundle for one run's output directory.

    Raises ``FileNotFoundError`` if the depth directory or actions CSV is
    missing — the caller is expected to skip or surface the error rather
    than silently returning empty metrics (callers like ``aggregate.py``
    catch and convert to an error row).
    """
    input_dir = Path(input_dir)
    opts = opts or EvaluateOptions()

    depth_dir = find_depth_dir(input_dir)
    if depth_dir is None:
        raise FileNotFoundError(f"Depth directory not found under: {input_dir}")

    actions_csv = find_actions_csv(input_dir)
    if actions_csv is None:
        raise FileNotFoundError(f"No actions CSV found in {input_dir}")

    detector = WarningDetector(
        warning_threshold_m=opts.warning_threshold_m,
        roi_params=opts.roi_params(),
    )
    total_steps, warning_steps, warning_rate = compute_warning_rate(
        input_dir, detector=detector, use_actions=opts.use_actions,
    )
    forward_steps, collision_steps, collision_rate = compute_collision_rate(
        actions_csv, collision_px_thresh=opts.collision_threshold_px,
    )
    success, efficiency_steps, distance_ratio = compute_success_efficiency_distance(
        actions_csv,
        success_dist_m=opts.success_dist_m,
    )

    # results.csv carries the run's final metadata. When present, override
    # the CSV-derived final distance + steps_taken with those authoritative
    # values (the CSV may include extra logged-but-not-acted-on frames).
    results_row = read_latest_results_row(input_dir / "results.csv")
    final_distance_px: Optional[float] = None
    final_distance_world: Optional[float] = None
    stop_reason = ""
    steps_taken: Optional[int] = None
    if results_row:
        try:
            final_distance_px = float(results_row.get("distance_px", ""))
        except (ValueError, TypeError):
            final_distance_px = None
        try:
            final_distance_world = float(results_row.get("distance_world", ""))
        except (ValueError, TypeError):
            final_distance_world = None
        stop_reason = results_row.get("stop_reason", "")
        try:
            steps_taken = int(float(results_row.get("steps_taken", "")))
        except (ValueError, TypeError):
            steps_taken = None

    if final_distance_world is not None:
        success = 1 if final_distance_world <= opts.success_dist_m else 0
    else:
        success = 0
    if steps_taken is not None:
        efficiency_steps = steps_taken

    return {
        "input_dir": str(input_dir),
        "depth_dir": depth_dir.name,
        "actions_csv": actions_csv.name,
        "total_steps": total_steps,
        "success_ratio": success,
        "efficiency_steps": efficiency_steps,
        "distance_ratio": distance_ratio,
        "final_distance_world": final_distance_world,
        "final_distance_px": final_distance_px,
        "stop_reason": stop_reason,
        "warning_steps": warning_steps,
        "warning_rate": warning_rate,
        "forward_steps": forward_steps,
        "collision_steps": collision_steps,
        "collision_rate": collision_rate,
    }
