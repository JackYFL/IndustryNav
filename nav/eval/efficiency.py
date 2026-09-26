"""Step-efficiency metrics normalized by an action-aware A* reference."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

from nav.config import ACTION_SPACE_AGENTS


DEFAULT_ASTAR_RESULTS_DIR = "astar_static_all_points"
DEFAULT_EFFICIENCY_STEP_MARGIN = 100
TaskKey = Tuple[str, str]


def _finite_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class AStarStepReference:
    """A* task cost in both native steps and continuous control effort."""

    astar_steps: int
    forward_signal_ticks: float
    turn_signal_ticks: float
    stop_steps: int = 1

    def agent_steps(
        self,
        *,
        forward_signal: float = ACTION_SPACE_AGENTS["forward"],
        turn_signal: float = abs(ACTION_SPACE_AGENTS["turn right"]),
        sim_steps_per_decision: int = 2,
    ) -> int:
        """Convert the A* trajectory to equivalent atomic agent steps."""
        forward_per_step = float(forward_signal) * int(sim_steps_per_decision)
        turn_per_step = float(turn_signal) * int(sim_steps_per_decision)
        if forward_per_step <= 0 or turn_per_step <= 0:
            raise ValueError("agent action signals and simulation steps must be positive")
        forward_steps = math.ceil(self.forward_signal_ticks / forward_per_step)
        turn_steps = math.ceil(self.turn_signal_ticks / turn_per_step)
        return int(forward_steps + turn_steps + self.stop_steps)


def compute_step_efficiency(
    actual_steps,
    optimal_steps,
    max_steps,
    success=1,
) -> float:
    """Return success-weighted, inverse min-max normalized step efficiency."""
    actual = _finite_float(actual_steps)
    optimal = _finite_float(optimal_steps)
    maximum = _finite_float(max_steps)
    success_weight = _finite_float(success)
    if None in (actual, optimal, maximum, success_weight):
        return float("nan")
    if actual < 0 or optimal < 0 or maximum <= optimal:
        return float("nan")
    success_weight = min(1.0, max(0.0, success_weight))
    normalized = (maximum - actual) / (maximum - optimal)
    return success_weight * min(1.0, max(0.0, normalized))


def compute_efficiency_step_budget_max(
    optimal_steps,
    step_margin=DEFAULT_EFFICIENCY_STEP_MARGIN,
) -> float:
    """Return the per-task normalization ceiling ``optimal_steps + K``."""
    optimal = _finite_float(optimal_steps)
    margin = _finite_float(step_margin)
    if margin is None or margin <= 0:
        raise ValueError("efficiency step margin K must be positive")
    if optimal is None or optimal < 0:
        return float("nan")
    return optimal + margin


def load_astar_step_references(
    outputs_root: Path,
    results_dir_name: str = DEFAULT_ASTAR_RESULTS_DIR,
) -> dict[TaskKey, AStarStepReference]:
    """Load successful A* trajectories keyed by ``(scene, point)``."""
    outputs_root = Path(outputs_root)
    references: dict[TaskKey, AStarStepReference] = {}
    pattern = f"scene*/point*/{results_dir_name}/results.csv"
    for results_csv in sorted(outputs_root.glob(pattern)):
        try:
            with results_csv.open(newline="") as stream:
                result_rows = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
        if not result_rows:
            continue
        result = result_rows[-1]
        distance = _finite_float(result.get("distance_world"))
        reach_m = _finite_float(result.get("reach_m"))
        steps = _finite_float(result.get("steps_taken"))
        sim_steps = _finite_float(result.get("sim_steps_per_decision")) or 1.0
        if (
            distance is None
            or reach_m is None
            or steps is None
            or distance > reach_m
            or steps < 0
            or sim_steps <= 0
        ):
            continue

        actions_csv = results_csv.parent / "astar_actions.csv"
        try:
            with actions_csv.open(newline="") as stream:
                action_rows = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
        if not action_rows:
            continue

        forward_signal_ticks = 0.0
        turn_signal_ticks = 0.0
        stop_steps = 0
        for row in action_rows:
            action_step = _finite_float(row.get("step"))
            if action_step is not None and action_step > steps:
                continue
            move = _finite_float(row.get("move")) or 0.0
            look = _finite_float(row.get("look")) or 0.0
            forward_signal_ticks += abs(move) * sim_steps
            turn_signal_ticks += abs(look) * sim_steps
            stop_steps += int(str(row.get("action", "")).strip().lower() == "stop")

        scene = result.get("scene_name") or results_csv.parts[-4]
        point = result.get("point_id") or results_csv.parts[-3]
        key = (str(scene), str(point))
        reference = AStarStepReference(
            astar_steps=int(round(steps)),
            forward_signal_ticks=forward_signal_ticks,
            turn_signal_ticks=turn_signal_ticks,
            stop_steps=max(1, stop_steps),
        )
        previous = references.get(key)
        if previous is None or reference.astar_steps < previous.astar_steps:
            references[key] = reference
    return references


def _uses_native_astar_steps(row: Mapping) -> bool:
    labels = (
        row.get("exec_mode", ""),
        row.get("model", ""),
        row.get("model_short", ""),
    )
    return any(str(label).strip().lower().startswith("astar") for label in labels)


def attach_step_efficiency(
    rows: Iterable[Mapping],
    astar_references: Mapping[TaskKey, AStarStepReference],
    *,
    step_margin: int = DEFAULT_EFFICIENCY_STEP_MARGIN,
) -> list[dict]:
    """Attach action-aware optima and ``max_steps = optimal_steps + K``."""
    augmented: list[dict] = []
    for source in rows:
        row = dict(source)
        key = (str(row.get("scene_name", "")), str(row.get("point_id", "")))
        reference = astar_references.get(key)
        if reference is None:
            optimal = None
        elif _uses_native_astar_steps(row):
            optimal = reference.astar_steps
        else:
            sim_steps = _finite_float(row.get("sim_steps_per_decision")) or 2.0
            optimal = reference.agent_steps(
                sim_steps_per_decision=max(1, int(round(sim_steps)))
            )

        maximum = compute_efficiency_step_budget_max(optimal, step_margin)
        row["optimal_steps"] = optimal if optimal is not None else float("nan")
        row["efficiency_max_steps"] = maximum
        row["efficiency"] = compute_step_efficiency(
            row.get("steps_taken"),
            optimal,
            maximum,
            row.get("success", row.get("success_ratio")),
        )
        augmented.append(row)
    return augmented
