"""Collision-rate detection from agent trajectories.

A collision is operationally defined as a ``forward`` action that resulted
in negligible pixel-space displacement. Two input paths are supported:

- :func:`compute_collision_rate` — the canonical workflow: reads the
  per-frame ``*_actions.csv`` produced by ``run_headless_benchmark.py``.
  Each row carries the chosen action plus pre-action ``(curr_px, curr_py)``;
  consecutive rows give the displacement.
- :func:`parse_log_file_collisions` — legacy: parses ``*.log`` files where
  the same trajectory was emitted as regex-matchable lines. Kept for
  backward compatibility with archived run dirs that no longer carry the
  actions CSV. Walked over a (scene, point, model) tree by
  :func:`walk_logs_for_collisions`.

Both implementations share the same definition of "collision" (Manhattan
pixel-distance below :data:`nav.config.EVAL_COLLISION_PX_THRESH`).
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nav.config import EVAL_COLLISION_PX_THRESH


# ---------------------------------------------------------------------------
# Canonical: actions CSV (the path used by stats_analysis_full / eval_run)
# ---------------------------------------------------------------------------

def compute_collision_rate(
    actions_csv: Path,
    collision_px_thresh: int = EVAL_COLLISION_PX_THRESH,
) -> Tuple[int, int, float]:
    """Compute the collision rate from one run's actions CSV.

    Returns ``(total_forward_steps, total_collision_steps, collision_rate)``.
    ``collision_rate`` is 0.0 when there were no forward steps (avoids
    divide-by-zero; the caller can decide whether NaN would be more
    appropriate for downstream aggregation).
    """
    rows: List[dict] = []
    with open(actions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step_num = int(float(row["step"]))
            except (ValueError, KeyError, TypeError):
                continue
            action = (row.get("action") or "").strip()
            pos: Optional[Tuple[int, int]] = None
            try:
                pos = (int(float(row["curr_px"])), int(float(row["curr_py"])))
            except (ValueError, KeyError, TypeError):
                pos = None
            rows.append({"step": step_num, "action": action, "pos": pos})

    rows.sort(key=lambda r: r["step"])
    total_forward_steps = 0
    total_collision_steps = 0
    for i in range(len(rows) - 1):
        cur, nxt = rows[i], rows[i + 1]
        if cur["action"] == "forward" and cur["pos"] and nxt["pos"]:
            total_forward_steps += 1
            pixel_change = (
                abs(nxt["pos"][0] - cur["pos"][0])
                + abs(nxt["pos"][1] - cur["pos"][1])
            )
            if pixel_change < collision_px_thresh:
                total_collision_steps += 1

    rate = (
        total_collision_steps / total_forward_steps
        if total_forward_steps > 0
        else 0.0
    )
    return total_forward_steps, total_collision_steps, rate


# ---------------------------------------------------------------------------
# Legacy: log-file regex parser
# ---------------------------------------------------------------------------

_RE_UNITY_POS = re.compile(
    r"\[Step:1\] Position: X=([-\d\.]+), Y=([-\d\.]+), Z=([-\d\.]+)"
)
_RE_MINIMAP_STATE = re.compile(
    r"\[Step:(\d+)\] STATE -> Pos:\((\d+),(\d+)\), .* Target:\((\d+),(\d+)\)"
)
_RE_LLM_ACTION = re.compile(
    r"\[Step:(\d+)\] LLM Action: (forward|turn right|turn left|stop)"
)


def parse_log_file_collisions(
    log_filepath: Path,
    collision_px_thresh: int = EVAL_COLLISION_PX_THRESH,
) -> Optional[dict]:
    """Parse a run's ``.log`` file and compute collision metrics.

    Used when the actions CSV isn't available (older archived runs).
    Returns a dict with positions, totals, and rate, or None if the file
    can't be opened.
    """
    start_pos_unity: Optional[str] = None
    start_pos_minimap: Optional[str] = None
    target_pos_minimap: Optional[str] = None
    step_data: Dict[int, dict] = {}

    try:
        with open(log_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if start_pos_unity is None:
                    m = _RE_UNITY_POS.search(line)
                    if m:
                        start_pos_unity = f"({m.group(1)}, {m.group(2)}, {m.group(3)})"
                        continue

                m = _RE_MINIMAP_STATE.search(line)
                if m:
                    step_num = int(m.group(1))
                    pos = (int(m.group(2)), int(m.group(3)))
                    step_data.setdefault(step_num, {})["pos"] = pos
                    if start_pos_minimap is None:
                        start_pos_minimap = f"({pos[0]}, {pos[1]})"
                        target_pos_minimap = f"({m.group(4)}, {m.group(5)})"
                    continue

                m = _RE_LLM_ACTION.search(line)
                if m:
                    step_num = int(m.group(1))
                    step_data.setdefault(step_num, {})["action"] = m.group(2)
    except OSError:
        return None

    total_forward = total_collision = 0
    sorted_steps = sorted(step_data.keys())
    for i in range(len(sorted_steps) - 1):
        cur = step_data.get(sorted_steps[i], {})
        nxt = step_data.get(sorted_steps[i + 1], {})
        if cur.get("action") == "forward" and "pos" in cur and "pos" in nxt:
            total_forward += 1
            pixel_change = (
                abs(nxt["pos"][0] - cur["pos"][0])
                + abs(nxt["pos"][1] - cur["pos"][1])
            )
            if pixel_change < collision_px_thresh:
                total_collision += 1

    rate = total_collision / total_forward if total_forward > 0 else 0.0
    return {
        "Start_Position_Unity (X,Y,Z)": start_pos_unity,
        "Start_Position_Minimap (X,Y)": start_pos_minimap,
        "Target_Position_Minimap (X,Y)": target_pos_minimap,
        "Total_Forward_Steps": total_forward,
        "Total_Collision_Steps": total_collision,
        "Collision_Rate": f"{rate:.2%}",
    }


_LOG_FIELDNAMES = [
    "Scene", "Point_Pair", "Model_Name",
    "Start_Position_Unity (X,Y,Z)",
    "Start_Position_Minimap (X,Y)",
    "Target_Position_Minimap (X,Y)",
    "Total_Forward_Steps", "Total_Collision_Steps", "Collision_Rate",
]


def walk_logs_for_collisions(
    benchmark_root: Path,
    output_dir: Path,
    *,
    collision_px_thresh: int = EVAL_COLLISION_PX_THRESH,
) -> Dict[str, List[dict]]:
    """Walk ``benchmark_root/<scene>/<point>/<model>/*.log`` and write per-scene CSVs.

    Returns the ``{scene_name: [row, ...]}`` map for in-memory consumers.
    Mirrors the legacy ``collision_analyzer.py`` walker but with
    parameterized roots (no hardcoded developer paths) and the canonical
    collision threshold.
    """
    benchmark_root = Path(benchmark_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_rows: Dict[str, List[dict]] = {}
    for root, _dirs, files in os.walk(benchmark_root):
        for filename in files:
            if not filename.endswith(".log"):
                continue
            log_path = Path(root) / filename
            relative = log_path.parent.relative_to(benchmark_root).parts
            if len(relative) != 3:
                continue
            scene, point_pair, model_name = relative
            row = parse_log_file_collisions(log_path, collision_px_thresh=collision_px_thresh)
            if row is None:
                continue
            row.update({"Scene": scene, "Point_Pair": point_pair, "Model_Name": model_name})
            scene_rows.setdefault(scene, []).append(row)

    for scene_name, rows in scene_rows.items():
        csv_path = output_dir / f"{scene_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return scene_rows
