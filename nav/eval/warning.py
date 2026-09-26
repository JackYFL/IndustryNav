"""Offline warning-rate aggregation over finished navigation runs.

The reusable per-frame detector lives in :mod:`nav.safety.warning`; it is
re-exported here so archived callers keep working.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nav.config import EVAL_ROI_PARAMS, EVAL_WARNING_THRESHOLD_M
from nav.eval.base import BinaryRateMetric
from nav.eval.io import (
    find_actions_csv,
    find_depth_dir,
    load_action_moves,
    load_action_steps,
    read_depth_npy,
)
from nav.safety.warning import WarningDetector


def compute_warning_rate(
    input_dir: Path,
    detector: Optional[WarningDetector] = None,
    *,
    use_actions: bool = True,
) -> Tuple[int, int, float]:
    """Return ``(total_steps, warning_steps, warning_rate)`` for one run."""
    depth_dir = find_depth_dir(input_dir)
    if depth_dir is None:
        return 0, 0, 0.0

    detector = detector or WarningDetector()
    actions_csv = find_actions_csv(input_dir)
    action_moves = load_action_moves(actions_csv) if actions_csv else {}
    step_filter: Optional[set] = None
    if use_actions and actions_csv is not None:
        step_filter = set(load_action_steps(actions_csv))

    metric = BinaryRateMetric()
    for depth_file in sorted(depth_dir.glob("*.npy")):
        try:
            step_id = int(depth_file.stem)
        except ValueError:
            step_id = None
        if step_filter is not None and step_id not in step_filter:
            continue
        depth = read_depth_npy(depth_file)
        if depth is None:
            continue
        verdict = detector.detect(
            depth,
            move_command=action_moves.get(step_id, 0.0),
        )
        metric.update(verdict["warning"] == "yes")
    result = metric.compute()
    return result.total, result.triggered, result.rate


_SCENE_FIELDNAMES = [
    "Scene",
    "Point_Pair",
    "Model_Name",
    "Total_Steps",
    "Total_Warning_Steps",
    "Warning_Rate",
]


def _process_model_dir(job: tuple) -> Optional[dict]:
    """Picklable worker used by :func:`run_benchmark_depth`."""
    scene_name, point_name, model_dir_str, warning_threshold_m, roi_params = job
    model_dir = Path(model_dir_str)
    depth_files = sorted(model_dir.glob("*.npy"))
    if not depth_files:
        return None
    detector = WarningDetector(
        warning_threshold_m=warning_threshold_m,
        roi_params=roi_params,
    )
    total = warning = 0
    for depth_file in depth_files:
        depth = read_depth_npy(depth_file)
        if depth is None or depth.size == 0 or not np.isfinite(depth).any():
            total += 1
            continue
        total += 1
        warning += int(detector.detect(depth)["warning"] == "yes")
    if total == 0:
        return None
    return {
        "Scene": scene_name,
        "Point_Pair": point_name,
        "Model_Name": model_dir.name,
        "Total_Steps": total,
        "Total_Warning_Steps": warning,
        "Warning_Rate": f"{warning / total * 100.0:.2f}%",
    }


def run_benchmark_depth(
    depth_root: Path,
    warning_root: Path,
    *,
    warning_threshold_m: float = EVAL_WARNING_THRESHOLD_M,
    roi_params: Optional[Dict[str, float]] = None,
    max_workers: int = 1,
) -> Dict[str, List[dict]]:
    """Evaluate a ``scene/point/model`` tree and write one CSV per scene."""
    depth_root = Path(depth_root)
    warning_root = Path(warning_root)
    warning_root.mkdir(parents=True, exist_ok=True)
    roi_params = dict(roi_params) if roi_params is not None else dict(EVAL_ROI_PARAMS)

    jobs = []
    for scene_dir in sorted(depth_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for point_dir in sorted(scene_dir.iterdir()):
            if not point_dir.is_dir():
                continue
            for model_dir in sorted(point_dir.iterdir()):
                if model_dir.is_dir() and any(model_dir.glob("*.npy")):
                    jobs.append(
                        (
                            scene_dir.name,
                            point_dir.name,
                            str(model_dir),
                            warning_threshold_m,
                            roi_params,
                        )
                    )

    rows_by_scene: Dict[str, List[dict]] = {}
    if max_workers > 1 and jobs:
        with mp.Pool(processes=max_workers) as pool:
            results = pool.imap_unordered(_process_model_dir, jobs)
            for row in results:
                if row is not None:
                    rows_by_scene.setdefault(row["Scene"], []).append(row)
    else:
        for job in jobs:
            row = _process_model_dir(job)
            if row is not None:
                rows_by_scene.setdefault(row["Scene"], []).append(row)

    for scene_name, rows in sorted(rows_by_scene.items()):
        with open(warning_root / f"{scene_name}.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_SCENE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    return rows_by_scene


__all__ = ["WarningDetector", "compute_warning_rate", "run_benchmark_depth"]
