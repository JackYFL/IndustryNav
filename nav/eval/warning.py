"""Depth-based forward-collision warning detection.

A warning frame is one where the agent's forward region of interest (a
trapezoid in the depth image) contains pixels closer than
:data:`nav.config.EVAL_WARNING_THRESHOLD_M`. The warning rate over a run is
the fraction of frames where this is true.

Three entry points by audience:

- :class:`WarningDetector` — per-frame primitive. One detector instance is
  reused across all frames of a run; the ROI polygon is lazily computed from
  the first depth map's shape.
- :func:`compute_warning_rate` — per-run helper. Takes an input directory,
  iterates over its depth frames (optionally filtered by action steps), and
  returns the fraction of warning frames.
- :func:`run_benchmark_depth` — batch walker for a tree of (scene, point,
  model) folders, with optional multiprocessing. Replaces the standalone
  ``warning_detect_mp.py`` script's main loop.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from nav.config import EVAL_ROI_PARAMS, EVAL_WARNING_THRESHOLD_M
from nav.eval.io import (
    find_actions_csv,
    find_depth_dir,
    load_action_steps,
    read_depth_npy,
)


# ---------------------------------------------------------------------------
# Per-frame primitive
# ---------------------------------------------------------------------------

class WarningDetector:
    """Forward-ROI warning detector parameterized by depth threshold + ROI shape.

    The ROI polygon is computed lazily from the first depth map's shape so
    callers don't have to commit to an image size up front. After the first
    frame the polygon is cached and reused for the rest of the run.
    """

    def __init__(
        self,
        warning_threshold_m: float = EVAL_WARNING_THRESHOLD_M,
        roi_params: Optional[Dict[str, float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.warning_threshold = float(warning_threshold_m)
        self.roi_params = dict(roi_params) if roi_params is not None else dict(EVAL_ROI_PARAMS)
        self.image_size = image_size
        self.roi_polygon: Optional[np.ndarray] = (
            self._compute_roi_polygon(image_size) if image_size is not None else None
        )

    def _compute_roi_polygon(self, image_size: Tuple[int, int]) -> np.ndarray:
        H, W = image_size
        r = self.roi_params
        p_bl = (int(W * r["bottom_pad"]), int(H * (1 - r["bottom_margin"])))
        p_br = (int(W * (1 - r["bottom_pad"])), int(H * (1 - r["bottom_margin"])))
        p_tr = (int(W * (1 - r["top_pad"])), int(H * r["top_margin"]))
        p_tl = (int(W * r["top_pad"]), int(H * r["top_margin"]))
        return np.array([p_bl, p_br, p_tr, p_tl], dtype=np.int32)

    def create_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Return a uint8 mask (1 inside the ROI, 0 outside) for ``shape``."""
        if self.roi_polygon is None:
            self.image_size = shape
            self.roi_polygon = self._compute_roi_polygon(shape)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], color=1)
        return mask

    def detect(self, depth_map: np.ndarray) -> dict:
        """Return a per-frame warning verdict.

        Output keys:

        - ``warning``: one of ``"yes" | "no" | "error"`` (``"error"`` when
          the ROI contains no valid (finite, positive) depth pixels).
        - ``min_depth_m``: minimum valid depth inside the ROI (``-1`` on error).
        - ``mean_depth_m``: mean valid depth inside the ROI (``-1`` on error).
        - ``threshold_m``: the configured warning threshold.
        - ``pixels_in_roi``: number of valid pixels considered.
        - ``warning_pixels``: number of pixels below the threshold.
        - ``warning_pixel_ratio``: ``warning_pixels / pixels_in_roi``.
        """
        if self.image_size is None:
            self.image_size = depth_map.shape[:2]
            self.roi_polygon = self._compute_roi_polygon(self.image_size)
        elif depth_map.shape[:2] != self.image_size:
            depth_map = cv2.resize(
                depth_map,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        roi_mask = self.create_roi_mask(depth_map.shape[:2])
        roi_depths = depth_map[roi_mask == 1]
        valid = roi_depths[np.isfinite(roi_depths) & (roi_depths > 0)]

        if len(valid) == 0:
            return {
                "warning": "error",
                "min_depth_m": -1.0,
                "mean_depth_m": -1.0,
                "threshold_m": float(self.warning_threshold),
                "pixels_in_roi": 0,
                "warning_pixels": 0,
                "warning_pixel_ratio": 0.0,
            }

        min_depth = float(np.min(valid))
        warning_pixels = int(np.sum(valid < self.warning_threshold))
        total = int(len(valid))
        return {
            "warning": "yes" if min_depth < self.warning_threshold else "no",
            "min_depth_m": min_depth,
            "mean_depth_m": float(np.mean(valid)),
            "threshold_m": float(self.warning_threshold),
            "pixels_in_roi": total,
            "warning_pixels": warning_pixels,
            "warning_pixel_ratio": warning_pixels / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Per-run aggregation
# ---------------------------------------------------------------------------

def compute_warning_rate(
    input_dir: Path,
    detector: Optional[WarningDetector] = None,
    *,
    use_actions: bool = True,
) -> Tuple[int, int, float]:
    """Compute the warning rate for one run's output directory.

    Returns ``(total_steps, warning_steps, warning_rate)``. If
    ``use_actions`` is True (default), only depth frames whose stem matches
    an integer step in the actions CSV are evaluated — this skips frames
    captured outside the agent's decision loop. Pass ``use_actions=False``
    to evaluate every ``.npy`` under the depth directory.

    If the depth directory is missing or contains no readable frames,
    returns ``(0, 0, 0.0)``.
    """
    depth_dir = find_depth_dir(input_dir)
    if depth_dir is None:
        return 0, 0, 0.0

    detector = detector or WarningDetector()

    step_filter: Optional[set] = None
    if use_actions:
        actions_csv = find_actions_csv(input_dir)
        if actions_csv is not None:
            step_filter = set(load_action_steps(actions_csv))

    total = warning = 0
    for df in sorted(depth_dir.glob("*.npy")):
        try:
            step_id = int(df.stem)
        except ValueError:
            step_id = None
        if step_filter is not None and step_id is not None and step_id not in step_filter:
            continue
        depth = read_depth_npy(df)
        if depth is None:
            continue
        info = detector.detect(depth)
        total += 1
        if info["warning"] == "yes":
            warning += 1

    rate = (warning / total) if total > 0 else 0.0
    return total, warning, rate


# ---------------------------------------------------------------------------
# Batch walker over (scene, point, model) trees
# ---------------------------------------------------------------------------

_SCENE_FIELDNAMES = [
    "Scene", "Point_Pair", "Model_Name",
    "Total_Steps", "Total_Warning_Steps", "Warning_Rate",
]


def _process_model_dir(job: tuple) -> Optional[dict]:
    """Worker for :func:`run_benchmark_depth`. Pure function, picklable.

    Each detector is instantiated fresh inside the worker so concurrent
    workers can't share mutable state (ROI cache).
    """
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
    for df in depth_files:
        depth = read_depth_npy(df)
        if depth is None or depth.size == 0 or not np.isfinite(depth).any():
            total += 1
            continue
        info = detector.detect(depth)
        total += 1
        if info["warning"] == "yes":
            warning += 1

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
    """Walk ``depth_root/<scene>/<point>/<model>/`` and write per-scene warning CSVs.

    Returns a ``{scene_name: [row_dict, ...]}`` map of what was written, for
    callers that want to do further processing in-memory. The CSVs land at
    ``warning_root/<scene>.csv``.

    Parallelism is controlled by ``max_workers``. The pool is created
    lazily inside this function (not at import time) so spinning up a
    multiprocessing worker doesn't trigger re-init in re-entrant test runs.
    """
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
                if not model_dir.is_dir() or not any(model_dir.glob("*.npy")):
                    continue
                jobs.append(
                    (scene_dir.name, point_dir.name, str(model_dir),
                     warning_threshold_m, roi_params)
                )

    scene_rows: Dict[str, List[dict]] = {}
    if max_workers > 1 and jobs:
        with mp.Pool(processes=max_workers) as pool:
            for row in pool.imap_unordered(_process_model_dir, jobs):
                if row is not None:
                    scene_rows.setdefault(row["Scene"], []).append(row)
    else:
        for job in jobs:
            row = _process_model_dir(job)
            if row is not None:
                scene_rows.setdefault(row["Scene"], []).append(row)

    for scene_name, rows in sorted(scene_rows.items()):
        csv_path = warning_root / f"{scene_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SCENE_FIELDNAMES)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return scene_rows
