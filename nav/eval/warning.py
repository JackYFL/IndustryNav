"""Depth-based forward-collision warning detection.

A warning frame is one where a sufficient fraction of the agent's forward
region of interest (a trapezoid in the depth image) is closer than the current
action's safety distance. That distance is the base clearance plus the
theoretical distance of a positive move command. The warning rate over a run
is the fraction of frames where this is true.

Three entry points by audience:

- :class:`WarningDetector` — per-frame primitive. One detector instance is
  reused across all frames of a run; inputs are normalized to one evaluation
  resolution before applying the cached ROI polygon.
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

from nav.config import (
    EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M,
    EVAL_ROI_PARAMS,
    EVAL_WARNING_IMAGE_SIZE,
    EVAL_WARNING_MIN_PIXEL_RATIO,
    EVAL_WARNING_THRESHOLD_M,
)
from nav.eval.io import (
    find_actions_csv,
    find_depth_dir,
    load_action_moves,
    load_action_steps,
    read_depth_npy,
)


# ---------------------------------------------------------------------------
# Per-frame primitive
# ---------------------------------------------------------------------------

class WarningDetector:
    """Action-aware forward-ROI warning detector.

    Depth maps are normalized to ``image_size`` before evaluation so runs with
    different sensor resolutions are comparable. A frame warns only when the
    proportion of valid ROI pixels inside the action-aware safety distance is
    at least ``min_warning_pixel_ratio``.
    """

    def __init__(
        self,
        warning_threshold_m: float = EVAL_WARNING_THRESHOLD_M,
        roi_params: Optional[Dict[str, float]] = None,
        image_size: Optional[Tuple[int, int]] = EVAL_WARNING_IMAGE_SIZE,
        min_warning_pixel_ratio: float = EVAL_WARNING_MIN_PIXEL_RATIO,
        forward_distance_per_move_unit_m: float = (
            EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
        ),
    ) -> None:
        self.warning_threshold = float(warning_threshold_m)
        self.roi_params = (
            dict(roi_params) if roi_params is not None else dict(EVAL_ROI_PARAMS)
        )
        self.image_size = tuple(image_size) if image_size is not None else None
        self.min_warning_pixel_ratio = float(min_warning_pixel_ratio)
        self.forward_distance_per_move_unit_m = float(
            forward_distance_per_move_unit_m
        )
        if self.warning_threshold < 0.0:
            raise ValueError("warning_threshold_m must be nonnegative")
        if not 0.0 <= self.min_warning_pixel_ratio <= 1.0:
            raise ValueError("min_warning_pixel_ratio must be in [0, 1]")
        if self.forward_distance_per_move_unit_m < 0.0:
            raise ValueError(
                "forward_distance_per_move_unit_m must be nonnegative"
            )
        if self.image_size is not None and (
            len(self.image_size) != 2 or min(self.image_size) <= 0
        ):
            raise ValueError("image_size must be a positive (height, width) pair")
        self.roi_polygon: Optional[np.ndarray] = (
            self._compute_roi_polygon(self.image_size)
            if self.image_size is not None
            else None
        )

    def _compute_roi_polygon(self, image_size: Tuple[int, int]) -> np.ndarray:
        H, W = image_size
        r = self.roi_params
        p_bl = (
            int(W * r["bottom_pad"]),
            int(H * (1 - r["bottom_margin"])),
        )
        p_br = (
            int(W * (1 - r["bottom_pad"])),
            int(H * (1 - r["bottom_margin"])),
        )
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

    @staticmethod
    def _resize_depth_min_preserving(
        depth_map: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize depth while preserving nearby surfaces during downsampling."""
        source_h, source_w = depth_map.shape[:2]
        target_h, target_w = image_size
        if (source_h, source_w) == image_size:
            return depth_map
        if (
            source_h >= target_h
            and source_w >= target_w
            and source_h % target_h == 0
            and source_w % target_w == 0
        ):
            scale_h = source_h // target_h
            scale_w = source_w // target_w
            valid = np.isfinite(depth_map) & (depth_map > 0)
            safe_depth = np.where(valid, depth_map, np.inf)
            resized = safe_depth.reshape(
                target_h, scale_h, target_w, scale_w
            ).min(axis=(1, 3))
            resized[~np.isfinite(resized)] = np.nan
            return resized.astype(np.float32, copy=False)
        return cv2.resize(
            depth_map,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    def detect(self, depth_map: np.ndarray, *, move_command: float = 0.0) -> dict:
        """Return a per-frame warning verdict.

        Output keys:

        - ``warning``: one of ``"yes" | "no" | "error"`` (``"error"`` when
          the ROI contains no valid (finite, positive) depth pixels).
        - ``min_depth_m``: minimum valid depth inside the ROI (``-1`` on error).
        - ``mean_depth_m``: mean valid depth inside the ROI (``-1`` on error).
        - ``threshold_m``: base clearance plus commanded forward distance.
        - ``pixels_in_roi``: number of valid pixels considered.
        - ``warning_pixels``: number of pixels below the threshold.
        - ``warning_pixel_ratio``: ``warning_pixels / pixels_in_roi``.
        """
        if depth_map.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got {depth_map.shape}")
        if self.image_size is None:
            self.image_size = depth_map.shape[:2]
            self.roi_polygon = self._compute_roi_polygon(self.image_size)
        elif depth_map.shape[:2] != self.image_size:
            depth_map = self._resize_depth_min_preserving(
                depth_map,
                self.image_size,
            )

        move = float(move_command)
        if not np.isfinite(move):
            move = 0.0
        effective_threshold = self.warning_threshold + (
            max(0.0, move) * self.forward_distance_per_move_unit_m
        )

        roi_mask = self.create_roi_mask(depth_map.shape[:2])
        roi_depths = depth_map[roi_mask == 1]
        valid = roi_depths[np.isfinite(roi_depths) & (roi_depths > 0)]

        if len(valid) == 0:
            return {
                "warning": "error",
                "min_depth_m": -1.0,
                "mean_depth_m": -1.0,
                "threshold_m": float(effective_threshold),
                "base_threshold_m": float(self.warning_threshold),
                "pixels_in_roi": 0,
                "warning_pixels": 0,
                "warning_pixel_ratio": 0.0,
                "min_warning_pixel_ratio": self.min_warning_pixel_ratio,
            }

        min_depth = float(np.min(valid))
        warning_pixels = int(np.sum(valid < effective_threshold))
        total = int(len(valid))
        warning_pixel_ratio = warning_pixels / total if total > 0 else 0.0
        warning = (
            warning_pixels > 0
            and warning_pixel_ratio >= self.min_warning_pixel_ratio
        )
        return {
            "warning": "yes" if warning else "no",
            "min_depth_m": min_depth,
            "mean_depth_m": float(np.mean(valid)),
            "threshold_m": float(effective_threshold),
            "base_threshold_m": float(self.warning_threshold),
            "pixels_in_roi": total,
            "warning_pixels": warning_pixels,
            "warning_pixel_ratio": warning_pixel_ratio,
            "min_warning_pixel_ratio": self.min_warning_pixel_ratio,
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

    When an actions CSV is available, its ``move`` value supplies the current
    theoretical forward distance. Frames without a matching move use only the
    base clearance. If the depth directory is missing or has no readable
    frames, returns ``(0, 0, 0.0)``.
    """
    depth_dir = find_depth_dir(input_dir)
    if depth_dir is None:
        return 0, 0, 0.0

    detector = detector or WarningDetector()

    actions_csv = find_actions_csv(input_dir)
    action_moves = (
        load_action_moves(actions_csv) if actions_csv is not None else {}
    )
    step_filter: Optional[set] = None
    if use_actions and actions_csv is not None:
        step_filter = set(load_action_steps(actions_csv))

    total = warning = 0
    for df in sorted(depth_dir.glob("*.npy")):
        try:
            step_id = int(df.stem)
        except ValueError:
            step_id = None
        if step_filter is not None and step_id not in step_filter:
            continue
        depth = read_depth_npy(df)
        if depth is None:
            continue
        info = detector.detect(
            depth,
            move_command=action_moves.get(step_id, 0.0),
        )
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
