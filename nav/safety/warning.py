"""Action-aware warning detection from an egocentric depth image."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from nav.config import (
    EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M,
    EVAL_ROI_PARAMS,
    EVAL_WARNING_MIN_PIXEL_RATIO,
    EVAL_WARNING_THRESHOLD_M,
)


class WarningDetector:
    """Detect nearby geometry inside a resolution-independent forward ROI."""

    def __init__(
        self,
        warning_threshold_m: float = EVAL_WARNING_THRESHOLD_M,
        roi_params: Optional[Dict[str, float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        min_warning_pixel_ratio: float = EVAL_WARNING_MIN_PIXEL_RATIO,
        forward_distance_per_move_unit_m: float = (
            EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
        ),
    ) -> None:
        self.warning_threshold = float(warning_threshold_m)
        self.roi_params = (
            dict(roi_params) if roi_params is not None else dict(EVAL_ROI_PARAMS)
        )
        self._validate_roi_params()
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
        self._roi_polygons: dict[Tuple[int, int], np.ndarray] = {}
        self.roi_polygon: Optional[np.ndarray] = None
        if self.image_size is not None:
            self.roi_polygon = self._compute_roi_polygon(self.image_size)
            self._roi_polygons[self.image_size] = self.roi_polygon

    def _validate_roi_params(self) -> None:
        required = {"bottom_margin", "top_margin", "bottom_pad", "top_pad"}
        missing = required - self.roi_params.keys()
        if missing:
            raise ValueError(f"roi_params is missing: {', '.join(sorted(missing))}")
        if any(not np.isfinite(float(self.roi_params[key])) for key in required):
            raise ValueError("roi_params values must be finite")
        if not 0.0 <= float(self.roi_params["top_margin"]) < 1.0:
            raise ValueError("top_margin must be in [0, 1)")
        if not 0.0 <= float(self.roi_params["bottom_margin"]) < 1.0:
            raise ValueError("bottom_margin must be in [0, 1)")
        for key in ("top_pad", "bottom_pad"):
            if not 0.0 <= float(self.roi_params[key]) < 0.5:
                raise ValueError(f"{key} must be in [0, 0.5)")
        if float(self.roi_params["top_margin"]) >= (
            1.0 - float(self.roi_params["bottom_margin"])
        ):
            raise ValueError("ROI top must be above its bottom")

    def _compute_roi_polygon(self, image_size: Tuple[int, int]) -> np.ndarray:
        height, width = image_size
        params = self.roi_params
        return np.array(
            [
                (
                    int(round((width - 1) * params["bottom_pad"])),
                    int(round((height - 1) * (1 - params["bottom_margin"]))),
                ),
                (
                    int(round((width - 1) * (1 - params["bottom_pad"]))),
                    int(round((height - 1) * (1 - params["bottom_margin"]))),
                ),
                (
                    int(round((width - 1) * (1 - params["top_pad"]))),
                    int(round((height - 1) * params["top_margin"])),
                ),
                (
                    int(round((width - 1) * params["top_pad"])),
                    int(round((height - 1) * params["top_margin"])),
                ),
            ],
            dtype=np.int32,
        )

    def create_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Return a uint8 mask (1 inside the ROI, 0 outside) for ``shape``."""
        shape = tuple(shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("shape must be a positive (height, width) pair")
        polygon = self._roi_polygons.get(shape)
        if polygon is None:
            polygon = self._compute_roi_polygon(shape)
            self._roi_polygons[shape] = polygon
        self.roi_polygon = polygon
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], color=1)
        return mask

    @staticmethod
    def _resize_depth_min_preserving(
        depth_map: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize depth while preserving nearby surfaces when possible."""
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
        """Return a per-frame warning verdict and supporting measurements."""
        if depth_map.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got {depth_map.shape}")
        if self.image_size is not None and depth_map.shape[:2] != self.image_size:
            depth_map = self._resize_depth_min_preserving(depth_map, self.image_size)

        move = float(move_command)
        if not np.isfinite(move):
            move = 0.0
        effective_threshold = self.warning_threshold + (
            max(0.0, move) * self.forward_distance_per_move_unit_m
        )
        roi_depths = depth_map[self.create_roi_mask(depth_map.shape[:2]) == 1]
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

        warning_pixels = int(np.sum(valid < effective_threshold))
        total = int(len(valid))
        ratio = warning_pixels / total if total else 0.0
        warning = warning_pixels > 0 and ratio >= self.min_warning_pixel_ratio
        return {
            "warning": "yes" if warning else "no",
            "min_depth_m": float(np.min(valid)),
            "mean_depth_m": float(np.mean(valid)),
            "threshold_m": float(effective_threshold),
            "base_threshold_m": float(self.warning_threshold),
            "pixels_in_roi": total,
            "warning_pixels": warning_pixels,
            "warning_pixel_ratio": ratio,
            "min_warning_pixel_ratio": self.min_warning_pixel_ratio,
        }
