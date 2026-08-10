"""Letterbox-aware visual-pixel → Unity-pixel coordinate mapping.

The minimap render is letterboxed inside the camera frame, so the visual
pixel a user/CLI specifies isn't the Unity-internal minimap pixel. We detect
the rendered minimap's tight bounding box (via Canny edges) once, then
rescale visual pixels into the selected runtime minimap space. The client owns
the final pixel→world transform.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from nav.config import MINIMAP_CANNY_HI, MINIMAP_CANNY_LO, UNITY_MAP_SIZE


def resolve_minimap_resolution(
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve a positive minimap size while preserving the canonical ratio."""
    canonical_w, canonical_h = (int(v) for v in UNITY_MAP_SIZE)
    if width is None and height is None:
        return canonical_w, canonical_h
    if width is None:
        if height is None or height <= 0:
            raise ValueError("--minimap_height must be a positive integer.")
        width = round(height * canonical_w / canonical_h)
    if width <= 0:
        raise ValueError("--minimap_width must be a positive integer.")

    expected_height = round(width * canonical_h / canonical_w)
    if height is None:
        height = expected_height
    if height <= 0:
        raise ValueError("--minimap_height must be a positive integer.")
    if height != expected_height:
        raise ValueError(
            "Minimap resolution must preserve the "
            f"{canonical_w}:{canonical_h} aspect ratio: width {width} requires "
            f"height {expected_height}, got {height}."
        )
    return int(width), int(height)


def minimap_axis_scales(
    minimap_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Return runtime/canonical scale factors for the x and y axes."""
    width, height = (int(v) for v in minimap_size)
    if width <= 0 or height <= 0:
        raise ValueError("Minimap size must contain positive width and height values.")
    canonical_w, canonical_h = (float(v) for v in UNITY_MAP_SIZE)
    return width / canonical_w, height / canonical_h


def minimap_pixel_scale(minimap_size: Tuple[int, int]) -> float:
    """Return the scalar used for isotropic pixel-distance parameters."""
    scale_x, scale_y = minimap_axis_scales(minimap_size)
    return (scale_x + scale_y) / 2.0


def canonical_to_minimap_coords(
    point: Tuple[float, float],
    minimap_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Scale a canonical ``862 x 512`` point into runtime minimap pixels."""
    scale_x, scale_y = minimap_axis_scales(minimap_size)
    width, height = (int(v) for v in minimap_size)
    x = int(np.clip(round(float(point[0]) * scale_x), 0, width - 1))
    y = int(np.clip(round(float(point[1]) * scale_y), 0, height - 1))
    return x, y


def minimap_to_canonical_coords(
    point: Tuple[float, float],
    minimap_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Scale a runtime minimap point back into canonical benchmark pixels."""
    scale_x, scale_y = minimap_axis_scales(minimap_size)
    return float(point[0]) / scale_x, float(point[1]) / scale_y


def canonical_minimap_distance_to_runtime(
    distance_px: float,
    minimap_size: Tuple[int, int],
) -> float:
    """Scale a canonical pixel-distance threshold for runtime use."""
    return float(distance_px) * minimap_pixel_scale(minimap_size)


def runtime_minimap_distance_to_canonical(
    first: Tuple[float, float],
    second: Tuple[float, float],
    minimap_size: Tuple[int, int],
) -> float:
    """Measure two runtime points in canonical benchmark pixel units."""
    first_x, first_y = minimap_to_canonical_coords(first, minimap_size)
    second_x, second_y = minimap_to_canonical_coords(second, minimap_size)
    return float(np.hypot(first_x - second_x, first_y - second_y))


def find_exact_map_bounds(
    logger: logging.Logger, minimap_rgb: Optional[np.ndarray]
) -> Optional[Tuple[float, float, float, float]]:
    """Return the ``(min_x, max_x, min_y, max_y)`` tight bounds of the rendered minimap.

    Uses Canny edges to find the non-letterbox region. Returns None if the
    frame is missing or edge detection finds nothing (blank frame).
    """
    if minimap_rgb is None:
        return None
    gray = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=MINIMAP_CANNY_LO, threshold2=MINIMAP_CANNY_HI)
    y_idx, x_idx = np.where(edges > 0)
    if len(x_idx) == 0 or len(y_idx) == 0:
        logger.warning("Minimap edge detection found no edges; cannot compute margin.")
        return None
    min_x, max_x = float(x_idx.min()), float(x_idx.max())
    min_y, max_y = float(y_idx.min()), float(y_idx.max())
    logger.info(
        f"Minimap effective bounds: x=[{min_x:.0f},{max_x:.0f}] y=[{min_y:.0f},{max_y:.0f}] "
        f"({max_x - min_x:.0f}x{max_y - min_y:.0f})"
    )
    return min_x, max_x, min_y, max_y


def visual_to_unity_coords(
    margin: Tuple[float, float, float, float],
    px: float,
    py: float,
    map_size: Tuple[float, float] = UNITY_MAP_SIZE,
) -> Tuple[float, float]:
    """Rescale a visual minimap pixel into the client's Unity-pixel space.

    ``margin`` is the ``(min_x, max_x, min_y, max_y)`` from
    :func:`find_exact_map_bounds`. The result is fed to the client's
    ``spawn_px`` / ``target_px`` env params; the client does pixel→world.
    """
    min_x, max_x, min_y, max_y = margin
    map_w, map_h = map_size
    u = (px - min_x) / (max_x - min_x)
    v = (py - min_y) / (max_y - min_y)
    return u * map_w, v * map_h
