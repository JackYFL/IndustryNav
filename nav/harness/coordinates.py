"""Letterbox-aware visual-pixel → Unity-pixel coordinate mapping.

The minimap render is letterboxed inside the camera frame, so the visual
pixel a user/CLI specifies isn't the Unity-internal minimap pixel. We detect
the rendered minimap's tight bounding box (via Canny edges) once, then
rescale visual pixels into the client's expected ``UNITY_MAP_SIZE`` space.
The client owns the final pixel→world transform.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from nav.config import MINIMAP_CANNY_HI, MINIMAP_CANNY_LO, UNITY_MAP_SIZE


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
    margin: Tuple[float, float, float, float], px: float, py: float
) -> Tuple[float, float]:
    """Rescale a visual minimap pixel into the client's Unity-pixel space.

    ``margin`` is the ``(min_x, max_x, min_y, max_y)`` from
    :func:`find_exact_map_bounds`. The result is fed to the client's
    ``spawn_px`` / ``target_px`` env params; the client does pixel→world.
    """
    min_x, max_x, min_y, max_y = margin
    map_w, map_h = UNITY_MAP_SIZE
    u = (px - min_x) / (max_x - min_x)
    v = (py - min_y) / (max_y - min_y)
    return u * map_w, v * map_h
