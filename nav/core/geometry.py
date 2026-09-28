"""Dependency-light geometry helpers shared by planners and evaluation."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple


Point = Tuple[int, int]
WorldPoint = Tuple[float, float]


def transformed_path_length_m(
    path: Sequence[Point],
    point_to_world: Optional[Callable[[Point], Optional[WorldPoint]]],
) -> float | None:
    """Return a pixel path's world-space length when calibration is available."""
    if point_to_world is None or len(path) < 2:
        return None
    world_points = [point_to_world(point) for point in path]
    if any(point is None for point in world_points):
        return None
    return float(
        sum(
            math.hypot(
                float(current[0]) - float(previous[0]),
                float(current[1]) - float(previous[1]),
            )
            for previous, current in zip(world_points, world_points[1:])
        )
    )


def remaining_path_length_m(
    path: Sequence[Point],
    current_world: WorldPoint,
    point_to_world: Optional[Callable[[Point], Optional[WorldPoint]]],
) -> float | None:
    """Return distance from the current pose along the remaining planned path.

    Cached A* paths retain points already traversed.  The closest projected
    path point is therefore used as the remaining-path anchor, followed by the
    path's world-space arc length from that anchor to the target.
    """
    if point_to_world is None or not path:
        return None
    projected = [point_to_world(point) for point in path]
    valid = [(index, point) for index, point in enumerate(projected) if point]
    if not valid:
        return None
    closest_index, closest_world = min(
        valid,
        key=lambda item: math.hypot(
            float(item[1][0]) - float(current_world[0]),
            float(item[1][1]) - float(current_world[1]),
        ),
    )
    remaining = math.hypot(
        float(closest_world[0]) - float(current_world[0]),
        float(closest_world[1]) - float(current_world[1]),
    )
    previous = closest_world
    for point in projected[closest_index + 1 :]:
        if point is None:
            return None
        remaining += math.hypot(
            float(point[0]) - float(previous[0]),
            float(point[1]) - float(previous[1]),
        )
        previous = point
    return float(remaining)
