"""Small data objects used at boundaries between baselines and runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Pose2D:
    """Unity world pose projected onto the navigation plane."""

    world_x: float
    world_z: float
    yaw_deg: float


@dataclass(frozen=True)
class PointGoal:
    """World-space PointGoal target on Unity's X/Z plane."""

    world_x: float
    world_z: float


@dataclass(frozen=True)
class NavigationObservation:
    """Baseline-neutral observation bundle for future adapters."""

    pose: Pose2D
    goal: PointGoal
    rgb: Optional[Any] = None
    depth: Optional[Any] = None
