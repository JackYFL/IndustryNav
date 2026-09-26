"""Stable, dependency-light contracts shared across navigation methods."""

from nav.core.agent import NavigationAgent
from nav.core.pointgoal import (
    POINTGOAL_ENCODINGS,
    POINTGOAL_ENCODING_LEGACY,
    POINTGOAL_ENCODING_UNITY,
    encode_pointgoal,
    encode_rl_goal,
)
from nav.core.types import NavigationObservation, PointGoal, Pose2D

__all__ = [
    "NavigationAgent",
    "NavigationObservation",
    "PointGoal",
    "Pose2D",
    "POINTGOAL_ENCODINGS",
    "POINTGOAL_ENCODING_LEGACY",
    "POINTGOAL_ENCODING_UNITY",
    "encode_pointgoal",
    "encode_rl_goal",
]
