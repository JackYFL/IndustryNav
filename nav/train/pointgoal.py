"""Shared PointGoal coordinate transforms for training and deployment."""

from __future__ import annotations

import math

import numpy as np


POINTGOAL_ENCODING_LEGACY = "legacy_v1"
POINTGOAL_ENCODING_UNITY = "unity_egocentric_v2"
POINTGOAL_ENCODINGS = (
    POINTGOAL_ENCODING_LEGACY,
    POINTGOAL_ENCODING_UNITY,
)


def encode_pointgoal(
    curr_world_x: float,
    curr_world_z: float,
    curr_yaw_deg: float,
    target_world_x: float,
    target_world_z: float,
    *,
    goal_rep: str,
    distance_scale_m: float,
    encoding: str = POINTGOAL_ENCODING_UNITY,
) -> np.ndarray:
    """Encode a world-space target in the agent's egocentric frame.

    Unity yaw zero faces world ``+Z`` and positive yaw turns toward ``+X``.
    The v2 Cartesian layout is ``[forward, right]``; its polar layout is
    ``[distance, bearing]``, where a positive bearing means turn right.  Both
    layouts therefore mirror horizontally by negating their second value.

    ``legacy_v1`` exists only so checkpoints trained before the coordinate fix
    retain their original inference semantics.
    """
    if goal_rep not in {"cartesian", "polar"}:
        raise ValueError(f"Unsupported goal representation: {goal_rep!r}")
    if distance_scale_m < 0.0:
        raise ValueError("distance_scale_m must be nonnegative")
    if encoding not in POINTGOAL_ENCODINGS:
        raise ValueError(f"Unsupported PointGoal encoding: {encoding!r}")

    dx = float(target_world_x) - float(curr_world_x)
    dz = float(target_world_z) - float(curr_world_z)
    yaw_rad = math.radians(float(curr_yaw_deg))
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)

    if encoding == POINTGOAL_ENCODING_LEGACY:
        first = cos_yaw * dx + sin_yaw * dz
        second = -sin_yaw * dx + cos_yaw * dz
    else:
        # Dot the target delta with Unity's forward and right unit vectors.
        first = sin_yaw * dx + cos_yaw * dz
        second = cos_yaw * dx - sin_yaw * dz

    if goal_rep == "polar":
        distance = math.hypot(first, second)
        bearing = math.atan2(second, first)
        if distance_scale_m > 0.0:
            distance /= distance_scale_m
            bearing /= math.pi
        return np.array([distance, bearing], dtype=np.float32)

    if distance_scale_m > 0.0:
        first /= distance_scale_m
        second /= distance_scale_m
    return np.array([first, second], dtype=np.float32)
