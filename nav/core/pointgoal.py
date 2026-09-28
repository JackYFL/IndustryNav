"""Shared PointGoal coordinate transforms for training and deployment.

This module belongs to :mod:`nav.core` because goal geometry is consumed by
datasets, learned baselines, and evaluation-time agents alike.
"""

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


def encode_rl_goal(
    curr_world_x: float,
    curr_world_z: float,
    curr_yaw_deg: float,
    target_world_x: float,
    target_world_z: float,
    distance_scale_m: float = 50.0,
) -> np.ndarray:
    """Return ``[scaled distance, sin(bearing), cos(bearing)]`` in Unity axes."""
    if distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    dx = float(target_world_x) - float(curr_world_x)
    dz = float(target_world_z) - float(curr_world_z)
    yaw = math.radians(float(curr_yaw_deg))
    forward = math.sin(yaw) * dx + math.cos(yaw) * dz
    right = math.cos(yaw) * dx - math.sin(yaw) * dz
    distance = math.hypot(forward, right)
    bearing = math.atan2(right, forward)
    return np.asarray(
        [distance / distance_scale_m, math.sin(bearing), math.cos(bearing)],
        dtype=np.float32,
    )


def encode_rl_goal_state(
    curr_world_x: float,
    curr_world_z: float,
    curr_yaw_deg: float,
    target_world_x: float,
    target_world_z: float,
    *,
    distance_scale_m: float = 50.0,
    scene_id: int | None = None,
    num_scenes: int = 24,
    world_coordinate_scale_m: float = 50.0,
    include_absolute_scene_state: bool = False,
    coordinate_fourier_bands: int = 0,
) -> np.ndarray:
    """Encode PointGoal plus optional absolute GPS and scene context.

    The optional context is useful when training and evaluating new point
    pairs in the same known layouts: four normalized world coordinates, a
    one-hot scene identity, normalized world-frame target delta, and explicit
    sine/cosine heading let the recurrent policy learn layout-specific detours
    without exposing the minimap or benchmark point pairs. New fields are
    appended after the legacy context so older checkpoints migrate exactly.
    """
    relative = encode_rl_goal(
        curr_world_x,
        curr_world_z,
        curr_yaw_deg,
        target_world_x,
        target_world_z,
        distance_scale_m,
    )
    if not include_absolute_scene_state:
        return relative
    if world_coordinate_scale_m <= 0.0:
        raise ValueError("world_coordinate_scale_m must be positive")
    if num_scenes <= 0:
        raise ValueError("num_scenes must be positive")
    if coordinate_fourier_bands < 0:
        raise ValueError("coordinate_fourier_bands must be nonnegative")
    if scene_id is None or not 0 <= int(scene_id) < num_scenes:
        raise ValueError(f"scene_id must be in [0, {num_scenes - 1}]")
    coordinates = np.asarray(
        [curr_world_x, curr_world_z, target_world_x, target_world_z],
        dtype=np.float32,
    ) / float(world_coordinate_scale_m)
    scene = np.zeros(num_scenes, dtype=np.float32)
    scene[int(scene_id)] = 1.0
    scale = float(world_coordinate_scale_m)
    yaw_rad = math.radians(float(curr_yaw_deg))
    planning_context = np.asarray(
        [
            (float(target_world_x) - float(curr_world_x)) / scale,
            (float(target_world_z) - float(curr_world_z)) / scale,
            math.sin(yaw_rad),
            math.cos(yaw_rad),
        ],
        dtype=np.float32,
    )
    fourier = np.asarray(
        [
            component
            for value in coordinates
            for band in range(int(coordinate_fourier_bands))
            for component in (
                math.sin(2.0 * math.pi * (2**band) * float(value)),
                math.cos(2.0 * math.pi * (2**band) * float(value)),
            )
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (relative, coordinates, scene, planning_context, fourier)
    )
