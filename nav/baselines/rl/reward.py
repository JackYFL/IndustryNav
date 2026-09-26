"""Reward functions owned by the PointGoal reinforcement-learning method."""

from __future__ import annotations

import numpy as np


TURN_ACTIONS = frozenset(("turn left", "turn right"))


def is_turn_reversal(previous_action: str | None, action: str | None) -> bool:
    """Return whether two consecutive actions reverse turning direction."""
    return (
        previous_action in TURN_ACTIONS
        and action in TURN_ACTIONS
        and previous_action != action
    )


def collision_started(collision: bool, previous_collision: bool) -> bool:
    """Return true only on the first step of a contiguous collision."""
    return bool(collision and not previous_collision)


def should_trigger_stuck_recovery(
    steps_without_progress: int,
    threshold: int,
    *,
    success: bool,
    timeout: bool,
) -> bool:
    """Terminate a stalled training episode early so sampling can recover."""
    return bool(
        threshold > 0
        and steps_without_progress >= threshold
        and not success
        and not timeout
    )


def compute_pointgoal_reward(
    previous_distance_m: float,
    current_distance_m: float,
    *,
    success: bool,
    timeout: bool,
    collision: bool,
    collision_onset: bool | None = None,
    warning: bool,
    path_progress_m: float | None = None,
    progress_scale: float = 1.0,
    step_penalty: float = 0.01,
    success_bonus: float = 5.0,
    timeout_penalty: float = 1.0,
    collision_penalty: float = 1.0,
    collision_step_penalty: float = 0.0,
    warning_penalty: float = 0.05,
    action_name: str | None = None,
    turn_reversal: bool = False,
    stagnation_steps: int = 0,
    rotation_penalty: float = 0.0,
    turn_reversal_penalty: float = 0.0,
    stagnation_penalty: float = 0.0,
    stagnation_start_steps: int = 12,
    safe_forward_bonus: float = 0.0,
    stuck_recovery: bool = False,
    stuck_recovery_penalty: float = 0.0,
) -> float:
    """Safety-aware dense PointGoal reward computed entirely in Python.

    Rotation, reversal, and stagnation terms close the reward-hacking loophole
    where an agent can avoid forward-collision penalties by oscillating in
    place. The safe-forward bonus is awarded only when depth and physical
    motion checks both agree that a translation was unobstructed.
    """
    raw_progress = (
        previous_distance_m - current_distance_m
        if path_progress_m is None
        else float(path_progress_m)
    )
    progress = float(np.clip(raw_progress, -1.0, 1.0))
    reward = progress_scale * progress - step_penalty
    if action_name in TURN_ACTIONS:
        reward -= rotation_penalty
    if turn_reversal:
        reward -= turn_reversal_penalty
    if stagnation_start_steps > 0 and stagnation_steps >= stagnation_start_steps:
        reward -= stagnation_penalty
    if action_name == "forward" and not collision and not warning:
        reward += safe_forward_bonus
    # Preserve the historical standalone API when collision_onset is omitted.
    # The Unity environment supplies it explicitly to avoid charging the same
    # physical contact once per blocked forward step.
    if collision if collision_onset is None else collision_onset:
        reward -= collision_penalty
    # A small per-step term makes repeated blocked-forward actions worse than
    # turning away after contact.  Keep it separate from the larger contact-
    # onset penalty so persistent contact does not dominate the whole return.
    if collision:
        reward -= collision_step_penalty
    if warning:
        reward -= warning_penalty
    if success:
        reward += success_bonus
    elif timeout:
        reward -= timeout_penalty
    elif stuck_recovery:
        reward -= stuck_recovery_penalty
    return float(reward)


__all__ = [
    "TURN_ACTIONS",
    "collision_started",
    "compute_pointgoal_reward",
    "is_turn_reversal",
    "should_trigger_stuck_recovery",
]
