"""Discrete action vocabulary and history helpers for PointGoal PPO."""

from __future__ import annotations

import math

import torch

PPO_ACTIONS = ("forward", "turn right", "turn left")
PPO_ACTION_TO_LABEL = {
    action: index for index, action in enumerate(PPO_ACTIONS)
}
PPO_BOS_LABEL = len(PPO_ACTIONS)


def pointgoal_homing_action(
    goal_bearing_sin: float,
    goal_bearing_cos: float,
    *,
    turn_tolerance_deg: float = 11.25,
) -> str:
    """Return the atomic action that most directly reduces goal bearing.

    This is used only by the optional near-goal fallback.  It consumes the
    same GPS/compass PointGoal bearing as the learned policy and never reads
    the privileged minimap.
    """
    if not 0.0 <= float(turn_tolerance_deg) <= 90.0:
        raise ValueError("turn_tolerance_deg must be in [0, 90]")
    bearing_deg = math.degrees(
        math.atan2(float(goal_bearing_sin), float(goal_bearing_cos))
    )
    if abs(bearing_deg) <= float(turn_tolerance_deg):
        return "forward"
    return "turn right" if bearing_deg > 0.0 else "turn left"


def initial_action_history(
    batch_size: int,
    history_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return a BOS-filled ``[batch, history]`` action tensor."""
    if batch_size <= 0 or history_len <= 0:
        raise ValueError("batch_size and history_len must be positive")
    return torch.full(
        (batch_size, history_len),
        PPO_BOS_LABEL,
        dtype=torch.long,
        device=device,
    )


def append_action_history(
    history: torch.Tensor,
    actions: torch.Tensor,
    *,
    episode_done: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append one action per batch and reset completed episodes to BOS."""
    if history.ndim != 2:
        raise ValueError("action history must have shape [batch, history]")
    actions = actions.reshape(-1).to(device=history.device, dtype=torch.long)
    if actions.shape[0] != history.shape[0]:
        raise ValueError("actions and history must use the same batch size")
    updated = torch.cat((history[:, 1:], actions.unsqueeze(-1)), dim=-1)
    if episode_done is not None:
        done = episode_done.reshape(-1).to(device=history.device, dtype=torch.bool)
        if done.shape[0] != history.shape[0]:
            raise ValueError("episode_done and history must use the same batch size")
        updated = updated.masked_fill(done.unsqueeze(-1), PPO_BOS_LABEL)
    return updated


__all__ = [
    "PPO_ACTIONS",
    "PPO_ACTION_TO_LABEL",
    "PPO_BOS_LABEL",
    "append_action_history",
    "initial_action_history",
    "pointgoal_homing_action",
]
