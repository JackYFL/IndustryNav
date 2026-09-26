"""Transition cues derived only from successive relative point-goal inputs."""

import math

import torch
from torch.nn import functional as F


MOTION_FEATURE_DIM = 6


def relative_motion_features(previous_goal, current_goal, previous_action, masks,
                             *, goal_rep, distance_scale_m):
    """Return validity, executed-action one-hot, progress and forward displacement.

    Goals use the BC window's two-component encoding: normalized distance and
    bearing/pi, or normalized Cartesian forward/right coordinates. The last two
    features are metres clipped to [-2, 2] and [0, 2], respectively. They are not
    collision labels. Forward displacement assumes heading is unchanged during
    a forward command; turns are explicitly excluded from that estimate.

    Reset/BOS transitions are zero. No absolute pose, target ID, depth warning,
    reward, planner, or future observation enters this function. Inputs come from
    the same cached window during rollout and PPO replay.
    """
    if goal_rep not in {"polar", "cartesian"}:
        raise ValueError("Motion features require polar or cartesian goals")
    if not math.isfinite(distance_scale_m) or distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be finite and positive")
    if previous_goal.ndim != 2 or previous_goal.shape[-1] != 2 or current_goal.shape != previous_goal.shape:
        raise ValueError("Motion goals must have matching [batch, 2] shapes")
    batch = previous_goal.shape[0]
    action = previous_action.reshape(batch)
    valid = (masks.reshape(batch) != 0) & (action >= 0) & (action < 3)
    # Mask before arithmetic: even stale NaNs in a reset episode must not leak.
    previous_goal = torch.where(valid[:, None], previous_goal, torch.zeros_like(previous_goal))
    current_goal = torch.where(valid[:, None], current_goal, torch.zeros_like(current_goal))

    def cartesian(goal):
        if goal_rep == "cartesian":
            return goal * distance_scale_m
        distance, bearing = goal[:, 0] * distance_scale_m, goal[:, 1] * math.pi
        return torch.stack((distance * bearing.cos(), distance * bearing.sin()), -1)

    before, after = cartesian(previous_goal), cartesian(current_goal)
    progress = (before.norm(dim=-1) - after.norm(dim=-1)).clamp(-2., 2.)
    displacement = (before - after).norm(dim=-1).clamp(0., 2.)
    displacement = torch.where(valid & (action == 0), displacement, torch.zeros_like(displacement))
    one_hot = F.one_hot(action.clamp(0, 2).long(), 3).to(current_goal.dtype)
    features = torch.cat((valid[:, None].to(current_goal.dtype), one_hot,
                          progress[:, None], displacement[:, None]), -1)
    return torch.where(valid[:, None], features, torch.zeros_like(features))
