"""Recurrent PointGoal actor-critic and PPO math shared by train/eval."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from nav.models.encoder import TimmEncoder


PPO_ACTIONS = ("forward", "turn right", "turn left")
PPO_ACTION_TO_LABEL = {action: index for index, action in enumerate(PPO_ACTIONS)}
PPO_BOS_LABEL = len(PPO_ACTIONS)


@dataclass(frozen=True)
class PointGoalPPOConfig:
    """Architecture and environment-independent PPO hyperparameters."""

    depth_backbone: str = "resnet50"
    img_size: int = 128
    visual_dim: int = 256
    goal_hidden: int = 64
    action_embed_dim: int = 16
    hidden_size: int = 512
    num_actions: int = len(PPO_ACTIONS)
    half_width: bool = False
    pretrained_depth: bool = False
    goal_distance_scale_m: float = 50.0
    analytic_goal_prior_strength: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


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


def depth_observation_uint8(depth_obs: np.ndarray) -> np.ndarray:
    """Normalize a Unity depth observation to one encoded uint8 CHW channel."""
    depth = np.asarray(depth_obs)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            pass
        elif depth.shape[-1] == 1:
            depth = np.moveaxis(depth, -1, 0)
        else:
            depth = depth[:1]
    elif depth.ndim == 2:
        depth = depth[None, ...]
    else:
        raise ValueError(f"Unsupported depth observation shape: {depth.shape}")
    if depth.dtype == np.uint8:
        return depth.copy()
    depth = depth.astype(np.float32)
    if np.nanmax(depth) <= 1.5:
        depth = depth * 255.0
    return np.clip(np.nan_to_num(depth), 0.0, 255.0).astype(np.uint8)


class PointGoalActorCritic(nn.Module):
    """Depth ResNet + PointGoal/action embeddings + a recurrent actor-critic."""

    def __init__(self, config: PointGoalPPOConfig) -> None:
        super().__init__()
        self.config = config
        self.depth_encoder = TimmEncoder(
            model_name=config.depth_backbone,
            in_channels=1,
            out_dim=config.visual_dim,
            pretrained=config.pretrained_depth,
            img_size=config.img_size,
            **(
                {"stem_width": 32, "base_width": 32}
                if config.half_width
                else {}
            ),
        )
        self.goal_mlp = nn.Sequential(
            nn.Linear(3, config.goal_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.goal_hidden, config.goal_hidden),
            nn.ReLU(inplace=True),
        )
        self.action_embed = nn.Embedding(
            config.num_actions + 1, config.action_embed_dim
        )
        feature_dim = (
            config.visual_dim + config.goal_hidden + config.action_embed_dim
        )
        self.gru = nn.GRUCell(feature_dim, config.hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, config.num_actions),
        )
        # A trainable residual initialized as a simple PointGoal controller.
        # This prevents an expensive on-policy run from starting with uniform
        # random wandering while the recurrent/visual branch learns obstacle
        # corrections. Goal layout is [distance, sin(bearing), cos(bearing)].
        self.goal_actor = nn.Linear(3, config.num_actions)
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
            nn.init.zeros_(self.goal_actor.weight)
            nn.init.zeros_(self.goal_actor.bias)
            strength = float(config.analytic_goal_prior_strength)
            self.goal_actor.weight[0, 2] = strength
            self.goal_actor.bias[0] = -0.5
            self.goal_actor.weight[1, 1] = strength
            self.goal_actor.weight[2, 1] = -strength

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.hidden_size, device=device)

    def forward(
        self,
        depth: torch.Tensor,
        goal: torch.Tensor,
        previous_action: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden * masks
        depth = depth.float()
        if depth.max().item() > 1.5:
            depth = depth / 255.0
        if depth.shape[-2:] != (self.config.img_size, self.config.img_size):
            depth = F.interpolate(
                depth,
                size=(self.config.img_size, self.config.img_size),
                mode="nearest",
            )
        visual = self.depth_encoder(depth)
        goal_feature = self.goal_mlp(goal)
        action_feature = self.action_embed(previous_action)
        hidden = self.gru(
            torch.cat((visual, goal_feature, action_feature), dim=-1), hidden
        )
        logits = self.actor(hidden) + self.goal_actor(goal)
        return logits, self.critic(hidden).squeeze(-1), hidden

    def act(
        self,
        depth: torch.Tensor,
        goal: torch.Tensor,
        previous_action: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, hidden = self(
            depth, goal, previous_action, hidden, masks
        )
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value, hidden


def load_bc_depth_encoder(
    model: PointGoalActorCritic, checkpoint_path: str | Path
) -> int:
    """Warm-start the matching depth encoder from a BC checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefix = "depth_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError(f"No {prefix!r} weights in {checkpoint_path}")
    model.depth_encoder.load_state_dict(encoder_state, strict=True)
    return len(encoder_state)


def compute_pointgoal_reward(
    previous_distance_m: float,
    current_distance_m: float,
    *,
    success: bool,
    timeout: bool,
    collision: bool,
    warning: bool,
    progress_scale: float = 1.0,
    step_penalty: float = 0.01,
    success_bonus: float = 5.0,
    timeout_penalty: float = 1.0,
    collision_penalty: float = 0.5,
    warning_penalty: float = 0.05,
) -> float:
    """Safety-aware dense PointGoal reward computed entirely in Python."""
    progress = float(np.clip(previous_distance_m - current_distance_m, -1.0, 1.0))
    reward = progress_scale * progress - step_penalty
    if collision:
        reward -= collision_penalty
    if warning:
        reward -= warning_penalty
    if success:
        reward += success_bonus
    elif timeout:
        reward -= timeout_penalty
    return float(reward)


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    nonterminal_masks: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets for ``[T, N]`` tensors."""
    if rewards.shape != values.shape or rewards.shape != nonterminal_masks.shape:
        raise ValueError("rewards, values, and nonterminal_masks must share shape")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    following_value = next_value
    for step in reversed(range(rewards.shape[0])):
        mask = nonterminal_masks[step]
        delta = rewards[step] + gamma * following_value * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
        following_value = values[step]
    return advantages, advantages + values


def round_robin_scene_tasks(tasks: Iterable[dict], limit: int = 0) -> list[dict]:
    """Interleave scenes while retaining deterministic per-scene task order."""
    by_scene: dict[str, list[dict]] = {}
    for task in tasks:
        by_scene.setdefault(str(task["scene_name"]), []).append(task)
    ordered: list[dict] = []
    depth = 0
    while True:
        added = False
        for scene in sorted(by_scene, key=lambda name: int(name.removeprefix("scene"))):
            if depth < len(by_scene[scene]):
                ordered.append(by_scene[scene][depth])
                added = True
                if limit > 0 and len(ordered) >= limit:
                    return ordered
        if not added:
            return ordered
        depth += 1
