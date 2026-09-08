"""Inference controller for recurrent PPO PointGoal checkpoints."""

from __future__ import annotations

import numpy as np
import torch

from nav.baselines.rl.pointgoal_ppo import (
    PPO_ACTIONS,
    PPO_BOS_LABEL,
    PointGoalActorCritic,
    PointGoalPPOConfig,
    depth_observation_uint8,
    encode_rl_goal,
)


class PPOPointGoalController:
    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.config = PointGoalPPOConfig(**checkpoint["model_config"])
        self.model = PointGoalActorCritic(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.hidden = self.model.initial_hidden(1, self.device)
        self.previous_action = torch.tensor(
            [PPO_BOS_LABEL], dtype=torch.long, device=self.device
        )
        self.mask = torch.zeros(1, 1, device=self.device)

    def predict_action(
        self,
        *,
        depth_obs,
        curr_world_x: float,
        curr_world_z: float,
        curr_yaw_deg: float,
        target_world_x: float,
        target_world_z: float,
    ) -> str:
        if depth_obs is None:
            raise RuntimeError("PPO PointGoal policy requires depth input")
        depth = torch.from_numpy(
            depth_observation_uint8(np.asarray(depth_obs))[None, ...]
        ).to(self.device)
        goal = torch.from_numpy(
            encode_rl_goal(
                curr_world_x,
                curr_world_z,
                curr_yaw_deg,
                target_world_x,
                target_world_z,
                self.config.goal_distance_scale_m,
            )[None, ...]
        ).to(self.device)
        with torch.no_grad():
            action, _, _, self.hidden = self.model.act(
                depth,
                goal,
                self.previous_action,
                self.hidden,
                self.mask,
                deterministic=True,
            )
        action_index = int(action.item())
        self.previous_action.fill_(action_index)
        self.mask.fill_(1.0)
        return PPO_ACTIONS[action_index]
