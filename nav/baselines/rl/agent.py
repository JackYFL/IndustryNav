"""Inference agent for recurrent PPO PointGoal checkpoints."""

from __future__ import annotations

import numpy as np
import torch

from nav.baselines.rl.actions import (
    PPO_ACTIONS,
    PPO_ACTION_TO_LABEL,
    append_action_history,
    initial_action_history,
)
from nav.core.agent import NavigationAgent
from nav.core.pointgoal import encode_rl_goal_state
from nav.data.preprocessing import depth_observation_float32, depth_observation_uint8
from nav.models.policies import (
    PointGoalActorCritic,
    PointGoalPPOConfig,
    append_visual_history,
    initial_visual_history,
    load_compatible_pointgoal_state_dict,
    build_pointgoal_actor_critic,
)


class PPOPointGoalController(NavigationAgent):
    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.config = PointGoalPPOConfig(**checkpoint["model_config"])
        if self.config.uses_dagger_transformer:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = build_pointgoal_actor_critic(self.config).to(self.device)
        load_compatible_pointgoal_state_dict(self.model, checkpoint["model"])
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.hidden = self.model.initial_hidden(1, self.device)
        self.action_history = initial_action_history(
            1,
            self.config.action_history_len,
            device=self.device,
        )
        self.visual_history = initial_visual_history(
            1,
            self.config.visual_history_len,
            self.config.visual_dim,
            device=self.device,
        )
        self.mask = torch.zeros(1, 1, device=self.device)

    def observe_executed_action(self, action: str) -> None:
        """Replace the newest history entry after an external intervention.

        ``predict_action`` records the policy sample immediately.  A safety
        shield may subsequently execute a different action, so the recurrent
        policy must be told what actually happened before its next decision.
        """
        if action not in PPO_ACTION_TO_LABEL:
            raise ValueError(f"Unsupported executed PPO action: {action!r}")
        self.action_history[:, -1] = PPO_ACTION_TO_LABEL[action]

    def predict_action(
        self,
        *,
        depth_obs,
        curr_world_x: float,
        curr_world_z: float,
        curr_yaw_deg: float,
        target_world_x: float,
        target_world_z: float,
        scene_id: int | None = None,
    ) -> str:
        if depth_obs is None:
            raise RuntimeError("PPO PointGoal policy requires depth input")
        preprocess = (
            depth_observation_float32
            if self.config.uses_dagger_transformer
            else depth_observation_uint8
        )
        depth = torch.from_numpy(preprocess(np.asarray(depth_obs))[None, ...]).to(self.device)
        goal = torch.from_numpy(
            encode_rl_goal_state(
                curr_world_x,
                curr_world_z,
                curr_yaw_deg,
                target_world_x,
                target_world_z,
                distance_scale_m=self.config.goal_distance_scale_m,
                scene_id=scene_id,
                num_scenes=self.config.num_scenes,
                world_coordinate_scale_m=self.config.world_coordinate_scale_m,
                include_absolute_scene_state=self.config.absolute_scene_state,
                coordinate_fourier_bands=self.config.coordinate_fourier_bands,
            )[None, ...]
        ).to(self.device)
        with torch.no_grad():
            action, _, _, self.hidden, current_visual = self.model.act(
                depth,
                goal,
                self.action_history,
                self.visual_history,
                self.hidden,
                self.mask,
                deterministic=True,
            )
        action_index = int(action.item())
        self.action_history = append_action_history(
            self.action_history,
            torch.tensor([action_index], device=self.device),
        )
        self.visual_history = append_visual_history(
            self.visual_history,
            current_visual,
        )
        self.mask.fill_(1.0)
        return PPO_ACTIONS[action_index]


PPOPointGoalAgent = PPOPointGoalController

__all__ = ["PPOPointGoalAgent", "PPOPointGoalController"]
