"""Inference wrapper for the learned scene-conditioned waypoint planner."""

from __future__ import annotations

import math

import torch

from nav.models.policies import SceneWaypointConfig, SceneWaypointPlanner


class SceneWaypointController:
    """Convert a final PointGoal into a learned short-horizon local target."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "auto",
        terminal_distance_m: float | None = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.config = SceneWaypointConfig(**checkpoint["model_config"])
        self.model = SceneWaypointPlanner(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.terminal_distance_m = (
            float(self.config.lookahead_m)
            if terminal_distance_m is None
            else float(terminal_distance_m)
        )
        if self.terminal_distance_m < 0.0:
            raise ValueError("terminal_distance_m must be nonnegative")

    def predict_waypoint(
        self,
        *,
        scene_id: int,
        curr_world_x: float,
        curr_world_z: float,
        target_world_x: float,
        target_world_z: float,
    ) -> tuple[float, float]:
        dx = float(target_world_x) - float(curr_world_x)
        dz = float(target_world_z) - float(curr_world_z)
        direct_distance = math.hypot(dx, dz)
        if direct_distance <= self.terminal_distance_m:
            return float(target_world_x), float(target_world_z)
        current = torch.tensor(
            [[curr_world_x, curr_world_z]],
            dtype=torch.float32,
            device=self.device,
        )
        target = torch.tensor(
            [[target_world_x, target_world_z]],
            dtype=torch.float32,
            device=self.device,
        )
        scene = torch.tensor([scene_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            normalized_delta = self.model(scene, current, target)[0]
        waypoint_delta = (
            normalized_delta.float().cpu().numpy()
            * float(self.config.lookahead_m)
        )
        waypoint_norm = math.hypot(
            float(waypoint_delta[0]), float(waypoint_delta[1])
        )
        if not math.isfinite(waypoint_norm) or waypoint_norm < 0.25:
            waypoint_delta = [
                dx / direct_distance * self.config.lookahead_m,
                dz / direct_distance * self.config.lookahead_m,
            ]
            waypoint_norm = float(self.config.lookahead_m)
        elif waypoint_norm > self.config.lookahead_m:
            scale = float(self.config.lookahead_m) / waypoint_norm
            waypoint_delta *= scale
        return (
            float(curr_world_x) + float(waypoint_delta[0]),
            float(curr_world_z) + float(waypoint_delta[1]),
        )


__all__ = ["SceneWaypointController"]
