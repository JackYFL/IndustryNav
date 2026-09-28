"""Scene-conditioned waypoint planner for hierarchical PointGoal navigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SceneWaypointConfig:
    """Architecture and normalization used by :class:`SceneWaypointPlanner`."""

    num_scenes: int = 24
    coordinate_scale_m: float = 50.0
    coordinate_fourier_bands: int = 4
    scene_embed_dim: int = 32
    hidden_size: int = 256
    lookahead_m: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


class SceneWaypointPlanner(nn.Module):
    """Predict a short world-frame route offset from GPS and scene identity.

    The network learns static layout structure from independent A* trajectories.
    It never consumes minimap pixels.  Its output is an offset divided by the
    configured lookahead, so each component is naturally near ``[-1, 1]``.
    """

    def __init__(self, config: SceneWaypointConfig) -> None:
        super().__init__()
        if config.num_scenes <= 0:
            raise ValueError("num_scenes must be positive")
        if config.coordinate_scale_m <= 0.0 or config.lookahead_m <= 0.0:
            raise ValueError("coordinate scale and lookahead must be positive")
        if config.coordinate_fourier_bands < 0:
            raise ValueError("coordinate_fourier_bands must be nonnegative")
        if config.scene_embed_dim <= 0 or config.hidden_size <= 0:
            raise ValueError("embedding and hidden sizes must be positive")
        self.config = config
        self.scene_embedding = nn.Embedding(
            config.num_scenes, config.scene_embed_dim
        )
        coordinate_features = 4 + 2 + 8 * config.coordinate_fourier_bands
        input_dim = coordinate_features + config.scene_embed_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_size // 2, 2),
            nn.Tanh(),
        )

    def _coordinate_features(
        self,
        current_world_xz: torch.Tensor,
        target_world_xz: torch.Tensor,
    ) -> torch.Tensor:
        if current_world_xz.ndim != 2 or current_world_xz.shape[-1] != 2:
            raise ValueError("current_world_xz must have shape [batch, 2]")
        if target_world_xz.shape != current_world_xz.shape:
            raise ValueError("target_world_xz must match current_world_xz")
        scale = float(self.config.coordinate_scale_m)
        coordinates = torch.cat(
            (current_world_xz, target_world_xz), dim=-1
        ) / scale
        delta = (target_world_xz - current_world_xz) / scale
        features = [coordinates, delta]
        if self.config.coordinate_fourier_bands:
            frequencies = 2.0 * torch.pi * torch.pow(
                coordinates.new_tensor(2.0),
                torch.arange(
                    self.config.coordinate_fourier_bands,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                ),
            )
            phases = coordinates.unsqueeze(-1) * frequencies
            features.extend(
                (torch.sin(phases).flatten(start_dim=1),
                 torch.cos(phases).flatten(start_dim=1))
            )
        return torch.cat(features, dim=-1)

    def forward(
        self,
        scene_id: torch.Tensor,
        current_world_xz: torch.Tensor,
        target_world_xz: torch.Tensor,
    ) -> torch.Tensor:
        scene_id = scene_id.reshape(-1).long()
        if scene_id.shape[0] != current_world_xz.shape[0]:
            raise ValueError("scene_id and coordinates must use the same batch")
        if torch.any(scene_id < 0) or torch.any(scene_id >= self.config.num_scenes):
            raise ValueError("scene_id is outside the configured scene range")
        features = self._coordinate_features(current_world_xz, target_world_xz)
        return self.network(
            torch.cat((features, self.scene_embedding(scene_id)), dim=-1)
        )


__all__ = ["SceneWaypointConfig", "SceneWaypointPlanner"]
