"""Recurrent PointGoal actor-critic architecture used by PPO baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from nav.models.encoders import TimmEncoder


@dataclass(frozen=True)
class PointGoalPPOConfig:
    """Architecture configuration for :class:`PointGoalActorCritic`."""

    depth_backbone: str = "resnet50"
    img_size: int = 128
    visual_dim: int = 256
    goal_hidden: int = 64
    action_embed_dim: int = 16
    # Keep one as the model-level compatibility default for checkpoints that
    # predate explicit action histories. New training sets this to five.
    action_history_len: int = 1
    # Zero preserves checkpoints that predate explicit visual-feature memory.
    # New training uses five previous encoder features.
    visual_history_len: int = 0
    hidden_size: int = 512
    num_actions: int = 3
    half_width: bool = False
    pretrained_depth: bool = False
    goal_distance_scale_m: float = 50.0
    analytic_goal_prior_strength: float = 3.0
    absolute_scene_state: bool = False
    num_scenes: int = 24
    world_coordinate_scale_m: float = 50.0
    coordinate_fourier_bands: int = 0
    # Optional nonlinear GPS/Compass route residual. Zero preserves all
    # historical checkpoints and the relative-only PointGoal architecture.
    route_actor_hidden: int = 0
    base_policy_logit_scale: float = 1.0
    route_actor_logit_scale: float = 1.0
    policy_architecture: str = "gru"
    bc_actor_config: dict | None = None
    freeze_dagger_transformer: bool = False
    persistent_memory_size: int = 0
    freeze_base_actor: bool = False
    # Optional, relative-observation-only transition cues for persistent memory.
    # False preserves all existing checkpoint architectures and tensor shapes.
    memory_motion_features: bool = False

    @property
    def uses_dagger_transformer(self) -> bool:
        return self.policy_architecture in {"dagger_transformer", "dagger_transformer_memory"}

    @property
    def goal_input_dim(self) -> int:
        # Relative PointGoal + legacy GPS/scene context + appended planning
        # context (world-frame delta and explicit sine/cosine yaw).
        return 3 + (
            8 + self.num_scenes + 8 * self.coordinate_fourier_bands
            if self.absolute_scene_state
            else 0
        )

    def to_dict(self) -> dict:
        return asdict(self)


class PointGoalActorCritic(nn.Module):
    """Depth encoder + PointGoal/action embeddings + recurrent actor-critic."""

    def __init__(self, config: PointGoalPPOConfig) -> None:
        super().__init__()
        if config.action_history_len <= 0:
            raise ValueError("action_history_len must be positive")
        if config.visual_history_len < 0:
            raise ValueError("visual_history_len must be nonnegative")
        if config.coordinate_fourier_bands < 0:
            raise ValueError("coordinate_fourier_bands must be nonnegative")
        if config.coordinate_fourier_bands and not config.absolute_scene_state:
            raise ValueError(
                "coordinate_fourier_bands requires absolute_scene_state"
            )
        if config.route_actor_hidden < 0:
            raise ValueError("route_actor_hidden must be nonnegative")
        if config.route_actor_hidden and not config.absolute_scene_state:
            raise ValueError("route_actor_hidden requires absolute_scene_state")
        if config.base_policy_logit_scale < 0.0:
            raise ValueError("base_policy_logit_scale must be nonnegative")
        if config.route_actor_logit_scale < 0.0:
            raise ValueError("route_actor_logit_scale must be nonnegative")
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
            nn.Linear(config.goal_input_dim, config.goal_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.goal_hidden, config.goal_hidden),
            nn.ReLU(inplace=True),
        )
        self.action_embed = nn.Embedding(
            config.num_actions + 1, config.action_embed_dim
        )
        if config.visual_history_len:
            self.visual_history_gru = nn.GRU(
                config.visual_dim,
                config.visual_dim,
                batch_first=True,
            )
            # Zero gating makes migration initially identical to the old
            # current-frame-only visual path.
            self.visual_history_gate = nn.Parameter(
                torch.zeros(config.visual_dim)
            )
        feature_dim = (
            config.visual_dim
            + config.goal_hidden
            + config.action_embed_dim * config.action_history_len
        )
        self.gru = nn.GRUCell(feature_dim, config.hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, config.num_actions),
        )
        # A trainable residual initialized as a simple PointGoal controller.
        # This avoids starting on-policy learning with uniform random walking.
        self.goal_actor = nn.Linear(config.goal_input_dim, config.num_actions)
        if config.route_actor_hidden:
            route_hidden = int(config.route_actor_hidden)
            self.route_actor = nn.Sequential(
                nn.Linear(config.goal_input_dim, route_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(route_hidden, route_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(route_hidden, config.num_actions),
            )
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
            if config.route_actor_hidden:
                nn.init.zeros_(self.route_actor[-1].weight)
                nn.init.zeros_(self.route_actor[-1].bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.hidden_size, device=device)

    def forward(
        self,
        depth: torch.Tensor,
        goal: torch.Tensor,
        previous_action: torch.Tensor,
        visual_history: torch.Tensor | None,
        hidden: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_visual = self.encode_depth(depth)
        return self.forward_from_visual(
            current_visual,
            goal,
            previous_action,
            visual_history,
            hidden,
            masks,
        )

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode depth using exactly the same preprocessing as online use."""
        depth = depth.float()
        if depth.max().item() > 1.5:
            depth = depth / 255.0
        if depth.shape[-2:] != (self.config.img_size, self.config.img_size):
            depth = F.interpolate(
                depth,
                size=(self.config.img_size, self.config.img_size),
                mode="nearest",
            )
        return self.depth_encoder(depth)

    def forward_from_visual(
        self,
        current_visual: torch.Tensor,
        goal: torch.Tensor,
        previous_action: torch.Tensor,
        visual_history: torch.Tensor | None,
        hidden: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run temporal heads from precomputed current-frame features.

        The offline imitation warm-up freezes the depth backbone and encodes
        many frames at once.  Reusing this head keeps its recurrent semantics
        identical to :meth:`forward` while making that training practical.
        """
        if (
            current_visual.ndim != 2
            or current_visual.shape[1] != self.config.visual_dim
        ):
            raise ValueError(
                "current visual features must have shape "
                f"[batch, {self.config.visual_dim}]"
            )
        batch_size = current_visual.shape[0]
        hidden = hidden * masks
        visual = current_visual
        if self.config.visual_history_len:
            expected_visual_shape = (
                batch_size,
                self.config.visual_history_len,
                self.config.visual_dim,
            )
            actual_visual_shape = (
                None if visual_history is None else tuple(visual_history.shape)
            )
            if actual_visual_shape != expected_visual_shape:
                raise ValueError(
                    "visual history must have shape "
                    f"{expected_visual_shape}, got {actual_visual_shape}"
                )
            # A zero recurrent mask also blocks stale cached features if a
            # caller forgets to reset its external history at an episode edge.
            visual_history = visual_history * masks.unsqueeze(-1)
            visual_sequence = torch.cat(
                (visual_history, current_visual.unsqueeze(1)), dim=1
            )
            _, temporal_state = self.visual_history_gru(visual_sequence)
            visual = current_visual + torch.tanh(
                self.visual_history_gate
            ) * torch.tanh(temporal_state[-1])
        if tuple(goal.shape) != (batch_size, self.config.goal_input_dim):
            raise ValueError(
                "goal state must have shape "
                f"{(batch_size, self.config.goal_input_dim)}, "
                f"got {tuple(goal.shape)}"
            )
        goal_feature = self.goal_mlp(goal)
        if previous_action.ndim == 1:
            previous_action = previous_action.unsqueeze(-1)
        expected_shape = (batch_size, self.config.action_history_len)
        if tuple(previous_action.shape) != expected_shape:
            raise ValueError(
                "action history must have shape "
                f"{expected_shape}, got {tuple(previous_action.shape)}"
            )
        action_feature = self.action_embed(previous_action).flatten(start_dim=1)
        hidden = self.gru(
            torch.cat((visual, goal_feature, action_feature), dim=-1), hidden
        )
        logits = self.actor(hidden) + self.goal_actor(goal)
        if self.config.route_actor_hidden:
            logits = (
                self.config.base_policy_logit_scale * logits
                + self.config.route_actor_logit_scale * self.route_actor(goal)
            )
        return logits, self.critic(hidden).squeeze(-1), hidden, current_visual

    def act(
        self,
        depth: torch.Tensor,
        goal: torch.Tensor,
        previous_action: torch.Tensor,
        visual_history: torch.Tensor | None,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits, value, hidden, current_visual = self(
            depth, goal, previous_action, visual_history, hidden, masks
        )
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value, hidden, current_visual


def initial_visual_history(
    batch_size: int,
    history_len: int,
    visual_dim: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Return an empty explicit visual-feature history, or ``None`` if disabled."""
    if history_len < 0 or batch_size <= 0 or visual_dim <= 0:
        raise ValueError(
            "batch_size and visual_dim must be positive; history_len must be nonnegative"
        )
    if history_len == 0:
        return None
    return torch.zeros(
        batch_size,
        history_len,
        visual_dim,
        dtype=torch.float32,
        device=device,
    )


def append_visual_history(
    history: torch.Tensor | None,
    current_visual: torch.Tensor,
    *,
    episode_done: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Append current encoder features and clear histories for finished episodes."""
    if history is None:
        return None
    if history.ndim != 3:
        raise ValueError("visual history must have shape [batch, history, feature]")
    current_visual = current_visual.detach().to(
        device=history.device,
        dtype=history.dtype,
    )
    if tuple(current_visual.shape) != (history.shape[0], history.shape[2]):
        raise ValueError("current visual features do not match history shape")
    updated = torch.cat(
        (history[:, 1:], current_visual.unsqueeze(1)), dim=1
    )
    if episode_done is not None:
        done = episode_done.reshape(-1).to(device=history.device, dtype=torch.bool)
        if done.shape[0] != history.shape[0]:
            raise ValueError("episode_done and history must use the same batch size")
        updated = updated.masked_fill(done[:, None, None], 0.0)
    return updated


def load_compatible_pointgoal_state_dict(
    model: PointGoalActorCritic,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, object] | None:
    """Load a checkpoint, initializing newly requested temporal inputs.

    The old action slot is aligned with the newest slot of the requested
    history. Newly introduced older slots start with zero GRU input weights,
    while the visual-history branch starts behind a zero residual gate. Thus
    migration initially reproduces the older checkpoint exactly.
    """
    if model.config.uses_dagger_transformer:
        model.load_state_dict(state_dict, strict=True)
        return None
    key = "gru.weight_ih"
    target_state = model.state_dict()
    source_weight = state_dict.get(key)
    target_weight = target_state[key]
    if source_weight is None:
        raise RuntimeError("Checkpoint is missing gru.weight_ih")
    adapted_state = dict(state_dict)
    migration: dict[str, object] = {}

    goal_widths = []
    for goal_key in ("goal_mlp.0.weight", "goal_actor.weight"):
        source_goal = adapted_state.get(goal_key)
        target_goal = target_state[goal_key]
        if source_goal is None:
            raise RuntimeError(f"Checkpoint is missing {goal_key}")
        if source_goal.shape == target_goal.shape:
            continue
        if (
            source_goal.ndim != 2
            or target_goal.ndim != 2
            or source_goal.shape[0] != target_goal.shape[0]
            or source_goal.shape[1] > target_goal.shape[1]
        ):
            raise RuntimeError(
                f"Cannot migrate {goal_key} from {tuple(source_goal.shape)} "
                f"to {tuple(target_goal.shape)}"
            )
        expanded_goal = torch.zeros_like(target_goal)
        expanded_goal[:, : source_goal.shape[1]] = source_goal
        adapted_state[goal_key] = expanded_goal
        goal_widths.append((source_goal.shape[1], target_goal.shape[1]))
    if goal_widths:
        if len(set(goal_widths)) != 1:
            raise RuntimeError("PointGoal context widths disagree across actor inputs")
        migration["goal_state"] = goal_widths[0]

    config = model.config
    base_features = config.visual_dim + config.goal_hidden
    action_width = config.action_embed_dim

    def history_slots(input_features: int) -> int:
        action_features = input_features - base_features
        if action_features <= 0 or action_features % action_width:
            raise RuntimeError(
                "Cannot infer action-history length from checkpoint GRU shape: "
                f"{tuple(source_weight.shape)}"
            )
        return action_features // action_width

    if source_weight.shape != target_weight.shape:
        source_slots = history_slots(source_weight.shape[1])
        target_slots = history_slots(target_weight.shape[1])
        if source_weight.shape[0] != target_weight.shape[0]:
            raise RuntimeError(
                "Checkpoint GRU hidden size does not match the requested policy"
            )

        adapted_weight = torch.zeros_like(target_weight)
        adapted_weight[:, :base_features] = source_weight[:, :base_features]
        shared_slots = min(source_slots, target_slots)
        source_start = base_features + (source_slots - shared_slots) * action_width
        target_start = base_features + (target_slots - shared_slots) * action_width
        adapted_weight[:, target_start:] = source_weight[:, source_start:]
        adapted_state[key] = adapted_weight
        migration["action_history"] = (source_slots, target_slots)

    visual_keys = [
        name
        for name in target_state
        if name == "visual_history_gate" or name.startswith("visual_history_gru.")
    ]
    initialized_visual_keys = [
        name for name in visual_keys if name not in adapted_state
    ]
    for name in initialized_visual_keys:
        adapted_state[name] = target_state[name]
    if initialized_visual_keys:
        migration["visual_history"] = (0, config.visual_history_len)

    route_keys = [name for name in target_state if name.startswith("route_actor.")]
    initialized_route_keys = [name for name in route_keys if name not in adapted_state]
    for name in initialized_route_keys:
        adapted_state[name] = target_state[name]
    if initialized_route_keys:
        migration["route_actor"] = (0, config.route_actor_hidden)

    model.load_state_dict(adapted_state, strict=True)
    return migration or None
