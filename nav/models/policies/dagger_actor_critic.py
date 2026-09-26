"""Preserve a DAgger Transformer actor while learning a PPO value head.

The fixed depth encoder makes cached window features exact, including during
PPO re-evaluation. Goal/action embeddings and all Transformer/actor weights
remain trainable. The critic receives detached features so a new value head
cannot erase the pretrained actor through its shared representation.
"""

from __future__ import annotations

import math
import torch
from torch import nn

from nav.core.pointgoal import POINTGOAL_ENCODING_UNITY
from nav.models.policies.bc import NavPolicyTransformer
from nav.models.policies.pointgoal_actor_critic import (
    PointGoalActorCritic,
    PointGoalPPOConfig,
)


def dagger_ppo_config(bc_config: dict) -> PointGoalPPOConfig:
    """Validate a depth-only navigation actor and retain its full architecture."""
    required = {
        "policy_type": "transformer",
        "use_depth": True,
        "use_rgb": False,
        "navigation_only_actions": True,
        "chunk_size": 1,
        "goal_encoding": POINTGOAL_ENCODING_UNITY,
    }
    for name, expected in required.items():
        if bc_config.get(name) != expected:
            raise ValueError(f"DAgger PPO requires {name}={expected!r}")
    if bc_config.get("goal_rep") not in {"polar", "cartesian"}:
        raise ValueError("DAgger PPO requires a polar or cartesian point goal")
    if float(bc_config.get("goal_distance_scale_m", 0)) <= 0:
        raise ValueError("DAgger PPO requires a positive goal distance scale")
    if int(bc_config.get("seq_len", 0)) < 6:
        raise ValueError("DAgger PPO requires at least five historical frames")
    return PointGoalPPOConfig(
        policy_architecture="dagger_transformer",
        bc_actor_config=dict(bc_config),
        depth_backbone=bc_config["depth_backbone"],
        img_size=int(bc_config["img_size"]),
        half_width=bool(bc_config.get("half_width", False)),
        goal_distance_scale_m=float(bc_config["goal_distance_scale_m"]),
        action_history_len=int(bc_config["seq_len"]),
        visual_history_len=int(bc_config["seq_len"]) - 1,
    )


class DaggerTransformerActorCritic(NavPolicyTransformer):
    """Same actor state dict and window padding as ``BCNavController``."""

    value_parameter_prefixes = ("critic.",)
    adaptation_parameter_prefixes = ()

    def __init__(self, config: PointGoalPPOConfig):
        bc = config.bc_actor_config
        if bc is None:
            raise ValueError("Missing bc_actor_config")
        dagger_ppo_config(bc)
        super().__init__(
            goal_dim=2, num_actions=3, use_depth=True, use_rgb=False,
            seq_len=int(bc["seq_len"]), num_layers=int(bc["num_layers"]),
            depth_backbone=bc["depth_backbone"],
            pretrained_depth=False, pretrained_rgb=False,
            half_width=bool(bc.get("half_width", False)),
            img_size=int(bc["img_size"]), chunk_size=1,
            goal_action_residual=bool(bc.get("goal_action_residual", False)),
            previous_action_vocab_size=4,
        )
        self.config = config
        self.depth_encoder.requires_grad_(False)
        width = self.pos_embed.shape[-1]
        self.critic = nn.Sequential(
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1),
        )
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        if config.freeze_dagger_transformer:
            for name, parameter in self.named_parameters():
                if not name.startswith(("head.", "critic.")):
                    parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # PPO likelihoods must use the same deterministic network in rollout
        # and optimization. eval() leaves autograd enabled.
        return super().train(False)

    def load_dagger_actor(self, state_dict: dict) -> None:
        result = self.load_state_dict(state_dict, strict=False)
        expected = {name for name in self.state_dict() if name.startswith("critic.")}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError(f"DAgger actor initialization mismatch: {result}")

    def initial_hidden(self, batch_size, device):
        # Each cached frame stores frozen vision, raw polar/cartesian goal,
        # and its previous executed action. Embeddings are recomputed by PPO.
        return torch.zeros(
            batch_size, self.seq_len - 1, self.config.visual_dim + 3,
            device=device,
        )

    encode_depth = PointGoalActorCritic.encode_depth
    act = PointGoalActorCritic.act

    def forward(self, depth, goal, previous_action, visual_history, hidden, masks):
        # The trainer can replay cached features because the encoder is frozen.
        with torch.no_grad():
            visual = depth if depth.ndim == 2 else self.encode_depth(depth)
        return self.forward_from_visual(
            visual, goal, previous_action, visual_history, hidden, masks,
        )

    def forward_from_visual(
        self, current_visual, goal, previous_action, visual_history, hidden, masks,
    ):
        context, bc_goal, next_window = self._window_context(
            current_visual, goal, previous_action, hidden, masks,
        )
        logits = self._actor_logits(context, bc_goal)
        value = self.critic(context.detach()).squeeze(-1)
        return logits, value, next_window, current_visual

    def reference_window_hidden(self, hidden):
        """Raw window suitable for the frozen, memory-free reference actor."""
        return hidden

    def _window_context(self, current_visual, goal, previous_action, hidden, masks):
        if self.config.bc_actor_config["goal_rep"] == "polar":
            bc_goal = torch.stack((goal[:, 0], torch.atan2(goal[:, 1], goal[:, 2]) / math.pi), -1)
        else:
            bc_goal = torch.stack((goal[:, 0] * goal[:, 2], goal[:, 0] * goal[:, 1]), -1)
        current = torch.cat((current_visual, bc_goal, previous_action[:, -1:].float()), -1)
        reset = (masks.reshape(-1) == 0)[:, None, None]
        first = current[:, None, :].expand(-1, self.seq_len - 1, -1).clone()
        first[..., -1] = 3  # BOS padding matches BCNavController exactly.
        history = torch.where(reset, first, hidden)
        window = torch.cat((history, current[:, None, :]), 1)
        feats = torch.cat((
            window[..., :self.config.visual_dim],
            self.goal_mlp(window[..., -3:-1]),
            self.action_embed(window[..., -1].long()),
        ), -1)
        tokens = self.token_proj(feats) + self.pos_embed
        context = self._run_encoder(tokens)[:, -1]
        return context, bc_goal, window[:, 1:].detach()

    def _actor_logits(self, context, bc_goal):
        logits = self.head(context)
        if self.goal_action_head is not None:
            logits = logits + self.goal_action_head(bc_goal)
        return logits


def build_pointgoal_actor_critic(config: PointGoalPPOConfig) -> nn.Module:
    if config.memory_motion_features and config.policy_architecture != "dagger_transformer_memory":
        raise ValueError("Motion features require persistent Transformer memory")
    if config.policy_architecture == "dagger_transformer":
        return DaggerTransformerActorCritic(config)
    if config.policy_architecture == "dagger_transformer_memory":
        from nav.models.policies.memory_dagger_actor_critic import MemoryDaggerTransformerActorCritic
        return MemoryDaggerTransformerActorCritic(config)
    if config.policy_architecture != "gru":
        raise ValueError(f"Unknown PPO architecture: {config.policy_architecture}")
    return PointGoalActorCritic(config)
