"""Persistent residual memory on top of the preserved PointGoal actor.

Only depth/relative-goal/executed-action history enters the policy. Zero output
heads preserve the complete initialized actor and critic exactly. Raw window
features remain detached; the learned memory retains its graph within BPTT.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nav.models.policies.dagger_actor_critic import DaggerTransformerActorCritic
from nav.models.policies.relative_motion import MOTION_FEATURE_DIM, relative_motion_features


class MemoryDaggerTransformerActorCritic(DaggerTransformerActorCritic):
    value_parameter_prefixes = ("critic.", "critic_memory.")
    adaptation_parameter_prefixes = ("recurrent_memory.", "memory_actor.")
    memory_parameter_prefixes = adaptation_parameter_prefixes + ("critic_memory.",)

    def __init__(self, config):
        if config.policy_architecture != "dagger_transformer_memory":
            raise ValueError("Persistent memory requires its explicit architecture identifier")
        if type(config.persistent_memory_size) is not int or config.persistent_memory_size <= 0:
            raise ValueError("persistent_memory_size must be a positive integer")
        if type(config.memory_motion_features) is not bool:
            raise ValueError("memory_motion_features must be a boolean")
        super().__init__(config)
        if config.freeze_base_actor:
            for name, parameter in self.named_parameters():
                if not name.startswith(self.value_parameter_prefixes):
                    parameter.requires_grad_(False)
        size = config.persistent_memory_size
        self.frame_width = config.visual_dim + 3
        self.memory_rows = math.ceil(size / self.frame_width)
        motion_width = MOTION_FEATURE_DIM if config.memory_motion_features else 0
        self.recurrent_memory = nn.GRUCell(self.pos_embed.shape[-1] + motion_width, size)
        self.memory_actor = nn.Linear(size, config.num_actions)
        self.critic_memory = nn.Linear(size, 1)
        # Start with a relatively persistent update gate; the recurrent weights
        # are learned only through PPO, never through an expert label loss.
        nn.init.zeros_(self.recurrent_memory.bias_ih)
        nn.init.zeros_(self.recurrent_memory.bias_hh)
        with torch.no_grad():
            self.recurrent_memory.bias_hh[size:2 * size].fill_(3.)
        for head in (self.memory_actor, self.critic_memory):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.eval()

    def load_base_policy(self, state_dict):
        result = self.load_state_dict(state_dict, strict=False)
        expected = {name for name in self.state_dict() if name.startswith(self.memory_parameter_prefixes)}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError(f"Persistent-memory initialization mismatch: {result}")

    def initial_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.seq_len - 1 + self.memory_rows,
                           self.frame_width, device=device)

    def reference_window_hidden(self, hidden):
        # The frozen reference has no learned memory and must never consume the
        # current actor's recurrent state in place of a historical visual frame.
        return hidden[:, :self.seq_len - 1]

    def forward_from_visual(self, current_visual, goal, previous_action, visual_history, hidden, masks):
        expected = (self.seq_len - 1 + self.memory_rows, self.frame_width)
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected:
            raise ValueError(f"Unexpected packed memory state: {tuple(hidden.shape)}")
        context, bc_goal, next_window = self._window_context(
            current_visual, goal, previous_action, self.reference_window_hidden(hidden), masks,
        )
        memory = hidden[:, self.seq_len - 1:].reshape(hidden.shape[0], -1)
        memory = memory[:, :self.config.persistent_memory_size] * masks.reshape(-1, 1)
        memory_input = context
        if self.config.memory_motion_features:
            motion = relative_motion_features(
                hidden[:, self.seq_len - 2, -3:-1], bc_goal,
                previous_action[:, -1], masks,
                goal_rep=self.config.bc_actor_config["goal_rep"],
                distance_scale_m=self.config.goal_distance_scale_m,
            )
            memory_input = torch.cat((context, motion), -1)
        memory = self.recurrent_memory(memory_input, memory)
        logits = self._actor_logits(context, bc_goal) + self.memory_actor(memory)
        value = (self.critic(context.detach()).squeeze(-1)
                 + self.critic_memory(memory.detach()).squeeze(-1))
        packed_memory = F.pad(memory, (0, self.memory_rows * self.frame_width - memory.shape[-1]))
        packed_memory = packed_memory.reshape(memory.shape[0], self.memory_rows, self.frame_width)
        # Keep the learned state connected across the current BPTT chunk.
        next_hidden = torch.cat((next_window, packed_memory), 1)
        return logits, value, next_hidden, current_visual
