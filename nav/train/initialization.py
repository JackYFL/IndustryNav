"""Cross-method checkpoint initialization helpers."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace

import torch

from nav.models.policies import (
    PointGoalActorCritic, PointGoalPPOConfig, DaggerTransformerActorCritic,
    MemoryDaggerTransformerActorCritic,
)


def load_bc_depth_encoder(
    model: PointGoalActorCritic, checkpoint_path: str | Path
) -> int:
    """Warm-start a PointGoal actor-critic's depth encoder from BC weights."""
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


def add_persistent_pointgoal_memory(checkpoint: dict, memory_size: int = 256, *,
                                  freeze_base_actor: bool = False, motion_features: bool = False):
    """Preserve a complete PPO actor/critic and audit a zero-residual extension.

    The probes contain synthetic features/relative goals/actions, not benchmark
    episodes or labels. No optimizer state is reused across architecture changes.
    """
    base_config = PointGoalPPOConfig(**checkpoint["model_config"])
    if base_config.policy_architecture != "dagger_transformer":
        raise ValueError("Persistent-memory initialization requires a memory-free Transformer PPO checkpoint")
    config = replace(base_config, policy_architecture="dagger_transformer_memory",
                     persistent_memory_size=memory_size, freeze_base_actor=freeze_base_actor,
                     memory_motion_features=motion_features)
    base = DaggerTransformerActorCritic(base_config)
    base.load_state_dict(checkpoint["model"], strict=True)
    # A motion-enabled initializer extends the SAME seeded context-only memory:
    # old GRU columns/hidden weights are retained, new motion columns start at
    # zero. This allows a paired ablation without changing the random memory.
    model = MemoryDaggerTransformerActorCritic(replace(config, memory_motion_features=False))
    model.load_base_policy(checkpoint["model"])
    if motion_features:
        extended = MemoryDaggerTransformerActorCritic(config)
        state = model.state_dict()
        width = extended.recurrent_memory.input_size - model.recurrent_memory.input_size
        state["recurrent_memory.weight_ih"] = torch.nn.functional.pad(
            state["recurrent_memory.weight_ih"], (0, width))
        extended.load_state_dict(state, strict=True)
        model = extended
    identical = all(torch.equal(value, model.state_dict()[name])
                    for name, value in base.state_dict().items())
    if not identical or not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("Memory initialization changed original weights or contains nonfinite tensors")
    generator = torch.Generator().manual_seed(20260913)
    history = torch.full((2, config.action_history_len), 3, dtype=torch.long)
    old_hidden, hidden = base.initial_hidden(2, "cpu"), model.initial_hidden(2, "cpu")
    errors, value_errors = [], []
    probes = 2 * base.seq_len + 3
    with torch.no_grad():
        for step in range(probes):
            visual = torch.randn(2, config.visual_dim, generator=generator)
            angle = torch.rand(2, generator=generator) * (2 * torch.pi) - torch.pi
            goal = torch.stack((torch.rand(2, generator=generator), angle.sin(), angle.cos()), -1)
            masks = torch.ones(2, 1)
            if step in (0, base.seq_len + 1):
                masks[0] = 0
                history[0].fill_(3)
            if step in (0, base.seq_len + 2):
                masks[1] = 0
                history[1].fill_(3)
            before, old_value, old_hidden, _ = base(visual, goal, history, None, old_hidden, masks)
            after, value, hidden, _ = model(visual, goal, history, None, hidden, masks)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            torch.testing.assert_close(old_value, value, rtol=0, atol=0)
            errors.append(float((before - after).abs().max()))
            value_errors.append(float((old_value - value).abs().max()))
            executed = (torch.arange(2) + step) % 3
            history = torch.cat((history[:, 1:], executed[:, None]), 1)
    audit = dict(original_weights_identical=identical, synthetic_probes=probes,
                 maximum_logit_error=max(errors), maximum_value_error=max(value_errors),
                 memory_size=memory_size, source_update=checkpoint.get("update"),
                 freeze_base_actor=freeze_base_actor,
                 motion_features=motion_features,
                 motion_input_initialization="zero-extended context-memory GRU" if motion_features else None,
                 source_global_steps=checkpoint.get("global_steps"),
                 initialization="Zero residual actor/value heads; all original actor/critic weights retained")
    payload = dict(model=model.state_dict(), model_config=asdict(config),
                   update=0, global_steps=0, initialization=audit)
    return payload, audit
