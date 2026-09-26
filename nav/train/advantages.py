"""Advantage estimators reusable across policy-gradient baselines."""

from __future__ import annotations

import torch


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
