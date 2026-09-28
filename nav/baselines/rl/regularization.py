"""Policy-preservation objectives for on-policy PPO (no expert labels)."""

import math

import torch


def temperature_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Parameterize the stochastic training policy without changing argmax.

    Rollout likelihoods, PPO replay, entropy and reference KL must all use the
    same temperature. The default returns the original tensor unchanged.
    This is not an evaluation-time exploration or action override mechanism.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Policy temperature must be finite and positive")
    return logits if temperature == 1.0 else logits / temperature


def reference_kl(logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Exact categorical KL(current || frozen reference), one value per state."""
    if logits.shape != reference_logits.shape:
        raise ValueError("Current and reference action distributions must match")
    logp = logits.log_softmax(-1)
    logq = reference_logits.detach().log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0.0)


def reference_coefficient(update: int, initial: float, final: float, decay_updates: int) -> float:
    if min(initial, final, decay_updates) < 0 or final > initial:
        raise ValueError("Reference KL schedule requires 0 <= final <= initial")
    fraction = min(max((update - 1) / max(1, decay_updates), 0.0), 1.0)
    return initial + (final - initial) * fraction if decay_updates else initial
