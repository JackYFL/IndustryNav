"""Explicit optimizer-resume controls and read-only gradient diagnostics."""

import math

import torch


def restore_optimizer_state(optimizer, state_dict: dict, *, use_config_lrs: bool = False) -> None:
    """Restore moments; optionally use the newly configured learning rates.

    Default behavior is exactly torch's ordinary checkpoint restore. Opting
    into new rates requires matching named groups and parameter counts; rates
    are the only saved group setting overridden (not eps, decay or betas).
    """
    rates = None
    if use_config_lrs:
        current, saved = optimizer.param_groups, state_dict["param_groups"]
        names = [group.get("group_name") for group in current]
        if (len(current) != len(saved) or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Configured-rate resume requires unique matching named optimizer groups")
        for group, old in zip(current, saved):
            if group["group_name"] != old.get("group_name") or len(group["params"]) != len(old["params"]):
                raise ValueError("Optimizer groups or parameter counts changed during rate override")
            if not math.isfinite(float(group["lr"])) or group["lr"] <= 0:
                raise ValueError("Configured optimizer learning rates must be finite and positive")
        rates = {group["group_name"]: group["lr"] for group in current}
    optimizer.load_state_dict(state_dict)
    if rates is not None:
        for group in optimizer.param_groups:
            group["lr"] = rates[group["group_name"]]


def gradient_l2_norm(parameters) -> torch.Tensor:
    """Measure pre-clipping gradients without changing gradients or state."""
    parameters = list(parameters)
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return parameters[0].new_zeros(()) if parameters else torch.tensor(0.)
    return torch.stack(norms).norm(2)


def clip_policy_value_gradients(model, max_norm: float, *, mode: str = "global") -> torch.Tensor:
    """Optionally decouple actor/value norm clipping on detached-critic actors.

    Return the pre-clipping total norm in either mode. Default delegates to the
    exact original PyTorch operation. Separate mode permits each disjoint group
    its own max_norm (so the combined norm can reach sqrt(2) * max_norm).
    It neither changes the loss nor creates actor gradients during warmup.
    """
    if mode == "global":
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if mode != "actor_value":
        raise ValueError(f"Unknown gradient clipping mode: {mode}")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Separate gradient clipping requires a finite positive max_norm")
    raw = model.module if hasattr(model, "module") else model
    if raw.config.policy_architecture not in {"dagger_transformer", "dagger_transformer_memory"}:
        raise ValueError("Separate gradient clipping requires a detached-critic DAgger PPO actor")
    prefixes = getattr(raw, "value_parameter_prefixes", ("critic.",))
    actor, value = [], []
    for name, parameter in raw.named_parameters():
        (value if name.startswith(prefixes) else actor).append(parameter)
    if not actor or not value:
        raise ValueError("Separate clipping requires nonempty disjoint actor and value groups")
    norm = gradient_l2_norm(raw.parameters())
    # Validate before either group is modified, avoiding partially clipped NaNs.
    if not torch.isfinite(norm).item():
        raise ValueError("Nonfinite gradient norm before separate clipping")
    torch.nn.utils.clip_grad_norm_(actor, max_norm)
    torch.nn.utils.clip_grad_norm_(value, max_norm)
    return norm
