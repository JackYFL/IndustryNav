"""Offline uniform parameter averaging, without an inference-time ensemble."""

from collections.abc import Mapping, Sequence

import torch


def average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    fixed_prefixes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Average compatible floating tensors; require fixed buffers to match.

    Inputs are never mutated. Identical tensors retain their exact bytes.
    Integer/bool state and caller-designated frozen encoders cannot be averaged.
    This does not verify that independently trained networks share an alignment;
    callers must establish common initialization and evaluate the result.
    """
    if len(states) < 2 or not states[0]:
        raise ValueError("At least two nonempty state dictionaries are required")
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("State dictionary keys differ")
    averaged = {}
    for key, source in states[0].items():
        values = [state[key] for state in states]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError(f"Non-tensor checkpoint state: {key}")
        if any(value.shape != source.shape or value.dtype != source.dtype for value in values):
            raise ValueError(f"Tensor shape/dtype mismatch: {key}")
        if source.is_complex() or source.layout != torch.strided:
            raise ValueError(f"Unsupported checkpoint tensor: {key}")
        values = [value.detach().cpu() for value in values]
        if source.is_floating_point() and any(not torch.isfinite(value).all() for value in values):
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
        identical = all(torch.equal(values[0], value) for value in values[1:])
        if identical:
            averaged[key] = values[0].clone()
        elif key.startswith(fixed_prefixes) or not source.is_floating_point():
            raise ValueError(f"Frozen encoder or non-floating buffer differs: {key}")
        else:
            # Accumulate in float64 to avoid FP16/FP32 summation overflow and
            # minimize rounding error before restoring the source dtype.
            total = torch.zeros_like(values[0], dtype=torch.float64)
            for value in values:
                total.add_(value.to(torch.float64) / len(values))
            averaged[key] = total.to(source.dtype)
            if not torch.isfinite(averaged[key]).all():
                raise ValueError(f"Nonfinite averaged tensor: {key}")
    return averaged
