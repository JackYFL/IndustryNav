"""Bounded cross-update replay for online PointGoal teacher supervision."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


_REPLAY_KEYS = (
    "depth",
    "goal",
    "action_history",
    "visual_history",
    "hidden",
    "masks",
    "teacher_actions",
)


class TeacherReplayBuffer:
    """Keep recent A*-labeled recurrent states in compact CPU tensors."""

    def __init__(self, capacity: int, *, img_size: int, seed: int) -> None:
        if capacity <= 0 or img_size <= 0:
            raise ValueError("capacity and img_size must be positive")
        self.capacity = int(capacity)
        self.img_size = int(img_size)
        self.generator = torch.Generator().manual_seed(int(seed))
        self._data: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        actions = self._data.get("teacher_actions")
        return 0 if actions is None else int(actions.shape[0])

    def add_rollout(self, rollout: Mapping[str, torch.Tensor]) -> None:
        missing = set(_REPLAY_KEYS) - rollout.keys()
        if missing:
            raise ValueError(f"Replay rollout is missing keys: {sorted(missing)}")
        flattened = {
            key: rollout[key].flatten(0, 1).detach().cpu()
            for key in _REPLAY_KEYS
        }
        depth = flattened["depth"]
        if depth.shape[-2:] != (self.img_size, self.img_size):
            depth = F.interpolate(
                depth.float(),
                size=(self.img_size, self.img_size),
                mode="nearest",
            ).to(torch.uint8)
        elif depth.dtype != torch.uint8:
            depth = depth.clamp(0, 255).to(torch.uint8)
        flattened["depth"] = depth
        # Cached visual histories and hidden states need no fp32 precision in
        # replay; conversion back happens when a sampled batch moves to CUDA.
        flattened["visual_history"] = flattened["visual_history"].half()
        flattened["hidden"] = flattened["hidden"].half()
        for key, value in flattened.items():
            if key in self._data:
                value = torch.cat((self._data[key], value), dim=0)
            self._data[key] = value[-self.capacity :].contiguous()

    def action_counts(self, num_actions: int) -> torch.Tensor:
        if len(self) == 0:
            return torch.zeros(num_actions, dtype=torch.float32)
        return torch.bincount(
            self._data["teacher_actions"].long(),
            minlength=num_actions,
        ).float()

    def sample(self, batch_size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self) == 0:
            raise RuntimeError("Cannot sample an empty teacher replay buffer")
        indices = torch.randint(
            len(self),
            (min(int(batch_size), len(self)),),
            generator=self.generator,
        )
        batch = {
            key: value[indices].to(device, non_blocking=True)
            for key, value in self._data.items()
        }
        batch["visual_history"] = batch["visual_history"].float()
        batch["hidden"] = batch["hidden"].float()
        return batch

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "img_size": self.img_size,
            "generator_state": self.generator.get_state(),
            "data": self._data,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Replay checkpoint capacity does not match")
        if int(state["img_size"]) != self.img_size:
            raise ValueError("Replay checkpoint image size does not match")
        data = dict(state["data"])
        if data and set(data) != set(_REPLAY_KEYS):
            raise ValueError("Replay checkpoint keys do not match")
        self._data = {key: value.cpu() for key, value in data.items()}
        self.generator.set_state(state["generator_state"])


def soft_inverse_class_weights(
    counts: torch.Tensor,
    *,
    power: float,
) -> torch.Tensor:
    """Temper inverse-frequency weights and preserve unit sample mean."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be in [0, 1]")
    counts = counts.float()
    present = counts > 0
    if not bool(present.any()):
        return torch.ones_like(counts)
    weights = torch.zeros_like(counts)
    inverse = counts[present].sum() / (counts[present] * present.sum())
    weights[present] = inverse.pow(float(power))
    sample_mean = (weights * counts).sum() / counts.sum()
    return weights / sample_mean.clamp_min(1.0e-8)


__all__ = ["TeacherReplayBuffer", "soft_inverse_class_weights"]
