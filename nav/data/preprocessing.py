"""Sensor preprocessing shared by policy training and inference."""

from __future__ import annotations

import numpy as np


def depth_observation_float32(depth_obs: np.ndarray) -> np.ndarray:
    """Preserve normalized Unity depth precision used by the DAgger actor."""
    depth = np.asarray(depth_obs, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[None, ...]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = np.moveaxis(depth, -1, 0)
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"Unsupported depth observation shape: {depth.shape}")
    if np.nanmax(depth) > 1.5:
        depth = depth / 255.0
    return depth.copy()


def depth_observation_uint8(depth_obs: np.ndarray) -> np.ndarray:
    """Normalize a Unity depth observation to one encoded uint8 CHW channel."""
    depth = np.asarray(depth_obs)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            pass
        elif depth.shape[-1] == 1:
            depth = np.moveaxis(depth, -1, 0)
        else:
            depth = depth[:1]
    elif depth.ndim == 2:
        depth = depth[None, ...]
    else:
        raise ValueError(f"Unsupported depth observation shape: {depth.shape}")
    if depth.dtype == np.uint8:
        return depth.copy()
    depth = depth.astype(np.float32)
    if np.nanmax(depth) <= 1.5:
        depth = depth * 255.0
    return np.clip(np.nan_to_num(depth), 0.0, 255.0).astype(np.uint8)
