"""Observation extraction + the ml-agents decode hotfix.

Helpers for pulling per-modality observations out of an ml-agents
``decision_steps`` tuple, plus the warmup poll that waits for the Unity
camera sensors to start emitting non-blank frames after reset.

:func:`patch_observation_decoding` installs a tolerance shim for a
CHW/HWC shape mismatch in the scene_all build's sensor layout; call it once
before constructing the ``UnityEnvironment``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import mlagents_envs.rpc_utils
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.exception import UnityObservationException

from nav.config import (
    ACTION_SPACE_ANNOTATION,
    MINIMAP_CANNY_HI,
    MINIMAP_CANNY_LO,
    MINIMAP_WARMUP_MAX_ATTEMPTS,
    MINIMAP_WARMUP_MIN_EDGES,
    MODALITY_TO_IDX,
    UNITY_MAP_SIZE,
)
from nav.utils import action2signal, obs_to_rgb


def patch_observation_decoding() -> None:
    """Install a tolerant ``_observation_to_np_array`` for sensor shape mismatches.

    The scene_all build's sensor layout can disagree with what mlagents-envs
    expects, raising "Decompressed observation did not have the expected
    shape". We fall back to decoding via ``process_pixels`` with an inferred
    channel count. Idempotent; safe to call once at startup before env launch.
    """
    orig = mlagents_envs.rpc_utils._observation_to_np_array

    def _patched(obs, expected_shape=None):
        try:
            return orig(obs, expected_shape)
        except UnityObservationException as e:
            if "Decompressed observation did not have the expected shape" in str(e):
                expected_channels = 3
                if hasattr(obs, "shape") and len(obs.shape) > 0 and 1 in obs.shape:
                    expected_channels = 1
                return mlagents_envs.rpc_utils.process_pixels(
                    obs.compressed_data,
                    expected_channels,
                    list(obs.compressed_channel_mapping),
                )
            raise

    mlagents_envs.rpc_utils._observation_to_np_array = _patched


def get_obs_safe(decision_steps, modality: str) -> Optional[np.ndarray]:
    """Return the squeezed observation for ``modality`` or None if unavailable."""
    if modality not in MODALITY_TO_IDX:
        return None
    idx = MODALITY_TO_IDX[modality]
    try:
        return decision_steps.obs[idx].squeeze(0)
    except Exception:
        return None


def get_minimap_rgb_for_init(
    env,
    behavior_name: str,
    max_attempts: int = MINIMAP_WARMUP_MAX_ATTEMPTS,
    min_edges: int = MINIMAP_WARMUP_MIN_EDGES,
) -> Optional[np.ndarray]:
    """Poll the env with STOP actions until the minimap renders a usable frame.

    Unity needs a few sim steps after reset before camera sensors emit
    non-empty frames; on the unified scene_all client this often takes
    10-30 steps. Returns the first minimap RGB with enough Canny edges, or
    None if none appeared within ``max_attempts``.
    """
    stop_signal = action2signal("stop", ACTION_SPACE_ANNOTATION)
    for _ in range(max_attempts):
        env.set_actions(behavior_name, ActionTuple(continuous=stop_signal))
        env.step()
        decision_steps, _ = env.get_steps(behavior_name)
        if len(decision_steps) == 0:
            continue
        mm = get_obs_safe(decision_steps, "minimap")
        if mm is None:
            continue
        rgb = obs_to_rgb(mm)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, MINIMAP_CANNY_LO, MINIMAP_CANNY_HI)
        h, w = rgb.shape[:2]
        edge_scale = (
            (w / float(UNITY_MAP_SIZE[0])) + (h / float(UNITY_MAP_SIZE[1]))
        ) / 2.0
        required_edges = max(5, round(min_edges * edge_scale))
        if int((edges > 0).sum()) >= required_edges:
            return rgb
    return None
