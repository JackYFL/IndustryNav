"""Inference-time BC controller consumed by the benchmark harness (bc_agent mode).

Wraps a trained :class:`nav.models.policy.NavPolicyTransformer` checkpoint and
turns per-step Unity observations into discrete navigation actions, maintaining
the sliding observation window the sequence model expects. Action chunking is
supported: when the checkpoint predicts a chunk, the extra steps are buffered
and replayed before the next model query.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np
import torch

from nav.config import BC_ACTION_TO_LABEL, BC_LABEL_TO_ACTION, IMAGENET_MEAN, IMAGENET_STD
from nav.models.policy import NavPolicyTransformer


class BCNavController:
    @staticmethod
    def _load_config(ckpt_path: str, ckpt: dict) -> dict:
        cfg = ckpt.get("config", {})
        if isinstance(cfg, dict) and cfg:
            return cfg
        cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg_json = json.load(f)
            if isinstance(cfg_json, dict):
                return cfg_json
        return {}

    def __init__(self, ckpt_path: str, device: str = "auto", seq_len_override: int = 0):
        self.seq_len_override = int(seq_len_override) if seq_len_override else 0
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")
        cfg = self._load_config(ckpt_path, ckpt)
        self.config = cfg

        self.goal_rep = str(cfg.get("goal_rep", "cartesian"))
        self.use_depth = bool(cfg.get("use_depth", True))
        self.use_rgb = bool(cfg.get("use_rgb", False))
        self.img_size = int(cfg.get("img_size", 256))
        self.seq_len = self.seq_len_override if self.seq_len_override > 0 else int(cfg.get("seq_len", 8))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.chunk_size = int(cfg.get("chunk_size", 1))

        self.model = NavPolicyTransformer(
            goal_dim=2,
            num_actions=4,
            use_depth=self.use_depth,
            use_rgb=self.use_rgb,
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            rgb_backbone=str(cfg.get("rgb_backbone", "resnet50")),
            depth_backbone=str(cfg.get("depth_backbone", "resnet50")),
            pretrained_rgb=bool(cfg.get("pretrained_rgb", False)),
            pretrained_depth=bool(cfg.get("pretrained_depth", False)),
            half_width=bool(cfg.get("half_width", True)),
            img_size=self.img_size,
            chunk_size=self.chunk_size,
        )
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.rgb_seq = deque(maxlen=self.seq_len)
        self.depth_seq = deque(maxlen=self.seq_len)
        self.goal_seq = deque(maxlen=self.seq_len)
        self.prev_action_seq = deque(maxlen=self.seq_len)
        self.last_action_idx = BC_ACTION_TO_LABEL["stop"]
        self._chunk_buffer: list = []

    def _prep_depth(self, depth_obs):
        if depth_obs is None:
            raise RuntimeError("Depth input is required by this checkpoint but depth_obs is None.")
        arr = np.asarray(depth_obs)
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
        if arr.ndim != 2:
            raise RuntimeError(f"Unsupported depth shape: {arr.shape}")
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.interpolate(ten, size=(self.img_size, self.img_size), mode="nearest")

    def _prep_rgb(self, ego_obs):
        if ego_obs is None:
            raise RuntimeError("RGB input is required by this checkpoint but ego_obs is None.")
        arr = np.asarray(ego_obs)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise RuntimeError(f"Unsupported rgb shape: {arr.shape}")
        arr = arr[..., :3].astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        ten = torch.nn.functional.interpolate(
            ten, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        mean = torch.tensor(IMAGENET_MEAN, dtype=ten.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=ten.dtype).view(1, 3, 1, 1)
        return (ten - mean) / std

    def _build_goal(self, curr_world_x, curr_world_z, curr_yaw_deg, target_world_x, target_world_z):
        dx = float(target_world_x) - float(curr_world_x)
        dz = float(target_world_z) - float(curr_world_z)
        yaw_rad = math.radians(float(curr_yaw_deg))
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        rel_x = cos_yaw * dx + sin_yaw * dz
        rel_z = -sin_yaw * dx + cos_yaw * dz
        if self.goal_rep == "polar":
            return np.array([math.sqrt(rel_x * rel_x + rel_z * rel_z), math.atan2(rel_z, rel_x)], dtype=np.float32)
        return np.array([rel_x, rel_z], dtype=np.float32)

    def reset(self) -> None:
        """Clear observation history and the action-chunk buffer."""
        self.rgb_seq.clear()
        self.depth_seq.clear()
        self.goal_seq.clear()
        self.prev_action_seq.clear()
        self.last_action_idx = BC_ACTION_TO_LABEL["stop"]
        self._chunk_buffer.clear()

    def _padded_stack(self, frames: list, pad_value) -> torch.Tensor:
        frames = list(frames)
        while len(frames) < self.seq_len:
            frames.insert(0, pad_value if pad_value is not None else frames[0])
        return frames[-self.seq_len :]

    def predict_action(
        self,
        ego_obs,
        depth_obs,
        curr_world_x: float,
        curr_world_z: float,
        curr_yaw_deg: float,
        target_world_x: float,
        target_world_z: float,
    ):
        # Update observation queues every call so a fresh query has full context
        # once the chunk buffer drains.
        if self.use_rgb:
            self.rgb_seq.append(self._prep_rgb(ego_obs))
        if self.use_depth:
            self.depth_seq.append(self._prep_depth(depth_obs))
        goal_np = self._build_goal(curr_world_x, curr_world_z, curr_yaw_deg, target_world_x, target_world_z)
        self.goal_seq.append(torch.from_numpy(goal_np).unsqueeze(0))
        self.prev_action_seq.append(self.last_action_idx)

        if self._chunk_buffer:
            action_idx = self._chunk_buffer.pop(0)
            self.last_action_idx = action_idx
            return BC_LABEL_TO_ACTION.get(action_idx, "stop")

        if self.use_rgb:
            rgb = torch.stack(self._padded_stack(self.rgb_seq, None), dim=1).to(self.device)
        else:
            rgb = torch.empty(0, device=self.device)
        if self.use_depth:
            depth = torch.stack(self._padded_stack(self.depth_seq, None), dim=1).to(self.device)
        else:
            depth = torch.empty(0, device=self.device)
        goal = torch.stack(self._padded_stack(self.goal_seq, None), dim=1).to(self.device)

        stop_label = BC_ACTION_TO_LABEL["stop"]
        prev_action = torch.tensor(
            self._padded_stack(self.prev_action_seq, stop_label), dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(rgb, depth, goal, prev_action)

        if self.chunk_size > 1:
            action_indices = logits[0].argmax(dim=-1).tolist()
            action_idx = action_indices[0]
            self._chunk_buffer = action_indices[1:]
        else:
            action_idx = int(torch.argmax(logits, dim=1).item())

        self.last_action_idx = action_idx
        return BC_LABEL_TO_ACTION.get(action_idx, "stop")
