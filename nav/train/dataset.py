"""BC navigation datasets over collected teleop episodes.

Two datasets share one base:

- :class:`NavEpisodeDataset` — one sample per frame, with a numeric *state*
  vector (for the single-frame ``mlp`` policy).
- :class:`NavEpisodeSequenceDataset` — one sample per sliding window, with a
  per-step *goal* vector + previous action (for the recurrent / transformer /
  diffusion policies).

:class:`_NavEpisodeBase` owns everything that was duplicated between them:
episode discovery + the deterministic seeded split, RGB/depth loading, and the
train-only augmentation (random resized crop + color jitter).
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    resize,
    resized_crop,
)

from nav.config import (
    BC_ACTION_TO_LABEL,
    BC_DEPTH_SUBDIR,
    BC_EPISODE_CSV,
    BC_RGB_SUBDIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def _row_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


class _NavEpisodeBase(Dataset):
    """Shared episode discovery, splitting, and image IO/augmentation."""

    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: int,
        split_ratios: Tuple[float, float, float],
        seed: int,
        normalize_rgb: bool,
        use_depth: bool,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.split_ratios = split_ratios
        self.seed = seed
        self.normalize_rgb = normalize_rgb
        self.use_depth = use_depth
        self.augment = split == "train"
        self._episodes = self._split_episodes()

    # -- episode discovery / split ------------------------------------------
    def _list_episodes(self) -> List[str]:
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"data_root not found: {self.data_root}")
        episodes = []
        for scene in sorted(os.listdir(self.data_root)):
            scene_dir = os.path.join(self.data_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for point in sorted(os.listdir(scene_dir)):
                ep_dir = os.path.join(scene_dir, point)
                if not os.path.isdir(ep_dir):
                    continue
                if not os.path.isfile(os.path.join(ep_dir, BC_EPISODE_CSV)):
                    continue
                if not os.path.isdir(os.path.join(ep_dir, BC_RGB_SUBDIR)):
                    continue
                if self.use_depth and not os.path.isdir(os.path.join(ep_dir, BC_DEPTH_SUBDIR)):
                    continue
                episodes.append(ep_dir)
        return episodes

    def _split_episodes(self) -> List[str]:
        episodes = self._list_episodes()
        if not episodes:
            raise RuntimeError(f"No episodes found in {self.data_root}")

        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(episodes))
        rng.shuffle(indices)

        n_total = len(episodes)
        n_train = int(self.split_ratios[0] * n_total)
        n_val = int(self.split_ratios[1] * n_total)
        if self.split == "train":
            sel = indices[:n_train]
        elif self.split == "val":
            sel = indices[n_train : n_train + n_val]
        elif self.split == "test":
            sel = indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return [episodes[i] for i in sel]

    def _episode_paths(self, ep_dir: str) -> Tuple[str, str, str]:
        return (
            os.path.join(ep_dir, BC_EPISODE_CSV),
            os.path.join(ep_dir, BC_RGB_SUBDIR),
            os.path.join(ep_dir, BC_DEPTH_SUBDIR),
        )

    # -- image IO / augmentation --------------------------------------------
    def _load_rgb_raw(self, path: str) -> torch.Tensor:
        """Load an RGB frame as a float [0,1] CHW tensor (no normalization)."""
        img = read_image(path, mode=ImageReadMode.RGB)
        img = resize(
            img, [self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        return img.float().div(255.0)

    def _load_depth(self, path: str) -> torch.Tensor:
        img = read_image(path, mode=ImageReadMode.GRAY)
        img = resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        return img.float().div(255.0)

    def _random_crop_params(self) -> Tuple[int, int, int, int]:
        scale = random.uniform(0.85, 1.0)
        crop_size = int(self.img_size * scale)
        top = random.randint(0, self.img_size - crop_size)
        left = random.randint(0, self.img_size - crop_size)
        return top, left, crop_size, crop_size

    def _color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        img = adjust_brightness(img, random.uniform(0.8, 1.2))
        img = adjust_contrast(img, random.uniform(0.8, 1.2))
        img = adjust_saturation(img, random.uniform(0.9, 1.1))
        return img

    def _normalize_rgb(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        std = torch.tensor(IMAGENET_STD)[:, None, None]
        return (img - mean) / std


class NavEpisodeDataset(_NavEpisodeBase):
    """One sample per frame: (RGB [+ depth], numeric state) -> action."""

    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: int = 512,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        state_mode: str = "relative",
        normalize_rgb: bool = True,
        use_depth: bool = True,
    ) -> None:
        self.state_mode = state_mode
        super().__init__(data_root, split, img_size, split_ratios, seed, normalize_rgb, use_depth)
        self._samples = self._build_samples()
        self.class_counts = self._count_classes()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, depth_path, state, action = self._samples[idx]
        rgb = self._load_rgb_raw(rgb_path)
        depth = self._load_depth(depth_path) if self.use_depth else torch.empty(0)

        if self.augment:
            crop = self._random_crop_params()
            rgb = self._color_jitter(resized_crop(rgb, *crop, [self.img_size, self.img_size]))
            if self.use_depth:
                depth = resized_crop(
                    depth, *crop, [self.img_size, self.img_size],
                    interpolation=InterpolationMode.NEAREST,
                )
        if self.normalize_rgb:
            rgb = self._normalize_rgb(rgb)

        return {
            "rgb": rgb,
            "depth": depth,
            "state": torch.from_numpy(state).float(),
            "action": torch.tensor(action, dtype=torch.long),
        }

    def _build_samples(self) -> List[Tuple[str, str, np.ndarray, int]]:
        samples = []
        for ep_dir in self._episodes:
            csv_path, rgb_dir, depth_dir = self._episode_paths(ep_dir)
            with open(csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    action_str = str(row.get("action", "")).strip().lower()
                    if action_str not in BC_ACTION_TO_LABEL:
                        continue
                    step = int(float(row.get("step", 0)))
                    rgb_path = os.path.join(rgb_dir, f"{step}.png")
                    depth_path = os.path.join(depth_dir, f"{step}.png")
                    if not os.path.isfile(rgb_path):
                        continue
                    if self.use_depth and not os.path.isfile(depth_path):
                        continue
                    samples.append(
                        (rgb_path, depth_path, self._build_state(row), BC_ACTION_TO_LABEL[action_str])
                    )
        if not samples:
            raise RuntimeError("No samples found after parsing episodes.")
        return samples

    def _build_state(self, row: Dict[str, str]) -> np.ndarray:
        yaw_rad = math.radians(_row_float(row, "curr_direction_y", _row_float(row, "init_direction")))
        sin_yaw, cos_yaw = math.sin(yaw_rad), math.cos(yaw_rad)
        curr_x, curr_z = _row_float(row, "curr_world_x"), _row_float(row, "curr_world_z")
        target_x, target_z = _row_float(row, "target_world_x"), _row_float(row, "target_world_z")
        dx, dz = target_x - curr_x, target_z - curr_z
        dist = _row_float(row, "distance_world", math.sqrt(dx * dx + dz * dz))

        if self.state_mode == "absolute":
            return np.array([sin_yaw, cos_yaw, dist, target_x, target_z], dtype=np.float32)
        if self.state_mode == "full":
            return np.array(
                [sin_yaw, cos_yaw, curr_x, curr_z, target_x, target_z, dist], dtype=np.float32
            )
        return np.array([sin_yaw, cos_yaw, dx, dz, dist], dtype=np.float32)  # relative

    def _count_classes(self) -> Dict[int, int]:
        counts = {i: 0 for i in range(4)}
        for _, _, _, action in self._samples:
            counts[action] += 1
        return counts


class NavEpisodeSequenceDataset(_NavEpisodeBase):
    """One sample per sliding window: (goal, prev-action[, vision]) seq -> action(s)."""

    def __init__(
        self,
        data_root: str,
        split: str,
        seq_len: int = 8,
        img_size: int = 256,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        goal_rep: str = "cartesian",
        normalize_rgb: bool = True,
        use_depth: bool = True,
        use_rgb: bool = False,
        chunk_size: int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.goal_rep = goal_rep
        self.use_rgb = use_rgb
        self.chunk_size = chunk_size
        super().__init__(data_root, split, img_size, split_ratios, seed, normalize_rgb, use_depth)
        self._episode_steps = self._build_episode_steps()
        self._index = self._build_index()
        self.class_counts = self._count_classes()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, step_idx = self._index[idx]
        steps = self._episode_steps[ep_idx]

        start = max(0, step_idx - self.seq_len + 1)
        seq = steps[start : step_idx + 1]
        if len(seq) < self.seq_len:
            seq = [steps[0]] * (self.seq_len - len(seq)) + seq

        rgb_seq, depth_seq, goal_seq, prev_action_seq = [], [], [], []
        for item in seq:
            if self.use_rgb:
                rgb_seq.append(self._load_rgb_raw(item["rgb_path"]))
            if self.use_depth:
                depth_seq.append(self._load_depth(item["depth_path"]))
            goal_seq.append(torch.from_numpy(item["goal"]).float())
            prev_action_seq.append(item["prev_action"])

        if self.augment and (rgb_seq or depth_seq):
            crop = self._random_crop_params()  # shared geometric crop across the window
            rgb_seq = [self._color_jitter(resized_crop(im, *crop, [self.img_size, self.img_size])) for im in rgb_seq]
            depth_seq = [
                resized_crop(im, *crop, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
                for im in depth_seq
            ]
        if rgb_seq and self.normalize_rgb:
            rgb_seq = [self._normalize_rgb(im) for im in rgb_seq]

        return {
            "rgb": torch.stack(rgb_seq) if self.use_rgb else torch.empty(0),
            "depth": torch.stack(depth_seq) if self.use_depth else torch.empty(0),
            "goal": torch.stack(goal_seq),
            "prev_action": torch.tensor(prev_action_seq, dtype=torch.long),
            "action": self._target_action(steps, step_idx),
        }

    def _target_action(self, steps, step_idx) -> torch.Tensor:
        if self.chunk_size == 1:
            return torch.tensor(steps[step_idx + 1]["action"], dtype=torch.long)
        chunk = []
        for k in range(self.chunk_size):
            future_idx = step_idx + 1 + k
            chunk.append(steps[future_idx]["action"] if future_idx < len(steps) else chunk[-1])
        return torch.tensor(chunk, dtype=torch.long)

    def _build_episode_steps(self) -> List[List[dict]]:
        stop_label = BC_ACTION_TO_LABEL["stop"]
        episodes_steps: List[List[dict]] = []
        for ep_dir in self._episodes:
            csv_path, rgb_dir, depth_dir = self._episode_paths(ep_dir)
            steps: List[dict] = []
            with open(csv_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    action_str = str(row.get("action", "")).strip().lower()
                    if action_str not in BC_ACTION_TO_LABEL:
                        continue
                    step = int(float(row.get("step", 0)))
                    rgb_path = os.path.join(rgb_dir, f"{step}.png")
                    depth_path = os.path.join(depth_dir, f"{step}.png")
                    if not os.path.isfile(rgb_path):
                        continue
                    if self.use_depth and not os.path.isfile(depth_path):
                        continue
                    action = BC_ACTION_TO_LABEL[action_str]
                    steps.append({
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "goal": self._build_goal(row),
                        "prev_action": steps[-1]["action"] if steps else stop_label,
                        "action": action,
                    })
            if steps:
                episodes_steps.append(steps)
        if not episodes_steps:
            raise RuntimeError("No steps found after parsing episodes.")
        return episodes_steps

    def _build_index(self) -> List[Tuple[int, int]]:
        # Drop windows whose target action is "stop" for train/val (the data is
        # stop-heavy at episode ends); keep all for test.
        drop_stop_target = self.split in {"train", "val"}
        stop_label = BC_ACTION_TO_LABEL["stop"]
        index = []
        for ep_idx, steps in enumerate(self._episode_steps):
            for step_idx in range(len(steps) - 1):
                if drop_stop_target and steps[step_idx + 1]["action"] == stop_label:
                    continue
                index.append((ep_idx, step_idx))
        return index

    def _build_goal(self, row: Dict[str, str]) -> np.ndarray:
        yaw_rad = math.radians(_row_float(row, "curr_direction_y", _row_float(row, "init_direction")))
        curr_x, curr_z = _row_float(row, "curr_world_x"), _row_float(row, "curr_world_z")
        target_x, target_z = _row_float(row, "target_world_x"), _row_float(row, "target_world_z")
        dx, dz = target_x - curr_x, target_z - curr_z
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        rel_x = cos_yaw * dx + sin_yaw * dz
        rel_z = -sin_yaw * dx + cos_yaw * dz
        if self.goal_rep == "polar":
            return np.array([math.sqrt(rel_x * rel_x + rel_z * rel_z), math.atan2(rel_z, rel_x)], dtype=np.float32)
        return np.array([rel_x, rel_z], dtype=np.float32)

    def _count_classes(self) -> Dict[int, int]:
        counts = {i: 0 for i in range(4)}
        for ep_idx, step_idx in self._index:
            counts[self._episode_steps[ep_idx][step_idx + 1]["action"]] += 1
        return counts
