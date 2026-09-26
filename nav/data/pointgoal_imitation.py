"""Episode-level A* demonstrations for recurrent PointGoal actor warm-up.

The PPO policy is recurrent and consumes explicit action and visual histories.
Training it from shuffled single frames therefore creates a train/deploy
mismatch.  This dataset keeps complete trajectories together so the offline
trainer can replay the action sequence exactly and use truncated BPTT without
ever exposing the evaluation ``input_points.json`` pairs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from nav.baselines.rl.actions import PPO_ACTION_TO_LABEL, PPO_BOS_LABEL
from nav.core.pointgoal import encode_rl_goal_state
from nav.data.preprocessing import depth_observation_uint8


def _scene_id(scene_name: str) -> int:
    try:
        scene_id = int(str(scene_name).removeprefix("scene")) - 1
    except ValueError as exc:
        raise ValueError(f"Unsupported scene name: {scene_name!r}") from exc
    if scene_id < 0:
        raise ValueError(f"Unsupported scene name: {scene_name!r}")
    return scene_id


def _float(row: dict[str, str], key: str, fallback: str | None = None) -> float:
    value = row.get(key)
    if (value is None or value == "") and fallback is not None:
        value = row.get(fallback)
    if value is None or value == "":
        raise ValueError(f"Missing numeric field {key!r}")
    return float(value)


class PointGoalImitationEpisodeDataset(Dataset):
    """Load complete navigation-only expert/DAgger trajectories.

    ``data_roots`` point at exported BC directories containing
    ``dataset_manifest.jsonl``.  The target action is the oracle ``action``
    column, while recurrent action history follows ``behavior_action`` when a
    DAgger trajectory records it.
    """

    def __init__(
        self,
        data_roots: Sequence[str | Path],
        *,
        split: str = "train",
        goal_distance_scale_m: float = 50.0,
        absolute_scene_state: bool = False,
        num_scenes: int = 24,
        world_coordinate_scale_m: float = 50.0,
        coordinate_fourier_bands: int = 0,
        episode_limit: int = 0,
        load_depth: bool = True,
    ) -> None:
        if not data_roots:
            raise ValueError("At least one imitation data root is required")
        self.data_roots = tuple(Path(root) for root in data_roots)
        self.split = str(split)
        self.goal_distance_scale_m = float(goal_distance_scale_m)
        self.absolute_scene_state = bool(absolute_scene_state)
        self.num_scenes = int(num_scenes)
        self.world_coordinate_scale_m = float(world_coordinate_scale_m)
        self.coordinate_fourier_bands = int(coordinate_fourier_bands)
        self.load_depth = bool(load_depth)
        if episode_limit < 0:
            raise ValueError("episode_limit must be nonnegative")
        self.episode_limit = int(episode_limit)
        self.episodes = self._load_episode_metadata()
        if self.episode_limit:
            self.episodes = self.episodes[: self.episode_limit]
        if not self.episodes:
            raise RuntimeError(
                f"No imitation episodes found for split {self.split!r}"
            )
        self.class_counts = Counter(
            step["action"]
            for episode in self.episodes
            for step in episode["steps"]
        )

    def _load_episode_metadata(self) -> list[dict]:
        episodes: list[dict] = []
        for root in self.data_roots:
            manifest = root / "dataset_manifest.jsonl"
            if not manifest.is_file():
                raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
            with manifest.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            for record in records:
                if self.split != "all" and record.get("split") != self.split:
                    continue
                episode_dir = root / str(record["episode_dir"])
                scene_name = str(record.get("scene_name") or episode_dir.parent.name)
                steps = self._read_steps(episode_dir, _scene_id(scene_name))
                if steps:
                    episodes.append(
                        {
                            "episode_dir": str(episode_dir),
                            "scene_name": scene_name,
                            "episode_id": str(record.get("episode_id", episode_dir.name)),
                            "steps": steps,
                        }
                    )
        return episodes

    def _read_steps(self, episode_dir: Path, scene_id: int) -> list[dict]:
        csv_path = episode_dir / "keyboard_actions.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Episode action CSV not found: {csv_path}")
        steps = []
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                action_name = str(row.get("action", "")).strip().lower()
                if action_name not in PPO_ACTION_TO_LABEL:
                    # PPO terminates by metric distance and has no stop class.
                    continue
                behavior_name = str(
                    row.get("behavior_action", action_name)
                ).strip().lower()
                if behavior_name not in PPO_ACTION_TO_LABEL:
                    raise ValueError(
                        f"Unsupported behavior action {behavior_name!r} in {csv_path}"
                    )
                frame = int(float(row.get("step", 0)))
                depth_path: Path | None = None
                if self.load_depth:
                    depth_dir = episode_dir / "keyboard_depth"
                    npy_path = depth_dir / f"{frame}.npy"
                    png_path = depth_dir / f"{frame}.png"
                    if npy_path.is_file():
                        depth_path = npy_path
                    elif png_path.is_file():
                        depth_path = png_path
                    else:
                        raise FileNotFoundError(
                            f"Depth frame {frame} not found below {depth_dir}"
                        )
                goal = encode_rl_goal_state(
                    _float(row, "curr_world_x"),
                    _float(row, "curr_world_z"),
                    _float(row, "curr_direction_y", "init_direction"),
                    _float(row, "target_world_x"),
                    _float(row, "target_world_z"),
                    distance_scale_m=self.goal_distance_scale_m,
                    scene_id=scene_id,
                    num_scenes=self.num_scenes,
                    world_coordinate_scale_m=self.world_coordinate_scale_m,
                    include_absolute_scene_state=self.absolute_scene_state,
                    coordinate_fourier_bands=self.coordinate_fourier_bands,
                )
                steps.append(
                    {
                        "depth_path": (
                            str(depth_path) if depth_path is not None else None
                        ),
                        "goal": goal,
                        "action": PPO_ACTION_TO_LABEL[action_name],
                        "behavior_action": PPO_ACTION_TO_LABEL[behavior_name],
                    }
                )
        return steps

    def __len__(self) -> int:
        return len(self.episodes)

    @staticmethod
    def _load_depth(path: str) -> torch.Tensor:
        depth_path = Path(path)
        if depth_path.suffix == ".npy":
            depth = depth_observation_uint8(np.load(depth_path))
            return torch.from_numpy(depth)
        return read_image(str(depth_path), mode=ImageReadMode.GRAY)

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[index]
        steps = episode["steps"]
        return {
            "depth": (
                torch.stack(
                    [self._load_depth(step["depth_path"]) for step in steps]
                )
                if self.load_depth
                else torch.empty((len(steps), 0, 0, 0), dtype=torch.uint8)
            ),
            "goal": torch.from_numpy(
                np.stack([step["goal"] for step in steps])
            ).float(),
            "action": torch.tensor(
                [step["action"] for step in steps], dtype=torch.long
            ),
            "behavior_action": torch.tensor(
                [step["behavior_action"] for step in steps], dtype=torch.long
            ),
            "length": len(steps),
            "scene_name": episode["scene_name"],
            "episode_id": episode["episode_id"],
        }


def collate_pointgoal_imitation_episodes(batch: Iterable[dict]) -> dict:
    """Pad a batch of complete episodes and return an explicit validity mask."""
    episodes = list(batch)
    if not episodes:
        raise ValueError("Cannot collate an empty imitation batch")
    lengths = torch.tensor([episode["length"] for episode in episodes])
    max_steps = int(lengths.max().item())
    batch_size = len(episodes)
    depth_shape = tuple(episodes[0]["depth"].shape[1:])
    goal_dim = int(episodes[0]["goal"].shape[-1])
    depth = torch.zeros((batch_size, max_steps, *depth_shape), dtype=torch.uint8)
    goal = torch.zeros((batch_size, max_steps, goal_dim), dtype=torch.float32)
    action = torch.zeros((batch_size, max_steps), dtype=torch.long)
    behavior_action = torch.full(
        (batch_size, max_steps), PPO_BOS_LABEL, dtype=torch.long
    )
    valid = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    for row, episode in enumerate(episodes):
        length = int(episode["length"])
        if tuple(episode["depth"].shape[1:]) != depth_shape:
            raise ValueError("All depth frames in a batch must share one resolution")
        depth[row, :length] = episode["depth"]
        goal[row, :length] = episode["goal"]
        action[row, :length] = episode["action"]
        behavior_action[row, :length] = episode["behavior_action"]
        valid[row, :length] = True
    return {
        "depth": depth,
        "goal": goal,
        "action": action,
        "behavior_action": behavior_action,
        "valid": valid,
        "lengths": lengths,
        "scene_name": [episode["scene_name"] for episode in episodes],
        "episode_id": [episode["episode_id"] for episode in episodes],
    }


__all__ = [
    "PointGoalImitationEpisodeDataset",
    "collate_pointgoal_imitation_episodes",
]
