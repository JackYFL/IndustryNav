from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nav.baselines.rl.imitation import (
    imitation_class_weights,
    run_imitation_epoch,
    run_route_epoch,
)
from nav.data.pointgoal_imitation import (
    PointGoalImitationEpisodeDataset,
    collate_pointgoal_imitation_episodes,
)
from nav.models.policies import PointGoalActorCritic, PointGoalPPOConfig


class PointGoalImitationTest(unittest.TestCase):
    def _write_episode(self, root: Path, episode_id: str, steps: int) -> None:
        episode = root / "scene1" / episode_id
        depth_dir = episode / "keyboard_depth"
        depth_dir.mkdir(parents=True)
        fields = [
            "step",
            "action",
            "behavior_action",
            "curr_world_x",
            "curr_world_z",
            "curr_direction_y",
            "target_world_x",
            "target_world_z",
        ]
        actions = ["forward", "turn right", "turn left"]
        with (episode / "keyboard_actions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(steps):
                np.save(
                    depth_dir / f"{step}.npy",
                    np.full((8, 12, 1), step / 10.0, dtype=np.float32),
                )
                writer.writerow(
                    {
                        "step": step,
                        "action": actions[step % 3],
                        "behavior_action": actions[(step + 1) % 3],
                        "curr_world_x": step,
                        "curr_world_z": 1,
                        "curr_direction_y": 90,
                        "target_world_x": 10,
                        "target_world_z": 5,
                    }
                )

    def test_episode_loading_and_padding_preserve_behavior_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_episode(root, "a", 3)
            self._write_episode(root, "b", 2)
            with (root / "dataset_manifest.jsonl").open("w") as stream:
                for episode_id in ("a", "b"):
                    stream.write(
                        json.dumps(
                            {
                                "episode_dir": f"scene1/{episode_id}",
                                "episode_id": episode_id,
                                "scene_name": "scene1",
                                "split": "train",
                            }
                        )
                        + "\n"
                    )
            dataset = PointGoalImitationEpisodeDataset(
                [root],
                split="train",
                absolute_scene_state=True,
                coordinate_fourier_bands=1,
            )
            batch = collate_pointgoal_imitation_episodes(
                [dataset[0], dataset[1]]
            )
            self.assertEqual(tuple(batch["depth"].shape), (2, 3, 1, 8, 12))
            self.assertEqual(tuple(batch["goal"].shape), (2, 3, 43))
            self.assertEqual(batch["valid"].tolist(), [[True] * 3, [True, True, False]])
            self.assertEqual(batch["behavior_action"][0].tolist(), [1, 2, 0])
            self.assertEqual(dict(dataset.class_counts), {0: 2, 1: 2, 2: 1})

    def test_soft_class_weights_have_unit_sample_mean(self):
        weights = imitation_class_weights({0: 100, 1: 25, 2: 25}, power=0.5)
        counts = torch.tensor([100.0, 25.0, 25.0])
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float((weights * counts).sum() / counts.sum()), 1.0)

    def test_precomputed_visual_head_matches_regular_forward(self):
        torch.manual_seed(3)
        config = PointGoalPPOConfig(
            depth_backbone="resnet18",
            img_size=32,
            visual_dim=8,
            goal_hidden=4,
            action_embed_dim=2,
            action_history_len=5,
            visual_history_len=5,
            hidden_size=8,
        )
        model = PointGoalActorCritic(config).eval()
        depth = torch.randint(0, 256, (2, 1, 20, 24), dtype=torch.uint8)
        goal = torch.randn(2, config.goal_input_dim)
        actions = torch.full((2, 5), 3, dtype=torch.long)
        visual_history = torch.randn(2, 5, config.visual_dim)
        hidden = torch.randn(2, config.hidden_size)
        masks = torch.ones(2, 1)
        with torch.no_grad():
            direct = model(depth, goal, actions, visual_history, hidden, masks)
            visual = model.encode_depth(depth)
            split = model.forward_from_visual(
                visual, goal, actions, visual_history, hidden, masks
            )
        for direct_tensor, split_tensor in zip(direct, split):
            torch.testing.assert_close(direct_tensor, split_tensor)

    def test_truncated_recurrent_training_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_episode(root, "a", 3)
            self._write_episode(root, "b", 3)
            with (root / "dataset_manifest.jsonl").open("w") as stream:
                for episode_id in ("a", "b"):
                    stream.write(
                        json.dumps(
                            {
                                "episode_dir": f"scene1/{episode_id}",
                                "episode_id": episode_id,
                                "scene_name": "scene1",
                                "split": "train",
                            }
                        )
                        + "\n"
                    )
            dataset = PointGoalImitationEpisodeDataset([root], split="train")
            loader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_pointgoal_imitation_episodes,
            )
            model = PointGoalActorCritic(
                PointGoalPPOConfig(
                    depth_backbone="resnet18",
                    img_size=32,
                    visual_dim=8,
                    goal_hidden=4,
                    action_embed_dim=2,
                    action_history_len=5,
                    visual_history_len=5,
                    hidden_size=8,
                )
            )
            for parameter in model.depth_encoder.parameters():
                parameter.requires_grad_(False)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=1e-3
            )
            metrics = run_imitation_epoch(
                model,
                loader,
                device=torch.device("cpu"),
                class_weights=torch.ones(3),
                bptt_len=2,
                encoder_batch_size=4,
                optimizer=optimizer,
            )
            self.assertEqual(metrics["samples"], 6)
            self.assertTrue(np.isfinite(metrics["loss"]))

    def test_route_only_training_does_not_load_depth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_episode(root, "a", 3)
            with (root / "dataset_manifest.jsonl").open("w") as stream:
                stream.write(
                    json.dumps(
                        {
                            "episode_dir": "scene1/a",
                            "episode_id": "a",
                            "scene_name": "scene1",
                            "split": "train",
                        }
                    )
                    + "\n"
                )
            # Prove metadata-only mode never touches image files.
            for depth_path in (root / "scene1" / "a" / "keyboard_depth").iterdir():
                depth_path.unlink()
            dataset = PointGoalImitationEpisodeDataset(
                [root],
                split="train",
                absolute_scene_state=True,
                coordinate_fourier_bands=1,
                load_depth=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                collate_fn=collate_pointgoal_imitation_episodes,
            )
            model = PointGoalActorCritic(
                PointGoalPPOConfig(
                    depth_backbone="resnet18",
                    img_size=32,
                    visual_dim=8,
                    goal_hidden=4,
                    action_embed_dim=2,
                    action_history_len=5,
                    visual_history_len=5,
                    hidden_size=8,
                    absolute_scene_state=True,
                    coordinate_fourier_bands=1,
                    route_actor_hidden=8,
                )
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.route_actor.parameters():
                parameter.requires_grad_(True)
            optimizer = torch.optim.Adam(model.route_actor.parameters(), lr=1e-3)
            metrics = run_route_epoch(
                model,
                loader,
                device=torch.device("cpu"),
                class_weights=torch.ones(3),
                optimizer=optimizer,
            )
            self.assertEqual(metrics["samples"], 3)
            self.assertEqual(tuple(dataset[0]["depth"].shape), (3, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
