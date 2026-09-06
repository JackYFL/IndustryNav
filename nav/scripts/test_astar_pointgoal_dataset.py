from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nav.config import ACTION_SPACE_POINTGOAL
from nav.baselines.astar import AStarBaseline
from nav.scripts.export_astar_pointgoal_dataset import export_episode
from nav.scripts.prepare_astar_pointgoal_pilot import build_manifest
from nav.scripts.run_benchmark_cell import action_space_for_baseline
from nav.train.controller import BCNavController
from nav.train.dataset import NavEpisodeSequenceDataset, mirror_pointgoal_sample
from nav.train.loop import get_class_weights


class PointGoalCollectionTest(unittest.TestCase):
    def test_horizontal_mirror_swaps_goal_and_turn_semantics(self):
        rgb = [torch.tensor([[[1.0, 2.0, 3.0]]])]
        depth = [torch.tensor([[[4.0, 5.0, 6.0]]])]
        goal = [torch.tensor([0.5, 0.25])]
        previous = [2, 3, 0]
        action = torch.tensor([2, 3, 0, 1])
        rgb, depth, goal, previous, action = mirror_pointgoal_sample(
            rgb, depth, goal, previous, action
        )
        self.assertTrue(torch.equal(rgb[0], torch.tensor([[[3.0, 2.0, 1.0]]])))
        self.assertTrue(torch.equal(depth[0], torch.tensor([[[6.0, 5.0, 4.0]]])))
        self.assertTrue(torch.equal(goal[0], torch.tensor([0.5, -0.25])))
        self.assertEqual(previous, [3, 2, 0])
        self.assertTrue(torch.equal(action, torch.tensor([3, 2, 0, 1])))

    def test_polar_goal_normalization(self):
        dataset = NavEpisodeSequenceDataset.__new__(NavEpisodeSequenceDataset)
        dataset.goal_rep = "polar"
        dataset.goal_distance_scale_m = 50.0
        goal = dataset._build_goal({
            "curr_world_x": "0",
            "curr_world_z": "0",
            "curr_direction_y": "0",
            "target_world_x": "3",
            "target_world_z": "4",
        })
        self.assertAlmostEqual(float(goal[0]), 0.1)
        self.assertAlmostEqual(float(goal[1]), math.atan2(4, 3) / math.pi)

        controller = BCNavController.__new__(BCNavController)
        controller.goal_rep = "polar"
        controller.goal_distance_scale_m = 50.0
        inference_goal = controller._build_goal(0, 0, 0, 3, 4)
        np.testing.assert_allclose(inference_goal, goal)

    def test_soft_class_weights_reduce_extreme_imbalance(self):
        counts = {0: 100, 1: 1, 2: 25, 3: 25}
        full = get_class_weights(counts, power=1.0)
        soft = get_class_weights(counts, power=0.5)
        self.assertGreater(float(full[1] / full[0]), float(soft[1] / soft[0]))
        self.assertAlmostEqual(float(soft[1] / soft[0]), 10.0, places=5)
        with self.assertRaises(ValueError):
            get_class_weights(counts, power=1.1)

    def test_manifest_is_scene_disjoint_and_has_240_tasks(self):
        points = {
            f"scene{scene}": [
                {
                    "point_id": f"point{point}",
                    "start": {"x": point, "z": point + 0.5, "direction": 90 * point},
                    "target": {"x": 10 * point, "y": 20 * point},
                }
                for point in range(1, 5)
            ]
            for scene in range(1, 25)
        }
        records = build_manifest(points, seeds=2)
        self.assertEqual(len(records), 240)
        self.assertEqual(sum(r["split"] == "train" for r in records), 160)
        self.assertEqual(sum(r["split"] == "val" for r in records), 40)
        self.assertEqual(sum(r["split"] == "test" for r in records), 40)
        self.assertTrue(all(r["source_start_point"] != r["source_target_point"] for r in records))

    def test_policy_astar_uses_agent_action_space(self):
        self.assertIs(action_space_for_baseline("astar", True), ACTION_SPACE_POINTGOAL)
        self.assertEqual(ACTION_SPACE_POINTGOAL["forward"], 7.5)
        planner = AStarBaseline(policy_actions=True)
        # An uncorrectable sub-half-step error should move forward instead of
        # oscillating around the desired heading.
        self.assertEqual(
            planner._action_from_steering_error(
                12.0,
                forward_tolerance_deg=8.0,
                drive_turn_tolerance_deg=30.0,
            ),
            "forward",
        )
        self.assertEqual(
            planner._action_from_steering_error(
                13.0,
                forward_tolerance_deg=20.0,
                drive_turn_tolerance_deg=30.0,
            ),
            "turn right",
        )
        smoothed = planner._smooth_path(
            np.ones((8, 8), dtype=bool),
            [(0, 0), (3, 3), (7, 7)],
        )
        self.assertEqual(smoothed[0], (0, 0))
        self.assertEqual(smoothed[-1], (7, 7))

    def test_policy_astar_recovery_turns_around_then_leaves_contact(self):
        planner = AStarBaseline(policy_actions=True, recovery_turn_steps=3)
        self.assertEqual(planner.recovery_turn_steps, 8)
        self.assertEqual(planner.recovery_forward_steps, 2)
        self.assertEqual(planner.stuck_block_ahead_m, 0.75)
        self.assertEqual(planner.stuck_block_radius_m, 0.9)
        self.assertEqual(planner.stuck_block_ttl_steps, 24)
        planner.recovery_count = planner.recovery_turn_steps
        planner.recovery_forward_count = planner.recovery_forward_steps
        actions = [planner._recovery_action() for _ in range(10)]
        self.assertEqual(actions[:8], ["turn right"] * 8)
        self.assertEqual(actions[8:], ["forward", "forward"])

        # The historical smooth A* baseline retains its three-turn recovery.
        benchmark_planner = AStarBaseline(policy_actions=False, recovery_turn_steps=3)
        self.assertEqual(benchmark_planner.recovery_turn_steps, 3)
        self.assertEqual(benchmark_planner.recovery_forward_steps, 0)

    def test_policy_astar_does_not_greedy_forward_through_temporary_block(self):
        planner = AStarBaseline(policy_actions=True, contrast_threshold=0)
        planner.virtual_obstacles = [(20, 20, 1, 1, 5)]
        action, reasoning, _ = planner.decide(
            np.zeros((64, 64, 3), dtype=np.uint8),
            (10, 10),
            (45, 45),
            agent_theta=0.0,
            curr_world_xz=(10.0, 10.0),
            target_world_xz=(45.0, 45.0),
            point_to_world=lambda point: (float(point[0]), float(point[1])),
        )
        self.assertEqual(action, "turn right")
        self.assertIn("temporary collision obstacle", reasoning)

    def test_astar_floor_threshold_adapts_to_randomized_bright_lighting(self):
        planner = AStarBaseline(contrast_threshold=0)
        gray = np.full((20, 20), 246, dtype=np.uint8)
        gray[0:4, :] = 100
        gray[4:6, :] = 255
        free, lower, upper, floor_reference = planner._threshold_free_space(gray)
        self.assertEqual(floor_reference, 246.0)
        self.assertEqual((lower, upper), (216, 254))
        self.assertTrue(np.all(free[6:, :]))
        self.assertFalse(np.any(free[:6, :]))

        normal = np.full((8, 8), 127, dtype=np.uint8)
        free, lower, upper, floor_reference = planner._threshold_free_space(normal)
        self.assertEqual(floor_reference, 127.0)
        self.assertEqual((lower, upper), (55, 190))
        self.assertTrue(np.all(free))

        bright_rgb = np.full((96, 96, 3), 246, dtype=np.uint8)
        bright_rgb[20, 20] = 80
        bright_rgb[40:48, 40:48] = 80
        planner = AStarBaseline(contrast_threshold=0, obstacle_clearance_m=0.0)
        planner._build_walkable_grid(bright_rgb, (8, 80), (80, 80))
        adapted_free = planner.last_debug["threshold_free"]
        self.assertTrue(planner.last_debug["lighting_adapted"])
        self.assertTrue(adapted_free[20, 20])
        self.assertFalse(np.any(adapted_free[40:48, 40:48]))

    def test_export_aligns_source_action_s_with_frame_s_minus_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            (raw / "astar_fp").mkdir(parents=True)
            (raw / "astar_depth").mkdir()
            for step, value in enumerate((23, 47)):
                Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(
                    raw / "astar_fp" / f"{step}.png"
                )
                Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(
                    raw / "astar_depth" / f"{step}.png"
                )
                np.save(raw / "astar_depth" / f"{step}.npy", np.full((4, 5), value))
            with (raw / "astar_actions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["step", "action"])
                writer.writeheader()
                writer.writerows([
                    {"step": 1, "action": "forward"},
                    {"step": 2, "action": "stop"},
                ])
            destination = root / "dataset" / "scene1" / "episode"
            destination.mkdir(parents=True)
            count, actions = export_episode({}, raw, destination)
            self.assertEqual(count, 2)
            self.assertEqual(actions["stop"], 1)
            with (destination / "keyboard_actions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["step"], "0")
            self.assertEqual(rows[0]["source_action_step"], "1")
            self.assertEqual(rows[0]["source_frame_step"], "0")
            exported = np.asarray(Image.open(destination / "keyboard_fp" / "0.png"))
            self.assertTrue(np.all(exported == 23))


if __name__ == "__main__":
    unittest.main()
