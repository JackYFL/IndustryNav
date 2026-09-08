from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from nav.config import (
    ACTION_SPACE_POINTGOAL,
    BC_NAV_ACTION_TO_LABEL,
    BC_NAV_BOS_LABEL,
)
from nav.baselines.astar import AStarBaseline
from nav.scripts.evaluation.evaluate_pointgoal_policy import (
    command_for as pointgoal_eval_command,
    select_held_out_tasks,
)
from nav.scripts.astar.export_astar_pointgoal_dataset import export_episode
from nav.scripts.bc.export_dagger_pointgoal_dataset import (
    export_episode as export_dagger_episode,
)
from nav.scripts.astar.prepare_astar_pointgoal_pilot import build_manifest
from nav.scripts.agent.run_benchmark_cell import action_space_for_baseline
from nav.harness.routing import execute_decision
from nav.models.policy import NavPolicyTransformer
from nav.train.controller import BCNavController
from nav.train.dataset import NavEpisodeSequenceDataset, mirror_pointgoal_sample
from nav.train.loop import (
    evaluate,
    get_class_weights,
    mask_untrained_stop_logits,
    turn_direction_loss,
    initialize_from_checkpoint,
)
from nav.train.pointgoal import (
    POINTGOAL_ENCODING_LEGACY,
    POINTGOAL_ENCODING_UNITY,
    encode_pointgoal,
)


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
        dataset.goal_encoding = POINTGOAL_ENCODING_UNITY
        dataset.goal_distance_scale_m = 50.0
        goal = dataset._build_goal({
            "curr_world_x": "0",
            "curr_world_z": "0",
            "curr_direction_y": "0",
            "target_world_x": "3",
            "target_world_z": "4",
        })
        self.assertAlmostEqual(float(goal[0]), 0.1)
        self.assertAlmostEqual(float(goal[1]), math.atan2(3, 4) / math.pi)

        controller = BCNavController.__new__(BCNavController)
        controller.goal_rep = "polar"
        controller.goal_encoding = POINTGOAL_ENCODING_UNITY
        controller.goal_distance_scale_m = 50.0
        inference_goal = controller._build_goal(0, 0, 0, 3, 4)
        np.testing.assert_allclose(inference_goal, goal)

    def test_unity_pointgoal_bearing_matches_forward_and_turn_directions(self):
        forward = encode_pointgoal(
            0, 0, 0, 0, 5,
            goal_rep="polar", distance_scale_m=0,
        )
        right = encode_pointgoal(
            0, 0, 0, 5, 0,
            goal_rep="polar", distance_scale_m=0,
        )
        left = encode_pointgoal(
            0, 0, 0, -5, 0,
            goal_rep="polar", distance_scale_m=0,
        )
        np.testing.assert_allclose(forward, [5, 0], atol=1e-6)
        np.testing.assert_allclose(right, [5, math.pi / 2], atol=1e-6)
        np.testing.assert_allclose(left, [5, -math.pi / 2], atol=1e-6)

        # At yaw=90 degrees Unity forward is world +X and right is world -Z.
        cartesian = encode_pointgoal(
            0, 0, 90, 5, -2,
            goal_rep="cartesian", distance_scale_m=1,
        )
        np.testing.assert_allclose(cartesian, [5, 2], atol=1e-6)

    def test_legacy_goal_encoding_remains_available_for_old_checkpoints(self):
        legacy = encode_pointgoal(
            0, 0, 0, 3, 4,
            goal_rep="polar", distance_scale_m=50,
            encoding=POINTGOAL_ENCODING_LEGACY,
        )
        self.assertAlmostEqual(float(legacy[0]), 0.1)
        self.assertAlmostEqual(float(legacy[1]), math.atan2(4, 3) / math.pi)

    def test_dagger_finetune_rejects_goal_encoding_mismatch(self):
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save({"model": model.state_dict(), "config": {}}, checkpoint)
            cfg = SimpleNamespace(
                init_checkpoint=str(checkpoint),
                goal_encoding=POINTGOAL_ENCODING_UNITY,
                navigation_only_actions=False,
            )
            with self.assertRaisesRegex(ValueError, "goal encoding mismatch"):
                initialize_from_checkpoint(model, cfg)

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

        combined = build_manifest(points, seeds=4, pairing="both")
        self.assertEqual(len(combined), 864)
        self.assertEqual(len({(r["scene_name"], r["episode_id"]) for r in combined}), 864)
        self.assertEqual(sum(r["split"] == "train" for r in combined), 576)
        self.assertEqual(
            {r["pairing"] for r in combined}, {"canonical", "cross"}
        )

    def test_macro_navigation_metric_and_turn_auxiliary_loss(self):
        logits = torch.tensor([
            [8.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
        ])

        class FixedPolicy(nn.Module):
            def forward(self, rgb, depth, goal, prev_action):
                return logits

        batch = {
            "rgb": torch.empty(0),
            "depth": torch.empty(0),
            "goal": torch.zeros(4, 1, 2),
            "prev_action": torch.zeros(4, 1, dtype=torch.long),
            "action": torch.tensor([0, 1, 2, 3]),
        }
        metrics = evaluate(FixedPolicy(), [batch], torch.device("cpu"), "transformer")
        self.assertAlmostEqual(metrics["acc"], 0.5)
        self.assertAlmostEqual(metrics["macro_acc"], 0.5)
        self.assertAlmostEqual(metrics["macro_nav_acc"], 2.0 / 3.0)
        self.assertEqual(metrics["support_class_3"], 1)
        self.assertLess(
            float(turn_direction_loss(logits[[2, 3]], torch.tensor([2, 2]))),
            0.01,
        )

    def test_absent_stop_class_has_zero_weight_and_is_masked(self):
        weights = get_class_weights({0: 33004, 1: 0, 2: 5631, 3: 5468}, power=0.5)
        self.assertEqual(float(weights[1]), 0.0)
        self.assertGreater(float(weights[0]), 0.0)
        self.assertGreater(float(weights[2]), float(weights[0]))

        logits = torch.tensor([[2.0, 100.0, 1.0, 0.0]])
        masked = mask_untrained_stop_logits(logits, include_stop_targets=False)
        self.assertEqual(int(masked.argmax(dim=-1).item()), 0)
        self.assertEqual(int(logits.argmax(dim=-1).item()), 1)

    def test_navigation_only_head_has_three_outputs_and_bos_history_token(self):
        weights = get_class_weights({0: 33004, 1: 5631, 2: 5468}, power=0.5)
        self.assertEqual(tuple(weights.shape), (3,))

        model = NavPolicyTransformer(
            goal_dim=2,
            num_actions=3,
            previous_action_vocab_size=4,
            use_depth=False,
            use_rgb=False,
            seq_len=2,
            num_layers=1,
        )
        logits = model(
            torch.empty(0),
            torch.empty(0),
            torch.tensor([[[0.5, -0.25], [0.4, -0.1]]]),
            torch.tensor([[BC_NAV_BOS_LABEL, BC_NAV_ACTION_TO_LABEL["forward"]]]),
        )
        self.assertEqual(tuple(logits.shape), (1, 3))

        _, _, _, previous, action = mirror_pointgoal_sample(
            [],
            [],
            [torch.tensor([0.5, 0.25])],
            [1, 2, 0, BC_NAV_BOS_LABEL],
            torch.tensor([1, 2, 0]),
            action_to_label=BC_NAV_ACTION_TO_LABEL,
        )
        self.assertEqual(previous, [2, 1, 0, BC_NAV_BOS_LABEL])
        self.assertTrue(torch.equal(action, torch.tensor([2, 1, 0])))

    def test_evaluation_masks_stop_for_three_action_pointgoal_policy(self):
        logits = torch.tensor([
            [8.0, 100.0, 0.0, 0.0],
            [0.0, 100.0, 8.0, 0.0],
            [0.0, 100.0, 0.0, 8.0],
        ])

        class FixedPolicy(nn.Module):
            def forward(self, rgb, depth, goal, prev_action):
                return logits

        batch = {
            "rgb": torch.empty(0),
            "depth": torch.empty(0),
            "goal": torch.zeros(3, 1, 2),
            "prev_action": torch.zeros(3, 1, dtype=torch.long),
            "action": torch.tensor([0, 2, 3]),
        }
        metrics = evaluate(
            FixedPolicy(),
            [batch],
            torch.device("cpu"),
            "transformer",
            include_stop_targets=False,
        )
        self.assertEqual(metrics["acc"], 1.0)
        self.assertEqual(metrics["predicted_class_1"], 0)

    def test_held_out_selection_excludes_failed_expert_episodes(self):
        records = [
            {"split": "test", "scene_name": "scene21", "episode_id": "failed"},
            {"split": "test", "scene_name": "scene21", "episode_id": "passed"},
            {"split": "test", "scene_name": "scene22", "episode_id": "passed"},
        ]
        eligible = [
            {"scene_name": "scene21", "episode_id": "passed"},
            {"scene_name": "scene22", "episode_id": "passed"},
        ]
        selected = select_held_out_tasks(records, 2, eligible)
        self.assertEqual(
            [(item["scene_name"], item["episode_id"]) for item in selected],
            [("scene21", "passed"), ("scene22", "passed")],
        )

    def test_pointgoal_evaluation_uses_scene_authored_lighting(self):
        args = SimpleNamespace(
            python="python",
            unity=Path("scene_all.x86_64"),
            checkpoint=Path("best.pt"),
            base_port=19507,
        )
        task = {
            "scene_id": 20,
            "scene_name": "scene21",
            "episode_id": "canonical01_seed0",
            "seed": 0,
            "dynamic_objects": "moving",
            "init_world_x": 34.66,
            "init_world_z": 56.13,
            "init_direction": 180.0,
            "target_x": 550,
            "target_y": 450,
            "motion_random_seed": 1000,
            "light_random_seed": 2000,
        }
        command = pointgoal_eval_command(args, task, 0, Path("eval/scene21"))
        for option in (
            "--global_light_intensity",
            "--light_intensity_multiplier",
            "--light_intensity_min",
            "--light_intensity_max",
            "--light_random_seed",
            "--light_fixed_exposure",
        ):
            self.assertNotIn(option, command)

    def test_transformer_goal_action_residual_is_optional(self):
        model = NavPolicyTransformer(
            goal_dim=2,
            use_depth=False,
            use_rgb=False,
            seq_len=2,
            num_layers=1,
            goal_action_residual=True,
        )
        logits = model(
            torch.empty(0),
            torch.empty(0),
            torch.tensor([[[0.5, -0.25], [0.4, -0.1]]]),
            torch.zeros(1, 2, dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertIsNotNone(model.goal_action_head)

    def test_policy_astar_uses_agent_action_space(self):
        self.assertIs(action_space_for_baseline("astar", True), ACTION_SPACE_POINTGOAL)
        self.assertIs(action_space_for_baseline("dagger"), ACTION_SPACE_POINTGOAL)
        self.assertIs(action_space_for_baseline("ppo"), ACTION_SPACE_POINTGOAL)
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

    def test_dagger_queries_expert_but_tracks_executed_policy_action(self):
        class FixedBC:
            def __init__(self):
                self.observed = []

            def predict_action(self, **_kwargs):
                return "turn left"

            def observe_executed_action(self, action):
                self.observed.append(action)

        class FixedAStar:
            plan_revision = 2
            last_plan_status = "target_reachable"
            last_tracking_metrics = {"cross_track_m": 0.25}

            def __init__(self):
                self.observed = []

            def decide(self, **_kwargs):
                return "turn right", "oracle", [(1, 2), (3, 4)]

            def observe_executed_action(self, action):
                self.observed.append(action)

        class FixedRng:
            def random(self):
                return 0.9

        bc = FixedBC()
        astar = FixedAStar()
        payload = {
            "bc_controller": bc,
            "astar_planner": astar,
            "dagger_beta": 0.5,
            "dagger_rng": FixedRng(),
            "ego_obs": None,
            "depth_obs": np.zeros((2, 2)),
            "curr_world_x": 0,
            "curr_world_z": 0,
            "curr_yaw_deg": 0,
            "target_world_x": 1,
            "target_world_z": 1,
            "minimap_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "curr_xy": (1, 2),
            "target_xy": (3, 4),
            "agent_theta": 0,
            "reach_m": 2,
            "step": 7,
            "curr_world_xz": (0, 0),
            "target_world_xz": (1, 1),
            "point_to_world": lambda point: point,
        }
        result = {}
        execute_decision("dagger", payload, result)
        self.assertFalse(result.get("error", False))
        self.assertEqual(result["action"], "turn left")
        self.assertEqual(result["dagger_policy_action"], "turn left")
        self.assertEqual(result["dagger_expert_action"], "turn right")
        self.assertFalse(result["dagger_used_expert"])
        self.assertEqual(bc.observed, ["turn left"])
        self.assertEqual(astar.observed, ["turn left"])

    def test_dagger_export_uses_oracle_label_not_behavior_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            destination = Path(tmp) / "exported"
            for subdir in ("dagger_fp", "dagger_depth"):
                (raw / subdir).mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(raw / "dagger_fp" / "0.png")
            Image.new("L", (2, 2)).save(raw / "dagger_depth" / "0.png")
            np.save(raw / "dagger_depth" / "0.npy", np.ones((2, 2)))
            action_row = {
                "step": 1,
                "action": "turn left",
                "move": 0,
                "strafe": 0,
                "look": -11.25,
                "curr_world_x": 0,
                "curr_world_z": 0,
                "curr_direction_y": 0,
                "target_world_x": 5,
                "target_world_z": 0,
            }
            with (raw / "dagger_actions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(action_row))
                writer.writeheader()
                writer.writerow(action_row)
            label_row = {
                "step": 1,
                "frame_step": 0,
                "policy_action": "turn left",
                "expert_action": "turn right",
                "executed_action": "turn left",
                "used_expert": False,
                "disagreement": True,
                "beta": 0.0,
                "expert_plan_status": "target_reachable",
            }
            with (raw / "dagger_labels.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(label_row))
                writer.writeheader()
                writer.writerow(label_row)

            destination.mkdir()
            steps, counts = export_dagger_episode(raw, destination)
            self.assertEqual(steps, 1)
            self.assertEqual(counts["turn right"], 1)
            with (destination / "keyboard_actions.csv").open(newline="") as stream:
                exported = next(csv.DictReader(stream))
            self.assertEqual(exported["action"], "turn right")
            self.assertEqual(exported["behavior_action"], "turn left")
            self.assertEqual(float(exported["look"]), 11.25)

    def test_dagger_sequence_history_uses_executed_behavior_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp) / "scene1" / "dagger_round1_ep"
            rgb_dir = episode / "keyboard_fp"
            depth_dir = episode / "keyboard_depth"
            rgb_dir.mkdir(parents=True)
            depth_dir.mkdir()
            rows = []
            for step, (oracle, behavior) in enumerate((
                ("turn right", "turn left"),
                ("forward", "forward"),
            )):
                Image.new("RGB", (2, 2)).save(rgb_dir / f"{step}.png")
                Image.new("L", (2, 2)).save(depth_dir / f"{step}.png")
                rows.append({
                    "step": step,
                    "action": oracle,
                    "behavior_action": behavior,
                    "curr_world_x": 0,
                    "curr_world_z": 0,
                    "curr_direction_y": 0,
                    "target_world_x": 0,
                    "target_world_z": 5,
                })
            with (episode / "keyboard_actions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            dataset = NavEpisodeSequenceDataset(
                tmp,
                split="train",
                split_ratios=(1.0, 0.0, 0.0),
                seq_len=1,
                img_size=2,
                use_depth=True,
                use_rgb=False,
                navigation_only_actions=True,
            )
            second = dataset._episode_steps[0][1]
            self.assertEqual(
                second["prev_action"],
                BC_NAV_ACTION_TO_LABEL["turn left"],
            )
            self.assertEqual(
                second["action"],
                BC_NAV_ACTION_TO_LABEL["forward"],
            )

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
