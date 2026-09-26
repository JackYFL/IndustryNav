from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nav.baselines.rl.trainer import (
    balanced_teacher_class_weights,
    optimizer_parameter_groups,
    read_tasks,
    resolve_training_device,
    scheduled_teacher_probability,
    select_rollout_action,
    teacher_auxiliary_scale,
)
from nav.baselines.rl.replay import TeacherReplayBuffer, soft_inverse_class_weights
from nav.harness.coordinates import (
    build_axis_aligned_projector,
    unity_to_world_coords,
    world_to_unity_coords,
)
from nav.core.geometry import remaining_path_length_m
from nav.harness.routing import execute_decision
from nav.safety import ReactiveSafetyShield
from nav.baselines.rl.pointgoal_ppo import (
    PPO_ACTIONS,
    PPO_BOS_LABEL,
    PointGoalActorCritic,
    PointGoalPPOConfig,
    append_action_history,
    append_visual_history,
    collision_started,
    compute_pointgoal_reward,
    depth_observation_uint8,
    encode_rl_goal,
    encode_rl_goal_state,
    generalized_advantage_estimate,
    initial_action_history,
    initial_visual_history,
    is_turn_reversal,
    load_compatible_pointgoal_state_dict,
    RandomSceneBlockTaskSampler,
    round_robin_scene_tasks,
    should_trigger_stuck_recovery,
)
from nav.baselines.rl.actions import pointgoal_homing_action
from nav.baselines.rl.route_graph import RouteGraphController
from nav.models.policies import SceneWaypointConfig, SceneWaypointPlanner
from nav.scripts.rl.train_scene_waypoint import episode_waypoint_samples
from nav.scripts.evaluation.evaluate_pointgoal_policy import (
    should_abandon_fallback,
    should_activate_fallback,
)


class PointGoalPPOTest(unittest.TestCase):
    @staticmethod
    def _small_model() -> PointGoalActorCritic:
        return PointGoalActorCritic(
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

    def test_action_history_keeps_five_actions_and_resets_on_done(self):
        history = initial_action_history(2, 5, device=torch.device("cpu"))
        self.assertEqual(history.tolist(), [[PPO_BOS_LABEL] * 5] * 2)

        for first, second in ((0, 1), (1, 2), (2, 0), (0, 1), (1, 2)):
            history = append_action_history(
                history, torch.tensor([first, second])
            )
        self.assertEqual(history.tolist(), [[0, 1, 2, 0, 1], [1, 2, 0, 1, 2]])

        history = append_action_history(
            history,
            torch.tensor([2, 0]),
            episode_done=torch.tensor([True, False]),
        )
        self.assertEqual(history[0].tolist(), [PPO_BOS_LABEL] * 5)
        self.assertEqual(history[1].tolist(), [2, 0, 1, 2, 0])

    def test_fallback_uses_live_collision_count_before_episode_ends(self):
        self.assertFalse(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=19,
            collision_threshold=20,
        ))
        self.assertTrue(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=0,
            collision_threshold=0,
            stagnation_steps=24,
            stagnation_threshold=24,
        ))
        self.assertTrue(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
        ))
        self.assertFalse(should_activate_fallback(
            fallback_available=True,
            fallback_active=True,
            collision_steps=21,
            collision_threshold=20,
        ))
        self.assertFalse(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            stagnation_steps=7,
            collision_stagnation_threshold=8,
        ))
        self.assertTrue(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            stagnation_steps=8,
            collision_stagnation_threshold=8,
        ))
        self.assertFalse(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            distance_m=10.0,
            min_distance_m=15.0,
        ))
        self.assertTrue(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            distance_m=16.0,
            min_distance_m=15.0,
        ))
        self.assertFalse(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            distance_m=7.0,
            min_distance_m=15.0,
            episode_steps=74,
            episode_max_steps=100,
            close_recovery_min_distance_m=6.5,
            close_recovery_max_distance_m=8.0,
            close_recovery_min_step_fraction=0.75,
        ))
        self.assertFalse(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            distance_m=20.0,
            min_distance_m=15.0,
            min_step_fraction=0.55,
            episode_steps=54,
            episode_max_steps=100,
        ))
        self.assertTrue(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            distance_m=35.0,
            min_distance_m=15.0,
            min_step_fraction=0.55,
            far_distance_m=35.0,
            episode_steps=20,
            episode_max_steps=100,
        ))
        self.assertTrue(should_activate_fallback(
            fallback_available=True,
            fallback_active=False,
            collision_steps=20,
            collision_threshold=20,
            distance_m=7.0,
            min_distance_m=15.0,
            episode_steps=75,
            episode_max_steps=100,
            close_recovery_min_distance_m=6.5,
            close_recovery_max_distance_m=8.0,
            close_recovery_min_step_fraction=0.75,
        ))

    def test_repeated_turn_escape_can_abandon_direct_fallback(self):
        self.assertFalse(should_abandon_fallback(
            fallback_active=True,
            turn_escape_interventions=6,
            activation_turn_escape_interventions=2,
            threshold=5,
        ))
        self.assertTrue(should_abandon_fallback(
            fallback_active=True,
            turn_escape_interventions=7,
            activation_turn_escape_interventions=2,
            threshold=5,
        ))
        self.assertFalse(should_abandon_fallback(
            fallback_active=False,
            turn_escape_interventions=7,
            activation_turn_escape_interventions=2,
            threshold=5,
        ))
    def test_route_graph_follows_demonstrated_detour(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "route_graph.json"
            checkpoint.write_text(json.dumps({
                "config": {
                    "lookahead_m": 2.0,
                    "max_snap_distance_m": 2.0,
                },
                "scenes": {
                    "0": {
                        "nodes": [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
                        "edges": [[0, 1], [1, 2]],
                    },
                },
            }), encoding="utf-8")
            controller = RouteGraphController(
                str(checkpoint), terminal_distance_m=0.5
            )
            waypoint = controller.predict_waypoint(
                scene_id=0,
                curr_world_x=0.0,
                curr_world_z=0.0,
                target_world_x=2.0,
                target_world_z=2.0,
            )
        self.assertEqual(waypoint, (0.0, 2.0))

    def test_safety_shield_turns_toward_clearer_depth_half(self):
        shield = ReactiveSafetyShield()
        depth = np.full((120, 160), 5.0, dtype=np.float32)
        roi = shield.warning_detector.create_roi_mask(depth.shape) == 1
        left = roi.copy()
        left[:, depth.shape[1] // 2 :] = False
        depth[left] = 0.5
        shield.begin_collision_recovery(
            depth,
            goal_bearing_sin=-1.0,
            move_command=7.5,
        )
        action, intervened = shield.filter_action("forward")
        self.assertTrue(intervened)
        self.assertEqual(action, "turn right")

    def test_safety_shield_preserves_safe_forward(self):
        shield = ReactiveSafetyShield()
        depth = np.full((120, 160), 5.0, dtype=np.float32)
        action, intervened = shield.filter_action("forward")
        self.assertFalse(intervened)
        self.assertEqual(action, "forward")

    def test_pointgoal_homing_uses_atomic_yaw_deadband(self):
        self.assertEqual(pointgoal_homing_action(0.0, 1.0), "forward")
        self.assertEqual(pointgoal_homing_action(1.0, 0.0), "turn right")
        self.assertEqual(pointgoal_homing_action(-1.0, 0.0), "turn left")
        self.assertEqual(
            pointgoal_homing_action(
                np.sin(np.deg2rad(15.0)),
                np.cos(np.deg2rad(15.0)),
                turn_tolerance_deg=20.0,
            ),
            "forward",
        )

    def test_scene_waypoint_planner_output_and_route_labels(self):
        model = SceneWaypointPlanner(
            SceneWaypointConfig(
                num_scenes=2,
                coordinate_fourier_bands=2,
                scene_embed_dim=4,
                hidden_size=16,
                lookahead_m=3.0,
            )
        )
        prediction = model(
            torch.tensor([0, 1]),
            torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            torch.tensor([[10.0, 0.0], [1.0, 8.0]]),
        )
        self.assertEqual(tuple(prediction.shape), (2, 2))
        self.assertTrue(torch.all(prediction.abs() <= 1.0))

        rows = [
            {
                "curr_world_x": str(x),
                "curr_world_z": "0",
                "target_world_x": "5",
                "target_world_z": "0",
            }
            for x in range(6)
        ]
        scene, current, target, label = episode_waypoint_samples(
            rows, scene_id=1, lookahead_m=3.0
        )
        self.assertEqual(scene.tolist(), [1] * 6)
        np.testing.assert_allclose(current[0], [0.0, 0.0])
        np.testing.assert_allclose(target[0], [5.0, 0.0])
        np.testing.assert_allclose(label[0], [1.0, 0.0])
        np.testing.assert_allclose(label[-1], [0.0, 0.0])

    def test_safety_shield_escapes_forward_only_after_depth_is_clear(self):
        shield = ReactiveSafetyShield()
        clear = np.full((120, 160), 5.0, dtype=np.float32)
        blocked = clear.copy()
        roi = shield.warning_detector.create_roi_mask(blocked.shape) == 1
        blocked[roi] = 0.2
        shield.begin_collision_recovery(
            clear,
            goal_bearing_sin=1.0,
            move_command=7.5,
            turns=1,
            escape_forwards=2,
        )
        self.assertEqual(
            shield.filter_action("forward", depth_m=clear)[0],
            "turn right",
        )
        # A newly visible obstacle extends the consistent clearance turn and
        # does not consume the requested escape translation.
        self.assertEqual(
            shield.filter_action("forward", depth_m=blocked)[0],
            "turn right",
        )
        self.assertEqual(shield.recovery_forwards_remaining, 2)
        self.assertEqual(
            shield.filter_action("turn left", depth_m=clear)[0],
            "forward",
        )
        self.assertEqual(shield.recovery_forwards_remaining, 1)

    def test_safety_shield_bounds_blocked_clearance_turns(self):
        shield = ReactiveSafetyShield(max_clearance_turns=2)
        blocked = np.full((120, 160), 5.0, dtype=np.float32)
        roi = shield.warning_detector.create_roi_mask(blocked.shape) == 1
        blocked[roi] = 0.2
        shield.begin_collision_recovery(
            blocked,
            goal_bearing_sin=1.0,
            move_command=7.5,
            turns=1,
            escape_forwards=2,
        )
        self.assertTrue(shield.filter_action("forward", depth_m=blocked)[1])
        self.assertTrue(shield.filter_action("forward", depth_m=blocked)[1])
        self.assertTrue(shield.filter_action("forward", depth_m=blocked)[1])
        action, intervened = shield.filter_action("turn left", depth_m=blocked)
        self.assertFalse(intervened)
        self.assertEqual(action, "turn left")
        self.assertEqual(shield.recovery_forwards_remaining, 0)

    def test_ppo_controller_tracks_shield_executed_action(self):
        from nav.baselines.rl.agent import PPOPointGoalController

        controller = PPOPointGoalController.__new__(PPOPointGoalController)
        controller.device = torch.device("cpu")
        controller.action_history = torch.tensor([[3, 3, 0, 0, 0]])
        controller.observe_executed_action("turn left")
        self.assertEqual(
            controller.action_history.tolist(),
            [[3, 3, 0, 0, PPO_ACTIONS.index("turn left")]],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported executed PPO action"):
            controller.observe_executed_action("stop")

    def test_teacher_auxiliary_scale_decays_to_floor(self):
        self.assertEqual(teacher_auxiliary_scale(1, 100, 0.1), 1.0)
        self.assertAlmostEqual(teacher_auxiliary_scale(51, 100, 0.1), 0.5)
        self.assertEqual(teacher_auxiliary_scale(101, 100, 0.1), 0.1)
        self.assertEqual(teacher_auxiliary_scale(999, 0, 0.1), 1.0)

    def test_teacher_action_probability_uses_absolute_floor(self):
        self.assertEqual(scheduled_teacher_probability(1, 0.5, 100, 0.1), 0.5)
        self.assertAlmostEqual(
            scheduled_teacher_probability(51, 0.5, 100, 0.1), 0.25
        )
        self.assertEqual(
            scheduled_teacher_probability(101, 0.5, 100, 0.1), 0.1
        )
        self.assertEqual(scheduled_teacher_probability(1, 0.0, 100, 0.0), 0.0)

    def test_teacher_class_weights_balance_forward_dominated_labels(self):
        counts = torch.tensor([70.0, 15.0, 15.0])
        weights = balanced_teacher_class_weights(counts)
        self.assertGreater(weights[1].item(), weights[0].item())
        self.assertGreater(weights[2].item(), weights[0].item())
        self.assertAlmostEqual(
            float((weights * counts).sum() / counts.sum()),
            1.0,
            places=6,
        )

    def test_teacher_replay_is_bounded_resized_and_restorable(self):
        buffer = TeacherReplayBuffer(4, img_size=8, seed=4)
        rollout = {
            "depth": torch.randint(0, 255, (2, 3, 1, 12, 10), dtype=torch.uint8),
            "goal": torch.randn(2, 3, 3),
            "action_history": torch.zeros(2, 3, 5, dtype=torch.long),
            "visual_history": torch.randn(2, 3, 5, 6),
            "hidden": torch.randn(2, 3, 7),
            "masks": torch.ones(2, 3, 1),
            "teacher_actions": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        }
        buffer.add_rollout(rollout)
        self.assertEqual(len(buffer), 4)
        batch = buffer.sample(3, device=torch.device("cpu"))
        self.assertEqual(tuple(batch["depth"].shape), (3, 1, 8, 8))
        self.assertEqual(batch["visual_history"].dtype, torch.float32)
        restored = TeacherReplayBuffer(4, img_size=8, seed=999)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(len(restored), 4)
        torch.testing.assert_close(restored.action_counts(3), torch.tensor([1.0, 1.0, 2.0]))

    def test_soft_replay_class_weights_are_tempered(self):
        counts = torch.tensor([100.0, 25.0, 25.0])
        weights = soft_inverse_class_weights(counts, power=0.5)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(
            float((weights * counts).sum() / counts.sum()), 1.0, places=6
        )

    def test_deterministic_dagger_rollout_uses_argmax(self):
        logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 1.0, 2.0]])
        self.assertEqual(
            select_rollout_action(logits, "argmax").tolist(),
            [1, 0],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            select_rollout_action(logits, "invalid")

    def test_calibrated_minimap_projection_round_trip(self):
        projector = build_axis_aligned_projector(
            (100, 400),
            (10, 20),
            (500, 100),
            (30, 60),
        )
        pixel = world_to_unity_coords(projector, 20, 40)
        np.testing.assert_allclose(pixel, [300, 250], atol=1e-6)
        world = unity_to_world_coords(projector, *pixel)
        np.testing.assert_allclose(world, [20, 40], atol=1e-6)

    def test_old_single_action_checkpoint_expands_to_five_slots(self):
        common = dict(
            depth_backbone="resnet18",
            img_size=32,
            visual_dim=8,
            goal_hidden=4,
            action_embed_dim=2,
            hidden_size=8,
        )
        old_model = PointGoalActorCritic(PointGoalPPOConfig(
            **common,
            action_history_len=1,
            visual_history_len=0,
        ))
        new_model = PointGoalActorCritic(
            PointGoalPPOConfig(
                **common,
                action_history_len=5,
                visual_history_len=5,
            )
        )
        old_state = old_model.state_dict()
        migration = load_compatible_pointgoal_state_dict(new_model, old_state)
        self.assertEqual(migration["action_history"], (1, 5))
        self.assertEqual(migration["visual_history"], (0, 5))

        base_features = common["visual_dim"] + common["goal_hidden"]
        expanded = new_model.state_dict()["gru.weight_ih"]
        original = old_state["gru.weight_ih"]
        torch.testing.assert_close(
            expanded[:, :base_features], original[:, :base_features]
        )
        self.assertTrue(
            torch.count_nonzero(expanded[:, base_features:-2]).item() == 0
        )
        torch.testing.assert_close(expanded[:, -2:], original[:, -2:])
        self.assertTrue(
            torch.count_nonzero(new_model.visual_history_gate).item() == 0
        )

    def test_visual_history_keeps_five_features_and_resets_on_done(self):
        history = initial_visual_history(
            2, 5, 3, device=torch.device("cpu")
        )
        self.assertEqual(tuple(history.shape), (2, 5, 3))
        for value in range(1, 7):
            current = torch.tensor(
                [[value] * 3, [value + 10] * 3], dtype=torch.float32
            )
            history = append_visual_history(history, current)
        self.assertEqual(history[0, :, 0].tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(history[1, :, 0].tolist(), [12, 13, 14, 15, 16])

        history = append_visual_history(
            history,
            torch.tensor([[7.0] * 3, [17.0] * 3]),
            episode_done=torch.tensor([True, False]),
        )
        self.assertTrue(torch.count_nonzero(history[0]).item() == 0)
        self.assertEqual(history[1, :, 0].tolist(), [13, 14, 15, 16, 17])

    def test_optimizer_uses_separate_policy_temporal_learning_rate(self):
        model = self._small_model()
        groups = optimizer_parameter_groups(
            model,
            lr=1.0e-4,
            encoder_lr_scale=0.1,
            policy_lr_scale=2.5,
        )
        by_name = {group["group_name"]: group for group in groups}
        self.assertEqual(set(by_name), {"depth_encoder", "policy_temporal", "value"})
        self.assertAlmostEqual(by_name["depth_encoder"]["lr"], 1.0e-5)
        self.assertAlmostEqual(by_name["policy_temporal"]["lr"], 2.5e-4)
        self.assertAlmostEqual(by_name["value"]["lr"], 1.0e-4)
        grouped_ids = [
            id(parameter)
            for group in groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(len(grouped_ids), len(list(model.parameters())))

    def test_cpu_training_device_is_available_for_single_process_smoke(self):
        device = resolve_training_device("cpu", world_size=1, local_rank=0)
        self.assertEqual(device, torch.device("cpu"))

    def test_cpu_training_device_rejects_distributed_training(self):
        with self.assertRaisesRegex(SystemExit, "requires CUDA"):
            resolve_training_device("cpu", world_size=2, local_rank=0)

    def test_routing_dispatches_ppo_controller(self):
        class Controller:
            def predict_action(self, **kwargs):
                self.kwargs = kwargs
                return "turn right"

        controller = Controller()
        result = {}
        execute_decision(
            "ppo",
            {
                "ppo_controller": controller,
                "depth_obs": np.zeros((1, 2, 2)),
                "curr_world_x": 1.0,
                "curr_world_z": 2.0,
                "curr_yaw_deg": 90.0,
                "target_world_x": 3.0,
                "target_world_z": 4.0,
            },
            result,
        )
        self.assertEqual(result["action"], "turn right")
        self.assertFalse(result.get("error", False))
        self.assertEqual(controller.kwargs["target_world_z"], 4.0)

    def test_goal_uses_continuous_sine_cosine_bearing(self):
        forward = encode_rl_goal(0, 0, 0, 0, 10, 50)
        right = encode_rl_goal(0, 0, 0, 10, 0, 50)
        np.testing.assert_allclose(forward, [0.2, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(right, [0.2, 1.0, 0.0], atol=1e-6)

    def test_optional_absolute_scene_state_does_not_expose_benchmark_points(self):
        relative = encode_rl_goal(1, 2, 90, 11, 12, 50)
        state = encode_rl_goal_state(
            1,
            2,
            90,
            11,
            12,
            distance_scale_m=50,
            scene_id=7,
            num_scenes=24,
            world_coordinate_scale_m=50,
            include_absolute_scene_state=True,
        )
        self.assertEqual(state.shape, (35,))
        np.testing.assert_allclose(state[:3], relative)
        np.testing.assert_allclose(state[3:7], [0.02, 0.04, 0.22, 0.24])
        self.assertEqual(state[7:31].sum(), 1.0)
        self.assertEqual(state[7 + 7], 1.0)
        np.testing.assert_allclose(state[31:], [0.2, 0.2, 1.0, 0.0], atol=1e-6)

    def test_absolute_scene_state_supports_multiscale_coordinate_features(self):
        state = encode_rl_goal_state(
            1,
            2,
            90,
            11,
            12,
            distance_scale_m=50,
            scene_id=7,
            num_scenes=24,
            world_coordinate_scale_m=50,
            include_absolute_scene_state=True,
            coordinate_fourier_bands=2,
        )
        self.assertEqual(state.shape, (51,))
        base_coordinates = np.asarray([0.02, 0.04, 0.22, 0.24])
        expected = np.asarray([
            component
            for value in base_coordinates
            for band in range(2)
            for component in (
                np.sin(2 * np.pi * (2**band) * value),
                np.cos(2 * np.pi * (2**band) * value),
            )
        ])
        np.testing.assert_allclose(state[35:], expected, atol=1e-6)

    def test_old_relative_goal_checkpoint_expands_context_at_zero(self):
        common = dict(
            depth_backbone="resnet18",
            img_size=32,
            visual_dim=8,
            goal_hidden=4,
            action_embed_dim=2,
            action_history_len=5,
            visual_history_len=5,
            hidden_size=8,
        )
        old_model = PointGoalActorCritic(PointGoalPPOConfig(**common))
        context_model = PointGoalActorCritic(PointGoalPPOConfig(
            **common,
            absolute_scene_state=True,
        ))
        migration = load_compatible_pointgoal_state_dict(
            context_model,
            old_model.state_dict(),
        )
        self.assertEqual(migration["goal_state"], (3, 35))
        for key in ("goal_mlp.0.weight", "goal_actor.weight"):
            old_weight = old_model.state_dict()[key]
            expanded = context_model.state_dict()[key]
            torch.testing.assert_close(expanded[:, :3], old_weight)
            self.assertEqual(torch.count_nonzero(expanded[:, 3:]).item(), 0)

    def test_legacy_absolute_context_appends_planning_state_at_zero(self):
        config = PointGoalPPOConfig(
            depth_backbone="resnet18",
            img_size=32,
            visual_dim=8,
            goal_hidden=4,
            action_embed_dim=2,
            action_history_len=5,
            visual_history_len=5,
            hidden_size=8,
            absolute_scene_state=True,
        )
        model = PointGoalActorCritic(config)
        legacy_state = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        for key in ("goal_mlp.0.weight", "goal_actor.weight"):
            legacy_state[key] = legacy_state[key][:, :31].clone()
        migrated = PointGoalActorCritic(config)
        migration = load_compatible_pointgoal_state_dict(migrated, legacy_state)
        self.assertEqual(migration["goal_state"], (31, 35))
        for key in ("goal_mlp.0.weight", "goal_actor.weight"):
            weight = migrated.state_dict()[key]
            torch.testing.assert_close(weight[:, :31], legacy_state[key])
            self.assertEqual(torch.count_nonzero(weight[:, 31:]).item(), 0)

    def test_reward_combines_progress_and_safety_penalties(self):
        reward = compute_pointgoal_reward(
            10.0,
            9.5,
            success=False,
            timeout=False,
            collision=True,
            warning=True,
        )
        self.assertAlmostEqual(reward, -0.56)
        success = compute_pointgoal_reward(
            2.1,
            1.9,
            success=True,
            timeout=False,
            collision=False,
            warning=False,
        )
        self.assertAlmostEqual(success, 5.19)

    def test_reward_can_use_geodesic_path_progress_for_detours(self):
        reward = compute_pointgoal_reward(
            10.0,
            10.5,
            success=False,
            timeout=False,
            collision=False,
            warning=False,
            path_progress_m=0.3,
            progress_scale=1.5,
        )
        self.assertAlmostEqual(reward, 0.44)

    def test_remaining_path_length_projects_pose_onto_cached_path(self):
        path = [(0, 0), (3, 0), (3, 4)]
        distance = remaining_path_length_m(
            path,
            current_world=(2.0, 0.0),
            point_to_world=lambda point: (float(point[0]), float(point[1])),
        )
        self.assertAlmostEqual(distance, 5.0)

    def test_reward_penalizes_turn_oscillation_and_stagnation(self):
        reward = compute_pointgoal_reward(
            10.0,
            10.0,
            success=False,
            timeout=False,
            collision=False,
            warning=False,
            action_name="turn left",
            turn_reversal=True,
            stagnation_steps=12,
            rotation_penalty=0.01,
            turn_reversal_penalty=0.05,
            stagnation_penalty=0.02,
            stagnation_start_steps=12,
        )
        self.assertAlmostEqual(reward, -0.09)
        self.assertTrue(is_turn_reversal("turn right", "turn left"))
        self.assertFalse(is_turn_reversal("forward", "turn left"))

    def test_reward_rewards_only_safe_forward_motion(self):
        safe = compute_pointgoal_reward(
            10.0,
            9.5,
            success=False,
            timeout=False,
            collision=False,
            warning=False,
            action_name="forward",
            progress_scale=1.5,
            safe_forward_bonus=0.02,
        )
        warned = compute_pointgoal_reward(
            10.0,
            9.5,
            success=False,
            timeout=False,
            collision=False,
            warning=True,
            action_name="forward",
            progress_scale=1.5,
            safe_forward_bonus=0.02,
        )
        self.assertAlmostEqual(safe, 0.76)
        self.assertAlmostEqual(warned, 0.69)

    def test_collision_penalty_applies_only_on_contact_onset(self):
        self.assertTrue(collision_started(True, False))
        self.assertFalse(collision_started(True, True))
        onset = compute_pointgoal_reward(
            10.0,
            10.0,
            success=False,
            timeout=False,
            collision=True,
            collision_onset=True,
            warning=False,
            collision_penalty=1.0,
        )
        persistent = compute_pointgoal_reward(
            10.0,
            10.0,
            success=False,
            timeout=False,
            collision=True,
            collision_onset=False,
            warning=False,
            collision_penalty=1.0,
        )
        self.assertAlmostEqual(onset, -1.01)
        self.assertAlmostEqual(persistent, -0.01)

    def test_collision_step_penalty_discourages_repeated_blocked_forward(self):
        persistent = compute_pointgoal_reward(
            10.0,
            10.0,
            success=False,
            timeout=False,
            collision=True,
            collision_onset=False,
            warning=False,
            collision_penalty=2.0,
            collision_step_penalty=0.1,
        )
        self.assertAlmostEqual(persistent, -0.11)

    def test_stuck_recovery_terminates_only_before_timeout(self):
        self.assertFalse(
            should_trigger_stuck_recovery(47, 48, success=False, timeout=False)
        )
        self.assertTrue(
            should_trigger_stuck_recovery(48, 48, success=False, timeout=False)
        )
        self.assertFalse(
            should_trigger_stuck_recovery(48, 48, success=True, timeout=False)
        )
        self.assertFalse(
            should_trigger_stuck_recovery(48, 48, success=False, timeout=True)
        )
        reward = compute_pointgoal_reward(
            10.0,
            10.0,
            success=False,
            timeout=False,
            collision=False,
            warning=False,
            stuck_recovery=True,
            stuck_recovery_penalty=2.0,
        )
        self.assertAlmostEqual(reward, -2.01)

    def test_random_sampler_mixes_tasks_in_bounded_scene_blocks(self):
        tasks = [
            {"scene_name": f"scene{scene}", "episode_id": f"s{scene}-{index}"}
            for scene in (1, 2)
            for index in range(5)
        ]
        first = RandomSceneBlockTaskSampler(
            tasks, seed=17, scene_block_episodes=3
        )
        second = RandomSceneBlockTaskSampler(
            tasks, seed=17, scene_block_episodes=3
        )
        sequence = [first.next_task() for _ in range(12)]
        self.assertEqual(
            [task["episode_id"] for task in sequence],
            [second.next_task()["episode_id"] for _ in range(12)],
        )
        for start in range(0, 12, 3):
            self.assertEqual(
                len({task["scene_name"] for task in sequence[start : start + 3]}),
                1,
            )
        for scene in ("scene1", "scene2"):
            ids = [
                task["episode_id"]
                for task in sequence
                if task["scene_name"] == scene
            ][:5]
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)

    def test_random_sampler_can_abandon_failed_scene_block(self):
        tasks = [
            {"scene_name": f"scene{scene}", "episode_id": f"s{scene}-{index}"}
            for scene in (1, 2)
            for index in range(3)
        ]
        sampler = RandomSceneBlockTaskSampler(
            tasks, seed=9, scene_block_episodes=3
        )
        failed = sampler.next_task()
        sampler.skip_current_scene()
        recovered = sampler.next_task()
        self.assertNotEqual(failed["scene_name"], recovered["scene_name"])

    def test_training_can_mix_short_and_long_distance_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            short = root / "short.jsonl"
            long = root / "long.jsonl"
            short.write_text(
                json.dumps({
                    "split": "train", "scene_name": "scene1",
                    "episode_id": "short01",
                }) + "\n",
                encoding="utf-8",
            )
            long.write_text(
                json.dumps({
                    "split": "train", "scene_name": "scene1",
                    "episode_id": "long01",
                }) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                manifest=short,
                extra_manifest=[long],
                split="train",
                scene_limit=0,
                episodes_per_scene=0,
                num_envs=1,
            )
            shards = read_tasks(args, rank=0, world_size=1)
        self.assertEqual(
            [task["episode_id"] for task in shards[0]],
            ["short01", "long01"],
        )

    def test_all_split_uses_every_resampled_scene_without_input_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "tasks.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps({
                        "split": split,
                        "scene_name": f"scene{index}",
                        "episode_id": f"resampled{index:02d}",
                    })
                    for index, split in enumerate(
                        ("train", "val", "test"), start=1
                    )
                ) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                manifest=manifest,
                extra_manifest=[],
                split="all",
                scene_limit=0,
                episodes_per_scene=0,
                num_envs=1,
            )
            shards = read_tasks(args, rank=0, world_size=1)
        self.assertEqual(len(shards[0]), 3)

    def test_gae_respects_episode_boundary(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.zeros_like(rewards)
        masks = torch.tensor([[1.0], [0.0]])
        advantages, returns = generalized_advantage_estimate(
            rewards,
            values,
            torch.tensor([10.0]),
            masks,
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, torch.tensor([[3.0], [2.0]]))
        torch.testing.assert_close(returns, advantages)

    def test_depth_conversion_accepts_hwc(self):
        depth = np.full((3, 4, 1), 0.5, dtype=np.float32)
        result = depth_observation_uint8(depth)
        self.assertEqual(result.shape, (1, 3, 4))
        self.assertTrue(np.all(result == 127))

    def test_round_robin_tasks_interleave_scenes(self):
        tasks = [
            {"scene_name": "scene2", "episode_id": "b1"},
            {"scene_name": "scene1", "episode_id": "a1"},
            {"scene_name": "scene1", "episode_id": "a2"},
            {"scene_name": "scene2", "episode_id": "b2"},
        ]
        ordered = round_robin_scene_tasks(tasks)
        self.assertEqual(
            [task["episode_id"] for task in ordered],
            ["a1", "b1", "a2", "b2"],
        )


if __name__ == "__main__":
    unittest.main()
