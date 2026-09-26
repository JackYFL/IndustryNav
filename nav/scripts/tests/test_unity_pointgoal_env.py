from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from mlagents_envs.exception import UnityEnvironmentException

from nav.envs.unity_pointgoal import UnityPointGoalEnv
from nav.harness.env_setup import EnvSetupError


class UnityPointGoalEnvTest(unittest.TestCase):
    @staticmethod
    def _task():
        return {
            "scene_id": 0,
            "scene_name": "scene1",
            "episode_id": "resampled01",
            "seed": 1,
            "init_world_x": 1.0,
            "init_world_z": 2.0,
            "init_direction": 0.0,
            "target_x": 100,
            "target_y": 200,
        }

    def test_launch_retries_transient_environment_setup_failure(self):
        task = self._task()
        primed = SimpleNamespace(target_world=(3.0, 4.0))
        env = UnityPointGoalEnv(
            [task],
            unity_path="scene_all.x86_64",
            output_dir="outputs/test",
            worker_id=0,
            base_port=50000,
            reward_fn=lambda *_args, **_kwargs: 0.0,
            logger=Mock(),
        )
        with patch.dict(
            "os.environ", {"INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS": "2"}
        ), patch(
            "nav.envs.unity_pointgoal.setup_and_prime",
            side_effect=[EnvSetupError("blank minimap"), primed],
        ) as setup:
            env._launch(task)

        self.assertIs(env.primed, primed)
        self.assertEqual(env.target_world, (3.0, 4.0))
        self.assertEqual(setup.call_count, 2)

    def _check_same_scene_reset_lifecycle(self, **kwargs):
        task = self._task()
        env = UnityPointGoalEnv(
            [task], unity_path="scene_all.x86_64", output_dir="outputs/test",
            worker_id=0, base_port=50000, task_sampling="sequential",
            reward_fn=lambda *_args, **_kwargs: 0.0, logger=Mock(), **kwargs,
        )
        env.task = task
        env.primed = object()
        env.distance_m = 10.0
        with patch.object(env, "_launch") as launch, patch.object(
            env, "_reset_same_scene",
        ) as reuse, patch.object(env, "_reset_online_teacher"), patch.object(
            env, "_read_observation", return_value={"depth": np.zeros((2, 2))},
        ):
            env.reset()
        self.assertEqual(env.task, task)
        return launch, reuse

    def test_same_scene_player_reuse_remains_default(self):
        launch, reuse = self._check_same_scene_reset_lifecycle()
        launch.assert_not_called()
        reuse.assert_called_once_with(self._task())

    def test_fresh_episode_relaunches_even_the_same_scene(self):
        launch, reuse = self._check_same_scene_reset_lifecycle(reuse_same_scene=False)
        launch.assert_called_once_with(self._task())
        reuse.assert_not_called()

    def test_training_cli_fresh_player_is_opt_in(self):
        from nav.scripts.rl.train_pointgoal_ppo import parse_args

        required = ["train_pointgoal_ppo", "--manifest", "unused.jsonl",
                    "--unity", "unused.x86_64", "--output-dir", "unused_output"]
        for extra, expected in (([], False), (["--fresh-unity-per-episode"], True)):
            with self.subTest(extra=extra), patch("sys.argv", required + extra):
                args = parse_args()
            self.assertIs(args.fresh_unity_per_episode, expected)
            self.assertEqual(args.scene_block_episodes, 4)

    def test_launch_retries_transient_unity_process_failure(self):
        task = self._task()
        primed = SimpleNamespace(target_world=(3.0, 4.0))
        env = UnityPointGoalEnv(
            [task],
            unity_path="scene_all.x86_64",
            output_dir="outputs/test",
            worker_id=0,
            base_port=50000,
            reward_fn=lambda *_args, **_kwargs: 0.0,
            logger=Mock(),
        )
        with patch.dict(
            "os.environ", {"INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS": "2"}
        ), patch(
            "nav.envs.unity_pointgoal.setup_and_prime",
            side_effect=[UnityEnvironmentException("SIGSEGV"), primed],
        ) as setup:
            env._launch(task)

        self.assertIs(env.primed, primed)
        self.assertEqual(setup.call_count, 2)
        self.assertEqual(setup.call_args.args[0].worker_id, 64)

    def test_reset_skips_task_after_exhausted_launch_attempts(self):
        bad_task = self._task()
        good_task = {
            **bad_task,
            "scene_id": 1,
            "scene_name": "scene2",
            "episode_id": "resampled02",
        }
        primed = SimpleNamespace(target_world=(3.0, 4.0))
        env = UnityPointGoalEnv(
            [bad_task, good_task],
            unity_path="scene_all.x86_64",
            output_dir="outputs/test",
            worker_id=0,
            base_port=50000,
            task_sampling="random",
            task_sampling_seed=7,
            scene_block_episodes=2,
            reward_fn=lambda *_args, **_kwargs: 0.0,
            logger=Mock(),
        )
        env.task_sampler.next_task = Mock(side_effect=[bad_task, good_task])
        env.task_sampler.skip_current_scene = Mock()
        with patch.object(
            env,
            "_launch",
            side_effect=[EnvSetupError("blank minimap"), None],
        ), patch.object(env, "_reset_online_teacher"), patch.object(
            env,
            "_read_observation",
            return_value={"depth": np.zeros((2, 2), dtype=np.uint8)},
        ):
            env.primed = primed
            env.target_world = primed.target_world
            observation = env.reset()

        self.assertEqual(env.task, good_task)
        self.assertIn("depth", observation)
        env.task_sampler.skip_current_scene.assert_called_once_with()

    def test_online_astar_teacher_receives_minimap_but_policy_does_not(self):
        task = self._task()
        env = UnityPointGoalEnv(
            [task],
            unity_path="scene_all.x86_64",
            output_dir="outputs/test",
            worker_id=0,
            base_port=50000,
            online_astar_teacher=True,
            reward_fn=lambda *_args, **_kwargs: 0.0,
            logger=Mock(),
        )
        env.task = task
        env.pose = (10.0, 20.0, 90.0)
        env.target_world = (30.0, 60.0)
        env.minimap_rgb = np.zeros((512, 862, 3), dtype=np.uint8)
        env.minimap_projector = {
            "spawn_pixel": (100.0, 400.0),
            "spawn_world": (10.0, 20.0),
            "target_pixel": (500.0, 100.0),
            "target_world": (30.0, 60.0),
        }
        env.primed = SimpleNamespace(
            margin=(0.0, 861.0, 6.0, 506.0),
            minimap_size=(862, 512),
        )
        env.astar_teacher = Mock()
        env.astar_teacher.decide.return_value = (
            "turn right",
            "test",
            [(100, 200), (110, 200)],
        )

        self.assertEqual(env.predict_astar_teacher_action(), "turn right")
        kwargs = env.astar_teacher.decide.call_args.kwargs
        self.assertIs(kwargs["minimap_rgb"], env.minimap_rgb)
        self.assertEqual(kwargs["curr_world_xz"], env.pose[:2])

    def test_dynamic_training_budget_uses_initial_distance_and_clamps(self):
        for distance, expected in ((1.0, 80), (34.5, 136), (200.0, 320)):
            with self.subTest(distance=distance):
                env = UnityPointGoalEnv(
                    [self._task()], unity_path="scene_all.x86_64",
                    output_dir="outputs/test", worker_id=0, base_port=50000,
                    max_steps=999, dynamic_step_budget=True,
                    step_budget_min=80, step_budget_max=320,
                    steps_per_path_meter=2.2, step_budget_overhead=60,
                    reward_fn=lambda *_args, **_kwargs: 0.0, logger=Mock(),
                )
                def observation():
                    env.distance_m = distance
                    return {"depth": np.zeros((2, 2), dtype=np.uint8)}
                with patch.object(env, "_launch"), patch.object(
                    env, "_reset_online_teacher",
                ), patch.object(env, "_read_observation", side_effect=observation):
                    env.reset()
                self.assertEqual(env.episode_max_steps, expected)
                self.assertEqual(env.episode_initial_distance_m, distance)

    def test_fixed_training_budget_remains_backward_compatible(self):
        env = UnityPointGoalEnv(
            [self._task()], unity_path="scene_all.x86_64",
            output_dir="outputs/test", worker_id=0, base_port=50000,
            max_steps=123, reward_fn=lambda *_args, **_kwargs: 0.0,
            logger=Mock(),
        )
        with patch.object(env, "_launch"), patch.object(
            env, "_reset_online_teacher",
        ), patch.object(env, "_read_observation", return_value={}):
            env.reset()
        self.assertEqual(env.episode_max_steps, 123)

    def test_online_astar_teacher_uses_configured_forward_tolerance(self):
        env = UnityPointGoalEnv(
            [self._task()],
            unity_path="scene_all.x86_64",
            output_dir="outputs/test",
            worker_id=0,
            base_port=50000,
            online_astar_teacher=True,
            astar_policy_forward_tolerance_deg=20.0,
            reward_fn=lambda *_args, **_kwargs: 0.0,
            logger=Mock(),
        )
        env.primed = SimpleNamespace(
            target_sc=SimpleNamespace(
                last_spawn_pixel=(1, 2),
                last_spawn_world=(3.0, 4.0),
                last_target_pixel=(5, 6),
                last_target_world=(7.0, 8.0),
            ),
            minimap_size=(862, 512),
        )
        with patch(
            "nav.envs.unity_pointgoal.build_axis_aligned_projector",
            return_value={"projector": True},
        ), patch("nav.envs.unity_pointgoal.AStarBaseline") as astar:
            env._reset_online_teacher()
        self.assertEqual(
            astar.call_args.kwargs["policy_forward_tolerance_deg"], 20.0
        )

    def test_online_astar_teacher_rejects_invalid_forward_tolerance(self):
        with self.assertRaisesRegex(ValueError, "must be in \\[0, 90\\]"):
            UnityPointGoalEnv(
                [self._task()],
                unity_path="scene_all.x86_64",
                output_dir="outputs/test",
                worker_id=0,
                base_port=50000,
                astar_policy_forward_tolerance_deg=91.0,
                reward_fn=lambda *_args, **_kwargs: 0.0,
                logger=Mock(),
            )

    def test_geodesic_progress_reward_requires_online_astar_teacher(self):
        with self.assertRaisesRegex(ValueError, "requires online_astar_teacher"):
            UnityPointGoalEnv(
                [self._task()],
                unity_path="scene_all.x86_64",
                output_dir="outputs/test",
                worker_id=0,
                base_port=50000,
                geodesic_progress_reward=True,
                reward_fn=lambda *_args, **_kwargs: 0.0,
                logger=Mock(),
            )


if __name__ == "__main__":
    unittest.main()
