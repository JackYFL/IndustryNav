from __future__ import annotations

import unittest

import numpy as np
import torch

from nav.harness.routing import execute_decision
from nav.baselines.rl.pointgoal_ppo import (
    compute_pointgoal_reward,
    depth_observation_uint8,
    encode_rl_goal,
    generalized_advantage_estimate,
    round_robin_scene_tasks,
)


class PointGoalPPOTest(unittest.TestCase):
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

    def test_reward_combines_progress_and_safety_penalties(self):
        reward = compute_pointgoal_reward(
            10.0,
            9.5,
            success=False,
            timeout=False,
            collision=True,
            warning=True,
        )
        self.assertAlmostEqual(reward, -0.06)
        success = compute_pointgoal_reward(
            2.1,
            1.9,
            success=True,
            timeout=False,
            collision=False,
            warning=False,
        )
        self.assertAlmostEqual(success, 5.19)

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
