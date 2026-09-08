"""Persistent Unity environment wrapper for Python-side PointGoal RL."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple

from nav.config import (
    ACTION_SPACE_POINTGOAL,
    BEHAVIOR_NAME,
    EVAL_COLLISION_MIN_FORWARD_RATIO,
    EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M,
)
from nav.eval.warning import WarningDetector
from nav.harness.coordinates import canonical_to_minimap_coords, visual_to_unity_coords
from nav.harness.env_setup import setup_and_prime
from nav.harness.observations import get_obs_safe
from nav.baselines.rl.pointgoal_ppo import (
    PPO_ACTIONS,
    compute_pointgoal_reward,
    depth_observation_uint8,
    encode_rl_goal,
)
from nav.utils import action2signal, decode_depth_observation_meters


class UnityPointGoalEnv:
    """Cycle a task shard through one Unity player and compute RL rewards."""

    def __init__(
        self,
        tasks: list[dict],
        *,
        unity_path: str,
        output_dir: str | Path,
        worker_id: int,
        base_port: int,
        reach_m: float = 2.0,
        max_steps: int = 200,
        ego_width: int = 320,
        ego_height: int = 240,
        minimap_width: int = 862,
        minimap_height: int = 512,
        dynamic_objects: str = "moving",
        goal_distance_scale_m: float = 50.0,
        reward_kwargs: dict | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("UnityPointGoalEnv requires at least one task")
        self.tasks = list(tasks)
        self.unity_path = str(unity_path)
        self.output_dir = Path(output_dir)
        self.worker_id = int(worker_id)
        self.base_port = int(base_port)
        self.reach_m = float(reach_m)
        self.max_steps = int(max_steps)
        self.ego_width = int(ego_width)
        self.ego_height = int(ego_height)
        self.minimap_width = int(minimap_width)
        self.minimap_height = int(minimap_height)
        self.dynamic_objects = str(dynamic_objects)
        self.goal_distance_scale_m = float(goal_distance_scale_m)
        self.reward_kwargs = dict(reward_kwargs or {})
        self.logger = logger or logging.getLogger(__name__)
        self.warning_detector = WarningDetector()
        self.task_index = -1
        self.task: dict | None = None
        self.args: argparse.Namespace | None = None
        self.primed = None
        self.pose: tuple[float, float, float] | None = None
        self.target_world: tuple[float, float] | None = None
        self.depth_obs: np.ndarray | None = None
        self.distance_m = float("inf")
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_collisions = 0
        self.episode_warnings = 0

    @staticmethod
    def _task_seed(task: dict) -> int:
        return int(task.get("motion_random_seed", task.get("seed", 0)))

    def _args_for(self, task: dict) -> argparse.Namespace:
        frame_dir = self.output_dir / f"env{self.worker_id}"
        return argparse.Namespace(
            file_name=self.unity_path,
            frame_save_dir=str(frame_dir),
            worker_id=self.worker_id,
            base_port=self.base_port,
            screen_width=1724,
            screen_height=1024,
            ego_width=self.ego_width,
            ego_height=self.ego_height,
            minimap_width=self.minimap_width,
            minimap_height=self.minimap_height,
            quality_level=3,
            scene_id=int(task["scene_id"]),
            scene_name=str(task["scene_name"]),
            point_id=str(task["episode_id"]),
            seed_id=str(task.get("seed", "ppo")),
            init_world_x=float(task["init_world_x"]),
            init_world_z=float(task["init_world_z"]),
            init_curr_direction=float(task["init_direction"]),
            target_x=int(task["target_x"]),
            target_y=int(task["target_y"]),
            dynamic_objects=self.dynamic_objects,
            human_speed_mps=None,
            human_speed_min_mps=0.8,
            human_speed_max_mps=1.6,
            vehicle_speed_mps=None,
            vehicle_speed_min_mps=1.5,
            vehicle_speed_max_mps=3.5,
            robot_speed_mps=None,
            robot_speed_min_mps=1.0,
            robot_speed_max_mps=2.0,
            motion_random_seed=self._task_seed(task),
            global_light_intensity=None,
            light_intensity_multiplier=None,
            light_intensity_min=None,
            light_intensity_max=None,
            light_random_seed=0,
            light_fixed_exposure=9.0,
        )

    def _close_player(self) -> None:
        if self.primed is not None:
            try:
                self.primed.env.close()
            finally:
                self.primed = None

    def close(self) -> None:
        self._close_player()

    def _launch(self, task: dict) -> None:
        self._close_player()
        self.args = self._args_for(task)
        self.primed = setup_and_prime(self.args, self.logger)
        if self.primed.target_world is None:
            raise RuntimeError("Unity did not acknowledge the PointGoal target")
        self.target_world = tuple(map(float, self.primed.target_world))

    def _reset_same_scene(self, task: dict) -> None:
        assert self.primed is not None and self.args is not None
        args = self.args
        for name in (
            "scene_id", "scene_name", "point_id", "seed_id", "init_world_x",
            "init_world_z", "init_curr_direction", "target_x", "target_y",
        ):
            setattr(args, name, self._args_for(task).__dict__[name])
        params = self.primed.env_params
        params.set_float_parameter("spawn_x", float(args.init_world_x))
        params.set_float_parameter("spawn_y", 0.5)
        params.set_float_parameter("spawn_z", float(args.init_world_z))
        params.set_float_parameter("spawn_rot", float(args.init_curr_direction))
        target_xy = canonical_to_minimap_coords(
            (args.target_x, args.target_y),
            (self.minimap_width, self.minimap_height),
        )
        target_px, target_py = visual_to_unity_coords(
            self.primed.margin,
            target_xy[0],
            target_xy[1],
            map_size=(self.minimap_width, self.minimap_height),
        )
        params.set_float_parameter("target_px", float(target_px))
        params.set_float_parameter("target_py", float(target_py))
        self.primed.target_sc.last_target_world = None
        self.primed.env.reset()
        if self.primed.target_sc.last_target_world is None:
            raise RuntimeError("Unity did not acknowledge the reset target")
        self.target_world = tuple(map(float, self.primed.target_sc.last_target_world))

    def _read_observation(self) -> dict:
        assert self.primed is not None and self.target_world is not None
        decision_steps, _ = self.primed.env.get_steps(BEHAVIOR_NAME)
        if len(decision_steps) == 0:
            raise RuntimeError("Unity returned no decision step")
        vector = decision_steps.obs[3]
        if vector.shape[1] < 6:
            raise RuntimeError(f"Unity pose vector is too short: {vector.shape}")
        pos_x, pos_z, yaw = (
            float(vector[0, 0]),
            float(vector[0, 2]),
            float(vector[0, 4]),
        )
        depth = get_obs_safe(decision_steps, "depth")
        if depth is None:
            raise RuntimeError("Unity depth observation is unavailable")
        self.pose = (pos_x, pos_z, yaw)
        self.depth_obs = np.asarray(depth)
        self.distance_m = float(
            np.hypot(pos_x - self.target_world[0], pos_z - self.target_world[1])
        )
        return {
            "depth": depth_observation_uint8(self.depth_obs),
            "goal": encode_rl_goal(
                pos_x,
                pos_z,
                yaw,
                self.target_world[0],
                self.target_world[1],
                self.goal_distance_scale_m,
            ),
        }

    def reset(self) -> dict:
        self.task_index = (self.task_index + 1) % len(self.tasks)
        next_task = self.tasks[self.task_index]
        if self.task is None or (
            str(next_task["scene_name"]) != str(self.task["scene_name"])
        ):
            self._launch(next_task)
        else:
            self._reset_same_scene(next_task)
        self.task = next_task
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_collisions = 0
        self.episode_warnings = 0
        return self._read_observation()

    def step(self, action_index: int) -> tuple[dict, float, bool, dict]:
        if self.primed is None or self.pose is None or self.depth_obs is None:
            raise RuntimeError("Call reset() before step()")
        action_index = int(action_index)
        if not 0 <= action_index < len(PPO_ACTIONS):
            raise ValueError(f"Invalid PPO action index: {action_index}")
        action_name = PPO_ACTIONS[action_index]
        signal = action2signal(action_name, ACTION_SPACE_POINTGOAL)
        previous_pose = self.pose
        previous_distance = self.distance_m
        warning = False
        if action_name == "forward":
            depth_m = decode_depth_observation_meters(self.depth_obs)
            warning = self.warning_detector.detect(
                depth_m,
                move_command=ACTION_SPACE_POINTGOAL["forward"],
            )["warning"] == "yes"

        self.primed.env.set_actions(
            BEHAVIOR_NAME, ActionTuple(continuous=signal)
        )
        self.primed.env.step()
        self.episode_steps += 1
        observation = self._read_observation()
        actual_displacement = float(
            np.hypot(self.pose[0] - previous_pose[0], self.pose[1] - previous_pose[1])
        )
        expected_displacement = (
            ACTION_SPACE_POINTGOAL["forward"]
            * EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
        )
        collision = (
            action_name == "forward"
            and actual_displacement
            < expected_displacement * EVAL_COLLISION_MIN_FORWARD_RATIO
        )
        success = self.distance_m <= self.reach_m
        timeout = self.episode_steps >= self.max_steps and not success
        done = success or timeout
        reward = compute_pointgoal_reward(
            previous_distance,
            self.distance_m,
            success=success,
            timeout=timeout,
            collision=collision,
            warning=warning,
            **self.reward_kwargs,
        )
        self.episode_return += reward
        self.episode_collisions += int(collision)
        self.episode_warnings += int(warning)
        info = {
            "scene_name": str(self.task["scene_name"]),
            "episode_id": str(self.task["episode_id"]),
            "distance_m": self.distance_m,
            "success": success,
            "timeout": timeout,
            "collision": collision,
            "warning": warning,
            "actual_displacement_m": actual_displacement,
        }
        if done:
            info.update(
                episode_return=self.episode_return,
                episode_steps=self.episode_steps,
                episode_collisions=self.episode_collisions,
                episode_warnings=self.episode_warnings,
            )
            observation = self.reset()
        return observation, reward, done, info
