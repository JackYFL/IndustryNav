"""Persistent Unity environment wrapper for Python-side PointGoal RL."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from mlagents_envs.exception import UnityException
from mlagents_envs.base_env import ActionTuple

from nav.baselines.astar import AStarBaseline
from nav.baselines.rl.reward import (
    collision_started,
    is_turn_reversal,
    should_trigger_stuck_recovery,
)
from nav.baselines.rl.tasks import RandomSceneBlockTaskSampler
from nav.config import ACTION_SPACE_POINTGOAL, BEHAVIOR_NAME
from nav.core.geometry import remaining_path_length_m
from nav.core.pointgoal import encode_rl_goal, encode_rl_goal_state
from nav.data.preprocessing import depth_observation_float32, depth_observation_uint8
from nav.envs.base import NavigationEnvironment
from nav.harness.coordinates import (
    build_axis_aligned_projector,
    canonical_to_minimap_coords,
    minimap_pixel_scale,
    visual_to_world_coords,
    visual_to_unity_coords,
    world_to_visual_coords,
)
from nav.harness.env_setup import EnvSetupError, setup_and_prime
from nav.harness.unity_lifecycle import configured_close_timeout, close_unity_environment
from nav.harness.observations import get_obs_safe
from nav.safety import CollisionDetector, WarningDetector
from nav.safety.pointgoal_protocol import PointGoalSafetyProtocol
from nav.utils import (
    action2signal,
    decode_depth_observation_meters,
    obs_to_rgb,
    unity_rotation_to_egocentric_theta,
)


class InvalidPointGoalSpawn(RuntimeError):
    """A moving obstacle displaced a sampled spawn beyond its safety margin."""


class UnityPointGoalEnv(NavigationEnvironment):
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
        absolute_scene_state: bool = False,
        num_scenes: int = 24,
        world_coordinate_scale_m: float = 50.0,
        coordinate_fourier_bands: int = 0,
        online_astar_teacher: bool = False,
        record_visuals: bool = False,
        astar_obstacle_clearance_m: float = 0.6,
        astar_policy_forward_tolerance_deg: float = 12.5,
        geodesic_progress_reward: bool = False,
        navigation_map_dir: str | Path | None = None,
        resample_invalid_spawns: bool = False,
        stagnation_progress_epsilon_m: float = 0.05,
        task_sampling: str = "random",
        task_sampling_seed: int = 0,
        scene_block_episodes: int = 4,
        stuck_recovery_steps: int = 48,
        dynamic_step_budget: bool = False,
        step_budget_min: int = 80,
        step_budget_max: int = 320,
        steps_per_path_meter: float = 2.2,
        step_budget_overhead: int = 60,
        auto_reset: bool = True,
        reuse_same_scene: bool = True,
        preserve_depth_precision: bool = False,
        training_safety_shield: bool = False,
        action_names: Sequence[str] = ("forward", "turn right", "turn left"),
        reward_fn: Callable[..., float] | None = None,
        reward_kwargs: dict | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("UnityPointGoalEnv requires at least one task")
        self.tasks = list(tasks)
        self.unity_path = str(unity_path)
        self.output_dir = Path(output_dir)
        self.worker_id = int(worker_id)
        # A Unity process that crashes during ML-Agents construction can leave
        # its Python-side RPC socket registered until interpreter shutdown.
        # Retrying on a disjoint worker id avoids reusing that poisoned port.
        self._unity_worker_generation = 0
        self.base_port = int(base_port)
        self.reach_m = float(reach_m)
        self.max_steps = int(max_steps)
        self.ego_width = int(ego_width)
        self.ego_height = int(ego_height)
        self.minimap_width = int(minimap_width)
        self.minimap_height = int(minimap_height)
        self.dynamic_objects = str(dynamic_objects)
        self.goal_distance_scale_m = float(goal_distance_scale_m)
        self.absolute_scene_state = bool(absolute_scene_state)
        self.num_scenes = int(num_scenes)
        self.world_coordinate_scale_m = float(world_coordinate_scale_m)
        self.coordinate_fourier_bands = int(coordinate_fourier_bands)
        if self.coordinate_fourier_bands < 0:
            raise ValueError("coordinate_fourier_bands must be nonnegative")
        if self.coordinate_fourier_bands and not self.absolute_scene_state:
            raise ValueError(
                "coordinate_fourier_bands requires absolute_scene_state"
            )
        self.online_astar_teacher = bool(online_astar_teacher)
        self.record_visuals = bool(record_visuals)
        self.astar_obstacle_clearance_m = float(astar_obstacle_clearance_m)
        if self.astar_obstacle_clearance_m < 0.0:
            raise ValueError("astar_obstacle_clearance_m must be nonnegative")
        self.astar_policy_forward_tolerance_deg = float(
            astar_policy_forward_tolerance_deg
        )
        if not 0.0 <= self.astar_policy_forward_tolerance_deg <= 90.0:
            raise ValueError(
                "astar_policy_forward_tolerance_deg must be in [0, 90]"
            )
        self.geodesic_progress_reward = bool(geodesic_progress_reward)
        self.navigation_map_dir = Path(navigation_map_dir) if navigation_map_dir else None
        if self.navigation_map_dir is not None and self.geodesic_progress_reward:
            raise ValueError("Static map rewards and online teacher-path rewards are exclusive")
        self._navigation_maps = {}
        self.navigation_map = None
        self.navigation_distance_field = None
        self.map_distance_m = None
        self.best_map_distance_m = None
        self.episode_map_valid_steps = 0
        self.resample_invalid_spawns = bool(resample_invalid_spawns)
        self.invalid_spawn_resets = 0
        if self.geodesic_progress_reward and not self.online_astar_teacher:
            raise ValueError(
                "geodesic_progress_reward requires online_astar_teacher"
            )
        self.stagnation_progress_epsilon_m = float(
            stagnation_progress_epsilon_m
        )
        if self.stagnation_progress_epsilon_m < 0.0:
            raise ValueError("stagnation_progress_epsilon_m must be nonnegative")
        if task_sampling not in {"random", "sequential"}:
            raise ValueError("task_sampling must be 'random' or 'sequential'")
        if scene_block_episodes <= 0:
            raise ValueError("scene_block_episodes must be positive")
        if stuck_recovery_steps < 0:
            raise ValueError("stuck_recovery_steps must be nonnegative")
        self.task_sampling = task_sampling
        self.task_sampler = (
            RandomSceneBlockTaskSampler(
                self.tasks,
                seed=task_sampling_seed,
                scene_block_episodes=scene_block_episodes,
            )
            if task_sampling == "random"
            else None
        )
        self.stuck_recovery_steps = int(stuck_recovery_steps)
        if step_budget_min <= 0 or step_budget_max < step_budget_min:
            raise ValueError("invalid dynamic step-budget bounds")
        if steps_per_path_meter <= 0.0 or step_budget_overhead < 0:
            raise ValueError("invalid dynamic step-budget coefficients")
        self.dynamic_step_budget = bool(dynamic_step_budget)
        self.step_budget_min = int(step_budget_min)
        self.step_budget_max = int(step_budget_max)
        self.steps_per_path_meter = float(steps_per_path_meter)
        self.step_budget_overhead = int(step_budget_overhead)
        self.auto_reset = bool(auto_reset)
        self.reuse_same_scene = bool(reuse_same_scene)
        self.preserve_depth_precision = bool(preserve_depth_precision)
        self.episode_max_steps = self.max_steps
        self.action_names = tuple(action_names)
        if not self.action_names:
            raise ValueError("action_names must not be empty")
        if training_safety_shield and set(self.action_names) != {
            "forward", "turn right", "turn left",
        }:
            raise ValueError("training safety shield requires navigation-only actions")
        self.training_shield = PointGoalSafetyProtocol() if training_safety_shield else None
        if reward_fn is None:
            raise ValueError("reward_fn is required")
        self.reward_fn = reward_fn
        self.reward_kwargs = dict(reward_kwargs or {})
        self.logger = logger or logging.getLogger(__name__)
        # Parse before launching a player, not only after a long run finishes.
        self.close_timeout_seconds = configured_close_timeout()
        if self.close_timeout_seconds is not None:
            self.logger.info("Unity exit timeout override: %.3fs (connection timeout unchanged)",
                             self.close_timeout_seconds)
        self.warning_detector = WarningDetector()
        self.collision_detector = CollisionDetector()
        self.task_index = -1
        self.task: dict | None = None
        self.args: argparse.Namespace | None = None
        self.primed = None
        self.pose: tuple[float, float, float] | None = None
        self.target_world: tuple[float, float] | None = None
        self.depth_obs: np.ndarray | None = None
        self.ego_obs: np.ndarray | None = None
        self.minimap_rgb: np.ndarray | None = None
        self.astar_teacher: AStarBaseline | None = None
        self.minimap_projector: dict | None = None
        self.astar_path: list[tuple[int, int]] = []
        self.astar_remaining_distance_m: float | None = None
        self.distance_m = float("inf")
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_collisions = 0
        self.episode_collision_events = 0
        self.episode_warnings = 0
        self.episode_forward_steps = 0
        self.episode_rotation_steps = 0
        self.episode_turn_reversals = 0
        self.episode_stagnant_steps = 0
        self.previous_action_name: str | None = None
        self.best_distance_m = float("inf")
        self.steps_without_progress = 0
        self.collision_active = False
        self.episode_initial_distance_m = float("inf")

    @staticmethod
    def _task_seed(task: dict) -> int:
        return int(task.get("motion_random_seed", task.get("seed", 0)))

    def _args_for(self, task: dict) -> argparse.Namespace:
        frame_dir = self.output_dir / f"env{self.worker_id}"
        return argparse.Namespace(
            file_name=self.unity_path,
            frame_save_dir=str(frame_dir),
            worker_id=self.worker_id + 64 * self._unity_worker_generation,
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
                close_unity_environment(self.primed.env, timeout_seconds=self.close_timeout_seconds,
                                        logger=self.logger)
            finally:
                self.primed = None

    def close(self) -> None:
        self._close_player()

    def _reset_online_teacher(self) -> None:
        """Reset the privileged A* labeler at an episode boundary.

        The planner is training-only: its minimap never enters the policy
        observation or the exported checkpoint.
        """
        if not self.online_astar_teacher and not self.record_visuals:
            self.astar_teacher = None
            self.minimap_projector = None
            self.astar_path = []
            self.astar_remaining_distance_m = None
            return
        assert self.primed is not None
        target_sc = self.primed.target_sc
        self.minimap_projector = build_axis_aligned_projector(
            target_sc.last_spawn_pixel,
            target_sc.last_spawn_world,
            target_sc.last_target_pixel,
            target_sc.last_target_world,
        )
        if self.minimap_projector is None:
            raise RuntimeError("A* teacher requires Unity minimap calibration")
        if not self.online_astar_teacher:
            self.astar_teacher = None
            self.astar_path = []
            self.astar_remaining_distance_m = None
            return
        self.astar_teacher = AStarBaseline(
            obstacle_clearance_m=self.astar_obstacle_clearance_m,
            pixel_scale=minimap_pixel_scale(self.primed.minimap_size),
            minimap_has_baked_markers=True,
            policy_actions=True,
            policy_forward_tolerance_deg=(
                self.astar_policy_forward_tolerance_deg
            ),
        )
        self.astar_path = []
        self.astar_remaining_distance_m = None

    def _astar_path_distance(self) -> float | None:
        if (
            not self.astar_path
            or self.primed is None
            or self.pose is None
            or self.minimap_projector is None
        ):
            return None

        def point_to_world(point):
            return visual_to_world_coords(
                self.primed.margin,
                point,
                self.minimap_projector,
                map_size=self.primed.minimap_size,
            )

        return remaining_path_length_m(
            self.astar_path,
            self.pose[:2],
            point_to_world,
        )

    def predict_astar_teacher_action(self) -> str:
        """Return an online A* expert label for the current policy state."""
        if not self.online_astar_teacher or self.astar_teacher is None:
            raise RuntimeError("Online A* teacher is not enabled")
        if (
            self.primed is None
            or self.pose is None
            or self.target_world is None
            or self.minimap_rgb is None
            or self.task is None
        ):
            raise RuntimeError("A* teacher requires a current Unity observation")
        map_size = self.primed.minimap_size
        current_xy = world_to_visual_coords(
            self.primed.margin,
            self.pose[0],
            self.pose[1],
            img_shape=self.minimap_rgb.shape,
            projector=self.minimap_projector,
            map_size=map_size,
        )
        target_xy = canonical_to_minimap_coords(
            (float(self.task["target_x"]), float(self.task["target_y"])),
            map_size,
        )

        def point_to_world(point):
            return visual_to_world_coords(
                self.primed.margin,
                point,
                self.minimap_projector,
                map_size=map_size,
            )

        action, _reasoning, path = self.astar_teacher.decide(
            minimap_rgb=self.minimap_rgb,
            curr_xy=current_xy,
            target_xy=target_xy,
            agent_theta=unity_rotation_to_egocentric_theta(self.pose[2]),
            reach_m=self.reach_m,
            step=self.episode_steps,
            curr_world_xz=self.pose[:2],
            target_world_xz=self.target_world,
            point_to_world=point_to_world,
        )
        self.astar_path = list(path)
        self.astar_remaining_distance_m = self._astar_path_distance()
        return action

    def _launch(self, task: dict) -> None:
        self._close_player()
        self.args = self._args_for(task)
        try:
            launch_attempts = int(
                os.environ.get("INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS", "3")
            )
        except ValueError as exc:
            raise ValueError(
                "INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS must be a positive integer"
            ) from exc
        if launch_attempts <= 0:
            raise ValueError(
                "INDUSTRYNAV_UNITY_LAUNCH_ATTEMPTS must be a positive integer"
            )
        self.primed = None
        for attempt in range(1, launch_attempts + 1):
            try:
                self.primed = setup_and_prime(self.args, self.logger)
                break
            except (EnvSetupError, UnityException) as exc:
                if isinstance(exc, UnityException):
                    self._unity_worker_generation += 1
                    self.args.worker_id = (
                        self.worker_id + 64 * self._unity_worker_generation
                    )
                if attempt == launch_attempts:
                    raise EnvSetupError(
                        "Unity launch/priming failed after "
                        f"{launch_attempts} attempt(s): {exc}"
                    ) from exc
                self.logger.warning(
                    "Unity initialization failed for %s/%s (attempt %d/%d); "
                    "relaunching the worker",
                    task["scene_name"],
                    task["episode_id"],
                    attempt,
                    launch_attempts,
                    exc_info=True,
                )
        assert self.primed is not None
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
        if self.record_visuals:
            ego = get_obs_safe(decision_steps, "ego")
            if ego is None:
                raise RuntimeError("Visual recording requires the ego observation")
            self.ego_obs = np.asarray(ego)
        else:
            self.ego_obs = None
        if self.online_astar_teacher or self.record_visuals:
            minimap = get_obs_safe(decision_steps, "minimap")
            if minimap is None:
                raise RuntimeError(
                    "A* teaching or visual recording requires the minimap observation"
                )
            self.minimap_rgb = obs_to_rgb(minimap)
        else:
            self.minimap_rgb = None
        self.distance_m = float(
            np.hypot(pos_x - self.target_world[0], pos_z - self.target_world[1])
        )
        return {
            "depth": (
                depth_observation_float32(self.depth_obs)
                if self.preserve_depth_precision
                else depth_observation_uint8(self.depth_obs)
            ),
            "goal": encode_rl_goal_state(
                pos_x,
                pos_z,
                yaw,
                self.target_world[0],
                self.target_world[1],
                distance_scale_m=self.goal_distance_scale_m,
                scene_id=int(self.task["scene_id"]),
                num_scenes=self.num_scenes,
                world_coordinate_scale_m=self.world_coordinate_scale_m,
                include_absolute_scene_state=self.absolute_scene_state,
                coordinate_fourier_bands=self.coordinate_fourier_bands,
            ),
        }

    def reset(self) -> dict:
        attempts = 4 if self.resample_invalid_spawns else 1
        for attempt in range(attempts):
            try:
                return self._reset_once()
            except InvalidPointGoalSpawn as exc:
                self.invalid_spawn_resets += 1
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with (self.output_dir / f"invalid_spawns_env{self.worker_id}.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(task=self.task, actual_pose=self.pose, error=str(exc)))+"\n")
                if attempt == attempts - 1:
                    raise
                self.logger.warning("Rejected sampled spawn; drawing another training task: %s", exc)

    def _reset_once(self) -> dict:
        try:
            task_attempts = int(
                os.environ.get("INDUSTRYNAV_TASK_RESET_ATTEMPTS", "4")
            )
        except ValueError as exc:
            raise ValueError(
                "INDUSTRYNAV_TASK_RESET_ATTEMPTS must be a positive integer"
            ) from exc
        if task_attempts <= 0:
            raise ValueError(
                "INDUSTRYNAV_TASK_RESET_ATTEMPTS must be a positive integer"
            )

        for task_attempt in range(1, task_attempts + 1):
            if self.task_sampler is None:
                self.task_index = (self.task_index + 1) % len(self.tasks)
                next_task = self.tasks[self.task_index]
            else:
                next_task = self.task_sampler.next_task()
            try:
                if (
                    self.primed is None
                    or self.task is None
                    or not self.reuse_same_scene
                    or str(next_task["scene_name"])
                    != str(self.task["scene_name"])
                ):
                    self._launch(next_task)
                else:
                    self._reset_same_scene(next_task)
                break
            except EnvSetupError:
                if task_attempt == task_attempts:
                    raise
                if self.task_sampler is not None:
                    self.task_sampler.skip_current_scene()
                self.logger.warning(
                    "Skipping failed Unity task %s/%s (task attempt %d/%d)",
                    next_task["scene_name"],
                    next_task["episode_id"],
                    task_attempt,
                    task_attempts,
                    exc_info=True,
                )
        self.task = next_task
        if self.training_shield is not None:
            self.training_shield.reset()
        self._reset_online_teacher()
        self.episode_steps = 0
        self.episode_return = 0.0
        self.episode_collisions = 0
        self.episode_collision_events = 0
        self.episode_warnings = 0
        self.episode_forward_steps = 0
        self.episode_rotation_steps = 0
        self.episode_turn_reversals = 0
        self.episode_stagnant_steps = 0
        self.previous_action_name = None
        self.steps_without_progress = 0
        self.collision_active = False
        observation = self._read_observation()
        self._reset_map_reward()
        self.best_distance_m = self.distance_m
        self.episode_initial_distance_m = self.distance_m
        if self.dynamic_step_budget:
            self.episode_max_steps = int(np.clip(
                np.ceil(
                    self.distance_m * self.steps_per_path_meter
                    + self.step_budget_overhead
                ),
                self.step_budget_min,
                self.step_budget_max,
            ))
        else:
            self.episode_max_steps = self.max_steps
        if self.dynamic_step_budget:
            self.logger.info(
                "Dynamic episode budget | scene=%s task=%s distance_m=%.3f max_steps=%d",
                self.task["scene_name"], self.task["episode_id"],
                self.distance_m, self.episode_max_steps,
            )
        return observation

    def step(self, action_index: int) -> tuple[dict, float, bool, dict]:
        if self.primed is None or self.pose is None or self.depth_obs is None:
            raise RuntimeError("Call reset() before step()")
        action_index = int(action_index)
        if not 0 <= action_index < len(self.action_names):
            raise ValueError(f"Invalid action index: {action_index}")
        action_name = self.action_names[action_index]
        proposed_action_index = action_index
        shield_intervened = False
        if self.training_shield is not None:
            assert self.target_world is not None
            bearing_sin = encode_rl_goal(*self.pose, *self.target_world)[1]
            action_name, shield_intervened = self.training_shield.filter_action(
                action_name,
                depth_m=decode_depth_observation_meters(self.depth_obs),
                goal_bearing_sin=float(bearing_sin),
            )
            action_index = self.action_names.index(action_name)
        signal = action2signal(action_name, ACTION_SPACE_POINTGOAL)
        previous_pose = self.pose
        previous_distance = self.distance_m
        previous_path_distance = (
            self.map_distance_m if self.navigation_map is not None
            else self.astar_remaining_distance_m
        )
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
        reversed_turn = is_turn_reversal(self.previous_action_name, action_name)
        actual_displacement = float(
            np.hypot(
                self.pose[0] - previous_pose[0],
                self.pose[1] - previous_pose[1],
            )
        )
        collision_assessment = self.collision_detector.detect(
            ACTION_SPACE_POINTGOAL["forward"] if action_name == "forward" else 0.0,
            previous_pose[:2],
            self.pose[:2],
        )
        collision = collision_assessment.triggered
        if self.training_shield is not None:
            self.training_shield.observe_step(
                collision=collision,
                depth_m=decode_depth_observation_meters(self.depth_obs),
                goal_bearing_sin=float(observation["goal"][1]),
            )
        collision_onset = collision_started(collision, self.collision_active)
        self.collision_active = collision
        current_path_distance = (
            self.navigation_map.distance_at(self.pose[:2], self.navigation_distance_field)
            if self.navigation_map is not None
            else self._astar_path_distance() if self.geodesic_progress_reward
            else None
        )
        path_progress_m = (
            previous_path_distance - current_path_distance
            if previous_path_distance is not None
            and current_path_distance is not None
            else None
        )
        map_progress_valid = previous_path_distance is not None and current_path_distance is not None
        if self.navigation_map is not None:
            self.map_distance_m = current_path_distance
            self.episode_map_valid_steps += int(map_progress_valid)
            # Missing map coverage must not silently become Euclidean reward.
            if path_progress_m is None:
                path_progress_m = 0.0
            if not map_progress_valid and self.episode_steps <= 100:
                self.logger.warning(
                    "Map coverage missing | scene=%s task=%s step=%d pose=%s cell=%s goal=%s collision=%s",
                    self.task["scene_name"], self.task["episode_id"], self.episode_steps,
                    self.pose, self.navigation_map.world_to_cell(self.pose[:2]).tolist(),
                    self.target_world, collision,
                )
        elif current_path_distance is not None:
            self.astar_remaining_distance_m = current_path_distance
        if self.navigation_map is not None:
            made_progress = bool(
                current_path_distance is not None and (
                    self.best_map_distance_m is None
                    or current_path_distance < self.best_map_distance_m - self.stagnation_progress_epsilon_m
                )
            )
            if made_progress:
                self.best_map_distance_m = current_path_distance
        elif self.geodesic_progress_reward and path_progress_m is not None:
            made_progress = (
                path_progress_m > self.stagnation_progress_epsilon_m
            )
        else:
            made_progress = self.distance_m < (
                self.best_distance_m - self.stagnation_progress_epsilon_m
            )
        if made_progress:
            self.best_distance_m = min(self.best_distance_m, self.distance_m)
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        success = self.distance_m <= self.reach_m
        timeout = self.episode_steps >= self.episode_max_steps and not success
        stuck_recovery = should_trigger_stuck_recovery(
            self.steps_without_progress,
            self.stuck_recovery_steps,
            success=success,
            timeout=timeout,
        )
        done = success or timeout or stuck_recovery
        reward = self.reward_fn(
            previous_distance,
            self.distance_m,
            success=success,
            timeout=timeout,
            collision=collision,
            collision_onset=collision_onset,
            warning=warning,
            path_progress_m=path_progress_m,
            action_name=action_name,
            turn_reversal=reversed_turn,
            stagnation_steps=self.steps_without_progress,
            stuck_recovery=stuck_recovery,
            **self.reward_kwargs,
        )
        self.episode_return += reward
        self.episode_collisions += int(collision)
        self.episode_collision_events += int(collision_onset)
        self.episode_warnings += int(warning)
        self.episode_forward_steps += int(action_name == "forward")
        self.episode_rotation_steps += int(action_name != "forward")
        self.episode_turn_reversals += int(reversed_turn)
        stagnant = self.steps_without_progress >= int(
            self.reward_kwargs.get("stagnation_start_steps", 12)
        )
        self.episode_stagnant_steps += int(stagnant)
        self.previous_action_name = action_name
        if self.astar_teacher is not None:
            # ``decide`` records the proposed expert action.  Replace it with
            # the action actually sampled/mixed into the Unity rollout.
            self.astar_teacher.observe_executed_action(action_name)
        info = {
            "scene_name": str(self.task["scene_name"]),
            "episode_id": str(self.task["episode_id"]),
            "distance_m": self.distance_m,
            "success": success,
            "timeout": timeout,
            "collision": collision,
            "collision_onset": collision_onset,
            "warning": warning,
            "actual_displacement_m": actual_displacement,
            "path_progress_m": path_progress_m,
            "map_progress_valid": map_progress_valid if self.navigation_map is not None else None,
            "map_distance_m": self.map_distance_m,
            "turn_reversal": reversed_turn,
            "stagnation_steps": self.steps_without_progress,
            "stuck_recovery": stuck_recovery,
            "episode_max_steps": self.episode_max_steps,
            "proposed_action_index": proposed_action_index,
            "executed_action_index": action_index,
            "shield_intervened": shield_intervened,
            "episode_shield_interventions": (
                self.training_shield.interventions if self.training_shield is not None else 0
            ),
        }
        if done:
            info.update(
                episode_return=self.episode_return,
                episode_steps=self.episode_steps,
                episode_initial_distance_m=self.episode_initial_distance_m,
                episode_collisions=self.episode_collisions,
                episode_collision_events=self.episode_collision_events,
                episode_warnings=self.episode_warnings,
                episode_forward_steps=self.episode_forward_steps,
                episode_rotation_steps=self.episode_rotation_steps,
                episode_turn_reversals=self.episode_turn_reversals,
                episode_stagnant_steps=self.episode_stagnant_steps,
                episode_stuck_recovery=int(stuck_recovery),
                episode_map_valid_steps=self.episode_map_valid_steps,
            )
            if self.auto_reset:
                observation = self.reset()
        return observation, reward, done, info

    def _reset_map_reward(self):
        """Load a train-only fixed occupancy field, without creating a teacher."""
        self.navigation_map = None
        self.navigation_distance_field = None
        self.map_distance_m = self.best_map_distance_m = None
        self.episode_map_valid_steps = 0
        if self.navigation_map_dir is None:
            return
        from nav.data.navigation_map import NavigationMap, NavigationMapCoverageError

        scene = self.task["scene_name"]
        if scene not in self._navigation_maps:
            geometry = NavigationMap.load(self.navigation_map_dir / scene)
            if geometry.metadata["scene_name"] != scene:
                raise ValueError("Navigation map scene does not match the task")
            self._navigation_maps[scene] = geometry
        self.navigation_map = self._navigation_maps[scene]
        expected_target = self.task.get("sampled_target_world_x"), self.task.get("sampled_target_world_z")
        if all(value is not None for value in expected_target):
            target_error = float(np.linalg.norm(np.asarray(expected_target) - self.target_world))
            if target_error > 0.5:
                raise RuntimeError(f"Sampled target calibration disagrees with Unity by {target_error:.3f} m")
        start_error = float(np.linalg.norm(np.asarray(self.pose[:2]) - (
            self.task["init_world_x"], self.task["init_world_z"],
        )))
        if start_error > 0.5:
            raise InvalidPointGoalSpawn(f"Sampled spawn disagrees with Unity by {start_error:.3f} m")
        try:
            self.navigation_distance_field = self.navigation_map.distance_field(
                self.target_world, reach_m=max(0.1, self.reach_m - 0.4),
            )
        except NavigationMapCoverageError:
            # Image-derived occupancy is not physics ground truth. Retain this
            # episode with sparse success/safety rewards, not a filtered task
            # or a fabricated Euclidean/geodesic progress signal.
            self.navigation_distance_field = np.full(
                self.navigation_map.walkable.shape, np.inf, dtype=np.float64,
            )
        self.map_distance_m = self.navigation_map.distance_at(self.pose[:2], self.navigation_distance_field)
        if self.map_distance_m is None:
            self.logger.warning(
                "Map coverage missing at reset; retaining physical task | scene=%s task=%s pose=%s goal=%s",
                scene, self.task["episode_id"], self.pose, self.target_world,
            )
        self.best_map_distance_m = self.map_distance_m
