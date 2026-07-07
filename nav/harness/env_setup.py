"""Launch the unified Unity client and prime spawn + target via side channels.

Encapsulates everything the benchmark entry does *before* its decision loop:
construct the ``UnityEnvironment`` + side channels, select the scene, detect the
minimap coordinate margin, and run the spawn/target pixel→world handshake. The
entry script then only owns its decision loop.

This is the same priming pipeline the data-collection entry uses, so it lives in
the harness rather than in any one script.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Optional, Tuple

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from nav.config import BEHAVIOR_NAME, UNITY_ENGINE_QUALITY_LEVEL
from nav.harness.coordinates import find_exact_map_bounds, visual_to_unity_coords
from nav.harness.observations import get_minimap_rgb_for_init, patch_observation_decoding
from nav.harness.side_channels import BoundsSideChannel, TargetSideChannel


class EnvSetupError(RuntimeError):
    """Env couldn't be launched, or spawn/target priming failed unrecoverably."""


@dataclass
class PrimedEnv:
    """A launched + primed Unity env and the coordinates the loop needs."""

    env: UnityEnvironment
    env_params: EnvironmentParametersChannel
    bounds_sc: BoundsSideChannel
    target_sc: TargetSideChannel
    margin: object
    init_world: Optional[Tuple[float, float]]
    target_world: Optional[Tuple[float, float]]
    target_xy: Tuple[int, int]


def _launch_env(args, logger):
    engine = EngineConfigurationChannel()
    bounds_sc = BoundsSideChannel()
    target_sc = TargetSideChannel()
    env_params = EnvironmentParametersChannel()

    unity_log_path = os.path.join(args.frame_save_dir, "unity_log.txt")
    os.makedirs(args.frame_save_dir, exist_ok=True)
    use_batchmode_default = "0" if platform.system() == "Linux" else "1"
    use_batchmode = os.environ.get("INDUSTRYNAV_UNITY_BATCHMODE", use_batchmode_default) == "1"
    unity_args = [
        "-logFile", unity_log_path,
        "-screen-width", str(args.screen_width),
        "-screen-height", str(args.screen_height),
    ]
    if use_batchmode:
        unity_args.insert(0, "-batchmode")

    # NOTE on "headless": passing `-nographics` SIGKILLs scene_all.app (camera
    # sensors need a renderer); `no_graphics=True` makes the window invisible but
    # blanks the sensor frames on macOS. macOS uses `-batchmode`; Linux defaults
    # to windowed because some Linux builds SIGSEGV during Input System init in
    # batchmode. Override with INDUSTRYNAV_UNITY_BATCHMODE=1/0.
    env = UnityEnvironment(
        file_name=args.file_name,
        no_graphics=False,
        side_channels=[engine, env_params, bounds_sc, target_sc],
        additional_args=unity_args,
        worker_id=int(args.worker_id),
        base_port=int(args.base_port),
        timeout_wait=120,
    )
    quality_level = int(getattr(args, "quality_level", UNITY_ENGINE_QUALITY_LEVEL))
    engine.set_configuration_parameters(
        time_scale=1,
        quality_level=quality_level,
        target_frame_rate=-1,
        width=args.screen_width,
        height=args.screen_height,
    )
    logger.info(
        f"Engine config: quality_level={quality_level}, "
        f"screen={args.screen_width}x{args.screen_height}"
    )

    # Scene selection MUST happen before the first reset that produces obs we use.
    env_params.set_float_parameter("scene_id", float(args.scene_id))
    # Ensure the Unity player applies ML-Agents actions even when launched
    # windowed on Linux, where Application.isBatchMode is false.
    env_params.set_float_parameter("use_ai_control", 1.0)
    logger.info(f"Starting Unity ({args.file_name}) scene_id={args.scene_id}")
    logger.info(f"Unity launch args: {unity_args}")
    env.reset()
    logger.info(f"Behaviors: {list(env.behavior_specs.keys())}")
    _ = env.behavior_specs[BEHAVIOR_NAME]
    logger.info(f"Connected to behavior: {BEHAVIOR_NAME}")
    return env, env_params, bounds_sc, target_sc


def _compute_margin(env, env_params, logger):
    minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
    if minimap_rgb is None:
        raise EnvSetupError("Minimap unavailable; cannot compute coordinate margin.")
    margin = find_exact_map_bounds(logger, minimap_rgb)
    if margin is None:
        raise EnvSetupError("Failed to detect minimap margin.")
    h, w = minimap_rgb.shape[:2]
    env_params.set_float_parameter("minimap_px_width", float(w))
    env_params.set_float_parameter("minimap_px_height", float(h))
    return margin


def _prime_spawn(env, env_params, target_sc, margin, args, logger) -> Optional[Tuple[float, float]]:
    # Two spawn paths:
    #   (a) World-coord (preferred; input_points.json start.{x,z} are world
    #       coords): send spawn_x/spawn_y/spawn_z.
    #   (b) Visual-pixel (legacy): convert visual pixel -> unity pixel and send
    #       spawn_px/spawn_py; the world coords come back via the side-channel ack.
    if args.init_world_x is not None and args.init_world_z is not None:
        env_params.set_float_parameter("spawn_x", float(args.init_world_x))
        env_params.set_float_parameter("spawn_y", 0.5)
        env_params.set_float_parameter("spawn_z", float(args.init_world_z))
        env_params.set_float_parameter("spawn_rot", float(args.init_curr_direction))
        env.reset()
        init_world = (float(args.init_world_x), float(args.init_world_z))
        if target_sc.last_spawn_pixel is not None and target_sc.last_spawn_world is not None:
            init_world = target_sc.last_spawn_world
            logger.info(
                f"Spawn world({init_world[0]:.2f},{init_world[1]:.2f}) -> "
                f"unity{target_sc.last_spawn_pixel} yaw={args.init_curr_direction}"
            )
        else:
            logger.info(
                f"Spawn world({init_world[0]:.2f},{init_world[1]:.2f}) "
                f"yaw={args.init_curr_direction}"
            )
        return init_world

    if args.init_curr_x is not None and args.init_curr_y is not None:
        spawn_upx, spawn_upy = visual_to_unity_coords(margin, args.init_curr_x, args.init_curr_y)
        env_params.set_float_parameter("spawn_px", float(spawn_upx))
        env_params.set_float_parameter("spawn_py", float(spawn_upy))
        env_params.set_float_parameter("spawn_wy", 0.5)
        env_params.set_float_parameter("spawn_rot", float(args.init_curr_direction))
        env.reset()
        if target_sc.last_spawn_world is not None:
            init_world = target_sc.last_spawn_world
            logger.info(
                f"Spawn visual({args.init_curr_x},{args.init_curr_y}) -> "
                f"unity{target_sc.last_spawn_pixel} -> world({init_world[0]:.2f},{init_world[1]:.2f}) "
                f"yaw={args.init_curr_direction}"
            )
            return init_world
        logger.warning("Spawn world ack not received; continuing with None world coords.")
        return None

    raise EnvSetupError(
        "No spawn coords given. Pass --init_world_x/--init_world_z (preferred) "
        "or --init_curr_x/--init_curr_y."
    )


def _prime_target(env, env_params, target_sc, margin, args, logger) -> Optional[Tuple[float, float]]:
    tgt_upx, tgt_upy = visual_to_unity_coords(margin, args.target_x, args.target_y)
    env_params.set_float_parameter("target_px", float(tgt_upx))
    env_params.set_float_parameter("target_py", float(tgt_upy))
    env.reset()
    if target_sc.last_target_world is not None:
        target_world = target_sc.last_target_world
        logger.info(
            f"Target visual({args.target_x},{args.target_y}) -> "
            f"unity{target_sc.last_target_pixel} -> world({target_world[0]:.2f},{target_world[1]:.2f})"
        )
        return target_world
    logger.warning("Target world ack not received; world distance will be unavailable.")
    return None


def setup_and_prime(args, logger) -> PrimedEnv:
    """Launch + scene-select + margin-detect + spawn/target prime.

    Raises :class:`EnvSetupError` (after closing the env) if the minimap margin
    or spawn coords are unusable. ``args`` is the benchmark entry's argparse
    namespace (the data collector's compatible flags work too).
    """
    # Tolerate the scene_all build's sensor-shape quirk before env launch.
    patch_observation_decoding()
    env, env_params, bounds_sc, target_sc = _launch_env(args, logger)
    try:
        margin = _compute_margin(env, env_params, logger)
        init_world = _prime_spawn(env, env_params, target_sc, margin, args, logger)
        target_world = _prime_target(env, env_params, target_sc, margin, args, logger)
    except EnvSetupError:
        env.close()
        raise

    return PrimedEnv(
        env=env,
        env_params=env_params,
        bounds_sc=bounds_sc,
        target_sc=target_sc,
        margin=margin,
        init_world=init_world,
        target_world=target_world,
        target_xy=(int(args.target_x), int(args.target_y)),
    )
