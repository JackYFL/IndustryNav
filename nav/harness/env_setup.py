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
from nav.harness.coordinates import (
    canonical_to_minimap_coords,
    find_exact_map_bounds,
    resolve_minimap_resolution,
    visual_to_unity_coords,
)
from nav.harness.observations import get_minimap_rgb_for_init, patch_observation_decoding
from nav.harness.lighting import configure_unity_lighting
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
    minimap_size: Tuple[int, int]


def _set_world_spawn_parameters(env_params, args) -> bool:
    """Send a world-coordinate spawn when one is available on ``args``."""
    init_world_x = getattr(args, "init_world_x", None)
    init_world_z = getattr(args, "init_world_z", None)
    if init_world_x is None or init_world_z is None:
        return False
    env_params.set_float_parameter("spawn_x", float(init_world_x))
    env_params.set_float_parameter("spawn_y", 0.5)
    env_params.set_float_parameter("spawn_z", float(init_world_z))
    env_params.set_float_parameter(
        "spawn_rot", float(getattr(args, "init_curr_direction", 0.0))
    )
    return True


def _resolve_dynamic_objects_mode(args) -> str:
    """Return the validated Unity environment-motion mode."""
    mode = str(getattr(args, "dynamic_objects", "moving")).strip().lower()
    if mode not in {"moving", "static"}:
        raise EnvSetupError(
            "dynamic_objects must be either 'moving' or 'static', "
            f"got {mode!r}."
        )
    return mode


def _launch_env(args, logger):
    engine = EngineConfigurationChannel()
    bounds_sc = BoundsSideChannel()
    target_sc = TargetSideChannel()
    env_params = EnvironmentParametersChannel()

    unity_log_path = os.path.abspath(os.path.join(args.frame_save_dir, "unity_log.txt"))
    os.makedirs(args.frame_save_dir, exist_ok=True)
    use_batchmode_default = "0" if platform.system() == "Linux" else "1"
    use_batchmode = os.environ.get("INDUSTRYNAV_UNITY_BATCHMODE", use_batchmode_default) == "1"
    ego_width = int(getattr(args, "ego_width", 512))
    ego_height = int(getattr(args, "ego_height", 512))
    if ego_width <= 0 or ego_height <= 0:
        raise EnvSetupError("Egocentric RGB/depth resolution must use positive width and height values.")
    try:
        minimap_width, minimap_height = resolve_minimap_resolution(
            getattr(args, "minimap_width", None),
            getattr(args, "minimap_height", None),
        )
    except ValueError as exc:
        raise EnvSetupError(str(exc)) from exc
    args.minimap_width = minimap_width
    args.minimap_height = minimap_height
    unity_args = [
        "-logFile", unity_log_path,
        "-screen-width", str(args.screen_width),
        "-screen-height", str(args.screen_height),
        "--ego-width", str(ego_width),
        "--ego-height", str(ego_height),
        "--minimap-width", str(minimap_width),
        "--minimap-height", str(minimap_height),
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
        f"screen={args.screen_width}x{args.screen_height}, "
        f"egocentric_rgb_depth={ego_width}x{ego_height}, "
        f"minimap={minimap_width}x{minimap_height}"
    )

    # Scene selection MUST happen before the first reset that produces obs we use.
    env_params.set_float_parameter("scene_id", float(args.scene_id))
    # Ensure the Unity player applies ML-Agents actions even when launched
    # windowed on Linux, where Application.isBatchMode is false.
    env_params.set_float_parameter("use_ai_control", 1.0)
    dynamic_objects = _resolve_dynamic_objects_mode(args)
    env_params.set_float_parameter(
        "dynamic_objects_enabled",
        1.0 if dynamic_objects == "moving" else 0.0,
    )
    env_params.set_float_parameter(
        "spline_speed_multiplier",
        float(getattr(args, "spline_speed_multiplier", 1.0)),
    )
    try:
        lighting = configure_unity_lighting(env_params, args, logger)
    except ValueError as exc:
        env.close()
        raise EnvSetupError(str(exc)) from exc
    if _set_world_spawn_parameters(env_params, args):
        logger.info("Seeded world spawn parameters before the initial Unity reset.")
    logger.info(
        f"Starting Unity ({args.file_name}) scene_id={args.scene_id} "
        f"dynamic_objects={dynamic_objects} lighting={lighting.mode}"
    )
    logger.info(f"Unity launch args: {unity_args}")
    env.reset()
    logger.info(f"Behaviors: {list(env.behavior_specs.keys())}")
    _ = env.behavior_specs[BEHAVIOR_NAME]
    logger.info(f"Connected to behavior: {BEHAVIOR_NAME}")
    return env, env_params, bounds_sc, target_sc


def _compute_margin(env, env_params, args, logger):
    minimap_sensor_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
    if minimap_sensor_rgb is None:
        raise EnvSetupError("Minimap unavailable; cannot compute coordinate margin.")
    sensor_h, sensor_w = minimap_sensor_rgb.shape[:2]
    expected_size = (int(args.minimap_width), int(args.minimap_height))
    if (sensor_w, sensor_h) != expected_size:
        raise EnvSetupError(
            "Unity minimap observation has unexpected resolution: "
            f"expected {expected_size[0]}x{expected_size[1]}, got {sensor_w}x{sensor_h}. "
            "Rebuild the Unity client with runtime minimap resolution support."
        )
    margin = find_exact_map_bounds(logger, minimap_sensor_rgb)
    if margin is None:
        raise EnvSetupError("Failed to detect minimap margin.")
    env_params.set_float_parameter("minimap_px_width", float(sensor_w))
    env_params.set_float_parameter("minimap_px_height", float(sensor_h))
    return margin


def _prime_spawn(env, env_params, target_sc, margin, args, logger) -> Optional[Tuple[float, float]]:
    # Two spawn paths:
    #   (a) World-coord (preferred; input_points.json start.{x,z} are world
    #       coords): send spawn_x/spawn_y/spawn_z.
    #   (b) Visual-pixel (legacy): convert visual pixel -> unity pixel and send
    #       spawn_px/spawn_py; the world coords come back via the side-channel ack.
    if args.init_world_x is not None and args.init_world_z is not None:
        _set_world_spawn_parameters(env_params, args)
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
        spawn_xy = canonical_to_minimap_coords(
            (args.init_curr_x, args.init_curr_y),
            (args.minimap_width, args.minimap_height),
        )
        spawn_upx, spawn_upy = visual_to_unity_coords(
            margin,
            spawn_xy[0],
            spawn_xy[1],
            map_size=(args.minimap_width, args.minimap_height),
        )
        env_params.set_float_parameter("spawn_px", float(spawn_upx))
        env_params.set_float_parameter("spawn_py", float(spawn_upy))
        env_params.set_float_parameter("spawn_wy", 0.5)
        env_params.set_float_parameter("spawn_rot", float(args.init_curr_direction))
        env.reset()
        if target_sc.last_spawn_world is not None:
            init_world = target_sc.last_spawn_world
            logger.info(
                f"Spawn canonical({args.init_curr_x},{args.init_curr_y}) -> "
                f"visual{spawn_xy} -> "
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


def _prime_target(
    env,
    env_params,
    target_sc,
    margin,
    target_xy,
    args,
    logger,
) -> Optional[Tuple[float, float]]:
    tgt_upx, tgt_upy = visual_to_unity_coords(
        margin,
        target_xy[0],
        target_xy[1],
        map_size=(args.minimap_width, args.minimap_height),
    )
    env_params.set_float_parameter("target_px", float(tgt_upx))
    env_params.set_float_parameter("target_py", float(tgt_upy))
    env.reset()
    if target_sc.last_target_world is not None:
        target_world = target_sc.last_target_world
        logger.info(
            f"Target canonical({args.target_x},{args.target_y}) -> "
            f"visual{target_xy} -> "
            f"unity{target_sc.last_target_pixel} -> world({target_world[0]:.2f},{target_world[1]:.2f})"
        )
        return target_world
    raise EnvSetupError(
        "Target world mapping was not received from Unity. This client cannot "
        "run the world-coordinate-only benchmark; rebuild it with the target "
        "side-channel mapping enabled."
    )


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
        margin = _compute_margin(env, env_params, args, logger)
        init_world = _prime_spawn(env, env_params, target_sc, margin, args, logger)
        target_xy = canonical_to_minimap_coords(
            (args.target_x, args.target_y),
            (args.minimap_width, args.minimap_height),
        )
        target_world = _prime_target(
            env,
            env_params,
            target_sc,
            margin,
            target_xy,
            args,
            logger,
        )
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
        target_xy=target_xy,
        minimap_size=(int(args.minimap_width), int(args.minimap_height)),
    )
