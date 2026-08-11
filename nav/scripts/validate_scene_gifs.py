"""Smoke-test Unity scenes and export RGB/depth/minimap GIFs.

Each scene is launched in a fresh Unity process because ``SceneSwitcher`` reads
``scene_id`` during scene startup. The validator uses either a fixed world pose
or the scene's unprimed random-spawn path, applies reproducible random agent
actions, and checks that all three camera observations remain non-blank.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from nav.config import (
    ACTION_SPACE_AGENTS,
    BEHAVIOR_NAME,
    SCENE_ID_MAP,
    UNITY_DEPTH_MAX_DISTANCE_M,
    UNITY_ENGINE_QUALITY_LEVEL,
)
from nav.harness.coordinates import (
    canonical_to_minimap_coords,
    find_exact_map_bounds,
    minimap_pixel_scale,
    visual_to_unity_coords,
)
from nav.harness.observations import (
    get_minimap_rgb_for_init,
    get_obs_safe,
    patch_observation_decoding,
)
from nav.harness.lighting import (
    add_lighting_args,
    configure_unity_lighting,
    resolve_lighting_config,
)
from nav.harness.motion import (
    add_motion_speed_args,
    configure_unity_motion_speed,
    resolve_motion_speed_config,
)
from nav.harness.side_channels import BoundsSideChannel, TargetSideChannel
from nav.scripts.run_benchmark_cell import (
    build_axis_aligned_projector,
    draw_curr_target_heading_rgb,
    remove_unity_red_marker_rgb,
    vector_marker_from_obs,
)
from nav.utils import (
    action2signal,
    decode_depth_observation_meters,
    depth_observation_to_visualization,
    obs_to_rgb,
)


ALL_SCENES = tuple(SCENE_ID_MAP.items())

RANDOM_ACTIONS = ("forward", "turn right", "turn left", "stop")
RANDOM_ACTION_PROBABILITIES = (0.45, 0.25, 0.25, 0.05)

INITIAL_EGO_MIN_STD = 5.0
INITIAL_EGO_MIN_EDGES = 500
INITIAL_DEPTH_MIN_STD = 0.5
INITIAL_MINIMAP_MIN_STD = 1.0
INITIAL_MINIMAP_MIN_EDGES = 20

TOP_PANEL_SIZE = (320, 240)
COMPOSITE_WIDTH = TOP_PANEL_SIZE[0] * 2
ROW_HEADER_HEIGHT = 30


@dataclass
class SceneResult:
    scene_name: str
    scene_id: int
    ok: bool
    frames: int
    attempts: int = 1
    ego_shape: list[int] | None = None
    depth_shape: list[int] | None = None
    minimap_shape: list[int] | None = None
    ego_std_min: float | None = None
    depth_std_min: float | None = None
    minimap_std_min: float | None = None
    ego_edges_min: int | None = None
    minimap_edges_min: int | None = None
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    target_xy: list[int] | None = None
    gif_path: str | None = None
    unity_log: str | None = None
    error: str | None = None
    light_randomization_mode: str = "disabled"
    light_intensity_multiplier: float = 1.0
    light_fixed_exposure: float | None = None
    human_speed_mps: float | None = None
    vehicle_speed_mps: float | None = None
    robot_speed_mps: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file-name",
        default="unity_clients/scene_all_24scenes_absolute_speed_v10.app",
        help="Path to the unified macOS Unity client.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/all_24_scene_rgb_depth_minimap_gifs",
        help="Directory for GIFs, PNG frames, Unity logs, and summary.json.",
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--gif-duration-ms", type=int, default=140)
    parser.add_argument("--random-seed", type=int, default=20260810)
    parser.add_argument("--ego-width", type=int, default=320)
    parser.add_argument("--ego-height", type=int, default=240)
    parser.add_argument("--minimap-width", type=int, default=862)
    parser.add_argument("--minimap-height", type=int, default=512)
    parser.add_argument(
        "--target-x",
        type=float,
        default=550.0,
        help="Target x in canonical 862x512 minimap pixels.",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=450.0,
        help="Target y in canonical 862x512 minimap pixels.",
    )
    parser.add_argument(
        "--init-world-x",
        type=float,
        default=None,
        help="Optional fixed Unity world X used for reproducible comparisons.",
    )
    parser.add_argument(
        "--init-world-z",
        type=float,
        default=None,
        help="Optional fixed Unity world Z used for reproducible comparisons.",
    )
    parser.add_argument(
        "--init-direction",
        type=float,
        default=180.0,
        help="Initial Unity yaw in degrees when fixed world coordinates are set.",
    )
    parser.add_argument(
        "--dynamic-objects",
        choices=("moving", "static"),
        default="moving",
        help="Run non-agent Unity objects normally or keep them frozen.",
    )
    add_motion_speed_args(parser)
    add_lighting_args(parser)
    parser.add_argument("--screen-width", type=int, default=1724)
    parser.add_argument("--screen-height", type=int, default=1024)
    parser.add_argument("--base-port", type=int, default=5507)
    parser.add_argument("--worker-id-start", type=int, default=30)
    parser.add_argument(
        "--max-attempts-per-scene",
        type=int,
        default=1,
        help="Retry random spawning when a run contains blank camera frames.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of scene codes from the built-in 24-scene list.",
    )
    return parser.parse_args()


def selected_scenes(names: list[str] | None) -> Iterable[tuple[str, int]]:
    if not names:
        return ALL_SCENES
    requested = set(names)
    known = {name for name, _ in ALL_SCENES}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown scene codes: {', '.join(sorted(unknown))}")
    return tuple(item for item in ALL_SCENES if item[0] in requested)


def image_stats(rgb: np.ndarray) -> tuple[float, int]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.std()), int((cv2.Canny(gray, 30, 100) > 0).sum())


def depth_to_rgb(depth_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed grayscale preview and the untouched metric depth."""
    depth_gray = depth_observation_to_visualization(depth_obs)
    depth_rgb = np.repeat(depth_gray[:, :, None], 3, axis=2)
    return depth_rgb, decode_depth_observation_meters(depth_obs)


def agent_pose_from_steps(decision_steps) -> tuple[float, float, float, float]:
    """Return agent local/world-compatible X, Y, Z and Unity yaw."""
    vector_obs = decision_steps.obs[-1]
    if vector_obs.ndim != 2 or vector_obs.shape[0] == 0 or vector_obs.shape[1] < 6:
        raise RuntimeError(f"Expected >=6 vector observations, got {vector_obs.shape}")
    return (
        float(vector_obs[0, 0]),
        float(vector_obs[0, 1]),
        float(vector_obs[0, 2]),
        float(vector_obs[0, 4]),
    )


def combined_frame(
    ego_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    minimap_rgb: np.ndarray,
    scene_name: str,
    scene_id: int,
    frame_index: int,
    agent_xy: tuple[int, int],
    agent_heading_xy: tuple[int, int],
    target_xy: tuple[int, int],
    agent_world: tuple[float, float],
    agent_yaw: float,
) -> Image.Image:
    ego = Image.fromarray(ego_rgb).resize(TOP_PANEL_SIZE, Image.Resampling.LANCZOS)
    depth = Image.fromarray(depth_rgb).resize(TOP_PANEL_SIZE, Image.Resampling.LANCZOS)
    pixel_scale = minimap_pixel_scale(
        (minimap_rgb.shape[1], minimap_rgb.shape[0])
    )
    clean_minimap = remove_unity_red_marker_rgb(
        minimap_rgb,
        pixel_scale=pixel_scale,
    )
    annotated_minimap = draw_curr_target_heading_rgb(
        clean_minimap,
        agent_xy,
        target_xy,
        curr_heading_xy=agent_heading_xy,
        pixel_scale=pixel_scale,
    )
    minimap_h = round(
        minimap_rgb.shape[0] * COMPOSITE_WIDTH / minimap_rgb.shape[1]
    )
    minimap = Image.fromarray(annotated_minimap).resize(
        (COMPOSITE_WIDTH, minimap_h), Image.Resampling.LANCZOS
    )
    top_h = ROW_HEADER_HEIGHT + TOP_PANEL_SIZE[1]
    canvas = Image.new(
        "RGB",
        (COMPOSITE_WIDTH, top_h + ROW_HEADER_HEIGHT + minimap_h),
        "black",
    )
    canvas.paste(ego, (0, ROW_HEADER_HEIGHT))
    canvas.paste(depth, (TOP_PANEL_SIZE[0], ROW_HEADER_HEIGHT))
    canvas.paste(minimap, (0, top_h + ROW_HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 9),
        f"RGB | {scene_name} | build index {scene_id} | step {frame_index:02d}",
        fill="white",
    )
    draw.text(
        (TOP_PANEL_SIZE[0] + 8, 9),
        f"DEPTH | 0-{UNITY_DEPTH_MAX_DISTANCE_M:g} m",
        fill="white",
    )
    draw.text(
        (8, top_h + 9),
        (
            f"MINIMAP | agent px={agent_xy} world=({agent_world[0]:.2f},"
            f"{agent_world[1]:.2f}) yaw={agent_yaw:.1f} deg | target px={target_xy}"
        ),
        fill="white",
    )
    return canvas


def write_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    quantized = [
        frame.quantize(
            colors=256,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def validate_scene(
    args: argparse.Namespace,
    scene_name: str,
    scene_id: int,
    worker_id: int,
) -> SceneResult:
    output_dir = Path(args.output_dir).resolve()
    scene_dir = output_dir / f"{scene_id:02d}_{scene_name}"
    frames_dir = scene_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.png"):
        old_frame.unlink()
    unity_log = scene_dir / "unity.log"
    gif_path = output_dir / f"{scene_id:02d}_{scene_name}_rgb_depth_minimap.gif"

    engine = EngineConfigurationChannel()
    env_params = EnvironmentParametersChannel()
    bounds_sc = BoundsSideChannel()
    target_sc = TargetSideChannel()
    env: UnityEnvironment | None = None

    result = SceneResult(
        scene_name=scene_name,
        scene_id=scene_id,
        ok=False,
        frames=0,
        unity_log=str(unity_log),
    )
    logger = logging.getLogger(f"{__name__}.{scene_name}")

    try:
        env = UnityEnvironment(
            file_name=str(Path(args.file_name).resolve()),
            no_graphics=False,
            side_channels=[engine, env_params, bounds_sc, target_sc],
            additional_args=[
                "-batchmode",
                "-logFile",
                str(unity_log),
                "-screen-width",
                str(args.screen_width),
                "-screen-height",
                str(args.screen_height),
                "--ego-width",
                str(args.ego_width),
                "--ego-height",
                str(args.ego_height),
                "--minimap-width",
                str(args.minimap_width),
                "--minimap-height",
                str(args.minimap_height),
            ],
            worker_id=worker_id,
            base_port=args.base_port,
            timeout_wait=180,
        )
        engine.set_configuration_parameters(
            time_scale=1,
            quality_level=UNITY_ENGINE_QUALITY_LEVEL,
            target_frame_rate=-1,
            width=args.screen_width,
            height=args.screen_height,
        )
        env_params.set_float_parameter("scene_id", float(scene_id))
        env_params.set_float_parameter("use_ai_control", 1.0)
        env_params.set_float_parameter(
            "dynamic_objects_enabled",
            1.0 if args.dynamic_objects == "moving" else 0.0,
        )
        motion_args = argparse.Namespace(
            scene_id=scene_id,
            scene_name=scene_name,
            point_id="scene_validation",
            seed_id=str(args.random_seed),
            motion_random_seed=args.motion_random_seed,
            **{
                f"{category}_speed_{suffix}": getattr(
                    args, f"{category}_speed_{suffix}"
                )
                for category in ("human", "vehicle", "robot")
                for suffix in ("mps", "min_mps", "max_mps")
            },
        )
        motion = configure_unity_motion_speed(env_params, motion_args, logger)
        result.human_speed_mps = motion.human.speed_mps
        result.vehicle_speed_mps = motion.vehicle.speed_mps
        result.robot_speed_mps = motion.robot.speed_mps
        lighting_args = argparse.Namespace(
            scene_id=scene_id,
            scene_name=scene_name,
            point_id="scene_validation",
            seed_id=str(args.random_seed),
            global_light_intensity=args.global_light_intensity,
            light_intensity_multiplier=args.light_intensity_multiplier,
            light_intensity_min=args.light_intensity_min,
            light_intensity_max=args.light_intensity_max,
            light_random_seed=args.light_random_seed,
            light_fixed_exposure=args.light_fixed_exposure,
        )
        lighting = configure_unity_lighting(env_params, lighting_args, logger)
        result.light_randomization_mode = lighting.mode
        result.light_intensity_multiplier = lighting.multiplier
        result.light_fixed_exposure = lighting.fixed_exposure
        env_params.set_float_parameter("minimap_px_width", float(args.minimap_width))
        env_params.set_float_parameter("minimap_px_height", float(args.minimap_height))
        if args.init_world_x is not None:
            env_params.set_float_parameter("spawn_x", float(args.init_world_x))
            env_params.set_float_parameter("spawn_y", 0.5)
            env_params.set_float_parameter("spawn_z", float(args.init_world_z))
            env_params.set_float_parameter("spawn_rot", float(args.init_direction))
        env.reset()

        if BEHAVIOR_NAME not in env.behavior_specs:
            raise RuntimeError(
                f"Missing behavior {BEHAVIOR_NAME}; got {list(env.behavior_specs)}"
            )

        # Preserve the scene's random/serialized initial pose, then reset once
        # with an explicit target so Unity returns exact spawn/target mappings.
        initial_minimap = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
        if initial_minimap is None:
            raise RuntimeError("Minimap did not render during target priming")
        margin = find_exact_map_bounds(logger, initial_minimap)
        if margin is None:
            raise RuntimeError("Could not detect minimap bounds during target priming")
        initial_steps, _ = env.get_steps(BEHAVIOR_NAME)
        if len(initial_steps) == 0:
            raise RuntimeError("Agent did not request a decision during target priming")
        spawn_x, spawn_y, spawn_z, spawn_yaw = agent_pose_from_steps(initial_steps)
        target_xy = canonical_to_minimap_coords(
            (args.target_x, args.target_y),
            (args.minimap_width, args.minimap_height),
        )
        target_upx, target_upy = visual_to_unity_coords(
            margin,
            target_xy[0],
            target_xy[1],
            map_size=(args.minimap_width, args.minimap_height),
        )
        env_params.set_float_parameter("spawn_x", spawn_x)
        env_params.set_float_parameter("spawn_y", spawn_y)
        env_params.set_float_parameter("spawn_z", spawn_z)
        env_params.set_float_parameter("spawn_rot", spawn_yaw)
        env_params.set_float_parameter("target_px", float(target_upx))
        env_params.set_float_parameter("target_py", float(target_upy))
        env.reset()
        minimap_projector = build_axis_aligned_projector(
            target_sc.last_spawn_pixel,
            target_sc.last_spawn_world,
            target_sc.last_target_pixel,
            target_sc.last_target_world,
        )
        if minimap_projector is None:
            raise RuntimeError("Unity did not acknowledge spawn/target mappings")
        result.target_xy = [int(target_xy[0]), int(target_xy[1])]

        gif_frames: list[Image.Image] = []
        ego_stds: list[float] = []
        depth_stds: list[float] = []
        minimap_stds: list[float] = []
        ego_edges: list[int] = []
        minimap_edges: list[int] = []
        depth_mins: list[float] = []
        depth_maxs: list[float] = []
        rng = np.random.default_rng(args.random_seed + scene_id)
        attempts = 0
        max_attempts = args.frames + 80

        while len(gif_frames) < args.frames and attempts < max_attempts:
            attempts += 1
            decision_steps, _ = env.get_steps(BEHAVIOR_NAME)
            if len(decision_steps) == 0:
                env.step()
                continue

            ego_obs = get_obs_safe(decision_steps, "ego")
            depth_obs = get_obs_safe(decision_steps, "depth")
            minimap_obs = get_obs_safe(decision_steps, "minimap")
            if ego_obs is None or depth_obs is None or minimap_obs is None:
                action = "stop"
            else:
                ego_rgb = obs_to_rgb(ego_obs)
                depth_rgb, depth_m = depth_to_rgb(depth_obs)
                minimap_rgb = obs_to_rgb(minimap_obs)
                pos_x, _, pos_z, rot_y = agent_pose_from_steps(decision_steps)
                curr_xy, curr_heading_xy = vector_marker_from_obs(
                    margin,
                    pos_x,
                    pos_z,
                    rot_y,
                    img_shape=minimap_rgb.shape,
                    projector=minimap_projector,
                    map_size=(args.minimap_width, args.minimap_height),
                    pixel_scale=minimap_pixel_scale(
                        (minimap_rgb.shape[1], minimap_rgb.shape[0])
                    ),
                )
                ego_std, ego_edge_count = image_stats(ego_rgb)
                depth_std = float(depth_rgb[:, :, 0].std())
                minimap_std, minimap_edge_count = image_stats(minimap_rgb)

                # Skip the renderer's initial blank frames, but retain all later
                # frames so intermittent sensor failures are reflected in metrics.
                if gif_frames or (
                    ego_std >= INITIAL_EGO_MIN_STD
                    and ego_edge_count >= INITIAL_EGO_MIN_EDGES
                    and depth_std >= INITIAL_DEPTH_MIN_STD
                    and minimap_std >= INITIAL_MINIMAP_MIN_STD
                    and minimap_edge_count >= INITIAL_MINIMAP_MIN_EDGES
                ):
                    frame_index = len(gif_frames)
                    combined = combined_frame(
                        ego_rgb,
                        depth_rgb,
                        minimap_rgb,
                        scene_name,
                        scene_id,
                        frame_index,
                        curr_xy,
                        curr_heading_xy,
                        target_xy,
                        (pos_x, pos_z),
                        rot_y,
                    )
                    combined.save(frames_dir / f"{frame_index:03d}.png")
                    gif_frames.append(combined)
                    ego_stds.append(ego_std)
                    depth_stds.append(depth_std)
                    minimap_stds.append(minimap_std)
                    ego_edges.append(ego_edge_count)
                    minimap_edges.append(minimap_edge_count)
                    valid_depth = depth_m[np.isfinite(depth_m) & (depth_m >= 0.0)]
                    if valid_depth.size:
                        depth_mins.append(float(valid_depth.min()))
                        depth_maxs.append(float(valid_depth.max()))
                action = str(
                    rng.choice(RANDOM_ACTIONS, p=RANDOM_ACTION_PROBABILITIES)
                )

            env.set_actions(
                BEHAVIOR_NAME,
                ActionTuple(continuous=action2signal(action, ACTION_SPACE_AGENTS)),
            )
            env.step()

        result.frames = len(gif_frames)
        if gif_frames:
            result.ego_shape = list(ego_rgb.shape)
            result.depth_shape = list(depth_rgb.shape)
            result.minimap_shape = list(minimap_rgb.shape)
            result.ego_std_min = round(min(ego_stds), 3)
            result.depth_std_min = round(min(depth_stds), 3)
            result.minimap_std_min = round(min(minimap_stds), 3)
            result.ego_edges_min = min(ego_edges)
            result.minimap_edges_min = min(minimap_edges)
            result.depth_min_m = round(min(depth_mins), 4) if depth_mins else None
            result.depth_max_m = round(max(depth_maxs), 4) if depth_maxs else None
            write_gif(gif_frames, gif_path, args.gif_duration_ms)
            result.gif_path = str(gif_path)

        result.ok = (
            len(gif_frames) == args.frames
            and min(ego_stds, default=0.0) >= 1.0
            and min(minimap_stds, default=0.0) >= 1.0
        )
        if not result.ok:
            result.error = (
                f"Captured {len(gif_frames)}/{args.frames} usable frames; "
                f"ego_std_min={min(ego_stds, default=0.0):.3f}, "
                f"depth_std_min={min(depth_stds, default=0.0):.3f}, "
                f"minimap_std_min={min(minimap_stds, default=0.0):.3f}"
            )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        (scene_dir / "python_error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    finally:
        if env is not None:
            env.close()

    return result


def save_summary(output_dir: Path, results: list[SceneResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": all(result.ok for result in results),
        "passed": sum(result.ok for result in results),
        "total": len(results),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    try:
        resolve_motion_speed_config(args)
        resolve_lighting_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.max_attempts_per_scene <= 0:
        raise SystemExit("--max-attempts-per-scene must be positive")
    if (args.init_world_x is None) != (args.init_world_z is None):
        raise SystemExit("--init-world-x and --init-world-z must be set together")
    app_path = Path(args.file_name).resolve()
    if not app_path.exists():
        raise SystemExit(f"Unity client not found: {app_path}")

    patch_observation_decoding()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SceneResult] = []

    for offset, (scene_name, scene_id) in enumerate(selected_scenes(args.scenes)):
        print(f"[{offset + 1:02d}] validating {scene_name} (build index {scene_id})", flush=True)
        for attempt in range(1, args.max_attempts_per_scene + 1):
            worker_id = (
                args.worker_id_start
                + offset * args.max_attempts_per_scene
                + attempt
                - 1
            )
            result = validate_scene(args, scene_name, scene_id, worker_id)
            result.attempts = attempt
            if result.ok or attempt == args.max_attempts_per_scene:
                break
            print(
                f"[RETRY] {scene_name}: random-spawn attempt "
                f"{attempt}/{args.max_attempts_per_scene} failed quality checks",
                flush=True,
            )
        results.append(result)
        save_summary(output_dir, results)
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {scene_name}: {result.frames}/{args.frames} frames", flush=True)
        if result.error:
            print(f"        {result.error}", flush=True)

    save_summary(output_dir, results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
