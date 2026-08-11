"""Headless async benchmarking entry point for the unified Unity client.

Targets the unified `scene_all` client (24 scenes bundled, selected at runtime via
the `scene_id` env parameter). Borrows the side-channel + spawn/target priming
pipeline from `run_headless_collect_data_assign_points_v2.py`, and the async
decision state machine + LLM/BC controller plumbing from `run_async.py`.

Designed to be the single benchmarking entry point across macOS/Linux/Windows
(macOS validated first). Requires CLI-supplied spawn pixel + direction + target
pixel — no interactive selection.
"""
import argparse
import csv
import datetime
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from mlagents_envs.base_env import ActionTuple

from nav.baselines.astar import AStarBaseline
from nav.config import (
    ACTIONS_CSV_FIELDS,
    ACTION_SPACE_AGENTS,
    ACTION_SPACE_ANNOTATION,
    ASTAR_DEFAULTS,
    BEHAVIOR_NAME,
    DEFAULT_REACH_DISTANCE_M,
    BENCHMARK_BASELINES,
    DEFAULT_PROMPT_VISION,
    LLM_DEFAULT_HISTORY_SIZE,
    LLM_DEFAULT_MAX_TOKENS,
    UNITY_SCENE_COUNT,
    UNITY_MAP_SIZE,
    resolve_scene_all_path,
)
from nav.harness.env_setup import EnvSetupError, setup_and_prime
from nav.harness.lighting import add_lighting_args, lighting_result_fields
from nav.harness.motion import add_motion_speed_args, motion_speed_result_fields
from nav.harness.coordinates import (
    canonical_to_minimap_coords,
    minimap_pixel_scale,
    minimap_to_canonical_coords,
    resolve_minimap_resolution,
    runtime_minimap_distance_to_canonical,
    visual_to_unity_coords,
)
from nav.harness.observations import get_obs_safe
from nav.harness.perception import detect_red_point
from nav.harness.prompt_assembly import format_history_for_prompt, render_nav_prompt
from nav.harness.routing import execute_decision
from nav.utils import (
    action2signal,
    append_results_csv,
    clear_and_reset_dir,
    get_allowed_actions,
    load_prompt_template,
    logger_config,
    obs_to_rgb,
    save_frame_from_obs,
    unity_rotation_to_egocentric_theta,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="IndustryNav headless async benchmarking against the unified Unity client."
    )
    p.add_argument("--baseline", type=str, default="llm", choices=BENCHMARK_BASELINES,
                   help="Decision baseline: random|llm|bc|astar.")
    p.add_argument("--frame_sleep", type=float, default=0.0,
                   help="Idle sleep while a decision thread is in flight.")
    p.add_argument("--max_steps", type=int, default=70, help="0 = run indefinitely.")

    p.add_argument("--frame_save_dir", type=str, default="./outputs/run")
    # Default to a per-run file under frame_save_dir to avoid touching the
    # historical experiment_results_v6.csv at repo root. Pass an explicit
    # path to override.
    p.add_argument("--results_csv", type=str, default="",
                   help="If empty, defaults to <frame_save_dir>/results.csv.")
    p.add_argument("--actions_csv", type=str, default="")

    p.add_argument("--modalities", type=str, default="ego,minimap,depth",
                   help="Comma-separated subset of {ego,minimap,depth,his}.")
    p.add_argument("--marker_source", choices=("red", "vector"), default="vector",
                   help="How to locate/draw the current minimap marker. "
                        "'vector' projects Unity vector obs into minimap pixels and "
                        "draws a Python-side red dot/arrow; 'red' uses legacy HSV "
                        "detection of the Unity-rendered red marker.")
    p.add_argument("--hide_unity_red_marker", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="When --marker_source=vector, remove the Unity-rendered red "
                        "cone/dot from saved/planned minimaps before drawing the "
                        "Python-side red marker. Use --no-hide_unity_red_marker to "
                        "keep raw Unity marker pixels.")

    # Unity client + scene selection (unified scene_all client)
    p.add_argument("--file_name", type=str, default="auto",
                   help="Path to the unified Unity executable (e.g. scene_all.app). "
                        "Pass 'auto' (default) to scan config.SCENE_ALL_BUILDS for the "
                        "current OS and use the first build that exists locally. Pass "
                        "an explicit path to override.")
    p.add_argument("--scene_id", type=int, required=True,
                   help=f"Zero-based scene id baked into the unified client "
                        f"(0..{UNITY_SCENE_COUNT - 1}).")
    p.add_argument("--scene_name", type=str, default="",
                   help="Scene code (e.g. 'scene1'). Recorded in results.csv as both "
                        "scene_name and the legacy exp_name column. Optional; the wrappers "
                        "and run_benchmark_grid.py pass it through.")
    p.add_argument("--point_id", type=str, default="",
                   help="Start-target pair identifier (e.g. 'point1'). Recorded in results.csv.")
    p.add_argument("--seed_id", type=str, default="",
                   help="Run replicate identifier. Closed-source LLMs ignore this for "
                        "sampling; we record it as a label so multiple runs of the same "
                        "(scene, point, model) cell are distinguishable.")
    p.add_argument("--vision_input", type=lambda s: str(s).lower() in {"1","true","yes","on"},
                   default=True,
                   help="If False, suppress the egocentric image when calling the LLM. "
                        "Pair with --prompt_file prompts/nav_state_history_no_vision.txt "
                        "for the matching no-vision-language prompt.")
    p.add_argument("--worker_id", type=int, default=1)
    p.add_argument("--base_port", type=int, default=5507)
    p.add_argument("--screen_width", type=int, default=1724,
                   help="Unity player/window width; does not set sensor resolution.")
    p.add_argument("--screen_height", type=int, default=1024,
                   help="Unity player/window height; does not set sensor resolution.")
    p.add_argument("--ego_width", type=int, default=512,
                   help="Width of the egocentric RGB and depth observations.")
    p.add_argument("--ego_height", type=int, default=512,
                   help="Height of the egocentric RGB and depth observations.")
    p.add_argument("--minimap_width", type=int, default=None,
                   help="Optional minimap width; derived from height when omitted.")
    p.add_argument("--minimap_height", type=int, default=None,
                   help="Optional minimap height; derived from width when omitted.")
    p.add_argument("--sim_steps_per_decision", type=int, default=2)
    p.add_argument(
        "--dynamic_objects",
        choices=("moving", "static"),
        default="moving",
        help="Run environment objects normally or freeze them in place. "
             "The navigation agent remains controllable.",
    )
    add_motion_speed_args(p)
    add_lighting_args(p)

    # Spawn. Two mutually-exclusive ways to specify position:
    #   (preferred) --init_world_x / --init_world_z : Unity world coords.
    #     `input_points.json` `start.{x,z}` are world coords and should be
    #     forwarded here. The unified client spawns directly at this point.
    #   (legacy)   --init_curr_x / --init_curr_y : visual minimap pixels.
    #     The script converts to Unity pixel via the letterbox-aware margin
    #     and the client does pixel->world internally.
    p.add_argument("--init_world_x", type=float, default=None,
                   help="Spawn x in Unity world coords.")
    p.add_argument("--init_world_z", type=float, default=None,
                   help="Spawn z in Unity world coords.")
    p.add_argument("--init_curr_x", type=int, default=None,
                   help="Spawn x in canonical 862x512 minimap pixels. Used only if "
                        "--init_world_x/--init_world_z are not provided.")
    p.add_argument("--init_curr_y", type=int, default=None,
                   help="Spawn y in canonical 862x512 minimap pixels. Used only if "
                        "--init_world_x/--init_world_z are not provided.")
    p.add_argument("--init_curr_direction", type=float, default=180.0,
                   help="Spawn yaw in degrees (Unity convention; 0=W,90=N,180=E,270=S).")

    # Target
    p.add_argument("--target_x", type=int, required=True,
                   help="Target x in canonical 862x512 minimap pixels.")
    p.add_argument("--target_y", type=int, required=True,
                   help="Target y in canonical 862x512 minimap pixels.")
    p.add_argument(
        "--reach_m",
        type=float,
        default=DEFAULT_REACH_DISTANCE_M,
        help=(
            "Success radius in Unity world meters (default: "
            f"{DEFAULT_REACH_DISTANCE_M:g})."
        ),
    )

    # LLM / BC
    p.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_VISION)
    p.add_argument("--model_id", type=str, default="openai/gpt-4o-mini")
    p.add_argument("--max_tokens", type=int, default=LLM_DEFAULT_MAX_TOKENS)
    p.add_argument("--history_size", type=int, default=LLM_DEFAULT_HISTORY_SIZE)
    p.add_argument("--bc_ckpt", type=str,
                   default="ckpts/nav_bc_resnet50_causal_transformer_depth_aug_remove_stop_seq_32_bs4_num_layers3/best.pt")
    p.add_argument("--bc_device", type=str, default="auto")
    p.add_argument("--bc_seq_len", type=int, default=0)

    # A* baseline (no LLM, no API key).
    p.add_argument(
        "--astar_obstacle_clearance_m",
        type=float,
        default=ASTAR_DEFAULTS.obstacle_clearance_m,
        help="Physical clearance around minimap obstacles in Unity world meters.",
    )
    p.add_argument(
        "--astar_proxy_stop_distance_m",
        "--astar_proxy_stop_real_dist_m",
        dest="astar_proxy_stop_distance_m",
        type=float,
        default=ASTAR_DEFAULTS.proxy_stop_distance_m,
        help="Blocked-target proxy threshold in Unity world meters.",
    )
    p.add_argument("--astar_debug_viz", action="store_true",
                   help="Dump A* walkable-grid / path overlays per step.")
    p.add_argument("--astar_debug_dir", type=str, default="",
                   help="Where to write A* debug images (default: <frame_save_dir>/astar_debug).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Python-side minimap marker overlay
# ---------------------------------------------------------------------------
def world2pixel(x, z, map_size=UNITY_MAP_SIZE):
    """Fallback affine from Unity world X/Z into Unity minimap pixels."""
    ax, bx, cx = 9.842864E-10, -0.0754061, 43.76626
    az, bz, cz = -0.06702765, 1.205371E-08, 68.32464
    det = ax * bz - bx * az
    if det == 0:
        return 0, 0
    px = (bz * (x - cx) - bx * (z - cz)) / det
    py = (ax * (z - cz) - az * (x - cx)) / det
    scale_x = float(map_size[0]) / float(UNITY_MAP_SIZE[0])
    scale_y = float(map_size[1]) / float(UNITY_MAP_SIZE[1])
    return float(px) * scale_x, float(py) * scale_y


def unity_to_visual_coords(
    margin,
    unity_px,
    unity_py,
    img_shape=None,
    unity_map_size=UNITY_MAP_SIZE,
):
    """Inverse of the letterbox-aware visual->Unity minimap coordinate mapping."""
    if margin is None:
        px, py = float(unity_px), float(unity_py)
    else:
        min_x, max_x, min_y, max_y = margin
        unity_w, unity_h = unity_map_size
        px = min_x + (float(unity_px) / unity_w) * (max_x - min_x)
        py = min_y + (float(unity_py) / unity_h) * (max_y - min_y)

    if img_shape is not None:
        h, w = img_shape[:2]
        px = np.clip(round(px), 0, w - 1)
        py = np.clip(round(py), 0, h - 1)
    return int(px), int(py)


def build_axis_aligned_projector(spawn_pixel, spawn_world, target_pixel, target_world):
    if (
        spawn_pixel is None
        or spawn_world is None
        or target_pixel is None
        or target_world is None
    ):
        return None
    return {
        "spawn_pixel": (float(spawn_pixel[0]), float(spawn_pixel[1])),
        "spawn_world": (float(spawn_world[0]), float(spawn_world[1])),
        "target_pixel": (float(target_pixel[0]), float(target_pixel[1])),
        "target_world": (float(target_world[0]), float(target_world[1])),
    }


def world_to_unity_coords(projector, world_x, world_z, map_size=UNITY_MAP_SIZE):
    """Project world X/Z into Unity minimap pixels using run calibration when available."""
    if projector is None:
        return world2pixel(float(world_x), float(world_z), map_size=map_size)

    spawn_px, spawn_py = projector["spawn_pixel"]
    spawn_x, spawn_z = projector["spawn_world"]
    target_px, target_py = projector["target_pixel"]
    target_x, target_z = projector["target_world"]

    fallback_px, fallback_py = world2pixel(float(world_x), float(world_z), map_size=map_size)

    if abs(target_z - spawn_z) > 1e-6:
        px = spawn_px + (float(world_z) - spawn_z) * (target_px - spawn_px) / (target_z - spawn_z)
    else:
        px = fallback_px

    if abs(target_x - spawn_x) > 1e-6:
        py = spawn_py + (float(world_x) - spawn_x) * (target_py - spawn_py) / (target_x - spawn_x)
    else:
        py = fallback_py

    return px, py


def unity_to_world_coords(projector, unity_px, unity_py):
    """Invert the run-calibrated axis-aligned world/minimap projection."""
    if projector is None:
        return None

    spawn_px, spawn_py = projector["spawn_pixel"]
    spawn_x, spawn_z = projector["spawn_world"]
    target_px, target_py = projector["target_pixel"]
    target_x, target_z = projector["target_world"]

    if abs(target_py - spawn_py) > 1e-6:
        world_x = spawn_x + (float(unity_py) - spawn_py) * (
            target_x - spawn_x
        ) / (target_py - spawn_py)
    else:
        world_x = spawn_x - 0.0754061 * (float(unity_py) - spawn_py)

    if abs(target_px - spawn_px) > 1e-6:
        world_z = spawn_z + (float(unity_px) - spawn_px) * (
            target_z - spawn_z
        ) / (target_px - spawn_px)
    else:
        world_z = spawn_z - 0.06702765 * (float(unity_px) - spawn_px)
    return float(world_x), float(world_z)


def visual_to_world_coords(
    margin,
    visual_xy,
    projector,
    map_size=UNITY_MAP_SIZE,
):
    """Convert a runtime minimap point to Unity world X/Z coordinates."""
    if projector is None:
        return None
    unity_px, unity_py = visual_to_unity_coords(
        margin,
        float(visual_xy[0]),
        float(visual_xy[1]),
        map_size=map_size,
    )
    return unity_to_world_coords(projector, unity_px, unity_py)


def world_to_visual_coords(
    margin,
    world_x,
    world_z,
    img_shape=None,
    projector=None,
    map_size=UNITY_MAP_SIZE,
):
    unity_px, unity_py = world_to_unity_coords(
        projector,
        world_x,
        world_z,
        map_size=map_size,
    )
    return unity_to_visual_coords(
        margin,
        unity_px,
        unity_py,
        img_shape=img_shape,
        unity_map_size=map_size,
    )


def heading_endpoint_from_screen_direction(center_xy, direction_deg, length=24, img_shape=None):
    """
    Build a screen-space arrow endpoint using the same convention as the
    collect_data init picker: 90=up, 270=down, 360/0=left, 180=right.
    """
    if center_xy is None or direction_deg is None:
        return None
    direction = float(direction_deg) % 360.0
    rad = np.deg2rad(direction)
    dx = -np.cos(rad) * float(length)
    dy = -np.sin(rad) * float(length)
    x = float(center_xy[0]) + dx
    y = float(center_xy[1]) + dy
    if img_shape is not None:
        h, w = img_shape[:2]
        x = np.clip(round(x), 0, w - 1)
        y = np.clip(round(y), 0, h - 1)
    return int(x), int(y)


def vector_marker_from_obs(
    margin,
    pos_x,
    pos_z,
    rot_y,
    img_shape=None,
    projector=None,
    map_size=UNITY_MAP_SIZE,
    pixel_scale=1.0,
):
    """Return (center_xy, heading_xy) for a Python-side minimap marker arrow.

    The marker center uses world->minimap projection, while the arrow endpoint
    follows collect_data's screen-space direction helper. This keeps the A*
    marker consistent with the init picker and avoids distorting the heading
    when the world->minimap projector is calibrated from only spawn/target
    points.
    """
    center_xy = world_to_visual_coords(
        margin,
        pos_x,
        pos_z,
        img_shape=img_shape,
        projector=projector,
        map_size=map_size,
    )
    heading_xy = heading_endpoint_from_screen_direction(
        center_xy,
        rot_y,
        length=max(1, round(24 * float(pixel_scale))),
        img_shape=img_shape,
    )
    return center_xy, heading_xy


def draw_curr_target_heading_rgb(
    minimap_rgb,
    curr_xy,
    target_xy,
    curr_heading_xy=None,
    pixel_scale=1.0,
):
    """Draw Python-side current marker, heading arrow, and target marker."""
    bgr = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)
    if curr_xy is not None:
        curr_radius = max(2, round(5 * float(pixel_scale)))
        cv2.circle(bgr, (int(curr_xy[0]), int(curr_xy[1])), curr_radius, (0, 0, 255), -1)
        if curr_heading_xy is not None:
            cv2.arrowedLine(
                bgr,
                (int(curr_xy[0]), int(curr_xy[1])),
                (int(curr_heading_xy[0]), int(curr_heading_xy[1])),
                (0, 0, 255),
                max(1, round(2 * float(pixel_scale))),
                cv2.LINE_AA,
                tipLength=0.35,
            )
    if target_xy is not None:
        target_radius = max(2, round(6 * float(pixel_scale)))
        cv2.circle(bgr, (int(target_xy[0]), int(target_xy[1])), target_radius, (0, 200, 0), -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def remove_unity_red_marker_rgb(minimap_rgb, pixel_scale=1.0):
    """Remove the old Unity-rendered red marker from an RGB minimap image."""
    if minimap_rgb is None:
        return minimap_rgb
    bgr = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)
    _, _, mask = detect_red_point(bgr, draw=False)
    if mask is None or cv2.countNonZero(mask) == 0:
        return minimap_rgb

    kernel_size = max(1, round(9 * float(pixel_scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpaint_radius = max(1, round(5 * float(pixel_scale)))
    cleaned_bgr = cv2.inpaint(bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)


def maybe_calibrate_projector_from_raw_marker(
    raw_minimap_rgb,
    margin,
    target_sc,
    world_x,
    world_z,
    logger,
    map_size=UNITY_MAP_SIZE,
):
    """Use the Unity red marker once to calibrate world->minimap projection.

    World-spawn benchmark runs do not receive a spawn pixel ack from Unity.
    Without that second calibration point, vector markers fall back to a static
    affine that can be wrong for some scenes. The raw Unity marker is still
    useful for one-time calibration; saved/planned minimaps can then hide it.
    """
    if (
        raw_minimap_rgb is None
        or target_sc.last_target_pixel is None
        or target_sc.last_target_world is None
    ):
        return None

    map_bgr = cv2.cvtColor(raw_minimap_rgb, cv2.COLOR_RGB2BGR)
    _, detected_xy, mask = detect_red_point(map_bgr, draw=False)
    if detected_xy is None or mask is None or cv2.countNonZero(mask) == 0:
        return None

    spawn_upx, spawn_upy = visual_to_unity_coords(
        margin,
        detected_xy[0],
        detected_xy[1],
        map_size=map_size,
    )
    projector = build_axis_aligned_projector(
        (spawn_upx, spawn_upy),
        (float(world_x), float(world_z)),
        target_sc.last_target_pixel,
        target_sc.last_target_world,
    )
    logger.info(
        "Calibrated vector marker projection from raw Unity marker: "
        f"visual={detected_xy} unity=({spawn_upx:.1f},{spawn_upy:.1f}) "
        f"world=({float(world_x):.2f},{float(world_z):.2f})"
    )
    return projector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def action_space_for_baseline(baseline: str) -> dict:
    """Use agent controls for learned/LLM agents; keep A* on annotation controls."""
    if baseline == "astar":
        return ACTION_SPACE_ANNOTATION
    return ACTION_SPACE_AGENTS


def target_reached(
    curr_world_xz,
    target_world_xz,
    reach_m,
):
    """Return target reach status using Unity world X/Z coordinates only."""
    if curr_world_xz is None or target_world_xz is None:
        raise ValueError("Target reach requires current and target world coordinates.")
    distance_m = float(
        np.hypot(
            curr_world_xz[0] - target_world_xz[0],
            curr_world_xz[1] - target_world_xz[1],
        )
    )
    return distance_m <= float(reach_m), distance_m


def main():
    args = parse_args()
    if args.reach_m <= 0:
        raise SystemExit("--reach_m must be positive.")
    try:
        args.minimap_width, args.minimap_height = resolve_minimap_resolution(
            args.minimap_width,
            args.minimap_height,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    minimap_size = (int(args.minimap_width), int(args.minimap_height))
    pixel_scale = minimap_pixel_scale(minimap_size)
    # Resolve --file_name="auto" against config.SCENE_ALL_BUILDS so a single
    # invocation works across dev machines without a per-host wrapper hack.
    args.file_name = resolve_scene_all_path(args.file_name)
    logger = logger_config(args.frame_save_dir)

    enabled_modalities = {m.strip() for m in args.modalities.split(",") if m.strip()}
    if "minimap" not in enabled_modalities:
        # We require minimap for red-dot position tracking + margin computation.
        enabled_modalities.add("minimap")
        logger.info("Adding 'minimap' modality (required for position tracking).")
    assert enabled_modalities.issubset({"ego", "minimap", "depth", "his"}), \
        "modalities must be subset of {ego,minimap,depth,his}"

    action_space = action_space_for_baseline(args.baseline)
    allowed_actions = get_allowed_actions(action_space)
    logger.info(
        "Action space: "
        f"{'ACTION_SPACE_ANNOTATION' if action_space is ACTION_SPACE_ANNOTATION else 'ACTION_SPACE_AGENTS'} "
        f"{action_space}"
    )
    prompt_template = load_prompt_template(args.prompt_file) if args.baseline == "llm" else ""
    history_deque = deque(maxlen=args.history_size if args.history_size > 0 else None)

    astar_planner = None
    if args.baseline == "astar":
        astar_debug_dir = (
            args.astar_debug_dir.strip()
            if args.astar_debug_dir.strip()
            else os.path.join(args.frame_save_dir, "astar_debug")
        )
        astar_planner = AStarBaseline(
            obstacle_clearance_m=args.astar_obstacle_clearance_m,
            proxy_stop_distance_m=args.astar_proxy_stop_distance_m,
            pixel_scale=pixel_scale,
            debug_viz=args.astar_debug_viz,
            debug_dir=astar_debug_dir,
        )
        logger.info(
            "A* baseline ready | "
            f"grid_cell_m={ASTAR_DEFAULTS.grid_cell_m:g} | "
            f"obstacle_clearance_m={args.astar_obstacle_clearance_m:g} | "
            f"proxy_stop_distance_m={args.astar_proxy_stop_distance_m:g} | "
            f"pixel_scale={pixel_scale:.4f} | "
            f"debug_viz={args.astar_debug_viz} | debug_dir={astar_debug_dir}"
        )

    bc_controller = None
    if args.baseline == "bc":
        from nav.train.controller import BCNavController
        bc_controller = BCNavController(
            ckpt_path=args.bc_ckpt,
            device=args.bc_device,
            seq_len_override=args.bc_seq_len,
        )
        logger.info(
            f"BC agent loaded from {args.bc_ckpt} | device={bc_controller.device} | "
            f"seq_len={bc_controller.seq_len}"
        )

    # ---- Launch Unity + prime spawn/target (see nav.harness.env_setup) ----
    try:
        primed = setup_and_prime(args, logger)
    except EnvSetupError as e:
        logger.error(str(e))
        sys.exit(2)
    env = primed.env
    init_world_x, init_world_z = primed.init_world or (None, None)
    target_world_x, target_world_z = primed.target_world or (None, None)
    target_xy = primed.target_xy
    logger.info(
        f"Minimap runtime space: {minimap_size[0]}x{minimap_size[1]} | "
        f"reach_m={args.reach_m:g}"
    )
    minimap_projector = build_axis_aligned_projector(
        primed.target_sc.last_spawn_pixel,
        primed.target_sc.last_spawn_world,
        primed.target_sc.last_target_pixel,
        primed.target_sc.last_target_world,
    )

    def astar_point_to_world(point):
        return visual_to_world_coords(
            primed.margin,
            point,
            minimap_projector,
            map_size=minimap_size,
        )

    # ---- Per-modality save dirs (prefixed by the baseline token) ----
    subdirs = {}
    for mod in ("ego", "minimap", "depth"):
        if args.baseline == "llm" and mod == "minimap":
            continue
        if mod in enabled_modalities:
            subdirs[mod] = os.path.join(args.frame_save_dir, f"{args.baseline}_{('fp' if mod=='ego' else mod)}")
            clear_and_reset_dir(subdirs[mod])
    subdir_ann = os.path.join(args.frame_save_dir, f"{args.baseline}_minimap_target")
    clear_and_reset_dir(subdir_ann)

    actions_csv_path = (
        args.actions_csv.strip()
        if args.actions_csv.strip()
        else os.path.join(args.frame_save_dir, f"{args.baseline}_actions.csv")
    )
    os.makedirs(os.path.dirname(os.path.abspath(actions_csv_path)), exist_ok=True)
    _action_csv_file = open(actions_csv_path, "w", newline="")
    _action_csv_writer = csv.DictWriter(
        _action_csv_file,
        fieldnames=ACTIONS_CSV_FIELDS,
    )
    _action_csv_writer.writeheader()
    _action_csv_file.flush()
    action_log = []

    agent_qa_path = os.path.join(args.frame_save_dir, "agent_qa.txt")
    _qa_file = open(agent_qa_path, "w", encoding="utf-8")

    # ---- Main async loop (derived from run_async.py) ----
    SIM_STEPS_PER_DECISION = max(1, int(args.sim_steps_per_decision))
    decision_thread = None
    decision_result = {}
    last_prompt = ""
    detected_init_xy = None
    last_curr_xy = None
    last_world_xz = None
    last_saved_step = None
    step_count = 0
    stop_reason = None

    try:
        while True:
            if args.max_steps > 0 and step_count >= args.max_steps:
                stop_reason = "max_steps"
                logger.info(f"Max steps reached: {step_count} >= {args.max_steps}")
                break

            decision_steps, _ = env.get_steps(BEHAVIOR_NAME)

            ego_obs = get_obs_safe(decision_steps, "ego") if "ego" in enabled_modalities else None
            minimap_obs_raw = get_obs_safe(decision_steps, "minimap") if "minimap" in enabled_modalities else None
            depth_obs_raw = get_obs_safe(decision_steps, "depth") if "depth" in enabled_modalities else None

            # Vector observation slot (4th sensor in the unified build).
            try:
                vector_obs = decision_steps.obs[3]
                if vector_obs.shape[1] < 6:
                    raise ValueError(
                        f"expected at least 6 values, got {vector_obs.shape[1]}"
                    )
                pos_x, pos_y, pos_z = vector_obs[0, 0], vector_obs[0, 1], vector_obs[0, 2]
                rot_x, rot_y, rot_z = vector_obs[0, 3], vector_obs[0, 4], vector_obs[0, 5]
            except Exception as exc:
                stop_reason = "world_pose_unavailable"
                logger.error(
                    "Unity world pose observation is unavailable; "
                    f"world-coordinate navigation cannot continue: {exc}"
                )
                break
            last_world_xz = (float(pos_x), float(pos_z))

            if decision_thread is not None and decision_thread.is_alive():
                # Decision in flight: advance Unity with STOP, but do not resave
                # the same decision-step frame under the same step_count.
                env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=action2signal("stop", action_space)))
                for _ in range(SIM_STEPS_PER_DECISION):
                    env.step()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)
                continue

            curr_xy = last_curr_xy
            curr_heading_xy = None
            sensor_minimap_rgb = (
                obs_to_rgb(minimap_obs_raw) if minimap_obs_raw is not None else None
            )
            raw_minimap_rgb = sensor_minimap_rgb
            if args.marker_source == "vector" and minimap_projector is None:
                calibrated_projector = maybe_calibrate_projector_from_raw_marker(
                    raw_minimap_rgb,
                    primed.margin,
                    primed.target_sc,
                    pos_x,
                    pos_z,
                    logger,
                    map_size=minimap_size,
                )
                if calibrated_projector is not None:
                    minimap_projector = calibrated_projector
            true_minimap_rgb = raw_minimap_rgb
            saved_minimap_rgb = sensor_minimap_rgb
            if (
                true_minimap_rgb is not None
                and args.marker_source == "vector"
                and args.hide_unity_red_marker
            ):
                true_minimap_rgb = remove_unity_red_marker_rgb(
                    true_minimap_rgb,
                    pixel_scale=pixel_scale,
                )
                saved_minimap_rgb = true_minimap_rgb
            ego_rgb = obs_to_rgb(ego_obs) if ego_obs is not None else None

            # The async decision loop observes a step both before launching the
            # worker and after it completes. Persist that step only once.
            should_save_frame = step_count != last_saved_step
            if should_save_frame:
                if ego_obs is not None and "ego" in subdirs:
                    save_frame_from_obs(ego_obs, save_dir=subdirs["ego"], filename=f"{step_count}.png")
                if saved_minimap_rgb is not None and "minimap" in subdirs:
                    cv2.imwrite(
                        os.path.join(subdirs["minimap"], f"{step_count}.png"),
                        cv2.cvtColor(saved_minimap_rgb, cv2.COLOR_RGB2BGR),
                    )
                if depth_obs_raw is not None and "depth" in subdirs:
                    if depth_obs_raw.ndim == 2:
                        depth_obs_raw = depth_obs_raw[None, ...]
                    if depth_obs_raw.ndim == 3:
                        save_frame_from_obs(depth_obs_raw, save_dir=subdirs["depth"], filename=f"{step_count}.png")

            # Position tracking / marker drawing. Default to vector obs so the
            # marker cannot disappear behind scene geometry or detach from the
            # Unity-side red cone. Keep HSV red detection as an explicit fallback.
            if args.marker_source == "vector":
                curr_xy, curr_heading_xy = vector_marker_from_obs(
                    primed.margin,
                    pos_x,
                    pos_z,
                    rot_y,
                    img_shape=true_minimap_rgb.shape if true_minimap_rgb is not None else None,
                    projector=minimap_projector,
                    map_size=minimap_size,
                    pixel_scale=pixel_scale,
                )
            elif (step_count == 0 and last_curr_xy is None
                    and args.init_curr_x is not None and args.init_curr_y is not None):
                curr_xy = canonical_to_minimap_coords(
                    (args.init_curr_x, args.init_curr_y),
                    minimap_size,
                )
                logger.info(f"[Frame 0] Using scaled CLI init position: {curr_xy}")
            elif true_minimap_rgb is not None:
                map_bgr = cv2.cvtColor(true_minimap_rgb, cv2.COLOR_RGB2BGR)
                _, detected_xy, _ = detect_red_point(map_bgr)
                curr_xy = detected_xy

            if curr_xy is not None:
                last_curr_xy = curr_xy
                if detected_init_xy is None:
                    detected_init_xy = curr_xy

            if should_save_frame and true_minimap_rgb is not None and subdir_ann is not None:
                annotated = draw_curr_target_heading_rgb(
                    true_minimap_rgb,
                    curr_xy,
                    target_xy,
                    curr_heading_xy=curr_heading_xy,
                    pixel_scale=pixel_scale,
                )
                cv2.imwrite(
                    os.path.join(subdir_ann, f"{step_count}.png"),
                    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                )
            if should_save_frame:
                last_saved_step = step_count

            # ----- Async decision state machine -----
            if decision_thread is not None and not decision_thread.is_alive():
                decision_thread.join()
                decision_thread = None
                step_count += 1
                chosen_action = decision_result.get("action", "stop")
                reasoning = decision_result.get("reasoning", "")
                if decision_result.get("error"):
                    stop_reason = "decision_error"
                    logger.error(f"Decision error at step {step_count}: {reasoning}")
                    break
                if args.baseline == "llm":
                    _qa_file.write(
                        f"=== Step {step_count} ===\n"
                        f"[Q]\n{decision_result.get('prompt', last_prompt)}\n\n"
                        f"[A]\nAction: {chosen_action}\nReasoning: {reasoning}\n"
                        f"{'=' * 60}\n\n"
                    )
                    _qa_file.flush()
                if args.baseline == "llm" and args.history_size > 0 and curr_xy is not None:
                    history_xy = minimap_to_canonical_coords(curr_xy, minimap_size)
                    history_distance_m = float(
                        np.hypot(
                            float(pos_x) - float(target_world_x),
                            float(pos_z) - float(target_world_z),
                        )
                    )
                    history_deque.append({
                        "step": step_count,
                        "position": (round(history_xy[0]), round(history_xy[1])),
                        "world_position": (float(pos_x), float(pos_z)),
                        "theta": unity_rotation_to_egocentric_theta(rot_y),
                        "action": chosen_action,
                        "distance_to_target_m": history_distance_m,
                    })
                logger.info(
                    f"[Step:{step_count}] Action: {chosen_action} | Reasoning: {str(reasoning)[:200]}"
                )
                if args.baseline == "astar" and chosen_action == "stop":
                    stop_reason = "astar_stop"
                continuous_actions = action2signal(chosen_action, action_space)
            else:
                # No decision in flight: either we've reached, or we kick one off.
                reached, reach_distance = target_reached(
                    (float(pos_x), float(pos_z)),
                    (
                        (float(target_world_x), float(target_world_z))
                        if target_world_x is not None and target_world_z is not None
                        else None
                    ),
                    args.reach_m,
                )
                if reached:
                    chosen_action = "stop"
                    step_count += 1
                    continuous_actions = action2signal(chosen_action, action_space)
                    stop_reason = "reached_vicinity"
                    logger.info(
                        f"Reached target at {reach_distance:.2f}m "
                        f"(threshold={args.reach_m:g}m)."
                    )
                else:
                    payload = None
                    if args.baseline == "random":
                        payload = {"action_space": list(action_space.keys())}
                    elif args.baseline == "llm":
                        if curr_xy is not None and target_xy is not None:
                            prompt_curr_xy = minimap_to_canonical_coords(curr_xy, minimap_size)
                            prompt_curr_xy = (
                                round(prompt_curr_xy[0]),
                                round(prompt_curr_xy[1]),
                            )
                            prompt_target_xy = (int(args.target_x), int(args.target_y))
                            distance_px = runtime_minimap_distance_to_canonical(
                                curr_xy,
                                target_xy,
                                minimap_size,
                            )
                            distance_m = float(
                                np.hypot(
                                    float(pos_x) - float(target_world_x),
                                    float(pos_z) - float(target_world_z),
                                )
                            )
                            theta = unity_rotation_to_egocentric_theta(rot_y)
                            history_str = format_history_for_prompt(history_deque)
                            prompt = render_nav_prompt(
                                prompt_template, prompt_curr_xy, prompt_target_xy, theta,
                                distance_px, allowed_actions, history_str,
                                curr_world_xz=(float(pos_x), float(pos_z)),
                                target_world_xz=(
                                    float(target_world_x),
                                    float(target_world_z),
                                ),
                                distance_m=distance_m,
                                reach_m=args.reach_m,
                                dynamic_objects=args.dynamic_objects,
                            )
                            last_prompt = prompt
                            agent_images = (
                                [("ego", ego_rgb)] if (args.vision_input and ego_rgb is not None) else []
                            )
                            payload = {
                                "prompt": prompt,
                                "images": agent_images,
                                "model_id": args.model_id,
                                "max_tokens": args.max_tokens,
                                "allowed_actions": allowed_actions,
                            }
                    elif args.baseline == "astar":
                        if curr_xy is not None and target_xy is not None:
                            payload = {
                                "astar_planner": astar_planner,
                                "minimap_rgb": true_minimap_rgb,
                                "curr_xy": curr_xy,
                                "target_xy": target_xy,
                                "agent_theta": unity_rotation_to_egocentric_theta(rot_y),
                                "reach_m": args.reach_m,
                                "step": step_count,
                                "curr_world_xz": (float(pos_x), float(pos_z)),
                                "target_world_xz": (
                                    (float(target_world_x), float(target_world_z))
                                    if target_world_x is not None
                                    and target_world_z is not None
                                    else None
                                ),
                                "point_to_world": astar_point_to_world,
                            }
                    elif args.baseline == "bc":
                        payload = {
                            "bc_controller": bc_controller,
                            "ego_obs": np.array(ego_obs, copy=True) if ego_obs is not None else None,
                            "depth_obs": np.array(depth_obs_raw, copy=True) if depth_obs_raw is not None else None,
                            "curr_world_x": float(pos_x),
                            "curr_world_z": float(pos_z),
                            "curr_yaw_deg": float(rot_y),
                            "target_world_x": target_world_x,
                            "target_world_z": target_world_z,
                        }

                    if payload is None:
                        chosen_action = "stop"
                        step_count += 1
                        continuous_actions = action2signal(chosen_action, action_space)
                    else:
                        decision_result = {"finished": False}
                        decision_thread = threading.Thread(
                            target=execute_decision,
                            args=(args.baseline, payload, decision_result),
                            daemon=True,
                        )
                        decision_thread.start()
                        env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=action2signal("stop", action_space)))
                        for _ in range(SIM_STEPS_PER_DECISION):
                            env.step()
                        if args.frame_sleep > 0:
                            time.sleep(args.frame_sleep)
                        continue

            # ---- Apply chosen action and step Unity ----
            action_sent = continuous_actions[0]
            logger.info(
                f"[Step:{step_count}] Sending Action -> Move:{action_sent[0]:.2f}, "
                f"Strafe:{action_sent[1]:.2f}, Look:{action_sent[2]:.2f}"
            )
            init_xy_canonical = (
                minimap_to_canonical_coords(detected_init_xy, minimap_size)
                if detected_init_xy is not None
                else None
            )
            curr_xy_canonical = (
                minimap_to_canonical_coords(curr_xy, minimap_size)
                if curr_xy is not None
                else None
            )
            row = {
                "step": step_count,
                "action": chosen_action,
                "move": float(action_sent[0]),
                "strafe": float(action_sent[1]),
                "look": float(action_sent[2]),
                "init_px": round(init_xy_canonical[0]) if init_xy_canonical is not None else None,
                "init_py": round(init_xy_canonical[1]) if init_xy_canonical is not None else None,
                "init_world_x": init_world_x,
                "init_world_z": init_world_z,
                "init_direction": float(args.init_curr_direction),
                "curr_px": round(curr_xy_canonical[0]) if curr_xy_canonical is not None else None,
                "curr_py": round(curr_xy_canonical[1]) if curr_xy_canonical is not None else None,
                "curr_world_x": float(pos_x),
                "curr_world_y": float(pos_y),
                "curr_world_z": float(pos_z),
                "curr_direction_x": float(rot_x),
                "curr_direction_y": float(rot_y),
                "curr_direction_z": float(rot_z),
                "target_px": int(args.target_x),
                "target_py": int(args.target_y),
                "target_world_x": target_world_x,
                "target_world_z": target_world_z,
                "marker_source": args.marker_source,
                "distance_px": (
                    runtime_minimap_distance_to_canonical(curr_xy, target_xy, minimap_size)
                    if curr_xy is not None
                    else None
                ),
                "distance_world": (
                    float(np.hypot(float(pos_x) - target_world_x, float(pos_z) - target_world_z))
                    if (target_world_x is not None and target_world_z is not None) else None
                ),
            }
            action_log.append(row)
            _action_csv_writer.writerow(row)
            _action_csv_file.flush()

            env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=continuous_actions))
            for _ in range(SIM_STEPS_PER_DECISION):
                env.step()

            if stop_reason in {"reached_vicinity", "astar_stop"} and chosen_action == "stop":
                logger.info(f"[Step:{step_count}] {stop_reason}. Stopping.")
                break

    except Exception as e:
        # Any unhandled exception (Unity crash, side-channel error, etc.):
        # convert to a stop_reason and fall through to the result-write block
        # so downstream stats scripts see this cell as a real failure rather
        # than a silently-missing row.
        stop_reason = stop_reason or f"exception:{type(e).__name__}"
        logger.exception(f"Unhandled exception in benchmark loop: {e}")
    finally:
        # ---- Final summary (always runs, even on exception) ----
        try:
            distance_world = (
                float(
                    np.hypot(
                        last_world_xz[0] - target_world_x,
                        last_world_xz[1] - target_world_z,
                    )
                )
                if last_world_xz is not None
                and target_world_x is not None
                and target_world_z is not None
                else None
            )
            result = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                # exp_name kept as a column alias of scene_name for backward compatibility
                # with experiment_results_v6.csv readers; both columns hold the same value.
                "exp_name": args.scene_name,
                "scene_name": args.scene_name,
                "point_id": args.point_id,
                "seed_id": args.seed_id,
                # Column name kept as exec_mode for RESULTS_CSV_FIELDS / historical
                # readers; the value is now the --baseline token (llm/bc/astar/...).
                "exec_mode": args.baseline,
                "provider": "openrouter",
                "model": args.model_id,
                "vision_input": bool(args.vision_input),
                "max_steps": int(args.max_steps),
                "reach_m": args.reach_m,
                "init_world_x": init_world_x,
                "init_world_z": init_world_z,
                "target_world_x": target_world_x,
                "target_world_z": target_world_z,
                "init_direction": float(args.init_curr_direction),
                "final_world_x": last_world_xz[0] if last_world_xz is not None else None,
                "final_world_z": last_world_xz[1] if last_world_xz is not None else None,
                "distance_world": distance_world,
                "stop_reason": stop_reason or "loop_end",
                "steps_taken": step_count,
                "frame_sleep": float(args.frame_sleep),
                "modalities": ",".join(sorted(enabled_modalities)),
                "sim_steps_per_decision": int(args.sim_steps_per_decision),
                "marker_source": args.marker_source,
                "dynamic_objects": args.dynamic_objects,
                **motion_speed_result_fields(args),
                **lighting_result_fields(args),
            }
            world_distance_label = (
                f"{distance_world:.2f}m"
                if distance_world is not None
                else "unavailable"
            )
            logger.info(
                f"[FINAL] stop={result['stop_reason']} steps={result['steps_taken']} "
                f"world_final={last_world_xz} "
                f"world_target={(target_world_x, target_world_z)} "
                f"world_dist={world_distance_label}"
            )
            results_csv_path = (
                args.results_csv.strip()
                if args.results_csv.strip()
                else os.path.join(args.frame_save_dir, "results.csv")
            )
            append_results_csv(results_csv_path, result)
            logger.info(f"Results appended to: {os.path.abspath(results_csv_path)}")
        except Exception:
            logger.exception("Failed to write results.csv")
        try:
            env.close()
        except Exception:
            pass
        _action_csv_file.close()
        _qa_file.close()
        if action_log:
            logger.info(f"Actions saved to: {os.path.abspath(actions_csv_path)}")
        logger.info(f"Agent Q&A saved to: {os.path.abspath(agent_qa_path)}")


if __name__ == "__main__":
    main()
