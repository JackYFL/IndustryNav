"""Interactive human-teleop data collection against the unified scene_all client.

Drives the unified client with keyboard control (WASD/arrows via an OpenCV
window), letting a human assign spawn/target points and walk a trajectory while
per-frame RGB/depth/minimap + a ``keyboard_actions.csv`` are recorded. The saved
layout (``keyboard_fp/`` / ``keyboard_depth/`` / ``keyboard_actions.csv``) is
what the BC datasets in :mod:`nav.train.dataset` consume.

Shares the side-channel + minimap-margin + obs-decode plumbing with the
benchmark entry via :mod:`nav.harness`. The spawn/target priming here is
*interactive* (point selection), so it does not use
``nav.harness.env_setup.setup_and_prime`` (which is the benchmark's
non-interactive CLI-driven path).
"""

import argparse
import csv
import datetime
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from nav.config import ACTION_SPACE_ANNOTATION as ACTION_SPACE
from nav.config import BEHAVIOR_NAME, UNITY_ENGINE_QUALITY_LEVEL
from nav.harness.coordinates import find_exact_map_bounds
from nav.harness.coordinates import visual_to_unity_coords as _visual_to_unity_coords
from nav.harness.observations import get_obs_safe, patch_observation_decoding
from nav.harness.perception import detect_red_point
from nav.harness.side_channels import BoundsSideChannel, TargetSideChannel
from nav.utils import (
    obs_to_rgb,
    action2signal,
    pix_dist,
    logger_config,
    save_frame_from_obs,
    clear_and_reset_dir,
    append_results_csv,
    ensure_parent_dir,
)

def parse_args():
    p = argparse.ArgumentParser(
        description="IndustryNav v6: Egocentric + State-based navigation with History"
    )
    p.add_argument(
        "--frame_sleep",
        type=float,
        default=0.0,
        help="Artificial delay between frames.",
    )
    p.add_argument("--max_steps", type=int, default=100, help="0 = run indefinitely.")

    p.add_argument(
        "--frame_save_dir",
        type=str,
        default="./exp_frames_v6",
        help="Base directory to save frames.",
    )
    p.add_argument("--results_csv", type=str, default="./experiment_results_v6.csv")
    p.add_argument(
        "--actions_csv",
        type=str,
        default="",
        help="Path to save per-frame action + control signals CSV (default: frame_save_dir/keyboard_actions.csv).",
    )

    p.add_argument(
        "--modalities",
        type=str,
        default="ego,minimap,depth",
        help="Comma-separated: ego,minimap,depth,his",
    )

    p.add_argument(
        "--sim_steps_per_decision",
        type=int,
        default=2,
        help="Unity sim steps per decision (lower for snappier keyboard control).",
    )
    p.add_argument(
        "--key_poll_ms",
        type=int,
        default=100,
        help="Keyboard polling interval in ms (1 = non-blocking).",
    )
    p.add_argument(
        "--buffer_max",
        type=int,
        default=0,
        help="Max frames to keep in memory before flushing to disk (0 = no flush until exit).",
    )
    p.add_argument(
        "--save_workers",
        type=int,
        default=4,
        help="Number of worker threads for saving images.",
    )
    p.add_argument(
        "--marker_source",
        choices=("red", "vector"),
        default="vector",
        help="How to locate/draw the current agent marker on the minimap. "
        "'red' uses cone color detection; 'vector' projects Unity vector obs "
        "(pos_x,pos_z,rot_y) into the minimap and draws a Python-side arrow.",
    )

    p.add_argument("--target_x", type=int, default=None, help="Target x on minimap.")
    p.add_argument("--target_y", type=int, default=None, help="Target y on minimap.")
    p.add_argument(
        "--reach_px", type=float, default=20.0, help="Stop threshold in pixels."
    )
    p.add_argument("--init_curr_x", type=int, default=None, help="Initial current x for frame 1")
    p.add_argument("--init_curr_y", type=int, default=None, help="Initial current y for frame 1")
    p.add_argument("--pixel_x", type=int, default=None, help="Preset clicked pixel x for init point (skip click if set with pixel_y)")
    p.add_argument("--pixel_y", type=int, default=None, help="Preset clicked pixel y for init point (skip click if set with pixel_x)")
    p.add_argument("--init_curr_direction", type=float, help="Initial direction of the agent")
    p.add_argument("--scene_id", type=int, default=3, help="Unity scene_id environment parameter.")
    p.add_argument(
        "--spline_speed_multiplier",
        type=float,
        default=2.0,
        help="Runtime multiplier for Unity SplineAnimate environment-object speed. "
             "1.0 uses the speed serialized in the scene.",
    )
    p.add_argument(
        "--screen_width",
        type=int,
        default=1724,
        help="Unity player width used to constrain window aspect ratio.",
    )
    p.add_argument(
        "--screen_height",
        type=int,
        default=1024,
        help="Unity player height used to constrain window aspect ratio.",
    )
    p.add_argument(
        "--ego_width",
        type=int,
        default=512,
        help="Width of the egocentric RGB and depth observations.",
    )
    p.add_argument(
        "--ego_height",
        type=int,
        default=512,
        help="Height of the egocentric RGB and depth observations.",
    )
    p.add_argument(
        "--file_name",
        type=str,
        default="/Users/liyifan/Documents/MyCodes/release/mac/scene_all.app",
        help="Path to Unity executable",
    )

    p.add_argument(
        "--model_id",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model id",
    )

    return p.parse_args()


def draw_curr_target_action(
    minimap_rgb,
    curr_xy,
    target_xy,
    action_label: str,
    curr_heading_xy=None,
):
    bgr = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)
    if curr_xy is not None:
        cv2.circle(bgr, (int(curr_xy[0]), int(curr_xy[1])), 4, (0, 0, 255), -1)
        if curr_heading_xy is not None:
            cv2.arrowedLine(
                bgr,
                (int(curr_xy[0]), int(curr_xy[1])),
                (int(curr_heading_xy[0]), int(curr_heading_xy[1])),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.35,
            )
    if target_xy is not None:
        cv2.circle(bgr, (int(target_xy[0]), int(target_xy[1])), 3, (0, 200, 0), -1)
    if action_label:
        cv2.putText(
            bgr,
            f"action: {action_label}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def heading_endpoint_from_screen_direction(center_xy, direction_deg, length=24, img_shape=None):
    """
    Build a screen-space arrow endpoint for the init picker.
    Existing init controls use: 90=up, 270=down, 360/0=left, 180=right.
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


def annotate_minimap_for_save(minimap_obs_raw, curr_xy, target_xy, curr_heading_xy=None):
    if minimap_obs_raw is None:
        return minimap_obs_raw
    rgb = obs_to_rgb(minimap_obs_raw)
    annotated_rgb = draw_curr_target_action(
        rgb,
        curr_xy,
        target_xy,
        action_label="",
        curr_heading_xy=curr_heading_xy,
    )
    return np.transpose(annotated_rgb, (2, 0, 1))


def window_to_image_coords(x, y, img_w, img_h, shown_w, shown_h, win_w, win_h):
    """Map window click to original image pixels with robust letterbox handling."""
    if img_w <= 0 or img_h <= 0 or shown_w <= 0 or shown_h <= 0 or win_w <= 0 or win_h <= 0:
        return None

    # Step 1: map from window coords to displayed-frame coords.
    scale_win = min(win_w / float(shown_w), win_h / float(shown_h))
    disp_w = shown_w * scale_win
    disp_h = shown_h * scale_win
    off_x = (win_w - disp_w) * 0.5
    off_y = (win_h - disp_h) * 0.5

    if x < off_x or x >= (off_x + disp_w) or y < off_y or y >= (off_y + disp_h):
        return None

    shown_x = (x - off_x) / scale_win
    shown_y = (y - off_y) / scale_win

    # Step 2: map from displayed-frame coords back to original image coords.
    sx = img_w / float(shown_w)
    sy = img_h / float(shown_h)
    px = shown_x * sx
    py = shown_y * sy
    px = int(np.clip(round(px), 0, img_w - 1))
    py = int(np.clip(round(py), 0, img_h - 1))
    return px, py


def visual_to_unity_coords(margin, px, py):
    if margin is not None:
        return _visual_to_unity_coords(margin, px, py)
    return float(px), float(py)


def world2pixel(x, z):
    ax, bx, cx = 9.842864E-10, -0.0754061, 43.76626
    az, bz, cz = -0.06702765, 1.205371E-08, 68.32464
    det = ax * bz - bx * az
    if det == 0:
        return 0, 0
    px = (bz * (x - cx) - bx * (z - cz)) / det
    py = (ax * (z - cz) - az * (x - cx)) / det
    return int(px), int(py)


def unity_to_visual_coords(margin, unity_px, unity_py, img_shape=None):
    """
    Inverse of visual_to_unity_coords. Maps Unity minimap pixels back to the
    displayed/saved minimap image coordinates.
    """
    if margin is None:
        px, py = float(unity_px), float(unity_py)
    else:
        min_x, max_x, min_y, max_y = margin
        unity_w, unity_h = 862.0, 512.0
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


def world_to_unity_coords(projector, world_x, world_z):
    """
    Project world X/Z into Unity minimap pixels. Prefer the current run's
    side-channel calibration, then fall back to the static camera affine.
    """
    if projector is None:
        return world2pixel(float(world_x), float(world_z))

    spawn_px, spawn_py = projector["spawn_pixel"]
    spawn_x, spawn_z = projector["spawn_world"]
    target_px, target_py = projector["target_pixel"]
    target_x, target_z = projector["target_world"]

    fallback_px, fallback_py = world2pixel(float(world_x), float(world_z))

    if abs(target_z - spawn_z) > 1e-6:
        px = spawn_px + (float(world_z) - spawn_z) * (target_px - spawn_px) / (target_z - spawn_z)
    else:
        px = fallback_px

    if abs(target_x - spawn_x) > 1e-6:
        py = spawn_py + (float(world_x) - spawn_x) * (target_py - spawn_py) / (target_x - spawn_x)
    else:
        py = fallback_py

    return px, py


def world_to_visual_coords(margin, world_x, world_z, img_shape=None, projector=None):
    """
    Project Unity world X/Z into minimap image coordinates using the same affine
    mapping as the current Unity minimap camera setup.
    """
    unity_px, unity_py = world_to_unity_coords(projector, world_x, world_z)
    return unity_to_visual_coords(margin, unity_px, unity_py, img_shape=img_shape)


def vector_marker_from_obs(margin, pos_x, pos_z, rot_y, img_shape=None, projector=None):
    """
    Return (center_xy, heading_xy) for a Python-side minimap arrow.
    Unity yaw convention: 0 deg faces +Z; 90 deg faces +X.
    """
    center_xy = world_to_visual_coords(
        margin, pos_x, pos_z, img_shape=img_shape, projector=projector
    )
    yaw_rad = np.deg2rad(float(rot_y))
    ahead_world_x = float(pos_x) + np.sin(yaw_rad) * 1.5
    ahead_world_z = float(pos_z) + np.cos(yaw_rad) * 1.5
    heading_xy = world_to_visual_coords(
        margin,
        ahead_world_x,
        ahead_world_z,
        img_shape=img_shape,
        projector=projector,
    )
    return center_xy, heading_xy


def signed_angle_to_target_deg(pos_x, pos_z, rot_y, target_world_x, target_world_z):
    if target_world_x is None or target_world_z is None:
        return None
    dx = float(target_world_x) - float(pos_x)
    dz = float(target_world_z) - float(pos_z)
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    desired_yaw = np.degrees(np.arctan2(dx, dz))
    return float((desired_yaw - float(rot_y) + 180.0) % 360.0 - 180.0)


def select_init_point(
    minimap_rgb,
    type="spawn",
    window_name="minimap_init",
    marker_xy=None,
    marker_direction=None,
    preview_selection=False,
):
    if minimap_rgb is None:
        return None

    img_h, img_w = minimap_rgb.shape[:2]
    base = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)
    overlay = base.copy()
    cv2.putText(
        overlay,
        f"Click to set {type} (ESC/Q to cancel)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Enter/Space to confirm; close window to cancel"
        if preview_selection
        else "Close window to cancel",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if marker_xy is not None:
        heading_xy = heading_endpoint_from_screen_direction(
            marker_xy,
            marker_direction,
            length=24,
            img_shape=minimap_rgb.shape,
        )
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_rgb = draw_curr_target_action(
            overlay_rgb,
            marker_xy,
            None,
            "",
            curr_heading_xy=heading_xy,
        )
        overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    display_scale = 2.0 if img_w < 800 else 1.0
    d_w, d_h = int(img_w * display_scale), int(img_h * display_scale)
    clicked_pt = {"value": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px = int(round(x / display_scale))
        py = int(round(y / display_scale))
        px = max(0, min(px, img_w - 1))
        py = max(0, min(py, img_h - 1))
        clicked_pt["value"] = (px, py)
        print(f"DEBUG: OpenCV Selected -> px:{px}, py:{py}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, d_w, d_h)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = overlay.copy()
        if preview_selection and clicked_pt["value"] is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = draw_curr_target_action(
                frame_rgb,
                clicked_pt["value"],
                None,
                "",
            )
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        display_frame = cv2.resize(frame, (d_w, d_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(20) & 0xFFFFFFFF

        if clicked_pt["value"] is not None and not preview_selection:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return clicked_pt["value"]

        if clicked_pt["value"] is not None and key in (13, 10, ord(" ")):
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return clicked_pt["value"]

        if key == 27 or key == ord("q"):
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return None

        # User clicked close button.
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return None

def select_init_direction(minimap_rgb=None, window_name="minimap_init", marker_xy=None):
    """
    Select initial direction using arrow keys.
    Returns rotation in degrees or None if cancelled.
    """
    if minimap_rgb is not None:
        img_h, img_w = minimap_rgb.shape[:2]
        base = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)
        overlay = base.copy()
        display_scale = 2.0 if img_w < 800 else 1.0
        d_w, d_h = int(img_w * display_scale), int(img_h * display_scale)
    else:
        overlay = np.zeros((400, 800, 3), dtype=np.uint8)
        d_w, d_h = 800, 400

    selected_direction = {"value": None}

    lines = [
        "Select initial direction (Arrow keys or WASD):",
        "Up/W: 90 (N), Down/S: 270 (S)",
        "Left/A: 360 (W), Right/D: 180 (E)",
        "Enter/Space to confirm selected arrow",
        "ESC/Q to cancel",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (15, 35 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, d_w, d_h)

    while True:
        frame = overlay.copy()
        if marker_xy is not None:
            heading_xy = None
            if selected_direction["value"] is not None:
                heading_xy = heading_endpoint_from_screen_direction(
                    marker_xy,
                    selected_direction["value"],
                    length=24,
                    img_shape=frame.shape,
                )
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = draw_curr_target_action(
                frame_rgb,
                marker_xy,
                None,
                "",
                curr_heading_xy=heading_xy,
            )
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        display_frame = cv2.resize(frame, (d_w, d_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(20) & 0xFFFFFFFF
        if key == 27 or key == ord("q"):
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return None
        # Arrow key codes
        if key in (2490368, 82) or key == ord("w"):   # Up
            if marker_xy is None:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
                return 90.0
            selected_direction["value"] = 90.0
            continue
        if key in (2621440, 84) or key == ord("s"):   # Down
            if marker_xy is None:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
                return 270.0
            selected_direction["value"] = 270.0
            continue
        if key in (2424832, 81) or key == ord("a"):   # Left
            if marker_xy is None:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
                return 360.0
            selected_direction["value"] = 360.0
            continue
        if key in (2555904, 83) or key == ord("d"):   # Right
            if marker_xy is None:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
                return 180.0
            selected_direction["value"] = 180.0
            continue
        if key in (13, 10) and selected_direction["value"] is not None:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return selected_direction["value"]
        if key == ord(" ") and selected_direction["value"] is not None:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            return selected_direction["value"]


def save_obs_buffered(save_dir, items, workers: int):
    if not items:
        return
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [
            executor.submit(save_frame_from_obs, obs, save_dir=save_dir, filename=fn)
            for fn, obs in items
        ]
        for f in futures:
            _ = f.result()


def get_minimap_rgb_for_init(env, behavior_name, max_attempts=10):
    for _ in range(max_attempts):
        decision_steps, _ = env.get_steps(behavior_name)
        if len(decision_steps) > 0:
            minimap_obs_raw = get_obs_safe(decision_steps, "minimap")
            if minimap_obs_raw is not None:
                return obs_to_rgb(minimap_obs_raw)
            depth_obs_raw = get_obs_safe(decision_steps, "depth")
            if depth_obs_raw is not None:
                return obs_to_rgb(depth_obs_raw)
            ego_obs = get_obs_safe(decision_steps, "ego")
            if ego_obs is not None:
                return obs_to_rgb(ego_obs)
        env.set_actions(
            behavior_name, ActionTuple(continuous=action2signal("stop", ACTION_SPACE))
        )
        env.step()
    return None


def main():
    args = parse_args()
    logger = logger_config(args.frame_save_dir)
    # Tolerate the scene_all build's sensor-shape quirk before env launch.
    patch_observation_decoding()

    minimap_aspect = 862.0 / 512.0
    requested_aspect = float(args.screen_width) / float(args.screen_height)
    logger.info(
        f"Requested player window: {args.screen_width}x{args.screen_height} "
        f"(aspect={requested_aspect:.6f}); minimap reference aspect={minimap_aspect:.6f}"
    )
    if abs(requested_aspect - minimap_aspect) > 1e-4:
        logger.warning(
            "Requested screen aspect does not match minimap reference aspect 862/512. "
            "PixelToWorld may still differ across platforms."
        )

    enabled_modalities = set(
        [m.strip() for m in args.modalities.split(",") if m.strip()]
    )
    if "minimap" not in enabled_modalities:
        enabled_modalities.add("minimap")
        logger.info("Keyboard mode requires minimap; adding 'minimap' to modalities.")
    assert enabled_modalities.issubset(
        {"ego", "minimap", "depth", "his"}
    ), "modalities must be subset of {ego,minimap,depth,his}"

    # Initialize history tracking
    # Unity env
    engine = EngineConfigurationChannel()
    # Create the channel
    bounds_channel = BoundsSideChannel()
    target_sc = TargetSideChannel()
    env_params = EnvironmentParametersChannel()

    use_batchmode_default = "0" if platform.system() == "Linux" else "1"
    use_batchmode = os.environ.get("INDUSTRYNAV_UNITY_BATCHMODE", use_batchmode_default) == "1"
    if args.ego_width <= 0 or args.ego_height <= 0:
        raise ValueError("Egocentric RGB/depth resolution must use positive width and height values.")
    unity_args = [
        "-logFile", "test.log",
        "-screen-width", str(args.screen_width),
        "-screen-height", str(args.screen_height),
        "--ego-width", str(args.ego_width),
        "--ego-height", str(args.ego_height),
    ]
    if use_batchmode:
        unity_args.insert(0, "-batchmode")

    env = UnityEnvironment(
        file_name=args.file_name,
        # file_name=None,
        no_graphics=False,                             # keep rendering for CameraSensor
        side_channels=[engine, env_params, bounds_channel, target_sc],
        additional_args=unity_args,
        # worker_id=1,
        worker_id=1,
        base_port=5507,
        timeout_wait=120,
    )

    engine.set_configuration_parameters(
        time_scale=1,
        # Project quality index 0 is "Ray Tracing (Realtime GI)", which can
        # add stochastic visual noise to CameraSensor frames.
        quality_level=UNITY_ENGINE_QUALITY_LEVEL,
        target_frame_rate=-1,
        width=args.screen_width,
        height=args.screen_height,
    )
    env_params.set_float_parameter("scene_id", float(args.scene_id))
    env_params.set_float_parameter("use_ai_control", 1.0)
    env_params.set_float_parameter("spline_speed_multiplier", float(args.spline_speed_multiplier))
    logger.info(f"Unity launch args: {unity_args}")
    logger.info("Starting Unity…")
    env.reset()
    logger.info(f"Behaviors: {list(env.behavior_specs.keys())}")

    logger.info("Connecting to Unity Editor...")
    env.reset()
    _ = env.behavior_specs[BEHAVIOR_NAME]
    logger.info(f"Connected to behavior: {BEHAVIOR_NAME}")

    margin = None
    target_world_x = None
    target_world_z = None
    minimap_projector = None

    if (args.pixel_x is None) != (args.pixel_y is None):
        logger.info("Only one of --pixel_x/--pixel_y is set; ignoring preset pixel and using click selection.")

    # If init not provided, show minimap and click to select before setting spawn params.
    if args.init_curr_x is None or args.init_curr_y is None:
        minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
        margin = find_exact_map_bounds(logger, minimap_rgb)
        if minimap_rgb is not None:
            h, w = minimap_rgb.shape[:2]
            env_params.set_float_parameter("minimap_px_width", float(w))
            env_params.set_float_parameter("minimap_px_height", float(h))
            if args.pixel_x is not None and args.pixel_y is not None:
                args.init_curr_x, args.init_curr_y = int(args.pixel_x), int(args.pixel_y)
                logger.info(f"Using preset initial pixel from args: ({args.init_curr_x}, {args.init_curr_y})")
            else:
                logger.info("Click to set initial position on minimap.")
                clicked = select_init_point(
                    minimap_rgb,
                    type="spawn point",
                    window_name="minimap_init",
                    preview_selection=True,
                )
                if clicked is not None:
                    args.init_curr_x, args.init_curr_y = clicked
                    logger.info(f"Selected initial position: {clicked}")
        else:
            logger.info("Minimap unavailable for init selection.")

    init_world_x = None
    init_world_z = None
    if args.init_curr_x is not None and args.init_curr_y is not None:
        minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
        if minimap_rgb is not None:
            h, w = minimap_rgb.shape[:2]
            env_params.set_float_parameter("minimap_px_width", float(w))
            env_params.set_float_parameter("minimap_px_height", float(h))
 
        # Step 1: send spawn pixels; Unity maps (px, py) -> (x, z)

        upx, upy = visual_to_unity_coords(margin, args.init_curr_x, args.init_curr_y)
        env_params.set_float_parameter("spawn_px", float(upx))
        env_params.set_float_parameter("spawn_py", float(upy))
        env_params.set_float_parameter("spawn_wy", 0.5)
        # Apply spawn position firstw
        env.reset()

        # Step 2: select direction (separate step)
        if args.init_curr_direction is None:
            logger.info("Select initial direction with arrow keys or WASD.")
            minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
            args.init_curr_direction = select_init_direction(
                minimap_rgb=minimap_rgb,
                window_name="minimap_init",
                marker_xy=(int(args.init_curr_x), int(args.init_curr_y)),
            )
            if args.init_curr_direction is not None:
                logger.info(f"Selected initial direction: {args.init_curr_direction}")

        # Apply rotation (keep position)
        if args.init_curr_direction is not None:
            env_params.set_float_parameter("spawn_rot", float(args.init_curr_direction))
        else:
            env_params.set_float_parameter("spawn_rot", 180.0)
            
        upx, upy = visual_to_unity_coords(margin, args.init_curr_x, args.init_curr_y)
        env_params.set_float_parameter("spawn_px", float(upx))
        env_params.set_float_parameter("spawn_py", float(upy))
        env_params.set_float_parameter("spawn_wy", 0.5)
        env.reset()
        
        # logger.info(f"target_sc.last_spawn_pixel: {target_sc.last_spawn_pixel}, target_sc.last_spawn_world: {target_sc.last_spawn_world}")
        # logger.info(f"{target_sc.last_spawn_pixel}, {(int(args.init_curr_x), int(args.init_curr_y))}, {target_sc.last_spawn_world}")
        init_world_x, init_world_z = target_sc.last_spawn_world
        # import ipdb; ipdb.set_trace()
        logger.info(
            f"Spawn in Visual Pixel coords ({int(args.init_curr_x)}, {int(args.init_curr_y)}) -> "
            f"Spawn in Unity Pixel coords {target_sc.last_spawn_pixel} -> "
            f"Spawn in World coords ({init_world_x:.2f}, {init_world_z:.2f})"
        )

    # If target not provided, show minimap and click to select target position.
    if args.target_x is None or args.target_y is None:
        minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
        if minimap_rgb is not None:
            logger.info("Click to set target position on minimap.")
            clicked = select_init_point(
                minimap_rgb,
                type="target point",
                window_name="minimap_target",
                marker_xy=(int(args.init_curr_x), int(args.init_curr_y))
                if args.init_curr_x is not None and args.init_curr_y is not None
                else None,
                marker_direction=args.init_curr_direction,
            )
            if clicked is not None:
                args.target_x, args.target_y = clicked
                logger.info(f"Selected target position: {clicked}")
        else:
            logger.info("Minimap unavailable for target selection.")

    if args.target_x is not None and args.target_y is not None:
        minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
        if minimap_rgb is not None:
            h, w = minimap_rgb.shape[:2]
            env_params.set_float_parameter("minimap_px_width", float(w))
            env_params.set_float_parameter("minimap_px_height", float(h))
        upx, upy = visual_to_unity_coords(margin, args.target_x, args.target_y) # coordinate transform
        env_params.set_float_parameter("target_px", float(upx))
        env_params.set_float_parameter("target_py", float(upy))
        env.reset()
        
         
        # logger.info(f"target_sc.last_target_pixel: {target_sc.last_target_pixel}, target_sc.last_target_world: {target_sc.last_target_world}")
  
        target_world_x, target_world_z = target_sc.last_target_world
        minimap_projector = build_axis_aligned_projector(
            target_sc.last_spawn_pixel,
            target_sc.last_spawn_world,
            target_sc.last_target_pixel,
            target_sc.last_target_world,
        )
        logger.info(
            f"Target in Visual Pixel coords ({int(args.target_x)}, {int(args.target_y)}) -> "
            f"Target in Unity Pixel coords {target_sc.last_target_pixel} -> "
            f"Target in World coords ({target_world_x:.2f}, {target_world_z:.2f})"
        )

    # per-modality save dirs
    subdirs = {}
    if "ego" in enabled_modalities:
        subdirs["ego"] = os.path.join(args.frame_save_dir, "keyboard_fp")
        clear_and_reset_dir(subdirs["ego"])
    if "minimap" in enabled_modalities:
        subdirs["minimap"] = os.path.join(args.frame_save_dir, "keyboard_minimap")
        clear_and_reset_dir(subdirs["minimap"])
    if "depth" in enabled_modalities:
        subdirs["depth"] = os.path.join(args.frame_save_dir, "keyboard_depth")
        clear_and_reset_dir(subdirs["depth"])

    # Buffer frames in memory; save on exit or when buffer_max reached
    frame_buffers = {}
    if "ego" in subdirs:
        frame_buffers["ego"] = []
    if "minimap" in subdirs:
        frame_buffers["minimap"] = []
    if "depth" in subdirs:
        frame_buffers["depth"] = []
    buffer_max = max(0, int(args.buffer_max))
    save_workers = max(1, int(args.save_workers))
    action_log = []
    actions_csv_path = (
        args.actions_csv.strip()
        if args.actions_csv.strip()
        else os.path.join(args.frame_save_dir, "keyboard_actions.csv")
    )

    # Define target variables before state initialization
    target_xy = (args.target_x, args.target_y) if args.target_x is not None and args.target_y is not None else None
    # if target_xy is not None and target_sc.last_target_pixel == (int(target_xy[0]), int(target_xy[1])) and target_sc.last_target_world is not None:
    #     target_world_x, target_world_z = target_sc.last_target_world
    # else:
    #     target_world_x, target_world_z = None, None

    # Shared state for real-time target updates
    state = {
        "target_xy": target_xy,
        "target_world_x": target_world_x,
        "target_world_z": target_world_z
    }

    mouse_ctx = {
        "img_shape": None,
        "shown_size": None,
    }

    def on_minimap_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get window dimensions to handle resizing if necessary
            # For now, assume window size matches image size as in select_init_point
            img_shape = param.get("img_shape")
            shown_size = param.get("shown_size")
            if img_shape is None or shown_size is None:
                return

            img_h, img_w = img_shape[:2]
            shown_w, shown_h = shown_size
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect("minimap")
                if win_w > 0 and win_h > 0:
                    mapped = window_to_image_coords(x, y, img_w, img_h, shown_w, shown_h, win_w, win_h)
                    if mapped is None:
                        return
                    px, py = mapped
                else:
                    px = int(np.clip(x, 0, img_w - 1))
                    py = int(np.clip(y, 0, img_h - 1))
            except:
                px = int(np.clip(x, 0, img_w - 1))
                py = int(np.clip(y, 0, img_h - 1))
            
            state["target_xy"] = (px, py) # Keep pure visual coordinates to draw the green marker accurately.
            # state["target_world_x"] = None
            # state["target_world_z"] = None
            
            # target_sc.send_target(px, py)
            # logger.info(f"New target set via click: Pixel ({px}, {py}); sent to Unity for mapping")
            # Convert to Unity coordinates before sending.
            upx, upy = visual_to_unity_coords(margin, px, py)
            target_sc.send_target(int(upx), int(upy))
            logger.info(f"New target set via click: Visual ({px}, {py}) -> Unity ({int(upx)}, {int(upy)})")


    last_curr_xy = None

    step_count = 0            # decision steps (frame ticks)
    forward_count = 0         # count of "forward" actions
    stop_reason = None

    # ===========================
    # KEY FIX (SYNC, NOT ASYNC):
    # After each decision, advance Unity multiple sim frames so animations move.
    # ===========================
    SIM_STEPS_PER_DECISION = max(1, int(args.sim_steps_per_decision))

    # Keep last action applied during sim-steps
    last_continuous_actions = action2signal("stop", ACTION_SPACE)
    last_keyboard_action = "stop"

    cv2.namedWindow("minimap", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("minimap", 1280, 720)
    # Callback will be set once we have the minimap shape in the loop
    callback_set = False

    try:
        while True:
            step_count += 1
            if args.max_steps > 0 and forward_count >= args.max_steps:
                stop_reason = "max_forward"
                logger.info(f"Max forward: {forward_count} >= {args.max_steps}")
                break

            decision_steps, _ = env.get_steps(BEHAVIOR_NAME)
            
            # print("--- SENSOR DIAGNOSTIC ---")
            # for i, obs in enumerate(decision_steps.obs):
            #     print(f"Sensor Index [{i}] Shape: {obs.shape}")
            # print("-------------------------")

            # If no agent is requesting a decision this tick, just step forward a bit.
            # (This avoids edge cases if DecisionRequester period > 1.)
            if len(decision_steps) == 0:
                env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=last_continuous_actions))
                env.step()
                if args.frame_sleep and args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)
                continue

            # Gather observations
            ego_obs = (
                get_obs_safe(decision_steps, "ego")
                if "ego" in enabled_modalities
                else None
            )
            minimap_obs_raw = (
                get_obs_safe(decision_steps, "minimap")
                if "minimap" in enabled_modalities
                else None
            )
            depth_obs_raw = (
                get_obs_safe(decision_steps, "depth")
                if "depth" in enabled_modalities
                else None
            )

            # Vector observations: position and rotation are usually the last item
            vector_obs = decision_steps.obs[-1]
            if vector_obs.shape[1] >= 3:
                pos_x, pos_y, pos_z = vector_obs[0, 0], vector_obs[0, 1], vector_obs[0, 2]
            else:
                logger.warning(f"Vector obs size is only {vector_obs.shape[1]}, setting position to 0.")
                pos_x, pos_y, pos_z = 0.0, 0.0, 0.0

            if vector_obs.shape[1] >= 6:
                rot_x, rot_y, rot_z = vector_obs[0, 3], vector_obs[0, 4], vector_obs[0, 5]
            else:
                rot_x, rot_y, rot_z = 0.0, 0.0, 0.0   

            logger.info(
                f"[Step:{step_count}] Position: X={float(pos_x):.2f}, Y={float(pos_y):.2f}, Z={float(pos_z):.2f}, "
                f"Rotation: RotX={float(rot_x):.2f}, RotY={float(rot_y):.2f}, RotZ={float(rot_z):.2f}"
            )

            # Save raw frames (same as original: once per decision step).
            # Minimap is appended after curr_xy is computed, so Python-side
            # vector markers are also present in saved minimap frames.
            if ego_obs is not None and "ego" in subdirs:
                frame_buffers["ego"].append((f"{step_count}.png", ego_obs))
            if depth_obs_raw is not None and "depth" in subdirs:
                frame_buffers["depth"].append((f"{step_count}.png", depth_obs_raw))

            # --------- Decide action (keyboard only) ----------
            true_minimap_rgb = (
                obs_to_rgb(minimap_obs_raw) if minimap_obs_raw is not None else None
            )

            if target_sc.last_target_world is not None:
                state["target_world_x"], state["target_world_z"] = target_sc.last_target_world
                minimap_projector = build_axis_aligned_projector(
                    target_sc.last_spawn_pixel,
                    target_sc.last_spawn_world,
                    target_sc.last_target_pixel,
                    target_sc.last_target_world,
                )

            curr_heading_xy = None
            heading_delta_deg = signed_angle_to_target_deg(
                pos_x,
                pos_z,
                rot_y,
                state["target_world_x"],
                state["target_world_z"],
            )

            if true_minimap_rgb is not None:
                if args.marker_source == "vector":
                    curr_xy, curr_heading_xy = vector_marker_from_obs(
                        margin,
                        pos_x,
                        pos_z,
                        rot_y,
                        img_shape=true_minimap_rgb.shape,
                        projector=minimap_projector,
                    )
                    detected_xy = curr_xy
                else:
                    map_bgr = cv2.cvtColor(true_minimap_rgb, cv2.COLOR_RGB2BGR)
                    _, detected_xy, _ = detect_red_point(
                        map_bgr, target_h_deg=350.0, target_s=1.0, target_v=0.827
                    )
            else:
                # Auto-calculate position from vector observation if minimap is missing
                if args.marker_source == "vector":
                    curr_xy, curr_heading_xy = vector_marker_from_obs(
                        margin,
                        pos_x,
                        pos_z,
                        rot_y,
                        img_shape=None,
                        projector=minimap_projector,
                    )
                    detected_xy = curr_xy
                else:
                    logger.error("Minimap RGB unavailable; cannot detect position.")
                    exit()


            if detected_xy is None and step_count == 1:
                if args.init_curr_x is not None and args.init_curr_y is not None:
                    curr_xy = (int(args.init_curr_x), int(args.init_curr_y))
                else:
                    curr_xy = None
            else:
                curr_xy = detected_xy

            if curr_xy is not None:
                last_curr_xy = curr_xy

            if minimap_obs_raw is not None and "minimap" in subdirs:
                minimap_save = annotate_minimap_for_save(
                    minimap_obs_raw,
                    curr_xy,
                    state["target_xy"],
                    curr_heading_xy=curr_heading_xy,
                )
                frame_buffers["minimap"].append((f"{step_count}.png", minimap_save))

            if buffer_max > 0:
                for key, items in frame_buffers.items():
                    if len(items) >= buffer_max:
                        save_dir = subdirs.get(key)
                        if save_dir:
                            save_obs_buffered(save_dir, items, save_workers)
                        frame_buffers[key].clear()

            # if state["target_xy"] is not None:
            #     expected_target = (int(state["target_xy"][0]), int(state["target_xy"][1]))
            #     if target_sc.last_target_pixel == expected_target and target_sc.last_target_world is not None:
            #         state["target_world_x"], state["target_world_z"] = target_sc.last_target_world

            display_rgb = true_minimap_rgb
            if display_rgb is None:
                display_rgb = obs_to_rgb(depth_obs_raw) if depth_obs_raw is not None else None
            if display_rgb is None:
                display_rgb = obs_to_rgb(ego_obs) if ego_obs is not None else None

            if display_rgb is not None:
                if not callback_set:
                    cv2.setMouseCallback("minimap", on_minimap_click, param=mouse_ctx)
                    callback_set = True

                annotated_rgb = draw_curr_target_action(
                    display_rgb,
                    curr_xy,
                    state["target_xy"],
                    last_keyboard_action,
                    curr_heading_xy=curr_heading_xy,
                )

                display_h, display_w = annotated_rgb.shape[:2]
                scale = 2  # Adjust display zoom as needed, e.g. 1.5 or 2.0.
                resized_for_show = cv2.resize(
                    annotated_rgb, 
                    (int(display_w * scale), int(display_h * scale)), 
                    interpolation=cv2.INTER_LINEAR
                )

                shown_h, shown_w = resized_for_show.shape[:2]
                mouse_ctx["img_shape"] = annotated_rgb.shape
                mouse_ctx["shown_size"] = (shown_w, shown_h)

                # cv2.imshow("minimap", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("minimap", cv2.cvtColor(resized_for_show, cv2.COLOR_RGB2BGR))
                # Non-blocking key polling; keep last action if no key
                key = cv2.waitKey(max(1, int(args.key_poll_ms))) & 0xFF
                if key == ord("w"):
                    last_keyboard_action = "forward"
                elif key == ord("a"):
                    last_keyboard_action = "turn left"
                elif key == ord("d"):
                    last_keyboard_action = "turn right"
                elif key == ord("s") or key == ord(" "):
                    last_keyboard_action = "stop"
                elif key == ord("q") or key == 27:
                    stop_reason = "user_quit"
                    logger.info("User requested quit.")
                    break

            if (curr_xy is not None) and (state["target_xy"] is not None):
                distance = pix_dist(curr_xy, state["target_xy"])
                if distance <= args.reach_px:
                    chosen_action = "stop"
                else:
                    chosen_action = last_keyboard_action
            else:
                chosen_action = "stop"

            continuous_actions = action2signal(chosen_action, ACTION_SPACE)

            # Send action (log same as original)
            action_sent = continuous_actions[0]
            logger.info(
                f"[Step:{step_count}] Sending Action -> Move:{action_sent[0]:.2f}, "
                f"Strafe:{action_sent[1]:.2f}, Look:{action_sent[2]:.2f}"
            )
            if chosen_action == "forward":
                forward_count += 1
            action_log.append(
                {
                    "step": step_count,
                    "action": chosen_action,
                    "move": float(action_sent[0]),
                    "strafe": float(action_sent[1]),
                    "look": float(action_sent[2]),
                    "init_px": int(args.init_curr_x) if args.init_curr_x is not None else None,
                    "init_py": int(args.init_curr_y) if args.init_curr_y is not None else None,
                    "init_world_x": init_world_x,
                    "init_world_z": init_world_z,
                    "init_direction": float(args.init_curr_direction)
                    if args.init_curr_direction is not None
                    else None,
                    "curr_px": int(curr_xy[0]) if curr_xy is not None else None,
                    "curr_py": int(curr_xy[1]) if curr_xy is not None else None,
                    "curr_world_x": float(pos_x),
                    "curr_world_y": float(pos_y),
                    "curr_world_z": float(pos_z),
                    "curr_direction_y": float(rot_y),
                    "target_px": int(state["target_xy"][0]) if state["target_xy"] is not None else None,
                    "target_py": int(state["target_xy"][1]) if state["target_xy"] is not None else None,
                    "target_world_x": state["target_world_x"],
                    "target_world_z": state["target_world_z"],
                    "heading_delta_deg": heading_delta_deg,
                    "marker_source": args.marker_source,
                    "distance_px": float(pix_dist(curr_xy, state["target_xy"]))
                    if (curr_xy is not None and state["target_xy"] is not None)
                    else None,
                    "distance_world": float(
                        np.hypot(float(pos_x) - state["target_world_x"], float(pos_z) - state["target_world_z"])
                    )
                    if (state["target_world_x"] is not None and state["target_world_z"] is not None)
                    else None,
                }
            )

            # Apply chosen action and advance Unity multiple sim frames so animations move
            last_continuous_actions = continuous_actions
            for step in range(max(1, SIM_STEPS_PER_DECISION)):
                if step == SIM_STEPS_PER_DECISION-1:
                    env.set_actions(BEHAVIOR_NAME, ActionTuple(continuous=last_continuous_actions))
                env.step()
                # if args.frame_sleep and args.frame_sleep > 0:
                #     time.sleep(args.frame_sleep)

            # Early stop when near goal (same logic)
            if (state["target_xy"] is not None) and (last_curr_xy is not None):
                if pix_dist(last_curr_xy, state["target_xy"]) <= args.reach_px and chosen_action == "stop":
                    stop_reason = "reached_vicinity"
                    logger.info(f"[Step:{step_count}] Reached target. Stopping.")
                    break

        # ===== End of loop: compute final distance & log =====
        final_xy = last_curr_xy
        if final_xy is not None and state["target_xy"] is not None:
            distance_px = float(pix_dist(final_xy, state["target_xy"]))
        else:
            distance_px = float("nan")

        result = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "exec_mode": "keyboard",
            "provider": "openrouter",
            "model": args.model_id,
            "max_steps": int(args.max_steps),
            "reach_px": float(args.reach_px),
            "init_direction": float(args.init_curr_direction)
            if args.init_curr_direction is not None
            else None,
            "target_x": int(state["target_xy"][0]) if state["target_xy"] is not None else None,
            "target_y": int(state["target_xy"][1]) if state["target_xy"] is not None else None,
            "final_x": int(final_xy[0]) if final_xy is not None else None,
            "final_y": int(final_xy[1]) if final_xy is not None else None,
            "distance_px": distance_px,
            "stop_reason": stop_reason or "loop_end",
            "steps_taken": forward_count,
            "frame_sleep": float(args.frame_sleep),
            "modalities": ",".join(sorted(enabled_modalities)),
            "sim_steps_per_decision": int(SIM_STEPS_PER_DECISION),
            "marker_source": args.marker_source,
            "spline_speed_multiplier": float(args.spline_speed_multiplier),
        }

        logger.info(
            f"[FINAL RESULT] Stop={result['stop_reason']} | Steps={result['steps_taken']} | "
            f"FinalPos={final_xy} | Target={state['target_xy']} | Distance={distance_px:.2f}px"
        )
        append_results_csv(args.results_csv, result)
        logger.info(f"Results saved to: {os.path.abspath(args.results_csv)}")

    finally:
        try:
            env.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # Flush buffered frames to disk
        for key, items in frame_buffers.items():
            save_dir = subdirs.get(key)
            if not save_dir:
                continue
            save_obs_buffered(save_dir, items, save_workers)
        if action_log:
            ensure_parent_dir(actions_csv_path)
            header = [
                "step",
                "action",
                "move",
                "strafe",
                "look",
                "init_px",
                "init_py",
                "init_world_x",
                "init_world_z",
                "init_direction",
                "curr_px",
                "curr_py",
                "curr_world_x",
                "curr_world_y",
                "curr_world_z",
                "curr_direction_y",
                "target_px",
                "target_py",
                "target_world_x",
                "target_world_z",
                "heading_delta_deg",
                "marker_source",
                "distance_px",
                "distance_world",
            ]
            with open(actions_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(action_log)
            logger.info(f"Actions saved to: {os.path.abspath(actions_csv_path)}")


if __name__ == "__main__":
    main()
