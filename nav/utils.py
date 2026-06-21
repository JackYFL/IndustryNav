"""Generic stateless utilities used across the codebase.

This module is intentionally a catch-all of reusable helpers grouped by
section below. Things that aren't generic (decision-loop logic, side-channel
codecs, scene-specific transforms beyond the unity↔egocentric primitive
helpers in the geometry section) live in their respective ``nav/<submodule>/``
homes.

If you find yourself adding a new function here, ask: would three unrelated
scripts plausibly call it? If not, it probably belongs in the submodule
that owns the concept.
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import os
import re
import shutil
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from nav.config import RESULTS_CSV_FIELDS


# Configure once on import. ``basicConfig`` is a no-op if the root logger
# already has handlers, so re-imports and test runs stay quiet.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Logging
# ============================================================================

def logger_config(log_dir: str) -> logging.Logger:
    """Configure the root logger to emit to both console and a timestamped file.

    Reuses the root logger so both ``logging.info(...)`` and any module-level
    ``logger.info(...)`` lines reach the same file. Existing handlers are
    cleared first so the function is safe to call multiple times in a session
    (notebooks, repeated runs).
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.handlers.clear()

    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    return root


# ============================================================================
# Filesystem
# ============================================================================

def clear_and_reset_dir(path: str) -> None:
    """Delete ``path`` (if it exists) and recreate it as an empty directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it doesn't already exist."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ============================================================================
# Results CSV
# ============================================================================

def append_results_csv(csv_path: str, row: dict) -> None:
    """Append one run's summary row to ``csv_path``, writing a header if new.

    The fixed schema lives in ``nav.config.RESULTS_CSV_FIELDS`` so callers
    can pre-validate or downstream readers can introspect column order
    without re-reading this function. Extra keys in ``row`` are dropped
    (``extrasaction="ignore"``).
    """
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ============================================================================
# Prompt template I/O (will move to nav/prompts/ in PR 2)
# ============================================================================

def load_prompt_template(path: str) -> str:
    """Read a prompt template file from disk and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# Image encoding (PNG / base64 / data URLs for LLM API calls)
# ============================================================================

def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    """Encode an HWC RGB array as PNG bytes using OpenCV."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def data_url_png_from_rgb(rgb: np.ndarray) -> str:
    """Encode an HWC RGB array as a ``data:image/png;base64,...`` URL."""
    return "data:image/png;base64," + base64.b64encode(
        png_bytes_from_rgb(rgb)
    ).decode("ascii")


# ============================================================================
# Observation / frame conversion
# ============================================================================

def obs_to_rgb(obs_chw: np.ndarray) -> np.ndarray:
    """Convert a Unity CHW observation to an HWC uint8 RGB image.

    Accepts float in [0, 1] or already-uint8 input. Drops the alpha channel
    if present (4-channel obs).
    """
    img = np.transpose(obs_chw, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def save_frame_from_obs(
    obs: np.ndarray, save_dir: str, filename: str = "frame.png"
) -> None:
    """Save a single CHW observation as an image file.

    Supports RGB (C=3) and depth (C=1). Depth is min-max-normalized to a
    grayscale visualization before writing.
    """
    os.makedirs(save_dir, exist_ok=True)
    C = obs.shape[0]

    if C == 3:
        rgb = obs_to_rgb(obs)
        img = Image.fromarray(rgb)
    elif C == 1:
        depth = obs.squeeze(0)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-5:
            depth = (depth - d_min) / (d_max - d_min)
        depth_viz = (depth * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(depth_viz)  # PIL infers mode="L" from 2D uint8
    else:
        raise ValueError(
            f"Unsupported channel count: {C}. Expected 1 (depth) or 3 (RGB)."
        )

    file_path = os.path.join(save_dir, filename)
    img.save(file_path)
    logger.info(f"Frame ({'RGB' if C == 3 else 'Depth'}) saved at {file_path}")


# ============================================================================
# Minimap drawing
# ============================================================================

def draw_curr_and_target_rgb(
    minimap_rgb: np.ndarray,
    curr_xy,
    target_xy,
    r_curr: int = 5,
    r_tgt: int = 6,
) -> np.ndarray:
    """Overlay a green target dot on a minimap.

    Returns an RGB copy. The ``curr_xy`` argument is accepted for signature
    compatibility with callers but only the target marker is drawn (per the
    visualization spec).
    """
    bgr = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)
    if target_xy is not None:
        cv2.circle(bgr, target_xy, r_tgt, (0, 200, 0), -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_trajectory_rgb(
    base_minimap_rgb: np.ndarray,
    history_points: List[Tuple[int, int]],
    target_xy: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Overlay a yellow trajectory polyline and an optional green target dot.

    Returns an RGB copy of ``base_minimap_rgb``.
    """
    out_bgr = cv2.cvtColor(base_minimap_rgb, cv2.COLOR_RGB2BGR)
    yellow = (0, 255, 255)

    if history_points and len(history_points) >= 2:
        pts = np.array(history_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            out_bgr, [pts], isClosed=False, color=yellow,
            thickness=2, lineType=cv2.LINE_AA,
        )
    if history_points:
        for p in history_points:
            cv2.circle(out_bgr, tuple(map(int, p)), 1, yellow, -1)

    if target_xy is not None:
        cv2.circle(
            out_bgr, (int(target_xy[0]), int(target_xy[1])), 3, (0, 200, 0), -1
        )
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ============================================================================
# Action helpers
# ============================================================================

def get_allowed_actions(action_space: dict) -> List[str]:
    """Return the canonical list of allowed action names."""
    return list(action_space.keys())


def action2signal(action: str, action_space: dict) -> np.ndarray:
    """Map an action string to the continuous control signal ``[move, strafe, look]``.

    Returns a shape-(1, 3) float32 array suitable for ``ActionTuple(continuous=...)``.
    Unrecognized actions resolve to the zero signal (stop) for safety.
    """
    sig = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    if action in ("astar forward turn right", "astar forward right"):
        sig[0, 0] = action_space.get("forward", 0.0) * 0.5
        sig[0, 2] = action_space.get("turn right", 0.0) * 0.25
    elif action in ("astar forward turn left", "astar forward left"):
        sig[0, 0] = action_space.get("forward", 0.0) * 0.5
        sig[0, 2] = action_space.get("turn left", 0.0) * 0.25
    elif action == "astar turn right":
        sig[0, 2] = action_space.get("turn right", 0.0) * 0.25
    elif action == "astar turn left":
        sig[0, 2] = action_space.get("turn left", 0.0) * 0.25
    elif action in ("forward turn right", "forward right"):
        sig[0, 0] = action_space.get("forward", 0.0)
        sig[0, 2] = action_space.get("turn right", 0.0)
    elif action in ("forward turn left", "forward left"):
        sig[0, 0] = action_space.get("forward", 0.0)
        sig[0, 2] = action_space.get("turn left", 0.0)
    elif action in ("forward", "back"):
        sig[0, 0] = action_space.get(action, 0.0)
    elif action in ("strafe right", "strafe left"):
        sig[0, 1] = action_space.get(action, 0.0)
    elif action in ("turn right", "turn left"):
        sig[0, 2] += action_space[action]
    # ``stop`` and unknowns fall through to the zero signal initialized above.
    return sig


def parse_json_action(
    text: str,
    allowed_actions: Optional[Iterable[str]] = None,
    action_space: Optional[dict] = None,
) -> Tuple[str, str]:
    """Extract ``(action, reasoning)`` from an LLM-emitted JSON blob.

    Looks for the first ``{...}`` substring and parses it. Returns
    ``("stop", "")`` on any parse failure or if the action is not in the
    allowed set. When ``allowed_actions`` is None, derives the allowed set
    from ``action_space.keys()``.
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return "stop", ""
    try:
        obj = json.loads(m.group(0))
        a = str(obj.get("action", "")).strip().lower()
        r = str(obj.get("reasoning", "")).strip()
        if allowed_actions is not None:
            allowed = list(allowed_actions)
        elif action_space is not None:
            allowed = get_allowed_actions(action_space)
        else:
            allowed = []
        if a not in allowed:
            a = "stop"
        return a, r
    except Exception:
        return "stop", ""


# ============================================================================
# Geometry (pixel-space distances + Unity↔egocentric heading conversions)
# ============================================================================

def pix_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance between two pixel coordinates."""
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def unity_rotation_to_egocentric_theta(unity_rot_y: float) -> float:
    """Convert Unity's RotY to egocentric theta (0-360, 0°=facing west).

    Egocentric theta convention (agent at origin, +Y down):
        0° = facing WEST, 90° = SOUTH, 180° = EAST, 270° = NORTH.
    """
    return unity_rot_y % 360


def calculate_bearing_angle(
    curr_xy: Tuple[int, int], target_xy: Tuple[int, int]
) -> float:
    """Bearing in image/screen coords: 0°=East (+X), 90°=South (+Y), range [-180, 180]."""
    dx = target_xy[0] - curr_xy[0]
    dy = target_xy[1] - curr_xy[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def get_minimap_heading(unity_rot_y: float, rotation_offset: float = -90) -> float:
    """Convert Unity RotY to minimap heading.

    Minimap heading convention: 0°=North/Up, 90°=East/Right, 180°=South/Down,
    270°=West/Left. The default ``rotation_offset`` of -90 aligns Unity's
    180° with the minimap's 90° (East).
    """
    return (unity_rot_y + rotation_offset) % 360


def cardinal_to_egocentric(cardinal_action: str, agent_heading: float) -> str:
    """Convert a cardinal direction to the egocentric action that achieves it.

    ``agent_heading`` is in degrees (0=N, 90=E, 180=S, 270=W). Unknown
    cardinal actions return ``"stop"``.
    """
    if cardinal_action == "stop":
        return "stop"
    cardinal_angles = {"north": 0, "east": 90, "south": 180, "west": 270}
    if cardinal_action not in cardinal_angles:
        return "stop"

    relative_angle = (cardinal_angles[cardinal_action] - agent_heading) % 360
    if relative_angle > 180:
        relative_angle -= 360

    if -45 <= relative_angle <= 45:
        return "forward"
    if 45 < relative_angle <= 135:
        return "strafe right"
    if relative_angle > 135 or relative_angle < -135:
        return "back"
    return "strafe left"


def convert_cardinal_actions(
    cardinal_actions: List[str], agent_heading: float
) -> List[str]:
    """Apply :func:`cardinal_to_egocentric` over a list, dropping duplicates."""
    out: List[str] = []
    for action in cardinal_actions:
        ego = cardinal_to_egocentric(action, agent_heading)
        if ego not in out:
            out.append(ego)
    return out


# ============================================================================
# GIF assembly (absorbed from former turn2gif.py)
# ============================================================================

def open_and_resize_image(
    img_path: str,
    scale_factor: float,
    resample: int = Image.Resampling.LANCZOS,
) -> Optional[Image.Image]:
    """Open an image, convert to RGB, and resize by ``scale_factor``.

    Returns ``None`` if the file is missing or unreadable, or if the
    resulting dimensions would be non-positive.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        logger.error(f"open_and_resize_image: cannot open {img_path}: {e}")
        return None

    w, h = img.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    if new_w <= 0 or new_h <= 0:
        logger.error(
            f"open_and_resize_image: scale {scale_factor} yields invalid size "
            f"({new_w}x{new_h}) for {img_path}"
        )
        return None
    return img.resize((new_w, new_h), resample)


def pngs_to_gif(
    input_dir: str,
    output_path: str,
    duration: int = 100,
    loop: int = 0,
    scale_factor: float = 1.0,
    resample: int = Image.Resampling.LANCZOS,
) -> None:
    """Assemble all PNGs in ``input_dir`` into an animated GIF, sorted numerically.

    Filenames are sorted by the first integer they contain (so ``frame_2.png``
    comes before ``frame_10.png``). Each frame is quantized to a 256-color
    adaptive palette without dithering to preserve small saturated markers
    (red/green dots).
    """
    def extract_number(filename: str) -> float:
        nums = re.findall(r"\d+", filename)
        return int(nums[0]) if nums else float("inf")

    png_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".png")],
        key=extract_number,
    )
    if not png_files:
        raise ValueError(f"No PNG files found in {input_dir}")

    frames: List[Image.Image] = []
    for file in png_files:
        img = open_and_resize_image(
            os.path.join(input_dir, file), scale_factor, resample
        )
        if img is None:
            continue
        img = img.quantize(
            colors=256,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
        frames.append(img)

    if not frames:
        raise ValueError(f"No valid frames loaded from {input_dir}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False,
        disposal=2,
    )
    logger.info(f"GIF saved at {os.path.abspath(output_path)}")
