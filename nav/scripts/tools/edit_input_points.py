"""Interactively edit input_points.json on cached Unity minimaps.

The first run launches every scene once and caches its minimap plus the
pixel/world projection. Later runs browse, drag, and save points entirely from
that cache. Targets remain canonical minimap pixels, matching the benchmark
input schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import tempfile
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageTk

from nav.config import (
    BEHAVIOR_NAME,
    REPO_ROOT,
    SCENE_CODES,
    SCENE_ID_MAP,
    UNITY_ENGINE_QUALITY_LEVEL,
    resolve_scene_all_path,
)
from nav.harness.coordinates import (
    canonical_to_minimap_coords,
    find_exact_map_bounds,
    minimap_to_canonical_coords,
    resolve_minimap_resolution,
    visual_to_unity_coords,
)
from nav.harness.env_setup import _launch_env
from nav.harness.lighting import add_lighting_args
from nav.harness.motion import add_motion_speed_args
from nav.harness.observations import get_minimap_rgb_for_init, patch_observation_decoding


LOGGER = logging.getLogger("industrynav.input_point_editor")
POINT_ID_RE = re.compile(r"^point([1-9][0-9]*)$")
DEFAULT_INPUT_FILE = REPO_ROOT / "input_points.json"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "input_point_editor"
CACHE_SCHEMA_VERSION = 1
CANONICAL_MAP_SIZE = (862, 512)
PAIR_COLORS = (
    "#00acc1",
    "#fb8c00",
    "#8e24aa",
    "#43a047",
    "#e53935",
    "#3949ab",
    "#00897b",
    "#c0a000",
)
DIRECTION_KEY_YAWS = {
    "Left": 0.0,
    "Up": 90.0,
    "Right": 180.0,
    "Down": 270.0,
}
_OBSERVATION_PATCHED = False


@dataclass
class SceneRuntime:
    """Cached scene data plus optional live Unity objects during calibration."""

    scene_code: str
    env: Any
    env_params: Any
    target_sc: Any
    margin: tuple[float, float, float, float]
    minimap_rgb: np.ndarray
    minimap_size: tuple[int, int]
    pixel_to_world_h: np.ndarray | None = None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.env_params = None
            self.target_sc = None

    @property
    def is_cached(self) -> bool:
        return self.env is None and self.pixel_to_world_h is not None


@dataclass(frozen=True)
class SelectedPair:
    """One confirmed editor selection in benchmark coordinate spaces."""

    start_pixel: tuple[int, int]
    start_world: tuple[float, float]
    target_pixel: tuple[int, int]
    target_world: tuple[float, float]
    direction: float

    @property
    def world_distance(self) -> float:
        return math.hypot(
            self.start_world[0] - self.target_world[0],
            self.start_world[1] - self.target_world[1],
        )


@dataclass(frozen=True)
class ExistingPair:
    """One saved point rendered from its Unity-acknowledged world spawn."""

    point_id: str
    start_pixel: tuple[int, int]
    start_world: tuple[float, float]
    target_pixel: tuple[int, int]
    direction: float


def move_pair_endpoint(
    pair: SelectedPair | ExistingPair,
    endpoint: str,
    destination: tuple[int, int],
    mapped_world: tuple[float, float],
) -> SelectedPair | ExistingPair:
    """Return a pair with one Unity-validated endpoint moved."""
    if endpoint not in {"start", "target"}:
        raise ValueError(f"Unknown pair endpoint: {endpoint!r}")
    if isinstance(pair, ExistingPair):
        return ExistingPair(
            point_id=pair.point_id,
            start_pixel=destination if endpoint == "start" else pair.start_pixel,
            start_world=mapped_world if endpoint == "start" else pair.start_world,
            target_pixel=destination if endpoint == "target" else pair.target_pixel,
            direction=pair.direction,
        )
    return SelectedPair(
        start_pixel=destination if endpoint == "start" else pair.start_pixel,
        start_world=mapped_world if endpoint == "start" else pair.start_world,
        target_pixel=destination if endpoint == "target" else pair.target_pixel,
        target_world=mapped_world if endpoint == "target" else pair.target_world,
        direction=pair.direction,
    )


def set_pair_direction(
    pair: SelectedPair | ExistingPair,
    direction: float,
) -> SelectedPair | ExistingPair:
    """Return a pair with a normalized initial Unity yaw."""
    value = float(direction)
    if not math.isfinite(value):
        raise ValueError("Initial yaw must be finite.")
    return replace(pair, direction=value % 360.0)


def direction_for_arrow_key(keysym: str) -> float | None:
    """Map a Tk arrow key to the current minimap's Unity yaw."""
    return DIRECTION_KEY_YAWS.get(keysym)


def normalize_scene_code(value: str | int) -> str:
    """Return ``sceneN`` for either an integer or a scene-code argument."""
    text = str(value).strip().lower()
    if text.isdigit():
        text = f"scene{int(text)}"
    if text not in SCENE_ID_MAP:
        raise ValueError(
            f"Unknown scene {value!r}; expected a number from 1 to 24."
        )
    return text


def load_input_points(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read and minimally validate the editable point database."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Input-point file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at the top level of {path}.")
    for scene_code, entries in data.items():
        if not isinstance(entries, list):
            raise ValueError(f"{scene_code} must contain a list of point entries.")
    return data


def scene_point_counts(path: Path) -> dict[str, int]:
    """Return a count for every canonical scene, including missing keys."""
    data = load_input_points(path)
    return {scene_code: len(data.get(scene_code, [])) for scene_code in SCENE_CODES}


def next_point_index(entries: Sequence[dict[str, Any]]) -> int:
    """Return the first index after the largest existing ``pointN`` id."""
    indices = []
    for entry in entries:
        match = POINT_ID_RE.fullmatch(str(entry.get("point_id", "")))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def selection_ready_to_save(mode: str, selected_count: int, limit: int) -> bool:
    """Return whether the current new-pair draft satisfies its save mode."""
    if mode == "append":
        return selected_count > 0
    if mode == "replace":
        return limit > 0 and selected_count == limit
    raise ValueError(f"Unknown save mode: {mode!r}")


def heading_endpoint(
    center: tuple[int, int],
    direction_deg: float,
    length: float,
) -> tuple[float, float]:
    """Return a minimap arrow endpoint using Unity's screen yaw convention."""
    radians = math.radians(float(direction_deg) % 360.0)
    return (
        float(center[0]) - math.cos(radians) * float(length),
        float(center[1]) - math.sin(radians) * float(length),
    )


def fit_pixel_to_world_homography(
    pixel_points: Sequence[tuple[float, float]],
    world_points: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Fit the projective minimap-pixel to world-X/Z transform."""
    if len(pixel_points) != len(world_points) or len(pixel_points) < 4:
        raise ValueError("At least four matching pixel/world samples are required.")
    rows: list[list[float]] = []
    values: list[float] = []
    for (px, py), (world_x, world_z) in zip(pixel_points, world_points):
        rows.append([px, py, 1.0, 0.0, 0.0, 0.0, -world_x * px, -world_x * py])
        values.append(world_x)
        rows.append([0.0, 0.0, 0.0, px, py, 1.0, -world_z * px, -world_z * py])
        values.append(world_z)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        rcond=None,
    )
    if rank < 8:
        raise ValueError("Pixel/world calibration samples are degenerate.")
    return np.asarray(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
            [coefficients[6], coefficients[7], 1.0],
        ],
        dtype=np.float64,
    )


def apply_homography(
    matrix: np.ndarray,
    point: tuple[float, float],
) -> tuple[float, float]:
    """Apply a 3x3 projective transform to one 2D point."""
    projected = np.asarray(matrix, dtype=np.float64) @ np.asarray(
        [point[0], point[1], 1.0], dtype=np.float64
    )
    if abs(float(projected[2])) < 1e-12:
        raise ValueError("Projective transform produced a point at infinity.")
    return (
        float(projected[0] / projected[2]),
        float(projected[1] / projected[2]),
    )


def _cache_fingerprint(args: argparse.Namespace, scene_code: str) -> dict[str, Any]:
    client_path = Path(args.file_name).expanduser().resolve()
    try:
        client_mtime_ns = client_path.stat().st_mtime_ns
    except OSError:
        client_mtime_ns = None
    render_settings = {
        key: value
        for key, value in vars(args).items()
        if ("light" in key or "exposure" in key)
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "scene_code": scene_code,
        "scene_id": SCENE_ID_MAP[scene_code],
        "client_path": str(client_path),
        "client_mtime_ns": client_mtime_ns,
        "minimap_size": [int(args.minimap_width), int(args.minimap_height)],
        "quality_level": int(args.quality_level),
        "dynamic_objects": str(args.dynamic_objects),
        "render_settings": render_settings,
    }


def _scene_cache_paths(cache_dir: Path, scene_code: str) -> tuple[Path, Path]:
    scene_code = normalize_scene_code(scene_code)
    return cache_dir / f"{scene_code}.json", cache_dir / f"{scene_code}.png"


def load_scene_cache(
    cache_dir: Path,
    args: argparse.Namespace,
    scene_code: str,
) -> SceneRuntime | None:
    """Load one valid cached minimap and projection, or return ``None``."""
    metadata_path, image_path = _scene_cache_paths(cache_dir, scene_code)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != _cache_fingerprint(args, scene_code):
            return None
        matrix = np.asarray(metadata["pixel_to_world_h"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return None
        with Image.open(image_path) as image:
            minimap_rgb = np.asarray(image.convert("RGB"))
        width, height = int(metadata["minimap_size"][0]), int(metadata["minimap_size"][1])
        if minimap_rgb.shape[:2] != (height, width):
            return None
        margin_values = tuple(float(value) for value in metadata["margin"])
        if len(margin_values) != 4:
            return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return SceneRuntime(
        scene_code=scene_code,
        env=None,
        env_params=None,
        target_sc=None,
        margin=margin_values,
        minimap_rgb=minimap_rgb,
        minimap_size=(width, height),
        pixel_to_world_h=matrix,
    )


def missing_scene_caches(
    cache_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Return scenes without a valid image and pixel/world projection cache."""
    return [
        scene_code
        for scene_code in SCENE_CODES
        if load_scene_cache(cache_dir, args, scene_code) is None
    ]


def save_scene_cache(
    cache_dir: Path,
    args: argparse.Namespace,
    runtime: SceneRuntime,
) -> None:
    """Atomically save one minimap image and its calibrated projection."""
    if runtime.pixel_to_world_h is None:
        raise ValueError("Cannot cache a scene without pixel/world calibration.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, image_path = _scene_cache_paths(cache_dir, runtime.scene_code)
    metadata = {
        "fingerprint": _cache_fingerprint(args, runtime.scene_code),
        "minimap_size": list(runtime.minimap_size),
        "margin": list(runtime.margin),
        "pixel_to_world_h": runtime.pixel_to_world_h.tolist(),
    }

    image_fd, image_temp = tempfile.mkstemp(
        dir=cache_dir, prefix=f".{runtime.scene_code}.", suffix=".png"
    )
    os.close(image_fd)
    metadata_fd, metadata_temp = tempfile.mkstemp(
        dir=cache_dir, prefix=f".{runtime.scene_code}.", suffix=".json"
    )
    try:
        Image.fromarray(runtime.minimap_rgb).save(image_temp, format="PNG")
        metadata_handle = os.fdopen(metadata_fd, "w", encoding="utf-8")
        metadata_fd = -1
        with metadata_handle as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(image_temp, image_path)
        os.replace(metadata_temp, metadata_path)
    finally:
        if metadata_fd >= 0:
            os.close(metadata_fd)
        for temp_path in (image_temp, metadata_temp):
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def pair_to_entry(selection: SelectedPair, point_index: int) -> dict[str, Any]:
    """Serialize a confirmed pair using the benchmark's mixed coordinate schema."""
    return {
        "point_id": f"point{point_index}",
        "start": {
            "x": round(float(selection.start_world[0]), 2),
            "z": round(float(selection.start_world[1]), 2),
            "direction": float(selection.direction),
        },
        "target": {
            "x": float(selection.target_pixel[0]),
            "y": float(selection.target_pixel[1]),
        },
    }


def existing_pair_to_entry(selection: ExistingPair) -> dict[str, Any]:
    """Serialize an edited saved point without changing its point id."""
    return {
        "point_id": selection.point_id,
        "start": {
            "x": round(float(selection.start_world[0]), 2),
            "z": round(float(selection.start_world[1]), 2),
            "direction": float(selection.direction),
        },
        "target": {
            "x": float(selection.target_pixel[0]),
            "y": float(selection.target_pixel[1]),
        },
    }


def save_scene_entries(
    path: Path,
    scene_code: str,
    entries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atomically replace one scene while preserving every other scene."""
    scene_code = normalize_scene_code(scene_code)
    data = load_input_points(path)
    updated_entries = list(entries)
    data[scene_code] = updated_entries

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return updated_entries


def save_scene_pairs(
    path: Path,
    scene_code: str,
    selections: Sequence[SelectedPair],
    mode: str,
) -> list[dict[str, Any]]:
    """Atomically replace or append one scene's entries in the latest file."""
    scene_code = normalize_scene_code(scene_code)
    if mode not in {"replace", "append"}:
        raise ValueError("Save mode must be 'replace' or 'append'.")
    if not selections:
        raise ValueError("At least one confirmed pair is required before saving.")

    data = load_input_points(path)
    existing = list(data.get(scene_code, []))
    start_index = 1 if mode == "replace" else next_point_index(existing)
    new_entries = [
        pair_to_entry(selection, start_index + offset)
        for offset, selection in enumerate(selections)
    ]
    updated_entries = new_entries if mode == "replace" else existing + new_entries
    return save_scene_entries(path, scene_code, updated_entries)


def _runtime_args(base_args: argparse.Namespace, scene_code: str) -> argparse.Namespace:
    """Build a fresh namespace so per-run lighting/speed caches never leak."""
    values = {
        key: value
        for key, value in vars(base_args).items()
        if not key.startswith("_")
    }
    values.update(
        scene_id=SCENE_ID_MAP[scene_code],
        scene_name=scene_code,
        point_id="point_editor",
        seed_id="0",
        init_world_x=None,
        init_world_z=None,
        init_curr_direction=0.0,
        frame_save_dir=str(Path(base_args.log_dir) / scene_code),
    )
    return argparse.Namespace(**values)


def _ensure_observation_patch() -> None:
    """Install the ML-Agents observation shim once per editor process."""
    global _OBSERVATION_PATCHED
    if not _OBSERVATION_PATCHED:
        patch_observation_decoding()
        _OBSERVATION_PATCHED = True


def launch_scene_runtime(
    base_args: argparse.Namespace,
    scene_code: str,
    logger: logging.Logger = LOGGER,
) -> SceneRuntime:
    """Launch one Unity scene and return a displayable minimap."""
    args = _runtime_args(base_args, scene_code)
    _ensure_observation_patch()
    env, env_params, _bounds_sc, target_sc = _launch_env(args, logger)
    try:
        minimap_rgb = get_minimap_rgb_for_init(env, BEHAVIOR_NAME)
        if minimap_rgb is None:
            raise RuntimeError("Unity did not produce a usable minimap frame.")
        height, width = minimap_rgb.shape[:2]
        expected = (int(args.minimap_width), int(args.minimap_height))
        if (width, height) != expected:
            raise RuntimeError(
                "Unexpected minimap resolution: "
                f"expected {expected[0]}x{expected[1]}, got {width}x{height}."
            )
        margin = find_exact_map_bounds(logger, minimap_rgb)
        if margin is None:
            raise RuntimeError("Could not detect the rendered minimap bounds.")
        env_params.set_float_parameter("minimap_px_width", float(width))
        env_params.set_float_parameter("minimap_px_height", float(height))
        return SceneRuntime(
            scene_code=scene_code,
            env=env,
            env_params=env_params,
            target_sc=target_sc,
            margin=margin,
            minimap_rgb=minimap_rgb,
            minimap_size=(width, height),
        )
    except Exception:
        env.close()
        raise


def calibrate_scene_projection(runtime: SceneRuntime) -> np.ndarray:
    """Sample Unity's mapper once and retain an offline projective transform."""
    if runtime.env is None or runtime.env_params is None or runtime.target_sc is None:
        raise RuntimeError("A live Unity runtime is required for calibration.")
    width, height = runtime.minimap_size
    xs = (0, (width - 1) // 2, width - 1)
    ys = (0, (height - 1) // 2, height - 1)
    pixel_points: list[tuple[float, float]] = []
    world_points: list[tuple[float, float]] = []
    for pixel_y in ys:
        for pixel_x in xs:
            runtime.target_sc.last_target_pixel = None
            runtime.target_sc.last_target_world = None
            runtime.env_params.set_float_parameter("target_px", float(pixel_x))
            runtime.env_params.set_float_parameter("target_py", float(pixel_y))
            runtime.env.reset()
            world = runtime.target_sc.last_target_world
            if world is None:
                raise RuntimeError(
                    "Unity did not return a target mapping during cache calibration."
                )
            pixel_points.append((float(pixel_x), float(pixel_y)))
            world_points.append((float(world[0]), float(world[1])))
    matrix = fit_pixel_to_world_homography(pixel_points, world_points)
    errors = [
        math.hypot(
            predicted[0] - expected[0],
            predicted[1] - expected[1],
        )
        for predicted, expected in (
            (apply_homography(matrix, pixel), world)
            for pixel, world in zip(pixel_points, world_points)
        )
    ]
    max_error = max(errors, default=0.0)
    if max_error > 0.02:
        raise RuntimeError(
            "Minimap projection is not stable enough to cache: "
            f"maximum calibration error is {max_error:.4f} m."
        )
    runtime.pixel_to_world_h = matrix
    LOGGER.info(
        "Cached %s pixel/world projection (max fit error %.6f m)",
        runtime.scene_code,
        max_error,
    )
    return matrix


def build_scene_caches(
    cache_dir: Path,
    args: argparse.Namespace,
    scene_codes: Sequence[str],
    logger: logging.Logger = LOGGER,
) -> list[str]:
    """Launch each scene once and persist its minimap plus projection."""
    built: list[str] = []
    total = len(scene_codes)
    for index, scene_code in enumerate(scene_codes, start=1):
        logger.info("Building minimap cache %d/%d: %s", index, total, scene_code)
        runtime = launch_scene_runtime(args, scene_code, logger)
        try:
            calibrate_scene_projection(runtime)
            save_scene_cache(cache_dir, args, runtime)
            built.append(scene_code)
        finally:
            runtime.close()
    return built


def canonical_to_unity_pixel(
    runtime: SceneRuntime,
    canonical_pixel: tuple[int, int],
) -> tuple[int, int]:
    """Convert a canonical editor point to the integer mapper pixel Unity uses."""
    runtime_pixel = canonical_to_minimap_coords(
        canonical_pixel, runtime.minimap_size
    )
    pixel_x, pixel_y = visual_to_unity_coords(
        runtime.margin,
        runtime_pixel[0],
        runtime_pixel[1],
        map_size=runtime.minimap_size,
    )
    width, height = runtime.minimap_size
    return (
        int(np.clip(round(pixel_x), 0, width - 1)),
        int(np.clip(round(pixel_y), 0, height - 1)),
    )


def cached_pixel_to_world(
    runtime: SceneRuntime,
    canonical_pixel: tuple[int, int],
) -> tuple[float, float]:
    """Resolve a canonical point using the scene's cached Unity calibration."""
    if runtime.pixel_to_world_h is None:
        raise RuntimeError("The scene cache has no pixel/world calibration.")
    unity_pixel = canonical_to_unity_pixel(runtime, canonical_pixel)
    return apply_homography(runtime.pixel_to_world_h, unity_pixel)


def map_start_pixel_through_unity(
    runtime: SceneRuntime,
    start_pixel: tuple[int, int],
    direction: float,
) -> tuple[float, float]:
    """Map one canonical start pixel using cached or live Unity calibration."""
    if runtime.pixel_to_world_h is not None:
        return cached_pixel_to_world(runtime, start_pixel)
    if runtime.env is None or runtime.env_params is None or runtime.target_sc is None:
        raise RuntimeError("No cached calibration or live Unity runtime is available.")
    spawn_upx, spawn_upy = canonical_to_unity_pixel(runtime, start_pixel)
    runtime.target_sc.last_spawn_pixel = None
    runtime.target_sc.last_spawn_world = None
    runtime.env_params.set_float_parameter("spawn_px", float(spawn_upx))
    runtime.env_params.set_float_parameter("spawn_py", float(spawn_upy))
    runtime.env_params.set_float_parameter("spawn_wy", 0.5)
    runtime.env_params.set_float_parameter("spawn_rot", float(direction))
    runtime.env.reset()
    start_world = runtime.target_sc.last_spawn_world
    if start_world is None:
        raise RuntimeError(
            "Unity did not acknowledge the selected start pixel. Choose a "
            "navigable floor location and try again."
        )
    return float(start_world[0]), float(start_world[1])


def map_target_pixel_through_unity(
    runtime: SceneRuntime,
    target_pixel: tuple[int, int],
) -> tuple[float, float]:
    """Map one canonical target pixel using cached or live Unity calibration."""
    if runtime.pixel_to_world_h is not None:
        return cached_pixel_to_world(runtime, target_pixel)
    if runtime.env is None or runtime.env_params is None or runtime.target_sc is None:
        raise RuntimeError("No cached calibration or live Unity runtime is available.")
    target_upx, target_upy = canonical_to_unity_pixel(runtime, target_pixel)
    runtime.target_sc.last_target_pixel = None
    runtime.target_sc.last_target_world = None
    runtime.env_params.set_float_parameter("target_px", float(target_upx))
    runtime.env_params.set_float_parameter("target_py", float(target_upy))
    runtime.env.reset()
    target_world = runtime.target_sc.last_target_world
    if target_world is None:
        raise RuntimeError(
            "Unity did not acknowledge the selected target pixel. Choose a "
            "navigable floor location and try again."
        )
    return float(target_world[0]), float(target_world[1])


def map_pair_through_unity(
    runtime: SceneRuntime,
    start_pixel: tuple[int, int],
    target_pixel: tuple[int, int],
    direction: float,
) -> SelectedPair:
    """Map one canonical pair through the cached Unity projection."""
    start_world = map_start_pixel_through_unity(runtime, start_pixel, direction)
    target_world = map_target_pixel_through_unity(runtime, target_pixel)

    return SelectedPair(
        start_pixel=start_pixel,
        start_world=start_world,
        target_pixel=target_pixel,
        target_world=target_world,
        direction=float(direction),
    )


def _unity_pixel_to_canonical(
    runtime: SceneRuntime,
    unity_pixel: tuple[int, int],
) -> tuple[int, int]:
    """Invert the editor's letterbox mapping for a Unity pixel ack."""
    min_x, max_x, min_y, max_y = runtime.margin
    map_width, map_height = runtime.minimap_size
    visual_x = min_x + (float(unity_pixel[0]) / map_width) * (max_x - min_x)
    visual_y = min_y + (float(unity_pixel[1]) / map_height) * (max_y - min_y)
    canonical_x, canonical_y = minimap_to_canonical_coords(
        (visual_x, visual_y),
        runtime.minimap_size,
    )
    return (
        int(np.clip(round(canonical_x), 0, CANONICAL_MAP_SIZE[0] - 1)),
        int(np.clip(round(canonical_y), 0, CANONICAL_MAP_SIZE[1] - 1)),
    )


def cached_world_to_canonical(
    runtime: SceneRuntime,
    world: tuple[float, float],
) -> tuple[int, int]:
    """Project world X/Z back onto the canonical cached minimap."""
    if runtime.pixel_to_world_h is None:
        raise RuntimeError("The scene cache has no pixel/world calibration.")
    world_to_pixel = np.linalg.inv(runtime.pixel_to_world_h)
    unity_x, unity_y = apply_homography(world_to_pixel, world)
    width, height = runtime.minimap_size
    unity_pixel = (
        int(np.clip(round(unity_x), 0, width - 1)),
        int(np.clip(round(unity_y), 0, height - 1)),
    )
    return _unity_pixel_to_canonical(runtime, unity_pixel)


def map_existing_pairs_through_unity(
    runtime: SceneRuntime,
    entries: Sequence[dict[str, Any]],
) -> list[ExistingPair]:
    """Resolve saved world starts through the cached or live Unity projection."""
    resolved: list[ExistingPair] = []
    for entry in entries:
        point_id = str(entry.get("point_id", "point?"))
        try:
            start = entry["start"]
            target = entry["target"]
            world_x = float(start["x"])
            world_z = float(start["z"])
            direction = float(start["direction"])
            target_pixel = (int(round(float(target["x"]))), int(round(float(target["y"]))))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid saved entry {point_id!r}: {entry!r}") from exc

        if runtime.pixel_to_world_h is not None:
            start_pixel = cached_world_to_canonical(runtime, (world_x, world_z))
        else:
            if runtime.env is None or runtime.env_params is None or runtime.target_sc is None:
                raise RuntimeError("No cached calibration or live Unity runtime is available.")
            runtime.target_sc.last_spawn_pixel = None
            runtime.target_sc.last_spawn_world = None
            runtime.env_params.set_float_parameter("spawn_x", world_x)
            runtime.env_params.set_float_parameter("spawn_y", 0.5)
            runtime.env_params.set_float_parameter("spawn_z", world_z)
            runtime.env_params.set_float_parameter("spawn_rot", direction)
            runtime.env.reset()
            spawn_pixel = runtime.target_sc.last_spawn_pixel
            if spawn_pixel is None:
                raise RuntimeError(
                    f"Unity did not return WorldToPixel mapping for {point_id}."
                )
            start_pixel = _unity_pixel_to_canonical(runtime, spawn_pixel)
        resolved.append(
            ExistingPair(
                point_id=point_id,
                start_pixel=start_pixel,
                start_world=(world_x, world_z),
                target_pixel=target_pixel,
                direction=direction,
            )
        )
    return resolved


class InputPointEditor:
    """Tk application coordinating point selection and serialized Unity calls."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.input_file = Path(args.input_file).expanduser().resolve()
        self.cache_dir = Path(args.cache_dir).expanduser().resolve()
        self.runtime: SceneRuntime | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unity-editor")
        self.busy = False
        self.closing = False
        self.cache_ready = False
        self.auto_load_pending = bool(args.auto_load)
        self.existing_pairs: list[ExistingPair] = []
        self.original_existing_pairs: list[ExistingPair] = []
        self.existing_history: list[list[ExistingPair]] = []
        self.existing_dirty = False
        self.selected_pairs: list[SelectedPair] = []
        self.selected_history: list[list[SelectedPair]] = []
        self.pending_start: tuple[int, int] | None = None
        self.pending_target: tuple[int, int] | None = None
        self.drag_ref: tuple[str, int, str] | None = None
        self.drag_origin: tuple[int, int] | None = None
        self.drag_preview: tuple[int, int] | None = None
        self.new_pair_drag_origin: tuple[int, int] | None = None
        self.map_photo: ImageTk.PhotoImage | None = None

        self.scene_var = tk.StringVar(value=args.scene)
        self.pair_count_var = tk.StringVar(value=str(args.pairs))
        self.mode_var = tk.StringVar(value=args.mode)
        self.direction_var = tk.StringVar(value=str(args.direction))
        self.status_var = tk.StringVar(value="Checking the 24-scene minimap cache...")
        self.coordinate_var = tk.StringVar(value="Map coordinate: -")
        self.progress_var = tk.StringVar(value="Confirmed pairs: 0")

        self._build_ui()
        self.pair_count_var.trace_add("write", self._on_selection_setting_changed)
        self.mode_var.trace_add("write", self._on_selection_setting_changed)
        self.direction_var.trace_add("write", self._on_direction_changed)
        self.refresh_scene_counts()
        self._refresh_point_table()
        self._refresh_controls()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Command-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-s>", self._save_shortcut)
        self.root.bind_all("<Command-s>", self._save_shortcut)
        for key in DIRECTION_KEY_YAWS:
            self.root.bind(f"<{key}>", self._on_direction_key, add="+")
        self.root.after(100, self.cache_all_scenes)

    def _build_ui(self) -> None:
        self.root.title("IndustryNav Input Point Editor")
        self.root.geometry("1300x760")
        self.root.minsize(1250, 680)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.Frame(outer, width=355)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Scenes", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.scene_tree = ttk.Treeview(
            controls,
            columns=("id", "points"),
            show="tree headings",
            height=12,
            selectmode="browse",
        )
        self.scene_tree.heading("#0", text="Scene")
        self.scene_tree.heading("id", text="ID")
        self.scene_tree.heading("points", text="Points")
        self.scene_tree.column("#0", width=105, stretch=True)
        self.scene_tree.column("id", width=45, anchor=tk.CENTER, stretch=False)
        self.scene_tree.column("points", width=60, anchor=tk.CENTER, stretch=False)
        self.scene_tree.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.scene_tree.bind("<<TreeviewSelect>>", self._on_scene_selected)
        self.scene_tree.bind("<Double-1>", lambda _event: self.load_scene())

        settings = ttk.LabelFrame(controls, text="Selection", padding=8)
        settings.grid(row=2, column=0, sticky="ew")
        settings.columnconfigure(1, weight=1)
        self.pair_count_label = ttk.Label(settings, text="Pairs to replace")
        self.pair_count_label.grid(row=0, column=0, sticky="w")
        self.pair_spin = ttk.Spinbox(
            settings,
            from_=1,
            to=100,
            textvariable=self.pair_count_var,
            width=8,
        )
        self.pair_spin.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(settings, text="Initial yaw").grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.direction_combo = ttk.Combobox(
            settings,
            textvariable=self.direction_var,
            values=("0", "45", "90", "135", "180", "225", "270", "315"),
        )
        self.direction_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))

        mode_frame = ttk.Frame(settings)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.replace_radio = ttk.Radiobutton(
            mode_frame, text="Replace scene", value="replace", variable=self.mode_var
        )
        self.replace_radio.pack(side=tk.LEFT)
        self.append_radio = ttk.Radiobutton(
            mode_frame, text="Append", value="append", variable=self.mode_var
        )
        self.append_radio.pack(side=tk.LEFT, padx=(10, 0))

        self.load_button = ttk.Button(
            controls, text="Open cached scene", command=self.load_scene
        )
        self.load_button.grid(row=3, column=0, sticky="ew", pady=(10, 4))

        actions = ttk.Frame(controls)
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.undo_button = ttk.Button(actions, text="Undo", command=self.undo)
        self.undo_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.confirm_button = ttk.Button(actions, text="Confirm pair", command=self.confirm_pair)
        self.confirm_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.save_button = ttk.Button(controls, text="Save new pairs", command=self.save)
        self.save_button.grid(row=5, column=0, sticky="ew", pady=(4, 8))

        ttk.Label(controls, textvariable=self.progress_var).grid(row=6, column=0, sticky="w")
        point_style = ttk.Style(self.root)
        point_style.configure("Point.Treeview", rowheight=28)
        point_style.map(
            "Point.Treeview",
            background=[("selected", "#0b6e99")],
            foreground=[("selected", "#ffffff")],
        )
        self.point_tree = ttk.Treeview(
            controls,
            columns=("start", "target", "yaw", "distance"),
            show="tree headings",
            height=8,
            style="Point.Treeview",
        )
        self.point_tree.heading("#0", text="Point")
        self.point_tree.heading("start", text="Start world")
        self.point_tree.heading("target", text="Target px")
        self.point_tree.heading("yaw", text="Yaw")
        self.point_tree.heading("distance", text="Dist")
        self.point_tree.column("#0", width=58, stretch=False)
        self.point_tree.column("start", width=90, stretch=True)
        self.point_tree.column("target", width=72, stretch=False)
        self.point_tree.column("yaw", width=44, anchor=tk.CENTER, stretch=False)
        self.point_tree.column("distance", width=48, stretch=False)
        self.point_tree.grid(row=7, column=0, sticky="ew", pady=(4, 8))
        self.point_tree.tag_configure("existing", foreground="#20252b")
        self.point_tree.bind("<<TreeviewSelect>>", self._on_point_selected)
        for key in DIRECTION_KEY_YAWS:
            self.point_tree.bind(f"<{key}>", self._on_direction_key)

        existing_actions = ttk.Frame(controls)
        existing_actions.grid(row=8, column=0, sticky="ew")
        existing_actions.columnconfigure(0, weight=1)
        existing_actions.columnconfigure(1, weight=1)
        self.apply_yaw_button = ttk.Button(
            existing_actions,
            text="Apply yaw",
            command=self.apply_yaw_to_selected,
        )
        self.apply_yaw_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.delete_existing_button = ttk.Button(
            existing_actions,
            text="Delete selected",
            command=self.delete_selected_existing,
        )
        self.delete_existing_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.save_existing_button = ttk.Button(
            controls,
            text="Save existing changes",
            command=self.save_existing_changes,
        )
        self.save_existing_button.grid(row=9, column=0, sticky="ew", pady=(4, 8))

        ttk.Separator(controls).grid(row=10, column=0, sticky="ew", pady=4)
        ttk.Label(controls, text="Drag empty map: start to target.").grid(
            row=11, column=0, sticky="w"
        )
        ttk.Label(controls, text="Drag a confirmed S/T marker to adjust it.").grid(
            row=12, column=0, sticky="w"
        )
        ttk.Label(controls, text="Arrow keys set the selected start yaw.").grid(
            row=13, column=0, sticky="w"
        )
        ttk.Label(controls, text="Right click or Cmd/Ctrl+Z: undo.").grid(
            row=14, column=0, sticky="w"
        )

        map_frame = ttk.Frame(outer)
        map_frame.grid(row=0, column=1, sticky="nsew")
        map_frame.columnconfigure(0, weight=1)
        map_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            map_frame,
            width=CANONICAL_MAP_SIZE[0],
            height=CANONICAL_MAP_SIZE[1],
            background="#202124",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.grid(row=0, column=0, sticky="n")
        self.canvas.bind("<ButtonPress-1>", self._on_map_press)
        self.canvas.bind("<B1-Motion>", self._on_map_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_map_release)
        self.canvas.bind("<Button-3>", lambda _event: self.undo())
        self.canvas.bind("<Motion>", self._on_map_motion)
        for key in DIRECTION_KEY_YAWS:
            self.canvas.bind(f"<{key}>", self._on_direction_key)

        footer = ttk.Frame(map_frame)
        footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, wraplength=750).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(footer, textvariable=self.coordinate_var).grid(row=0, column=1, sticky="e")

    def _point_limit(self) -> int:
        try:
            value = int(self.pair_count_var.get())
        except ValueError as exc:
            raise ValueError("Pair count must be an integer.") from exc
        if value < 1 or value > 100:
            raise ValueError("Pair count must be between 1 and 100.")
        return value

    def _direction(self) -> float:
        try:
            value = float(self.direction_var.get())
        except ValueError as exc:
            raise ValueError("Initial yaw must be a number in degrees.") from exc
        if not math.isfinite(value):
            raise ValueError("Initial yaw must be finite.")
        return value % 360.0

    def _selected_scene(self) -> str:
        selection = self.scene_tree.selection()
        value = selection[0] if selection else self.scene_var.get()
        return normalize_scene_code(value)

    def refresh_scene_counts(self) -> None:
        counts = scene_point_counts(self.input_file)
        selected = self.scene_var.get()
        for item in self.scene_tree.get_children():
            self.scene_tree.delete(item)
        for scene_code in SCENE_CODES:
            self.scene_tree.insert(
                "",
                tk.END,
                iid=scene_code,
                text=scene_code,
                values=(SCENE_ID_MAP[scene_code], counts[scene_code]),
            )
        if selected in SCENE_ID_MAP:
            self.scene_tree.selection_set(selected)
            self.scene_tree.see(selected)

    def _on_scene_selected(self, _event=None) -> None:
        selection = self.scene_tree.selection()
        if selection:
            self.scene_var.set(selection[0])
            if self.runtime is not None and self.runtime.scene_code != selection[0]:
                self.status_var.set(
                    f"{selection[0]} is selected. Click Open cached scene before editing it."
                )
            self._refresh_point_table()
            self._refresh_controls()

    def _on_point_selected(self, _event=None) -> None:
        reference = self._selected_pair_reference()
        if reference is not None and not self.busy:
            kind, index = reference
            if kind == "existing":
                label = self.existing_pairs[index].point_id
                pair = self.existing_pairs[index]
            else:
                label = self.point_tree.item(f"new:{index}", "text")
                pair = self.selected_pairs[index]
            self.direction_var.set(f"{pair.direction:g}")
            self.status_var.set(
                f"{label} selected (yaw={pair.direction:g} deg). "
                "Use arrow keys to change yaw or drag its S/T marker."
            )
        self._draw_overlays()
        self._refresh_controls()

    def _on_selection_setting_changed(self, *_args) -> None:
        self.root.after_idle(self._refresh_point_table)
        self.root.after_idle(self._refresh_controls)

    def _on_direction_changed(self, *_args) -> None:
        self.root.after_idle(self._draw_overlays)

    def _on_direction_key(self, event) -> str | None:
        if event.widget in {self.scene_tree, self.pair_spin}:
            return None
        direction = direction_for_arrow_key(event.keysym)
        if (
            direction is None
            or self.busy
            or self.runtime is None
            or self.runtime.scene_code != self.scene_var.get()
        ):
            return None

        return self._set_direction_from_key(direction)

    def _set_direction_from_key(self, direction: float) -> str:
        """Apply a cardinal yaw to the active pair or the next pair draft."""
        direction = float(direction) % 360.0

        if self.pending_start is not None:
            self.direction_var.set(f"{direction:g}")
            self.status_var.set(
                f"Start yaw set to {direction:g} deg. "
                "Select a target if needed, then click Confirm pair."
            )
            self._draw_overlays()
            self._refresh_controls()
            return "break"

        reference = self._selected_pair_reference()
        if reference is None:
            self.direction_var.set(f"{direction:g}")
            self.status_var.set(
                f"Next start yaw set to {direction:g} deg. Select a start and target."
            )
            self._draw_overlays()
            self._refresh_controls()
            return "break"
        kind, index = reference
        if kind == "existing":
            if self.selected_pairs:
                self.status_var.set(
                    "Save or undo new pairs before changing a saved pair yaw."
                )
                return "break"
            pair = self.existing_pairs[index]
            if pair.direction != direction:
                self.existing_history.append(list(self.existing_pairs))
                self.existing_pairs[index] = set_pair_direction(pair, direction)
                self._update_existing_dirty()
            label = pair.point_id
        else:
            if self.existing_dirty:
                self.status_var.set(
                    "Save or undo existing-point changes before changing a new pair yaw."
                )
                return "break"
            pair = self.selected_pairs[index]
            if pair.direction != direction:
                self.selected_history.append(list(self.selected_pairs))
                self.selected_pairs[index] = set_pair_direction(pair, direction)
            label = self.point_tree.item(f"new:{index}", "text")
        self.direction_var.set(f"{direction:g}")
        self._refresh_point_table()
        self._draw_overlays()
        self.status_var.set(
            f"Set {label} initial yaw to {direction:g} deg. Save the point changes."
        )
        self._refresh_controls()
        return "break"

    def _discard_unsaved(self) -> bool:
        if (
            not self.selected_pairs
            and self.pending_start is None
            and not self.existing_dirty
            and self.drag_ref is None
        ):
            return True
        return messagebox.askyesno(
            "Discard selections?",
            "Loading another scene will discard all unsaved point changes.",
        )

    def cache_all_scenes(self) -> None:
        """Generate every missing/stale scene cache once at editor startup."""
        if self.busy or not self._discard_unsaved():
            return
        scene_codes = (
            list(SCENE_CODES)
            if self.args.refresh_cache
            else missing_scene_caches(self.cache_dir, self.args)
        )
        if not scene_codes:
            self.cache_ready = True
            self.status_var.set(
                "Scene cache ready: 24/24. Select a scene and open its cached minimap."
            )
            self._refresh_controls()
            if self.auto_load_pending:
                self.auto_load_pending = False
                self.root.after_idle(self.load_scene)
            return

        previous_runtime = self.runtime
        self.runtime = None
        self.cache_ready = False
        self.existing_pairs.clear()
        self.original_existing_pairs.clear()
        self.existing_history.clear()
        self.existing_dirty = False
        self.selected_pairs.clear()
        self.selected_history.clear()
        self.pending_start = None
        self.pending_target = None
        self.drag_ref = None
        self.drag_origin = None
        self.drag_preview = None
        self.new_pair_drag_origin = None
        self._refresh_point_table()
        self.canvas.delete("all")

        def task() -> tuple[int, int]:
            if previous_runtime is not None:
                previous_runtime.close()
            built = build_scene_caches(
                self.cache_dir,
                self.args,
                scene_codes,
            )
            return len(built), len(SCENE_CODES)

        def success(result: tuple[int, int]) -> None:
            generated_count, total_count = result
            self.cache_ready = True
            self.args.refresh_cache = False
            self.status_var.set(
                f"Scene cache ready: {total_count}/{total_count}; "
                f"generated {generated_count}, reused {total_count - generated_count}."
            )
            self._refresh_controls()
            if self.auto_load_pending:
                self.auto_load_pending = False
                self.root.after_idle(self.load_scene)

        self._run_background(
            f"Building {len(scene_codes)} missing/stale minimap caches. "
            "Each scene will launch once...",
            task,
            success,
        )

    def load_scene(self) -> None:
        if self.busy or not self.cache_ready or not self._discard_unsaved():
            return
        try:
            scene_code = self._selected_scene()
            self._point_limit()
        except ValueError as exc:
            messagebox.showerror("Invalid selection", str(exc))
            return

        previous_runtime = self.runtime
        self.runtime = None
        self.existing_pairs.clear()
        self.original_existing_pairs.clear()
        self.existing_history.clear()
        self.existing_dirty = False
        self.selected_pairs.clear()
        self.selected_history.clear()
        self.pending_start = None
        self.pending_target = None
        self.drag_ref = None
        self.drag_origin = None
        self.drag_preview = None
        self.new_pair_drag_origin = None
        self._refresh_point_table()
        self.canvas.delete("all")

        def task() -> tuple[SceneRuntime, list[ExistingPair]]:
            if previous_runtime is not None:
                previous_runtime.close()
            runtime = load_scene_cache(self.cache_dir, self.args, scene_code)
            if runtime is None:
                raise RuntimeError(
                    f"The {scene_code} cache is missing or stale. Restart the editor "
                    "with --refresh-cache to rebuild all scene caches."
                )
            try:
                entries = load_input_points(self.input_file).get(scene_code, [])
                existing = map_existing_pairs_through_unity(runtime, entries)
                return runtime, existing
            except Exception:
                runtime.close()
                raise

        def success(result: tuple[SceneRuntime, list[ExistingPair]]) -> None:
            runtime, existing = result
            self.runtime = runtime
            self.existing_pairs = existing
            self.original_existing_pairs = list(existing)
            self.existing_history.clear()
            self.existing_dirty = False
            self._show_minimap(runtime.minimap_rgb)
            self._refresh_point_table()
            self.status_var.set(
                f"Opened cached {scene_code} "
                f"(scene_id={SCENE_ID_MAP[scene_code]}). "
                f"Showing {len(existing)} saved points. Drag an S/T marker or click a new start."
            )
            self._refresh_controls()

        self._run_background(f"Loading cached {scene_code}...", task, success)

    def _run_background(
        self,
        status: str,
        task: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        self.busy = True
        self.status_var.set(status)
        self._refresh_controls()
        future = self.executor.submit(task)
        self._poll_future(future, on_success)

    def _poll_future(self, future: Future, on_success: Callable[[Any], None]) -> None:
        if self.closing:
            return
        if not future.done():
            self.root.after(100, self._poll_future, future, on_success)
            return
        self.busy = False
        try:
            result = future.result()
        except Exception as exc:
            LOGGER.exception("Input-point editor background operation failed")
            self.status_var.set(f"Error: {exc}")
            messagebox.showerror("Operation failed", str(exc))
            self._refresh_controls()
            return
        on_success(result)

    def _show_minimap(self, minimap_rgb: np.ndarray) -> None:
        image = Image.fromarray(minimap_rgb)
        self.map_photo = ImageTk.PhotoImage(image=image)
        width, height = image.size
        self.canvas.configure(width=width, height=height, scrollregion=(0, 0, width, height))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo, tags=("base",))
        self._draw_overlays()

    def _runtime_to_canonical(self, x: int, y: int) -> tuple[int, int]:
        if self.runtime is None:
            raise RuntimeError("No scene is loaded.")
        canonical = minimap_to_canonical_coords((x, y), self.runtime.minimap_size)
        return (
            int(np.clip(round(canonical[0]), 0, CANONICAL_MAP_SIZE[0] - 1)),
            int(np.clip(round(canonical[1]), 0, CANONICAL_MAP_SIZE[1] - 1)),
        )

    def _canonical_to_runtime(self, point: tuple[int, int]) -> tuple[int, int]:
        if self.runtime is None:
            return point
        return canonical_to_minimap_coords(point, self.runtime.minimap_size)

    def _on_map_motion(self, event) -> None:
        if self.runtime is None:
            return
        point = self._runtime_to_canonical(event.x, event.y)
        self.coordinate_var.set(f"Map coordinate: ({point[0]}, {point[1]})")
        if self.drag_ref is not None:
            self.canvas.configure(cursor="fleur")
        elif self.new_pair_drag_origin is not None:
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(
                cursor="hand2" if self._hit_test_marker(event.x, event.y) else "crosshair"
            )

    def _hit_test_marker(self, x: int, y: int) -> tuple[str, int, str] | None:
        """Return the nearest draggable start/target handle under the pointer."""
        if self.runtime is None or self.pending_start is not None:
            return None
        candidates: list[tuple[tuple[str, int, str], tuple[int, int]]] = []
        if self.selected_pairs and not self.existing_dirty:
            for index, pair in enumerate(self.selected_pairs):
                candidates.extend(
                    [
                        (("new", index, "start"), pair.start_pixel),
                        (("new", index, "target"), pair.target_pixel),
                    ]
                )
        elif not self.selected_pairs:
            for index, pair in enumerate(self.existing_pairs):
                candidates.extend(
                    [
                        (("existing", index, "start"), pair.start_pixel),
                        (("existing", index, "target"), pair.target_pixel),
                    ]
                )
        nearest: tuple[str, int, str] | None = None
        nearest_distance = 13.0
        for reference, point in candidates:
            marker_x, marker_y = self._canonical_to_runtime(point)
            distance = math.hypot(float(x - marker_x), float(y - marker_y))
            if distance < nearest_distance:
                nearest = reference
                nearest_distance = distance
        return nearest

    def _point_for_drag_ref(self, reference: tuple[str, int, str]) -> tuple[int, int]:
        kind, index, endpoint = reference
        pair = self.existing_pairs[index] if kind == "existing" else self.selected_pairs[index]
        return pair.start_pixel if endpoint == "start" else pair.target_pixel

    def _on_map_press(self, event) -> str | None:
        if (
            self.busy
            or self.runtime is None
            or self.runtime.scene_code != self.scene_var.get()
        ):
            return None
        self.canvas.focus_set()
        reference = self._hit_test_marker(event.x, event.y)
        if reference is None:
            previous_start = self.pending_start
            self._on_map_click(event)
            if (
                previous_start is None
                and self.pending_start is not None
                and self.pending_target is None
            ):
                self.new_pair_drag_origin = (event.x, event.y)
                self.canvas.configure(cursor="crosshair")
            return None
        self.drag_ref = reference
        self.drag_origin = self._point_for_drag_ref(reference)
        self.drag_preview = self.drag_origin
        kind, index, endpoint = reference
        if kind == "existing":
            point_id = self.existing_pairs[index].point_id
            iid = f"existing:{point_id}"
            point_label = point_id
        else:
            iid = f"new:{index}"
            point_label = f"new point {index + 1}"
        if self.point_tree.exists(iid):
            self.point_tree.selection_set(iid)
            self.point_tree.see(iid)
        self.status_var.set(
            f"Dragging {point_label} {endpoint}. Release it on a navigable location."
        )
        self.canvas.configure(cursor="fleur")
        self._draw_overlays()
        self._refresh_controls()
        return "break"

    def _on_map_drag(self, event) -> str | None:
        if self.runtime is None:
            return None
        if self.drag_ref is None:
            return self._on_new_pair_drag(event)
        min_x, max_x, min_y, max_y = self.runtime.margin
        clamped_x = int(np.clip(event.x, min_x, max_x))
        clamped_y = int(np.clip(event.y, min_y, max_y))
        self.drag_preview = self._runtime_to_canonical(clamped_x, clamped_y)
        self.coordinate_var.set(
            f"Map coordinate: ({self.drag_preview[0]}, {self.drag_preview[1]})"
        )
        self._draw_overlays()
        return "break"

    def _on_new_pair_drag(self, event) -> str | None:
        """Preview a new start-target pair while the pointer is held."""
        if self.new_pair_drag_origin is None or self.runtime is None:
            return None
        origin_x, origin_y = self.new_pair_drag_origin
        if math.hypot(event.x - origin_x, event.y - origin_y) < 4.0:
            return "break"
        min_x, max_x, min_y, max_y = self.runtime.margin
        drag_x = int(np.clip(event.x, min_x, max_x))
        drag_y = int(np.clip(event.y, min_y, max_y))
        self.pending_target = self._runtime_to_canonical(drag_x, drag_y)
        self.coordinate_var.set(
            f"Map coordinate: ({self.pending_target[0]}, {self.pending_target[1]})"
        )
        self.status_var.set(
            f"New pair: {self.pending_start} -> {self.pending_target}. "
            "Release to confirm it."
        )
        self._draw_overlays()
        self._refresh_controls()
        return "break"

    def _on_map_release(self, event) -> str | None:
        if self.drag_ref is None:
            return self._on_new_pair_release(event)
        reference = self.drag_ref
        origin = self.drag_origin
        destination = self.drag_preview
        if self.runtime is not None:
            min_x, max_x, min_y, max_y = self.runtime.margin
            release_x = int(np.clip(event.x, min_x, max_x))
            release_y = int(np.clip(event.y, min_y, max_y))
            destination = self._runtime_to_canonical(release_x, release_y)
        self.drag_ref = None
        self.drag_origin = None
        self.drag_preview = None
        self.canvas.configure(cursor="")
        if destination is None or origin == destination:
            self.status_var.set("Point selected. Drag its S or T marker to modify it.")
            self._draw_overlays()
            self._refresh_controls()
            return "break"

        kind, index, endpoint = reference
        runtime = self.runtime
        if runtime is None:
            return "break"
        pair = self.existing_pairs[index] if kind == "existing" else self.selected_pairs[index]

        def task() -> tuple[float, float]:
            if endpoint == "start":
                return map_start_pixel_through_unity(runtime, destination, pair.direction)
            return map_target_pixel_through_unity(runtime, destination)

        def success(mapped_world: tuple[float, float]) -> None:
            if kind == "existing":
                self.existing_history.append(list(self.existing_pairs))
                old = self.existing_pairs[index]
                replacement = move_pair_endpoint(
                    old, endpoint, destination, mapped_world
                )
                self.existing_pairs[index] = replacement
                self._update_existing_dirty()
                point_label = old.point_id
            else:
                self.selected_history.append(list(self.selected_pairs))
                old = self.selected_pairs[index]
                replacement = move_pair_endpoint(
                    old, endpoint, destination, mapped_world
                )
                self.selected_pairs[index] = replacement
                point_label = f"new point {index + 1}"
            self._refresh_point_table()
            iid = (
                f"existing:{point_label}"
                if kind == "existing"
                else f"new:{index}"
            )
            if self.point_tree.exists(iid):
                self.point_tree.selection_set(iid)
                self.point_tree.see(iid)
            self._draw_overlays()
            self.status_var.set(
                f"Moved {point_label} {endpoint}. Save the corresponding point changes."
            )
            self._refresh_controls()

        self._draw_overlays()
        self._run_background(
            f"Resolving dragged {endpoint} through the scene cache...",
            task,
            success,
        )
        return "break"

    def _on_new_pair_release(self, event) -> str | None:
        """Finish a drag-created pair, or keep a click as the pending start."""
        if self.new_pair_drag_origin is None:
            return None
        origin_x, origin_y = self.new_pair_drag_origin
        moved = math.hypot(event.x - origin_x, event.y - origin_y) >= 4.0
        if moved:
            self._on_new_pair_drag(event)
        self.new_pair_drag_origin = None
        self.canvas.configure(cursor="crosshair")
        if not moved or self.pending_target is None:
            self.pending_target = None
            self.status_var.set(
                f"Start selected at {self.pending_start}. Use arrow keys for yaw, "
                "then click or drag to the target."
            )
            self._draw_overlays()
            self._refresh_controls()
            return "break"
        self.status_var.set(
            "Pair selected. Use arrow keys to set the start yaw, then click Confirm pair."
        )
        self._refresh_controls()
        return "break"

    def _selected_existing_pair(self) -> ExistingPair | None:
        selection = self.point_tree.selection()
        if not selection or not selection[0].startswith("existing:"):
            return None
        point_id = selection[0].split(":", 1)[1]
        return next(
            (pair for pair in self.existing_pairs if pair.point_id == point_id),
            None,
        )

    def _selected_pair_reference(self) -> tuple[str, int] | None:
        selection = self.point_tree.selection()
        if not selection:
            return None
        item_id = selection[0]
        if item_id.startswith("existing:"):
            point_id = item_id.split(":", 1)[1]
            return next(
                (
                    ("existing", index)
                    for index, pair in enumerate(self.existing_pairs)
                    if pair.point_id == point_id
                ),
                None,
            )
        if item_id.startswith("new:"):
            try:
                index = int(item_id.split(":", 1)[1])
            except ValueError:
                return None
            if 0 <= index < len(self.selected_pairs):
                return "new", index
        return None

    def _update_existing_dirty(self) -> None:
        self.existing_dirty = self.existing_pairs != self.original_existing_pairs

    def apply_yaw_to_selected(self) -> None:
        reference = self._selected_pair_reference()
        if self.busy or reference is None:
            messagebox.showinfo("Select a point", "Select a point in the table first.")
            return
        try:
            direction = self._direction()
        except ValueError as exc:
            messagebox.showerror("Invalid yaw", str(exc))
            return

        kind, index = reference
        if kind == "existing":
            if self.selected_pairs or self.pending_start is not None:
                messagebox.showinfo(
                    "Finish current selection",
                    "Save or undo new-point selections before editing a saved yaw.",
                )
                return
            pair = self.existing_pairs[index]
            if pair.direction == direction:
                self.status_var.set(f"{pair.point_id} already uses yaw {direction:g} deg.")
                return
            self.existing_history.append(list(self.existing_pairs))
            self.existing_pairs[index] = set_pair_direction(pair, direction)
            self._update_existing_dirty()
            label = pair.point_id
            save_label = "Save existing changes"
        else:
            if self.existing_dirty or self.pending_start is not None:
                messagebox.showinfo(
                    "Finish current edit",
                    "Save or undo the current edit before changing this yaw.",
                )
                return
            pair = self.selected_pairs[index]
            if pair.direction == direction:
                self.status_var.set(f"Selected point already uses yaw {direction:g} deg.")
                return
            self.selected_history.append(list(self.selected_pairs))
            self.selected_pairs[index] = set_pair_direction(pair, direction)
            label = self.point_tree.item(f"new:{index}", "text")
            save_label = "Save new pairs"

        self._refresh_point_table()
        iid = f"existing:{label}" if kind == "existing" else f"new:{index}"
        if self.point_tree.exists(iid):
            self.point_tree.selection_set(iid)
            self.point_tree.see(iid)
        self._draw_overlays()
        self.status_var.set(
            f"Updated {label} to yaw {direction:g} deg. Click {save_label} to persist it."
        )
        self._refresh_controls()

    def _renumber_existing_pairs(self) -> None:
        self.existing_pairs = [
            ExistingPair(
                point_id=f"point{index}",
                start_pixel=pair.start_pixel,
                start_world=pair.start_world,
                target_pixel=pair.target_pixel,
                direction=pair.direction,
            )
            for index, pair in enumerate(self.existing_pairs, start=1)
        ]

    def delete_selected_existing(self) -> None:
        pair = self._selected_existing_pair()
        if pair is None:
            messagebox.showinfo("Select a point", "Select a saved point in the table first.")
            return
        if self.selected_pairs or self.pending_start is not None:
            messagebox.showinfo(
                "Finish current selection",
                "Confirm or undo the current selection before deleting a saved point.",
            )
            return
        if not messagebox.askyesno(
            "Delete saved point?",
            f"Remove {pair.point_id} from the current scene draft? "
            "The JSON file will not change until you save existing changes.",
        ):
            return
        self.existing_history.append(list(self.existing_pairs))
        self.existing_pairs = [
            existing
            for existing in self.existing_pairs
            if existing.point_id != pair.point_id
        ]
        self._renumber_existing_pairs()
        self._update_existing_dirty()
        self._refresh_point_table()
        self._draw_overlays()
        self.status_var.set(
            f"Deleted {pair.point_id} in the draft. Save existing changes to persist it."
        )
        self._refresh_controls()

    def _on_map_click(self, event) -> None:
        if (
            self.busy
            or self.runtime is None
            or self.runtime.scene_code != self.scene_var.get()
        ):
            return
        try:
            limit = self._point_limit()
        except ValueError as exc:
            messagebox.showerror("Invalid pair count", str(exc))
            return
        if self.existing_dirty:
            self.status_var.set(
                "Save or undo existing-point changes before adding new pairs."
            )
            return
        if len(self.selected_pairs) >= limit:
            self.status_var.set("Requested pair count reached. Save or undo a pair.")
            return
        min_x, max_x, min_y, max_y = self.runtime.margin
        if not (min_x <= event.x <= max_x and min_y <= event.y <= max_y):
            self.status_var.set("Click inside the rendered minimap, not its border.")
            return
        point = self._runtime_to_canonical(event.x, event.y)
        if self.pending_start is None:
            selection = self.point_tree.selection()
            if selection:
                self.point_tree.selection_remove(*selection)
            self.pending_start = point
            self.status_var.set(
                f"Start selected at {point}. Use arrow keys for yaw, then select target."
            )
        elif self.pending_target is None:
            self.pending_target = point
            self.status_var.set(
                f"Target selected at {point}. Use arrow keys for yaw, "
                "then click Confirm pair."
            )
        else:
            self.status_var.set("Both points are selected. Confirm or undo first.")
        self._draw_overlays()
        self._refresh_controls()

    def confirm_pair(self) -> None:
        if (
            self.busy
            or self.runtime is None
            or self.runtime.scene_code != self.scene_var.get()
        ):
            return
        if self.pending_start is None or self.pending_target is None:
            messagebox.showinfo("Incomplete pair", "Click a start and target point first.")
            return
        try:
            limit = self._point_limit()
            direction = self._direction()
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        if len(self.selected_pairs) >= limit:
            messagebox.showinfo("Pair count reached", "Undo a pair or save the current set.")
            return

        start = self.pending_start
        target = self.pending_target
        runtime = self.runtime

        def task() -> SelectedPair:
            return map_pair_through_unity(runtime, start, target, direction)

        def success(selection: SelectedPair) -> None:
            self.selected_history.append(list(self.selected_pairs))
            self.selected_pairs.append(selection)
            self.pending_start = None
            self.pending_target = None
            self._refresh_point_table()
            new_iid = f"new:{len(self.selected_pairs) - 1}"
            if self.point_tree.exists(new_iid):
                self.point_tree.selection_set(new_iid)
                self.point_tree.see(new_iid)
            self._draw_overlays()
            limit_now = self._point_limit()
            if len(self.selected_pairs) == limit_now:
                self.status_var.set(
                    f"Confirmed {limit_now} pairs. Review them, then save."
                )
            else:
                self.status_var.set(
                    f"Pair confirmed ({selection.world_distance:.1f} m straight-line). "
                    "Click the next start point or drag an S/T marker to refine it."
                )
            self._refresh_controls()

        self._run_background(
            "Mapping start and target through the scene cache...", task, success
        )

    def undo(self) -> None:
        if self.busy:
            return
        if self.new_pair_drag_origin is not None:
            self.new_pair_drag_origin = None
            self.pending_start = None
            self.pending_target = None
            self.canvas.configure(cursor="crosshair")
            self.status_var.set("Cancelled the new-pair drag.")
        elif self.drag_ref is not None:
            self.drag_ref = None
            self.drag_origin = None
            self.drag_preview = None
            self.canvas.configure(cursor="")
            self.status_var.set("Cancelled the current drag.")
        elif self.pending_target is not None:
            self.pending_target = None
            self.status_var.set("Target click undone. Click a new target point.")
        elif self.pending_start is not None:
            self.pending_start = None
            self.status_var.set("Start click undone. Click a new start point.")
        elif self.selected_history:
            self.selected_pairs = self.selected_history.pop()
            self.status_var.set("Restored the previous new-point draft.")
            self._refresh_point_table()
        elif self.selected_pairs:
            removed = self.selected_pairs.pop()
            self.status_var.set(
                f"Removed the last confirmed pair ending at {removed.target_pixel}."
            )
            self._refresh_point_table()
        elif self.existing_history:
            self.existing_pairs = self.existing_history.pop()
            self._update_existing_dirty()
            self.status_var.set("Restored the previous saved-point draft.")
            self._refresh_point_table()
        self._draw_overlays()
        self._refresh_controls()

    def _preview_start_index(self) -> int:
        if self.mode_var.get() == "replace":
            return 1
        data = load_input_points(self.input_file)
        return next_point_index(data.get(self.scene_var.get(), []))

    def _refresh_point_table(self) -> None:
        selected_iid = next(iter(self.point_tree.selection()), None)
        for item in self.point_tree.get_children():
            self.point_tree.delete(item)
        for pair in self.existing_pairs:
            self.point_tree.insert(
                "",
                tk.END,
                iid=f"existing:{pair.point_id}",
                text=pair.point_id,
                values=(
                    f"{pair.start_world[0]:.2f},{pair.start_world[1]:.2f}",
                    f"{pair.target_pixel[0]},{pair.target_pixel[1]}",
                    f"{pair.direction:.0f}",
                    "saved",
                ),
                tags=("existing",),
            )
        try:
            start_index = self._preview_start_index()
        except (ValueError, OSError):
            start_index = 1
        for offset, pair in enumerate(self.selected_pairs):
            self.point_tree.insert(
                "",
                tk.END,
                iid=f"new:{offset}",
                text=f"point{start_index + offset}",
                values=(
                    f"{pair.start_world[0]:.2f},{pair.start_world[1]:.2f}",
                    f"{pair.target_pixel[0]},{pair.target_pixel[1]}",
                    f"{pair.direction:.0f}",
                    f"{pair.world_distance:.1f}m",
                ),
            )
        try:
            limit = self._point_limit()
            if self.mode_var.get() == "append":
                self.progress_var.set(
                    f"New pairs ready: {len(self.selected_pairs)} (limit {limit})"
                )
            else:
                self.progress_var.set(
                    f"Replacement pairs: {len(self.selected_pairs)} / {limit}"
                )
        except ValueError:
            self.progress_var.set(f"Confirmed pairs: {len(self.selected_pairs)}")
        if selected_iid and self.point_tree.exists(selected_iid):
            self.point_tree.selection_set(selected_iid)

    def _draw_marker(
        self,
        point: tuple[int, int],
        color: str,
        label: str,
        *,
        selected: bool = False,
        outline: str = "#ffffff",
    ) -> None:
        x, y = self._canonical_to_runtime(point)
        radius = 10 if selected else 8
        if selected:
            halo = radius + 4
            self.canvas.create_oval(
                x - halo,
                y - halo,
                x + halo,
                y + halo,
                fill="",
                outline="#ffffff",
                width=3,
                tags=("overlay",),
            )
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline=outline,
            width=3 if selected else 2,
            tags=("overlay",),
        )
        self.canvas.create_text(
            x + radius + 5,
            y - radius - 1,
            text=label,
            fill="#101418",
            anchor=tk.SW,
            font=("TkDefaultFont", 11 if selected else 10, "bold"),
            tags=("overlay",),
        )
        self.canvas.create_text(
            x + radius + 4,
            y - radius - 2,
            text=label,
            fill="white",
            anchor=tk.SW,
            font=("TkDefaultFont", 11 if selected else 10, "bold"),
            tags=("overlay",),
        )

    def _draw_heading_arrow(
        self,
        start_xy: tuple[int, int],
        direction: float,
        color: str,
        *,
        selected: bool,
    ) -> None:
        end_x, end_y = heading_endpoint(
            start_xy,
            direction,
            44.0 if selected else 34.0,
        )
        arrow_shape = (14, 16, 6) if selected else (11, 13, 5)
        self.canvas.create_line(
            start_xy[0],
            start_xy[1],
            end_x,
            end_y,
            fill="#111820",
            width=8 if selected else 6,
            arrow=tk.LAST,
            arrowshape=arrow_shape,
            tags=("overlay",),
        )
        self.canvas.create_line(
            start_xy[0],
            start_xy[1],
            end_x,
            end_y,
            fill=color,
            width=4 if selected else 3,
            arrow=tk.LAST,
            arrowshape=arrow_shape,
            tags=("overlay",),
        )

    def _draw_overlays(self) -> None:
        self.canvas.delete("overlay")
        if self.runtime is None:
            return

        selected_reference = self._selected_pair_reference()
        try:
            new_start_index = self._preview_start_index()
        except (ValueError, OSError):
            new_start_index = 1
        draw_items: list[
            tuple[str, int, SelectedPair | ExistingPair, int]
        ] = [
            (
                "existing",
                index,
                pair,
                int(pair.point_id.removeprefix("point"))
                if pair.point_id.removeprefix("point").isdigit()
                else index + 1,
            )
            for index, pair in enumerate(self.existing_pairs)
        ]
        draw_items.extend(
            ("new", index, pair, new_start_index + index)
            for index, pair in enumerate(self.selected_pairs)
        )
        draw_items.sort(
            key=lambda item: (item[0], item[1]) == selected_reference
        )

        for kind, index, pair, point_number in draw_items:
            is_selected = selected_reference == (kind, index)
            pair_color = PAIR_COLORS[(point_number - 1) % len(PAIR_COLORS)]
            start_pixel = (
                self.drag_preview
                if self.drag_ref == (kind, index, "start")
                and self.drag_preview is not None
                else pair.start_pixel
            )
            target_pixel = (
                self.drag_preview
                if self.drag_ref == (kind, index, "target")
                and self.drag_preview is not None
                else pair.target_pixel
            )
            sx, sy = self._canonical_to_runtime(start_pixel)
            tx, ty = self._canonical_to_runtime(target_pixel)
            self.canvas.create_line(
                sx,
                sy,
                tx,
                ty,
                fill="#111820" if is_selected else pair_color,
                width=8 if is_selected else 4,
                tags=("overlay",),
            )
            if is_selected:
                self.canvas.create_line(
                    sx,
                    sy,
                    tx,
                    ty,
                    fill=pair_color,
                    width=5,
                    tags=("overlay",),
                )
            self._draw_heading_arrow(
                (sx, sy),
                pair.direction,
                pair_color,
                selected=is_selected,
            )
            self._draw_marker(
                start_pixel,
                "#00a86b",
                f"S{point_number}",
                selected=is_selected,
                outline=pair_color,
            )
            self._draw_marker(
                target_pixel,
                "#d32f2f",
                f"T{point_number}",
                selected=is_selected,
                outline=pair_color,
            )
        if self.pending_start is not None:
            try:
                pending_direction = self._direction()
            except ValueError:
                pending_direction = 0.0
            self._draw_heading_arrow(
                self._canonical_to_runtime(self.pending_start),
                pending_direction,
                "#00e5ff",
                selected=True,
            )
            self._draw_marker(
                self.pending_start,
                "#00a86b",
                "START",
                selected=True,
                outline="#00e5ff",
            )
        if self.pending_target is not None:
            sx, sy = self._canonical_to_runtime(self.pending_start)
            tx, ty = self._canonical_to_runtime(self.pending_target)
            self.canvas.create_line(
                sx,
                sy,
                tx,
                ty,
                fill="#ffb300",
                width=5,
                tags=("overlay",),
            )
            self._draw_marker(
                self.pending_target,
                "#d32f2f",
                "TARGET",
                selected=True,
                outline="#ffca28",
            )

    def _refresh_controls(self) -> None:
        runtime_ready = (
            self.runtime is not None
            and self.runtime.scene_code == self.scene_var.get()
            and not self.busy
        )
        try:
            limit = self._point_limit()
        except ValueError:
            limit = 0
        append_mode = self.mode_var.get() == "append"
        can_save_new_pairs = selection_ready_to_save(
            self.mode_var.get(),
            len(self.selected_pairs),
            limit,
        )
        selected_existing = self._selected_existing_pair()
        selected_reference = self._selected_pair_reference()
        has_undo = bool(
            self.pending_start is not None
            or self.pending_target is not None
            or self.drag_ref is not None
            or self.selected_pairs
            or self.selected_history
            or self.existing_history
        )
        self.load_button.configure(
            state=tk.NORMAL if self.cache_ready and not self.busy else tk.DISABLED
        )
        setting_state = tk.DISABLED if self.busy else tk.NORMAL
        self.scene_tree.configure(selectmode="none" if self.busy else "browse")
        self.pair_spin.configure(state=setting_state)
        self.pair_count_label.configure(
            text="Append limit" if append_mode else "Pairs to replace"
        )
        self.save_button.configure(
            text="Append new pairs" if append_mode else "Replace scene points"
        )
        self.direction_combo.configure(state=setting_state)
        self.replace_radio.configure(state=setting_state)
        self.append_radio.configure(state=setting_state)
        self.undo_button.configure(
            state=tk.NORMAL if runtime_ready and has_undo else tk.DISABLED
        )
        self.confirm_button.configure(
            state=(
                tk.NORMAL
                if runtime_ready
                and self.pending_start is not None
                and self.pending_target is not None
                and self.new_pair_drag_origin is None
                and len(self.selected_pairs) < limit
                else tk.DISABLED
            )
        )
        self.save_button.configure(
            state=(
                tk.NORMAL
                if runtime_ready
                and can_save_new_pairs
                and not self.existing_dirty
                and self.pending_start is None
                and self.drag_ref is None
                and self.new_pair_drag_origin is None
                else tk.DISABLED
            )
        )
        can_change_existing = bool(
            runtime_ready
            and selected_existing is not None
            and not self.selected_pairs
            and self.pending_start is None
            and self.drag_ref is None
        )
        can_apply_yaw = bool(
            runtime_ready
            and selected_reference is not None
            and self.pending_start is None
            and self.drag_ref is None
            and self.new_pair_drag_origin is None
            and (
                (selected_reference[0] == "existing" and not self.selected_pairs)
                or (selected_reference[0] == "new" and not self.existing_dirty)
            )
        )
        self.apply_yaw_button.configure(
            state=tk.NORMAL if can_apply_yaw else tk.DISABLED
        )
        self.delete_existing_button.configure(
            state=tk.NORMAL if can_change_existing else tk.DISABLED
        )
        self.save_existing_button.configure(
            state=(
                tk.NORMAL
                if runtime_ready
                and self.existing_dirty
                and not self.selected_pairs
                and self.pending_start is None
                and self.drag_ref is None
                else tk.DISABLED
            )
        )

    def _save_shortcut(self, _event=None) -> str:
        """Route Ctrl/Cmd+S to the active existing- or new-point draft."""
        if self.busy:
            return "break"
        if (
            self.pending_start is not None
            or self.drag_ref is not None
            or self.new_pair_drag_origin is not None
        ):
            self.status_var.set(
                "Confirm or undo the current pair/drag before saving."
            )
            return "break"
        if self.existing_dirty:
            self.save_existing_changes()
        elif self.selected_pairs:
            self.save()
        else:
            self.status_var.set("There are no unsaved point changes.")
        return "break"

    def save_existing_changes(self) -> None:
        if self.busy or not self.existing_dirty:
            return
        try:
            scene_code = self._selected_scene()
        except ValueError as exc:
            messagebox.showerror("Invalid scene", str(exc))
            return
        if self.runtime is None or self.runtime.scene_code != scene_code:
            messagebox.showerror("Scene mismatch", "Load the selected scene before saving.")
            return
        if not messagebox.askyesno(
            "Save existing-point changes?",
            f"Write the edited list of {len(self.existing_pairs)} points to {scene_code}?",
        ):
            return
        entries = [existing_pair_to_entry(pair) for pair in self.existing_pairs]
        try:
            save_scene_entries(self.input_file, scene_code, entries)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.original_existing_pairs = list(self.existing_pairs)
        self.existing_history.clear()
        self.existing_dirty = False
        self.refresh_scene_counts()
        self._refresh_point_table()
        self._draw_overlays()
        self.status_var.set(
            f"Saved {len(self.existing_pairs)} existing points for {scene_code}."
        )
        self._refresh_controls()
        messagebox.showinfo("Saved", self.status_var.get())

    def save(self) -> None:
        if self.busy:
            return
        if (
            self.pending_start is not None
            or self.drag_ref is not None
            or self.new_pair_drag_origin is not None
        ):
            messagebox.showinfo(
                "Incomplete edit",
                "Confirm or undo the current pair/drag before saving.",
            )
            return
        try:
            scene_code = self._selected_scene()
            limit = self._point_limit()
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        if self.runtime is None or self.runtime.scene_code != scene_code:
            messagebox.showerror("Scene mismatch", "Load the selected scene before saving.")
            return
        mode = self.mode_var.get()
        if mode == "replace" and len(self.selected_pairs) != limit:
            messagebox.showinfo(
                "Incomplete selection",
                f"Confirm exactly {limit} pairs before saving.",
            )
            return
        if mode == "append" and not self.selected_pairs:
            messagebox.showinfo(
                "No new pairs",
                "Add at least one start-target pair before saving.",
            )
            return
        existing_count = scene_point_counts(self.input_file)[scene_code]
        action = "replace" if mode == "replace" else "append to"
        selected_count = len(self.selected_pairs)
        if not messagebox.askyesno(
            "Confirm save",
            f"{action.capitalize()} {scene_code} ({existing_count} existing points) "
            f"with {selected_count} selected pairs?",
        ):
            return
        try:
            saved_selections = list(self.selected_pairs)
            updated = save_scene_pairs(
                self.input_file,
                scene_code,
                self.selected_pairs,
                mode,
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        if mode == "replace":
            retained_existing: list[ExistingPair] = []
        else:
            retained_existing = list(self.existing_pairs)
        saved_entries = updated[-len(saved_selections):]
        retained_existing.extend(
            ExistingPair(
                point_id=str(entry["point_id"]),
                start_pixel=selection.start_pixel,
                start_world=selection.start_world,
                target_pixel=selection.target_pixel,
                direction=selection.direction,
            )
            for entry, selection in zip(saved_entries, saved_selections)
        )
        self.existing_pairs = retained_existing
        self.original_existing_pairs = list(retained_existing)
        self.existing_history.clear()
        self.existing_dirty = False
        self.selected_pairs.clear()
        self.selected_history.clear()
        self.pending_start = None
        self.pending_target = None
        self.new_pair_drag_origin = None
        self.refresh_scene_counts()
        self._refresh_point_table()
        self._draw_overlays()
        self.status_var.set(
            f"Saved {scene_code}: {len(updated)} total points in {self.input_file}."
        )
        self._refresh_controls()
        messagebox.showinfo("Saved", self.status_var.get())

    def close(self) -> None:
        if self.closing:
            return
        if self.busy:
            messagebox.showinfo(
                "Background operation in progress",
                "Wait for cache preparation or coordinate mapping to finish, then close.",
            )
            return
        has_unsaved = bool(
            self.selected_pairs
            or self.pending_start is not None
            or self.existing_dirty
            or self.drag_ref is not None
        )
        if has_unsaved and not messagebox.askyesno(
            "Discard selections?", "Close without saving the current point changes?"
        ):
            return
        self.closing = True
        runtime = self.runtime
        self.runtime = None
        if runtime is not None:
            self.executor.submit(runtime.close)
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-name", "--file_name", default="auto")
    parser.add_argument(
        "--input-file",
        "--input_file",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to input_points.json.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for cached minimaps and pixel/world projections.",
    )
    parser.add_argument(
        "--scene",
        type=normalize_scene_code,
        default="scene1",
        help="Initially selected scene number/code (1-24).",
    )
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--mode", choices=("replace", "append"), default="replace")
    parser.add_argument("--direction", type=float, default=180.0)
    parser.add_argument(
        "--auto-load",
        action="store_true",
        help="Open the initially selected cached scene when cache preparation finishes.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild all 24 scene caches once, even when valid caches exist.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(REPO_ROOT / "outputs" / "input_point_editor"),
    )
    parser.add_argument("--worker-id", type=int, default=40)
    parser.add_argument("--base-port", type=int, default=5507)
    parser.add_argument("--screen-width", type=int, default=1724)
    parser.add_argument("--screen-height", type=int, default=1024)
    parser.add_argument("--ego-width", type=int, default=320)
    parser.add_argument("--ego-height", type=int, default=240)
    parser.add_argument("--minimap-width", type=int, default=862)
    parser.add_argument("--minimap-height", type=int, default=512)
    parser.add_argument("--quality-level", type=int, default=UNITY_ENGINE_QUALITY_LEVEL)
    parser.add_argument(
        "--dynamic-objects",
        choices=("moving", "static"),
        default="static",
    )
    add_motion_speed_args(parser)
    add_lighting_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs < 1 or args.pairs > 100:
        raise SystemExit("--pairs must be between 1 and 100.")
    try:
        args.minimap_width, args.minimap_height = resolve_minimap_resolution(
            args.minimap_width,
            args.minimap_height,
        )
        args.file_name = resolve_scene_all_path(args.file_name)
        load_input_points(Path(args.input_file).expanduser().resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    root = tk.Tk()
    InputPointEditor(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
