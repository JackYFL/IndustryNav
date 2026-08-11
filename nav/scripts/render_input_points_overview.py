"""Render a 4x6 overview of cached scene minimaps and benchmark points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nav.config import REPO_ROOT, SCENE_CODES
from nav.scripts.edit_input_points import (
    CANONICAL_MAP_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_INPUT_FILE,
    PAIR_COLORS,
    SceneRuntime,
    load_input_points,
    map_existing_pairs_through_unity,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "input_points_overview_4x6.png"
START_COLOR = "#00a86b"
TARGET_COLOR = "#d32f2f"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_cached_runtime(cache_dir: Path, scene_code: str) -> SceneRuntime:
    metadata_path = cache_dir / f"{scene_code}.json"
    image_path = cache_dir / f"{scene_code}.png"
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as source:
            minimap_rgb = np.asarray(source.convert("RGB")).copy()
        minimap_size = tuple(int(value) for value in metadata["minimap_size"])
        margin = tuple(float(value) for value in metadata["margin"])
        matrix = np.asarray(metadata["pixel_to_world_h"], dtype=np.float64)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid cache for {scene_code}: {cache_dir}") from exc
    if (
        len(minimap_size) != 2
        or len(margin) != 4
        or matrix.shape != (3, 3)
        or minimap_rgb.shape[:2] != (minimap_size[1], minimap_size[0])
    ):
        raise ValueError(f"Incompatible cache metadata for {scene_code}.")
    return SceneRuntime(
        scene_code=scene_code,
        env=None,
        env_params=None,
        target_sc=None,
        margin=margin,
        minimap_rgb=minimap_rgb,
        minimap_size=minimap_size,
        pixel_to_world_h=matrix,
    )


def _scaled_point(
    point: tuple[int, int],
    runtime_size: tuple[int, int],
    panel_size: tuple[int, int],
) -> tuple[int, int]:
    canonical_width, canonical_height = CANONICAL_MAP_SIZE
    runtime_x = float(point[0]) * runtime_size[0] / canonical_width
    runtime_y = float(point[1]) * runtime_size[1] / canonical_height
    return (
        round(runtime_x * panel_size[0] / runtime_size[0]),
        round(runtime_y * panel_size[1] / runtime_size[1]),
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    direction: float,
    color: str,
    scale: float,
) -> None:
    radians = math.radians(float(direction) % 360.0)
    unit_x = -math.cos(radians)
    unit_y = -math.sin(radians)
    length = max(24.0, 38.0 * scale)
    end_x = start[0] + unit_x * length
    end_y = start[1] + unit_y * length
    width = max(3, round(4 * scale))
    draw.line((start[0], start[1], end_x, end_y), fill="#101820", width=width + 3)
    draw.line((start[0], start[1], end_x, end_y), fill=color, width=width)
    head_length = max(8.0, 12.0 * scale)
    head_width = max(5.0, 7.0 * scale)
    base_x = end_x - unit_x * head_length
    base_y = end_y - unit_y * head_length
    perp_x, perp_y = -unit_y, unit_x
    draw.polygon(
        (
            (end_x, end_y),
            (base_x + perp_x * head_width, base_y + perp_y * head_width),
            (base_x - perp_x * head_width, base_y - perp_y * head_width),
        ),
        fill=color,
        outline="#101820",
    )


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    fill: str,
    outline: str,
    label: str,
    scale: float,
    font: ImageFont.ImageFont,
) -> None:
    radius = max(6, round(9 * scale))
    draw.ellipse(
        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
        fill=fill,
        outline=outline,
        width=max(2, round(3 * scale)),
    )
    draw.text(
        (point[0] + radius + 3, point[1] - radius - 2),
        label,
        fill="white",
        font=font,
        stroke_width=max(1, round(2 * scale)),
        stroke_fill="#101820",
    )


def render_overview(
    input_file: Path,
    cache_dir: Path,
    output: Path,
    *,
    columns: int = 4,
    panel_width: int = 862,
) -> Path:
    if columns <= 0 or len(SCENE_CODES) % columns:
        raise ValueError("Columns must divide the 24 scenes evenly.")
    if panel_width < 320:
        raise ValueError("Panel width must be at least 320 pixels.")

    entries_by_scene = load_input_points(input_file)
    first_runtime = _load_cached_runtime(cache_dir, SCENE_CODES[0])
    panel_height = round(
        panel_width * first_runtime.minimap_size[1] / first_runtime.minimap_size[0]
    )
    title_height = max(34, round(panel_width * 0.047))
    gutter = max(6, round(panel_width * 0.01))
    rows = len(SCENE_CODES) // columns
    canvas_width = columns * panel_width + (columns + 1) * gutter
    canvas_height = rows * (panel_height + title_height) + (rows + 1) * gutter
    overview = Image.new("RGB", (canvas_width, canvas_height), "#e9edf0")
    title_font = _font(max(18, round(title_height * 0.55)), bold=True)
    label_font = _font(max(13, round(panel_width * 0.018)), bold=True)
    scale = panel_width / CANONICAL_MAP_SIZE[0]

    for scene_index, scene_code in enumerate(SCENE_CODES):
        runtime = (
            first_runtime
            if scene_index == 0
            else _load_cached_runtime(cache_dir, scene_code)
        )
        pairs = map_existing_pairs_through_unity(
            runtime,
            entries_by_scene.get(scene_code, []),
        )
        minimap = Image.fromarray(runtime.minimap_rgb).resize(
            (panel_width, panel_height),
            Image.Resampling.LANCZOS,
        )
        panel = Image.new("RGB", (panel_width, panel_height + title_height), "white")
        panel.paste(minimap, (0, title_height))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel_width - 1, title_height - 1), fill="#18232d")
        draw.text(
            (round(14 * scale), title_height // 2),
            scene_code,
            fill="white",
            font=title_font,
            anchor="lm",
        )

        for index, pair in enumerate(pairs):
            color = PAIR_COLORS[index % len(PAIR_COLORS)]
            start_x, start_y = _scaled_point(
                pair.start_pixel, runtime.minimap_size, (panel_width, panel_height)
            )
            target_x, target_y = _scaled_point(
                pair.target_pixel, runtime.minimap_size, (panel_width, panel_height)
            )
            start = (start_x, start_y + title_height)
            target = (target_x, target_y + title_height)
            draw.line(
                (start[0], start[1], target[0], target[1]),
                fill="#101820",
                width=max(5, round(7 * scale)),
            )
            draw.line(
                (start[0], start[1], target[0], target[1]),
                fill=color,
                width=max(3, round(4 * scale)),
            )
            _draw_arrow(draw, start, pair.direction, color, scale)
            point_number = index + 1
            _draw_marker(
                draw, start, START_COLOR, color, f"S{point_number}", scale, label_font
            )
            _draw_marker(
                draw, target, TARGET_COLOR, color, f"T{point_number}", scale, label_font
            )

        row, column = divmod(scene_index, columns)
        offset_x = gutter + column * (panel_width + gutter)
        offset_y = gutter + row * (panel_height + title_height + gutter)
        overview.paste(panel, (offset_x, offset_y))

    output.parent.mkdir(parents=True, exist_ok=True)
    overview.save(output, format="PNG", optimize=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=862)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_overview(
        Path(args.input_file).expanduser().resolve(),
        Path(args.cache_dir).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        columns=args.columns,
        panel_width=args.panel_width,
    )
    print(output)


if __name__ == "__main__":
    main()
