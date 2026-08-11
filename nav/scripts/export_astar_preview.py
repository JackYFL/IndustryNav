"""Compose an A* benchmark run into the README navigation preview GIF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TOP_PANEL_SIZE = (320, 240)
COMPOSITE_WIDTH = TOP_PANEL_SIZE[0] * 2
ROW_HEADER_HEIGHT = 30
DEPTH_PANEL_BOX = (
    TOP_PANEL_SIZE[0],
    ROW_HEADER_HEIGHT,
    COMPOSITE_WIDTH,
    ROW_HEADER_HEIGHT + TOP_PANEL_SIZE[1],
)
GIF_COLOR_LEVELS = 128
GIF_DEPTH_LEVELS = 256 - GIF_COLOR_LEVELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RGB, depth, and A* path overlays as a GIF."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/_readme_preview/astar_scene1_point3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/industrynav_navigation_example.gif"),
    )
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--duration-ms", type=int, default=140)
    return parser.parse_args()


def numeric_files(directory: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        try:
            files[int(path.name[: -len(suffix)])] = path
        except ValueError:
            continue
    return files


def path_debug_files(directory: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in directory.glob("*_path.png"):
        try:
            files[int(path.name.removesuffix("_path.png"))] = path
        except ValueError:
            continue
    return files


def load_action_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["step"]) - 1: row for row in rows if row.get("step")}


def load_last_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def choose_steps(steps: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if len(steps) <= max_frames:
        return steps
    indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
    return [steps[index] for index in indices]


def compose_frame(
    step: int,
    rgb_path: Path,
    depth_path: Path,
    astar_path: Path,
    action_row: dict[str, str] | None,
    path_overlay: bool,
) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB").resize(
        TOP_PANEL_SIZE,
        Image.Resampling.LANCZOS,
    )
    depth = Image.open(depth_path).convert("L").convert("RGB").resize(
        TOP_PANEL_SIZE,
        Image.Resampling.LANCZOS,
    )
    minimap_source = Image.open(astar_path).convert("RGB")
    minimap_height = round(
        minimap_source.height * COMPOSITE_WIDTH / minimap_source.width
    )
    minimap = minimap_source.resize(
        (COMPOSITE_WIDTH, minimap_height),
        Image.Resampling.LANCZOS,
    )

    top_height = ROW_HEADER_HEIGHT + TOP_PANEL_SIZE[1]
    canvas = Image.new(
        "RGB",
        (COMPOSITE_WIDTH, top_height + ROW_HEADER_HEIGHT + minimap_height),
        "black",
    )
    canvas.paste(rgb, (0, ROW_HEADER_HEIGHT))
    canvas.paste(depth, (TOP_PANEL_SIZE[0], ROW_HEADER_HEIGHT))
    canvas.paste(minimap, (0, top_height + ROW_HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), f"RGB | A* navigation | step {step:02d}", fill="white")
    draw.text(
        (TOP_PANEL_SIZE[0] + 8, 9),
        "DEPTH | UNITY SENSOR ENCODING",
        fill="white",
    )
    distance_world = action_row.get("distance_world", "") if action_row else ""
    distance_px = action_row.get("distance_px", "") if action_row else ""
    if distance_world:
        distance_label = f" | distance {float(distance_world):.2f} m"
    elif distance_px:
        distance_label = f" | distance {float(distance_px):.1f} px"
    else:
        distance_label = ""
    minimap_label = (
        f"A* PLAN{distance_label} | red agent | green target | "
        "cyan waypoint | yellow route"
        if path_overlay
        else f"FINAL POSITION{distance_label} | red agent | green target"
    )
    draw.text((8, top_height + 9), minimap_label, fill="white")
    return canvas


def write_gif(frames: list[Image.Image], output: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError("No frames were composed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    quantized = [quantize_frame_for_gif(frame) for frame in frames]
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def quantize_frame_for_gif(frame: Image.Image) -> Image.Image:
    """Reserve half of the GIF palette for depth grayscale detail."""
    color_source = frame.copy()
    color_source.paste("black", DEPTH_PANEL_BOX)
    color_quantized = color_source.quantize(
        colors=GIF_COLOR_LEVELS,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    palette = (color_quantized.getpalette() or [])[: GIF_COLOR_LEVELS * 3]
    for value in np.linspace(0, 255, GIF_DEPTH_LEVELS).round().astype(np.uint8):
        palette.extend((int(value), int(value), int(value)))
    palette.extend([0] * (768 - len(palette)))

    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(palette)
    result = frame.quantize(
        palette=palette_image,
        dither=Image.Dither.FLOYDSTEINBERG,
    )

    indices = np.asarray(result).copy()
    depth = np.asarray(frame.crop(DEPTH_PANEL_BOX).convert("L"), dtype=np.float32)
    depth_indices = GIF_COLOR_LEVELS + np.rint(
        depth * (GIF_DEPTH_LEVELS - 1) / 255.0
    ).astype(np.uint8)
    left, top, right, bottom = DEPTH_PANEL_BOX
    indices[top:bottom, left:right] = depth_indices

    result = Image.fromarray(indices)
    result.putpalette(palette)
    return result


def main() -> None:
    args = parse_args()
    rgb_files = numeric_files(args.run_dir / "astar_fp", ".png")
    depth_files = numeric_files(args.run_dir / "astar_depth", ".png")
    path_files = path_debug_files(args.run_dir / "astar_debug")
    astar_files = numeric_files(args.run_dir / "astar_minimap_target", ".png")
    astar_files.update(path_files)
    common_steps = sorted(rgb_files.keys() & depth_files.keys() & astar_files.keys())
    if not common_steps:
        raise FileNotFoundError(
            "No matching RGB/depth/A* debug frames found. Run A* with "
            "--astar_debug_viz first."
        )

    steps = choose_steps(common_steps, args.max_frames)
    action_rows = load_action_rows(args.run_dir / "astar_actions.csv")
    final_result = load_last_csv_row(args.run_dir / "results.csv")
    if steps and final_result:
        action_rows.setdefault(steps[-1], final_result)
    frames = [
        compose_frame(
            step,
            rgb_files[step],
            depth_files[step],
            astar_files[step],
            action_rows.get(step),
            step in path_files,
        )
        for step in steps
    ]
    write_gif(frames, args.output, args.duration_ms)
    print(
        f"Wrote {len(frames)} frames ({frames[0].width}x{frames[0].height}) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
