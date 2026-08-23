"""Compose A* benchmark runs into RGB/depth/top-view GIFs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nav.baselines.astar import AStarBaseline
from nav.config import UNITY_MAP_SIZE


TOP_PANEL_WIDTH = 320
COMPOSITE_WIDTH = TOP_PANEL_WIDTH * 2
ROW_HEADER_HEIGHT = 30
GIF_COLOR_LEVELS = 128
GIF_DEPTH_LEVELS = 256 - GIF_COLOR_LEVELS
PathPoint = tuple[int, int]
PlannedPath = list[PathPoint]


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
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help="Scan <root>/scene*/point*/astar/results.csv and export every run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/astar_gifs"),
        help="Batch output root. GIFs retain their scene/point directory layout.",
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


def load_planned_paths(path: Path) -> dict[int, PlannedPath]:
    if not path.is_file():
        return {}
    paths: dict[int, PlannedPath] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                step = int(record["step"])
                points = [
                    (int(point[0]), int(point[1]))
                    for point in record.get("path", [])
                ]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid A* path record at {path}:{line_number}."
                ) from exc
            paths[step] = points
    return paths


def choose_steps(steps: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if len(steps) <= max_frames:
        return steps
    indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
    return [steps[index] for index in indices]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def row_float(row: dict[str, str], key: str, default: float) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def final_status(final_result: dict[str, str]) -> tuple[bool, float, float]:
    reach_m = row_float(final_result, "reach_m", 2.0)
    distance_m = row_float(final_result, "distance_world", float("inf"))
    return distance_m <= reach_m, distance_m, reach_m


def scaled_minimap_point(
    row: dict[str, str],
    x_key: str,
    y_key: str,
    image_size: tuple[int, int],
) -> PathPoint | None:
    x = row_float(row, x_key, float("nan"))
    y = row_float(row, y_key, float("nan"))
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    width, height = image_size
    scaled_x = round(x * width / UNITY_MAP_SIZE[0])
    scaled_y = round(y * height / UNITY_MAP_SIZE[1])
    return (
        int(np.clip(scaled_x, 0, width - 1)),
        int(np.clip(scaled_y, 0, height - 1)),
    )


def build_point_to_world(
    calibration_row: dict[str, str],
    image_size: tuple[int, int],
) -> Callable[[PathPoint], tuple[float, float]]:
    init_px = row_float(calibration_row, "init_px", 0.0)
    init_py = row_float(calibration_row, "init_py", 0.0)
    init_world_x = row_float(calibration_row, "init_world_x", 0.0)
    init_world_z = row_float(calibration_row, "init_world_z", 0.0)
    target_px = row_float(calibration_row, "target_px", init_px)
    target_py = row_float(calibration_row, "target_py", init_py)
    target_world_x = row_float(
        calibration_row,
        "target_world_x",
        init_world_x,
    )
    target_world_z = row_float(
        calibration_row,
        "target_world_z",
        init_world_z,
    )
    width, height = image_size

    def point_to_world(point: PathPoint) -> tuple[float, float]:
        unity_px = float(point[0]) * UNITY_MAP_SIZE[0] / width
        unity_py = float(point[1]) * UNITY_MAP_SIZE[1] / height
        if abs(target_py - init_py) > 1e-6:
            world_x = init_world_x + (unity_py - init_py) * (
                target_world_x - init_world_x
            ) / (target_py - init_py)
        else:
            world_x = init_world_x - 0.0754061 * (unity_py - init_py)
        if abs(target_px - init_px) > 1e-6:
            world_z = init_world_z + (unity_px - init_px) * (
                target_world_z - init_world_z
            ) / (target_px - init_px)
        else:
            world_z = init_world_z - 0.06702765 * (unity_px - init_px)
        return float(world_x), float(world_z)

    return point_to_world


def replay_planned_paths(
    steps: list[int],
    minimap_files: dict[int, Path],
    action_rows: dict[int, dict[str, str]],
    calibration_row: dict[str, str],
    reach_m: float,
) -> dict[int, PlannedPath]:
    """Replay saved observations when an older run has no path log."""
    paths: dict[int, PlannedPath] = {}
    planner: AStarBaseline | None = None
    point_to_world: Callable[[PathPoint], tuple[float, float]] | None = None
    image_size: tuple[int, int] | None = None

    for step in steps:
        row = action_rows.get(step)
        minimap_path = minimap_files.get(step)
        if row is None or minimap_path is None:
            continue
        with Image.open(minimap_path) as image:
            minimap = image.convert("RGB")
            current_size = minimap.size
            minimap_rgb = np.asarray(minimap)

        if planner is None or image_size != current_size:
            scale_x = current_size[0] / UNITY_MAP_SIZE[0]
            scale_y = current_size[1] / UNITY_MAP_SIZE[1]
            planner = AStarBaseline(pixel_scale=(scale_x + scale_y) / 2.0)
            point_to_world = build_point_to_world(calibration_row, current_size)
            image_size = current_size

        curr_xy = scaled_minimap_point(row, "curr_px", "curr_py", current_size)
        target_xy = scaled_minimap_point(
            row,
            "target_px",
            "target_py",
            current_size,
        )
        curr_world_x = row_float(row, "curr_world_x", float("nan"))
        curr_world_z = row_float(row, "curr_world_z", float("nan"))
        target_world_x = row_float(row, "target_world_x", float("nan"))
        target_world_z = row_float(row, "target_world_z", float("nan"))
        if (
            curr_xy is None
            or target_xy is None
            or not np.all(
                np.isfinite(
                    [
                        curr_world_x,
                        curr_world_z,
                        target_world_x,
                        target_world_z,
                    ]
                )
            )
            or point_to_world is None
        ):
            continue

        _, _, path = planner.decide(
            minimap_rgb=minimap_rgb,
            curr_xy=curr_xy,
            target_xy=target_xy,
            agent_theta=row_float(row, "curr_direction_y", 0.0) % 360.0,
            reach_m=reach_m,
            curr_world_xz=(curr_world_x, curr_world_z),
            target_world_xz=(target_world_x, target_world_z),
            point_to_world=point_to_world,
        )
        paths[step] = [(int(point[0]), int(point[1])) for point in path]
    return paths


def draw_planned_path(minimap: Image.Image, path: PlannedPath) -> Image.Image:
    if len(path) < 2:
        return minimap
    annotated = minimap.convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    scale = max(1.0, annotated.width / UNITY_MAP_SIZE[0])
    outer_width = max(4, round(7 * scale))
    inner_width = max(2, round(4 * scale))
    draw.line(path, fill=(12, 18, 24, 220), width=outer_width, joint="curve")
    draw.line(path, fill=(255, 215, 45, 255), width=inner_width, joint="curve")
    return annotated.convert("RGB")


def success_zone_geometry(
    calibration_row: dict[str, str],
    reach_m: float,
) -> tuple[float, float, float, float]:
    target_px = row_float(calibration_row, "target_px", 0.0)
    target_py = row_float(calibration_row, "target_py", 0.0)
    init_px = row_float(calibration_row, "init_px", target_px)
    init_py = row_float(calibration_row, "init_py", target_py)
    init_world_x = row_float(calibration_row, "init_world_x", 0.0)
    init_world_z = row_float(calibration_row, "init_world_z", 0.0)
    target_world_x = row_float(calibration_row, "target_world_x", init_world_x)
    target_world_z = row_float(calibration_row, "target_world_z", init_world_z)

    delta_world_z = target_world_z - init_world_z
    delta_world_x = target_world_x - init_world_x
    pixels_per_world_z = (
        abs((target_px - init_px) / delta_world_z)
        if abs(delta_world_z) > 1e-6
        else 1.0 / 0.06702765
    )
    pixels_per_world_x = (
        abs((target_py - init_py) / delta_world_x)
        if abs(delta_world_x) > 1e-6
        else 1.0 / 0.0754061
    )
    return (
        target_px,
        target_py,
        max(1.0, reach_m * pixels_per_world_z),
        max(1.0, reach_m * pixels_per_world_x),
    )


def annotate_top_view(
    minimap: Image.Image,
    calibration_row: dict[str, str],
    final_result: dict[str, str],
) -> tuple[Image.Image, str, float]:
    success, _, reach_m = final_status(final_result)
    target_x, target_y, radius_x, radius_y = success_zone_geometry(
        calibration_row,
        reach_m,
    )

    annotated = minimap.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(
        (
            target_x - radius_x,
            target_y - radius_y,
            target_x + radius_x,
            target_y + radius_y,
        ),
        fill=(38, 190, 92, 48),
        outline=(105, 255, 155, 255),
        width=3,
    )

    status = "SUCCESS" if success else "FAILED"
    status_color = (75, 220, 120, 255) if success else (245, 90, 90, 255)
    font = load_font(22)
    text_box = draw.textbbox((0, 0), status, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    padding_x = 12
    padding_y = 8
    right = annotated.width - 12
    left = right - text_width - 2 * padding_x
    top = 12
    bottom = top + text_height + 2 * padding_y
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=4,
        fill=(10, 14, 18, 220),
        outline=status_color,
        width=3,
    )
    draw.text(
        (left + padding_x, top + padding_y - text_box[1]),
        status,
        font=font,
        fill=status_color,
    )
    return Image.alpha_composite(annotated, overlay).convert("RGB"), status, reach_m


def compose_frame(
    step: int,
    rgb_path: Path,
    depth_path: Path,
    astar_path: Path,
    action_row: dict[str, str] | None,
    path_overlay: bool,
    planned_path: PlannedPath | None,
    calibration_row: dict[str, str],
    final_result: dict[str, str],
) -> Image.Image:
    rgb_source = Image.open(rgb_path).convert("RGB")
    top_panel_size = (
        TOP_PANEL_WIDTH,
        max(1, round(rgb_source.height * TOP_PANEL_WIDTH / rgb_source.width)),
    )
    rgb = rgb_source.resize(
        top_panel_size,
        Image.Resampling.LANCZOS,
    )
    depth = Image.open(depth_path).convert("L").convert("RGB").resize(
        top_panel_size,
        Image.Resampling.LANCZOS,
    )
    minimap_source = Image.open(astar_path).convert("RGB")
    if planned_path:
        minimap_source = draw_planned_path(minimap_source, planned_path)
    minimap_source, status, reach_m = annotate_top_view(
        minimap_source,
        calibration_row,
        final_result,
    )
    minimap_height = round(
        minimap_source.height * COMPOSITE_WIDTH / minimap_source.width
    )
    minimap = minimap_source.resize(
        (COMPOSITE_WIDTH, minimap_height),
        Image.Resampling.LANCZOS,
    )

    top_height = ROW_HEADER_HEIGHT + top_panel_size[1]
    canvas = Image.new(
        "RGB",
        (COMPOSITE_WIDTH, top_height + ROW_HEADER_HEIGHT + minimap_height),
        "black",
    )
    canvas.paste(rgb, (0, ROW_HEADER_HEIGHT))
    canvas.paste(depth, (TOP_PANEL_WIDTH, ROW_HEADER_HEIGHT))
    canvas.paste(minimap, (0, top_height + ROW_HEADER_HEIGHT))
    canvas.info["depth_panel_box"] = (
        TOP_PANEL_WIDTH,
        ROW_HEADER_HEIGHT,
        COMPOSITE_WIDTH,
        ROW_HEADER_HEIGHT + top_panel_size[1],
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), f"RGB | A* navigation | step {step:02d}", fill="white")
    draw.text(
        (TOP_PANEL_WIDTH + 8, 9),
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
    view_label = "A* PLAN" if path_overlay else "TOP VIEW"
    minimap_label = (
        f"{view_label}{distance_label} | {status} | "
        f"green zone <= {reach_m:.2f} m"
    )
    if path_overlay:
        minimap_label += " | yellow route"
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
    depth_panel_box = frame.info.get("depth_panel_box")
    if not isinstance(depth_panel_box, tuple) or len(depth_panel_box) != 4:
        raise ValueError("GIF frame is missing its depth-panel bounds.")
    color_source = frame.copy()
    color_source.paste("black", depth_panel_box)
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
    depth = np.asarray(frame.crop(depth_panel_box).convert("L"), dtype=np.float32)
    depth_indices = GIF_COLOR_LEVELS + np.rint(
        depth * (GIF_DEPTH_LEVELS - 1) / 255.0
    ).astype(np.uint8)
    left, top, right, bottom = depth_panel_box
    indices[top:bottom, left:right] = depth_indices

    result = Image.fromarray(indices)
    result.putpalette(palette)
    return result


def export_run(
    run_dir: Path,
    output: Path,
    max_frames: int,
    duration_ms: int,
) -> tuple[int, tuple[int, int], str]:
    rgb_files = numeric_files(run_dir / "astar_fp", ".png")
    depth_files = numeric_files(run_dir / "astar_depth", ".png")
    path_files = path_debug_files(run_dir / "astar_debug")
    top_view_files = numeric_files(run_dir / "astar_minimap_target", ".png")
    raw_minimap_files = numeric_files(run_dir / "astar_minimap", ".png")
    astar_files = dict(top_view_files)
    astar_files.update(path_files)
    common_steps = sorted(rgb_files.keys() & depth_files.keys() & astar_files.keys())
    if not common_steps:
        raise FileNotFoundError(
            f"No matching RGB/depth/top-view frames found under {run_dir}."
        )

    steps = choose_steps(common_steps, max_frames)
    action_rows = load_action_rows(run_dir / "astar_actions.csv")
    final_result = load_last_csv_row(run_dir / "results.csv")
    if not action_rows:
        raise ValueError(f"No action rows found under {run_dir}.")
    if not final_result:
        raise ValueError(f"No final result found under {run_dir}.")
    calibration_row = action_rows[min(action_rows)]
    planned_paths = load_planned_paths(run_dir / "astar_paths.jsonl")
    missing_path_steps = [
        step
        for step in common_steps
        if step not in path_files and step not in planned_paths
    ]
    if missing_path_steps:
        replay_sources = {
            step: raw_minimap_files.get(step, top_view_files.get(step))
            for step in common_steps
        }
        replayed_paths = replay_planned_paths(
            common_steps,
            {
                step: path
                for step, path in replay_sources.items()
                if path is not None
            },
            action_rows,
            calibration_row,
            final_status(final_result)[2],
        )
        for step in missing_path_steps:
            if step in replayed_paths:
                planned_paths[step] = replayed_paths[step]
    action_rows.setdefault(steps[-1], final_result)
    frames = [
        compose_frame(
            step,
            rgb_files[step],
            depth_files[step],
            astar_files[step],
            action_rows.get(step),
            step in path_files or len(planned_paths.get(step, [])) >= 2,
            None if step in path_files else planned_paths.get(step),
            calibration_row,
            final_result,
        )
        for step in steps
    ]
    write_gif(frames, output, duration_ms)
    status = "SUCCESS" if final_status(final_result)[0] else "FAILED"
    return len(frames), (frames[0].width, frames[0].height), status


def batch_run_dirs(batch_root: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, int]:
        scene = int(path.parts[-4].removeprefix("scene"))
        point = int(path.parts[-3].removeprefix("point"))
        return scene, point

    result_files = list(batch_root.glob("scene*/point*/astar/results.csv"))
    return [path.parent for path in sorted(result_files, key=sort_key)]


def main() -> None:
    args = parse_args()
    if args.batch_root is None:
        frame_count, size, status = export_run(
            args.run_dir,
            args.output,
            args.max_frames,
            args.duration_ms,
        )
        print(
            f"Wrote {frame_count} frames ({size[0]}x{size[1]}, {status}) "
            f"to {args.output}"
        )
        return

    run_dirs = batch_run_dirs(args.batch_root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No scene*/point*/astar/results.csv files under {args.batch_root}."
        )
    for run_dir in run_dirs:
        relative = run_dir.relative_to(args.batch_root)
        output = args.output_dir / relative.parent / f"{relative.name}.gif"
        frame_count, size, status = export_run(
            run_dir,
            output,
            args.max_frames,
            args.duration_ms,
        )
        print(
            f"Wrote {frame_count} frames ({size[0]}x{size[1]}, {status}) "
            f"to {output}"
        )
    print(f"Exported {len(run_dirs)} A* run GIFs under {args.output_dir}.")


if __name__ == "__main__":
    main()
