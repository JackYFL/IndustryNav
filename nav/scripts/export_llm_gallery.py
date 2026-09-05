"""Export navigation trajectories as annotated GIFs and an English HTML gallery."""

from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from nav.config import (
    EVAL_COLLISION_MIN_FORWARD_RATIO,
    EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M,
    UNITY_MAP_SIZE,
)
from nav.eval.collision import compute_collision_rate
from nav.eval.metrics import (
    compute_success_at_thresholds,
    compute_success_efficiency_distance,
)
from nav.eval.warning import WarningDetector


REPO_ROOT = Path(__file__).resolve().parents[2]
CANVAS_WIDTH = 720
PANEL_SIZE = 360
HEADER_HEIGHT = 64
MAP_HEIGHT = 428
CANVAS_HEIGHT = HEADER_HEIGHT + PANEL_SIZE + MAP_HEIGHT
TARGET_RADIUS_CANONICAL = 14
WARNING_RADIUS_CANONICAL = 9
COLLISION_RADIUS_CANONICAL = 6
TRAJECTORY_COLOR = (56, 189, 248)
TRAJECTORY_POINT_COLOR = (150, 225, 250)
TRAJECTORY_LINE_WIDTH_CANONICAL = 10
TRAJECTORY_POINT_RADIUS_CANONICAL = 6
CURRENT_POSITION_RADIUS_CANONICAL = 12
WARNING_COLOR = (250, 204, 21)
WARNING_OUTLINE_COLOR = (133, 77, 14)
COLLISION_COLOR = (220, 38, 38)
COLLISION_OUTLINE_COLOR = (127, 29, 29)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RGB/depth/minimap LLM trajectories and an HTML gallery."
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=None,
        help=(
            "Run-directory glob relative to the repo root. Repeat this option "
            "to combine multiple result sets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/gpt-5.6-sol_gif_gallery"),
    )
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--duration-ms", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Refresh the English HTML page from the existing manifest without regenerating GIFs.",
    )
    parser.add_argument(
        "--gallery-title",
        default="",
        help="Optional page title. Derived from the result models when omitted.",
    )
    args = parser.parse_args()
    if args.html_only and (args.input_glob or args.limit):
        parser.error("--html-only uses the existing manifest; do not combine it with --input-glob or --limit.")
    return args


def numeric_files(directory: Path, suffix: str = ".png") -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        try:
            files[int(path.name[: -len(suffix)])] = path
        except ValueError:
            continue
    return files


def choose_steps(steps: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if len(steps) <= max_frames:
        return steps
    indices = np.linspace(0, len(steps) - 1, max_frames, dtype=int)
    return [steps[index] for index in indices]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def collision_steps(rows: list[dict[str, str]]) -> set[int]:
    result: set[int] = set()
    for index, (current, following) in enumerate(zip(rows, rows[1:])):
        try:
            move = float(current["move"])
            current_step = int(float(current["step"]))
            following_step = int(float(following["step"]))
            current_pos = (
                float(current["curr_world_x"]),
                float(current["curr_world_z"]),
            )
            following_pos = (
                float(following["curr_world_x"]),
                float(following["curr_world_z"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if move <= 0 or following_step != current_step + 1:
            continue
        actual = math.hypot(
            following_pos[0] - current_pos[0],
            following_pos[1] - current_pos[1],
        )
        expected = move * EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
        if actual < expected * EVAL_COLLISION_MIN_FORWARD_RATIO:
            result.add(index)
    return result


def rotation_step_count(rows: list[dict[str, str]]) -> int:
    """Count steps that command any yaw change, including drive-and-turn."""
    count = 0
    for row in rows:
        try:
            look = float(row.get("look", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if abs(look) > 1e-8:
            count += 1
    return count


def warning_steps(
    run_dir: Path,
    stream_prefix: str,
    rows: list[dict[str, str]],
    common_steps: list[int],
) -> set[int]:
    detector = WarningDetector()
    result: set[int] = set()
    for step in common_steps:
        depth_path = run_dir / f"{stream_prefix}_depth" / f"{step}.npy"
        if not depth_path.is_file():
            continue
        try:
            depth = np.load(depth_path).astype(np.float32)
            move = float(rows[step].get("move", 0.0)) if step < len(rows) else 0.0
        except (OSError, TypeError, ValueError):
            continue
        if detector.detect(depth, move_command=move)["warning"] == "yes":
            result.add(step)
    return result


def trajectory_points(rows: list[dict[str, str]], step: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for row in rows[: step + 1]:
        try:
            points.append((int(float(row["curr_px"])), int(float(row["curr_py"]))))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def event_points(
    rows: list[dict[str, str]],
    event_steps: set[int],
    step: int,
) -> list[tuple[int, int]]:
    """Return cumulative event positions through the current replay step."""
    points: list[tuple[int, int]] = []
    for event_step in sorted(index for index in event_steps if index <= step):
        if event_step < 0 or event_step >= len(rows):
            continue
        try:
            row = rows[event_step]
            points.append((int(float(row["curr_px"])), int(float(row["curr_py"]))))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def target_point(row: dict[str, str]) -> tuple[int, int] | None:
    try:
        return (int(float(row["target_px"])), int(float(row["target_py"])))
    except (KeyError, TypeError, ValueError):
        return None


def scale_trajectory_points(
    points: list[tuple[int, int]],
    image_size: tuple[int, int],
    canonical_size: tuple[float, float] = UNITY_MAP_SIZE,
) -> list[tuple[int, int]]:
    """Map canonical 862x512 CSV coordinates into a saved minimap image."""
    scale_x = float(image_size[0]) / float(canonical_size[0])
    scale_y = float(image_size[1]) / float(canonical_size[1])
    return [
        (int(round(x * scale_x)), int(round(y * scale_y)))
        for x, y in points
    ]


def add_trajectory(
    image: Image.Image,
    points: list[tuple[int, int]],
    target: tuple[int, int] | None = None,
    warning_points: list[tuple[int, int]] | None = None,
    collision_points: list[tuple[int, int]] | None = None,
) -> Image.Image:
    result = image.convert("RGB")
    warning_points = warning_points or []
    collision_points = collision_points or []
    if not points and target is None and not warning_points and not collision_points:
        return result
    points = scale_trajectory_points(points, result.size)
    warning_points = scale_trajectory_points(warning_points, result.size)
    collision_points = scale_trajectory_points(collision_points, result.size)
    draw = ImageDraw.Draw(result)
    marker_scale = min(
        result.size[0] / UNITY_MAP_SIZE[0],
        result.size[1] / UNITY_MAP_SIZE[1],
    )
    trajectory_width = max(
        2,
        int(round(TRAJECTORY_LINE_WIDTH_CANONICAL * marker_scale)),
    )
    trajectory_point_radius = max(
        2,
        int(round(TRAJECTORY_POINT_RADIUS_CANONICAL * marker_scale)),
    )
    current_position_radius = max(
        4,
        int(round(CURRENT_POSITION_RADIUS_CANONICAL * marker_scale)),
    )
    if len(points) >= 2:
        draw.line(
            points,
            fill=TRAJECTORY_COLOR,
            width=trajectory_width,
            joint="curve",
        )
    for point in points[:: max(1, len(points) // 12)]:
        x, y = point
        draw.ellipse(
            (
                x - trajectory_point_radius,
                y - trajectory_point_radius,
                x + trajectory_point_radius,
                y + trajectory_point_radius,
            ),
            fill=TRAJECTORY_POINT_COLOR,
        )
    if target is not None:
        x, y = scale_trajectory_points([target], result.size)[0]
        radius = max(6, int(round(TARGET_RADIUS_CANONICAL * marker_scale)))
        outline_width = max(2, int(round(3 * marker_scale)))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(34, 197, 94),
            outline=(12, 116, 58),
            width=outline_width,
        )
    if points:
        x, y = points[-1]
        draw.ellipse(
            (
                x - current_position_radius,
                y - current_position_radius,
                x + current_position_radius,
                y + current_position_radius,
            ),
            fill=(255, 111, 97),
        )
    warning_radius = max(5, int(round(WARNING_RADIUS_CANONICAL * marker_scale)))
    collision_radius = max(4, int(round(COLLISION_RADIUS_CANONICAL * marker_scale)))
    event_outline_width = max(1, int(round(2 * marker_scale)))
    for x, y in warning_points:
        draw.ellipse(
            (
                x - warning_radius,
                y - warning_radius,
                x + warning_radius,
                y + warning_radius,
            ),
            fill=WARNING_COLOR,
            outline=WARNING_OUTLINE_COLOR,
            width=event_outline_width,
        )
    for x, y in collision_points:
        draw.ellipse(
            (
                x - collision_radius,
                y - collision_radius,
                x + collision_radius,
                y + collision_radius,
            ),
            fill=COLLISION_COLOR,
            outline=COLLISION_OUTLINE_COLOR,
            width=event_outline_width,
        )
    return result


def panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def draw_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    text_color: tuple[int, int, int] | str = "white",
) -> None:
    font = load_font(13, bold=True)
    box = draw.textbbox(xy, text, font=font)
    padded = (box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4)
    draw.rounded_rectangle(padded, radius=6, fill=color)
    draw.text(xy, text, fill=text_color, font=font)


def compose_frame(
    run_dir: Path,
    step: int,
    rgb_path: Path,
    depth_path: Path,
    minimap_path: Path,
    row: dict[str, str],
    result_row: dict[str, str],
    trail: list[tuple[int, int]],
    warning_points: list[tuple[int, int]],
    collision_points: list[tuple[int, int]],
    collisions: set[int],
    warnings: set[int],
    total_frames: int,
) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (12, 18, 29))
    rgb = panel(Image.open(rgb_path), (PANEL_SIZE, PANEL_SIZE))
    depth = panel(Image.open(depth_path), (PANEL_SIZE, PANEL_SIZE))
    minimap_source = add_trajectory(
        Image.open(minimap_path),
        trail,
        target_point(row),
        warning_points,
        collision_points,
    )
    minimap = panel(minimap_source, (CANVAS_WIDTH, MAP_HEIGHT))
    canvas.paste(rgb, (0, HEADER_HEIGHT))
    canvas.paste(depth, (PANEL_SIZE, HEADER_HEIGHT))
    canvas.paste(minimap, (0, HEADER_HEIGHT + PANEL_SIZE))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(19, bold=True)
    detail_font = load_font(14)
    scene, point, _ = run_identity(run_dir)
    scene = scene.upper()
    point = point.upper()
    final_distance = float(result_row.get("distance_world", "inf"))
    success = final_distance <= float(result_row.get("reach_m", 2.0))
    status = "SUCCESS" if success else "FAILED"
    status_color = (20, 150, 105) if success else (193, 63, 63)
    model = result_row.get("model", run_dir.parts[-2]).upper()
    draw.text((14, 8), f"{scene} / {point}   {model}", fill="white", font=title_font)
    draw_badge(draw, (CANVAS_WIDTH - 94, 12), status, status_color)

    action = row.get("action", "-").upper()
    distance = row.get("distance_world", "")
    distance_label = f"{float(distance):.2f} m" if distance else "-"
    draw.text(
        (14, 38),
        f"frame {step + 1}/{total_frames}   action {action}   distance {distance_label}",
        fill=(184, 199, 220),
        font=detail_font,
    )

    label_font = load_font(14, bold=True)
    draw.rectangle((0, HEADER_HEIGHT, 112, HEADER_HEIGHT + 26), fill=(7, 12, 20))
    draw.text((10, HEADER_HEIGHT + 5), "EGO RGB", fill="white", font=label_font)
    draw.rectangle(
        (PANEL_SIZE, HEADER_HEIGHT, PANEL_SIZE + 102, HEADER_HEIGHT + 26),
        fill=(7, 12, 20),
    )
    draw.text((PANEL_SIZE + 10, HEADER_HEIGHT + 5), "DEPTH", fill="white", font=label_font)
    map_y = HEADER_HEIGHT + PANEL_SIZE
    draw.rectangle((0, map_y, 452, map_y + 26), fill=(7, 12, 20))
    draw.text((10, map_y + 5), "MINIMAP + TRAJECTORY", fill="white", font=label_font)
    draw.ellipse((225, map_y + 8, 235, map_y + 18), fill=WARNING_COLOR)
    draw.text((241, map_y + 5), "WARNING", fill=(255, 228, 122), font=label_font)
    draw.ellipse((330, map_y + 8, 340, map_y + 18), fill=COLLISION_COLOR)
    draw.text((346, map_y + 5), "COLLISION", fill=(255, 169, 169), font=label_font)
    if step in warnings:
        warning_x = CANVAS_WIDTH - 206 if step in collisions else CANVAS_WIDTH - 100
        draw_badge(
            draw,
            (warning_x, HEADER_HEIGHT + 12),
            "WARNING",
            WARNING_COLOR,
            (66, 40, 5),
        )
    if step in collisions:
        draw_badge(
            draw,
            (CANVAS_WIDTH - 104, HEADER_HEIGHT + 12),
            "COLLISION",
            COLLISION_COLOR,
        )
    return canvas


def export_run(
    run_dir: Path,
    output_path: Path,
    max_frames: int,
    duration_ms: int,
) -> dict:
    action_paths = sorted(run_dir.glob("*_actions.csv"))
    if len(action_paths) != 1:
        raise FileNotFoundError(
            f"Expected one *_actions.csv under {run_dir}, found {len(action_paths)}"
        )
    actions_path = action_paths[0]
    stream_prefix = actions_path.name.removesuffix("_actions.csv")
    rgb_files = numeric_files(run_dir / f"{stream_prefix}_fp")
    depth_files = numeric_files(run_dir / f"{stream_prefix}_depth")
    minimap_files = numeric_files(run_dir / f"{stream_prefix}_minimap_target")
    common_steps = sorted(rgb_files.keys() & depth_files.keys() & minimap_files.keys())
    if not common_steps:
        raise FileNotFoundError(f"No matching frames under {run_dir}")

    rows = load_csv_rows(actions_path)
    result_rows = load_csv_rows(run_dir / "results.csv")
    if not rows or not result_rows:
        raise ValueError(f"Missing actions or results under {run_dir}")
    result_row = result_rows[-1]
    selected_steps = choose_steps(common_steps, max_frames)
    collisions = collision_steps(rows)
    warnings = warning_steps(run_dir, stream_prefix, rows, common_steps)
    frames = [
        compose_frame(
            run_dir,
            step,
            rgb_files[step],
            depth_files[step],
            minimap_files[step],
            rows[step] if step < len(rows) else {},
            result_row,
            trajectory_points(rows, step),
            event_points(rows, warnings, step),
            event_points(rows, collisions, step),
            collisions,
            warnings,
            len(common_steps),
        )
        for step in selected_steps
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized = [
        frame.quantize(
            colors=256,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    success, steps, distance_ratio = compute_success_efficiency_distance(
        actions_path
    )
    final_distance = float(result_row["distance_world"])
    success = int(final_distance <= float(result_row.get("reach_m", 2.0)))
    threshold_success = compute_success_at_thresholds(final_distance)
    forward_steps, collision_count, collision_rate = compute_collision_rate(
        actions_path
    )
    action_counts: dict[str, int] = {}
    for row in rows:
        action = row.get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    scene, point, run_name = run_identity(run_dir)
    return {
        "scene": scene,
        "scene_number": int(scene.removeprefix("scene")),
        "point": point,
        "model": result_row.get("model", run_name),
        "run_name": run_name,
        "stream_prefix": stream_prefix,
        "success": success,
        **threshold_success,
        "steps": int(float(result_row.get("steps_taken", steps))),
        "final_distance_m": round(final_distance, 3),
        "distance_ratio": round(float(distance_ratio), 5),
        "forward_steps": forward_steps,
        "collision_steps": collision_count,
        "collision_rate": round(collision_rate, 5),
        "warning_steps": len(warnings),
        "warning_frames": len(common_steps),
        "warning_rate": round(len(warnings) / len(common_steps), 5),
        "rotation_steps": rotation_step_count(rows),
        "action_steps": len(rows),
        "action_counts": action_counts,
        "gif": output_path.name,
        "run_dir": str(run_dir),
    }


def gallery_html(items: list[dict], gallery_title: str) -> str:
    gallery_title = escape(gallery_title)
    totals = {
        "runs": len(items),
        "successes": sum(item["success"] for item in items),
        "forward": sum(item["forward_steps"] for item in items),
        "collisions": sum(item["collision_steps"] for item in items),
        "warning_frames": sum(item["warning_frames"] for item in items),
        "warnings": sum(item["warning_steps"] for item in items),
        "rotations": sum(item.get("rotation_steps", 0) for item in items),
        "actions": sum(
            item.get("action_steps", sum(item["action_counts"].values()))
            for item in items
        ),
    }
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    summary = json.dumps(totals).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{gallery_title} · IndustryNav GIF Gallery</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--panel2:#13243b;--text:#edf4ff;--muted:#91a5c2;--line:#223a59;--good:#3ddc97;--bad:#ff6b6b;--warn:#ffbf47;--cyan:#5bc0eb}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#102844 0,#07111f 46%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
.shell{{max-width:1500px;margin:auto;padding:34px 28px 80px}}h1{{margin:0;font-size:31px;letter-spacing:-.03em}}.sub{{color:var(--muted);margin:7px 0 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:18px}}.kpi{{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:15px 17px}}.kpi b{{display:block;font-size:25px}}.kpi span{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}
.toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;padding:12px;margin:0 0 18px;background:#091627e8;backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:13px}}select,input{{color:var(--text);background:#0c1b2e;border:1px solid #294461;border-radius:9px;padding:9px 11px}}input{{min-width:220px}}#count{{margin-left:auto;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}.card{{overflow:hidden;background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 14px 35px #0004}}.card.success{{border-color:#246d55}}.visual{{display:block;aspect-ratio:720/852;background:#050b13}}.visual img{{display:block;width:100%;height:100%;object-fit:cover}}.body{{padding:13px 15px 15px}}.title{{display:flex;align-items:center;gap:9px;font-size:16px;font-weight:750}}.badge{{font-size:10px;padding:3px 7px;border-radius:99px;letter-spacing:.08em}}.success .badge{{color:#8bf4c9;background:#173f36}}.failure .badge{{color:#ffabab;background:#4a232a}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:11px}}.metric{{background:#0a1727;border-radius:8px;padding:8px}}.metric b{{display:block;font-size:14px}}.metric span{{font-size:10px;color:var(--muted)}}
.empty{{display:none;text-align:center;color:var(--muted);padding:70px}}@media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,1fr)}}.kpis{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{.shell{{padding:22px 14px}}.grid{{grid-template-columns:1fr}}.kpis{{grid-template-columns:repeat(2,1fr)}}.toolbar{{flex-wrap:wrap;position:static}}#count{{margin-left:0;width:100%}}}}
</style></head><body><main class="shell">
<h1>{gallery_title} · Navigation Replay</h1><p class="sub">RGB / egocentric depth with warning and collision overlays / minimap trajectory</p>
<section class="kpis" id="kpis"></section>
<section class="toolbar"><select id="model"><option value="all">All Models</option></select><select id="scene"><option value="all">All Scenes</option></select><select id="outcome"><option value="all">All Results</option><option value="success">Success Only</option><option value="failure">Failures Only</option></select><select id="sort"><option value="scene">Scene Order</option><option value="distance">Final Distance</option><option value="collision">Highest Collision Rate</option><option value="warning">Highest Warning Rate</option></select><input id="search" placeholder="Search model / scene / point"><span id="count"></span></section>
<section class="grid" id="grid"></section><p class="empty" id="empty">No trajectories match the current filters.</p>
</main><script>
const items={payload}, totals={summary};
const pct=x=>`${{(100*x).toFixed(1)}}%`;
function updateKpis(rows){{const t={{runs:rows.length,success2:rows.reduce((n,x)=>n+x.success_at_2m,0),success5:rows.reduce((n,x)=>n+x.success_at_5m,0),success10:rows.reduce((n,x)=>n+x.success_at_10m,0),forward:rows.reduce((n,x)=>n+x.forward_steps,0),collisions:rows.reduce((n,x)=>n+x.collision_steps,0),warning_frames:rows.reduce((n,x)=>n+x.warning_frames,0),warnings:rows.reduce((n,x)=>n+x.warning_steps,0),rotations:rows.reduce((n,x)=>n+(x.rotation_steps||0),0),actions:rows.reduce((n,x)=>n+(x.action_steps??Object.values(x.action_counts).reduce((a,b)=>a+b,0)),0)}};const sr=(n)=>`${{n}}/${{t.runs}} · ${{pct(n/Math.max(1,t.runs))}}`;document.querySelector('#kpis').innerHTML=[['Runs',t.runs],['Success@2m',sr(t.success2)],['Success@5m',sr(t.success5)],['Success@10m',sr(t.success10)],['Forward CR',`${{t.collisions}}/${{t.forward}} · ${{pct(t.collisions/Math.max(1,t.forward))}}`],['Warning Rate',`${{t.warnings}}/${{t.warning_frames}} · ${{pct(t.warnings/Math.max(1,t.warning_frames))}}`],['Rotation Ratio',pct(t.rotations/Math.max(1,t.actions))]].map(([a,b])=>`<article class="kpi"><b>${{b}}</b><span>${{a}}</span></article>`).join('');}}
const modelSelect=document.querySelector('#model');[...new Set(items.map(x=>x.model))].sort().forEach(model=>modelSelect.insertAdjacentHTML('beforeend',`<option value="${{model}}">${{model}}</option>`));
const sceneSelect=document.querySelector('#scene');for(let i=1;i<=24;i++)sceneSelect.insertAdjacentHTML('beforeend',`<option value="${{i}}">Scene ${{i}}</option>`);
const grid=document.querySelector('#grid'), count=document.querySelector('#count'), empty=document.querySelector('#empty');let observer;
function render(){{const model=modelSelect.value,scene=sceneSelect.value,outcome=document.querySelector('#outcome').value,q=document.querySelector('#search').value.toLowerCase(),sort=document.querySelector('#sort').value;let shown=items.filter(x=>(model==='all'||x.model===model)&&(scene==='all'||x.scene_number===+scene)&&(outcome==='all'||(outcome==='success')===!!x.success)&&(`${{x.model}} ${{x.scene}} ${{x.point}}`.toLowerCase().includes(q)));shown.sort((a,b)=>sort==='distance'?a.final_distance_m-b.final_distance_m:sort==='collision'?b.collision_rate-a.collision_rate:sort==='warning'?b.warning_rate-a.warning_rate:(a.model.localeCompare(b.model)||a.scene_number-b.scene_number||parseInt(a.point.slice(5))-parseInt(b.point.slice(5))));grid.innerHTML=shown.map(x=>`<article class="card ${{x.success?'success':'failure'}}"><a class="visual" href="gifs/${{x.gif}}" target="_blank"><img loading="lazy" data-src="gifs/${{x.gif}}" alt="${{x.model}} ${{x.scene}} ${{x.point}} navigation replay"></a><div class="body"><div class="title">${{x.model}} · ${{x.scene.toUpperCase()}} / ${{x.point.toUpperCase()}} <span class="badge">${{x.success?'SUCCESS':'FAILED'}}</span></div><div class="metrics"><div class="metric"><b>${{x.final_distance_m.toFixed(2)}} m</b><span>FINAL DIST</span></div><div class="metric"><b>${{pct(x.distance_ratio)}}</b><span>PROGRESS</span></div><div class="metric"><b>${{pct(x.collision_rate)}}</b><span>COLLISION</span></div><div class="metric"><b>${{pct(x.warning_rate)}}</b><span>WARNING</span></div></div></div></article>`).join('');count.textContent=`Showing ${{shown.length}} / ${{items.length}}`;empty.style.display=shown.length?'none':'block';updateKpis(shown);if(observer)observer.disconnect();observer=new IntersectionObserver(entries=>entries.forEach(e=>{{if(e.isIntersecting){{const img=e.target;if(!img.src)img.src=img.dataset.src;observer.unobserve(img)}}}}),{{rootMargin:'500px'}});document.querySelectorAll('img[data-src]').forEach(img=>observer.observe(img));}}
['model','scene','outcome','sort','search'].forEach(id=>document.querySelector('#'+id).addEventListener(id==='search'?'input':'change',render));render();
</script></body></html>"""


def run_identity(run_dir: Path) -> tuple[str, str, str]:
    """Return ``(scene, point, run_name)`` for seeded or unseeded run layouts."""
    parts = run_dir.parts
    for index in range(len(parts) - 1):
        scene = parts[index]
        point = parts[index + 1]
        if (
            scene.startswith("scene")
            and scene.removeprefix("scene").isdigit()
            and point.startswith("point")
            and point.removeprefix("point").isdigit()
        ):
            run_name = parts[index + 2] if index + 2 < len(parts) else run_dir.name
            return scene, point, run_name
    raise ValueError(f"Cannot identify scene/point from run directory: {run_dir}")


def discover_run_dirs(patterns: str | list[str]) -> list[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]

    def sort_key(path: Path) -> tuple[str, int, int]:
        scene, point, run_name = run_identity(path)
        return (
            run_name,
            int(scene.removeprefix("scene")),
            int(point.removeprefix("point")),
        )

    matched = {
        path.resolve()
        for pattern in patterns
        for path in REPO_ROOT.glob(pattern)
        if path.is_dir()
    }
    return sorted(matched, key=sort_key)


def write_gallery_page(output_dir: Path, items: list[dict], gallery_title: str = "") -> Path:
    """Write the English UI while preserving the original result metadata."""
    models = sorted({item["model"] for item in items})
    title = gallery_title.strip()
    if not title:
        title = models[0] if len(models) == 1 else "Navigation Baseline Comparison"
    index_path = output_dir / "index.html"
    index_path.write_text(gallery_html(items, title), encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.html_only:
        manifest_path = output_dir / "manifest.json"
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = write_gallery_page(output_dir, items, args.gallery_title)
        print(f"Gallery refreshed: {index_path} ({len(items)} existing trajectories)")
        return
    input_globs = args.input_glob or ["outputs/scene*/point*/gpt-5.6-sol/seed0"]
    run_dirs = discover_run_dirs(input_globs)
    if args.limit > 0:
        run_dirs = run_dirs[: args.limit]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched {input_globs!r}")

    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    pair_counts: dict[tuple[str, str], int] = {}
    for run_dir in run_dirs:
        scene, point, _ = run_identity(run_dir)
        pair = (scene, point)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for run_dir in run_dirs:
            scene, point, run_name = run_identity(run_dir)
            filename = f"{scene}_{point}.gif"
            if pair_counts[(scene, point)] > 1:
                filename = f"{run_name}__{filename}"
            output_path = gifs_dir / filename
            future = executor.submit(
                export_run,
                run_dir,
                output_path,
                args.max_frames,
                args.duration_ms,
            )
            futures[future] = run_dir
        for completed, future in enumerate(as_completed(futures), start=1):
            run_dir = futures[future]
            item = future.result()
            items.append(item)
            print(
                f"[{completed}/{len(run_dirs)}] {item['scene']}/{item['point']} "
                f"success={item['success']} -> {item['gif']}",
                flush=True,
            )

    items.sort(
        key=lambda item: (
            item["model"],
            item["scene_number"],
            int(item["point"][5:]),
        )
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Gallery: {write_gallery_page(output_dir, items, args.gallery_title)}")


if __name__ == "__main__":
    main()
