"""Export 96 top-down multi-agent trajectory comparisons and an HTML gallery."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from nav.config import EVAL_ROI_PARAMS, EVAL_WARNING_THRESHOLD_M, UNITY_MAP_SIZE
from nav.scripts.export_llm_gallery import (
    collision_steps,
    event_points,
    load_csv_rows,
    load_font,
    numeric_files,
    target_point,
    trajectory_points,
    warning_steps,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_SIZE = (int(UNITY_MAP_SIZE[0]), int(UNITY_MAP_SIZE[1]))
PAGE_TITLE = "Top-Down Multi-Agent Trajectory Comparison"
DEFAULT_EXCLUDED_MODELS = ("openai/gpt-4o-mini",)
WARNING_COLOR = (250, 204, 21)
COLLISION_COLOR = (220, 38, 38)
TARGET_COLOR = (34, 197, 94)
# Match the current-agent marker used by the existing GIF gallery.
START_COLOR = (255, 111, 97)
PATH_ALPHA = 128
PATH_HALO_ALPHA = 42
PATH_WIDTH = 4
PATH_RENDER_SCALE = 3
EVENT_ALPHA = 160
EVENT_OUTLINE_ALPHA = 210
MODEL_COLORS = (
    (14, 165, 233),
    (168, 85, 247),
    (236, 72, 153),
    (249, 115, 22),
    (20, 184, 166),
    (37, 99, 235),
    (244, 63, 94),
    (99, 102, 241),
    (107, 114, 128),
    (132, 204, 22),
    (6, 182, 212),
    (217, 70, 239),
)
MODEL_LABELS = {
    "astar": "A*",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "deepseek/deepseek-v4-flash-vision-exp": "DeepSeek V4 Flash Vision",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "gpt-5-mini": "GPT-5 mini",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "openai/gpt-4o-mini": "GPT-4o mini",
    "qwen/qwen3.8-flash": "Qwen 3.8 Flash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay every available agent trajectory for each of the 96 "
            "scene/point tasks on a top-down map."
        )
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("analysis/cli_agent_gif_gallery/manifest.json"),
        help="Manifest produced by export_llm_gallery.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/topdown_trajectory_comparison_gallery"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=list(DEFAULT_EXCLUDED_MODELS),
        help=(
            "Model id to omit from comparisons. Repeat to exclude more models; "
            "openai/gpt-4o-mini is excluded by default."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render only the first N tasks (useful for a smoke test).",
    )
    return parser.parse_args()


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def model_color_map(items: list[dict]) -> dict[str, tuple[int, int, int]]:
    models = sorted({str(item["model"]) for item in items})
    return {
        model: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, model in enumerate(models)
    }


def exclude_models(items: list[dict], excluded: list[str]) -> list[dict]:
    excluded_set = {model.strip() for model in excluded if model.strip()}
    return [item for item in items if str(item.get("model")) not in excluded_set]


def group_manifest(items: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """Group a gallery manifest into the canonical 24 x 4 task order."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        key = (str(item["scene"]), str(item["point"]))
        grouped.setdefault(key, []).append(item)
    result = []
    for scene_number in range(1, 25):
        for point_number in range(1, 5):
            scene, point = f"scene{scene_number}", f"point{point_number}"
            task_items = sorted(
                grouped.get((scene, point), []), key=lambda item: str(item["model"])
            )
            result.append((scene, point, task_items))
    return result


def _numeric_image_paths(directory: Path) -> list[Path]:
    return [path for _, path in sorted(numeric_files(directory).items())]


def _reference_item(task_items: list[dict]) -> dict:
    for item in task_items:
        if item.get("model") == "astar":
            return item
    if not task_items:
        raise ValueError("Task has no trajectories")
    return task_items[0]


def clean_reference_map(task_items: list[dict], sample_count: int = 9) -> Image.Image:
    """Median-composite minimap frames to suppress the moving red agent arrow."""
    item = _reference_item(task_items)
    run_dir = Path(item["run_dir"])
    prefix = str(item["stream_prefix"])
    paths = _numeric_image_paths(run_dir / f"{prefix}_minimap")
    if not paths:
        paths = _numeric_image_paths(run_dir / f"{prefix}_minimap_target")
    if not paths:
        raise FileNotFoundError(f"No minimap frames under {run_dir}")
    count = min(sample_count, len(paths))
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    frames = [
        np.asarray(
            ImageOps.fit(
                Image.open(paths[index]).convert("RGB"),
                MAP_SIZE,
                method=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        for index in indices
    ]
    composite = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return Image.fromarray(composite, "RGB")


def load_trajectory(item: dict) -> dict:
    run_dir = Path(item["run_dir"])
    action_paths = sorted(run_dir.glob("*_actions.csv"))
    if len(action_paths) != 1:
        raise FileNotFoundError(
            f"Expected one *_actions.csv under {run_dir}, found {len(action_paths)}"
        )
    rows = load_csv_rows(action_paths[0])
    if not rows:
        raise ValueError(f"Empty action log: {action_paths[0]}")
    prefix = action_paths[0].name.removesuffix("_actions.csv")
    depth_steps = sorted(numeric_files(run_dir / f"{prefix}_depth", ".npy"))
    warnings = warning_steps(run_dir, prefix, rows, depth_steps)
    collisions = collision_steps(rows)
    last_index = len(rows) - 1
    return {
        "model": str(item["model"]),
        "points": trajectory_points(rows, last_index),
        "warning_points": event_points(rows, warnings, last_index),
        "collision_points": event_points(rows, collisions, last_index),
        "target": target_point(rows[-1]),
        "run_dir": str(run_dir),
    }


def _draw_dot(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    radius: int,
    fill: tuple[int, ...],
    outline: tuple[int, ...],
    width: int = 2,
) -> None:
    x, y = point
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=outline,
        width=width,
    )


def _draw_triangle(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    radius: int,
    fill: tuple[int, ...],
    outline: tuple[int, ...],
    width: int = 2,
) -> None:
    x, y = point
    draw.polygon(
        (
            (x, y - radius),
            (x - radius, y + radius),
            (x + radius, y + radius),
        ),
        fill=fill,
        outline=outline,
        width=width,
    )


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    radius: int,
    fill: tuple[int, ...],
    outline: tuple[int, ...],
    width: int = 3,
) -> None:
    x, y = point
    segments = (
        (x - radius, y - radius, x + radius, y + radius),
        (x - radius, y + radius, x + radius, y - radius),
    )
    for segment in segments:
        draw.line(segment, fill=outline, width=width + 3)
    for segment in segments:
        draw.line(segment, fill=fill, width=width)


def render_map(
    base: Image.Image,
    trajectories: list[dict],
    colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    """Draw model paths first, then warning/collision, start, and target markers."""
    result = base.convert("RGBA")
    render_size = tuple(value * PATH_RENDER_SCALE for value in MAP_SIZE)
    path_layer = Image.new("RGBA", render_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(path_layer)

    def scaled(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [
            (x * PATH_RENDER_SCALE, y * PATH_RENDER_SCALE)
            for x, y in points
        ]

    # A faint neutral halo separates coincident paths without making the
    # trajectory layer compete visually with warning/collision markers.
    for trajectory in trajectories:
        points = scaled(trajectory["points"])
        if len(points) >= 2:
            draw.line(
                points,
                fill=(8, 15, 29, PATH_HALO_ALPHA),
                width=(PATH_WIDTH + 3) * PATH_RENDER_SCALE,
                joint="curve",
            )
    for trajectory in trajectories:
        points = scaled(trajectory["points"])
        color = colors[trajectory["model"]]
        if len(points) >= 2:
            draw.line(
                points,
                fill=(*color, PATH_ALPHA),
                width=PATH_WIDTH * PATH_RENDER_SCALE,
                joint="curve",
            )
    path_layer = path_layer.resize(MAP_SIZE, Image.Resampling.LANCZOS)
    result = Image.alpha_composite(result, path_layer)

    event_layer = Image.new("RGBA", MAP_SIZE, (0, 0, 0, 0))
    event_draw = ImageDraw.Draw(event_layer)
    for trajectory in trajectories:
        outline = colors[trajectory["model"]]
        for point in trajectory["warning_points"]:
            _draw_triangle(
                event_draw,
                point,
                7,
                (*WARNING_COLOR, EVENT_ALPHA),
                (*outline, EVENT_OUTLINE_ALPHA),
                2,
            )
        for point in trajectory["collision_points"]:
            _draw_cross(
                event_draw,
                point,
                7,
                (*COLLISION_COLOR, EVENT_ALPHA),
                (*outline, EVENT_OUTLINE_ALPHA),
                3,
            )
    result = Image.alpha_composite(result, event_layer)
    draw = ImageDraw.Draw(result)

    starts = [trajectory["points"][0] for trajectory in trajectories if trajectory["points"]]
    if starts:
        _draw_dot(draw, starts[0], 12, START_COLOR, START_COLOR, 1)
    target = next(
        (trajectory["target"] for trajectory in trajectories if trajectory["target"]),
        None,
    )
    if target is not None:
        _draw_dot(draw, target, 14, TARGET_COLOR, (12, 116, 58), 3)
    return result.convert("RGB")


def render_card(
    scene: str,
    point: str,
    task_items: list[dict],
    colors: dict[str, tuple[int, int, int]],
    output_path: Path,
) -> dict:
    if not task_items:
        raise ValueError(f"No manifest entries for {scene}/{point}")
    trajectories, errors = [], []
    for item in task_items:
        try:
            trajectories.append(load_trajectory(item))
        except (FileNotFoundError, OSError, ValueError) as exc:
            errors.append(f"{item.get('model', 'unknown')}: {exc}")
    if not trajectories:
        raise ValueError(f"No readable trajectories for {scene}/{point}")

    map_image = render_map(clean_reference_map(task_items), trajectories, colors)
    legend_width = 330
    header_height = 70
    footer_height = 44
    canvas = Image.new(
        "RGB",
        (MAP_SIZE[0] + legend_width, header_height + MAP_SIZE[1] + footer_height),
        (8, 15, 29),
    )
    canvas.paste(map_image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    body_font = load_font(17)
    small_font = load_font(14)
    draw.text(
        (24, 17),
        f"{scene.upper()} / {point.upper()} · {len(trajectories)} agents",
        font=title_font,
        fill=(241, 245, 249),
    )
    panel_x = MAP_SIZE[0] + 22
    draw.text((panel_x, header_height + 18), "TRAJECTORIES", font=body_font, fill=(148, 163, 184))
    y = header_height + 54
    for trajectory in trajectories:
        color = colors[trajectory["model"]]
        draw.line((panel_x, y + 8, panel_x + 34, y + 8), fill=color, width=5)
        draw.text(
            (panel_x + 46, y - 2),
            model_label(trajectory["model"]),
            font=small_font,
            fill=(226, 232, 240),
        )
        y += 34
    y = max(y + 12, header_height + 405)
    draw.text((panel_x, y), "SAFETY EVENTS", font=body_font, fill=(148, 163, 184))
    y += 34
    _draw_triangle(draw, (panel_x + 8, y + 7), 8, WARNING_COLOR, (133, 77, 14), 2)
    draw.text((panel_x + 26, y - 2), "Warning", font=small_font, fill=(226, 232, 240))
    _draw_cross(draw, (panel_x + 130, y + 7), 7, COLLISION_COLOR, (127, 29, 29), 3)
    draw.text((panel_x + 146, y - 2), "Collision", font=small_font, fill=(226, 232, 240))
    y += 32
    _draw_dot(draw, (panel_x + 8, y + 7), 8, START_COLOR, START_COLOR, 1)
    draw.text((panel_x + 26, y - 2), "Start", font=small_font, fill=(226, 232, 240))
    _draw_dot(draw, (panel_x + 130, y + 7), 9, TARGET_COLOR, (12, 116, 58), 2)
    draw.text((panel_x + 146, y - 2), "Target", font=small_font, fill=(226, 232, 240))

    warning_count = sum(len(item["warning_points"]) for item in trajectories)
    collision_count = sum(len(item["collision_points"]) for item in trajectories)
    draw.text(
        (24, header_height + MAP_SIZE[1] + 12),
        f"Warning events: {warning_count}   ·   Collision events: {collision_count}",
        font=small_font,
        fill=(203, 213, 225),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return {
        "scene": scene,
        "scene_number": int(scene.removeprefix("scene")),
        "point": point,
        "point_number": int(point.removeprefix("point")),
        "image": output_path.name,
        "agent_count": len(trajectories),
        "models": [trajectory["model"] for trajectory in trajectories],
        "warning_events": warning_count,
        "collision_events": collision_count,
        "errors": errors,
    }


def gallery_html(items: list[dict], colors: dict[str, tuple[int, int, int]]) -> str:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    color_payload = json.dumps(
        {model: f"rgb({color[0]}, {color[1]}, {color[2]})" for model, color in colors.items()}
    ).replace("</", "<\\/")
    model_chips = "".join(
        f'<span class="chip"><i style="background:rgb({color[0]},{color[1]},{color[2]})"></i>'
        f"{escape(model_label(model))}</span>"
        for model, color in colors.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(PAGE_TITLE)}</title><style>
:root{{--bg:#07101f;--panel:#101b2f;--line:#26364e;--text:#edf2f7;--muted:#9baabd}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#14213a 0,var(--bg) 38rem);color:var(--text);font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}}
header{{max-width:1500px;margin:auto;padding:38px 24px 22px}}h1{{margin:0 0 8px;font-size:clamp(28px,4vw,48px)}}header p{{margin:0;color:var(--muted);max-width:920px}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}}.chip{{display:inline-flex;align-items:center;gap:7px;padding:5px 9px;border:1px solid var(--line);border-radius:999px;background:#0d1728;color:#dbe5f0;font-size:12px}}.chip i{{width:16px;height:4px;border-radius:3px}}
.controls{{position:sticky;top:0;z-index:3;display:flex;gap:12px;align-items:center;padding:13px 24px;background:#07101fea;border-block:1px solid var(--line);backdrop-filter:blur(10px)}}select,input{{border:1px solid #34445e;border-radius:8px;background:#101b2f;color:var(--text);padding:9px 11px}}input{{min-width:230px}}#count{{margin-left:auto;color:var(--muted)}}
main{{max-width:1500px;margin:auto;padding:24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,570px),1fr));gap:22px}}article{{overflow:hidden;border:1px solid var(--line);border-radius:14px;background:var(--panel);box-shadow:0 14px 34px #0004}}article img{{display:block;width:100%;height:auto;background:#08111f}}.meta{{display:flex;justify-content:space-between;gap:16px;padding:12px 15px;color:var(--muted)}}.meta b{{color:var(--text)}}
@media(max-width:650px){{.controls{{flex-wrap:wrap}}#count{{width:100%;margin:0}}input{{min-width:0;flex:1}}}}
</style></head><body><header><h1>{escape(PAGE_TITLE)}</h1>
<p>One top-down comparison for each of 24 scenes x 4 targets. Lines identify agents; yellow points are depth warnings, red points are inferred collisions, coral is the start, and green is the target.</p>
<div class="legend">{model_chips}</div></header>
<div class="controls"><select id="scene"><option value="all">All Scenes</option></select><input id="search" type="search" placeholder="Search scene or point"><span id="count"></span></div><main id="grid"></main>
<script>const items={payload},colors={color_payload};const scene=document.querySelector('#scene');for(let i=1;i<=24;i++)scene.insertAdjacentHTML('beforeend',`<option value="${{i}}">Scene ${{i}}</option>`);const grid=document.querySelector('#grid'),count=document.querySelector('#count'),search=document.querySelector('#search');function render(){{const s=scene.value,q=search.value.toLowerCase();const shown=items.filter(x=>(s==='all'||x.scene_number===+s)&&(`${{x.scene}} ${{x.point}}`.toLowerCase().includes(q)));grid.innerHTML=shown.map(x=>`<article><a href="images/${{x.image}}" target="_blank"><img loading="lazy" src="images/${{x.image}}" alt="${{x.scene}} ${{x.point}} top-down trajectory comparison"></a><div class="meta"><b>${{x.scene.toUpperCase()}} / ${{x.point.toUpperCase()}}</b><span>${{x.agent_count}} agents · ${{x.warning_events}} warnings · ${{x.collision_events}} collisions</span></div></article>`).join('');count.textContent=`Showing ${{shown.length}} / ${{items.length}} comparisons`}}scene.addEventListener('change',render);search.addEventListener('input',render);render();</script></body></html>"""


def main() -> None:
    args = parse_args()
    source_manifest = args.source_manifest
    if not source_manifest.is_absolute():
        source_manifest = REPO_ROOT / source_manifest
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    items = exclude_models(
        json.loads(source_manifest.read_text(encoding="utf-8")),
        args.exclude_model,
    )
    colors = model_color_map(items)
    tasks = group_manifest(items)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for scene, point, task_items in tasks:
            filename = f"{scene}_{point}.png"
            future = executor.submit(
                render_card,
                scene,
                point,
                task_items,
                colors,
                image_dir / filename,
            )
            futures[future] = (scene, point)
        for future in as_completed(futures):
            scene, point = futures[future]
            item = future.result()
            completed.append(item)
            print(
                f"[{len(completed):02d}/{len(tasks):02d}] {scene}/{point}: "
                f"{item['agent_count']} agents, {item['warning_events']} warnings, "
                f"{item['collision_events']} collisions",
                flush=True,
            )

    completed.sort(key=lambda item: (item["scene_number"], item["point_number"]))
    (output_dir / "manifest.json").write_text(
        json.dumps(completed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        gallery_html(completed, colors), encoding="utf-8"
    )
    print(f"Wrote {len(completed)} comparisons to {output_dir}")


if __name__ == "__main__":
    main()
