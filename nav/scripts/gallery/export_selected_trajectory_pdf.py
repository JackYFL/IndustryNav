"""Export a publication-style, multi-page trajectory comparison PDF.

The Unity minimap is embedded at its native resolution.  Trajectories, event
markers, legend swatches, and all typography remain vector objects in the PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nav.config import UNITY_MAP_SIZE
from nav.scripts.gallery.export_topdown_comparison_gallery import (
    clean_reference_map,
    load_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MAP_WIDTH, MAP_HEIGHT = (int(UNITY_MAP_SIZE[0]), int(UNITY_MAP_SIZE[1]))

SELECTED_MODELS = (
    "claude-sonnet-5",
    "gpt-5-mini",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "google/gemini-3.8-flash",
    "qwen/qwen3.8-flash",
    "minimax/minimax-m3",
    "deepseek/deepseek-v4-flash-vision-exp",
    "z-ai/glm-5.3-flash",
)

BASELINE_MODELS = (
    "astar-static",
    "pointgoal-dagger-static",
    "pointgoal-ppo-static",
)

DYNAMIC_BASELINE_MODELS = (
    "astar",
    "pointgoal-dagger-all24-round1",
    "pointgoal-ppo-mixed400-visual-replay",
)

MODEL_LABELS = {
    "claude-sonnet-5": "Claude Sonnet 5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-6-astra": "GPT-6 Astra",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "qwen/qwen3.8-flash": "Qwen 3.8 Flash",
    "minimax/minimax-m3": "MiniMax M3",
    "deepseek/deepseek-v4-flash-vision-exp": "DeepSeek V4 Flash",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
}

# Stable high-contrast palette.  The same model keeps the same color on every
# page and in future full-gallery exports.
MODEL_COLORS = {
    "claude-sonnet-5": "#4CC9F0",
    "gpt-5-mini": "#F72585",
    "gpt-5.6-luna": "#FF9F1C",
    "gpt-5.6-terra": "#2EC4B6",
    "gpt-5.6-sol": "#9B5DE5",
    "gpt-6-astra": "#E63946",
    "google/gemini-3.8-flash": "#3A86FF",
    "qwen/qwen3.8-flash": "#8AC926",
    "minimax/minimax-m3": "#FFCA3A",
    "deepseek/deepseek-v4-flash-vision-exp": "#00B4D8",
    "z-ai/glm-5.3-flash": "#B8B8FF",
}

# Each model also has a unique dash rhythm.  This makes overlapping paths
# distinguishable in grayscale and separates the visually similar cyan/blue
# and orange/yellow pairs without changing the established color mapping.
MODEL_LINE_STYLES = {
    "claude-sonnet-5": "solid",
    "gpt-5-mini": (0, (6, 2.5)),
    "gpt-5.6-luna": (0, (3.8, 1.8)),
    "gpt-5.6-terra": (0, (1, 1.5)),
    "gpt-5.6-sol": (0, (5, 1.6, 1, 1.6)),
    "gpt-6-astra": (0, (3, 1.4, 1, 1.4)),
    "google/gemini-3.8-flash": (0, (5.5, 1.5, 2, 1.5)),
    "qwen/qwen3.8-flash": (0, (2, 2.5)),
    "minimax/minimax-m3": (0, (4, 1.3, 1, 1.3, 1, 1.3)),
    "deepseek/deepseek-v4-flash-vision-exp": (0, (1.8, 1, 1.8, 2.5)),
    "z-ai/glm-5.3-flash": (0, (5.5, 1.3, 1, 1.3, 1, 1.3)),
}

MODEL_LINE_STYLE_NAMES = {
    "claude-sonnet-5": "solid",
    "gpt-5-mini": "long dash",
    "gpt-5.6-luna": "medium dash",
    "gpt-5.6-terra": "dot",
    "gpt-5.6-sol": "dash-dot",
    "gpt-6-astra": "short dash-dot",
    "google/gemini-3.8-flash": "long-short dash",
    "qwen/qwen3.8-flash": "short dash",
    "minimax/minimax-m3": "dash-dot-dot",
    "deepseek/deepseek-v4-flash-vision-exp": "paired dash",
    "z-ai/glm-5.3-flash": "long dash-dot-dot",
}

BASELINE_LABELS = {
    "astar-static": "A*",
    "pointgoal-dagger-static": "DAgger",
    "pointgoal-ppo-static": "PPO",
}

BASELINE_COLORS = {
    "astar-static": "#2563EB",
    "pointgoal-dagger-static": "#F97316",
    "pointgoal-ppo-static": "#8B5CF6",
}

BASELINE_LINE_STYLES = {
    "astar-static": "solid",
    "pointgoal-dagger-static": (0, (6, 2.2)),
    "pointgoal-ppo-static": (0, (1.4, 1.8)),
}

BASELINE_LINE_STYLE_NAMES = {
    "astar-static": "solid",
    "pointgoal-dagger-static": "long dash",
    "pointgoal-ppo-static": "dot",
}

DYNAMIC_BASELINE_LABELS = {
    "astar": "A*",
    "pointgoal-dagger-all24-round1": "DAgger",
    "pointgoal-ppo-mixed400-visual-replay": "PPO",
}

DYNAMIC_BASELINE_COLORS = {
    "astar": "#2563EB",
    "pointgoal-dagger-all24-round1": "#F97316",
    "pointgoal-ppo-mixed400-visual-replay": "#8B5CF6",
}

DYNAMIC_BASELINE_LINE_STYLES = {
    "astar": "solid",
    "pointgoal-dagger-all24-round1": (0, (6, 2.2)),
    "pointgoal-ppo-mixed400-visual-replay": (0, (1.4, 1.8)),
}

DYNAMIC_BASELINE_LINE_STYLE_NAMES = {
    "astar": "solid",
    "pointgoal-dagger-all24-round1": "long dash",
    "pointgoal-ppo-mixed400-visual-replay": "dot",
}

PAGE_COLOR = "#ffffff"
PANEL_COLOR = "#ffffff"
PRIMARY_TEXT = "#111827"
MUTED_TEXT = "#475569"
WARNING_COLOR = "#facc15"
COLLISION_COLOR = "#ef4444"
START_COLOR = "#ff6f61"
START_OUTLINE_COLOR = "#dc2626"
TARGET_COLOR = "#22c55e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        choices=("agents", "baselines", "baselines-dynamic"),
        default="agents",
        help="Select the 11-agent comparison or the static A*/DAgger/PPO comparison.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("analysis/cli_agent_gif_gallery/manifest.json"),
    )
    parser.add_argument("--scene", type=int, default=9, choices=range(1, 25))
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Export all 24 scenes plus one combined 48-page PDF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/topdown_trajectory_comparison_gallery/selected_models_preview/"
            "scene09_trajectory_comparison.pdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/topdown_trajectory_comparison_gallery/"
            "selected_models_all_scenes"
        ),
        help="Destination directory used with --all-scenes.",
    )
    return parser.parse_args()


def _configure_comparison(comparison: str) -> None:
    global SELECTED_MODELS, MODEL_LABELS, MODEL_COLORS
    global MODEL_LINE_STYLES, MODEL_LINE_STYLE_NAMES
    if comparison == "baselines":
        SELECTED_MODELS = BASELINE_MODELS
        MODEL_LABELS = BASELINE_LABELS
        MODEL_COLORS = BASELINE_COLORS
        MODEL_LINE_STYLES = BASELINE_LINE_STYLES
        MODEL_LINE_STYLE_NAMES = BASELINE_LINE_STYLE_NAMES
    elif comparison == "baselines-dynamic":
        SELECTED_MODELS = DYNAMIC_BASELINE_MODELS
        MODEL_LABELS = DYNAMIC_BASELINE_LABELS
        MODEL_COLORS = DYNAMIC_BASELINE_COLORS
        MODEL_LINE_STYLES = DYNAMIC_BASELINE_LINE_STYLES
        MODEL_LINE_STYLE_NAMES = DYNAMIC_BASELINE_LINE_STYLE_NAMES


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _task_rows(rows: list[dict], scene: str, point: str) -> list[dict]:
    return [
        _with_local_run_dir(row)
        for row in rows
        if str(row.get("scene")) == scene and str(row.get("point")) == point
    ]


def _with_local_run_dir(row: dict) -> dict:
    """Resolve server-authored manifest paths against this repository."""
    result = dict(row)
    run_dir = Path(str(result.get("run_dir", "")))
    if run_dir.exists():
        return result
    marker = "/MyCodes/IndustryNav/"
    raw = str(run_dir)
    if marker in raw:
        local = REPO_ROOT / raw.split(marker, 1)[1]
        if local.exists():
            result["run_dir"] = str(local)
    return result


def _raw_fallback_run_dir(scene: str, point: str, model: str) -> Path | None:
    """Return a readable raw trajectory omitted by the result-only gallery."""
    candidates: list[Path] = []
    if model == "gpt-5.6-terra":
        candidates.append(
            REPO_ROOT / "outputs" / scene / point
            / "cli_agent_gpt-5.6-terra" / "seed0"
        )
    elif model == "qwen/qwen3.8-flash":
        candidates.append(
            REPO_ROOT / "outputs" / "_api_kiro_v1" / scene / point
            / "qwen3.8-flash" / "seed0"
        )
    elif model == "gpt-5-mini":
        candidates.append(
            REPO_ROOT / "outputs" / "_api_kiro_v1" / scene / point
            / "gpt-5-mini" / "seed0"
        )
    elif model == "pointgoal-dagger-static":
        candidates.append(
            REPO_ROOT / "outputs" / "pointgoal_static_all24_20260915"
            / "dagger" / scene / point
        )
    elif model == "pointgoal-ppo-static":
        candidates.append(
            REPO_ROOT / "outputs" / "pointgoal_static_all24_20260915"
            / "ppo" / scene / point
        )
    for candidate in candidates:
        if len(list(candidate.glob("*_actions.csv"))) == 1:
            return candidate
    return None


def _supplement_raw_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Add action-log-only runs when the metrics gallery excluded a task."""
    supplemented = list(rows)
    additions: list[dict] = []
    present = {
        (str(row.get("scene")), str(row.get("point")), str(row.get("model")))
        for row in rows
    }
    for scene_number in range(1, 25):
        scene = f"scene{scene_number}"
        for point_number in range(1, 5):
            point = f"point{point_number}"
            for model in SELECTED_MODELS:
                if (scene, point, model) in present:
                    continue
                run_dir = _raw_fallback_run_dir(scene, point, model)
                if run_dir is None:
                    continue
                addition = {
                    "scene": scene,
                    "point": point,
                    "model": model,
                    "run_dir": str(run_dir),
                    "trajectory_source": "raw action log fallback",
                }
                supplemented.append(addition)
                additions.append(addition)
                present.add((scene, point, model))
    return supplemented, additions


def _selected_rows(task_rows: list[dict], allow_missing: bool = False) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in task_rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    missing = [model for model in SELECTED_MODELS if model not in by_model]
    duplicates = [model for model in SELECTED_MODELS if len(by_model.get(model, [])) > 1]
    if duplicates or (missing and not allow_missing):
        raise ValueError(
            f"Selected-model coverage is invalid; missing={missing}, duplicates={duplicates}"
        )
    return [by_model[model][0] for model in SELECTED_MODELS if model in by_model]


def _line_with_halo(axis, xs, ys, model: str) -> None:
    line, = axis.plot(
        xs,
        ys,
        color=MODEL_COLORS[model],
        linestyle=MODEL_LINE_STYLES[model],
        linewidth=2.35,
        alpha=0.88,
        solid_capstyle="round",
        dash_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    line.set_path_effects(
        [path_effects.Stroke(linewidth=4.0, foreground="#07101f", alpha=0.20),
         path_effects.Normal()]
    )


def _render_page(
    pdf: PdfPages,
    scene: str,
    point: str,
    task_rows: list[dict],
    selected_rows: list[dict],
) -> dict:
    trajectories = [load_trajectory(row) for row in selected_rows]
    background = clean_reference_map(task_rows)
    warning_count = sum(len(item["warning_points"]) for item in trajectories)
    collision_count = sum(len(item["collision_points"]) for item in trajectories)

    figure = plt.figure(figsize=(15.4, 7.15), facecolor=PAGE_COLOR)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(3.75, 1.25),
        height_ratios=(0.105, 0.895),
        left=0.018,
        right=0.988,
        bottom=0.040,
        top=0.978,
        wspace=0.018,
        hspace=0.012,
    )
    title_axis = figure.add_subplot(grid[0, :])
    map_axis = figure.add_subplot(grid[1, 0])
    legend_axis = figure.add_subplot(grid[1, 1])

    for axis in (title_axis, map_axis, legend_axis):
        axis.set_facecolor(PAGE_COLOR)
    title_axis.axis("off")
    title_axis.text(
        0.008,
        0.60,
        f"{scene.upper()}  /  {point.upper()}",
        color=PRIMARY_TEXT,
        fontsize=11.8,
        fontweight="bold",
        ha="left",
        va="center",
    )
    map_axis.imshow(background, extent=(0, MAP_WIDTH, MAP_HEIGHT, 0), interpolation="lanczos")
    map_axis.set_xlim(0, MAP_WIDTH)
    map_axis.set_ylim(MAP_HEIGHT, 0)
    map_axis.set_aspect("equal")
    map_axis.axis("off")

    for trajectory in trajectories:
        points = trajectory["points"]
        if len(points) >= 2:
            _line_with_halo(
                map_axis,
                [point[0] for point in points],
                [point[1] for point in points],
                trajectory["model"],
            )
        if trajectory["warning_points"]:
            warning_x = [p[0] for p in trajectory["warning_points"]]
            warning_y = [p[1] for p in trajectory["warning_points"]]
            # Dark outer stroke keeps the yellow triangle visible on both the
            # pale floor and bright overlapping trajectories.
            map_axis.scatter(
                warning_x,
                warning_y,
                marker="^",
                s=80,
                facecolors="none",
                edgecolors="#111827",
                linewidths=3.0,
                alpha=0.88,
                zorder=4,
            )
            map_axis.scatter(
                warning_x,
                warning_y,
                marker="^",
                s=52,
                facecolors=WARNING_COLOR,
                edgecolors=MODEL_COLORS[trajectory["model"]],
                linewidths=1.45,
                alpha=0.98,
                zorder=5,
            )
        if trajectory["collision_points"]:
            collision_x = [p[0] for p in trajectory["collision_points"]]
            collision_y = [p[1] for p in trajectory["collision_points"]]
            # A broad dark underlay turns the red X into a high-contrast marker
            # without rasterizing it or hiding its shape in a dense path bundle.
            map_axis.scatter(
                collision_x,
                collision_y,
                marker="x",
                s=90,
                color="#111827",
                linewidths=5.0,
                alpha=0.88,
                zorder=6,
            )
            map_axis.scatter(
                collision_x,
                collision_y,
                marker="x",
                s=65,
                color=COLLISION_COLOR,
                linewidths=2.6,
                alpha=1.0,
                zorder=7,
            )

    start = next((item["points"][0] for item in trajectories if item["points"]), None)
    target = next((item["target"] for item in trajectories if item["target"]), None)
    if start:
        map_axis.scatter(
            *start, s=180, color=START_COLOR,
            edgecolor=START_OUTLINE_COLOR, linewidth=1.6, zorder=8,
        )
    if target:
        map_axis.scatter(*target, s=215, color=TARGET_COLOR, edgecolor="#0c743a", linewidth=1.6, zorder=8)

    legend_axis.set_facecolor(PANEL_COLOR)
    legend_axis.set_xlim(0, 1)
    legend_axis.set_ylim(0, 1)
    legend_axis.axis("off")
    legend_axis.text(
        0.075, 0.972, "TRAJECTORIES", color=MUTED_TEXT,
        fontsize=14.2, fontweight="bold", va="top",
    )
    start_y, step = 0.875, 0.0605
    for index, model in enumerate(SELECTED_MODELS):
        y = start_y - index * step
        legend_axis.plot(
            (0.075, 0.225), (y, y), color=MODEL_COLORS[model],
            linestyle=MODEL_LINE_STYLES[model], linewidth=4.3,
            solid_capstyle="round", dash_capstyle="round", clip_on=False,
        )
        legend_axis.text(
            0.265, y, MODEL_LABELS[model], color=PRIMARY_TEXT,
            fontsize=14.2, va="center", ha="left",
        )

    safety_y = 0.212
    legend_axis.text(
        0.075, safety_y, "MAP MARKERS", color=MUTED_TEXT,
        fontsize=14.2, fontweight="bold", va="top",
    )
    marker_y = safety_y - 0.097
    legend_axis.scatter(
        0.095, marker_y, marker="^", s=132, facecolors="none",
        edgecolors="#111827", linewidths=3.2,
    )
    legend_axis.scatter(
        0.095, marker_y, marker="^", s=92, color=WARNING_COLOR,
        edgecolor="#854d0e", linewidth=1.3,
    )
    legend_axis.text(0.145, marker_y, "Warning", color=PRIMARY_TEXT, fontsize=14.2, va="center")
    legend_axis.scatter(0.525, marker_y, marker="x", s=132, color="#111827", linewidth=5.0)
    legend_axis.scatter(0.525, marker_y, marker="x", s=96, color=COLLISION_COLOR, linewidth=2.8)
    legend_axis.text(0.575, marker_y, "Collision", color=PRIMARY_TEXT, fontsize=14.2, va="center")
    marker_y -= 0.069
    legend_axis.scatter(
        0.095, marker_y, s=76, color=START_COLOR,
        edgecolor=START_OUTLINE_COLOR, linewidth=1.2,
    )
    legend_axis.text(0.145, marker_y, "Start", color=PRIMARY_TEXT, fontsize=14.2, va="center")
    legend_axis.scatter(0.525, marker_y, s=82, color=TARGET_COLOR, edgecolor="#0c743a", linewidth=1.0)
    legend_axis.text(0.575, marker_y, "Target", color=PRIMARY_TEXT, fontsize=14.2, va="center")

    pdf.savefig(figure, facecolor=figure.get_facecolor())
    plt.close(figure)
    return {
        "scene": scene,
        "point": point,
        "models": list(SELECTED_MODELS),
        "warning_events": warning_count,
        "collision_events": collision_count,
    }


def _paired_task(
    scene: str,
    point: str,
    task_rows: list[dict],
    selected_rows: list[dict],
) -> dict:
    trajectories = [load_trajectory(row) for row in selected_rows]
    available_models = [trajectory["model"] for trajectory in trajectories]
    background_rows = (
        selected_rows
        if any(row.get("model") == "astar-static" for row in selected_rows)
        else task_rows
    )
    return {
        "scene": scene,
        "point": point,
        "models": available_models,
        "missing_models": [
            model for model in SELECTED_MODELS if model not in available_models
        ],
        "background": clean_reference_map(background_rows),
        "trajectories": trajectories,
        "warning_events": sum(len(item["warning_points"]) for item in trajectories),
        "collision_events": sum(len(item["collision_points"]) for item in trajectories),
    }


def _draw_paired_map(map_axis, task: dict) -> None:
    trajectories = task["trajectories"]
    map_axis.set_facecolor(PAGE_COLOR)
    map_axis.imshow(
        task["background"],
        extent=(0, MAP_WIDTH, MAP_HEIGHT, 0),
        interpolation="lanczos",
    )
    map_axis.set_xlim(0, MAP_WIDTH)
    map_axis.set_ylim(MAP_HEIGHT, 0)
    map_axis.set_aspect("equal")
    map_axis.axis("off")
    for trajectory in trajectories:
        points = trajectory["points"]
        if len(points) >= 2:
            _line_with_halo(
                map_axis,
                [point[0] for point in points],
                [point[1] for point in points],
                trajectory["model"],
            )
        if trajectory["warning_points"]:
            warning_x = [point[0] for point in trajectory["warning_points"]]
            warning_y = [point[1] for point in trajectory["warning_points"]]
            map_axis.scatter(
                warning_x, warning_y, marker="^", s=80,
                facecolors="none", edgecolors="#111827", linewidths=3.0,
                alpha=0.88, zorder=4,
            )
            map_axis.scatter(
                warning_x, warning_y, marker="^", s=52,
                facecolors=WARNING_COLOR,
                edgecolors=MODEL_COLORS[trajectory["model"]],
                linewidths=1.45, alpha=0.98, zorder=5,
            )
        if trajectory["collision_points"]:
            collision_x = [point[0] for point in trajectory["collision_points"]]
            collision_y = [point[1] for point in trajectory["collision_points"]]
            map_axis.scatter(
                collision_x, collision_y, marker="x", s=90,
                color="#111827", linewidths=5.0, alpha=0.88, zorder=6,
            )
            map_axis.scatter(
                collision_x, collision_y, marker="x", s=65,
                color=COLLISION_COLOR, linewidths=2.6, alpha=1.0, zorder=7,
            )
    start = next(
        (item["points"][0] for item in trajectories if item["points"]), None
    )
    target = next(
        (item["target"] for item in trajectories if item["target"]), None
    )
    if start:
        map_axis.scatter(
            *start, s=180, color=START_COLOR,
            edgecolor=START_OUTLINE_COLOR, linewidth=1.6, zorder=8,
        )
    if target:
        map_axis.scatter(
            *target, s=215, color=TARGET_COLOR, edgecolor="#0c743a",
            linewidth=1.6, zorder=8,
        )


def _draw_shared_legend(legend_axis) -> None:
    legend_axis.set_facecolor(PANEL_COLOR)
    legend_axis.set_xlim(0, 1)
    legend_axis.set_ylim(0, 1)
    legend_axis.axis("off")
    legend_axis.text(
        0.075, 0.972, "TRAJECTORIES", color=MUTED_TEXT,
        fontsize=14.2, fontweight="bold", va="top",
    )
    if len(SELECTED_MODELS) <= 3:
        start_y, step, safety_y = 0.835, 0.145, 0.355
    else:
        start_y, step, safety_y = 0.875, 0.0605, 0.212
    for index, model in enumerate(SELECTED_MODELS):
        y = start_y - index * step
        legend_axis.plot(
            (0.075, 0.225), (y, y), color=MODEL_COLORS[model],
            linestyle=MODEL_LINE_STYLES[model], linewidth=4.3,
            solid_capstyle="round", dash_capstyle="round", clip_on=False,
        )
        legend_axis.text(
            0.265, y, MODEL_LABELS[model], color=PRIMARY_TEXT,
            fontsize=14.2, va="center", ha="left",
        )
    legend_axis.text(
        0.075, safety_y, "MAP MARKERS", color=MUTED_TEXT,
        fontsize=14.2, fontweight="bold", va="top",
    )
    marker_y = safety_y - 0.097
    legend_axis.scatter(
        0.095, marker_y, marker="^", s=132, facecolors="none",
        edgecolors="#111827", linewidths=3.2,
    )
    legend_axis.scatter(
        0.095, marker_y, marker="^", s=92, color=WARNING_COLOR,
        edgecolor="#854d0e", linewidth=1.3,
    )
    legend_axis.text(
        0.145, marker_y, "Warning", color=PRIMARY_TEXT,
        fontsize=14.2, va="center",
    )
    legend_axis.scatter(
        0.525, marker_y, marker="x", s=132,
        color="#111827", linewidth=5.0,
    )
    legend_axis.scatter(
        0.525, marker_y, marker="x", s=96,
        color=COLLISION_COLOR, linewidth=2.8,
    )
    legend_axis.text(
        0.575, marker_y, "Collision", color=PRIMARY_TEXT,
        fontsize=14.2, va="center",
    )
    marker_y -= 0.069
    legend_axis.scatter(
        0.095, marker_y, s=76, color=START_COLOR,
        edgecolor=START_OUTLINE_COLOR, linewidth=1.2,
    )
    legend_axis.text(
        0.145, marker_y, "Start", color=PRIMARY_TEXT,
        fontsize=14.2, va="center",
    )
    legend_axis.scatter(
        0.525, marker_y, s=82, color=TARGET_COLOR,
        edgecolor="#0c743a", linewidth=1.0,
    )
    legend_axis.text(
        0.575, marker_y, "Target", color=PRIMARY_TEXT,
        fontsize=14.2, va="center",
    )


def _render_pair_page(
    pdf_targets: list[PdfPages], scene: str, tasks: list[dict]
) -> dict:
    if len(tasks) != 2:
        raise ValueError("A paired comparison page requires exactly two tasks")
    figure = plt.figure(figsize=(19.2, 4.95), facecolor=PAGE_COLOR)
    grid = figure.add_gridspec(
        2, 3,
        width_ratios=(1.0, 1.0, 0.46),
        height_ratios=(0.075, 0.925),
        left=0.012, right=0.992, bottom=0.012, top=0.995,
        wspace=0.018, hspace=0.010,
    )
    title_axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    map_axes = [figure.add_subplot(grid[1, index]) for index in range(2)]
    legend_axis = figure.add_subplot(grid[:, 2])
    for title_axis, map_axis, task in zip(title_axes, map_axes, tasks):
        title_axis.set_facecolor(PAGE_COLOR)
        title_axis.axis("off")
        title_axis.text(
            0.008, 0.30,
            f"{scene.upper()}  /  {task['point'].upper()}",
            color=PRIMARY_TEXT, fontsize=15.0, fontweight="bold",
            ha="left", va="center",
        )
        if task["missing_models"]:
            title_axis.text(
                0.995, 0.30,
                f"{len(task['models'])}/{len(SELECTED_MODELS)} trajectories",
                color="#b91c1c", fontsize=9.5, fontweight="bold",
                ha="right", va="center",
            )
        _draw_paired_map(map_axis, task)
    _draw_shared_legend(legend_axis)
    for pdf in pdf_targets:
        pdf.savefig(figure, facecolor=figure.get_facecolor())
    plt.close(figure)
    return {
        "scene": scene,
        "points": [
            {
                key: task[key]
                for key in (
                    "point", "models", "missing_models",
                    "warning_events", "collision_events"
                )
            }
            for task in tasks
        ],
    }


def _prepare_scene(rows: list[dict], scene: str) -> list[list[dict]]:
    paired_pages: list[list[dict]] = []
    for first_point in (1, 3):
        paired_tasks = []
        for point_number in (first_point, first_point + 1):
            point = f"point{point_number}"
            all_task_rows = _task_rows(rows, scene, point)
            if not all_task_rows:
                raise ValueError(f"No gallery records for {scene}/{point}")
            selected = _selected_rows(all_task_rows, allow_missing=True)
            paired_tasks.append(
                _paired_task(scene, point, all_task_rows, selected)
            )
            missing = [
                model for model in SELECTED_MODELS
                if model not in {str(row.get("model")) for row in selected}
            ]
            suffix = f"; missing={','.join(missing)}" if missing else ""
            print(
                f"Prepared {scene}/{point} with {len(selected)} models{suffix}",
                flush=True,
            )
        paired_pages.append(paired_tasks)
    return paired_pages


def _pdf_metadata(title: str) -> dict[str, str]:
    return {
        "Title": title,
        "Author": "IndustryNav",
        "Subject": (
            "Paired canonical navigation points with vector trajectories "
            "and labels"
        ),
    }


def _scene_manifest(
    source: Path,
    scene: str,
    pdf_name: str,
    pages: list[dict],
) -> dict:
    return {
        "source_manifest": str(source),
        "scene": scene,
        "page_count": len(pages),
        "layout": "two trajectory maps side by side with a shared right legend",
        "pdf": pdf_name,
        "models": list(SELECTED_MODELS),
        "line_styles": {
            model: MODEL_LINE_STYLE_NAMES[model] for model in SELECTED_MODELS
        },
        "pages": pages,
        "vector_content": ["trajectories", "event markers", "legend", "text"],
        "raster_content": ["Unity minimap background"],
    }


def main() -> None:
    args = parse_args()
    _configure_comparison(args.comparison)
    source = _absolute(args.source_manifest)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("The source gallery manifest must be a JSON list")
    rows, fallback_rows = _supplement_raw_rows(rows)
    if fallback_rows:
        print(
            f"Added {len(fallback_rows)} raw trajectory fallback records",
            flush=True,
        )

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "savefig.transparent": False,
    })
    if not args.all_scenes:
        scene = f"scene{args.scene}"
        output = _absolute(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared_pages = _prepare_scene(rows, scene)
        pages = []
        with PdfPages(
            output,
            metadata=_pdf_metadata(
                f"{scene.upper()} Selected-Agent Trajectory Comparison"
            ),
        ) as pdf:
            for paired_tasks in prepared_pages:
                pages.append(_render_pair_page([pdf], scene, paired_tasks))
        manifest = _scene_manifest(source, scene, output.name, pages)
        manifest["comparison"] = args.comparison
        manifest["raw_fallback_records"] = fallback_rows
        output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(pages)} pages to {output}")
        return

    output_dir = _absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_output = output_dir / "all_scenes_trajectory_comparison.pdf"
    scene_manifests = []
    with PdfPages(
        combined_output,
        metadata=_pdf_metadata(
            "All Scenes Selected-Agent Trajectory Comparison"
        ),
    ) as combined_pdf:
        for scene_number in range(1, 25):
            scene = f"scene{scene_number}"
            scene_output = output_dir / f"scene{scene_number:02d}_trajectory_comparison.pdf"
            prepared_pages = _prepare_scene(rows, scene)
            pages = []
            with PdfPages(
                scene_output,
                metadata=_pdf_metadata(
                    f"{scene.upper()} Selected-Agent Trajectory Comparison"
                ),
            ) as scene_pdf:
                for paired_tasks in prepared_pages:
                    pages.append(
                        _render_pair_page(
                            [scene_pdf, combined_pdf], scene, paired_tasks
                        )
                    )
            scene_manifest = _scene_manifest(
                source, scene, scene_output.name, pages
            )
            scene_manifest["comparison"] = args.comparison
            scene_output.with_suffix(".json").write_text(
                json.dumps(scene_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            scene_manifests.append(scene_manifest)
            print(f"Wrote {scene_output}", flush=True)

    combined_manifest = {
        "source_manifest": str(source),
        "comparison": args.comparison,
        "scene_count": len(scene_manifests),
        "page_count": sum(item["page_count"] for item in scene_manifests),
        "combined_pdf": combined_output.name,
        "models": list(SELECTED_MODELS),
        "raw_fallback_records": fallback_rows,
        "scenes": scene_manifests,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(combined_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {combined_manifest['page_count']} pages across "
        f"{combined_manifest['scene_count']} scenes to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
