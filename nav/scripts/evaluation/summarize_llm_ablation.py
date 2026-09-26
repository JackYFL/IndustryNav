"""Summarize the fixed 32-task LLM memory/top-down ablation.

The report keeps the historical reference separate from newly run conditions.
The top-down-only extension is opt-in. It is safe to run while the grid is in
progress: missing and incomplete cells remain visible instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_DIR = (
    REPO_ROOT / "analysis" / "llm_memory_topdown_ablation_32_20260914"
)
HISTORICAL_PROMPT_SHA256 = (
    "0a138c1a8294d766b3b6d35158d4d89f74945f7d94a99c5307a76d97104fda3f"
)
MODELS = (
    ("google/gemini-3.8-flash", "gemini-3.8-flash", "Gemini 3.8 Flash"),
    ("z-ai/glm-5.3-flash", "glm-5.3-flash", "GLM 5.3 Flash"),
)


@dataclass(frozen=True)
class Condition:
    key: str
    memory: bool
    topdown: bool
    historical: bool = False
    egocentric: bool = True

    @property
    def label(self) -> str:
        if self.historical:
            return "Original reference (memory on, map off)"
        memory = "on" if self.memory else "off"
        if not self.egocentric and self.topdown:
            return f"Memory {memory} / top-down only (no egocentric RGB)"
        topdown = "on" if self.topdown else "off"
        return f"Memory {memory} / top-down {topdown}"


CONDITIONS = (
    Condition("original", True, False, historical=True),
    Condition("m0_td0", False, False),
    Condition("m0_td1", False, True),
    Condition("m1_td1", True, True),
)
TOPDOWN_ONLY_CONDITION = Condition("m0_td_only", False, True, egocentric=False)


def selected_conditions(args) -> tuple[Condition, ...]:
    if getattr(args, "include_topdown_only", False):
        return (*CONDITIONS, TOPDOWN_ONLY_CONDITION)
    return CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-file", type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "tasks_32.json",
    )
    parser.add_argument(
        "--ablation-root", type=Path,
        default=REPO_ROOT / "outputs" / "llm_memory_topdown_ablation_32_20260914",
    )
    parser.add_argument("--original-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--original-reference-csv", type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "original_reference.csv",
        help=(
            "Portable frozen metrics for the historical memory-on/map-off cell. "
            "It is created from --original-root when absent."
        ),
    )
    parser.add_argument(
        "--include-topdown-only", action="store_true",
        help="Include the additional history-0, minimap-only condition using nav_minimap_only.txt.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail unless every selected new condition contains 32 valid tasks per model.",
    )
    return parser.parse_args()


def _load_tasks(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("tasks", [])
    tasks = [(str(item["scene_name"]), str(item["point_id"])) for item in raw]
    if len(tasks) != 32 or len(set(tasks)) != 32:
        raise ValueError(f"Expected 32 unique tasks in {path}; found {len(set(tasks))}.")
    source = REPO_ROOT / str(payload.get("source", "input_points.json"))
    expected_sha = str(payload.get("source_sha256", ""))
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError(
            f"Canonical point source changed: expected {expected_sha}, found {actual_sha}."
        )
    return tasks


def _cell_dir(
    condition: Condition,
    scene: str,
    point: str,
    model_short: str,
    original_root: Path,
    ablation_root: Path,
) -> Path:
    if condition.historical:
        return original_root / scene / point / model_short / "seed0"
    root = ablation_root
    history_size = 5 if condition.memory else 0
    if history_size != 5:
        root = root / "_history_size" / f"hs{history_size}"
    model_dir = (
        model_short + ("" if condition.egocentric else "_novision")
        + ("_topdown" if condition.topdown else "")
    )
    return root / scene / point / model_dir / "seed0"


def _read_last_result(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def ensure_original_reference(
    path: Path,
    tasks: list[tuple[str, str]],
    original_root: Path,
) -> None:
    """Freeze the small historical control table for portable remote reports."""
    if path.exists():
        return
    reference_rows: list[dict[str, object]] = []
    for model_id, model_short, model_label in MODELS:
        for scene, point in tasks:
            folder = original_root / scene / point / model_short / "seed0"
            result_path = folder / "results.csv"
            config_path = folder / "run_config.json"
            result = _read_last_result(result_path)
            if result is None:
                raise ValueError(f"Cannot freeze missing original result: {result_path}")
            reference_rows.append({
                "model_id": model_id,
                "model": model_label,
                "scene_name": scene,
                "point_id": point,
                "stop_reason": result.get("stop_reason", ""),
                "distance_world": result.get("distance_world", ""),
                "steps_taken": result.get("steps_taken", ""),
                "result_sha256": _sha256(result_path),
                "run_config_sha256": _sha256(config_path),
                "source_run_dir": str(folder.relative_to(REPO_ROOT)),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)


def _load_original_reference(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(MODELS) * 32
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows in {path}; found {len(rows)}.")
    return {
        (row["model_id"], row["scene_name"], row["point_id"]): row
        for row in rows
    }


def _config_error(
    folder: Path,
    condition: Condition,
    model_id: str,
    scene: str,
    point: str,
) -> str:
    path = folder / "run_config.json"
    if not path.exists():
        return "missing run_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"invalid run_config.json: {exc}"
    settings = config.get("settings", {})
    expected_history = 5 if condition.memory else 0
    checks = {
        "model_id": model_id,
        "scene_name": scene,
        "point_id": point,
        "history_size": expected_history,
        "vision_input": condition.egocentric,
    }
    for name, expected in checks.items():
        if settings.get(name) != expected:
            return f"config {name}={settings.get(name)!r}; expected {expected!r}"
    recorded_topdown = bool(settings.get("topdown_input", False))
    if recorded_topdown != condition.topdown:
        return f"config topdown_input={recorded_topdown}; expected {condition.topdown}"
    modalities = config.get("input_modalities", [])
    expected_modalities = (
        (["ego"] if condition.egocentric else [])
        + (["topdown"] if condition.topdown else [])
    )
    if modalities != expected_modalities:
        return f"config input_modalities={modalities!r}; expected {expected_modalities!r}"
    if not condition.topdown and config.get("prompt_sha256") != HISTORICAL_PROMPT_SHA256:
        return "map-off prompt does not match the historical reference prompt"
    if condition.key == TOPDOWN_ONLY_CONDITION.key:
        prompt_path = REPO_ROOT / "nav" / "prompts" / "nav_minimap_only.txt"
        if config.get("prompt_sha256") != _sha256(prompt_path):
            return "top-down-only prompt does not match nav_minimap_only.txt"
    return ""


def collect_rows(
    args: argparse.Namespace,
    tasks: list[tuple[str, str]],
    original_reference: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id, model_short, model_label in MODELS:
        for condition in selected_conditions(args):
            for scene, point in tasks:
                folder = _cell_dir(
                    condition, scene, point, model_short,
                    args.original_root, args.ablation_root,
                )
                result = _read_last_result(folder / "results.csv")
                if result is None and condition.historical:
                    result = original_reference.get((model_id, scene, point))
                error = ""
                if result is None:
                    status = "missing"
                else:
                    if not condition.historical:
                        error = _config_error(
                            folder, condition, model_id, scene, point,
                        )
                    stop_reason = str(result.get("stop_reason", ""))
                    if error:
                        status = "invalid_config"
                    elif stop_reason not in {"max_steps", "reached_vicinity"}:
                        status = "incomplete"
                    else:
                        status = "complete"
                distance = None
                steps = None
                if result is not None:
                    try:
                        distance = float(result["distance_world"])
                        steps = int(float(result["steps_taken"]))
                    except (KeyError, TypeError, ValueError):
                        if status == "complete":
                            status = "invalid_result"
                rows.append({
                    "model_id": model_id,
                    "model": model_label,
                    "condition": condition.key,
                    "condition_label": condition.label,
                    "memory": condition.memory,
                    "egocentric": condition.egocentric,
                    "topdown": condition.topdown,
                    "historical_reference": condition.historical,
                    "scene_name": scene,
                    "point_id": point,
                    "status": status,
                    "stop_reason": result.get("stop_reason", "") if result else "",
                    "distance_world": distance,
                    "steps_taken": steps,
                    "success_at_2m": distance is not None and distance <= 2.0,
                    "success_at_5m": distance is not None and distance <= 5.0,
                    "success_at_10m": distance is not None and distance <= 10.0,
                    "config_error": error,
                    "run_dir": str(folder),
                })
    return rows


def summarize(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = list(rows)
    summaries: list[dict[str, object]] = []
    for model_id, _short, model_label in MODELS:
        original_success = None
        model_rows = [row for row in rows if row["model_id"] == model_id]
        for condition in (*CONDITIONS, TOPDOWN_ONLY_CONDITION):
            selected = [row for row in model_rows if row["condition"] == condition.key]
            if not selected:
                continue
            complete = [row for row in selected if row["status"] == "complete"]
            distances = [float(row["distance_world"]) for row in complete]
            steps = [int(row["steps_taken"]) for row in complete]
            success_2 = sum(bool(row["success_at_2m"]) for row in complete)
            if condition.historical:
                original_success = success_2 / len(complete) if complete else None
            rate_2 = success_2 / len(complete) if complete else None
            summaries.append({
                "model_id": model_id,
                "model": model_label,
                "condition": condition.key,
                "condition_label": condition.label,
                "memory": condition.memory,
                "egocentric": condition.egocentric,
                "topdown": condition.topdown,
                "historical_reference": condition.historical,
                "expected_tasks": len(selected),
                "completed_tasks": len(complete),
                "missing_or_invalid_tasks": len(selected) - len(complete),
                "success_at_2m_count": success_2,
                "success_at_2m": rate_2,
                "success_at_5m": (
                    sum(bool(row["success_at_5m"]) for row in complete) / len(complete)
                    if complete else None
                ),
                "success_at_10m": (
                    sum(bool(row["success_at_10m"]) for row in complete) / len(complete)
                    if complete else None
                ),
                "mean_final_distance_m": statistics.fmean(distances) if distances else None,
                "median_final_distance_m": statistics.median(distances) if distances else None,
                "mean_steps": statistics.fmean(steps) if steps else None,
                "delta_success_at_2m_vs_original": (
                    rate_2 - original_success
                    if rate_2 is not None and original_success is not None
                    and not condition.historical else None
                ),
            })
    return summaries


def _format_percent(value: object) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}%"


def _format_float(value: object) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Gemini / GLM Memory and Top-down Ablation Results",
        "",
        "The original reference uses the same frozen 32 tasks and is not counted as a rerun.",
        "New conditions are reported only over completed, configuration-validated tasks.",
        "All original 2 x 2 conditions include egocentric RGB. The optional top-down-only condition uses nav_minimap_only.txt, no history, and no egocentric/depth image. Its prompt differs from the RGB + map condition, so it is not a prompt-controlled image-only comparison.",
        "",
        "| Model | Condition | Complete | Success@2m | Success@5m | Success@10m | Mean final distance (m) | Mean steps | Delta S@2m vs original |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        delta = row["delta_success_at_2m_vs_original"]
        lines.append(
            f"| {row['model']} | {row['condition_label']} | "
            f"{row['completed_tasks']}/{row['expected_tasks']} | "
            f"{_format_percent(row['success_at_2m'])} | "
            f"{_format_percent(row['success_at_5m'])} | "
            f"{_format_percent(row['success_at_10m'])} | "
            f"{_format_float(row['mean_final_distance_m'])} | "
            f"{_format_float(row['mean_steps'])} | "
            f"{('--' if delta is None else f'{100.0 * float(delta):+.2f} pp')} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tasks = _load_tasks(args.tasks_file)
    ensure_original_reference(
        args.original_reference_csv, tasks, args.original_root,
    )
    original_reference = _load_original_reference(args.original_reference_csv)
    rows = collect_rows(args, tasks, original_reference)
    summaries = summarize(rows)
    write_outputs(args.output_dir, rows, summaries)
    new_rows = [row for row in summaries if not row["historical_reference"]]
    completed = sum(int(row["completed_tasks"]) for row in new_rows)
    expected = sum(int(row["expected_tasks"]) for row in new_rows)
    print(f"Ablation completion: {completed}/{expected} validated episodes")
    print(f"Report: {args.output_dir / 'results.md'}")
    if args.strict and completed != expected:
        raise SystemExit(f"Strict completion check failed: {completed}/{expected}")


if __name__ == "__main__":
    main()
