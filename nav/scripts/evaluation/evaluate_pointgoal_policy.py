"""Run a small closed-loop learned PointGoal evaluation on held-out scenes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--eligible-dataset-manifest",
        type=Path,
        default=None,
        help=(
            "Optional exported dataset_manifest.jsonl. When provided, evaluate "
            "only tasks whose A* demonstrations passed the expert success gate."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", choices=("bc", "ppo"), default="bc")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=19507)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_held_out_tasks(
    records: list[dict],
    scene_count: int,
    eligible_records: list[dict] | None = None,
) -> list[dict]:
    eligible = None
    if eligible_records is not None:
        eligible = {
            (str(record["scene_name"]), str(record["episode_id"]))
            for record in eligible_records
        }

    selected = []
    seen_scenes = set()
    for record in records:
        key = (str(record["scene_name"]), str(record["episode_id"]))
        if record["split"] != "test" or record["scene_name"] in seen_scenes:
            continue
        if eligible is not None and key not in eligible:
            continue
        selected.append(record)
        seen_scenes.add(record["scene_name"])
        if len(selected) >= scene_count:
            break
    return selected


def latest_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def command_for(
    args: argparse.Namespace,
    task: dict,
    task_index: int,
    episode_dir: Path,
) -> list[str]:
    """Build one evaluation command using scene-authored lighting/exposure."""
    baseline = getattr(args, "baseline", "bc")
    command = [
        str(args.python), "-m", "nav.scripts.agent.run_benchmark_cell",
        "--baseline", baseline,
        "--file_name", str(args.unity),
        "--scene_id", str(task["scene_id"]),
        "--scene_name", str(task["scene_name"]),
        "--point_id", str(task["episode_id"]),
        "--seed_id", str(task["seed"]),
        "--worker_id", "0", "--base_port", str(args.base_port + task_index),
        "--max_steps", "320", "--dynamic_step_budget",
        "--step_budget_min", "80", "--step_budget_max", "320",
        "--steps_per_path_meter", "2.2", "--step_budget_overhead", "60",
        "--reach_m", "2.0", "--sim_steps_per_decision", "1",
        "--ego_width", "320", "--ego_height", "240",
        "--minimap_width", "862", "--modalities", "ego,minimap,depth",
        "--dynamic_objects", str(task["dynamic_objects"]),
        "--marker_source", "vector", "--frame_save_dir", str(episode_dir),
        "--model_id", f"pointgoal-{baseline}",
        "--init_world_x", str(task["init_world_x"]),
        "--init_world_z", str(task["init_world_z"]),
        "--init_curr_direction", str(task["init_direction"]),
        "--target_x", str(task["target_x"]), "--target_y", str(task["target_y"]),
        "--motion_random_seed", str(task["motion_random_seed"]),
        # Match A* teacher collection on every platform. Runtime HDRP exposure
        # overrides saturate the minimap and darken the egocentric camera.
        # Photometric diversity is applied by the training data pipeline.
    ]
    if baseline == "bc":
        command.extend([
            "--bc_ckpt", str(args.checkpoint), "--bc_device", "cuda",
            "--bc_pointgoal_actions",
        ])
    else:
        command.extend([
            "--ppo_ckpt", str(args.checkpoint), "--ppo_device", "cuda",
        ])
    if platform.system() == "Linux" and shutil.which("xvfb-run"):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1724x1024x24", *command]
    return command


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    records = read_jsonl(args.manifest)
    eligible_records = (
        read_jsonl(args.eligible_dataset_manifest)
        if args.eligible_dataset_manifest is not None
        else None
    )
    selected = select_held_out_tasks(records, args.scenes, eligible_records)
    if len(selected) < args.scenes:
        raise SystemExit(f"Only found {len(selected)} held-out scenes")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, task in enumerate(selected):
        episode_dir = args.output_root / task["scene_name"] / task["episode_id"]
        command = command_for(args, task, index, episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        with (episode_dir / "runner.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
        result = latest_result(episode_dir / "results.csv") or {}
        success = (
            result.get("stop_reason") == "reached_vicinity"
            and float(result.get("distance_world", "inf")) <= 2.0
        )
        summaries.append({
            "scene_name": task["scene_name"], "episode_id": task["episode_id"],
            "returncode": process.returncode, "success": success,
            "stop_reason": result.get("stop_reason"),
            "distance_world": result.get("distance_world"),
            "steps_taken": result.get("steps_taken"),
        })
        print(json.dumps(summaries[-1]), flush=True)
    report = {
        "checkpoint": str(args.checkpoint), "episodes": summaries,
        "successes": sum(item["success"] for item in summaries),
        "success_rate": sum(item["success"] for item in summaries) / len(summaries),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
