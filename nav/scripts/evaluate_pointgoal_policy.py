"""Run a small closed-loop BC Point-Goal evaluation on held-out scenes."""

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=19507)
    return parser.parse_args()


def latest_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    records = [
        json.loads(line) for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    selected = []
    seen_scenes = set()
    for record in records:
        if record["split"] != "test" or record["scene_name"] in seen_scenes:
            continue
        selected.append(record)
        seen_scenes.add(record["scene_name"])
        if len(selected) >= args.scenes:
            break
    if len(selected) < args.scenes:
        raise SystemExit(f"Only found {len(selected)} held-out scenes")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, task in enumerate(selected):
        episode_dir = args.output_root / task["scene_name"] / task["episode_id"]
        command = [
            str(args.python), "-m", "nav.scripts.run_benchmark_cell",
            "--baseline", "bc",
            "--file_name", str(args.unity),
            "--scene_id", str(task["scene_id"]),
            "--scene_name", str(task["scene_name"]),
            "--point_id", str(task["episode_id"]),
            "--seed_id", str(task["seed"]),
            "--worker_id", "0", "--base_port", str(args.base_port + index),
            "--max_steps", "320", "--dynamic_step_budget",
            "--step_budget_min", "80", "--step_budget_max", "320",
            "--steps_per_path_meter", "2.2", "--step_budget_overhead", "60",
            "--reach_m", "2.0", "--sim_steps_per_decision", "1",
            "--ego_width", "320", "--ego_height", "240",
            "--minimap_width", "862", "--modalities", "ego,minimap,depth",
            "--dynamic_objects", str(task["dynamic_objects"]),
            "--marker_source", "vector", "--frame_save_dir", str(episode_dir),
            "--model_id", "pointgoal-bc-polar",
            "--bc_ckpt", str(args.checkpoint), "--bc_device", "cuda",
            "--bc_pointgoal_actions",
            "--init_world_x", str(task["init_world_x"]),
            "--init_world_z", str(task["init_world_z"]),
            "--init_curr_direction", str(task["init_direction"]),
            "--target_x", str(task["target_x"]), "--target_y", str(task["target_y"]),
            "--motion_random_seed", str(task["motion_random_seed"]),
            "--light_intensity_min", "0.8", "--light_intensity_max", "1.2",
            "--light_random_seed", str(task["light_random_seed"]),
            "--light_fixed_exposure", "9.0",
        ]
        if platform.system() == "Linux" and shutil.which("xvfb-run"):
            command = ["xvfb-run", "-a", "-s", "-screen 0 1724x1024x24", *command]
        episode_dir.mkdir(parents=True, exist_ok=True)
        with (episode_dir / "runner.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
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
