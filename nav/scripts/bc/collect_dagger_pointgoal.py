"""Collect A*-labeled states visited by a PointGoal behavior policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from nav.scripts.astar.collect_astar_pointgoal import atomic_json, latest_result, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split", default="train")
    parser.add_argument("--round-id", default="round1")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=30500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=0,
        help="Optional balanced cap applied independently to every selected scene.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def label_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def command_for(
    args: argparse.Namespace,
    task: dict,
    task_index: int,
    episode_dir: Path,
) -> list[str]:
    command = [
        str(args.python), "-m", "nav.scripts.agent.run_benchmark_cell",
        "--baseline", "dagger",
        "--file_name", str(args.unity),
        "--scene_id", str(task["scene_id"]),
        "--scene_name", str(task["scene_name"]),
        "--point_id", f"dagger_{args.round_id}_{task['episode_id']}",
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
        "--model_id", f"pointgoal-dagger-{args.round_id}",
        "--bc_ckpt", str(args.checkpoint), "--bc_device", "cuda",
        "--dagger_beta", str(args.beta),
        "--dagger_seed", str(args.seed + task_index),
        "--astar_obstacle_clearance_m", "0.6",
        "--astar_dynamic_replan_lookahead_m", "8.0",
        "--astar_dynamic_replan_confirm_steps", "2",
        "--init_world_x", str(task["init_world_x"]),
        "--init_world_z", str(task["init_world_z"]),
        "--init_curr_direction", str(task["init_direction"]),
        "--target_x", str(task["target_x"]), "--target_y", str(task["target_y"]),
        "--motion_random_seed", str(task["motion_random_seed"]),
        "--human_speed_min_mps", "0.8", "--human_speed_max_mps", "1.6",
        "--vehicle_speed_min_mps", "1.5", "--vehicle_speed_max_mps", "3.5",
        "--robot_speed_min_mps", "1.0", "--robot_speed_max_mps", "2.0",
    ]
    if platform.system() == "Linux" and shutil.which("xvfb-run"):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1724x1024x24", *command]
    return command


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.beta <= 1.0:
        raise SystemExit("--beta must be in [0, 1]")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.unity.is_file():
        raise SystemExit(f"Unity executable not found: {args.unity}")

    tasks = [task for task in read_jsonl(args.manifest) if task["split"] == args.split]
    if args.episodes_per_scene > 0:
        scene_counts: dict[str, int] = {}
        balanced = []
        for task in tasks:
            scene = str(task["scene_name"])
            if scene_counts.get(scene, 0) >= args.episodes_per_scene:
                continue
            balanced.append(task)
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        tasks = balanced
    if args.max_episodes > 0:
        tasks = tasks[: args.max_episodes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = []
    for index, task in enumerate(tasks):
        episode_dir = args.output_root / task["scene_name"] / task["episode_id"]
        completed = latest_result(episode_dir / "results.csv") is not None
        labels = label_count(episode_dir / "dagger_labels.csv")
        completed = completed and labels > 0
        if args.resume and completed:
            skipped.append((task, labels))
            continue
        pending.append((index, task, episode_dir))

    state_path = args.output_root / "collection_state.json"
    state_lock = threading.Lock()
    state = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "round_id": args.round_id,
        "beta": args.beta,
        "dagger_seed": args.seed,
        "episodes_per_scene": args.episodes_per_scene,
        "selected_tasks": len(tasks),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": {
            f"{task['scene_name']}/{task['episode_id']}": {
                "status": "skipped_complete",
                "labels": labels,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            for task, labels in skipped
        },
    }
    atomic_json(state_path, state)

    def save_status(task: dict, status: str, **extra) -> None:
        with state_lock:
            state["tasks"][f"{task['scene_name']}/{task['episode_id']}"] = {
                "status": status,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                **extra,
            }
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_json(state_path, state)

    print(
        f"round={args.round_id} beta={args.beta:g} tasks={len(tasks)} "
        f"pending={len(pending)} workers={args.workers}"
    )
    if args.dry_run:
        for item in pending:
            print(" ".join(command_for(args, item[1], item[0], item[2])))
        return

    def run_one(item) -> tuple[bool, int]:
        index, task, episode_dir = item
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "collection_task.json").write_text(
            json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = command_for(args, task, index, episode_dir)
        save_status(task, "running", command=command)
        with (episode_dir / "runner.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
        labels = label_count(episode_dir / "dagger_labels.csv")
        usable = process.returncode == 0 and labels > 0
        result = latest_result(episode_dir / "results.csv") or {}
        save_status(
            task, "usable" if usable else "failed",
            returncode=process.returncode, labels=labels,
            stop_reason=result.get("stop_reason"),
            distance_world=result.get("distance_world"),
        )
        print(
            f"[{task['scene_name']}/{task['episode_id']}] "
            f"rc={process.returncode} labels={labels} usable={usable}",
            flush=True,
        )
        return usable, labels

    usable = labels = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, item) for item in pending]
        for future in as_completed(futures):
            ok, count = future.result()
            usable += int(ok)
            labels += count
    print(
        f"collection finished: usable={usable} labels={labels} "
        f"skipped={len(skipped)}"
    )


if __name__ == "__main__":
    main()
