"""Collect policy-compatible point-goal demonstrations from a manifest.

Each episode is independently restartable.  ``--resume`` skips every task that
already has a terminal ``results.csv`` and reruns only interrupted directories.
"""

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


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def latest_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=15507)
    parser.add_argument(
        "--gpu-ids",
        default="",
        help=(
            "Optional comma-separated Unity graphics device indices. Tasks are "
            "assigned round-robin through INDUSTRYNAV_UNITY_DEVICE_INDEX."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Optional exact scene/episode_id filter; repeat for multiple tasks.",
    )
    return parser.parse_args()


def episode_success(result: dict | None) -> bool:
    if not result:
        return False
    try:
        distance = float(result.get("distance_world", "inf"))
    except (TypeError, ValueError):
        return False
    return result.get("stop_reason") in {"astar_stop", "reached_vicinity"} and distance <= 2.0


def command_for(args: argparse.Namespace, task: dict, task_index: int, episode_dir: Path) -> list[str]:
    command = [
        str(args.python), "-m", "nav.scripts.agent.run_benchmark_cell",
        "--baseline", "astar",
        "--file_name", str(args.unity),
        "--scene_id", str(task["scene_id"]),
        "--scene_name", str(task["scene_name"]),
        "--point_id", str(task["episode_id"]),
        "--seed_id", str(task["seed"]),
        "--worker_id", "0",
        "--base_port", str(args.base_port + task_index),
        "--max_steps", "320",
        "--astar_dynamic_step_budget",
        "--astar_step_budget_min", "80",
        "--astar_step_budget_max", "320",
        "--astar_steps_per_path_meter", "2.2",
        "--astar_step_budget_overhead", "60",
        "--reach_m", "2.0",
        # One Unity step makes ACTION_SPACE_POINTGOAL +/-11.25 an actual
        # 22.5-degree decision. Two steps would rotate about 45 degrees and
        # induce aliasing.
        "--sim_steps_per_decision", "1",
        "--ego_width", "320",
        "--ego_height", "240",
        "--minimap_width", "862",
        "--dynamic_objects", str(task["dynamic_objects"]),
        "--modalities", "ego,minimap,depth",
        "--marker_source", "vector",
        "--frame_save_dir", str(episode_dir),
        "--model_id", "astar-policy-teacher-v1",
        "--astar_obstacle_clearance_m", "0.6",
        "--astar_dynamic_replan_lookahead_m", "8.0",
        "--astar_dynamic_replan_confirm_steps", "2",
        "--astar_policy_actions",
        "--init_world_x", str(task["init_world_x"]),
        "--init_world_z", str(task["init_world_z"]),
        "--init_curr_direction", str(task["init_direction"]),
        "--target_x", str(task["target_x"]),
        "--target_y", str(task["target_y"]),
        "--motion_random_seed", str(task["motion_random_seed"]),
        "--human_speed_min_mps", "0.8", "--human_speed_max_mps", "1.6",
        "--vehicle_speed_min_mps", "1.5", "--vehicle_speed_max_mps", "3.5",
        "--robot_speed_min_mps", "1.0", "--robot_speed_max_mps", "2.0",
        # Keep scene-authored lighting/exposure for teacher collection. Runtime
        # HDRP exposure randomization can saturate the minimap and erase thin
        # obstacle contrast; photometric diversity belongs in train-time image
        # augmentation, where it cannot corrupt A* geometry labels.
    ]
    if platform.system() == "Linux" and shutil.which("xvfb-run"):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1724x1024x24", *command]
    return command


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    try:
        gpu_ids = tuple(
            int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()
        )
    except ValueError as exc:
        raise SystemExit("--gpu-ids must be a comma-separated list of integers") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise SystemExit("--gpu-ids values must be non-negative")
    if not args.unity.is_file():
        raise SystemExit(f"Unity executable not found: {args.unity}")
    tasks = read_jsonl(args.manifest)
    if args.episode:
        selected = set(args.episode)
        tasks = [
            task for task in tasks
            if f"{task['scene_name']}/{task['episode_id']}" in selected
        ]
        missing = selected - {
            f"{task['scene_name']}/{task['episode_id']}" for task in tasks
        }
        if missing:
            raise SystemExit(f"Unknown --episode value(s): {sorted(missing)}")
    if args.max_episodes > 0:
        tasks = tasks[: args.max_episodes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    state_path = args.output_root / "collection_state.json"
    state_lock = threading.Lock()
    state = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_digest,
        "workers": args.workers,
        "gpu_ids": list(gpu_ids),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": {},
    }

    def save_status(task: dict, status: str, **extra) -> None:
        with state_lock:
            state["tasks"][f"{task['scene_name']}/{task['episode_id']}"] = {
                "status": status,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                **extra,
            }
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_json(state_path, state)

    pending = []
    for index, task in enumerate(tasks):
        episode_dir = args.output_root / task["scene_name"] / task["episode_id"]
        result = latest_result(episode_dir / "results.csv")
        if args.resume and result is not None and (episode_success(result) or not args.retry_failed):
            save_status(task, "skipped_success" if episode_success(result) else "skipped_failed")
            continue
        pending.append((index, task, episode_dir))

    print(f"manifest={args.manifest} tasks={len(tasks)} pending={len(pending)} workers={args.workers}")
    if args.dry_run:
        for index, task, episode_dir in pending:
            print(" ".join(command_for(args, task, index, episode_dir)))
        return

    def run_one(item) -> tuple[dict, int, bool]:
        index, task, episode_dir = item
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "collection_task.json").write_text(
            json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command = command_for(args, task, index, episode_dir)
        gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
        save_status(task, "running", command=command, gpu_id=gpu_id)
        process_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if gpu_id is not None:
            process_env["INDUSTRYNAV_UNITY_DEVICE_INDEX"] = str(gpu_id)
        with (episode_dir / "runner.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=process_env,
                check=False,
            )
        result = latest_result(episode_dir / "results.csv")
        success = process.returncode == 0 and episode_success(result)
        save_status(
            task,
            "success" if success else "failed",
            returncode=process.returncode,
            stop_reason=(result or {}).get("stop_reason"),
            distance_world=(result or {}).get("distance_world"),
            gpu_id=gpu_id,
        )
        print(f"[{task['scene_name']}/{task['episode_id']}] rc={process.returncode} success={success}", flush=True)
        return task, process.returncode, success

    succeeded = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, item) for item in pending]
        for future in as_completed(futures):
            _task, _returncode, success = future.result()
            succeeded += int(success)
            failed += int(not success)
    print(f"collection finished: success={succeeded} failed={failed} skipped={len(tasks)-len(pending)}")


if __name__ == "__main__":
    main()
