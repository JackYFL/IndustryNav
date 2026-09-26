"""Run a small closed-loop learned PointGoal evaluation on held-out scenes."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nav.config import (
    ACTIONS_CSV_FIELDS,
    ACTION_SPACE_POINTGOAL,
    SCENE_CODES,
    SCENE_ID_MAP,
)
from nav.utils import action2signal, append_results_csv, save_frame_from_obs
from nav.eval.budgets import (
    apply_reference_budget, load_reference_budgets, validate_completed_budgets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--manifest", type=Path)
    task_source.add_argument(
        "--input-points",
        type=Path,
        help="Evaluate the canonical input_points.json benchmark directly.",
    )
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
    parser.add_argument("--step-budget-reference", type=Path,
                        help="Pin each task to this prior report's budget; persistent evaluation only.")
    parser.add_argument("--step-budget-reference-sha256",
                        help="Required exact SHA256 when pinning a step-budget reference.")
    parser.add_argument(
        "--fallback-ppo-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional second PPO policy kept recurrently synchronized and "
            "activated after the configured collision-step threshold."
        ),
    )
    parser.add_argument(
        "--fallback-bc-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional BC/DAgger recovery policy kept synchronized with the "
            "primary PPO and activated by the same collision/stagnation trigger."
        ),
    )
    parser.add_argument(
        "--fallback-collision-steps",
        type=int,
        default=0,
        help=(
            "Cumulative physical-collision steps that activate the fallback "
            "policy; zero disables this trigger."
        ),
    )
    parser.add_argument(
        "--fallback-stagnation-steps",
        type=int,
        default=0,
        help=(
            "Consecutive steps without a new best goal distance that activate "
            "the fallback policy; zero disables this trigger."
        ),
    )
    parser.add_argument(
        "--fallback-collision-stagnation-steps",
        type=int,
        default=0,
        help=(
            "When positive, a collision-triggered switch additionally "
            "requires this many consecutive no-progress steps."
        ),
    )
    parser.add_argument(
        "--second-fallback-bc-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional final BC/DAgger recovery stage after a PPO fallback. "
            "It is synchronized from episode start."
        ),
    )
    parser.add_argument(
        "--second-fallback-collision-steps",
        type=int,
        default=0,
        help=(
            "Cumulative physical-collision steps that activate the final "
            "BC/DAgger recovery stage."
        ),
    )
    parser.add_argument(
        "--second-fallback-stagnation-steps",
        type=int,
        default=0,
        help=(
            "Consecutive no-progress steps that activate the final BC/DAgger "
            "recovery stage."
        ),
    )
    parser.add_argument(
        "--second-fallback-collision-stagnation-steps",
        type=int,
        default=0,
        help=(
            "When positive, a collision-triggered switch to the final "
            "BC/DAgger stage additionally requires this many consecutive "
            "no-progress steps."
        ),
    )
    parser.add_argument(
        "--second-fallback-min-distance-m",
        type=float,
        default=0.0,
        help=(
            "Only activate the final recovery stage at or beyond this goal "
            "distance. Zero disables distance gating."
        ),
    )
    parser.add_argument(
        "--second-fallback-min-step-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum consumed episode-budget fraction for the normal final "
            "recovery gate; zero allows immediate activation."
        ),
    )
    parser.add_argument(
        "--second-fallback-far-distance-m",
        type=float,
        default=0.0,
        help=(
            "Goal distance that bypasses --second-fallback-min-step-fraction "
            "for urgent long-range recovery; zero disables the bypass."
        ),
    )
    parser.add_argument(
        "--second-fallback-close-recovery-min-distance-m",
        type=float,
        default=0.0,
        help=(
            "Optional lower bound for a late-episode close-recovery band. "
            "The band can bypass --second-fallback-min-distance-m."
        ),
    )
    parser.add_argument(
        "--second-fallback-close-recovery-max-distance-m",
        type=float,
        default=0.0,
        help=(
            "Optional upper bound for the late-episode close-recovery band; "
            "zero disables the band."
        ),
    )
    parser.add_argument(
        "--second-fallback-close-recovery-min-step-fraction",
        type=float,
        default=0.75,
        help=(
            "Minimum consumed episode-budget fraction for close recovery "
            "(default: 0.75)."
        ),
    )
    parser.add_argument(
        "--waypoint-planner-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional learned scene-conditioned local-waypoint planner. It "
            "uses coordinates and scene identity, never minimap pixels."
        ),
    )
    parser.add_argument(
        "--waypoint-terminal-distance-m",
        type=float,
        default=-1.0,
        help=(
            "Use the final goal directly inside this distance; -1 uses the "
            "planner checkpoint's lookahead."
        ),
    )
    parser.add_argument(
        "--route-graph-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional topological route memory built only from independent "
            "A* demonstrations; it never reads minimap pixels at evaluation."
        ),
    )
    parser.add_argument(
        "--route-graph-lookahead-m",
        type=float,
        default=-1.0,
        help="Route-graph local waypoint distance; -1 uses the graph default.",
    )
    parser.add_argument(
        "--route-graph-terminal-distance-m",
        type=float,
        default=-1.0,
        help="Use the final goal directly inside this distance; -1 uses lookahead.",
    )
    parser.add_argument(
        "--route-graph-max-snap-distance-m",
        type=float,
        default=-1.0,
        help="Maximum GPS-to-graph snap distance; -1 uses the graph default.",
    )
    parser.add_argument(
        "--route-graph-activation-collision-steps",
        type=int,
        default=0,
        help=(
            "Delay route-graph waypoints until this many cumulative physical "
            "collision steps; zero enables them from episode start."
        ),
    )
    parser.add_argument("--baseline", choices=("bc", "ppo"), default="bc")
    parser.add_argument(
        "--model-id",
        default="",
        help=(
            "Model label written to per-episode results and galleries. "
            "Defaults to pointgoal-<baseline>."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch inference device for the learned policy (default: auto).",
    )
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=1,
        help="Number of held-out episodes to evaluate in each selected scene.",
    )
    parser.add_argument("--base-port", type=int, default=19507)
    parser.add_argument(
        "--dynamic-objects",
        choices=("moving", "static"),
        default=None,
        help=(
            "Override environment motion for all evaluation tasks. Static "
            "freezes objects in place while keeping the agent controllable; "
            "omitting this flag preserves the existing moving benchmark."
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Fixed dynamic-object seed used for input_points.json evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent Unity evaluation processes.",
    )
    parser.add_argument(
        "--persistent-policy",
        "--persistent-unity",
        dest="persistent_policy",
        action="store_true",
        help=(
            "Load one policy per worker while running isolated Unity episodes. "
            "This is the recommended full-benchmark mode."
        ),
    )
    parser.add_argument(
        "--safety-shield",
        action="store_true",
        help="Recover from sustained physical contact with a clearance-seeking turn.",
    )
    parser.add_argument(
        "--shield-collision-streak",
        type=int,
        default=8,
        help="Consecutive blocked forward steps before collision recovery (default: 8).",
    )
    parser.add_argument(
        "--shield-recovery-turns",
        type=int,
        default=2,
        help="Consistent turn actions scheduled by each collision recovery (default: 2).",
    )
    parser.add_argument(
        "--shield-recovery-forward-steps",
        type=int,
        default=0,
        help=(
            "Clear-depth forward actions after a recovery turn; zero preserves "
            "the contact-turn-only shield (default: 0)."
        ),
    )
    parser.add_argument(
        "--shield-max-clearance-turns",
        type=int,
        default=-1,
        help=(
            "Maximum extra turns while an escape-forward remains depth-blocked; "
            "-1 preserves the legacy unbounded behavior."
        ),
    )
    parser.add_argument(
        "--shield-warning-streak",
        type=int,
        default=0,
        help=(
            "Consecutive warned forward proposals before a pre-contact "
            "clearance turn; zero disables proactive warning recovery."
        ),
    )
    parser.add_argument(
        "--shield-warning-turns",
        type=int,
        default=2,
        help="Turns scheduled by proactive warning recovery (default: 2).",
    )
    parser.add_argument(
        "--shield-turn-streak",
        type=int,
        default=0,
        help=(
            "Consecutive executed turns before a clear-depth forward escape; "
            "zero disables turn-loop recovery."
        ),
    )
    parser.add_argument(
        "--shield-turn-escape-steps",
        type=int,
        default=4,
        help="Maximum clear forward steps used to break a turn loop (default: 4).",
    )
    parser.add_argument(
        "--fallback-turn-streak",
        type=int,
        default=0,
        help=(
            "Consecutive turns before a clear-depth forward escape while a "
            "fallback controller is active; zero keeps the global setting."
        ),
    )
    parser.add_argument(
        "--fallback-turn-escape-steps",
        type=int,
        default=4,
        help=(
            "Maximum clear forward steps used to break a fallback-only turn "
            "loop (default: 4)."
        ),
    )
    parser.add_argument(
        "--fallback-turn-final-stage-only",
        action="store_true",
        help=(
            "In a three-controller cascade, apply fallback turn-loop "
            "escape only after the final recovery controller is active."
        ),
    )
    parser.add_argument(
        "--fallback-abandon-turn-escapes",
        type=int,
        default=0,
        help=(
            "Return from the deepest active fallback to its preceding "
            "synchronized controller after this many turn-escape "
            "interventions; zero disables abandonment."
        ),
    )
    parser.add_argument(
        "--shield-terminal-homing-distance-m",
        type=float,
        default=0.0,
        help=(
            "Inside this distance, prefer the direct PointGoal bearing when "
            "the requested forward step is depth-clear; zero disables."
        ),
    )
    parser.add_argument(
        "--shield-terminal-homing-tolerance-deg",
        type=float,
        default=11.25,
        help="Bearing deadband for optional terminal homing (default: 11.25).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed persistent worker episodes from the output root.",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help=(
            "Save ego/depth/minimap frames plus gallery-compatible action and "
            "result CSV files during persistent evaluation."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def should_activate_fallback(
    *,
    fallback_available: bool,
    fallback_active: bool,
    collision_steps: int,
    collision_threshold: int,
    stagnation_steps: int = 0,
    stagnation_threshold: int = 0,
    collision_stagnation_threshold: int = 0,
    distance_m: float = 0.0,
    min_distance_m: float = 0.0,
    min_step_fraction: float = 0.0,
    far_distance_m: float = 0.0,
    episode_steps: int = 0,
    episode_max_steps: int = 1,
    close_recovery_min_distance_m: float = 0.0,
    close_recovery_max_distance_m: float = 0.0,
    close_recovery_min_step_fraction: float = 0.75,
) -> bool:
    """Return whether a synchronized fallback policy should take control.

    ``UnityPointGoalEnv.step`` only adds episode totals to ``info`` on the
    terminal transition.  Callers therefore pass the environment's live
    collision counter here instead of reading a terminal-only ``info`` key.
    """
    collision_triggered = (
        collision_threshold > 0
        and collision_steps >= collision_threshold
        and (
            collision_stagnation_threshold <= 0
            or stagnation_steps >= collision_stagnation_threshold
        )
    )
    stagnation_triggered = (
        stagnation_threshold > 0 and stagnation_steps >= stagnation_threshold
    )
    step_fraction = episode_steps / max(episode_max_steps, 1)
    normal_recovery_eligible = (
        (min_distance_m <= 0.0 or distance_m >= min_distance_m)
        and step_fraction >= min_step_fraction
    )
    urgent_far_recovery_eligible = (
        far_distance_m > 0.0 and distance_m >= far_distance_m
    )
    close_recovery_eligible = (
        close_recovery_max_distance_m > 0.0
        and distance_m >= close_recovery_min_distance_m
        and distance_m <= close_recovery_max_distance_m
        and step_fraction >= close_recovery_min_step_fraction
    )
    return (
        fallback_available
        and not fallback_active
        and (collision_triggered or stagnation_triggered)
        and (
            normal_recovery_eligible
            or urgent_far_recovery_eligible
            or close_recovery_eligible
        )
    )


def should_abandon_fallback(
    *,
    fallback_active: bool,
    turn_escape_interventions: int,
    activation_turn_escape_interventions: int,
    threshold: int,
) -> bool:
    """Return to the preceding controller when recovery repeatedly spins."""
    return bool(
        fallback_active
        and threshold > 0
        and turn_escape_interventions - activation_turn_escape_interventions
        >= threshold
    )


def tasks_from_input_points(
    path: Path,
    *,
    eval_seed: int = 0,
    dynamic_objects: str = "moving",
) -> list[dict]:
    """Translate canonical benchmark points into evaluator task records."""
    points = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for scene_name in SCENE_CODES:
        for entry in points.get(scene_name, []):
            tasks.append({
                "split": "test",
                "scene_id": SCENE_ID_MAP[scene_name],
                "scene_name": scene_name,
                "episode_id": str(entry["point_id"]),
                "seed": int(eval_seed),
                "motion_random_seed": int(eval_seed),
                "dynamic_objects": dynamic_objects,
                "init_world_x": float(entry["start"]["x"]),
                "init_world_z": float(entry["start"]["z"]),
                "init_direction": float(entry["start"]["direction"]),
                "target_x": int(round(float(entry["target"]["x"]))),
                "target_y": int(round(float(entry["target"]["y"]))),
            })
    return tasks


def select_held_out_tasks(
    records: list[dict],
    scene_count: int,
    eligible_records: list[dict] | None = None,
    episodes_per_scene: int = 1,
) -> list[dict]:
    eligible = None
    if eligible_records is not None:
        eligible = {
            (str(record["scene_name"]), str(record["episode_id"]))
            for record in eligible_records
        }

    if scene_count <= 0:
        raise ValueError("scene_count must be positive")
    if episodes_per_scene <= 0:
        raise ValueError("episodes_per_scene must be positive")

    selected = []
    selected_scenes: list[str] = []
    episode_counts: dict[str, int] = {}
    for record in records:
        key = (str(record["scene_name"]), str(record["episode_id"]))
        scene_name = str(record["scene_name"])
        if record["split"] != "test":
            continue
        if eligible is not None and key not in eligible:
            continue
        if scene_name not in episode_counts:
            if len(selected_scenes) >= scene_count:
                continue
            selected_scenes.append(scene_name)
            episode_counts[scene_name] = 0
        if episode_counts[scene_name] >= episodes_per_scene:
            continue
        selected.append(record)
        episode_counts[scene_name] += 1
        if len(selected) >= scene_count * episodes_per_scene:
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
    device = getattr(args, "device", "auto")
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
        "--model_id", (
            getattr(args, "model_id", "") or f"pointgoal-{baseline}"
        ),
        "--init_world_x", str(task["init_world_x"]),
        "--init_world_z", str(task["init_world_z"]),
        "--init_curr_direction", str(task["init_direction"]),
        "--target_x", str(int(round(float(task["target_x"])))),
        "--target_y", str(int(round(float(task["target_y"])))),
        "--motion_random_seed", str(task["motion_random_seed"]),
        # Match A* teacher collection on every platform. Runtime HDRP exposure
        # overrides saturate the minimap and darken the egocentric camera.
        # Photometric diversity is applied by the training data pipeline.
    ]
    if baseline == "bc":
        command.extend([
            "--bc_ckpt", str(args.checkpoint), "--bc_device", str(device),
            "--bc_pointgoal_actions",
        ])
    else:
        command.extend([
            "--ppo_ckpt", str(args.checkpoint), "--ppo_device", str(device),
        ])
    if platform.system() == "Linux" and shutil.which("xvfb-run"):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1724x1024x24", *command]
    return command


def evaluate_task(args: argparse.Namespace, task: dict, task_index: int) -> dict:
    episode_dir = args.output_root / task["scene_name"] / task["episode_id"]
    command = command_for(args, task, task_index, episode_dir)
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
    return {
        "scene_name": task["scene_name"],
        "episode_id": task["episode_id"],
        "returncode": process.returncode,
        "success": success,
        "stop_reason": result.get("stop_reason"),
        "distance_world": result.get("distance_world"),
        "steps_taken": result.get("steps_taken"),
    }


def shard_tasks_by_scene(tasks: list[dict], workers: int) -> list[list[dict]]:
    """Balance whole scenes across workers so Unity relaunches stay minimal."""
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_scene[str(task["scene_name"])].append(task)
    shards: list[list[dict]] = [[] for _ in range(min(workers, len(by_scene)))]
    for scene_index, scene_tasks in enumerate(by_scene.values()):
        shards[scene_index % len(shards)].extend(scene_tasks)
    return shards


class PersistentVisualRecorder:
    """Write a persistent-policy episode in the standard gallery layout."""

    def __init__(
        self,
        args: argparse.Namespace,
        env,
        task: dict,
    ) -> None:
        from nav.harness.coordinates import (
            minimap_to_canonical_coords,
            world_to_visual_coords,
        )

        self.args = args
        self.env = env
        self.task = task
        self.prefix = str(args.baseline)
        self.episode_dir = (
            args.output_root / str(task["scene_name"]) / str(task["episode_id"])
        )
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dirs = {
            "ego": self.episode_dir / f"{self.prefix}_fp",
            "depth": self.episode_dir / f"{self.prefix}_depth",
            "minimap": self.episode_dir / f"{self.prefix}_minimap_target",
        }
        for directory in self.frame_dirs.values():
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
        for filename in (f"{self.prefix}_actions.csv", "results.csv"):
            path = self.episode_dir / filename
            if path.exists():
                path.unlink()
        self.actions_path = self.episode_dir / f"{self.prefix}_actions.csv"
        self._stream = self.actions_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=ACTIONS_CSV_FIELDS)
        self._writer.writeheader()

        if env.primed is None or env.pose is None:
            raise RuntimeError("Visual recording requires a primed Unity episode")
        initial_runtime = world_to_visual_coords(
            env.primed.margin,
            env.pose[0],
            env.pose[1],
            img_shape=(env.minimap_height, env.minimap_width, 3),
            projector=env.minimap_projector,
            map_size=env.primed.minimap_size,
        )
        self.initial_pixel = minimap_to_canonical_coords(
            initial_runtime,
            env.primed.minimap_size,
        )

    def _current_pixel(self) -> tuple[float, float]:
        from nav.harness.coordinates import (
            minimap_to_canonical_coords,
            world_to_visual_coords,
        )

        if self.env.primed is None or self.env.pose is None:
            raise RuntimeError("Visual recording lost the Unity pose")
        runtime_pixel = world_to_visual_coords(
            self.env.primed.margin,
            self.env.pose[0],
            self.env.pose[1],
            img_shape=(self.env.minimap_height, self.env.minimap_width, 3),
            projector=self.env.minimap_projector,
            map_size=self.env.primed.minimap_size,
        )
        return minimap_to_canonical_coords(
            runtime_pixel,
            self.env.primed.minimap_size,
        )

    def record(self, action_name: str) -> None:
        from PIL import Image

        if (
            self.env.ego_obs is None
            or self.env.depth_obs is None
            or self.env.minimap_rgb is None
            or self.env.pose is None
            or self.env.target_world is None
        ):
            raise RuntimeError("Persistent visual observations are unavailable")
        step = int(self.env.episode_steps)
        filename = f"{step}.png"
        save_frame_from_obs(
            self.env.ego_obs,
            str(self.frame_dirs["ego"]),
            filename,
        )
        depth = self.env.depth_obs
        if depth.ndim == 2:
            depth = depth[None, ...]
        save_frame_from_obs(depth, str(self.frame_dirs["depth"]), filename)
        Image.fromarray(self.env.minimap_rgb).save(
            self.frame_dirs["minimap"] / filename
        )

        curr_px, curr_py = self._current_pixel()
        target_px = float(self.task["target_x"])
        target_py = float(self.task["target_y"])
        signal = action2signal(action_name, ACTION_SPACE_POINTGOAL)[0]
        self._writer.writerow({
            "step": step,
            "action": action_name,
            "move": float(signal[0]),
            "strafe": float(signal[1]),
            "look": float(signal[2]),
            "init_px": round(self.initial_pixel[0]),
            "init_py": round(self.initial_pixel[1]),
            "init_world_x": float(self.task["init_world_x"]),
            "init_world_z": float(self.task["init_world_z"]),
            "init_direction": float(self.task["init_direction"]),
            "curr_px": round(curr_px),
            "curr_py": round(curr_py),
            "curr_world_x": float(self.env.pose[0]),
            "curr_world_y": 0.5,
            "curr_world_z": float(self.env.pose[1]),
            "curr_direction_x": 0.0,
            "curr_direction_y": float(self.env.pose[2]),
            "curr_direction_z": 0.0,
            "target_px": int(round(target_px)),
            "target_py": int(round(target_py)),
            "target_world_x": float(self.env.target_world[0]),
            "target_world_z": float(self.env.target_world[1]),
            "marker_source": "vector",
            "distance_px": math.hypot(curr_px - target_px, curr_py - target_py),
            "distance_world": float(self.env.distance_m),
        })
        self._stream.flush()

    def finish(self, summary: dict) -> None:
        self.close()
        if self.env.pose is None or self.env.target_world is None:
            raise RuntimeError("Visual recording lost the final Unity pose")
        model_id = self.args.model_id or f"pointgoal-{self.args.baseline}"
        append_results_csv(str(self.episode_dir / "results.csv"), {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "exp_name": str(self.task["scene_name"]),
            "scene_name": str(self.task["scene_name"]),
            "point_id": str(self.task["episode_id"]),
            "seed_id": str(self.task.get("seed", 0)),
            "exec_mode": str(self.args.baseline),
            "provider": "",
            "model": model_id,
            "vision_input": True,
            "max_steps": int(summary["step_budget"]),
            "step_budget_max": 320,
            "step_budget_mode": "dynamic",
            "initial_step_budget": int(summary["step_budget"]),
            "reach_m": 2.0,
            "init_world_x": float(self.task["init_world_x"]),
            "init_world_z": float(self.task["init_world_z"]),
            "init_direction": float(self.task["init_direction"]),
            "target_world_x": float(self.env.target_world[0]),
            "target_world_z": float(self.env.target_world[1]),
            "final_world_x": float(self.env.pose[0]),
            "final_world_z": float(self.env.pose[1]),
            "distance_world": float(summary["distance_world"]),
            "stop_reason": str(summary["stop_reason"]),
            "steps_taken": int(summary["steps_taken"]),
            "resume_count": 0,
            "frame_sleep": 0.0,
            "modalities": "depth,ego,minimap",
            "sim_steps_per_decision": 1,
            "marker_source": "vector",
            "dynamic_objects": str(self.task.get("dynamic_objects", "moving")),
            "light_fixed_exposure": 9.0,
        })

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


def evaluate_persistent_worker(
    args: argparse.Namespace,
    tasks: list[dict],
    worker_index: int,
) -> list[dict]:
    """Evaluate a scene shard while loading the policy and Unity only once."""
    from nav.baselines.rl import (
        PPO_ACTIONS,
        RouteGraphController,
        SceneWaypointController,
        compute_pointgoal_reward,
        pointgoal_homing_action,
    )
    from nav.envs.unity_pointgoal import UnityPointGoalEnv
    from nav.safety import ReactiveSafetyShield
    from nav.utils import decode_depth_observation_meters

    worker_result_path = args.output_root / f"worker_{worker_index}.jsonl"
    existing = (
        read_jsonl(worker_result_path)
        if args.resume and worker_result_path.is_file()
        else []
    )
    if any(
        row.get("dynamic_objects", "moving") != (args.dynamic_objects or "moving")
        for row in existing
    ):
        raise ValueError("Cannot resume results from a different dynamic-objects mode")
    fixed_budgets = getattr(args, "_reference_budgets", None)
    validate_completed_budgets(existing, fixed_budgets)
    completed = {
        (str(row["scene_name"]), str(row["episode_id"])) for row in existing
    }
    pending_tasks = [
        task
        for task in tasks
        if (str(task["scene_name"]), str(task["episode_id"])) not in completed
    ]
    if not pending_tasks:
        return existing

    if args.baseline == "ppo":
        from nav.baselines.rl.agent import PPOPointGoalController

        controller = PPOPointGoalController(
            str(args.checkpoint), device=args.device
        )
        fallback_controller = (
            PPOPointGoalController(
                str(args.fallback_ppo_checkpoint), device=args.device
            )
            if args.fallback_ppo_checkpoint is not None
            else None
        )
        if args.fallback_bc_checkpoint is not None:
            from nav.baselines.bc.agent import BCNavController

            fallback_controller = BCNavController(
                str(args.fallback_bc_checkpoint), device=args.device
            )
        second_fallback_controller = None
        if args.second_fallback_bc_checkpoint is not None:
            from nav.baselines.bc.agent import BCNavController

            second_fallback_controller = BCNavController(
                str(args.second_fallback_bc_checkpoint), device=args.device
            )
        fallback_is_ppo = args.fallback_ppo_checkpoint is not None
        waypoint_controller = (
            SceneWaypointController(
                str(args.waypoint_planner_checkpoint),
                device=args.device,
                terminal_distance_m=(
                    None
                    if args.waypoint_terminal_distance_m < 0.0
                    else args.waypoint_terminal_distance_m
                ),
            )
            if args.waypoint_planner_checkpoint is not None
            else None
        )
        route_graph_controller = (
            RouteGraphController(
                str(args.route_graph_checkpoint),
                lookahead_m=(
                    None
                    if args.route_graph_lookahead_m < 0.0
                    else args.route_graph_lookahead_m
                ),
                terminal_distance_m=(
                    None
                    if args.route_graph_terminal_distance_m < 0.0
                    else args.route_graph_terminal_distance_m
                ),
                max_snap_distance_m=(
                    None
                    if args.route_graph_max_snap_distance_m < 0.0
                    else args.route_graph_max_snap_distance_m
                ),
            )
            if args.route_graph_checkpoint is not None
            else None
        )
    else:
        from nav.baselines.bc.agent import BCNavController

        controller = BCNavController(str(args.checkpoint), device=args.device)
        fallback_controller = None
        second_fallback_controller = None
        fallback_is_ppo = False
        waypoint_controller = None
        route_graph_controller = None
    env = UnityPointGoalEnv(
        pending_tasks,
        unity_path=str(args.unity),
        output_dir=args.output_root / "unity",
        worker_id=worker_index,
        base_port=args.base_port,
        reach_m=2.0,
        max_steps=320,
        ego_width=320,
        ego_height=240,
        minimap_width=862,
        dynamic_objects=args.dynamic_objects or "moving",
        absolute_scene_state=(
            bool(controller.config.absolute_scene_state)
            if args.baseline == "ppo"
            else False
        ),
        num_scenes=(
            int(controller.config.num_scenes)
            if args.baseline == "ppo"
            else 24
        ),
        world_coordinate_scale_m=(
            float(controller.config.world_coordinate_scale_m)
            if args.baseline == "ppo"
            else 50.0
        ),
        task_sampling="sequential",
        stuck_recovery_steps=0,
        dynamic_step_budget=True,
        step_budget_min=80,
        step_budget_max=320,
        steps_per_path_meter=2.2,
        step_budget_overhead=60,
        auto_reset=False,
        # A fresh Unity process reproduces the canonical one-task benchmark's
        # moving-object initial state while still keeping the model resident.
        reuse_same_scene=False,
        record_visuals=args.save_visuals,
        action_names=PPO_ACTIONS,
        reward_fn=compute_pointgoal_reward,
    )
    summaries: list[dict] = list(existing)
    shield = (
        ReactiveSafetyShield(
            max_clearance_turns=(
                None
                if args.shield_max_clearance_turns < 0
                else args.shield_max_clearance_turns
            )
        )
        if args.safety_shield
        else None
    )
    if not args.resume:
        worker_result_path.write_text("", encoding="utf-8")
    recorder: PersistentVisualRecorder | None = None
    try:
        observation = env.reset()
        for task_index, task in enumerate(pending_tasks):
            if env.task is not task:
                raise RuntimeError(
                    "Persistent evaluator task order diverged from environment shard"
                )
            dynamic_budget = apply_reference_budget(env, task, fixed_budgets)
            controller.reset()
            if fallback_controller is not None:
                fallback_controller.reset()
            if second_fallback_controller is not None:
                second_fallback_controller.reset()
            if route_graph_controller is not None:
                route_graph_controller.reset()
            if shield is not None:
                shield.reset()
            recorder = (
                PersistentVisualRecorder(args, env, task)
                if args.save_visuals
                else None
            )
            collision_streak = 0
            warning_streak = 0
            proactive_warning_recoveries = 0
            turn_streak = 0
            turn_escape_remaining = 0
            turn_escape_interventions = 0
            terminal_homing_interventions = 0
            fallback_active = False
            fallback_ever_activated = False
            fallback_disabled = False
            fallback_activation_turn_escape_interventions = 0
            fallback_abandon_step: int | None = None
            fallback_abandon_distance_m: float | None = None
            fallback_trigger_step: int | None = None
            fallback_trigger_distance_m: float | None = None
            fallback_trigger_stagnation_steps: int | None = None
            fallback_action_counts: Counter[str] = Counter()
            second_fallback_active = False
            second_fallback_ever_activated = False
            second_fallback_disabled = False
            second_fallback_activation_turn_escape_interventions = 0
            second_fallback_abandon_step: int | None = None
            second_fallback_abandon_distance_m: float | None = None
            second_fallback_trigger_step: int | None = None
            second_fallback_trigger_distance_m: float | None = None
            second_fallback_trigger_step_fraction: float | None = None
            second_fallback_action_counts: Counter[str] = Counter()
            predicted_action_counts: Counter[str] = Counter()
            executed_action_counts: Counter[str] = Counter()
            while True:
                assert env.pose is not None and env.target_world is not None
                state = {
                    "depth_obs": env.depth_obs,
                    "curr_world_x": env.pose[0],
                    "curr_world_z": env.pose[1],
                    "curr_yaw_deg": env.pose[2],
                    "target_world_x": env.target_world[0],
                    "target_world_z": env.target_world[1],
                }
                route_graph_active = (
                    route_graph_controller is not None
                    and int(env.episode_collisions)
                    >= args.route_graph_activation_collision_steps
                )
                if route_graph_active:
                    waypoint_x, waypoint_z = route_graph_controller.predict_waypoint(
                        scene_id=int(task["scene_id"]),
                        curr_world_x=env.pose[0],
                        curr_world_z=env.pose[1],
                        target_world_x=env.target_world[0],
                        target_world_z=env.target_world[1],
                    )
                    state["target_world_x"] = waypoint_x
                    state["target_world_z"] = waypoint_z
                elif waypoint_controller is not None:
                    waypoint_x, waypoint_z = waypoint_controller.predict_waypoint(
                        scene_id=int(task["scene_id"]),
                        curr_world_x=env.pose[0],
                        curr_world_z=env.pose[1],
                        target_world_x=env.target_world[0],
                        target_world_z=env.target_world[1],
                    )
                    state["target_world_x"] = waypoint_x
                    state["target_world_z"] = waypoint_z
                primary_action = (
                    controller.predict_action(
                        **state,
                        scene_id=int(task["scene_id"]),
                    )
                    if args.baseline == "ppo"
                    else controller.predict_action(ego_obs=None, **state)
                )
                fallback_action = (
                    (
                        fallback_controller.predict_action(
                            **state,
                            scene_id=int(task["scene_id"]),
                        )
                        if fallback_is_ppo
                        else fallback_controller.predict_action(
                            ego_obs=None,
                            **state,
                        )
                    )
                    if fallback_controller is not None
                    else None
                )
                second_fallback_action = (
                    second_fallback_controller.predict_action(
                        ego_obs=None,
                        **state,
                    )
                    if second_fallback_controller is not None
                    else None
                )
                predicted_action = (
                    second_fallback_action
                    if (
                        second_fallback_active
                        and second_fallback_action is not None
                    )
                    else (
                        fallback_action
                        if fallback_active and fallback_action is not None
                        else primary_action
                    )
                )
                if second_fallback_active:
                    second_fallback_action_counts[predicted_action] += 1
                elif fallback_active:
                    fallback_action_counts[predicted_action] += 1
                action_name = predicted_action
                predicted_action_counts[predicted_action] += 1
                intervened = False
                if shield is not None:
                    depth_m = decode_depth_observation_meters(env.depth_obs)
                    forward_warning = shield.warning_detector.detect(
                        depth_m,
                        move_command=7.5,
                    )["warning"] == "yes"
                    if (
                        args.shield_terminal_homing_distance_m > 0.0
                        and env.distance_m
                        <= args.shield_terminal_homing_distance_m
                    ):
                        homing_action = pointgoal_homing_action(
                            float(observation["goal"][1]),
                            float(observation["goal"][2]),
                            turn_tolerance_deg=(
                                args.shield_terminal_homing_tolerance_deg
                            ),
                        )
                        # Never replace a learned avoidance turn with a
                        # forward command that the metric-depth detector has
                        # already classified as unsafe.
                        if homing_action != "forward" or not forward_warning:
                            if homing_action != action_name:
                                action_name = homing_action
                                intervened = True
                                terminal_homing_interventions += 1
                    warned_forward = (
                        action_name == "forward"
                        and forward_warning
                    )
                    warning_streak = warning_streak + 1 if warned_forward else 0
                    if (
                        args.shield_warning_streak > 0
                        and warning_streak >= args.shield_warning_streak
                    ):
                        shield.begin_collision_recovery(
                            depth_m,
                            goal_bearing_sin=float(observation["goal"][1]),
                            move_command=7.5,
                            turns=args.shield_warning_turns,
                            escape_forwards=args.shield_recovery_forward_steps,
                        )
                        warning_streak = 0
                        proactive_warning_recoveries += 1
                    action_name, shield_intervened = shield.filter_action(
                        action_name,
                        depth_m=depth_m,
                        move_command=7.5,
                    )
                    intervened = intervened or shield_intervened
                    if not intervened and turn_escape_remaining > 0:
                        if forward_warning:
                            turn_escape_remaining = 0
                        else:
                            action_name = "forward"
                            intervened = True
                            turn_escape_remaining -= 1
                            turn_escape_interventions += 1
                    prospective_turn_streak = (
                        turn_streak + 1
                        if action_name in {"turn left", "turn right"}
                        else 0
                    )
                    fallback_stage_turn_escape_active = (
                        second_fallback_active
                        if args.fallback_turn_final_stage_only
                        else (fallback_active or second_fallback_active)
                    )
                    fallback_turn_escape = (
                        fallback_stage_turn_escape_active
                        and args.fallback_turn_streak > 0
                    )
                    active_turn_streak = (
                        args.fallback_turn_streak
                        if fallback_turn_escape
                        else args.shield_turn_streak
                    )
                    active_turn_escape_steps = (
                        args.fallback_turn_escape_steps
                        if fallback_turn_escape
                        else args.shield_turn_escape_steps
                    )
                    if (
                        not intervened
                        and active_turn_streak > 0
                        and prospective_turn_streak >= active_turn_streak
                        and not forward_warning
                    ):
                        action_name = "forward"
                        intervened = True
                        turn_escape_remaining = max(
                            active_turn_escape_steps - 1,
                            0,
                        )
                        turn_escape_interventions += 1
                    turn_streak = (
                        turn_streak + 1
                        if action_name in {"turn left", "turn right"}
                        else 0
                    )
                if fallback_controller is not None:
                    # Both recurrent policies advance every step so the
                    # fallback is ready at the actual visited state. Correct
                    # each newest action token to what Unity really received.
                    if action_name != primary_action:
                        controller.observe_executed_action(action_name)
                    if action_name != fallback_action:
                        fallback_controller.observe_executed_action(action_name)
                elif intervened:
                    # Keep recurrent/action-token history aligned to the
                    # action Unity actually receives, not the overridden
                    # policy proposal.
                    controller.observe_executed_action(action_name)
                if second_fallback_controller is not None:
                    if action_name != second_fallback_action:
                        second_fallback_controller.observe_executed_action(
                            action_name
                        )
                executed_action_counts[action_name] += 1
                if recorder is not None:
                    recorder.record(action_name)
                observation, _, done, info = env.step(
                    PPO_ACTIONS.index(action_name)
                )
                if should_abandon_fallback(
                    fallback_active=second_fallback_active,
                    turn_escape_interventions=turn_escape_interventions,
                    activation_turn_escape_interventions=(
                        second_fallback_activation_turn_escape_interventions
                    ),
                    threshold=args.fallback_abandon_turn_escapes,
                ):
                    second_fallback_active = False
                    second_fallback_disabled = True
                    second_fallback_abandon_step = int(env.episode_steps)
                    second_fallback_abandon_distance_m = float(env.distance_m)
                    turn_escape_remaining = 0
                    turn_streak = 0
                    # The preceding fallback regains control here. Do not
                    # charge it for turn-escape interventions accumulated by
                    # the final stage that just relinquished control.
                    fallback_activation_turn_escape_interventions = (
                        turn_escape_interventions
                    )
                elif should_abandon_fallback(
                    fallback_active=fallback_active,
                    turn_escape_interventions=turn_escape_interventions,
                    activation_turn_escape_interventions=(
                        fallback_activation_turn_escape_interventions
                    ),
                    threshold=args.fallback_abandon_turn_escapes,
                ):
                    fallback_active = False
                    fallback_disabled = True
                    fallback_abandon_step = int(env.episode_steps)
                    fallback_abandon_distance_m = float(env.distance_m)
                    turn_escape_remaining = 0
                    turn_streak = 0
                if should_activate_fallback(
                    fallback_available=(
                        fallback_controller is not None
                        and not fallback_disabled
                    ),
                    fallback_active=fallback_active,
                    collision_steps=int(env.episode_collisions),
                    collision_threshold=args.fallback_collision_steps,
                    stagnation_steps=int(env.steps_without_progress),
                    stagnation_threshold=args.fallback_stagnation_steps,
                    collision_stagnation_threshold=(
                        args.fallback_collision_stagnation_steps
                    ),
                ):
                    fallback_active = True
                    fallback_ever_activated = True
                    fallback_activation_turn_escape_interventions = (
                        turn_escape_interventions
                    )
                    fallback_trigger_step = int(env.episode_steps)
                    fallback_trigger_distance_m = float(env.distance_m)
                    fallback_trigger_stagnation_steps = int(
                        env.steps_without_progress
                    )
                if should_activate_fallback(
                    fallback_available=(
                        second_fallback_controller is not None
                        and not second_fallback_disabled
                    ),
                    fallback_active=second_fallback_active,
                    collision_steps=int(env.episode_collisions),
                    collision_threshold=args.second_fallback_collision_steps,
                    stagnation_steps=int(env.steps_without_progress),
                    stagnation_threshold=(
                        args.second_fallback_stagnation_steps
                    ),
                    collision_stagnation_threshold=(
                        args.second_fallback_collision_stagnation_steps
                    ),
                    distance_m=float(env.distance_m),
                    min_distance_m=args.second_fallback_min_distance_m,
                    min_step_fraction=(
                        args.second_fallback_min_step_fraction
                    ),
                    far_distance_m=args.second_fallback_far_distance_m,
                    episode_steps=int(env.episode_steps),
                    episode_max_steps=int(env.episode_max_steps),
                    close_recovery_min_distance_m=(
                        args.second_fallback_close_recovery_min_distance_m
                    ),
                    close_recovery_max_distance_m=(
                        args.second_fallback_close_recovery_max_distance_m
                    ),
                    close_recovery_min_step_fraction=(
                        args.second_fallback_close_recovery_min_step_fraction
                    ),
                ):
                    second_fallback_active = True
                    second_fallback_ever_activated = True
                    second_fallback_activation_turn_escape_interventions = (
                        turn_escape_interventions
                    )
                    second_fallback_trigger_step = int(env.episode_steps)
                    second_fallback_trigger_distance_m = float(env.distance_m)
                    second_fallback_trigger_step_fraction = float(
                        env.episode_steps / max(env.episode_max_steps, 1)
                    )
                collision_streak = collision_streak + 1 if info["collision"] else 0
                if (
                    shield is not None
                    and collision_streak >= args.shield_collision_streak
                ):
                    shield.begin_collision_recovery(
                        decode_depth_observation_meters(env.depth_obs),
                        goal_bearing_sin=float(observation["goal"][1]),
                        move_command=7.5,
                        turns=args.shield_recovery_turns,
                        escape_forwards=args.shield_recovery_forward_steps,
                    )
                    collision_streak = 0
                if done:
                    break
            summary = {
                "scene_name": str(task["scene_name"]),
                "episode_id": str(task["episode_id"]),
                "dynamic_objects": args.dynamic_objects or "moving",
                "returncode": 0,
                "success": bool(info["success"]),
                "stop_reason": (
                    "reached_vicinity" if info["success"] else "max_steps"
                ),
                "distance_world": float(info["distance_m"]),
                "steps_taken": int(info["episode_steps"]),
                "step_budget": int(info["episode_max_steps"]),
                "collision_steps": int(info["episode_collisions"]),
                "collision_events": int(info["episode_collision_events"]),
                "warning_steps": int(info["episode_warnings"]),
                "safety_shield_interventions": (
                    shield.interventions
                    + turn_escape_interventions
                    + terminal_homing_interventions
                    if shield is not None
                    else 0
                ),
                "predicted_action_counts": dict(predicted_action_counts),
                "executed_action_counts": dict(executed_action_counts),
                "proactive_warning_recoveries": proactive_warning_recoveries,
                "turn_escape_interventions": turn_escape_interventions,
                "terminal_homing_interventions": (
                    terminal_homing_interventions
                ),
                "fallback_activated": fallback_ever_activated,
                "fallback_active_at_end": fallback_active,
                "fallback_abandon_step": fallback_abandon_step,
                "fallback_abandon_distance_m": fallback_abandon_distance_m,
                "fallback_trigger_step": fallback_trigger_step,
                "fallback_trigger_distance_m": fallback_trigger_distance_m,
                "fallback_trigger_stagnation_steps": (
                    fallback_trigger_stagnation_steps
                ),
                "fallback_action_counts": dict(fallback_action_counts),
                "second_fallback_activated": second_fallback_ever_activated,
                "second_fallback_active_at_end": second_fallback_active,
                "second_fallback_abandon_step": second_fallback_abandon_step,
                "second_fallback_abandon_distance_m": (
                    second_fallback_abandon_distance_m
                ),
                "second_fallback_trigger_step": second_fallback_trigger_step,
                "second_fallback_trigger_distance_m": (
                    second_fallback_trigger_distance_m
                ),
                "second_fallback_trigger_step_fraction": (
                    second_fallback_trigger_step_fraction
                ),
                "second_fallback_action_counts": dict(
                    second_fallback_action_counts
                ),
            }
            if dynamic_budget is not None:
                summary["reset_dynamic_step_budget"] = dynamic_budget
                summary["step_budget_reference_sha256"] = args._budget_reference_sha256
            summaries.append(summary)
            if recorder is not None:
                recorder.finish(summary)
                recorder = None
            with worker_result_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, sort_keys=True) + "\n")
            print(json.dumps(summary), flush=True)
            if task_index + 1 < len(pending_tasks):
                observation = env.reset()
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    return summaries


def evaluate_persistent(args: argparse.Namespace, tasks: list[dict]) -> list[dict]:
    shards = shard_tasks_by_scene(tasks, args.workers)
    summaries: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = {
            executor.submit(evaluate_persistent_worker, args, shard, index): index
            for index, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            summaries.extend(future.result())
    order = {
        (str(task["scene_name"]), str(task["episode_id"])): index
        for index, task in enumerate(tasks)
    }
    summaries.sort(
        key=lambda row: order[(row["scene_name"], row["episode_id"])]
    )
    return summaries


def main() -> None:
    args = parse_args()
    if bool(args.step_budget_reference) != bool(args.step_budget_reference_sha256):
        raise SystemExit("Both step-budget reference path and SHA256 are required")
    if args.step_budget_reference is not None and not args.persistent_policy:
        raise SystemExit("A step-budget reference requires --persistent-policy")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.save_visuals and not args.persistent_policy:
        raise SystemExit("--save-visuals requires --persistent-policy")
    if args.shield_collision_streak <= 0:
        raise SystemExit("--shield-collision-streak must be positive")
    if args.shield_recovery_turns <= 0:
        raise SystemExit("--shield-recovery-turns must be positive")
    if args.shield_recovery_forward_steps < 0:
        raise SystemExit("--shield-recovery-forward-steps must be nonnegative")
    if args.shield_max_clearance_turns < -1:
        raise SystemExit("--shield-max-clearance-turns must be -1 or nonnegative")
    if args.shield_warning_streak < 0:
        raise SystemExit("--shield-warning-streak must be nonnegative")
    if args.shield_warning_turns <= 0:
        raise SystemExit("--shield-warning-turns must be positive")
    if args.shield_turn_streak < 0:
        raise SystemExit("--shield-turn-streak must be nonnegative")
    if args.shield_turn_escape_steps <= 0:
        raise SystemExit("--shield-turn-escape-steps must be positive")
    if args.fallback_turn_streak < 0:
        raise SystemExit("--fallback-turn-streak must be nonnegative")
    if args.fallback_turn_escape_steps <= 0:
        raise SystemExit("--fallback-turn-escape-steps must be positive")
    if args.fallback_abandon_turn_escapes < 0:
        raise SystemExit("--fallback-abandon-turn-escapes must be nonnegative")
    if args.shield_terminal_homing_distance_m < 0.0:
        raise SystemExit(
            "--shield-terminal-homing-distance-m must be nonnegative"
        )
    if not 0.0 <= args.shield_terminal_homing_tolerance_deg <= 90.0:
        raise SystemExit(
            "--shield-terminal-homing-tolerance-deg must be in [0, 90]"
        )
    if args.shield_terminal_homing_distance_m > 0.0 and not args.safety_shield:
        raise SystemExit("terminal homing requires --safety-shield")
    if args.waypoint_terminal_distance_m < -1.0:
        raise SystemExit("--waypoint-terminal-distance-m must be -1 or nonnegative")
    if args.route_graph_lookahead_m < -1.0:
        raise SystemExit("--route-graph-lookahead-m must be -1 or positive")
    if args.route_graph_lookahead_m == 0.0:
        raise SystemExit("--route-graph-lookahead-m must be -1 or positive")
    if args.route_graph_terminal_distance_m < -1.0:
        raise SystemExit(
            "--route-graph-terminal-distance-m must be -1 or nonnegative"
        )
    if args.route_graph_max_snap_distance_m < -1.0:
        raise SystemExit(
            "--route-graph-max-snap-distance-m must be -1 or positive"
        )
    if args.route_graph_max_snap_distance_m == 0.0:
        raise SystemExit(
            "--route-graph-max-snap-distance-m must be -1 or positive"
        )
    if args.route_graph_activation_collision_steps < 0:
        raise SystemExit(
            "--route-graph-activation-collision-steps must be nonnegative"
        )
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.fallback_collision_steps < 0:
        raise SystemExit("--fallback-collision-steps must be nonnegative")
    if args.fallback_stagnation_steps < 0:
        raise SystemExit("--fallback-stagnation-steps must be nonnegative")
    if args.fallback_collision_stagnation_steps < 0:
        raise SystemExit(
            "--fallback-collision-stagnation-steps must be nonnegative"
        )
    if args.second_fallback_collision_steps < 0:
        raise SystemExit(
            "--second-fallback-collision-steps must be nonnegative"
        )
    if args.second_fallback_stagnation_steps < 0:
        raise SystemExit(
            "--second-fallback-stagnation-steps must be nonnegative"
        )
    if args.second_fallback_collision_stagnation_steps < 0:
        raise SystemExit(
            "--second-fallback-collision-stagnation-steps must be nonnegative"
        )
    if args.second_fallback_min_distance_m < 0.0:
        raise SystemExit("--second-fallback-min-distance-m must be nonnegative")
    if not 0.0 <= args.second_fallback_min_step_fraction <= 1.0:
        raise SystemExit(
            "--second-fallback-min-step-fraction must be in [0, 1]"
        )
    if args.second_fallback_far_distance_m < 0.0:
        raise SystemExit("--second-fallback-far-distance-m must be nonnegative")
    close_min = args.second_fallback_close_recovery_min_distance_m
    close_max = args.second_fallback_close_recovery_max_distance_m
    if close_min < 0.0 or close_max < 0.0:
        raise SystemExit("second-fallback close-recovery distances must be nonnegative")
    if close_max > 0.0 and close_min > close_max:
        raise SystemExit(
            "second-fallback close-recovery minimum distance must not exceed maximum"
        )
    if not 0.0 <= args.second_fallback_close_recovery_min_step_fraction <= 1.0:
        raise SystemExit(
            "--second-fallback-close-recovery-min-step-fraction must be in [0, 1]"
        )
    fallback_checkpoints = [
        checkpoint
        for checkpoint in (
            args.fallback_ppo_checkpoint,
            args.fallback_bc_checkpoint,
        )
        if checkpoint is not None
    ]
    if len(fallback_checkpoints) > 1:
        raise SystemExit(
            "--fallback-ppo-checkpoint and --fallback-bc-checkpoint are "
            "mutually exclusive"
        )
    if fallback_checkpoints:
        if args.baseline != "ppo" or not args.persistent_policy:
            raise SystemExit(
                "fallback policy requires --baseline ppo and --persistent-policy"
            )
        if (
            args.fallback_collision_steps <= 0
            and args.fallback_stagnation_steps <= 0
        ):
            raise SystemExit(
                "fallback checkpoint requires a positive collision or "
                "stagnation trigger"
            )
        fallback_checkpoint = fallback_checkpoints[0]
        if not fallback_checkpoint.is_file():
            raise SystemExit(
                f"Fallback checkpoint not found: {fallback_checkpoint}"
            )
    elif (
        args.fallback_collision_steps > 0
        or args.fallback_stagnation_steps > 0
    ):
        raise SystemExit(
            "fallback triggers require a PPO or BC fallback checkpoint"
        )
    if args.second_fallback_bc_checkpoint is not None:
        if (
            args.baseline != "ppo"
            or not args.persistent_policy
            or args.fallback_ppo_checkpoint is None
        ):
            raise SystemExit(
                "second BC fallback requires a persistent PPO baseline and "
                "--fallback-ppo-checkpoint"
            )
        if (
            args.second_fallback_collision_steps <= 0
            and args.second_fallback_stagnation_steps <= 0
        ):
            raise SystemExit(
                "--second-fallback-bc-checkpoint requires a positive "
                "collision or stagnation trigger"
            )
        if not args.second_fallback_bc_checkpoint.is_file():
            raise SystemExit(
                "Second fallback checkpoint not found: "
                f"{args.second_fallback_bc_checkpoint}"
            )
        if (
            args.second_fallback_collision_steps > 0
            and args.fallback_collision_steps > 0
            and args.second_fallback_collision_steps
            <= args.fallback_collision_steps
        ):
            raise SystemExit(
                "second fallback collision threshold must exceed the first"
            )
    elif (
        args.second_fallback_collision_steps > 0
        or args.second_fallback_stagnation_steps > 0
        or args.second_fallback_collision_stagnation_steps > 0
    ):
        raise SystemExit(
            "second fallback triggers require "
            "--second-fallback-bc-checkpoint"
        )
    if args.waypoint_planner_checkpoint is not None:
        if args.baseline != "ppo" or not args.persistent_policy:
            raise SystemExit(
                "waypoint planner requires --baseline ppo and --persistent-policy"
            )
        if not args.waypoint_planner_checkpoint.is_file():
            raise SystemExit(
                "Waypoint planner checkpoint not found: "
                f"{args.waypoint_planner_checkpoint}"
            )
    if args.route_graph_checkpoint is not None:
        if args.baseline != "ppo" or not args.persistent_policy:
            raise SystemExit(
                "route graph requires --baseline ppo and --persistent-policy"
            )
        if args.waypoint_planner_checkpoint is not None:
            raise SystemExit(
                "route graph and learned waypoint planner are mutually exclusive"
            )
        if not args.route_graph_checkpoint.is_file():
            raise SystemExit(
                f"Route graph checkpoint not found: {args.route_graph_checkpoint}"
            )
    records = (
        tasks_from_input_points(
            args.input_points,
            eval_seed=args.eval_seed,
            dynamic_objects=args.dynamic_objects or "moving",
        )
        if args.input_points is not None
        else read_jsonl(args.manifest)
    )
    if args.dynamic_objects is not None:
        records = [dict(task, dynamic_objects=args.dynamic_objects) for task in records]
    eligible_records = (
        read_jsonl(args.eligible_dataset_manifest)
        if args.eligible_dataset_manifest is not None
        else None
    )
    selected = select_held_out_tasks(
        records,
        args.scenes,
        eligible_records,
        args.episodes_per_scene,
    )
    expected_episodes = args.scenes * args.episodes_per_scene
    if len(selected) < expected_episodes:
        raise SystemExit(
            f"Only found {len(selected)} held-out episodes; expected "
            f"{expected_episodes} across {args.scenes} scenes"
        )
    if args.step_budget_reference is not None:
        args._reference_budgets, args._budget_reference_sha256 = load_reference_budgets(
            args.step_budget_reference, args.step_budget_reference_sha256,
            [(task["scene_name"], task["episode_id"]) for task in selected],
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.persistent_policy:
        summaries = evaluate_persistent(args, selected)
    else:
        payloads = [(args, task, index) for index, task in enumerate(selected)]
        if args.workers == 1:
            summaries = [evaluate_task(*payload) for payload in payloads]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                summaries = list(
                    executor.map(lambda payload: evaluate_task(*payload), payloads)
                )
    for summary in summaries:
        print(json.dumps(summary), flush=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "fallback_ppo_checkpoint": (
            str(args.fallback_ppo_checkpoint)
            if args.fallback_ppo_checkpoint is not None
            else None
        ),
        "fallback_bc_checkpoint": (
            str(args.fallback_bc_checkpoint)
            if args.fallback_bc_checkpoint is not None
            else None
        ),
        "fallback_collision_steps": int(args.fallback_collision_steps),
        "fallback_stagnation_steps": int(args.fallback_stagnation_steps),
        "fallback_collision_stagnation_steps": int(
            args.fallback_collision_stagnation_steps
        ),
        "second_fallback_bc_checkpoint": (
            str(args.second_fallback_bc_checkpoint)
            if args.second_fallback_bc_checkpoint is not None
            else None
        ),
        "second_fallback_collision_steps": int(
            args.second_fallback_collision_steps
        ),
        "second_fallback_stagnation_steps": int(
            args.second_fallback_stagnation_steps
        ),
        "second_fallback_collision_stagnation_steps": int(
            args.second_fallback_collision_stagnation_steps
        ),
        "second_fallback_min_distance_m": float(
            args.second_fallback_min_distance_m
        ),
        "second_fallback_min_step_fraction": float(
            args.second_fallback_min_step_fraction
        ),
        "second_fallback_far_distance_m": float(
            args.second_fallback_far_distance_m
        ),
        "second_fallback_close_recovery_min_distance_m": float(
            args.second_fallback_close_recovery_min_distance_m
        ),
        "second_fallback_close_recovery_max_distance_m": float(
            args.second_fallback_close_recovery_max_distance_m
        ),
        "second_fallback_close_recovery_min_step_fraction": float(
            args.second_fallback_close_recovery_min_step_fraction
        ),
        "waypoint_planner_checkpoint": (
            str(args.waypoint_planner_checkpoint)
            if args.waypoint_planner_checkpoint is not None
            else None
        ),
        "waypoint_terminal_distance_m": float(
            args.waypoint_terminal_distance_m
        ),
        "route_graph_checkpoint": (
            str(args.route_graph_checkpoint)
            if args.route_graph_checkpoint is not None
            else None
        ),
        "route_graph_lookahead_m": float(args.route_graph_lookahead_m),
        "route_graph_terminal_distance_m": float(
            args.route_graph_terminal_distance_m
        ),
        "route_graph_max_snap_distance_m": float(
            args.route_graph_max_snap_distance_m
        ),
        "route_graph_activation_collision_steps": int(
            args.route_graph_activation_collision_steps
        ),
        "input_points": (
            str(args.input_points) if args.input_points is not None else None
        ),
        "eval_seed": args.eval_seed,
        "dynamic_objects": args.dynamic_objects or "moving",
        "reach_m": 2.0,
        "persistent_policy": bool(args.persistent_policy),
        "safety_shield": bool(args.safety_shield),
        "shield_collision_streak": int(args.shield_collision_streak),
        "shield_recovery_turns": int(args.shield_recovery_turns),
        "shield_recovery_forward_steps": int(
            args.shield_recovery_forward_steps
        ),
        "shield_max_clearance_turns": int(args.shield_max_clearance_turns),
        "shield_warning_streak": int(args.shield_warning_streak),
        "shield_warning_turns": int(args.shield_warning_turns),
        "shield_turn_streak": int(args.shield_turn_streak),
        "shield_turn_escape_steps": int(args.shield_turn_escape_steps),
        "fallback_turn_streak": int(args.fallback_turn_streak),
        "fallback_turn_escape_steps": int(args.fallback_turn_escape_steps),
        "fallback_turn_final_stage_only": bool(
            args.fallback_turn_final_stage_only
        ),
        "fallback_abandon_turn_escapes": int(
            args.fallback_abandon_turn_escapes
        ),
        "shield_terminal_homing_distance_m": float(
            args.shield_terminal_homing_distance_m
        ),
        "shield_terminal_homing_tolerance_deg": float(
            args.shield_terminal_homing_tolerance_deg
        ),
        "episodes": summaries,
        "successes": sum(item["success"] for item in summaries),
        "success_rate": sum(item["success"] for item in summaries) / len(summaries),
    }
    if args.step_budget_reference is not None:
        report["step_budget_reference"] = str(args.step_budget_reference)
        report["step_budget_reference_sha256"] = args._budget_reference_sha256
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
