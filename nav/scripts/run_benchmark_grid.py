"""Grid orchestrator for the unified-client benchmark cell runner.

Walks a (model x scene x point x seed x vision_modality x history_size) grid and
dispatches one ``nav.scripts.run_benchmark_cell`` invocation per cell into a
bounded ProcessPoolExecutor:

  python -m nav.scripts.run_benchmark_grid \
      --models anthropic/claude-sonnet-4.6 google/gemini-3-flash-preview \
      --scenes scene1 scene2 ... \
      --seeds 0 1 2 \
      --vision_input both \
      --history_sizes 0 5 10 \
      --max_concurrency 4 \
      --max_steps 70

Design choices:
  * One subprocess per cell. Crash isolation is the whole point — a bad LLM call
    or a Unity SIGSEGV in one cell never blocks the others, the executor keeps
    draining the queue.
  * Each cell gets its own xvfb-run display (Linux) and a fresh free TCP base
    port (acquired by the worker just before launching mlagents, which avoids
    the pre-allocate-vs-actually-bind race).
  * Default-history cells live at outputs/<scene>/<point>/<model_short>[_novision]/seed<k>/
    so eval_metrics.py + the stats loader keep working. Non-default history sizes
    are routed under outputs/_history_size/hs<k>/<...> (the loader skips
    ``_``-prefixed roots) so the history sweep never pollutes the main leaderboard.
  * Aggregates land under analysis/grid_runs/<timestamp>/{runs,failures}.csv;
    outputs/ stays per-cell telemetry only.
  * --resume skips cells whose results.csv already has a row, so re-launching
    after a partial completion is safe.
  * --skip_existing_dirs skips every cell whose output directory already
    exists, including interrupted/partial runs.
  * --max_retries N retries a failed cell up to N times before recording a
    permanent failure in failures.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from nav.config import (
    ANALYSIS_ROOT,
    ASTAR_DEFAULTS,
    DEFAULT_REACH_DISTANCE_M,
    DEFAULT_PROMPT_NOVISION,
    DEFAULT_PROMPT_VISION,
    GRID_CSV_FIELDS,
    LLM_DEFAULT_HISTORY_SIZE,
    LLM_DEFAULT_MAX_TOKENS,
    SCENE_ID_MAP,
    SCENE_CODES,
)
from nav.harness.navigation_protocol import (
    NAVIGATION_PROTOCOL_VERSION, check_navigation_run_config, navigation_run_config,
    resolve_navigation_sensors,
)
from nav.utils import load_prompt_template
from nav.harness.lighting import (
    add_lighting_args,
    lighting_result_fields,
    resolve_lighting_config,
)
from nav.harness.motion import (
    MOTION_CATEGORIES,
    add_motion_speed_args,
    motion_speed_result_fields,
    resolve_motion_speed_config,
)


# nav/scripts/run_benchmark_grid.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_POINTS = REPO_ROOT / "input_points.json"
#: Subtree for non-default history-size sweeps (skipped by the stats loader,
#: which ignores ``_``-prefixed roots under outputs/).
HISTORY_SWEEP_ROOT = REPO_ROOT / "outputs" / "_history_size"


@dataclass(frozen=True)
class Cell:
    model: str          # e.g. "anthropic/claude-sonnet-4.6"
    scene_name: str     # e.g. "scene1"
    point_id: str       # e.g. "point1"
    seed_id: str        # e.g. "0"
    vision_input: bool  # True | False
    history_size: int   # LLM prompt history depth
    # Spawn is in Unity world coords (input_points.json `start.{x,z}`).
    init_world_x: float
    init_world_z: float
    init_direction: float
    target_x: int
    target_y: int
    output_root: Optional[str] = None

    @property
    def model_short(self) -> str:
        # "anthropic/claude-sonnet-4.6" -> "claude-sonnet-4.6"
        return self.model.split("/")[-1]

    @property
    def model_dir(self) -> str:
        return self.model_short + ("" if self.vision_input else "_novision")

    @property
    def frame_save_dir(self) -> Path:
        # Default-history runs use the canonical tree the stats loader reads;
        # other sizes go under outputs/_history_size/hs<k>/ to stay out of it.
        base = Path(self.output_root) if self.output_root else REPO_ROOT / "outputs"
        if self.history_size != LLM_DEFAULT_HISTORY_SIZE:
            base = base / "_history_size" / f"hs{self.history_size}"
        return base / self.scene_name / self.point_id / self.model_dir / f"seed{self.seed_id}"

    @property
    def results_csv(self) -> Path:
        return self.frame_save_dir / "results.csv"

    @property
    def label(self) -> str:
        v = "v" if self.vision_input else "x"
        return f"{self.scene_name}/{self.point_id}/{self.model_short}/seed{self.seed_id}/{v}/hs{self.history_size}"


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def load_input_points() -> dict:
    return json.loads(INPUT_POINTS.read_text(encoding="utf-8"))


def build_grid(
    models: List[str],
    scenes: List[str],
    seeds: List[str],
    vision_modes: List[bool],
    history_sizes: List[int],
    points: Optional[List[str]] = None,
    output_root: Optional[str] = None,
) -> List[Cell]:
    pts = load_input_points()
    unknown_scenes = [scene for scene in scenes if scene not in SCENE_ID_MAP]
    if unknown_scenes:
        raise SystemExit(
            f"Unknown scene code(s): {unknown_scenes}. "
            f"Expected one of: {list(SCENE_ID_MAP)}"
        )
    cells: List[Cell] = []
    for scene in scenes:
        if scene not in pts:
            raise SystemExit(f"Scene '{scene}' not in input_points.json keys: {list(pts)}")
        for entry in pts[scene]:
            pid = entry["point_id"]
            if points and pid not in points:
                continue
            init_world_x = float(entry["start"]["x"])
            init_world_z = float(entry["start"]["z"])
            init_dir = float(entry["start"]["direction"])
            tx = int(entry["target"]["x"])
            ty = int(entry["target"]["y"])
            for model in models:
                for seed in seeds:
                    for vis in vision_modes:
                        for hs in history_sizes:
                            cells.append(Cell(
                                model=model,
                                scene_name=scene,
                                point_id=pid,
                                seed_id=str(seed),
                                vision_input=vis,
                                history_size=int(hs),
                                init_world_x=init_world_x,
                                init_world_z=init_world_z,
                                init_direction=init_dir,
                                target_x=tx, target_y=ty,
                                output_root=output_root,
                            ))
    return cells


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------
def free_tcp_port() -> int:
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def cell_run_config(args, cell: Cell, file_name: str) -> dict:
    settings = argparse.Namespace(**{
        **vars(args),
        "model_id": cell.model, "scene_id": SCENE_ID_MAP[cell.scene_name],
        "scene_name": cell.scene_name, "point_id": cell.point_id,
        "seed_id": cell.seed_id, "vision_input": cell.vision_input,
        "history_size": cell.history_size, "init_world_x": cell.init_world_x,
        "init_world_z": cell.init_world_z, "init_curr_direction": cell.init_direction,
        "target_x": cell.target_x, "target_y": cell.target_y, "file_name": file_name,
    })
    prompt = DEFAULT_PROMPT_VISION if cell.vision_input else DEFAULT_PROMPT_NOVISION
    return navigation_run_config(settings, load_prompt_template(prompt))


def cell_completed(cell: Cell) -> bool:
    if not cell.results_csv.exists():
        return False
    with cell.results_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return bool(rows) and rows[-1].get("stop_reason") in {"max_steps", "reached_vicinity"}


def run_cell(args_dict: dict) -> dict:
    """Worker function. Runs in a child process; returns a status dict."""
    cell = Cell(**args_dict["cell"])
    file_name = args_dict["file_name"]
    python_bin = args_dict["python_bin"]
    llm_provider = args_dict["llm_provider"]
    llm_min_request_interval_sec = args_dict["llm_min_request_interval_sec"]
    max_tokens = args_dict["max_tokens"]
    max_steps = args_dict["max_steps"]
    dynamic_step_budget = args_dict["dynamic_step_budget"]
    step_budget_min = args_dict["step_budget_min"]
    step_budget_max = args_dict["step_budget_max"]
    steps_per_path_meter = args_dict["steps_per_path_meter"]
    step_budget_overhead = args_dict["step_budget_overhead"]
    reach_m = args_dict["reach_m"]
    use_xvfb = args_dict["use_xvfb"]
    xvfb_screen = args_dict["xvfb_screen"]
    ego_width = args_dict["ego_width"]
    ego_height = args_dict["ego_height"]
    minimap_width = args_dict["minimap_width"]
    minimap_height = args_dict["minimap_height"]
    dynamic_objects = args_dict["dynamic_objects"]
    motion_args = argparse.Namespace(
        scene_id=SCENE_ID_MAP[cell.scene_name],
        scene_name=cell.scene_name,
        point_id=cell.point_id,
        seed_id=cell.seed_id,
        motion_random_seed=args_dict["motion_random_seed"],
        **{
            f"{category}_speed_{suffix}": args_dict[
                f"{category}_speed_{suffix}"
            ]
            for category in MOTION_CATEGORIES
            for suffix in ("mps", "min_mps", "max_mps")
        },
    )
    motion = resolve_motion_speed_config(motion_args)
    motion_fields = motion_speed_result_fields(motion_args)
    lighting_args = argparse.Namespace(
        scene_id=SCENE_ID_MAP[cell.scene_name],
        scene_name=cell.scene_name,
        point_id=cell.point_id,
        seed_id=cell.seed_id,
        global_light_intensity=args_dict["global_light_intensity"],
        light_intensity_multiplier=args_dict["light_intensity_multiplier"],
        light_intensity_min=args_dict["light_intensity_min"],
        light_intensity_max=args_dict["light_intensity_max"],
        light_random_seed=args_dict["light_random_seed"],
        light_fixed_exposure=args_dict["light_fixed_exposure"],
    )
    lighting = resolve_lighting_config(lighting_args)
    lighting_fields = lighting_result_fields(lighting_args)

    check_navigation_run_config(
        cell.frame_save_dir, cell_run_config(argparse.Namespace(**args_dict), cell, file_name),
    )
    cell.frame_save_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell.frame_save_dir / "run.log"

    prompt_file = DEFAULT_PROMPT_VISION if cell.vision_input else DEFAULT_PROMPT_NOVISION
    base_port = free_tcp_port()

    cmd: List[str] = []
    if use_xvfb:
        cmd += ["xvfb-run", "-a", "-s", f"-screen 0 {xvfb_screen}"]
    cmd += [
        python_bin, "-m", "nav.scripts.run_benchmark_cell",
        "--baseline", "llm",
        "--file_name", file_name,
        "--scene_id", str(SCENE_ID_MAP[cell.scene_name]),
        "--scene_name", cell.scene_name,
        "--point_id", cell.point_id,
        "--seed_id", cell.seed_id,
        "--vision_input", "true" if cell.vision_input else "false",
        "--history_size", str(cell.history_size),
        "--worker_id", "0",
        "--base_port", str(base_port),
        "--max_steps", str(max_steps),
        (
            "--dynamic_step_budget"
            if dynamic_step_budget
            else "--no-dynamic_step_budget"
        ),
        "--step_budget_min", str(step_budget_min),
        "--step_budget_max", str(step_budget_max),
        "--steps_per_path_meter", str(steps_per_path_meter),
        "--step_budget_overhead", str(step_budget_overhead),
        "--reach_m", str(reach_m),
        "--ego_width", str(ego_width),
        "--ego_height", str(ego_height),
        "--minimap_width", str(minimap_width),
        "--minimap_height", str(minimap_height),
        "--dynamic_objects", dynamic_objects,
        "--frame_save_dir", str(cell.frame_save_dir),
        "--prompt_file", prompt_file,
        "--model_id", cell.model,
        "--llm_provider", llm_provider,
        "--llm_min_request_interval_sec", str(llm_min_request_interval_sec),
        "--max_tokens", str(max_tokens),
        "--init_world_x", str(cell.init_world_x),
        "--init_world_z", str(cell.init_world_z),
        "--init_curr_direction", str(cell.init_direction),
        "--target_x", str(cell.target_x),
        "--target_y", str(cell.target_y),
    ]
    for category in MOTION_CATEGORIES:
        fixed = args_dict[f"{category}_speed_mps"]
        minimum = args_dict[f"{category}_speed_min_mps"]
        maximum = args_dict[f"{category}_speed_max_mps"]
        if fixed is not None:
            cmd += [f"--{category}_speed_mps", str(fixed)]
        elif minimum is not None:
            cmd += [
                f"--{category}_speed_min_mps", str(minimum),
                f"--{category}_speed_max_mps", str(maximum),
            ]
    cmd += ["--motion_random_seed", str(args_dict["motion_random_seed"])]
    if args_dict["global_light_intensity"] is not None:
        cmd += ["--global_light_intensity", str(args_dict["global_light_intensity"])]
    if args_dict["light_intensity_multiplier"] is not None:
        cmd += [
            "--light_intensity_multiplier",
            str(args_dict["light_intensity_multiplier"]),
        ]
    if lighting.enabled and lighting.mode == "range":
        cmd += [
            "--light_intensity_min", str(args_dict["light_intensity_min"]),
            "--light_intensity_max", str(args_dict["light_intensity_max"]),
        ]
    cmd += [
        "--light_random_seed", str(args_dict["light_random_seed"]),
        "--light_fixed_exposure", str(args_dict["light_fixed_exposure"]),
    ]

    started = time.time()
    try:
        with open(log_path, "w") as logf:
            logf.write("# command: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                check=False,
                timeout=args_dict.get("per_cell_timeout_sec"),
            )
        duration = time.time() - started
        ok = (proc.returncode == 0) and cell_completed(cell)
        return {
            "label": cell.label,
            "ok": ok,
            "returncode": proc.returncode,
            "duration_sec": duration,
            "frame_save_dir": str(cell.frame_save_dir),
            "log_path": str(log_path),
            "cell": asdict(cell),
            "dynamic_objects": dynamic_objects,
            "motion": motion_fields,
            "lighting": lighting_fields,
            "error": "" if ok else "cell did not complete a valid navigation episode",
        }
    except subprocess.TimeoutExpired:
        return {
            "label": cell.label,
            "ok": False,
            "returncode": -999,
            "duration_sec": time.time() - started,
            "frame_save_dir": str(cell.frame_save_dir),
            "log_path": str(log_path),
            "cell": asdict(cell),
            "dynamic_objects": dynamic_objects,
            "motion": motion_fields,
            "lighting": lighting_fields,
            "error": "timeout",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "label": cell.label,
            "ok": False,
            "returncode": -1000,
            "duration_sec": time.time() - started,
            "frame_save_dir": str(cell.frame_save_dir),
            "log_path": str(log_path),
            "cell": asdict(cell),
            "dynamic_objects": dynamic_objects,
            "motion": motion_fields,
            "lighting": lighting_fields,
            "error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# Aggregate CSV (fields in nav.config.GRID_CSV_FIELDS)
# ---------------------------------------------------------------------------
def append_grid_row(grid_csv: Path, status: dict):
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not grid_csv.exists()
    cell = status["cell"]
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scene_name": cell["scene_name"],
        "point_id": cell["point_id"],
        "model": cell["model"],
        "seed_id": cell["seed_id"],
        "vision_input": cell["vision_input"],
        "history_size": cell["history_size"],
        "dynamic_objects": status["dynamic_objects"],
        **status["motion"],
        **status["lighting"],
        "ok": status["ok"],
        "returncode": status["returncode"],
        "duration_sec": round(status["duration_sec"], 2),
        "frame_save_dir": status["frame_save_dir"],
        "results_csv_present": Path(status["frame_save_dir"], "results.csv").exists(),
        "log_path": status["log_path"],
    }
    with open(grid_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GRID_CSV_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def append_failure_row(failures_csv: Path, status: dict, attempts: int):
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not failures_csv.exists()
    cell = status["cell"]
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scene_name": cell["scene_name"],
        "point_id": cell["point_id"],
        "model": cell["model"],
        "seed_id": cell["seed_id"],
        "vision_input": cell["vision_input"],
        "history_size": cell["history_size"],
        "dynamic_objects": status.get("dynamic_objects", ""),
        **status.get("motion", {}),
        **status.get("lighting", {}),
        "attempts": attempts,
        "returncode": status["returncode"],
        "error": status.get("error", ""),
        "log_path": status["log_path"],
    }
    fields = list(row.keys())
    with open(failures_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Grid runner for nav.scripts.run_benchmark_cell.")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model ids to benchmark. Example: google/gemini-3.8-flash")
    p.add_argument(
        "--llm_provider",
        choices=("openrouter", "gemini", "openai"),
        default="openrouter",
        help="LLM transport for every cell in this grid.",
    )
    p.add_argument(
        "--llm_min_request_interval_sec",
        type=float,
        default=0.0,
        help="Minimum seconds between direct-provider calls within each cell process.",
    )
    p.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Scene codes to run (scene1..scene24). Defaults to all 24 scenes "
            "in config.SCENE_CODES."
        ),
    )
    p.add_argument("--points", nargs="*", default=None,
                   help="Restrict to specific point_ids (e.g. point1 point2). Default: all points per scene.")
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2"],
                   help="Run-replicate ids. Closed-source LLMs ignore these for sampling; the script just labels each replicate.")
    p.add_argument("--vision_input", choices=["on", "off", "both"], default="on",
                   help="Whether to enable vision (egocentric image). 'both' runs each cell twice.")
    p.add_argument("--history_sizes", nargs="+", type=int, default=[LLM_DEFAULT_HISTORY_SIZE],
                   help="LLM prompt history depths to sweep (6th grid axis). Default-size cells "
                        "land in the canonical outputs/ tree; other sizes under "
                        "outputs/_history_size/hs<k>/ so the stats loader isn't polluted.")
    p.add_argument("--output_root", type=str, default=str(REPO_ROOT / "outputs"),
                   help="Output root; use a fresh directory to preserve older protocol results.")
    p.add_argument("--max_concurrency", type=int, default=4,
                   help="Max parallel cells. Start at 4; bump up if you don't see OpenRouter 429s.")
    p.add_argument(
        "--max_tokens",
        type=int,
        default=LLM_DEFAULT_MAX_TOKENS,
        help="Maximum completion tokens for each LLM decision request.",
    )
    p.add_argument("--max_steps", type=int, default=70)
    p.add_argument(
        "--dynamic_step_budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize each LLM cell's step budget from its start-target world "
            "distance (default: enabled). Use --no-dynamic_step_budget with "
            "--max_steps to reproduce a fixed-budget run."
        ),
    )
    p.add_argument(
        "--step_budget_min",
        type=int,
        default=ASTAR_DEFAULTS.step_budget_min,
    )
    p.add_argument(
        "--step_budget_max",
        type=int,
        default=ASTAR_DEFAULTS.step_budget_max,
    )
    p.add_argument(
        "--steps_per_path_meter",
        type=float,
        default=ASTAR_DEFAULTS.steps_per_path_meter,
    )
    p.add_argument(
        "--step_budget_overhead",
        type=int,
        default=ASTAR_DEFAULTS.step_budget_overhead,
    )
    p.add_argument(
        "--reach_m",
        type=float,
        default=DEFAULT_REACH_DISTANCE_M,
        help=f"Success radius in Unity world meters (default: {DEFAULT_REACH_DISTANCE_M:g}).",
    )
    p.add_argument("--ego_width", type=int, default=None,
                   help="Egocentric RGB/depth width (Kiro default: 320).")
    p.add_argument("--ego_height", type=int, default=None,
                   help="Egocentric RGB/depth height (Kiro default: 240).")
    p.add_argument("--minimap_width", type=int, default=None,
                   help="Optional minimap width; derived from height when omitted.")
    p.add_argument("--minimap_height", type=int, default=None,
                   help="Optional minimap height; derived from width when omitted.")
    p.add_argument(
        "--dynamic_objects",
        choices=("moving", "static"),
        default="moving",
        help="Run environment objects normally or freeze them for every grid cell.",
    )
    add_motion_speed_args(p)
    add_lighting_args(p)
    p.add_argument("--max_retries", type=int, default=2,
                   help="Retry a failed cell this many times before logging it to failures.csv.")
    p.add_argument("--resume", action="store_true",
                   help="Skip normally completed cells with matching run_config.json.")
    p.add_argument(
        "--skip_existing_dirs",
        action="store_true",
        help="Skip cells whose output directory already exists, even without results.csv.",
    )
    p.add_argument("--grid_csv", type=str, default="",
                   help="Aggregate runs CSV. Default: analysis/grid_runs/<timestamp>/runs.csv.")
    p.add_argument("--failures_csv", type=str, default="",
                   help="Failures CSV. Default: analysis/grid_runs/<timestamp>/failures.csv.")
    p.add_argument("--per_cell_timeout_sec", type=int, default=900,
                   help="Hard timeout per cell. 15 min default. Set to 0 to disable.")
    p.add_argument("--file_name", type=str, default="auto",
                   help="Path to scene_all client. Pass 'auto' (default) or empty to "
                        "scan config.SCENE_ALL_BUILDS for the current OS and use the "
                        "first build that exists locally. Pass an explicit path to override. "
                        "Legacy SCENE_ALL_APP / SCENE_ALL_BIN env-var fallbacks still work.")
    p.add_argument("--python_bin", type=str, default=str(REPO_ROOT / ".venv" / "bin" / "python"))
    p.add_argument("--xvfb_screen", type=str, default="1724x1024x24")
    p.add_argument("--no_xvfb", action="store_true",
                   help="Skip xvfb-run wrapping (use a real DISPLAY). Default Linux uses xvfb.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the planned grid and exit without running anything.")
    return p.parse_args()


def resolve_file_name(arg_file_name: str) -> str:
    """Resolve --file_name. Precedence:
      1. Explicit non-"auto" path on the CLI wins.
      2. Legacy env vars SCENE_ALL_APP (Darwin) / SCENE_ALL_BIN (other)
         kept as a fallback for older wrappers / muscle memory.
      3. Delegate to config.resolve_scene_all_path("auto") which scans
         config.SCENE_ALL_BUILDS for the current OS.
    """
    from nav.config import resolve_scene_all_path
    if arg_file_name and arg_file_name.strip().lower() not in {"", "auto"}:
        return arg_file_name
    env_var = "SCENE_ALL_APP" if platform.system() == "Darwin" else "SCENE_ALL_BIN"
    env_val = os.environ.get(env_var, "").strip()
    if env_val and env_val.lower() != "auto":
        return env_val
    return resolve_scene_all_path("auto")


def main():
    args = parse_args()
    try:
        motion = resolve_motion_speed_config(args)
        lighting = resolve_lighting_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(size < 0 for size in args.history_sizes):
        raise SystemExit("--history_sizes must be nonnegative.")
    if args.reach_m <= 0:
        raise SystemExit("--reach_m must be positive.")
    if args.max_tokens <= 0:
        raise SystemExit("--max_tokens must be positive.")
    if args.step_budget_min <= 0:
        raise SystemExit("--step_budget_min must be positive.")
    if args.step_budget_max < args.step_budget_min:
        raise SystemExit(
            "--step_budget_max must be greater than or equal to "
            "--step_budget_min."
        )
    if args.steps_per_path_meter <= 0:
        raise SystemExit("--steps_per_path_meter must be positive.")
    if args.step_budget_overhead < 0:
        raise SystemExit("--step_budget_overhead must be nonnegative.")
    try:
        resolve_navigation_sensors(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    vision_modes = {"on": [True], "off": [False], "both": [True, False]}[args.vision_input]
    scenes = args.scenes if args.scenes is not None else list(SCENE_CODES)
    cells = build_grid(
        models=args.models,
        scenes=scenes,
        seeds=args.seeds,
        vision_modes=vision_modes,
        history_sizes=args.history_sizes,
        points=args.points,
        output_root=str(Path(args.output_root).resolve()),
    )

    # Aggregates default to a per-invocation timestamped dir under analysis/.
    run_dir = REPO_ROOT / ANALYSIS_ROOT / "grid_runs" / time.strftime("%Y%m%d_%H%M%S")
    grid_csv = Path(args.grid_csv) if args.grid_csv else run_dir / "runs.csv"
    failures_csv = Path(args.failures_csv) if args.failures_csv else run_dir / "failures.csv"

    file_name = resolve_file_name(args.file_name)
    # Validate before opening any logs or launching a worker. --dry_run remains
    # read-only; --resume must also validate so it cannot silently skip old runs.
    if not args.dry_run or args.resume:
        for cell in cells:
            try:
                check_navigation_run_config(cell.frame_save_dir, cell_run_config(args, cell, file_name))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc

    if args.skip_existing_dirs:
        before = len(cells)
        cells = [c for c in cells if not c.frame_save_dir.exists()]
        print(
            f"[skip-existing] skipping {before - len(cells)} previously started "
            f"cells (output directory present); {len(cells)} remain.",
            flush=True,
        )
    elif args.resume:
        before = len(cells)
        cells = [c for c in cells if not cell_completed(c)]
        print(f"[resume] skipping {before - len(cells)} already-complete cells "
              f"(matching config and normal stop); {len(cells)} remain.", flush=True)

    file_name = resolve_file_name(args.file_name)
    if not Path(file_name).exists():
        raise SystemExit(f"Unity client not found at: {file_name}. Set --file_name or the env var.")
    if not Path(args.python_bin).exists():
        raise SystemExit(f"Python interpreter not found at: {args.python_bin}. Run `uv sync` or pass --python_bin.")

    use_xvfb = (platform.system() == "Linux") and (not args.no_xvfb)
    if use_xvfb and not shutil.which("xvfb-run"):
        raise SystemExit("xvfb-run not found on PATH. `sudo apt install xvfb` or pass --no_xvfb.")

    motion_label = ",".join(
        f"{category}="
        + (
            f"{setting.mode}:{setting.speed_mps:.3f}m/s"
            if setting.speed_mps is not None
            else "authored"
        )
        for category in MOTION_CATEGORIES
        for setting in (motion.for_category(category),)
    )
    print(f"[grid] cells={len(cells)} | concurrency={args.max_concurrency} | "
          f"protocol={NAVIGATION_PROTOCOL_VERSION} | "
          f"provider={args.llm_provider} | "
          f"request_interval={args.llm_min_request_interval_sec:g}s | "
          f"max_tokens={args.max_tokens} | "
          f"max_steps={args.max_steps} | dynamic_step_budget={args.dynamic_step_budget} | "
          f"budget_range={args.step_budget_min}-{args.step_budget_max} | "
          f"steps_per_meter={args.steps_per_path_meter:g} | "
          f"budget_overhead={args.step_budget_overhead} | vision={args.vision_input} | "
          f"reach_m={args.reach_m:g} | "
          f"ego={args.ego_width}x{args.ego_height} | "
          f"minimap={args.minimap_width}x{args.minimap_height} | "
          f"dynamic_objects={args.dynamic_objects} | "
          f"motion_speed={motion_label} | "
          f"lighting={lighting.mode} | "
          f"file={file_name} | xvfb={use_xvfb} | output_root={args.output_root}", flush=True)

    if args.dry_run:
        for c in cells:
            print("  ", c.label)
        print(f"[dry_run] {len(cells)} cells planned. Exiting.")
        return

    # Track attempts per cell for retry accounting.
    attempts: dict[str, int] = {c.label: 0 for c in cells}
    pending = list(cells)
    started_at = time.time()
    total_done = 0
    total_failed = 0

    while pending:
        with ProcessPoolExecutor(max_workers=args.max_concurrency) as ex:
            future_to_cell = {}
            for c in pending:
                attempts[c.label] += 1
                fut = ex.submit(
                    run_cell,
                    {
                        "cell": asdict(c),
                        "file_name": file_name,
                        "python_bin": args.python_bin,
                        "llm_provider": args.llm_provider,
                        "llm_min_request_interval_sec": (
                            args.llm_min_request_interval_sec
                        ),
                        "max_tokens": args.max_tokens,
                        "max_steps": args.max_steps,
                        "dynamic_step_budget": args.dynamic_step_budget,
                        "step_budget_min": args.step_budget_min,
                        "step_budget_max": args.step_budget_max,
                        "steps_per_path_meter": args.steps_per_path_meter,
                        "step_budget_overhead": args.step_budget_overhead,
                        "reach_m": args.reach_m,
                        "ego_width": args.ego_width,
                        "ego_height": args.ego_height,
                        "minimap_width": args.minimap_width,
                        "minimap_height": args.minimap_height,
                        "dynamic_objects": args.dynamic_objects,
                        "motion_random_seed": args.motion_random_seed,
                        **{
                            f"{category}_speed_{suffix}": getattr(
                                args, f"{category}_speed_{suffix}"
                            )
                            for category in MOTION_CATEGORIES
                            for suffix in ("mps", "min_mps", "max_mps")
                        },
                        "global_light_intensity": args.global_light_intensity,
                        "light_intensity_multiplier": args.light_intensity_multiplier,
                        "light_intensity_min": args.light_intensity_min,
                        "light_intensity_max": args.light_intensity_max,
                        "light_random_seed": args.light_random_seed,
                        "light_fixed_exposure": args.light_fixed_exposure,
                        "use_xvfb": use_xvfb,
                        "xvfb_screen": args.xvfb_screen,
                        "per_cell_timeout_sec": (args.per_cell_timeout_sec or None),
                    },
                )
                future_to_cell[fut] = c

            requeue: List[Cell] = []
            for fut in as_completed(future_to_cell):
                c = future_to_cell[fut]
                try:
                    status = fut.result()
                except Exception as e:  # noqa: BLE001
                    status = {
                        "label": c.label, "ok": False, "returncode": -2000,
                        "duration_sec": 0, "frame_save_dir": str(c.frame_save_dir),
                        "log_path": str(c.frame_save_dir / "run.log"),
                        "cell": asdict(c), "dynamic_objects": args.dynamic_objects,
                        "motion": motion_speed_result_fields(args),
                        "lighting": lighting_result_fields(args),
                        "error": f"future failed: {type(e).__name__}: {e}",
                    }

                if status["ok"]:
                    total_done += 1
                    append_grid_row(grid_csv, status)
                    print(f"  [✓ {total_done}/{len(cells)}] {c.label}  "
                          f"({status['duration_sec']:.1f}s)", flush=True)
                else:
                    if attempts[c.label] <= args.max_retries:
                        print(f"  [retry {attempts[c.label]}/{args.max_retries}] "
                              f"{c.label} rc={status['returncode']} "
                              f"err={status.get('error','')}", flush=True)
                        requeue.append(c)
                    else:
                        total_failed += 1
                        append_grid_row(grid_csv, status)
                        append_failure_row(failures_csv, status, attempts[c.label])
                        print(f"  [✗ failed] {c.label} after {attempts[c.label]} attempts "
                              f"rc={status['returncode']} err={status.get('error','')}",
                              flush=True)

            pending = requeue

    elapsed = time.time() - started_at
    print(f"[grid] done. ok={total_done} failed={total_failed} "
          f"elapsed={elapsed/60:.1f} min  ({len(cells)} cells planned)", flush=True)
    print(f"[grid] aggregate: {grid_csv}")
    if total_failed:
        print(f"[grid] failures: {failures_csv}")


if __name__ == "__main__":
    main()
