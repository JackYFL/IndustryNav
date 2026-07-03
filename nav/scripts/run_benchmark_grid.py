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
    DEFAULT_PROMPT_NOVISION,
    DEFAULT_PROMPT_VISION,
    GRID_CSV_FIELDS,
    LLM_DEFAULT_HISTORY_SIZE,
    SCENE_ID_MAP,
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
        if self.history_size == LLM_DEFAULT_HISTORY_SIZE:
            base = REPO_ROOT / "outputs"
        else:
            base = HISTORY_SWEEP_ROOT / f"hs{self.history_size}"
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
) -> List[Cell]:
    pts = load_input_points()
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


def run_cell(args_dict: dict) -> dict:
    """Worker function. Runs in a child process; returns a status dict."""
    cell = Cell(**args_dict["cell"])
    file_name = args_dict["file_name"]
    python_bin = args_dict["python_bin"]
    max_steps = args_dict["max_steps"]
    use_xvfb = args_dict["use_xvfb"]
    xvfb_screen = args_dict["xvfb_screen"]

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
        "--frame_save_dir", str(cell.frame_save_dir),
        "--prompt_file", prompt_file,
        "--model_id", cell.model,
        "--init_world_x", str(cell.init_world_x),
        "--init_world_z", str(cell.init_world_z),
        "--init_curr_direction", str(cell.init_direction),
        "--target_x", str(cell.target_x),
        "--target_y", str(cell.target_y),
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
        ok = (proc.returncode == 0) and cell.results_csv.exists()
        return {
            "label": cell.label,
            "ok": ok,
            "returncode": proc.returncode,
            "duration_sec": duration,
            "frame_save_dir": str(cell.frame_save_dir),
            "log_path": str(log_path),
            "cell": asdict(cell),
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
                   help="OpenRouter model ids to benchmark. Example: anthropic/claude-sonnet-4.6 google/gemini-3-flash-preview")
    p.add_argument("--scenes", nargs="+", default=list(SCENE_ID_MAP.keys()),
                   help="Scene names to run; defaults to all 12.")
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
    p.add_argument("--max_concurrency", type=int, default=4,
                   help="Max parallel cells. Start at 4; bump up if you don't see OpenRouter 429s.")
    p.add_argument("--max_steps", type=int, default=70)
    p.add_argument("--max_retries", type=int, default=2,
                   help="Retry a failed cell this many times before logging it to failures.csv.")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells whose results.csv already exists.")
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

    vision_modes = {"on": [True], "off": [False], "both": [True, False]}[args.vision_input]
    cells = build_grid(
        models=args.models,
        scenes=args.scenes,
        seeds=args.seeds,
        vision_modes=vision_modes,
        history_sizes=args.history_sizes,
        points=args.points,
    )

    # Aggregates default to a per-invocation timestamped dir under analysis/.
    run_dir = REPO_ROOT / ANALYSIS_ROOT / "grid_runs" / time.strftime("%Y%m%d_%H%M%S")
    grid_csv = Path(args.grid_csv) if args.grid_csv else run_dir / "runs.csv"
    failures_csv = Path(args.failures_csv) if args.failures_csv else run_dir / "failures.csv"

    if args.resume:
        before = len(cells)
        cells = [c for c in cells if not c.results_csv.exists()]
        print(f"[resume] skipping {before - len(cells)} already-complete cells "
              f"(results.csv present); {len(cells)} remain.", flush=True)

    file_name = resolve_file_name(args.file_name)
    if not Path(file_name).exists():
        raise SystemExit(f"Unity client not found at: {file_name}. Set --file_name or the env var.")
    if not Path(args.python_bin).exists():
        raise SystemExit(f"Python interpreter not found at: {args.python_bin}. Run `uv sync` or pass --python_bin.")

    use_xvfb = (platform.system() == "Linux") and (not args.no_xvfb)
    if use_xvfb and not shutil.which("xvfb-run"):
        raise SystemExit("xvfb-run not found on PATH. `sudo apt install xvfb` or pass --no_xvfb.")

    print(f"[grid] cells={len(cells)} | concurrency={args.max_concurrency} | "
          f"max_steps={args.max_steps} | vision={args.vision_input} | "
          f"file={file_name} | xvfb={use_xvfb}", flush=True)

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
                        "max_steps": args.max_steps,
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
                        "cell": asdict(c), "error": f"future failed: {type(e).__name__}: {e}",
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
