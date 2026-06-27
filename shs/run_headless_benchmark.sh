#!/usr/bin/env bash
# Unified headless benchmark wrapper (macOS + Linux) for the scene_all client.
# Loops every point of a scene from input_points.json, dispatching one
# `python -m nav.scripts.run_benchmark_cell` invocation per point.
#
# Usage:
#   bash shs/run_headless_benchmark.sh <scene_name> [model_id]
#   BASELINE=astar bash shs/run_headless_benchmark.sh yifan1
#
# OS handling (auto via `uname -s`): Linux wraps each invocation in `xvfb-run`
# (unless USE_XVFB=0) and defaults the client to the .x86_64 ELF; macOS uses the
# .app and no xvfb. The Python module itself is OS-agnostic.
#
# Env:
#   BASELINE     llm | astar | navid | bc | random        (default: llm)
#   MODEL_ID     OpenRouter model id (BASELINE=llm only)   (default: google/gemini-3-flash-preview)
#   SCENE_ALL_APP / SCENE_ALL_BIN  client path override     (default: auto -> config.SCENE_ALL_BUILDS,
#                                  which prefers the in-repo unity_client/ build, no path needed)
#   SCENE_ID     override the scene_name->scene_id mapping
#   MAX_STEPS    per-point decision-step cap               (default: 70)
#   PYTHON_BIN   interpreter                               (default: <repo>/.venv/bin/python)
#   USE_XVFB / XVFB_SCREEN   Linux virtual-display knobs    (default: 1 / 1724x1024x24)

set -euo pipefail

SCENE_NAME="${1:-}"
if [[ -z "$SCENE_NAME" ]]; then
  echo "Usage: $0 <scene_name> [model_id]"
  echo "scene_name must be one of: yifan1 yifan2 yifan3 yifan4 yicheng lichi1 lichi2 xinyu1 xinyu2 anh1 anh2 anh3"
  echo "Set BASELINE={llm,astar,navid,bc,random} (default llm)."
  exit 1
fi

BASELINE="${BASELINE:-llm}"
MODEL_ID="${2:-${MODEL_ID:-google/gemini-3-flash-preview}}"
# Output subdir is the model short-name for llm, otherwise the baseline token.
if [[ "$BASELINE" == "llm" ]]; then
  RUN_NAME="${MODEL_ID##*/}"
else
  RUN_NAME="$BASELINE"
fi

# scene_name -> scene_id baked into the unified client. Override via SCENE_ID.
case "$SCENE_NAME" in
  yifan1)  SCENE_ID_DEFAULT=1  ;;
  yifan2)  SCENE_ID_DEFAULT=2  ;;
  yifan3)  SCENE_ID_DEFAULT=3  ;;
  yifan4)  SCENE_ID_DEFAULT=4  ;;
  yicheng) SCENE_ID_DEFAULT=5  ;;
  lichi1)  SCENE_ID_DEFAULT=6  ;;
  lichi2)  SCENE_ID_DEFAULT=7  ;;
  xinyu1)  SCENE_ID_DEFAULT=8  ;;
  xinyu2)  SCENE_ID_DEFAULT=9  ;;
  anh1)    SCENE_ID_DEFAULT=10 ;;
  anh2)    SCENE_ID_DEFAULT=11 ;;
  anh3)    SCENE_ID_DEFAULT=12 ;;
  *)       echo "Unknown scene_name: $SCENE_NAME"; exit 1 ;;
esac
SCENE_ID="${SCENE_ID:-$SCENE_ID_DEFAULT}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
# Make `python -m nav.scripts.run_benchmark_cell` resolve regardless of CWD.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at: $PYTHON_BIN"
  echo "Run 'uv sync' from the repo root, or set PYTHON_BIN."
  exit 1
fi

# OS branch: client path default + (Linux) xvfb prefix.
OS="$(uname -s)"
XVFB_PREFIX=()
if [[ "$OS" == "Linux" ]]; then
  CLIENT="${SCENE_ALL_BIN:-auto}"
  if [[ "$CLIENT" != "auto" && ! -x "$CLIENT" ]]; then
    echo "Unified Unity client (Linux ELF) not found or not executable at: $CLIENT"
    echo "Set SCENE_ALL_BIN=auto (default) or an explicit path; chmod +x if needed."
    exit 1
  fi
  USE_XVFB="${USE_XVFB:-1}"
  XVFB_SCREEN="${XVFB_SCREEN:-1724x1024x24}"
  if [[ "$USE_XVFB" == "1" ]]; then
    if ! command -v xvfb-run >/dev/null 2>&1; then
      echo "xvfb-run not found. Install with: sudo apt install -y xvfb (or set USE_XVFB=0)."
      exit 1
    fi
    XVFB_PREFIX=(xvfb-run -a -s "-screen 0 ${XVFB_SCREEN}")
  fi
else
  CLIENT="${SCENE_ALL_APP:-auto}"
  if [[ "$CLIENT" != "auto" && ! -e "$CLIENT" ]]; then
    echo "Unified Unity client not found at: $CLIENT"
    echo "Set SCENE_ALL_APP=auto (default) or an explicit path to scene_all.app."
    exit 1
  fi
fi

if [[ "$BASELINE" == "llm" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Warning: OPENROUTER_API_KEY is not set. LLM calls will fail."
fi

INPUT_FILE="${REPO_ROOT}/input_points.json"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing $INPUT_FILE"
  exit 1
fi

MAX_STEPS="${MAX_STEPS:-70}"

idx=0
"$PYTHON_BIN" - <<'PY' "$INPUT_FILE" "$SCENE_NAME" | while IFS='|' read -r point_id init_wx init_wz init_dir target_x target_y init_px init_py; do
import json, sys
from pathlib import Path

path, scene = sys.argv[1], sys.argv[2]
data = json.loads(Path(path).read_text(encoding="utf-8"))
for entry in data.get(scene, []):
    pid = entry["point_id"]
    # input_points.json start.{x,z} are Unity WORLD coords (not pixels) — pass
    # them straight through as --init_world_x/--init_world_z so the client spawns
    # at that world point. (target.{x,y} ARE visual minimap pixels.)
    sx = entry["start"]["x"]
    sz = entry["start"]["z"]
    sd = entry["start"]["direction"]
    start_pixel = entry.get("start_pixel", {})
    tx = int(entry["target"]["x"])
    ty = int(entry["target"]["y"])
    px = int(start_pixel["x"]) if "x" in start_pixel else ""
    py = int(start_pixel["y"]) if "y" in start_pixel else ""
    print(f"{pid}|{sx}|{sz}|{sd}|{tx}|{ty}|{px}|{py}")
PY
  echo "[benchmark] scene=${SCENE_NAME} (scene_id=${SCENE_ID}) point=${point_id} baseline=${BASELINE} run=${RUN_NAME}"
  echo "[benchmark] init_world=(${init_wx},${init_wz},dir=${init_dir}) target_px=(${target_x},${target_y})"
  if [[ -n "$init_px" && -n "$init_py" ]]; then
    echo "[benchmark] init_pixel_calibration=(${init_px},${init_py})"
  fi

  WORKER_ID=$((1 + idx))
  BASE_PORT="$($PYTHON_BIN -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()')"
  FRAME_SAVE_DIR="outputs/${SCENE_NAME}/${point_id}/${RUN_NAME}"

  CMD=("$PYTHON_BIN" -m nav.scripts.run_benchmark_cell
       --baseline "$BASELINE"
       --file_name "$CLIENT"
       --scene_id "$SCENE_ID"
       --scene_name "$SCENE_NAME"
       --point_id "$point_id"
       --worker_id "$WORKER_ID"
       --base_port "$BASE_PORT"
       --max_steps "$MAX_STEPS"
       --frame_save_dir "$FRAME_SAVE_DIR"
       --model_id "$MODEL_ID"
       --init_world_x "$init_wx"
       --init_world_z "$init_wz"
       --init_curr_direction "$init_dir"
       --target_x "$target_x"
       --target_y "$target_y")
  if [[ -n "$init_px" && -n "$init_py" ]]; then
    CMD+=(--init_curr_x "$init_px" --init_curr_y "$init_py")
  fi
  # Prepend the xvfb prefix only when non-empty (safe under `set -u` on bash 3.2).
  if [[ ${#XVFB_PREFIX[@]} -gt 0 ]]; then
    CMD=("${XVFB_PREFIX[@]}" "${CMD[@]}")
  fi
  "${CMD[@]}"
  idx=$((idx + 1))
done
