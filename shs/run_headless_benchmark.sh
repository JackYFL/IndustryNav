#!/usr/bin/env bash
# Unified headless benchmark wrapper (macOS + Linux) for the scene_all client.
# Loops every point of a scene from input_points.json, dispatching one
# `python -m nav.scripts.run_benchmark_cell` invocation per point.
#
# Usage:
#   bash shs/run_headless_benchmark.sh <scene_code> [model_id]
#   BASELINE=astar bash shs/run_headless_benchmark.sh scene1
#
# OS handling (auto via `uname -s`): Linux wraps each invocation in `xvfb-run`
# (unless USE_XVFB=0) and defaults the client to the .x86_64 ELF; macOS uses the
# .app and no xvfb. The Python module itself is OS-agnostic.
#
# Env:
#   BASELINE     llm | astar | bc | random        (default: llm)
#   MODEL_ID     OpenRouter model id (BASELINE=llm only)   (default: google/gemini-3-flash-preview)
#   SCENE_ALL_APP / SCENE_ALL_BIN  client path override     (default: auto -> config.SCENE_ALL_BUILDS,
#                                  which prefers the in-repo unity_client/ build, no path needed)
#   SCENE_ID     override the scene_code->scene_id mapping
#   MAX_STEPS    per-point decision-step cap               (default: 70)
#   REACH_M      success radius in Unity world meters       (default: 2.0)
#   EGO_WIDTH / EGO_HEIGHT  egocentric RGB/depth sensor size (default: 512 / 512)
#   MINIMAP_WIDTH / MINIMAP_HEIGHT  minimap size; set either value and derive
#                                  the other using 862:512 (default: 862x512)
#   DYNAMIC_OBJECTS  moving | static                       (default: moving)
#   HUMAN_SPEED_MPS / VEHICLE_SPEED_MPS / ROBOT_SPEED_MPS
#       Category-wide absolute speeds in meters/second. Defaults applied by
#       Python: human 1.2, vehicle 2.5, robot 1.5.
#   <CATEGORY>_SPEED_MIN_MPS / <CATEGORY>_SPEED_MAX_MPS
#       Optional deterministic per-run ranges.
#   MOTION_RANDOM_SEED  range-sampling seed                 (default: 0)
#   LIGHT_INTENSITY_MULTIPLIER  optional fixed global light multiplier
#   LIGHT_INTENSITY_MIN / LIGHT_INTENSITY_MAX  optional per-run range
#   LIGHT_RANDOM_SEED / LIGHT_FIXED_EXPOSURE    defaults: 0 / 9.0 EV
#   PYTHON_BIN   interpreter                               (default: <repo>/.venv/bin/python)
#   USE_XVFB / XVFB_SCREEN   Linux virtual-display knobs    (default: 1 / 1724x1024x24)
#   INDUSTRYNAV_UNITY_BATCHMODE  pass -batchmode to Unity    (default: 0 on Linux, 1 elsewhere)

set -euo pipefail

SCENE_CODE="${1:-}"
if [[ -z "$SCENE_CODE" ]]; then
  echo "Usage: $0 <scene_code> [model_id]"
  echo "scene_code must be in the range scene1..scene24"
  echo "Set BASELINE={llm,astar,bc,random} (default llm)."
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at: $PYTHON_BIN"
  echo "Run 'uv sync' from the repo root, or set PYTHON_BIN."
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

# scene_code -> zero-based Unity build index. Override via SCENE_ID.
if ! SCENE_ID_DEFAULT="$("$PYTHON_BIN" - "$SCENE_CODE" <<'PY'
import sys

from nav.config import SCENE_ID_MAP

scene_code = sys.argv[1]
if scene_code not in SCENE_ID_MAP:
    raise SystemExit(1)
print(SCENE_ID_MAP[scene_code])
PY
)"; then
  echo "Unknown scene_code: $SCENE_CODE"
  exit 1
fi
SCENE_ID="${SCENE_ID:-$SCENE_ID_DEFAULT}"

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
REACH_M="${REACH_M:-2.0}"
EGO_WIDTH="${EGO_WIDTH:-512}"
EGO_HEIGHT="${EGO_HEIGHT:-512}"
MINIMAP_WIDTH="${MINIMAP_WIDTH:-}"
MINIMAP_HEIGHT="${MINIMAP_HEIGHT:-}"
DYNAMIC_OBJECTS="${DYNAMIC_OBJECTS:-moving}"
HUMAN_SPEED_MPS="${HUMAN_SPEED_MPS:-}"
HUMAN_SPEED_MIN_MPS="${HUMAN_SPEED_MIN_MPS:-}"
HUMAN_SPEED_MAX_MPS="${HUMAN_SPEED_MAX_MPS:-}"
VEHICLE_SPEED_MPS="${VEHICLE_SPEED_MPS:-}"
VEHICLE_SPEED_MIN_MPS="${VEHICLE_SPEED_MIN_MPS:-}"
VEHICLE_SPEED_MAX_MPS="${VEHICLE_SPEED_MAX_MPS:-}"
ROBOT_SPEED_MPS="${ROBOT_SPEED_MPS:-}"
ROBOT_SPEED_MIN_MPS="${ROBOT_SPEED_MIN_MPS:-}"
ROBOT_SPEED_MAX_MPS="${ROBOT_SPEED_MAX_MPS:-}"
MOTION_RANDOM_SEED="${MOTION_RANDOM_SEED:-}"
LIGHT_INTENSITY_MULTIPLIER="${LIGHT_INTENSITY_MULTIPLIER:-${GLOBAL_LIGHT_INTENSITY:-}}"
LIGHT_INTENSITY_MIN="${LIGHT_INTENSITY_MIN:-}"
LIGHT_INTENSITY_MAX="${LIGHT_INTENSITY_MAX:-}"
LIGHT_RANDOM_SEED="${LIGHT_RANDOM_SEED:-}"
LIGHT_FIXED_EXPOSURE="${LIGHT_FIXED_EXPOSURE:-}"
if [[ "$DYNAMIC_OBJECTS" != "moving" && "$DYNAMIC_OBJECTS" != "static" ]]; then
  echo "DYNAMIC_OBJECTS must be 'moving' or 'static', got: $DYNAMIC_OBJECTS"
  exit 1
fi
validate_category_speed() {
  local label="$1" fixed="$2" minimum="$3" maximum="$4"
  if [[ -n "$fixed" && ( -n "$minimum" || -n "$maximum" ) ]]; then
    echo "${label}_SPEED_MPS cannot be combined with ${label}_SPEED_MIN_MPS/MAX_MPS."
    exit 1
  fi
  if [[ -n "$minimum" && -z "$maximum" ]] || [[ -z "$minimum" && -n "$maximum" ]]; then
    echo "${label}_SPEED_MIN_MPS and ${label}_SPEED_MAX_MPS must be set together."
    exit 1
  fi
}
validate_category_speed HUMAN "$HUMAN_SPEED_MPS" "$HUMAN_SPEED_MIN_MPS" "$HUMAN_SPEED_MAX_MPS"
validate_category_speed VEHICLE "$VEHICLE_SPEED_MPS" "$VEHICLE_SPEED_MIN_MPS" "$VEHICLE_SPEED_MAX_MPS"
validate_category_speed ROBOT "$ROBOT_SPEED_MPS" "$ROBOT_SPEED_MIN_MPS" "$ROBOT_SPEED_MAX_MPS"
if [[ -n "$LIGHT_INTENSITY_MULTIPLIER" && ( -n "$LIGHT_INTENSITY_MIN" || -n "$LIGHT_INTENSITY_MAX" ) ]]; then
  echo "LIGHT_INTENSITY_MULTIPLIER cannot be combined with LIGHT_INTENSITY_MIN/MAX."
  exit 1
fi
if [[ -n "$LIGHT_INTENSITY_MIN" && -z "$LIGHT_INTENSITY_MAX" ]] || [[ -z "$LIGHT_INTENSITY_MIN" && -n "$LIGHT_INTENSITY_MAX" ]]; then
  echo "LIGHT_INTENSITY_MIN and LIGHT_INTENSITY_MAX must be set together."
  exit 1
fi
if [[ -z "$MINIMAP_WIDTH" && -z "$MINIMAP_HEIGHT" ]]; then
  MINIMAP_WIDTH=862
fi

idx=0
"$PYTHON_BIN" - <<'PY' "$INPUT_FILE" "$SCENE_CODE" | while IFS='|' read -r point_id init_wx init_wz init_dir target_x target_y; do
import json, sys
from pathlib import Path

path, scene = sys.argv[1], sys.argv[2]
data = json.loads(Path(path).read_text(encoding="utf-8"))
entries = data.get(scene)
if not entries:
    raise SystemExit(f"No input points found for scene: {scene}")
for entry in entries:
    pid = entry["point_id"]
    # input_points.json start.{x,z} are Unity WORLD coords (not pixels) — pass
    # them straight through as --init_world_x/--init_world_z so the client spawns
    # at that world point. (target.{x,y} ARE visual minimap pixels.)
    sx = entry["start"]["x"]
    sz = entry["start"]["z"]
    sd = entry["start"]["direction"]
    tx = int(entry["target"]["x"])
    ty = int(entry["target"]["y"])
    print(f"{pid}|{sx}|{sz}|{sd}|{tx}|{ty}")
PY
  echo "[benchmark] scene=${SCENE_CODE} (scene_id=${SCENE_ID}) point=${point_id} baseline=${BASELINE} run=${RUN_NAME}"
  echo "[benchmark] init_world=(${init_wx},${init_wz},dir=${init_dir}) target_px=(${target_x},${target_y})"

  WORKER_ID=$((1 + idx))
  BASE_PORT="$($PYTHON_BIN -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()')"
  FRAME_SAVE_DIR="outputs/${SCENE_CODE}/${point_id}/${RUN_NAME}"

  CMD=("$PYTHON_BIN" -m nav.scripts.run_benchmark_cell
       --baseline "$BASELINE"
       --file_name "$CLIENT"
       --scene_id "$SCENE_ID"
       --scene_name "$SCENE_CODE"
       --point_id "$point_id"
       --worker_id "$WORKER_ID"
       --base_port "$BASE_PORT"
       --max_steps "$MAX_STEPS"
       --reach_m "$REACH_M"
       --ego_width "$EGO_WIDTH"
       --ego_height "$EGO_HEIGHT"
       --dynamic_objects "$DYNAMIC_OBJECTS"
       --frame_save_dir "$FRAME_SAVE_DIR"
       --model_id "$MODEL_ID"
       --init_world_x "$init_wx"
       --init_world_z "$init_wz"
       --init_curr_direction "$init_dir"
       --target_x "$target_x"
       --target_y "$target_y")
  if [[ -n "$MINIMAP_WIDTH" ]]; then
    CMD+=(--minimap_width "$MINIMAP_WIDTH")
  fi
  if [[ -n "$MINIMAP_HEIGHT" ]]; then
    CMD+=(--minimap_height "$MINIMAP_HEIGHT")
  fi
  motion_speed_configured=0
  if [[ -n "$HUMAN_SPEED_MPS" ]]; then
    CMD+=(--human_speed_mps "$HUMAN_SPEED_MPS"); motion_speed_configured=1
  elif [[ -n "$HUMAN_SPEED_MIN_MPS" ]]; then
    CMD+=(--human_speed_min_mps "$HUMAN_SPEED_MIN_MPS" --human_speed_max_mps "$HUMAN_SPEED_MAX_MPS"); motion_speed_configured=1
  fi
  if [[ -n "$VEHICLE_SPEED_MPS" ]]; then
    CMD+=(--vehicle_speed_mps "$VEHICLE_SPEED_MPS"); motion_speed_configured=1
  elif [[ -n "$VEHICLE_SPEED_MIN_MPS" ]]; then
    CMD+=(--vehicle_speed_min_mps "$VEHICLE_SPEED_MIN_MPS" --vehicle_speed_max_mps "$VEHICLE_SPEED_MAX_MPS"); motion_speed_configured=1
  fi
  if [[ -n "$ROBOT_SPEED_MPS" ]]; then
    CMD+=(--robot_speed_mps "$ROBOT_SPEED_MPS"); motion_speed_configured=1
  elif [[ -n "$ROBOT_SPEED_MIN_MPS" ]]; then
    CMD+=(--robot_speed_min_mps "$ROBOT_SPEED_MIN_MPS" --robot_speed_max_mps "$ROBOT_SPEED_MAX_MPS"); motion_speed_configured=1
  fi
  if [[ "$motion_speed_configured" == "1" && -n "$MOTION_RANDOM_SEED" ]]; then
    CMD+=(--motion_random_seed "$MOTION_RANDOM_SEED")
  fi
  if [[ -n "$LIGHT_INTENSITY_MULTIPLIER" ]]; then
    CMD+=(--light_intensity_multiplier "$LIGHT_INTENSITY_MULTIPLIER")
  elif [[ -n "$LIGHT_INTENSITY_MIN" ]]; then
    CMD+=(--light_intensity_min "$LIGHT_INTENSITY_MIN" --light_intensity_max "$LIGHT_INTENSITY_MAX")
  fi
  if [[ -n "$LIGHT_INTENSITY_MULTIPLIER" || -n "$LIGHT_INTENSITY_MIN" ]]; then
    if [[ -n "$LIGHT_RANDOM_SEED" ]]; then
      CMD+=(--light_random_seed "$LIGHT_RANDOM_SEED")
    fi
    if [[ -n "$LIGHT_FIXED_EXPOSURE" ]]; then
      CMD+=(--light_fixed_exposure "$LIGHT_FIXED_EXPOSURE")
    fi
  fi
  # Prepend the xvfb prefix only when non-empty (safe under `set -u` on bash 3.2).
  if [[ ${#XVFB_PREFIX[@]} -gt 0 ]]; then
    CMD=("${XVFB_PREFIX[@]}" "${CMD[@]}")
  fi
  "${CMD[@]}"
  idx=$((idx + 1))
done
