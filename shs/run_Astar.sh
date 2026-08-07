#!/usr/bin/env bash
# Convenience wrapper for running the A* navigation baseline against the
# unified scene_all client.
#
# Usage:
#   bash shs/run_Astar.sh <scene_code|all> [point_id]
#
# Examples:
#   bash shs/run_Astar.sh scene1
#   bash shs/run_Astar.sh scene1 point1
#   ASTAR_DEBUG_VIZ=1 bash shs/run_Astar.sh scene1 point1
#   MAX_STEPS=120 bash shs/run_Astar.sh all
#
# Env:
#   SCENE_ALL_APP / SCENE_ALL_BIN
#       Unity client path override. Defaults to "auto", which lets
#       nav.config.resolve_scene_all_path discover a local scene_all build.
#   PYTHON_BIN
#       Python interpreter. Defaults to <repo>/.venv/bin/python.
#   MAX_STEPS
#       Per-point decision-step cap. Default: 70.
#   SIM_STEPS_PER_DECISION
#       Unity simulation steps per A* action. Default: 2.
#   REACH_PX
#       Success radius in minimap pixels. Default: 20.
#   MODALITIES
#       Sensor modalities to save. Default: ego,minimap,depth.
#   EGO_WIDTH / EGO_HEIGHT
#       Egocentric RGB/depth sensor resolution. Defaults: 512 / 512.
#   MARKER_SOURCE
#       vector | red. Default: vector. vector draws the Python-side red
#       dot/arrow from Unity vector observations; red uses legacy HSV detection.
#   HIDE_UNITY_RED_MARKER
#       1 | 0. Default: 1. When using vector marker, remove the old Unity
#       red cone/dot from saved/planned minimaps before drawing the Python marker.
#   ASTAR_DEBUG_VIZ
#       Set to 1 to save A* walkable-grid/path debug images.
#   ASTAR_DEBUG_DIR
#       Optional explicit debug image directory. Default:
#       <frame_save_dir>/astar_debug when ASTAR_DEBUG_VIZ=1.
#   ASTAR_OBSTACLE_INFLATE_PX
#       Optional obstacle dilation in minimap pixels. Default: 8, except
#       scene1/point2 defaults to 24 to avoid the shelf collider corner.
#   ASTAR_MARKER_CLEAR_PX
#       Optional marker clearing radius. Default: 16.
#   ASTAR_PROXY_STOP_REAL_DIST_PX
#       Optional real-target distance threshold for blocked-target proxy stop.
#       Default: 65.
#   DRY_RUN
#       Set to 1 to print commands without launching Unity.
#   BASE_PORT_START
#       Fallback base port when automatic free-port probing is unavailable.
#       Default: 5507.
#   USE_XVFB / XVFB_SCREEN
#       Linux only. Defaults: 1 / 1724x1024x24.
#   INDUSTRYNAV_UNITY_BATCHMODE
#       Whether Python passes -batchmode to Unity. Defaults to 0 on Linux and
#       1 elsewhere.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SCENE_ARG="${1:-}"
POINT_FILTER="${2:-}"

if [[ -z "$SCENE_ARG" ]]; then
  echo "Usage: $0 <scene_code|all> [point_id]"
  echo "scene_code must be one of: scene1 scene2 scene3 scene4 scene5 scene6 scene7 scene8 scene9 scene10 scene11 scene12"
  echo "Use 'all' to run every scene. Optional point_id filters to one point, e.g. point1."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at: $PYTHON_BIN"
  echo "Run 'uv sync' from the repo root, or set PYTHON_BIN."
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

INPUT_FILE="${INPUT_FILE:-${REPO_ROOT}/input_points.json}"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing input points file: $INPUT_FILE"
  exit 1
fi

scene_id_for() {
  case "$1" in
    scene1)  echo 1  ;;
    scene2)  echo 2  ;;
    scene3)  echo 3  ;;
    scene4)  echo 4  ;;
    scene5)  echo 5  ;;
    scene6)  echo 6  ;;
    scene7)  echo 7  ;;
    scene8)  echo 8  ;;
    scene9)  echo 9  ;;
    scene10) echo 10 ;;
    scene11) echo 11 ;;
    scene12) echo 12 ;;
    *)       return 1 ;;
  esac
}

if [[ "$SCENE_ARG" == "all" ]]; then
  SCENES=(scene1 scene2 scene3 scene4 scene5 scene6 scene7 scene8 scene9 scene10 scene11 scene12)
else
  if ! scene_id_for "$SCENE_ARG" >/dev/null; then
    echo "Unknown scene_code: $SCENE_ARG"
    exit 1
  fi
  SCENES=("$SCENE_ARG")
fi

OS="$(uname -s)"
XVFB_PREFIX=()
if [[ "$OS" == "Linux" ]]; then
  CLIENT="${SCENE_ALL_BIN:-auto}"
  if [[ "$CLIENT" != "auto" && ! -x "$CLIENT" ]]; then
    echo "Unified Unity client (Linux ELF) not found or not executable at: $CLIENT"
    echo "Set SCENE_ALL_BIN=auto, set an explicit path, or chmod +x the binary."
    exit 1
  fi

  USE_XVFB="${USE_XVFB:-1}"
  XVFB_SCREEN="${XVFB_SCREEN:-1724x1024x24}"
  if [[ "$USE_XVFB" == "1" ]]; then
    if ! command -v xvfb-run >/dev/null 2>&1; then
      echo "xvfb-run not found. Install with: sudo apt install -y xvfb, or set USE_XVFB=0."
      exit 1
    fi
    XVFB_PREFIX=(xvfb-run -a -s "-screen 0 ${XVFB_SCREEN}")
  fi
else
  CLIENT="${SCENE_ALL_APP:-auto}"
  if [[ "$CLIENT" != "auto" && ! -e "$CLIENT" ]]; then
    echo "Unified Unity client not found at: $CLIENT"
    echo "Set SCENE_ALL_APP=auto, or set an explicit path to scene_all.app."
    exit 1
  fi
fi

if [[ "$CLIENT" == "auto" ]]; then
  CLIENT="$("$PYTHON_BIN" - <<'PY'
from nav.config import resolve_scene_all_path
print(resolve_scene_all_path("auto"))
PY
)"
fi

MAX_STEPS_WAS_SET="${MAX_STEPS+x}"
MAX_STEPS="${MAX_STEPS:-70}"
SIM_STEPS_PER_DECISION="${SIM_STEPS_PER_DECISION:-2}"
REACH_PX="${REACH_PX:-20}"
MODALITIES="${MODALITIES:-ego,minimap,depth}"
EGO_WIDTH="${EGO_WIDTH:-512}"
EGO_HEIGHT="${EGO_HEIGHT:-512}"
MARKER_SOURCE="${MARKER_SOURCE:-vector}"
HIDE_UNITY_RED_MARKER="${HIDE_UNITY_RED_MARKER:-1}"
RUN_NAME="${RUN_NAME:-astar}"
ASTAR_DEBUG_VIZ="${ASTAR_DEBUG_VIZ:-0}"
ASTAR_MARKER_CLEAR_PX="${ASTAR_MARKER_CLEAR_PX:-16}"
ASTAR_PROXY_STOP_REAL_DIST_PX="${ASTAR_PROXY_STOP_REAL_DIST_PX:-65}"
DRY_RUN="${DRY_RUN:-0}"
BASE_PORT_START="${BASE_PORT_START:-5507}"

pick_base_port() {
  local fallback="$1"
  local port
  if port="$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null)"; then
    echo "$port"
  else
    echo "$fallback"
  fi
}

echo "[astar] repo=${REPO_ROOT}"
echo "[astar] client=${CLIENT}"
echo "[astar] scenes=${SCENES[*]}"
if [[ -n "$POINT_FILTER" ]]; then
  echo "[astar] point_filter=${POINT_FILTER}"
fi
echo "[astar] max_steps=${MAX_STEPS} sim_steps_per_decision=${SIM_STEPS_PER_DECISION} reach_px=${REACH_PX} ego=${EGO_WIDTH}x${EGO_HEIGHT}"
echo "[astar] marker_source=${MARKER_SOURCE}"
echo "[astar] hide_unity_red_marker=${HIDE_UNITY_RED_MARKER}"

idx=0
for scene_name in "${SCENES[@]}"; do
  scene_id="$(scene_id_for "$scene_name")"

  "$PYTHON_BIN" - <<'PY' "$INPUT_FILE" "$scene_name" "$POINT_FILTER" | while IFS='|' read -r point_id init_wx init_wz init_dir target_x target_y; do
import json
import sys
from pathlib import Path

path, scene, point_filter = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.loads(Path(path).read_text(encoding="utf-8"))
entries = data.get(scene)
if not entries:
    raise SystemExit(f"No input points found for scene: {scene}")

matched = 0
for entry in entries:
    pid = str(entry["point_id"])
    if point_filter and pid != point_filter:
        continue
    start = entry["start"]
    target = entry["target"]
    print(
        f"{pid}|{float(start['x'])}|{float(start['z'])}|"
        f"{float(start.get('direction', 180.0))}|"
        f"{int(target['x'])}|{int(target['y'])}",
        flush=True,
    )
    matched += 1

if point_filter and matched == 0:
    raise SystemExit(f"Point {point_filter!r} not found in scene {scene!r}")
PY
    echo "[astar] scene=${scene_name} scene_id=${scene_id} point=${point_id}"
    echo "[astar] init_world=(${init_wx},${init_wz},dir=${init_dir}) target_px=(${target_x},${target_y})"

    worker_id=$((31 + idx))
    base_port="$(pick_base_port "$((BASE_PORT_START + idx))")"
    frame_save_dir="outputs/${scene_name}/${point_id}/${RUN_NAME}"
    max_steps_for_point="$MAX_STEPS"
    if [[ -z "$MAX_STEPS_WAS_SET" && "$scene_name" == "scene1" && "$point_id" == "point4" ]]; then
      max_steps_for_point=100
    fi
    astar_obstacle_inflate_px="${ASTAR_OBSTACLE_INFLATE_PX:-8}"
    if [[ -z "${ASTAR_OBSTACLE_INFLATE_PX:-}" && "$scene_name" == "scene1" && "$point_id" == "point2" ]]; then
      astar_obstacle_inflate_px=24
    fi

    cmd=("$PYTHON_BIN" -m nav.scripts.run_benchmark_cell
         --baseline astar
         --file_name "$CLIENT"
         --scene_id "$scene_id"
         --scene_name "$scene_name"
         --point_id "$point_id"
         --worker_id "$worker_id"
         --base_port "$base_port"
         --max_steps "$max_steps_for_point"
         --sim_steps_per_decision "$SIM_STEPS_PER_DECISION"
         --ego_width "$EGO_WIDTH"
         --ego_height "$EGO_HEIGHT"
         --reach_px "$REACH_PX"
         --modalities "$MODALITIES"
         --marker_source "$MARKER_SOURCE"
         --frame_save_dir "$frame_save_dir"
         --model_id astar
         --astar_obstacle_inflate_px "$astar_obstacle_inflate_px"
         --astar_marker_clear_px "$ASTAR_MARKER_CLEAR_PX"
         --astar_proxy_stop_real_dist_px "$ASTAR_PROXY_STOP_REAL_DIST_PX"
         --init_world_x "$init_wx"
         --init_world_z "$init_wz"
         --init_curr_direction "$init_dir"
         --target_x "$target_x"
         --target_y "$target_y")

    if [[ "$HIDE_UNITY_RED_MARKER" == "0" || "$HIDE_UNITY_RED_MARKER" == "false" || "$HIDE_UNITY_RED_MARKER" == "off" ]]; then
      cmd+=(--no-hide_unity_red_marker)
    else
      cmd+=(--hide_unity_red_marker)
    fi

    if [[ "$ASTAR_DEBUG_VIZ" == "1" || "$ASTAR_DEBUG_VIZ" == "true" || "$ASTAR_DEBUG_VIZ" == "on" ]]; then
      cmd+=(--astar_debug_viz)
      if [[ -n "${ASTAR_DEBUG_DIR:-}" ]]; then
        cmd+=(--astar_debug_dir "$ASTAR_DEBUG_DIR")
      fi
    fi

    if [[ ${#XVFB_PREFIX[@]} -gt 0 ]]; then
      cmd=("${XVFB_PREFIX[@]}" "${cmd[@]}")
    fi

    echo "[astar] output=${frame_save_dir}"
    echo "[astar] max_steps_for_point=${max_steps_for_point}"
    echo "[astar] obstacle_inflate_px=${astar_obstacle_inflate_px} marker_clear_px=${ASTAR_MARKER_CLEAR_PX} proxy_stop_real_dist_px=${ASTAR_PROXY_STOP_REAL_DIST_PX}"
    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "on" ]]; then
      printf '[astar] dry-run command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
    idx=$((idx + 1))
  done
done
