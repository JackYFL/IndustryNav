#!/usr/bin/env bash
# Export RGB/depth/minimap GIFs for all 24 scenes or an optional scene subset.
#
# Usage:
#   bash shs/export_scene_gifs.sh
#   TARGET_POSITION=700,180 bash shs/export_scene_gifs.sh
#   TARGET_POSITION=700,180 bash shs/export_scene_gifs.sh scene1 scene17
#
# Env:
#   TARGET_POSITION  Canonical minimap target as x,y. Default: 550,450.
#   FRAMES           Random decision steps per scene. Default: 30.
#   OUTPUT_DIR       Export directory. Default: outputs/all_24_scene_rgb_depth_minimap_gifs.
#   SCENE_ALL_APP    Unity client path override.
#   DYNAMIC_OBJECTS  moving | static. Default: moving.
#   HUMAN_SPEED_MPS / VEHICLE_SPEED_MPS / ROBOT_SPEED_MPS
#       Category-wide absolute speeds in meters/second. Defaults applied by
#       Python: human 1.2, vehicle 2.5, robot 1.5.
#   <CATEGORY>_SPEED_MIN_MPS / <CATEGORY>_SPEED_MAX_MPS
#       Optional deterministic category speed ranges. MOTION_RANDOM_SEED defaults to 0.
#   LIGHT_INTENSITY_MULTIPLIER or LIGHT_INTENSITY_MIN/LIGHT_INTENSITY_MAX
#       Optional fixed or deterministic range-based global light multiplier.
#   LIGHT_RANDOM_SEED / LIGHT_FIXED_EXPOSURE  Defaults: 0 / 9.0 EV.
#   PYTHON_BIN       Python interpreter. Default: <repo>/.venv/bin/python.
#   DRY_RUN          Set to 1 to print the command without launching Unity.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CLIENT="${SCENE_ALL_APP:-unity_clients/scene_all_24scenes_absolute_speed_v10.app}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/all_24_scene_rgb_depth_minimap_gifs}"
TARGET_POSITION="${TARGET_POSITION:-550,450}"
FRAMES="${FRAMES:-30}"
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
MOTION_RANDOM_SEED="${MOTION_RANDOM_SEED:-0}"
LIGHT_INTENSITY_MULTIPLIER="${LIGHT_INTENSITY_MULTIPLIER:-${GLOBAL_LIGHT_INTENSITY:-}}"
LIGHT_INTENSITY_MIN="${LIGHT_INTENSITY_MIN:-}"
LIGHT_INTENSITY_MAX="${LIGHT_INTENSITY_MAX:-}"
LIGHT_RANDOM_SEED="${LIGHT_RANDOM_SEED:-0}"
LIGHT_FIXED_EXPOSURE="${LIGHT_FIXED_EXPOSURE:-9.0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at: $PYTHON_BIN"
  exit 1
fi
if [[ ! -e "$CLIENT" ]]; then
  echo "Unity client not found at: $CLIENT"
  exit 1
fi
if [[ "$TARGET_POSITION" != *,* ]]; then
  echo "TARGET_POSITION must use x,y format, got: $TARGET_POSITION"
  exit 1
fi

TARGET_X="${TARGET_POSITION%%,*}"
TARGET_Y="${TARGET_POSITION#*,}"
if [[ -z "$TARGET_X" || -z "$TARGET_Y" || "$TARGET_Y" == *,* ]]; then
  echo "TARGET_POSITION must contain exactly two values, got: $TARGET_POSITION"
  exit 1
fi
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

CMD=(
  "$PYTHON_BIN" -m nav.scripts.validate_scene_gifs
  --file-name "$CLIENT"
  --output-dir "$OUTPUT_DIR"
  --frames "$FRAMES"
  --ego-width 320
  --ego-height 240
  --minimap-width 862
  --minimap-height 512
  --target-x "$TARGET_X"
  --target-y "$TARGET_Y"
  --dynamic-objects "$DYNAMIC_OBJECTS"
)
if [[ $# -gt 0 ]]; then
  CMD+=(--scenes "$@")
fi
motion_speed_configured=0
if [[ -n "$HUMAN_SPEED_MPS" ]]; then
  CMD+=(--human-speed-mps "$HUMAN_SPEED_MPS"); motion_speed_configured=1
elif [[ -n "$HUMAN_SPEED_MIN_MPS" ]]; then
  CMD+=(--human-speed-min-mps "$HUMAN_SPEED_MIN_MPS" --human-speed-max-mps "$HUMAN_SPEED_MAX_MPS"); motion_speed_configured=1
fi
if [[ -n "$VEHICLE_SPEED_MPS" ]]; then
  CMD+=(--vehicle-speed-mps "$VEHICLE_SPEED_MPS"); motion_speed_configured=1
elif [[ -n "$VEHICLE_SPEED_MIN_MPS" ]]; then
  CMD+=(--vehicle-speed-min-mps "$VEHICLE_SPEED_MIN_MPS" --vehicle-speed-max-mps "$VEHICLE_SPEED_MAX_MPS"); motion_speed_configured=1
fi
if [[ -n "$ROBOT_SPEED_MPS" ]]; then
  CMD+=(--robot-speed-mps "$ROBOT_SPEED_MPS"); motion_speed_configured=1
elif [[ -n "$ROBOT_SPEED_MIN_MPS" ]]; then
  CMD+=(--robot-speed-min-mps "$ROBOT_SPEED_MIN_MPS" --robot-speed-max-mps "$ROBOT_SPEED_MAX_MPS"); motion_speed_configured=1
fi
if [[ "$motion_speed_configured" == "1" ]]; then
  CMD+=(--motion-random-seed "$MOTION_RANDOM_SEED")
fi
if [[ -n "$LIGHT_INTENSITY_MULTIPLIER" ]]; then
  CMD+=(--light-intensity-multiplier "$LIGHT_INTENSITY_MULTIPLIER")
elif [[ -n "$LIGHT_INTENSITY_MIN" ]]; then
  CMD+=(--light-intensity-min "$LIGHT_INTENSITY_MIN" --light-intensity-max "$LIGHT_INTENSITY_MAX")
fi
if [[ -n "$LIGHT_INTENSITY_MULTIPLIER" || -n "$LIGHT_INTENSITY_MIN" ]]; then
  CMD+=(--light-random-seed "$LIGHT_RANDOM_SEED" --light-fixed-exposure "$LIGHT_FIXED_EXPOSURE")
fi

echo "[scene-gifs] target=(${TARGET_X},${TARGET_Y}) frames=${FRAMES} output=${OUTPUT_DIR}"
if [[ $# -gt 0 ]]; then
  echo "[scene-gifs] scenes=$*"
else
  echo "[scene-gifs] scenes=all"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
