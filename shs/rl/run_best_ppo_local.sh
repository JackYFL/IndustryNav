#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="${PROFILE:-exact}"
case "${PROFILE}" in
  exact)
    : "${DEVICE:=cuda}"
    : "${NPROC_PER_NODE:=2}"
    : "${NUM_ENVS_PER_RANK:=12}"
    : "${TOTAL_UPDATES:=500}"
    : "${ROLLOUT_STEPS:=128}"
    : "${USE_REFERENCE_ANCHOR:=1}"
    ;;
  local)
    : "${DEVICE:=auto}"
    : "${NPROC_PER_NODE:=1}"
    : "${NUM_ENVS_PER_RANK:=4}"
    : "${TOTAL_UPDATES:=500}"
    : "${ROLLOUT_STEPS:=128}"
    : "${USE_REFERENCE_ANCHOR:=0}"
    ;;
  smoke)
    : "${DEVICE:=cpu}"
    : "${NPROC_PER_NODE:=1}"
    : "${NUM_ENVS_PER_RANK:=1}"
    : "${TOTAL_UPDATES:=1}"
    : "${ROLLOUT_STEPS:=2}"
    : "${USE_REFERENCE_ANCHOR:=0}"
    ;;
  *) echo "PROFILE must be exact, local, or smoke" >&2; exit 2 ;;
esac

if [[ -z "${UNITY_CLIENT:-}" ]]; then
  case "$(uname -s)" in
    Darwin) UNITY_CLIENT="unity_clients/scene_all.app" ;;
    Linux) UNITY_CLIENT="clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64" ;;
    *) echo "Set UNITY_CLIENT for this platform" >&2; exit 2 ;;
  esac
fi

export DEVICE NPROC_PER_NODE NUM_ENVS_PER_RANK TOTAL_UPDATES ROLLOUT_STEPS UNITY_CLIENT
export INITIAL_ACTOR="${INITIAL_ACTOR:-analysis/ppo_dagger52_20260912/repro_full_update100/ppo_lr1e6/update_000100.pt}"
export REFERENCE_ACTOR="${REFERENCE_ACTOR:-analysis/ppo_dagger52_20260912/repro_full_update100/initialization_fp32/initial_actor.pt}"
export MANIFEST="${MANIFEST:-datasets/pointgoal_ppo_independent_final/train_manifest.jsonl}"
export NAVIGATION_MAP_DIR="${NAVIGATION_MAP_DIR:-analysis/ppo80_20260912/reward_maps_v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-experiments/local_best_pointgoal/ppo_stage2}"
export BASE_PORT="${BASE_PORT:-64000}"
export CRITIC_WARMUP_UPDATES="${CRITIC_WARMUP_UPDATES:-8}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"

EXTRA_ARGS=()
if [[ "${USE_REFERENCE_ANCHOR}" == "1" ]]; then
  ANCHOR="${REFERENCE_RESUME_ANCHOR:-experiments/ppo80_v4_20260913/resume_after_unity255_update141/update_000141.pt}"
  ANCHOR_PATH="${ANCHOR}"
  if [[ "${ANCHOR_PATH}" != /* ]]; then
    ANCHOR_PATH="${REPO_ROOT}/${ANCHOR_PATH}"
  fi
  EXTRA_ARGS+=(--resume)
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "# prepare ${OUTPUT_DIR}/latest.pt from ${ANCHOR}, then resume" >&2
  else
    [[ -f "${ANCHOR_PATH}" ]] || { echo "Missing resume anchor: ${ANCHOR_PATH}" >&2; exit 2; }
    if [[ -e "${OUTPUT_DIR}" ]]; then
      [[ -f "${OUTPUT_DIR}/latest.pt" ]] || {
        echo "Existing OUTPUT_DIR has no latest.pt: ${OUTPUT_DIR}" >&2
        exit 2
      }
    else
      mkdir -p "${OUTPUT_DIR}"
      ln -s "${ANCHOR_PATH}" "${OUTPUT_DIR}/latest.pt"
    fi
  fi
elif [[ "${RESUME:-0}" == "1" ]]; then
  [[ -f "${OUTPUT_DIR}/latest.pt" ]] || {
    echo "Cannot resume without ${OUTPUT_DIR}/latest.pt" >&2
    exit 2
  }
  EXTRA_ARGS+=(--resume)
elif [[ "${DRY_RUN:-0}" != "1" && -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing PPO stage-2 output: ${OUTPUT_DIR}" >&2
  exit 2
fi

if (( ${#EXTRA_ARGS[@]} )); then
  exec bash shs/rl/run_pointgoal_ppo_independent.sh "${EXTRA_ARGS[@]}" "$@"
fi
exec bash shs/rl/run_pointgoal_ppo_independent.sh "$@"
