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
    : "${TOTAL_UPDATES:=100}"
    : "${ROLLOUT_STEPS:=64}"
    ;;
  local)
    : "${DEVICE:=auto}"
    : "${NPROC_PER_NODE:=1}"
    : "${NUM_ENVS_PER_RANK:=4}"
    : "${TOTAL_UPDATES:=100}"
    : "${ROLLOUT_STEPS:=64}"
    ;;
  smoke)
    : "${DEVICE:=cpu}"
    : "${NPROC_PER_NODE:=1}"
    : "${NUM_ENVS_PER_RANK:=1}"
    : "${TOTAL_UPDATES:=1}"
    : "${ROLLOUT_STEPS:=2}"
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
export INITIAL_ACTOR="${INITIAL_ACTOR:-analysis/ppo_dagger52_20260912/repro_full_update100/initialization_fp32/initial_actor.pt}"
export MANIFEST="${MANIFEST:-analysis/ppo_dagger52_20260912/repro_full_update100/initialization_fp32/train_manifest.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-experiments/local_best_pointgoal/ppo_stage1}"
export BASE_PORT="${BASE_PORT:-47000}"
export CRITIC_WARMUP_UPDATES="${CRITIC_WARMUP_UPDATES:-4}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

EXTRA_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  [[ -f "${OUTPUT_DIR}/latest.pt" ]] || {
    echo "Cannot resume without ${OUTPUT_DIR}/latest.pt" >&2
    exit 2
  }
  EXTRA_ARGS+=(--resume)
elif [[ "${DRY_RUN:-0}" != "1" && -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing PPO stage-1 output: ${OUTPUT_DIR}" >&2
  exit 2
fi

if (( ${#EXTRA_ARGS[@]} )); then
  exec bash shs/rl/run_pointgoal_ppo_dagger_actor.sh "${EXTRA_ARGS[@]}" "$@"
fi
exec bash shs/rl/run_pointgoal_ppo_dagger_actor.sh "$@"
