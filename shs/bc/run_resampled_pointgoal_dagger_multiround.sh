#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/pointgoal_dagger_resampled_v1/collection_manifest.jsonl}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-ckpts/astar_pointgoal_resampled_v1_fromscratch/best.pt}"
INITIAL_DATA_ROOT="${INITIAL_DATA_ROOT:-datasets/astar_pointgoal_resampled_v1/bc}"
START_ROUND="${START_ROUND:-1}"
END_ROUND="${END_ROUND:-3}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
EPISODES_PER_SCENE="${EPISODES_PER_SCENE:-8}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
BASE_PORT="${BASE_PORT:-38000}"
MIN_TRAIN_EPISODES="${MIN_TRAIN_EPISODES:-250}"
MIN_VAL_EPISODES="${MIN_VAL_EPISODES:-40}"
MIN_TEST_EPISODES="${MIN_TEST_EPISODES:-40}"

if (( START_ROUND < 1 || END_ROUND > 3 || START_ROUND > END_ROUND )); then
  echo "START_ROUND and END_ROUND must select an inclusive range within 1..3" >&2
  exit 2
fi

if [[ -n "${WAIT_FOR_PID}" ]]; then
  echo "[dagger-multiround] waiting for PID ${WAIT_FOR_PID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 30
  done
fi

beta_for_round() {
  case "$1" in
    1) echo "0.5" ;;
    2) echo "0.25" ;;
    3) echo "0.1" ;;
  esac
}

for round in $(seq "${START_ROUND}" "${END_ROUND}"); do
  round_id="resampled_round${round}"
  if (( round == 1 )); then
    init_checkpoint="${INITIAL_CHECKPOINT}"
    base_data_root="${INITIAL_DATA_ROOT}"
    train_epochs=6
  else
    previous_round=$((round - 1))
    init_checkpoint="ckpts/astar_pointgoal_resampled_dagger_round${previous_round}/best.pt"
    base_data_root="datasets/astar_pointgoal_resampled_dagger_round${previous_round}"
    train_epochs=4
  fi

  if [[ ! -f "${init_checkpoint}" ]]; then
    echo "Missing checkpoint required by ${round_id}: ${init_checkpoint}" >&2
    exit 1
  fi
  if [[ ! -f "${base_data_root}/dataset_manifest.jsonl" ]]; then
    echo "Missing dataset required by ${round_id}: ${base_data_root}" >&2
    exit 1
  fi

  round_port=$((BASE_PORT + (round - 1) * 300))
  echo "[dagger-multiround] start ${round_id}, beta=$(beta_for_round "${round}")"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    MANIFEST="${MANIFEST}" \
    UNITY_CLIENT="${UNITY_CLIENT}" \
    ROUND_ID="${round_id}" \
    BETA="$(beta_for_round "${round}")" \
    EPISODES_PER_SCENE="${EPISODES_PER_SCENE}" \
    COLLECT_WORKERS="${COLLECT_WORKERS}" \
    BASE_PORT="${round_port}" \
    INIT_CHECKPOINT="${init_checkpoint}" \
    BASE_DATA_ROOT="${base_data_root}" \
    RAW_ROOT="outputs/dagger_pointgoal_${round_id}_raw" \
    DAGGER_DATA_ROOT="datasets/dagger_pointgoal_${round_id}" \
    AGGREGATE_ROOT="datasets/astar_pointgoal_resampled_dagger_round${round}" \
    TRAIN_OUTPUT="ckpts/astar_pointgoal_resampled_dagger_round${round}" \
    EVAL_OUTPUT="outputs/astar_pointgoal_resampled_dagger_round${round}_eval" \
    TRAIN_EPOCHS="${train_epochs}" \
    MIN_TRAIN_EPISODES="${MIN_TRAIN_EPISODES}" \
    MIN_VAL_EPISODES="${MIN_VAL_EPISODES}" \
    MIN_TEST_EPISODES="${MIN_TEST_EPISODES}" \
    bash shs/bc/run_pointgoal_dagger_round.sh
done

echo "[dagger-multiround] rounds ${START_ROUND}-${END_ROUND} complete"
