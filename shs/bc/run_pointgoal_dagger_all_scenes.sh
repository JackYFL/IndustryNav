#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SOURCE_BASE_DATA_ROOT="${SOURCE_BASE_DATA_ROOT:-datasets/astar_pointgoal_resampled_v1/bc}"
BASE_DATA_ROOT="${BASE_DATA_ROOT:-datasets/astar_pointgoal_resampled_v1_all24}"
MANIFEST="${MANIFEST:-datasets/pointgoal_dagger_long_v1/collection_manifest.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpts/astar_pointgoal_dagger_long_v1_round1/best.pt}"
ROUND_ID="${ROUND_ID:-long_v1_all24_round1}"

"${PYTHON_BIN}" -m nav.scripts.tools.resplit_pointgoal_dataset \
  --source-root "${SOURCE_BASE_DATA_ROOT}" \
  --output-root "${BASE_DATA_ROOT}" \
  --val-per-scene "${VAL_PER_SCENE:-1}" \
  --test-per-scene "${TEST_PER_SCENE:-1}" \
  --seed "${SPLIT_SEED:-20260910}" \
  --expected-scenes 24 \
  --overwrite

env \
  PYTHON_BIN="${PYTHON_BIN}" \
  MANIFEST="${MANIFEST}" \
  INITIAL_CHECKPOINT="${INIT_CHECKPOINT}" \
  INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
  BASE_DATA_ROOT="${BASE_DATA_ROOT}" \
  COLLECT_SPLIT=all \
  ROUND_ID="${ROUND_ID}" \
  BETA="${BETA:-0.25}" \
  EPISODES_PER_SCENE="${EPISODES_PER_SCENE:-4}" \
  COLLECT_WORKERS="${COLLECT_WORKERS:-4}" \
  BASE_PORT="${BASE_PORT:-43000}" \
  RAW_ROOT="${RAW_ROOT:-outputs/dagger_pointgoal_${ROUND_ID}_raw}" \
  DAGGER_DATA_ROOT="${DAGGER_DATA_ROOT:-datasets/dagger_pointgoal_${ROUND_ID}}" \
  AGGREGATE_ROOT="${AGGREGATE_ROOT:-datasets/astar_pointgoal_dagger_${ROUND_ID}}" \
  TRAIN_OUTPUT="${TRAIN_OUTPUT:-ckpts/astar_pointgoal_dagger_${ROUND_ID}}" \
  EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/astar_pointgoal_dagger_${ROUND_ID}_eval}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}" \
  MIN_TRAIN_EPISODES="${MIN_TRAIN_EPISODES:-300}" \
  MIN_VAL_EPISODES="${MIN_VAL_EPISODES:-24}" \
  MIN_TEST_EPISODES="${MIN_TEST_EPISODES:-24}" \
  bash shs/bc/run_pointgoal_dagger_round.sh
