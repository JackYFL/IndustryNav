#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/astar_pointgoal_expanded_v3/collection_manifest.jsonl}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
ROUND_ID="${ROUND_ID:-round1}"
BETA="${BETA:-0.5}"
EPISODES_PER_SCENE="${EPISODES_PER_SCENE:-4}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
BASE_PORT="${BASE_PORT:-30700}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpts/astar_pointgoal_pilot_tuned_v5_goalfix/best.pt}"
BASE_DATA_ROOT="${BASE_DATA_ROOT:-datasets/astar_pointgoal_expanded_v3/bc}"
RAW_ROOT="${RAW_ROOT:-outputs/dagger_pointgoal_${ROUND_ID}_raw}"
DAGGER_DATA_ROOT="${DAGGER_DATA_ROOT:-datasets/dagger_pointgoal_${ROUND_ID}}"
AGGREGATE_ROOT="${AGGREGATE_ROOT:-datasets/astar_pointgoal_dagger_${ROUND_ID}}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-ckpts/astar_pointgoal_dagger_${ROUND_ID}}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/astar_pointgoal_dagger_${ROUND_ID}_eval}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
MIN_TRAIN_EPISODES="${MIN_TRAIN_EPISODES:-400}"
MIN_VAL_EPISODES="${MIN_VAL_EPISODES:-80}"
MIN_TEST_EPISODES="${MIN_TEST_EPISODES:-80}"

echo "[dagger] collect ${ROUND_ID} with beta=${BETA}"
"${PYTHON_BIN}" -m nav.scripts.bc.collect_dagger_pointgoal \
  --manifest "${MANIFEST}" \
  --output-root "${RAW_ROOT}" \
  --checkpoint "${INIT_CHECKPOINT}" \
  --unity "${UNITY_CLIENT}" \
  --python "${PYTHON_BIN}" \
  --round-id "${ROUND_ID}" \
  --beta "${BETA}" \
  --episodes-per-scene "${EPISODES_PER_SCENE}" \
  --workers "${COLLECT_WORKERS}" \
  --base-port "${BASE_PORT}" \
  --resume

echo "[dagger] export policy-visited observations with A* labels"
"${PYTHON_BIN}" -m nav.scripts.bc.export_dagger_pointgoal_dataset \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${DAGGER_DATA_ROOT}" \
  --round-id "${ROUND_ID}" \
  --overwrite

echo "[dagger] aggregate expert and recovery datasets"
"${PYTHON_BIN}" -m nav.scripts.bc.aggregate_pointgoal_datasets \
  --dataset-root "${BASE_DATA_ROOT}" \
  --dataset-root "${DAGGER_DATA_ROOT}" \
  --output-root "${AGGREGATE_ROOT}" \
  --overwrite

"${PYTHON_BIN}" -m nav.scripts.bc.validate_pointgoal_dataset "${AGGREGATE_ROOT}" \
  --min-train-episodes "${MIN_TRAIN_EPISODES}" \
  --min-val-episodes "${MIN_VAL_EPISODES}" \
  --min-test-episodes "${MIN_TEST_EPISODES}" \
  --require-all-actions

echo "[dagger] fine-tune the compatible v5 policy"
"${PYTHON_BIN}" -m nav.scripts.bc.train_bc \
  --base resnet50 \
  --data_root "${AGGREGATE_ROOT}" \
  --output_dir "${TRAIN_OUTPUT}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --epochs "${TRAIN_EPOCHS}" \
  --lr 0.00003 \
  --batch_size 8 \
  --num_workers 8 \
  --seq_len 12 \
  --goal_rep polar \
  --goal_encoding unity_egocentric_v2 \
  --goal_distance_scale_m 50 \
  --horizontal_flip_prob 0.5 \
  --previous_action_noise_prob 0.15 \
  --class_weight_power 0.5 \
  --label_smoothing 0 \
  --turn_aux_loss_weight 0.1 \
  --checkpoint_metric macro_nav_acc \
  --goal_action_residual \
  --chunk_size 1 \
  --no-pretrained_depth \
  --sequence_action_offset 0 \
  --no-include_stop_targets \
  --navigation_only_actions

echo "[dagger] evaluate held-out scenes"
"${PYTHON_BIN}" -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --manifest "${MANIFEST}" \
  --eligible-dataset-manifest "${BASE_DATA_ROOT}/dataset_manifest.jsonl" \
  --checkpoint "${TRAIN_OUTPUT}/best.pt" \
  --output-root "${EVAL_OUTPUT}" \
  --unity "${UNITY_CLIENT}" \
  --python "${PYTHON_BIN}" \
  --scenes 4 \
  --base-port "$((BASE_PORT + 100))"

echo "[dagger] ${ROUND_ID} complete"
