#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST_DIR="${MANIFEST_DIR:-datasets/astar_pointgoal_expanded_v3}"
MANIFEST="${MANIFEST:-${MANIFEST_DIR}/collection_manifest.jsonl}"
# Reuse completed canonical episodes and collect only missing cross-route/new-seed tasks.
RAW_ROOT="${RAW_ROOT:-datasets/astar_pointgoal_pilot_tuned_v1/raw}"
DATA_ROOT="${DATA_ROOT:-${MANIFEST_DIR}/bc}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-ckpts/astar_pointgoal_pilot_tuned_v3}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/astar_pointgoal_pilot_tuned_v3_eval}"
SEEDS="${SEEDS:-4}"
WORKERS="${WORKERS:-4}"
BASE_PORT="${BASE_PORT:-27700}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-28700}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"

echo "[pointgoal-v3] preparing canonical + cross-route manifest"
"${PYTHON_BIN}" -m nav.scripts.astar.prepare_astar_pointgoal_pilot \
  --input input_points.json \
  --output "${MANIFEST}" \
  --pairing both \
  --seeds "${SEEDS}"

echo "[pointgoal-v3] collecting missing A* demonstrations"
"${PYTHON_BIN}" -m nav.scripts.astar.collect_astar_pointgoal \
  --manifest "${MANIFEST}" \
  --output-root "${RAW_ROOT}" \
  --unity "${UNITY_CLIENT}" \
  --workers "${WORKERS}" \
  --base-port "${BASE_PORT}" \
  --resume \
  --retry-failed

echo "[pointgoal-v3] exporting successful demonstrations"
"${PYTHON_BIN}" -m nav.scripts.astar.export_astar_pointgoal_dataset \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --overwrite

echo "[pointgoal-v3] validating expanded dataset"
"${PYTHON_BIN}" -m nav.scripts.bc.validate_pointgoal_dataset "${DATA_ROOT}" \
  --min-train-episodes 400 \
  --min-val-episodes 80 \
  --min-test-episodes 80 \
  --require-all-actions

echo "[pointgoal-v3] training goal-conditioned closed-loop-robust policy"
"${PYTHON_BIN}" -m nav.scripts.bc.train_bc \
  --base resnet50 \
  --data_root "${DATA_ROOT}" \
  --output_dir "${TRAIN_OUTPUT}" \
  --epochs "${TRAIN_EPOCHS}" \
  --batch_size 8 \
  --num_workers 8 \
  --seq_len 12 \
  --goal_rep polar \
  --goal_distance_scale_m 50 \
  --horizontal_flip_prob 0.5 \
  --previous_action_noise_prob 0.15 \
  --class_weight_power 0.5 \
  --label_smoothing 0.05 \
  --turn_aux_loss_weight 0.5 \
  --checkpoint_metric macro_nav_acc \
  --goal_action_residual \
  --chunk_size 1 \
  --pretrained_depth \
  --sequence_action_offset 0 \
  --no-include_stop_targets

echo "[pointgoal-v3] evaluating held-out scenes"
"${PYTHON_BIN}" -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --manifest "${MANIFEST}" \
  --checkpoint "${TRAIN_OUTPUT}/best.pt" \
  --output-root "${EVAL_OUTPUT}" \
  --unity "${UNITY_CLIENT}" \
  --python "${PYTHON_BIN}" \
  --scenes 4 \
  --base-port "${EVAL_BASE_PORT}"

echo "[pointgoal-v3] complete"
