#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/astar_pointgoal_expanded_v3/collection_manifest.jsonl}"
DATA_ROOT="${DATA_ROOT:-datasets/astar_pointgoal_expanded_v3/bc}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-ckpts/astar_pointgoal_pilot_tuned_v4}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/astar_pointgoal_pilot_tuned_v4_eval}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-12}"
EVAL_SCENES="${EVAL_SCENES:-4}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-29700}"

echo "[pointgoal-v4] validating successful A* demonstrations"
"${PYTHON_BIN}" -m nav.scripts.bc.validate_pointgoal_dataset "${DATA_ROOT}" \
  --min-train-episodes 400 \
  --min-val-episodes 80 \
  --min-test-episodes 80 \
  --require-all-actions

echo "[pointgoal-v4] training three-action distance-terminated policy"
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
  --label_smoothing 0 \
  --turn_aux_loss_weight 0.1 \
  --checkpoint_metric macro_nav_acc \
  --goal_action_residual \
  --chunk_size 1 \
  --pretrained_depth \
  --sequence_action_offset 0 \
  --no-include_stop_targets \
  --navigation_only_actions

echo "[pointgoal-v4] evaluating successful held-out expert routes"
"${PYTHON_BIN}" -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --manifest "${MANIFEST}" \
  --eligible-dataset-manifest "${DATA_ROOT}/dataset_manifest.jsonl" \
  --checkpoint "${TRAIN_OUTPUT}/best.pt" \
  --output-root "${EVAL_OUTPUT}" \
  --unity "${UNITY_CLIENT}" \
  --python "${PYTHON_BIN}" \
  --scenes "${EVAL_SCENES}" \
  --base-port "${EVAL_BASE_PORT}"

echo "[pointgoal-v4] complete"
