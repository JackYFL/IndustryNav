#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-datasets/astar_pointgoal_pilot_tuned_v1/bc}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-ckpts/astar_pointgoal_pilot_tuned_v2}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/astar_pointgoal_pilot_tuned_v2_eval}"
MANIFEST="${MANIFEST:-datasets/astar_pointgoal_pilot_canonical/collection_manifest.jsonl}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-26900}"

echo "[pointgoal-v2] validating exported expert dataset"
"${PYTHON_BIN}" -m nav.scripts.validate_pointgoal_dataset "${DATA_ROOT}" \
  --min-train-episodes 100 \
  --min-val-episodes 20 \
  --min-test-episodes 20 \
  --require-all-actions

echo "[pointgoal-v2] training ${TRAIN_EPOCHS} epochs"
"${PYTHON_BIN}" -m nav.scripts.train_bc \
  --base resnet50 \
  --data_root "${DATA_ROOT}" \
  --output_dir "${TRAIN_OUTPUT}" \
  --epochs "${TRAIN_EPOCHS}" \
  --num_workers 8 \
  --goal_rep polar \
  --goal_distance_scale_m 50 \
  --horizontal_flip_prob 0.5 \
  --class_weight_power 0.5 \
  --chunk_size 1 \
  --pretrained_depth \
  --sequence_action_offset 0 \
  --include_stop_targets

echo "[pointgoal-v2] evaluating one episode in each held-out scene"
"${PYTHON_BIN}" -m nav.scripts.evaluate_pointgoal_policy \
  --manifest "${MANIFEST}" \
  --checkpoint "${TRAIN_OUTPUT}/best.pt" \
  --output-root "${EVAL_OUTPUT}" \
  --unity "${UNITY_CLIENT}" \
  --python "${PYTHON_BIN}" \
  --scenes 4 \
  --base-port "${EVAL_BASE_PORT}"

echo "[pointgoal-v2] complete"
