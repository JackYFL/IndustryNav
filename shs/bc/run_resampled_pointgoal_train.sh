#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-datasets/astar_pointgoal_resampled_v1/bc}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/astar_pointgoal_resampled_v1_fromscratch}"
EPOCHS="${EPOCHS:-16}"

# Intentionally train only on the independently resampled dataset.  There is
# no --init_checkpoint: old input_points-derived policy weights are not reused.
exec "${PYTHON_BIN}" -m nav.scripts.bc.train_bc \
  --base resnet50 \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --lr 0.0001 \
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
  --pretrained_depth \
  --sequence_action_offset 0 \
  --no-include_stop_targets \
  --navigation_only_actions
