#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-datasets/astar_pointgoal_dagger_long_v1_all24_round1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpts/astar_pointgoal_dagger_long_v1_round1/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/astar_pointgoal_dagger_long_v1_all24_round1_local}"

CMD=(
  "${PYTHON_BIN}" -m nav.scripts.bc.train_bc
  --base resnet50
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --init_checkpoint "${INIT_CHECKPOINT}"
  --img_size 256
  --batch_size 8
  --num_workers "${NUM_WORKERS:-8}"
  --epochs "${EPOCHS:-6}"
  --lr 0.00003
  --weight_decay 0.0001
  --seed 42
  --state_mode relative
  --policy_type transformer
  --seq_len 12
  --num_layers 3
  --chunk_size 1
  --sequence_action_offset 0
  --no-include_stop_targets
  --navigation_only_actions
  --goal_rep polar
  --goal_encoding unity_egocentric_v2
  --goal_distance_scale_m 50
  --horizontal_flip_prob 0.5
  --previous_action_noise_prob 0.15
  --class_weight_power 0.5
  --label_smoothing 0
  --turn_aux_loss_weight 0.1
  --checkpoint_metric macro_nav_acc
  --goal_action_residual
  --rgb_backbone resnet50
  --depth_backbone resnet50
  --backbone_lr_scale 0.1
  --use_depth
  --no-use_rgb
  --normalize_rgb
  --pretrained_rgb
  --no-pretrained_depth
  --no-half_width
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing DAgger dataset: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${INIT_CHECKPOINT}" ]] || { echo "Missing DAgger initialization: ${INIT_CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Refusing to overwrite existing DAgger output: ${OUTPUT_DIR}" >&2
  exit 2
}

exec "${CMD[@]}" "$@"
