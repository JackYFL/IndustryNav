#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/pointgoal_dagger_resampled_v1/collection_manifest.jsonl}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/pointgoal_ddppo_resampled_v1}"
INIT_BC_CHECKPOINT="${INIT_BC_CHECKPOINT:-ckpts/astar_pointgoal_resampled_v1_fromscratch/best.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NUM_ENVS_PER_RANK="${NUM_ENVS_PER_RANK:-4}"
TOTAL_UPDATES="${TOTAL_UPDATES:-200}"
BASE_PORT="${BASE_PORT:-45000}"
# Set INDUSTRYNAV_UNITY_DEVICE_IDS to physical graphics-device indices, such
# as "4,5". Each torch rank selects its corresponding entry while CUDA uses
# the devices made visible through CUDA_VISIBLE_DEVICES.

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${NPROC_PER_NODE}" \
  -m nav.scripts.rl.train_pointgoal_ppo \
  --manifest "${MANIFEST}"
)
if [[ -n "${EXTRA_MANIFEST:-}" ]]; then
  CMD+=(--extra-manifest "${EXTRA_MANIFEST}")
fi
if [[ -n "${TEACHER_BC_CHECKPOINT:-}" ]]; then
  CMD+=(--teacher-bc-checkpoint "${TEACHER_BC_CHECKPOINT}")
fi
if [[ "${ONLINE_ASTAR_TEACHER:-0}" == "1" ]]; then
  CMD+=(--online-astar-teacher)
fi
if [[ "${GEODESIC_PROGRESS_REWARD:-0}" == "1" ]]; then
  CMD+=(--geodesic-progress-reward)
fi
if [[ "${TEACHER_CLASS_BALANCE:-0}" == "1" ]]; then
  CMD+=(--teacher-class-balance)
fi
if [[ -n "${INIT_PPO_CHECKPOINT:-}" ]]; then
  CMD+=(--init-ppo-checkpoint "${INIT_PPO_CHECKPOINT}")
else
  CMD+=(--init-bc-checkpoint "${INIT_BC_CHECKPOINT}")
fi
if [[ "${ABSOLUTE_SCENE_STATE:-0}" == "1" ]]; then
  CMD+=(--absolute-scene-state)
fi
CMD+=(
  --unity "${UNITY_CLIENT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT:-train}" \
  --num-envs "${NUM_ENVS_PER_RANK}" \
  --total-updates "${TOTAL_UPDATES}" \
  --base-port "${BASE_PORT}" \
  --device "${DEVICE:-cuda}" \
  --rollout-steps "${ROLLOUT_STEPS:-32}" \
  --bptt-len "${BPTT_LEN:-8}" \
  --lr "${LR:-0.0001}" \
  --encoder-lr-scale "${ENCODER_LR_SCALE:-0.1}" \
  --policy-lr-scale "${POLICY_LR_SCALE:-2.5}" \
  --teacher-bc-coef "${TEACHER_BC_COEF:-0.0}" \
  --teacher-bc-decay-updates "${TEACHER_BC_DECAY_UPDATES:-200}" \
  --teacher-bc-min-scale "${TEACHER_BC_MIN_SCALE:-0.1}" \
  --teacher-action-prob "${TEACHER_ACTION_PROB:-0.0}" \
  --teacher-action-decay-updates "${TEACHER_ACTION_DECAY_UPDATES:-100}" \
  --teacher-action-min-prob "${TEACHER_ACTION_MIN_PROB:-0.0}" \
  --teacher-replay-capacity "${TEACHER_REPLAY_CAPACITY:-0}" \
  --teacher-replay-steps "${TEACHER_REPLAY_STEPS:-0}" \
  --teacher-replay-batch-size "${TEACHER_REPLAY_BATCH_SIZE:-64}" \
  --teacher-replay-coef "${TEACHER_REPLAY_COEF:-1.0}" \
  --teacher-replay-class-weight-power "${TEACHER_REPLAY_CLASS_WEIGHT_POWER:-0.5}" \
  --astar-obstacle-clearance-m "${ASTAR_OBSTACLE_CLEARANCE_M:-0.6}" \
  --astar-policy-forward-tolerance-deg "${ASTAR_POLICY_FORWARD_TOLERANCE_DEG:-12.5}" \
  --action-history-len "${ACTION_HISTORY_LEN:-5}" \
  --visual-history-len "${VISUAL_HISTORY_LEN:-5}" \
  --coordinate-fourier-bands "${COORDINATE_FOURIER_BANDS:-0}" \
  --route-actor-hidden "${ROUTE_ACTOR_HIDDEN:-0}" \
  --base-policy-logit-scale "${BASE_POLICY_LOGIT_SCALE:-1.0}" \
  --route-actor-logit-scale "${ROUTE_ACTOR_LOGIT_SCALE:-1.0}" \
  --ppo-epochs "${PPO_EPOCHS:-2}" \
  --chunks-per-minibatch "${CHUNKS_PER_MINIBATCH:-8}" \
  --policy-loss-coef "${POLICY_LOSS_COEF:-1.0}" \
  --rollout-action-mode "${ROLLOUT_ACTION_MODE:-sample}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-200}" \
  --task-sampling "${TASK_SAMPLING:-random}" \
  --scene-block-episodes "${SCENE_BLOCK_EPISODES:-4}" \
  --scene-limit "${SCENE_LIMIT:-16}" \
  --progress-scale "${PROGRESS_SCALE:-1.5}" \
  --success-bonus "${SUCCESS_BONUS:-10.0}" \
  --timeout-penalty "${TIMEOUT_PENALTY:-2.0}" \
  --collision-penalty "${COLLISION_PENALTY:-1.0}" \
  --collision-step-penalty "${COLLISION_STEP_PENALTY:-0.0}" \
  --warning-penalty "${WARNING_PENALTY:-0.02}" \
  --rotation-penalty "${ROTATION_PENALTY:-0.005}" \
  --turn-reversal-penalty "${TURN_REVERSAL_PENALTY:-0.05}" \
  --stagnation-penalty "${STAGNATION_PENALTY:-0.02}" \
  --stagnation-start-steps "${STAGNATION_START_STEPS:-12}" \
  --stagnation-progress-epsilon-m "${STAGNATION_PROGRESS_EPSILON_M:-0.05}" \
  --safe-forward-bonus "${SAFE_FORWARD_BONUS:-0.02}" \
  --stuck-recovery-steps "${STUCK_RECOVERY_STEPS:-48}" \
  --stuck-recovery-penalty "${STUCK_RECOVERY_PENALTY:-2.0}"
)

exec "${CMD[@]}" "$@"
