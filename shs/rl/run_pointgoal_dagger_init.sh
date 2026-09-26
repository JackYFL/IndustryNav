#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${INIT_PPO_CHECKPOINT:-}" ]]; then
  echo "INIT_PPO_CHECKPOINT must point to a PPO-compatible actor checkpoint" >&2
  exit 2
fi

# Train the PPO architecture as a DAgger learner. The learned policy executes
# every rollout action; A* only supplies labels on the states it visits.
export ONLINE_ASTAR_TEACHER=1
export POLICY_LOSS_COEF=0
export ROLLOUT_ACTION_MODE=argmax
export TEACHER_BC_COEF="${TEACHER_BC_COEF:-1.0}"
export TEACHER_BC_DECAY_UPDATES="${TEACHER_BC_DECAY_UPDATES:-0}"
export TEACHER_BC_MIN_SCALE="${TEACHER_BC_MIN_SCALE:-1.0}"
export TEACHER_ACTION_PROB=0
export TEACHER_ACTION_MIN_PROB=0
export TEACHER_ACTION_DECAY_UPDATES=0
export TEACHER_CLASS_BALANCE="${TEACHER_CLASS_BALANCE:-1}"
export TEACHER_REPLAY_CAPACITY="${TEACHER_REPLAY_CAPACITY:-32768}"
export TEACHER_REPLAY_STEPS="${TEACHER_REPLAY_STEPS:-2}"
export TEACHER_REPLAY_BATCH_SIZE="${TEACHER_REPLAY_BATCH_SIZE:-128}"
export TEACHER_REPLAY_COEF="${TEACHER_REPLAY_COEF:-1.0}"
export TEACHER_REPLAY_CLASS_WEIGHT_POWER="${TEACHER_REPLAY_CLASS_WEIGHT_POWER:-0.5}"

exec bash "${SCRIPT_DIR}/run_pointgoal_ddppo.sh" \
  --entropy-coef 0 \
  --value-loss-coef 0 \
  "$@"
