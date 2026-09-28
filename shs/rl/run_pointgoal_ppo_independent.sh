#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
: "${NAVIGATION_MAP_DIR:?Calibrated training-only scene maps are required}"
: "${REFERENCE_ACTOR:?Frozen initialization checkpoint is required}"
# INITIAL_ACTOR, MANIFEST, UNITY_CLIENT and OUTPUT_DIR are checked by the
# shared full-actor launcher. No benchmark task file is read by training.
export TOTAL_UPDATES="${TOTAL_UPDATES:-500}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
export CRITIC_WARMUP_UPDATES="${CRITIC_WARMUP_UPDATES:-8}"

exec bash shs/rl/run_pointgoal_ppo_dagger_actor.sh \
  --navigation-map-dir "${NAVIGATION_MAP_DIR}" \
  --reference-policy-checkpoint "${REFERENCE_ACTOR}" \
  --reference-kl-coef "${REFERENCE_KL_COEF:-0.05}" \
  --reference-kl-final-coef "${REFERENCE_KL_FINAL_COEF:-0.01}" \
  --reference-kl-decay-updates "${REFERENCE_KL_DECAY_UPDATES:-400}" \
  --parallel-env-steps --training-safety-shield --dynamic-step-budget --resample-invalid-spawns \
  --gae-lambda 0.98 --stuck-recovery-steps 100 "$@"
