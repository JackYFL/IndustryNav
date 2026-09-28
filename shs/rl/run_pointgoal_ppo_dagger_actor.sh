#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
: "${INITIAL_ACTOR:?Path to the full converted DAgger actor is required}"
: "${MANIFEST:?Audited non-benchmark training manifest is required}"
: "${UNITY_CLIENT:?Unity executable is required}"
: "${OUTPUT_DIR:?Use a new output directory for each PPO experiment}"

NPROC="${NPROC_PER_NODE:-2}"
if [[ "${NPROC}" == "1" ]]; then
  # A single worker does not need a rendezvous.  Avoiding torchrun here also
  # works around PyTorch's localhost:0 TCPStore issue on macOS.
  CMD=("${PYTHON_BIN:-.venv/bin/python}" -m nav.scripts.rl.train_pointgoal_ppo)
else
  CMD=(
    "${PYTHON_BIN:-.venv/bin/python}" -m torch.distributed.run
    --standalone --nproc-per-node "${NPROC}"
    -m nav.scripts.rl.train_pointgoal_ppo
  )
fi

CMD+=(
  --manifest "${MANIFEST}" --split train --scene-limit 24
  --init-ppo-checkpoint "${INITIAL_ACTOR}"
  --unity "${UNITY_CLIENT}" --output-dir "${OUTPUT_DIR}"
  --num-envs "${NUM_ENVS_PER_RANK:-12}" --device "${DEVICE:-cuda}"
  --base-port "${BASE_PORT:-47000}"
  --total-updates "${TOTAL_UPDATES:-200}" --rollout-steps "${ROLLOUT_STEPS:-64}"
  --bptt-len 1 --chunks-per-minibatch 64 --ppo-epochs 2
  --lr "${CRITIC_LR:-0.0001}" --encoder-lr-scale 0.1
  --policy-lr-scale "${POLICY_LR_SCALE:-0.01}"
  --critic-warmup-updates "${CRITIC_WARMUP_UPDATES:-4}"
  --target-kl 0.01 --clip-param 0.1 --entropy-coef 0.001
  --gamma 0.995 --gae-lambda 0.95 --max-grad-norm 0.5
  --max-episode-steps 320 --stuck-recovery-steps 80
  --task-sampling random --scene-block-episodes 4
  --progress-scale 1.0 --success-bonus 10 --timeout-penalty 1
  --step-penalty 0.005 --collision-penalty 0.25
  --collision-step-penalty 0.1 --warning-penalty 0.005
  --rotation-penalty 0 --turn-reversal-penalty 0.005
  --stagnation-penalty 0.005 --stagnation-start-steps 16
  --stuck-recovery-penalty 1 --safe-forward-bonus 0
  --teacher-bc-coef 0 --teacher-action-prob 0 --teacher-replay-capacity 0
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-10}" --seed "${SEED:-20260912}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}" "$@"
