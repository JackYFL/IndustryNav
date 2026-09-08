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

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${NPROC_PER_NODE}" \
  -m nav.scripts.rl.train_pointgoal_ppo \
  --manifest "${MANIFEST}" \
  --unity "${UNITY_CLIENT}" \
  --output-dir "${OUTPUT_DIR}" \
  --init-bc-checkpoint "${INIT_BC_CHECKPOINT}" \
  --num-envs "${NUM_ENVS_PER_RANK}" \
  --total-updates "${TOTAL_UPDATES}" \
  --base-port "${BASE_PORT}" \
  --rollout-steps "${ROLLOUT_STEPS:-32}" \
  --bptt-len "${BPTT_LEN:-8}" \
  --ppo-epochs "${PPO_EPOCHS:-2}" \
  --chunks-per-minibatch "${CHUNKS_PER_MINIBATCH:-8}" \
  --max-episode-steps "${MAX_EPISODE_STEPS:-200}" \
  --scene-limit "${SCENE_LIMIT:-16}" \
  --collision-penalty "${COLLISION_PENALTY:-0.5}" \
  --warning-penalty "${WARNING_PENALTY:-0.05}" \
  "$@"
