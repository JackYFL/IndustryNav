#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${WAIT_PID:?WAIT_PID must identify the running DAgger launcher}"
: "${DAGGER_CHECKPOINT:?DAGGER_CHECKPOINT must point to its final checkpoint}"

EXPECTED_DAGGER_UPDATE="${EXPECTED_DAGGER_UPDATE:-100}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

if [[ ! -f "${DAGGER_CHECKPOINT}" ]]; then
  echo "DAgger checkpoint is missing: ${DAGGER_CHECKPOINT}" >&2
  exit 1
fi
ACTUAL_DAGGER_UPDATE="$(${PYTHON_BIN} -c \
  'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("update", -1))' \
  "${DAGGER_CHECKPOINT}")"
if [[ "${ACTUAL_DAGGER_UPDATE}" != "${EXPECTED_DAGGER_UPDATE}" ]]; then
  echo "DAgger stopped at update ${ACTUAL_DAGGER_UPDATE}; expected ${EXPECTED_DAGGER_UPDATE}" >&2
  exit 1
fi

# The second stage is ordinary stochastic PPO. No teacher, expert action,
# replay supervision, geodesic A* reward, or evaluation fallback is enabled.
export INIT_PPO_CHECKPOINT="${DAGGER_CHECKPOINT}"
export TEACHER_BC_CHECKPOINT=
export ONLINE_ASTAR_TEACHER=0
export GEODESIC_PROGRESS_REWARD=0
export TEACHER_CLASS_BALANCE=0
export TEACHER_BC_COEF=0
export TEACHER_ACTION_PROB=0
export TEACHER_REPLAY_CAPACITY=0
export TEACHER_REPLAY_STEPS=0
export POLICY_LOSS_COEF="${POLICY_LOSS_COEF:-1.0}"
export ROLLOUT_ACTION_MODE="${ROLLOUT_ACTION_MODE:-sample}"

exec xvfb-run -a -s "${XVFB_SCREEN:--screen 0 1724x1024x24}" \
  bash "${SCRIPT_DIR}/run_pointgoal_ddppo.sh" "$@"
