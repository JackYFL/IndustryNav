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
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
INPUT_POINTS="${INPUT_POINTS:-input_points.json}"
MODEL_ID="${MODEL_ID:-pointgoal-dagger-ppoarch}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/pointgoal_dagger_ppoarch_input_points}"
GALLERY_OUTPUT_DIR="${GALLERY_OUTPUT_DIR:-analysis/pointgoal_dagger_ppoarch_gallery}"
TRAJECTORY_GALLERY_OUTPUT_DIR="${TRAJECTORY_GALLERY_OUTPUT_DIR:-analysis/pointgoal_dagger_ppoarch_trajectory_gallery}"

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

"${PYTHON_BIN}" -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --baseline ppo \
  --model-id "${MODEL_ID}" \
  --input-points "${INPUT_POINTS}" \
  --checkpoint "${DAGGER_CHECKPOINT}" \
  --output-root "${EVAL_OUTPUT_ROOT}" \
  --unity "${UNITY_CLIENT}" \
  --device cuda \
  --eval-seed "${EVAL_SEED:-0}" \
  --base-port "${EVAL_BASE_PORT:-60000}" \
  --scenes 24 \
  --episodes-per-scene 4 \
  --workers "${EVAL_WORKERS:-4}"

"${PYTHON_BIN}" -m nav.scripts.gallery.export_llm_gallery \
  --input-glob "${EVAL_OUTPUT_ROOT}/scene*/point*" \
  --output-dir "${GALLERY_OUTPUT_DIR}" \
  --workers "${GALLERY_WORKERS:-4}" \
  --normal-only \
  --gallery-title "${GALLERY_TITLE:-PointGoal DAgger (PPO Architecture)}"

"${PYTHON_BIN}" -m nav.scripts.gallery.export_topdown_comparison_gallery \
  --source-manifest "${GALLERY_OUTPUT_DIR}/manifest.json" \
  --output-dir "${TRAJECTORY_GALLERY_OUTPUT_DIR}" \
  --workers "${GALLERY_WORKERS:-4}"
