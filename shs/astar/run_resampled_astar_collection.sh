#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-datasets/astar_pointgoal_expanded_v3/bc}"
MANIFEST_DIR="${MANIFEST_DIR:-datasets/astar_pointgoal_resampled_v1}"
MANIFEST="${MANIFEST:-${MANIFEST_DIR}/collection_manifest.jsonl}"
RAW_ROOT="${RAW_ROOT:-${MANIFEST_DIR}/raw}"
DATA_ROOT="${DATA_ROOT:-${MANIFEST_DIR}/bc}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
PAIRS_PER_SCENE="${PAIRS_PER_SCENE:-12}"
WORKERS="${WORKERS:-12}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
BASE_PORT="${BASE_PORT:-33000}"
export INDUSTRYNAV_UNITY_TIMEOUT_SECONDS="${INDUSTRYNAV_UNITY_TIMEOUT_SECONDS:-300}"

echo "[resampled-pointgoal] generate ${PAIRS_PER_SCENE} independent pairs for all 24 scenes"
"${PYTHON_BIN}" -m nav.scripts.tools.sample_resampled_pointgoal_pairs \
  --trajectory-root "${SOURCE_DATA_ROOT}" \
  --benchmark-points input_points.json \
  --output "${MANIFEST}" \
  --pairs-per-scene "${PAIRS_PER_SCENE}"

echo "[resampled-pointgoal] collect A* demonstrations with ${WORKERS} Unity workers"
"${PYTHON_BIN}" -m nav.scripts.astar.collect_astar_pointgoal \
  --manifest "${MANIFEST}" \
  --output-root "${RAW_ROOT}" \
  --unity "${UNITY_CLIENT}" \
  --workers "${WORKERS}" \
  --gpu-ids "${GPU_IDS}" \
  --base-port "${BASE_PORT}" \
  --resume \
  --retry-failed

echo "[resampled-pointgoal] export successful demonstrations"
"${PYTHON_BIN}" -m nav.scripts.astar.export_astar_pointgoal_dataset \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --overwrite

echo "[resampled-pointgoal] collection complete"
