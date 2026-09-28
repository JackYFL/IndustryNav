#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-datasets/astar_pointgoal_pilot_canonical/collection_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-datasets/astar_pointgoal_pilot_tuned_v1}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-ckpts/astar_pointgoal_pilot_tuned_v1_smoke}"
EVAL_OUTPUT="${EVAL_OUTPUT:-outputs/astar_pointgoal_pilot_tuned_v1_smoke_eval}"
WORKERS="${WORKERS:-4}"
BASE_PORT="${BASE_PORT:-25700}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-26700}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

mkdir -p "${OUTPUT_ROOT}"

echo "[pointgoal] collecting $(wc -l < "${MANIFEST}") episodes"
"${PYTHON_BIN}" -m nav.scripts.astar.collect_astar_pointgoal \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}/raw" \
  --unity "${UNITY_CLIENT}" \
  --workers "${WORKERS}" \
  --base-port "${BASE_PORT}"

echo "[pointgoal] exporting successful expert episodes"
"${PYTHON_BIN}" -m nav.scripts.astar.export_astar_pointgoal_dataset \
  --manifest "${MANIFEST}" \
  --raw-root "${OUTPUT_ROOT}/raw" \
  --output-root "${OUTPUT_ROOT}/bc" \
  --overwrite

echo "[pointgoal] validating dataset quality gate"
"${PYTHON_BIN}" -m nav.scripts.bc.validate_pointgoal_dataset \
  "${OUTPUT_ROOT}/bc" \
  --min-train-episodes 100 \
  --min-val-episodes 20 \
  --min-test-episodes 20 \
  --require-all-actions

echo "[pointgoal] training ${TRAIN_EPOCHS}-epoch smoke policy"
"${PYTHON_BIN}" -m nav.scripts.bc.train_bc \
  --base resnet50 \
  --data_root "${OUTPUT_ROOT}/bc" \
  --output_dir "${TRAIN_OUTPUT}" \
  --epochs "${TRAIN_EPOCHS}" \
  --num_workers 8 \
  --goal_rep polar \
  --goal_distance_scale_m 50 \
  --horizontal_flip_prob 0.5 \
  --class_weight_power 0.5 \
  --chunk_size 1 \
  --pretrained_depth \
  --sequence_action_offset 0 \
  --include_stop_targets

echo "[pointgoal] evaluating on held-out scenes"
"${PYTHON_BIN}" -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --manifest "${MANIFEST}" \
  --checkpoint "${TRAIN_OUTPUT}/best.pt" \
  --output-root "${EVAL_OUTPUT}" \
  --unity "${UNITY_CLIENT}" \
  --python "${PYTHON_BIN}" \
  --scenes 4 \
  --base-port "${EVAL_BASE_PORT}"

echo "[pointgoal] pipeline complete"
