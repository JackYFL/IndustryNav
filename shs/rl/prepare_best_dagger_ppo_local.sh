#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-ckpts/astar_pointgoal_dagger_long_v1_all24_round1/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/local_best_pointgoal/initialization}"

CMD=(
  "${PYTHON_BIN}" -m nav.scripts.rl.prepare_dagger_ppo
  --checkpoint "${CHECKPOINT}"
  --manifest datasets/pointgoal_dagger_resampled_v1/collection_manifest.jsonl
  --manifest datasets/pointgoal_dagger_long_v1/collection_manifest.jsonl
  --manifest datasets/pointgoal_dagger_dense_v2/collection_manifest.jsonl
  --input-points input_points.json
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE:-cpu}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Refusing to overwrite existing conversion output: ${OUTPUT_DIR}" >&2
  exit 2
}
exec "${CMD[@]}" "$@"
