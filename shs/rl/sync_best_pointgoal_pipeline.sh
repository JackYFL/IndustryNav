#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-action30}"
REMOTE_ROOT="${REMOTE_ROOT:-/egr/research-actionlab/liyifa11/MyCodes/IndustryNav}"
RSYNC_ARGS=(-a --partial)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_ARGS+=(-n --itemize-changes)
else
  RSYNC_ARGS+=(--progress)
fi

sync_dir() {
  local relative="$1"
  mkdir -p "${relative}"
  rsync "${RSYNC_ARGS[@]}" "${REMOTE_HOST}:${REMOTE_ROOT}/${relative}/" "${relative}/"
}

sync_file() {
  local relative="$1"
  mkdir -p "$(dirname "${relative}")"
  rsync "${RSYNC_ARGS[@]}" "${REMOTE_HOST}:${REMOTE_ROOT}/${relative}" "${relative}"
}

# Preserve both data sources because the aggregate contains relative symlinks.
sync_dir datasets/astar_pointgoal_resampled_v1/bc
sync_dir datasets/dagger_pointgoal_long_v1_all24_round1
sync_dir datasets/astar_pointgoal_dagger_long_v1_all24_round1
sync_dir ckpts/astar_pointgoal_dagger_long_v1_all24_round1
sync_dir experiments/ppo_dagger52_20260912/nav
sync_dir experiments/ppo80_v8_20260913/nav
sync_file experiments/ppo80_v4_20260913/resume_after_unity255_update141/update_000141.pt
sync_file experiments/ppo80_v8_20260913/ppo_worker_cap_probe/update_000400.pt
sync_file experiments/ppo80_v8_20260913/ppo_worker_cap_probe/run_config.json
sync_file experiments/ppo80_v8_20260913/ppo_worker_cap_probe/metrics.jsonl

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${PYTHON_BIN:-.venv/bin/python}" -m nav.scripts.tools.verify_best_pointgoal_pipeline \
    --require-reference-outputs
fi
