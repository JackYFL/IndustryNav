#!/usr/bin/env bash
# Gemini 3.8 Flash H3/H10 sweep on the frozen 32-task RGB+minimap cohort.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXPERIMENT_DIR="${REPO_ROOT}/analysis/llm_memory_topdown_ablation_32_20260914"
TASKS_FILE="${TASKS_FILE:-${EXPERIMENT_DIR}/tasks_32.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/llm_memory_topdown_ablation_32_20260914}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EXPERIMENT_DIR}/history_size_3_10_20260919}"
FILE_NAME="${FILE_NAME:-auto}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-60000}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PER_CELL_TIMEOUT_SEC="${PER_CELL_TIMEOUT_SEC:-7200}"
MAX_REQUESTS="${MAX_REQUESTS:-5000}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "${CONFIRM_GEMINI_HISTORY_API:-0}" != "1" ]]; then
    echo "Paid history-size evaluation requires explicit user authorization." >&2
    exit 1
  fi
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY must be exported before launch." >&2
    exit 1
  fi
fi

mkdir -p "$ANALYSIS_DIR"
mode_args=(--resume)
if [[ "$DRY_RUN" == "1" ]]; then
  mode_args=(--dry_run)
fi

# A dedicated persistent counter covers both history sizes and all provider
# requests. Resume keeps this file; never delete or reset it during the sweep.
export OPENROUTER_MAX_REQUESTS="$MAX_REQUESTS"
export OPENROUTER_REQUEST_COUNTER_FILE="$ANALYSIS_DIR/openrouter_requests.sqlite3"

"$PYTHON_BIN" -m nav.scripts.agent.run_benchmark_grid \
  --models google/gemini-3.8-flash \
  --seeds 0 --history_sizes 3 10 \
  --vision_input on --topdown_input on \
  --tasks_file "$TASKS_FILE" --output_root "$OUTPUT_ROOT" \
  --prompt_file "$REPO_ROOT/nav/prompts/nav_ego_state_history_kiro_v1.txt" \
  --topdown_prompt_file "$REPO_ROOT/nav/prompts/nav_ego_topdown_state_history_kiro_v1.txt" \
  --llm_provider openrouter --max_tokens "$MAX_TOKENS" \
  --max_concurrency "$MAX_CONCURRENCY" --max_retries "$MAX_RETRIES" \
  --per_cell_timeout_sec "$PER_CELL_TIMEOUT_SEC" --file_name "$FILE_NAME" \
  --grid_csv "$ANALYSIS_DIR/history_size_runs.csv" \
  --failures_csv "$ANALYSIS_DIR/history_size_failures.csv" \
  "${mode_args[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[history-ablation] dry-run complete: H3/H10 x 32 = 64 episodes; no API calls."
fi
