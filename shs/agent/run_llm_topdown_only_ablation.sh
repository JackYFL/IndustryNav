#!/usr/bin/env bash
# Additional 32-task-per-model minimap-only evaluation; never rerun the 2x2 grid.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXPERIMENT_DIR="${REPO_ROOT}/analysis/llm_memory_topdown_ablation_32_20260914"
TASKS_FILE="${TASKS_FILE:-${EXPERIMENT_DIR}/tasks_32.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/llm_memory_topdown_ablation_32_20260914}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EXPERIMENT_DIR}/topdown_only_20260915}"
FILE_NAME="${FILE_NAME:-auto}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-60000}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PER_CELL_TIMEOUT_SEC="${PER_CELL_TIMEOUT_SEC:-7200}"
MAX_REQUESTS="${MAX_REQUESTS:-6000}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "${CONFIRM_TOPDOWN_ONLY_API:-0}" != "1" ]]; then
    echo "New paid evaluation requires separate user authorization before launch." >&2
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
# A dedicated persistent counter includes provider retries across both models.
# Resume retains this counter; it must not be deleted or reset.
export OPENROUTER_MAX_REQUESTS="$MAX_REQUESTS"
export OPENROUTER_REQUEST_COUNTER_FILE="$ANALYSIS_DIR/openrouter_requests.sqlite3"

"$PYTHON_BIN" -m nav.scripts.agent.run_benchmark_grid \
  --models google/gemini-3.8-flash z-ai/glm-5.3-flash \
  --seeds 0 --history_sizes 0 --vision_input off --topdown_input on \
  --tasks_file "$TASKS_FILE" --output_root "$OUTPUT_ROOT" \
  --topdown_prompt_file "$REPO_ROOT/nav/prompts/nav_minimap_only.txt" \
  --llm_provider openrouter --max_tokens "$MAX_TOKENS" \
  --max_concurrency "$MAX_CONCURRENCY" --max_retries "$MAX_RETRIES" \
  --per_cell_timeout_sec "$PER_CELL_TIMEOUT_SEC" --file_name "$FILE_NAME" \
  --grid_csv "$ANALYSIS_DIR/topdown_only_runs.csv" \
  --failures_csv "$ANALYSIS_DIR/topdown_only_failures.csv" \
  "${mode_args[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[topdown-only] dry-run complete; planned 2 x 32 = 64 new episodes. No API calls."
  exit 0
fi
"$PYTHON_BIN" -m nav.scripts.evaluation.summarize_llm_ablation \
  --include-topdown-only --strict \
  --tasks-file "$TASKS_FILE" --ablation-root "$OUTPUT_ROOT" \
  --output-dir "$ANALYSIS_DIR" \
  --original-reference-csv "$EXPERIMENT_DIR/original_reference.csv"
