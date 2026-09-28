#!/usr/bin/env bash
# Run the three missing cells of the fixed 32-task Gemini/GLM 2x2 ablation.
# The memory-on/map-off cell comes from the original outputs and is not rerun.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
TASKS_FILE="${TASKS_FILE:-${REPO_ROOT}/analysis/llm_memory_topdown_ablation_32_20260914/tasks_32.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/llm_memory_topdown_ablation_32_20260914}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/analysis/llm_memory_topdown_ablation_32_20260914}"
FILE_NAME="${FILE_NAME:-auto}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-60000}"
MAX_RETRIES="${MAX_RETRIES:-2}"
PER_CELL_TIMEOUT_SEC="${PER_CELL_TIMEOUT_SEC:-1800}"
LLM_MIN_REQUEST_INTERVAL_SEC="${LLM_MIN_REQUEST_INTERVAL_SEC:-0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODELS=(google/gemini-3.8-flash z-ai/glm-5.3-flash)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be exported before launching the ablation." >&2
  exit 1
fi

mkdir -p "$ANALYSIS_DIR"

run_condition() {
  local tag="$1"
  local history_size="$2"
  local topdown="$3"
  local mode_args=(--llm_min_request_interval_sec "$LLM_MIN_REQUEST_INTERVAL_SEC")
  if [[ "$RESUME" == "1" && "$DRY_RUN" != "1" ]]; then
    mode_args+=(--resume)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    mode_args+=(--dry_run)
  fi
  echo "[ablation] condition=${tag} history=${history_size} topdown=${topdown}"
  "$PYTHON_BIN" -m nav.scripts.agent.run_benchmark_grid \
    --models "${MODELS[@]}" \
    --seeds 0 \
    --history_sizes "$history_size" \
    --vision_input on \
    --topdown_input "$topdown" \
    --tasks_file "$TASKS_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --prompt_file "$REPO_ROOT/nav/prompts/nav_ego_state_history_kiro_v1.txt" \
    --topdown_prompt_file "$REPO_ROOT/nav/prompts/nav_ego_topdown_state_history_kiro_v1.txt" \
    --llm_provider openrouter \
    --max_tokens "$MAX_TOKENS" \
    --max_concurrency "$MAX_CONCURRENCY" \
    --max_retries "$MAX_RETRIES" \
    --per_cell_timeout_sec "$PER_CELL_TIMEOUT_SEC" \
    --file_name "$FILE_NAME" \
    --grid_csv "$ANALYSIS_DIR/${tag}_runs.csv" \
    --failures_csv "$ANALYSIS_DIR/${tag}_failures.csv" \
    "${mode_args[@]}"
}

# Complete the remaining three cells of the 2x2 design. Each invocation plans
# 64 episodes: two models x the same frozen 32 tasks.
run_condition memory_off_topdown_off 0 off
run_condition memory_off_topdown_on 0 on
run_condition memory_on_topdown_on 5 on

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[ablation] dry-run complete; planned 3 x 64 = 192 new episodes."
  exit 0
fi

"$PYTHON_BIN" -m nav.scripts.evaluation.summarize_llm_ablation --strict
