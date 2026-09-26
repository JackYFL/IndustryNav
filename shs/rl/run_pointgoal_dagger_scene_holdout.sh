#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# The Linux Unity player needs a display because camera observations are blank
# under -nographics and this build can SIGSEGV without an X server.
if [[ "$(uname -s)" == "Linux" && -z "${DISPLAY:-}" && "${INDUSTRYNAV_XVFB_WRAPPED:-0}" != "1" ]]; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "xvfb-run is required for Linux scene-holdout training" >&2
    exit 2
  fi
  exec env INDUSTRYNAV_XVFB_WRAPPED=1 xvfb-run -a \
    -s "${XVFB_SCREEN:--screen 0 1724x1024x24}" bash "$0" "$@"
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-datasets/pointgoal_dagger_dense_v2/collection_manifest.jsonl}"
INPUT_POINTS="${INPUT_POINTS:-input_points.json}"
UNITY_CLIENT="${UNITY_CLIENT:-clients/scene_all_24scenes_exact_pillar_markers_v14/scene_all.x86_64}"
TRAIN_SCENE_START="${TRAIN_SCENE_START:-1}"
TRAIN_SCENE_END="${TRAIN_SCENE_END:-16}"
TEST_SCENE_START="${TEST_SCENE_START:-17}"
TEST_SCENE_END="${TEST_SCENE_END:-24}"
TOTAL_UPDATES="${TOTAL_UPDATES:-100}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NUM_ENVS_PER_RANK="${NUM_ENVS_PER_RANK:-8}"
BASE_PORT="${BASE_PORT:-65000}"
EVAL_BASE_PORT="${EVAL_BASE_PORT:-54000}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/pointgoal_dagger_scene01_16_holdout17_24}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/pointgoal_dagger_scene01_16_holdout17_24}"
SPLIT_DIR="${SPLIT_DIR:-${OUTPUT_DIR}/splits}"
TRAIN_MANIFEST="${SPLIT_DIR}/train_scene${TRAIN_SCENE_START}_${TRAIN_SCENE_END}.jsonl"
TEST_INPUT_POINTS="${SPLIT_DIR}/input_points_scene${TEST_SCENE_START}_${TEST_SCENE_END}.json"

if [[ ! -x "${PYTHON_BIN}" || ! -f "${SOURCE_MANIFEST}" || ! -f "${INPUT_POINTS}" || ! -x "${UNITY_CLIENT}" ]]; then
  echo "Python, source manifest, input points, and Unity client must exist" >&2
  exit 2
fi
if (( TRAIN_SCENE_START < 1 || TRAIN_SCENE_END < TRAIN_SCENE_START )); then
  echo "Invalid training scene range" >&2
  exit 2
fi
if (( TEST_SCENE_START < 1 || TEST_SCENE_END < TEST_SCENE_START )); then
  echo "Invalid test scene range" >&2
  exit 2
fi
if (( TRAIN_SCENE_END >= TEST_SCENE_START && TEST_SCENE_END >= TRAIN_SCENE_START )); then
  echo "Training and test scene ranges must not overlap" >&2
  exit 2
fi

mkdir -p "${SPLIT_DIR}" "${OUTPUT_DIR}" "${EVAL_OUTPUT_ROOT}"

"${PYTHON_BIN}" - "${SOURCE_MANIFEST}" "${TRAIN_MANIFEST}" \
  "${TRAIN_SCENE_START}" "${TRAIN_SCENE_END}" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
start, end = map(int, sys.argv[3:5])
records = []
for line in source.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    scene_number = int(str(record["scene_name"]).removeprefix("scene"))
    if start <= scene_number <= end:
        records.append(record)
destination.write_text(
    "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    encoding="utf-8",
)
if not records:
    raise SystemExit("Filtered training manifest is empty")
print(f"Prepared {len(records)} training point pairs in {destination}")
PY

"${PYTHON_BIN}" - "${INPUT_POINTS}" "${TEST_INPUT_POINTS}" \
  "${TEST_SCENE_START}" "${TEST_SCENE_END}" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
start, end = map(int, sys.argv[3:5])
points = json.loads(source.read_text(encoding="utf-8"))
selected = {
    name: entries
    for name, entries in points.items()
    if start <= int(name.removeprefix("scene")) <= end
}
destination.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if len(selected) != end - start + 1:
    raise SystemExit("Filtered input points do not cover every requested test scene")
print(
    f"Prepared {sum(map(len, selected.values()))} held-out input points "
    f"across {len(selected)} scenes in {destination}"
)
PY

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${NPROC_PER_NODE}" \
  -m nav.scripts.rl.train_pointgoal_ppo \
  --manifest "${TRAIN_MANIFEST}" \
  --unity "${UNITY_CLIENT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split all \
  --num-envs "${NUM_ENVS_PER_RANK}" \
  --total-updates "${TOTAL_UPDATES}" \
  --base-port "${BASE_PORT}" \
  --device cuda \
  --rollout-steps "${ROLLOUT_STEPS:-64}" \
  --bptt-len "${BPTT_LEN:-8}" \
  --lr "${LR:-0.00005}" \
  --encoder-lr-scale "${ENCODER_LR_SCALE:-1.0}" \
  --policy-lr-scale "${POLICY_LR_SCALE:-2.0}" \
  --online-astar-teacher \
  --teacher-bc-coef "${TEACHER_BC_COEF:-1.0}" \
  --teacher-bc-decay-updates 0 \
  --teacher-bc-min-scale 1.0 \
  --teacher-action-prob 0 \
  --teacher-action-min-prob 0 \
  --teacher-action-decay-updates 0 \
  --teacher-class-balance \
  --teacher-replay-capacity "${TEACHER_REPLAY_CAPACITY:-32768}" \
  --teacher-replay-steps "${TEACHER_REPLAY_STEPS:-2}" \
  --teacher-replay-batch-size "${TEACHER_REPLAY_BATCH_SIZE:-128}" \
  --teacher-replay-coef "${TEACHER_REPLAY_COEF:-1.0}" \
  --teacher-replay-class-weight-power "${TEACHER_REPLAY_CLASS_WEIGHT_POWER:-0.5}" \
  --astar-obstacle-clearance-m "${ASTAR_OBSTACLE_CLEARANCE_M:-0.6}" \
  --astar-policy-forward-tolerance-deg "${ASTAR_POLICY_FORWARD_TOLERANCE_DEG:-20.0}" \
  --action-history-len "${ACTION_HISTORY_LEN:-5}" \
  --visual-history-len "${VISUAL_HISTORY_LEN:-5}" \
  --ppo-epochs "${PPO_EPOCHS:-1}" \
  --chunks-per-minibatch "${CHUNKS_PER_MINIBATCH:-8}" \
  --policy-loss-coef 0 \
  --rollout-action-mode argmax \
  --max-episode-steps "${MAX_EPISODE_STEPS:-300}" \
  --task-sampling random \
  --scene-block-episodes "${SCENE_BLOCK_EPISODES:-8}" \
  --entropy-coef 0 \
  --value-loss-coef 0 \
  --weight-decay "${WEIGHT_DECAY:-0}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-10}"

FINAL_UPDATE="$(${PYTHON_BIN} -c \
  'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("update", -1))' \
  "${OUTPUT_DIR}/latest.pt")"
if [[ "${FINAL_UPDATE}" != "${TOTAL_UPDATES}" ]]; then
  echo "DAgger stopped at update ${FINAL_UPDATE}; expected ${TOTAL_UPDATES}" >&2
  exit 1
fi

TEST_SCENE_COUNT=$((TEST_SCENE_END - TEST_SCENE_START + 1))
TEST_EPISODE_COUNT="$(${PYTHON_BIN} -c \
  'import json, sys; print(sum(map(len, json.load(open(sys.argv[1])).values())))' \
  "${TEST_INPUT_POINTS}")"
if (( TEST_EPISODE_COUNT % TEST_SCENE_COUNT != 0 )); then
  echo "Held-out input points are not balanced across scenes" >&2
  exit 1
fi

INDUSTRYNAV_UNITY_DEVICE_INDEX="${EVAL_UNITY_DEVICE_INDEX:-${INDUSTRYNAV_UNITY_DEVICE_INDEX:-0}}" \
"${PYTHON_BIN}" -m nav.scripts.evaluation.evaluate_pointgoal_policy \
  --baseline ppo \
  --model-id "${MODEL_ID:-pointgoal-dagger-train01-16-test17-24}" \
  --input-points "${TEST_INPUT_POINTS}" \
  --checkpoint "${OUTPUT_DIR}/latest.pt" \
  --output-root "${EVAL_OUTPUT_ROOT}" \
  --unity "${UNITY_CLIENT}" \
  --device cuda \
  --eval-seed "${EVAL_SEED:-0}" \
  --base-port "${EVAL_BASE_PORT}" \
  --scenes "${TEST_SCENE_COUNT}" \
  --episodes-per-scene "$((TEST_EPISODE_COUNT / TEST_SCENE_COUNT))" \
  --workers "${EVAL_WORKERS:-8}" \
  --persistent-policy

echo "Experiment complete: ${EVAL_OUTPUT_ROOT}/summary.json"
