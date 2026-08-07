#!/usr/bin/env bash
set -euo pipefail

# Run the 6-task fixed-2.17s completion-check benchmark with Gemma 4 12B.
# Each task runs 11 rollouts; episode 0 is the warm-up and episodes 1-10 are
# used for latency / correctness statistics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

TEST_NUM=${TEST_NUM:-11}
SEED=${SEED:-10000}
RUN_TAG=${RUN_TAG:-"gemma4_12b_status_n10_$(date +%Y%m%d_%H%M%S)"}
OUT_ROOT=${OUT_ROOT:-"${WORK_ROOT}/profile_results/robotwin_multitask_${RUN_TAG}"}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/robotwin_eval"}
ROBOTWIN_EVAL_RESULT_ROOT=${ROBOTWIN_EVAL_RESULT_ROOT:-"${OUT_ROOT}/eval_results"}

TASKS=(
  "place_burger_fries:2.17"
  "place_cans_plasticbox:2.25"
  "put_bottles_dustbin:4.01"
  "stack_blocks_three:2.25"
  "stack_bowls_three:4.53"
  "place_dual_shoes:2.61"
)

mkdir -p "${OUT_ROOT}" "${RUNTIME_CACHE_ROOT}" "${ROBOTWIN_EVAL_RESULT_ROOT}"

for entry in "${TASKS[@]}"; do
  task_name=${entry%%:*}
  forced_boundary_s=${entry##*:}
  log_dir="${OUT_ROOT}/${task_name}"

  echo "=== ${task_name}: forced boundary ${forced_boundary_s}s ==="
  env \
    TEST_NUM="${TEST_NUM}" \
    SEED="${SEED}" \
    TASK_NAME="${task_name}" \
    TASK_CONFIG=demo_clean \
    INSTRUCTION_TYPE=unseen \
    HANDOFF_TRIGGER=forced_boundary \
    FORCED_BOUNDARY_S="${forced_boundary_s}" \
    MAX_FORCED_BOUNDARIES=1 \
    STOP_AFTER_ORACLE_HANDOFF=False \
    STOP_AFTER_MEASUREMENT_HANDOFF=False \
    HL_BACKEND=gemma4_vllm \
    HL_PROMPT_MODE=completion_status \
    HL_PLANNER_MODE=${HL_PLANNER_MODE:-vlm_plan_next} \
    HL_ASYNC_ENABLED=False \
    HL_FORCE_VISUAL_ON_DECISION=True \
    HL_RECENT_FRAMES_LENGTH=1 \
    HL_VLM_MAX_IMAGES=1 \
    HL_VLM_MAX_NEW_TOKENS=48 \
    GEMMA4_SERVER_URL=${GEMMA4_SERVER_URL:-http://127.0.0.1:8003} \
    GEMMA4_MODEL_NAME=${GEMMA4_MODEL_NAME:-gemma-4-12B-it} \
    GEMMA4_REQUEST_TIMEOUT_S=${GEMMA4_REQUEST_TIMEOUT_S:-120} \
    GEMMA4_JPEG_QUALITY=${GEMMA4_JPEG_QUALITY:-90} \
    GEMMA4_IMAGE_SELECTION=${GEMMA4_IMAGE_SELECTION:-latest} \
    GEMMA4_IMAGE_MAX_SIDE=${GEMMA4_IMAGE_MAX_SIDE:-0} \
    GEMMA4_OUTPUT_SCHEMA=${GEMMA4_OUTPUT_SCHEMA:-completion_status} \
    ROBOTWIN_EVAL_VIDEO_LOG=false \
    RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT}" \
    ROBOTWIN_EVAL_RESULT_ROOT="${ROBOTWIN_EVAL_RESULT_ROOT}" \
    LOG_DIR="${log_dir}" \
    bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh"
done
