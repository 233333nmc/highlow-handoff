#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

TEST_NUM=${TEST_NUM:-3}
SEED=${SEED:-10000}
RUN_TAG=${RUN_TAG:-"next_minimal_timing_n3_$(date +%Y%m%d_%H%M%S)"}
OUT_ROOT=${OUT_ROOT:-"${WORK_ROOT}/profile_results/${RUN_TAG}"}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/next_minimal_timing"}
START_FROM=${START_FROM:-}
skip_until_start=true

mkdir -p "${OUT_ROOT}" "${RUNTIME_CACHE_ROOT}"

run_one() {
  local mode=$1
  local task=$2
  local boundary=$3
  local history_fps=$4
  local visual_mode=$5
  local max_images=$6
  local selection=$7
  local run_key="${mode}_next_${task}"
  if [[ -n "${START_FROM}" && "${skip_until_start}" == true ]]; then
    if [[ "${run_key}" != "${START_FROM}" ]]; then
      return
    fi
    skip_until_start=false
  fi
  local log_dir="${OUT_ROOT}/${mode}_next_${task}"

  mkdir -p "${log_dir}"
  echo "=== ${mode}_next_${task} ===" | tee -a "${OUT_ROOT}/run_order.log"
  printf 'history_fps=%s visual_mode=%s max_images=%s selection=%s\n' \
    "${history_fps}" "${visual_mode}" "${max_images}" "${selection}" \
    | tee "${log_dir}/protocol.txt"
  env \
    TEST_NUM="${TEST_NUM}" \
    SEED="${SEED}" \
    TASK_NAME="${task}" \
    TASK_CONFIG=demo_clean \
    INSTRUCTION_TYPE=unseen \
    HANDOFF_TRIGGER=forced_boundary \
    FORCED_BOUNDARY_S="${boundary}" \
    MAX_FORCED_BOUNDARIES=1 \
    STOP_AFTER_ORACLE_HANDOFF=False \
    STOP_AFTER_MEASUREMENT_HANDOFF=True \
    STOP_AFTER_DECISION_HANDOFF=False \
    HL_BACKEND=gemma4_vllm \
    HL_PROMPT_MODE=completion_status \
    HL_PLANNER_MODE=vlm_plan_next \
    HL_INCLUDE_PHASE_PROFILE_HINTS=False \
    HL_ASYNC_ENABLED=False \
    HL_FORCE_VISUAL_ON_DECISION=True \
    HL_VISUAL_CONTEXT_MODE="${visual_mode}" \
    HL_HISTORY_FPS="${history_fps}" \
    HL_RECENT_FRAMES_LENGTH="${max_images}" \
    HL_VLM_MAX_IMAGES="${max_images}" \
    HL_VLM_MAX_NEW_TOKENS=48 \
    GEMMA4_SERVER_URL=${GEMMA4_SERVER_URL:-http://127.0.0.1:8003} \
    GEMMA4_MODEL_NAME=${GEMMA4_MODEL_NAME:-gemma-4-12B-it} \
    GEMMA4_REQUEST_TIMEOUT_S=${GEMMA4_REQUEST_TIMEOUT_S:-120} \
    GEMMA4_JPEG_QUALITY=${GEMMA4_JPEG_QUALITY:-90} \
    GEMMA4_IMAGE_SELECTION="${selection}" \
    GEMMA4_IMAGE_MAX_SIDE=${GEMMA4_IMAGE_MAX_SIDE:-0} \
    GEMMA4_OUTPUT_SCHEMA=completion_bool \
    ROBOTWIN_EVAL_VIDEO_LOG=true \
    LOW_LEVEL_TRACE_EVERY_STEP=False \
    RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT}" \
    ROBOTWIN_EVAL_RESULT_ROOT="${log_dir}/eval_results" \
    LOG_DIR="${log_dir}" \
    bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh" \
      >"${log_dir}/console.log" 2>&1
}

for task_boundary in "place_burger_fries:2.37" "put_bottles_dustbin:4.01"; do
  task=${task_boundary%%:*}
  boundary=${task_boundary##*:}
  run_one single "${task}" "${boundary}" 25.0 recent 1 latest
  run_one 2s "${task}" "${boundary}" 0.5 unseen_memory 8 tail
  run_one 1hz "${task}" "${boundary}" 1.0 unseen_memory 8 tail
done

echo "${OUT_ROOT}"
