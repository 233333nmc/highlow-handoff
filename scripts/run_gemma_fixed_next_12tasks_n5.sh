#!/usr/bin/env bash
set -u
set -o pipefail

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_ROOT=${WORK_ROOT:-"$(cd "${HY_VLA_DIR}/.." && pwd)"}
RUN_ROOT=${RUN_ROOT:-"${WORK_ROOT}/profile_results/gemma_fixed_next_12tasks_n5_20260818"}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
SEED=${SEED:-10000}
TEST_NUM=${TEST_NUM:-5}
CALIBRATE_BOUNDARIES=${CALIBRATE_BOUNDARIES:-True}
BOUNDARY_MARGIN_S=${BOUNDARY_MARGIN_S:-0.10}
BOUNDARY_CALIBRATION_FILE=${BOUNDARY_CALIBRATION_FILE:-}

# Ten additional multi-step tasks. Burger and three-block stacking are kept
# separate below so their five-sample runs are easy to find.
TASKS=(
  place_cans_plasticbox
  stack_bowls_three
  stack_bowls_two
  place_dual_shoes
  hanging_mug
  put_object_cabinet
  handover_block
  handover_mic
  stack_blocks_two
  blocks_ranking_rgb
  place_burger_fries
  stack_blocks_three
)

# Fixed-boundary start times are task-specific.  Values for the original six
# tasks come from the existing simulator-oracle calibration traces, with a
# small 0.1 s margin so the boundary is reached after the first subtask's
# normal completion.  The added tasks use the closest calibrated motion class
# until their first oracle-only calibration is available.
declare -A BOUNDARY_BY_TASK=(
  [place_burger_fries]=2.27
  [place_cans_plasticbox]=2.30
  [stack_bowls_three]=4.63
  [stack_bowls_two]=4.63
  [place_dual_shoes]=2.71
  [hanging_mug]=2.70
  [put_object_cabinet]=2.70
  [handover_block]=2.60
  [handover_mic]=2.50
  [stack_blocks_two]=2.35
  [blocks_ranking_rgb]=2.35
  [stack_blocks_three]=2.35
)

mkdir -p "${RUN_ROOT}"
SUMMARY_PATH="${RUN_ROOT}/batch_summary.tsv"
printf 'task\tstatus\tlog_dir\n' > "${SUMMARY_PATH}"
CALIBRATION_SUMMARY_PATH="${RUN_ROOT}/boundary_calibration.tsv"
printf 'task\toracle_first_completion_s\tforced_boundary_s\tsource\n' > "${CALIBRATION_SUMMARY_PATH}"

calibrate_boundary() {
  local task=$1
  local calibration_dir="${RUN_ROOT}/${task}/oracle_calibration"
  local decisions_path="${calibration_dir}/hierarchical_decisions.jsonl"
  local oracle_s=""
  local boundary_s="${BOUNDARY_BY_TASK[${task}]:-}"

  if [[ -n "${BOUNDARY_CALIBRATION_FILE}" && -f "${BOUNDARY_CALIBRATION_FILE}" ]]; then
    boundary_s=$(awk -F '\t' -v task="${task}" '$1 == task { print $3; exit }' \
      "${BOUNDARY_CALIBRATION_FILE}")
    if [[ -n "${boundary_s}" && "${boundary_s}" != "forced_boundary_s" ]]; then
      printf '%s\t%s\t%s\treused_calibration\n' \
        "${task}" "" "${boundary_s}" >> "${CALIBRATION_SUMMARY_PATH}"
      printf '%s\n' "${boundary_s}"
      return 0
    fi
    echo "Warning: no reusable boundary for ${task} in ${BOUNDARY_CALIBRATION_FILE}" >&2
  fi

  if [[ "${CALIBRATE_BOUNDARIES}" == "True" || "${CALIBRATE_BOUNDARIES}" == "true" ]]; then
    mkdir -p "${calibration_dir}"
    echo "Calibrating ${task}: waiting for the first simulator-oracle completion" >&2
    (
      cd "${WORK_ROOT}"
      env \
        TEST_NUM=1 \
        SEED="${SEED}" \
        TASK_NAME="${task}" \
        TASK_CONFIG="${TASK_CONFIG}" \
        INSTRUCTION_TYPE=unseen \
        HANDOFF_TRIGGER=task_oracle \
        ORACLE_STABILITY_STEPS=3 \
        STOP_AFTER_ORACLE_HANDOFF=True \
        CALIBRATION_MAX_SIM_TIME_S=8.0 \
        MAX_FORCED_BOUNDARIES=1 \
        HL_BACKEND=scripted \
        HL_PLANNER_MODE=profile_all \
        HL_PROMPT_MODE=completion_status \
        HL_ASYNC_ENABLED=False \
        HL_FORCE_VISUAL_ON_DECISION=False \
        ROBOTWIN_EVAL_VIDEO_LOG=False \
        LOW_LEVEL_TRACE_EVERY_STEP=False \
        ROBOTWIN_EVAL_RESULT_ROOT="${calibration_dir}/eval_results" \
        LOG_DIR="${calibration_dir}" \
        RUNTIME_CACHE_ROOT="${WORK_ROOT}/runtime_cache/gemma_fixed_next_12tasks_calibration" \
        bash "${WORK_ROOT}/Hy-Embodied-0.5-VLA/scripts/eval_robotwin_burger_fries_hierarchical.sh" \
        > "${calibration_dir}/console.log" 2>&1
    ) || true
    if [[ -s "${decisions_path}" ]]; then
      oracle_s=$(jq -r 'select(.episode_index == 0 and .trigger != "initial") | .timestamp_s' \
        "${decisions_path}" | head -n 1)
    fi
    if [[ -n "${oracle_s}" && "${oracle_s}" != "null" ]]; then
      boundary_s=$(awk -v t="${oracle_s}" -v m="${BOUNDARY_MARGIN_S}" \
        'BEGIN { printf "%.2f", t + m }')
      printf '%s\t%s\t%s\toracle_plus_margin\n' \
        "${task}" "${oracle_s}" "${boundary_s}" >> "${CALIBRATION_SUMMARY_PATH}"
      printf '%s\n' "${boundary_s}"
      return 0
    fi
    echo "Warning: oracle calibration did not produce a timestamp for ${task}; using fallback ${boundary_s}s" >&2
  fi

  if [[ -z "${boundary_s}" ]]; then
    echo "Missing task-specific boundary fallback for ${task}" >&2
    return 1
  fi
  printf '%s\t%s\t%s\tfallback_profile_class\n' \
    "${task}" "" "${boundary_s}" >> "${CALIBRATION_SUMMARY_PATH}"
  printf '%s\n' "${boundary_s}"
}

for task in "${TASKS[@]}"; do
  task_dir="${RUN_ROOT}/${task}"
  boundary_s="$(calibrate_boundary "${task}")" || exit 2
  if [[ -z "${boundary_s}" ]]; then
    echo "Missing task-specific boundary for ${task}" >&2
    exit 2
  fi
  mkdir -p "${task_dir}"
  echo "============================================================"
  echo "Running ${task} (${TEST_NUM} episodes; first boundary ${boundary_s}s)"
  echo "Output: ${task_dir}"
  printf 'task=%s\nforced_boundary_s=%s\nmax_forced_boundaries=10\n' \
    "${task}" "${boundary_s}" > "${task_dir}/protocol.txt"

  if (
    cd "${WORK_ROOT}"
    env \
      TEST_NUM="${TEST_NUM}" \
      SEED="${SEED}" \
      TASK_NAME="${task}" \
      TASK_CONFIG="${TASK_CONFIG}" \
      INSTRUCTION_TYPE=unseen \
      ROBOTWIN_EVAL_VIDEO_LOG=True \
      ROBOTWIN_EVAL_RESULT_ROOT="${task_dir}/eval_results" \
      FORCED_BOUNDARY_S="${boundary_s}" \
      MAX_FORCED_BOUNDARIES=10 \
      HANDOFF_TRIGGER=forced_boundary \
      HANDOFF_PAUSE_STEPS=0 \
      INSTRUCTION_MODE=subtask \
      HL_BACKEND=gemma4_vllm \
      HL_PLANNER_MODE=vlm_plan_next \
      HL_PROMPT_MODE=completion_status \
      HL_INCLUDE_PHASE_PROFILE_HINTS=False \
      HL_ONE_OBJECT_ONLY="${ONE_OBJECT_ONLY:-False}" \
      HL_ASYNC_ENABLED=False \
      HL_FORCE_VISUAL_ON_DECISION=True \
      HL_VISUAL_CONTEXT_MODE=unseen_memory \
      HL_HISTORY_FPS=1.0 \
      HL_SOURCE_FPS=25.0 \
      HL_VLM_MAX_IMAGES=8 \
      HL_VLM_MAX_NEW_TOKENS=48 \
      GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
      GEMMA4_MODEL_NAME=gemma-4-12B-it \
      GEMMA4_IMAGE_SELECTION=all \
      GEMMA4_OUTPUT_SCHEMA=completion_bool \
      LOW_LEVEL_TRACE_EVERY_STEP=True \
      STOP_AFTER_MEASUREMENT_HANDOFF=False \
      STOP_AFTER_DECISION_HANDOFF=False \
      LOG_DIR="${task_dir}" \
      RUNTIME_CACHE_ROOT="${WORK_ROOT}/runtime_cache/gemma_fixed_next_12tasks" \
      bash "${WORK_ROOT}/Hy-Embodied-0.5-VLA/scripts/eval_robotwin_burger_fries_hierarchical.sh" \
      2>&1 | tee "${task_dir}/console.log"
  ); then
    printf '%s\tcompleted\t%s\n' "${task}" "${task_dir}" >> "${SUMMARY_PATH}"
  else
    printf '%s\tfailed\t%s\n' "${task}" "${task_dir}" >> "${SUMMARY_PATH}"
    echo "Task ${task} failed; continuing with the next task."
  fi
done

echo "Batch summary: ${SUMMARY_PATH}"
