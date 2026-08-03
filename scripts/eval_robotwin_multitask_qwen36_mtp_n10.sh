#!/usr/bin/env bash
set -euo pipefail

# Run a uniform n=10 steady-state handoff benchmark. Each task runs 11
# rollouts; episode 0 is retained as a runtime warm-up and excluded later.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

TEST_NUM=${TEST_NUM:-11}
SEED=${SEED:-10000}
RUN_TAG=${RUN_TAG:-"qwen36_mtp_n10_$(date +%Y%m%d_%H%M%S)"}
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
    STOP_AFTER_MEASUREMENT_HANDOFF=True \
    HL_VLM_MAX_IMAGES=3 \
    HL_VLM_MAX_NEW_TOKENS=64 \
    ROBOTWIN_EVAL_VIDEO_LOG=false \
    RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT}" \
    ROBOTWIN_EVAL_RESULT_ROOT="${ROBOTWIN_EVAL_RESULT_ROOT}" \
    LOG_DIR="${log_dir}" \
    bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_forced_qwen36_mtp.sh"
done
