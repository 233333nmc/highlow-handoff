#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

RUN_ROOT=${RUN_ROOT:-"${WORK_ROOT}/profile_results/language_sensitivity/stack_blocks_three_oracle_n2_$(date +%Y%m%d_%H%M%S)"}
TEST_NUM=${TEST_NUM:-2}
SEED=${SEED:-10000}
VARIANTS=${VARIANTS:-"correct wrong empty full no_switch"}

RED="Place the red block at the center of the table."
GREEN="Stack the green block on the red block."
BLUE="Stack the blue block on the green block."
FULL="Transfer red block, green block, and blue block to the middle, then stack blue block over green block and green block over red block."

cd "${HY_VLA_DIR}"

for variant in ${VARIANTS}; do
  case "${variant}" in
    correct)
      phases="${RED}|||${GREEN}|||${BLUE}"
      max_handoffs=2
      ;;
    wrong)
      phases="${RED}|||${BLUE}|||${GREEN}"
      max_handoffs=2
      ;;
    empty)
      phases="${RED}||||||"
      max_handoffs=2
      ;;
    full)
      phases="${RED}|||${FULL}|||${FULL}"
      max_handoffs=2
      ;;
    no_switch)
      phases="${RED}|||${GREEN}|||${BLUE}"
      max_handoffs=0
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 2
      ;;
  esac

  echo "=== stack_blocks_three language sensitivity: ${variant} ==="
  LOG_DIR="${RUN_ROOT}/${variant}" \
  ROBOTWIN_EVAL_RESULT_ROOT="${RUN_ROOT}/${variant}/eval_results" \
  TEST_NUM="${TEST_NUM}" \
  SEED="${SEED}" \
  TASK_NAME=stack_blocks_three \
  HANDOFF_TRIGGER=task_oracle \
  MAX_FORCED_BOUNDARIES="${max_handoffs}" \
  ORACLE_STABILITY_STEPS=1 \
  HL_BACKEND=scripted \
  HL_PLANNER_MODE=profile_all \
  HL_PROMPT_MODE=completion_status \
  HL_PHASE_PROMPTS_JSON="${phases}" \
  HL_FORCE_VISUAL_ON_DECISION=False \
  ROBOTWIN_EVAL_VIDEO_LOG=True \
  LOW_LEVEL_TRACE_EVERY_STEP=True \
  bash scripts/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh
done

echo "Results: ${RUN_ROOT}"
