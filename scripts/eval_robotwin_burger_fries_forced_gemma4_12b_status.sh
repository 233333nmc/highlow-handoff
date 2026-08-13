#!/usr/bin/env bash
set -euo pipefail

# Fixed-boundary completion-check benchmark for Gemma 4 12B.
# By default this uses unseen visual memory: low-frequency frames are stored
# during low-level execution, and the verifier receives all unseen frames plus
# the current decision observation at the handoff boundary.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

export TASK_NAME=${TASK_NAME:-place_burger_fries}
export TEST_NUM=${TEST_NUM:-11}
export HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
export FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-2.17}
export MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-1}
export STOP_AFTER_MEASUREMENT_HANDOFF=${STOP_AFTER_MEASUREMENT_HANDOFF:-False}

export HL_BACKEND=${HL_BACKEND:-gemma4_vllm}
export HL_PROMPT_MODE=${HL_PROMPT_MODE:-completion_status}
export HL_PLANNER_MODE=${HL_PLANNER_MODE:-vlm_plan_next}
export HL_ASYNC_ENABLED=${HL_ASYNC_ENABLED:-False}
export HL_FORCE_VISUAL_ON_DECISION=${HL_FORCE_VISUAL_ON_DECISION:-True}
export HL_VISUAL_CONTEXT_MODE=${HL_VISUAL_CONTEXT_MODE:-unseen_memory}
export HL_HISTORY_FPS=${HL_HISTORY_FPS:-1.0}
export HL_RECENT_FRAMES_LENGTH=${HL_RECENT_FRAMES_LENGTH:-8}
export HL_VLM_MAX_IMAGES=${HL_VLM_MAX_IMAGES:-8}
export HL_VLM_MAX_NEW_TOKENS=${HL_VLM_MAX_NEW_TOKENS:-48}

export GEMMA4_SERVER_URL=${GEMMA4_SERVER_URL:-http://127.0.0.1:8003}
export GEMMA4_MODEL_NAME=${GEMMA4_MODEL_NAME:-gemma-4-12B-it}
export GEMMA4_REQUEST_TIMEOUT_S=${GEMMA4_REQUEST_TIMEOUT_S:-120}
export GEMMA4_JPEG_QUALITY=${GEMMA4_JPEG_QUALITY:-90}
export GEMMA4_IMAGE_SELECTION=${GEMMA4_IMAGE_SELECTION:-tail}
export GEMMA4_IMAGE_MAX_SIDE=${GEMMA4_IMAGE_MAX_SIDE:-0}
export GEMMA4_OUTPUT_SCHEMA=${GEMMA4_OUTPUT_SCHEMA:-completion_status}

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="${WORK_ROOT}/profile_results/robotwin_${TASK_NAME}/forced_gemma4_12b_status_$(date +%Y%m%d_%H%M%S)"
fi

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
