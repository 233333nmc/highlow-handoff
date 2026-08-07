#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the asynchronous high-level planner path with the smallest
# status-aware Qwen3.6 schema. Start `serve_qwen36_vllm_mtp.sh` first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

export TASK_NAME=${TASK_NAME:-place_burger_fries}
export TEST_NUM=${TEST_NUM:-3}
export HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
export FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-2.0}
export MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-10}
export STOP_AFTER_MEASUREMENT_HANDOFF=${STOP_AFTER_MEASUREMENT_HANDOFF:-False}

export HL_BACKEND=${HL_BACKEND:-qwen36_vllm_mtp}
export HL_PROMPT_MODE=${HL_PROMPT_MODE:-completion_status}
export HL_PLANNER_MODE=${HL_PLANNER_MODE:-vlm_plan_next}
export HL_ASYNC_ENABLED=${HL_ASYNC_ENABLED:-True}
export HL_ASYNC_DROP_STALE_S=${HL_ASYNC_DROP_STALE_S:-0.0}
export HL_VLM_MAX_IMAGES=${HL_VLM_MAX_IMAGES:-1}
export HL_VLM_MAX_NEW_TOKENS=${HL_VLM_MAX_NEW_TOKENS:-48}

export QWEN36_SERVER_URL=${QWEN36_SERVER_URL:-http://127.0.0.1:8001}
export QWEN36_MODEL_NAME=${QWEN36_MODEL_NAME:-qwen3.6-27b-mtp}
export QWEN36_REQUEST_TIMEOUT_S=${QWEN36_REQUEST_TIMEOUT_S:-120}
export QWEN36_JPEG_QUALITY=${QWEN36_JPEG_QUALITY:-90}
export QWEN36_IMAGE_SELECTION=${QWEN36_IMAGE_SELECTION:-latest}
export QWEN36_IMAGE_MAX_SIDE=${QWEN36_IMAGE_MAX_SIDE:-0}
export QWEN36_OUTPUT_SCHEMA=${QWEN36_OUTPUT_SCHEMA:-completion_status}

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="${WORK_ROOT}/profile_results/robotwin_${TASK_NAME}/async_qwen36_mtp_$(date +%Y%m%d_%H%M%S)"
fi

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
