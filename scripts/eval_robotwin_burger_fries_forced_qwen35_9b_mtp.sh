#!/usr/bin/env bash
set -euo pipefail

# Evaluate Qwen3.5-9B with native MTP as the high-level planner. Start
# `serve_qwen35_9b_vllm_mtp.sh` first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

export HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
export FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-2.17}
export MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-1}
export TASK_NAME=${TASK_NAME:-place_burger_fries}
export HL_BACKEND=${HL_BACKEND:-qwen35_vllm_mtp}
export HL_PROMPT_MODE=${HL_PROMPT_MODE:-completion_status}
export HL_PLANNER_MODE=${HL_PLANNER_MODE:-vlm_plan_next}
export HL_FORCE_VISUAL_ON_DECISION=${HL_FORCE_VISUAL_ON_DECISION:-True}
export HL_RECENT_FRAMES_LENGTH=${HL_RECENT_FRAMES_LENGTH:-1}
export HL_VLM_MAX_IMAGES=${HL_VLM_MAX_IMAGES:-1}
export HL_VLM_MAX_NEW_TOKENS=${HL_VLM_MAX_NEW_TOKENS:-48}
export QWEN35_SERVER_URL=${QWEN35_SERVER_URL:-http://127.0.0.1:8001}
export QWEN35_MODEL_NAME=${QWEN35_MODEL_NAME:-qwen3.5-9b-mtp}
export QWEN35_REQUEST_TIMEOUT_S=${QWEN35_REQUEST_TIMEOUT_S:-120}
export QWEN35_JPEG_QUALITY=${QWEN35_JPEG_QUALITY:-90}
export QWEN35_IMAGE_SELECTION=${QWEN35_IMAGE_SELECTION:-latest}
export QWEN35_IMAGE_MAX_SIDE=${QWEN35_IMAGE_MAX_SIDE:-0}
export QWEN35_OUTPUT_SCHEMA=${QWEN35_OUTPUT_SCHEMA:-completion_status}

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="${WORK_ROOT}/profile_results/robotwin_${TASK_NAME}/forced_qwen35_9b_mtp_$(date +%Y%m%d_%H%M%S)"
fi

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
