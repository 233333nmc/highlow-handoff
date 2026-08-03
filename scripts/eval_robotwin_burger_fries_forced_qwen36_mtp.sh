#!/usr/bin/env bash
set -euo pipefail

# Evaluate Qwen3.6-27B with native checkpoint BF16 weights and native MTP as a
# high-level planner. Start `serve_qwen36_vllm_mtp.sh` first, then invoke this
# script. Runtime FP8 is opt-in in the service script because vLLM 0.19.1
# cannot quantize this checkpoint's visual layer online.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

export HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
export FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-2.17}
export MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-1}
export TASK_NAME=${TASK_NAME:-place_burger_fries}
export HL_BACKEND=${HL_BACKEND:-qwen36_vllm_mtp}
export HL_VLM_MAX_IMAGES=${HL_VLM_MAX_IMAGES:-3}
export HL_VLM_MAX_NEW_TOKENS=${HL_VLM_MAX_NEW_TOKENS:-64}
export QWEN36_SERVER_URL=${QWEN36_SERVER_URL:-http://127.0.0.1:8001}
export QWEN36_MODEL_NAME=${QWEN36_MODEL_NAME:-qwen3.6-27b-mtp}
export QWEN36_REQUEST_TIMEOUT_S=${QWEN36_REQUEST_TIMEOUT_S:-120}
export QWEN36_JPEG_QUALITY=${QWEN36_JPEG_QUALITY:-90}

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="${WORK_ROOT}/profile_results/robotwin_${TASK_NAME}/forced_qwen36_mtp_$(date +%Y%m%d_%H%M%S)"
fi

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
