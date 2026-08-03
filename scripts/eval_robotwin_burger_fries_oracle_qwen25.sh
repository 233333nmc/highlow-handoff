#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

export HANDOFF_TRIGGER="${HANDOFF_TRIGGER:-burger_oracle}"
export ORACLE_STABILITY_STEPS="${ORACLE_STABILITY_STEPS:-3}"
export HL_BACKEND="${HL_BACKEND:-qwen25_vl}"
export HL_VLM_MAX_IMAGES="${HL_VLM_MAX_IMAGES:-6}"
export HL_VLM_MAX_NEW_TOKENS="${HL_VLM_MAX_NEW_TOKENS:-128}"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-"${WORK_ROOT}/checkpoints/Qwen2.5-VL-32B-Instruct"}"
export QWEN_DEVICE="${QWEN_DEVICE:-auto}"
export QWEN_ATTENTION_IMPLEMENTATION="${QWEN_ATTENTION_IMPLEMENTATION:-flash_attention_2}"

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="${WORK_ROOT}/profile_results/robotwin_burger_fries/qwen25_$(date +%Y%m%d_%H%M%S)"
fi

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
