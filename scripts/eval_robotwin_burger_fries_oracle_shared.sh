#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HANDOFF_TRIGGER="${HANDOFF_TRIGGER:-task_oracle}"
export ORACLE_STABILITY_STEPS="${ORACLE_STABILITY_STEPS:-3}"
export HL_BACKEND="${HL_BACKEND:-shared_hunyuan_vlm}"
export HL_VLM_MAX_IMAGES="${HL_VLM_MAX_IMAGES:-6}"
export HL_VLM_MAX_NEW_TOKENS="${HL_VLM_MAX_NEW_TOKENS:-96}"

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
