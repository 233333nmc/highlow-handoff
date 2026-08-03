#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TASK_NAME:-}" ]]; then
  echo "TASK_NAME is required, for example: TASK_NAME=place_cans_plasticbox" >&2
  exit 2
fi

export HANDOFF_TRIGGER="${HANDOFF_TRIGGER:-task_oracle}"
export ORACLE_STABILITY_STEPS="${ORACLE_STABILITY_STEPS:-3}"
export STOP_AFTER_ORACLE_HANDOFF="${STOP_AFTER_ORACLE_HANDOFF:-True}"
export CALIBRATION_MAX_SIM_TIME_S="${CALIBRATION_MAX_SIM_TIME_S:-8.0}"
export INSTRUCTION_MODE="${INSTRUCTION_MODE:-original}"
export HL_BACKEND="${HL_BACKEND:-scripted}"
export HL_VLM_MAX_IMAGES="${HL_VLM_MAX_IMAGES:-6}"
export HL_VLM_MAX_NEW_TOKENS="${HL_VLM_MAX_NEW_TOKENS:-96}"

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
