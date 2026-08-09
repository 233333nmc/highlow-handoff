#!/usr/bin/env bash
set -euo pipefail

# Diagnostic probe for Hy-VLA language conditioning.
# High-level is disabled. The script optionally overrides the exact
# instruction sent to the low-level policy for the whole rollout and records
# every real model-forward instruction.

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_ROOT=${WORK_ROOT:-"$(cd "${HY_VLA_DIR}/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-"${WORK_ROOT}/RoboTwin"}
CKPT_PATH=${CKPT_PATH:-"${WORK_ROOT}/checkpoints/Hy-Embodied-0.5-VLA-RoboTwin"}
DEFAULT_PYTHON="${WORK_ROOT}/envs/robotwin_hyvla_eval/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="${HY_VLA_DIR}/.venv/bin/python"
fi
PYTHON=${PYTHON:-"${DEFAULT_PYTHON}"}

TASK_NAME=${TASK_NAME:-place_burger_fries}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
TEST_NUM=${TEST_NUM:-11}
SEED=${SEED:-10000}
INSTRUCTION_VARIANT=${INSTRUCTION_VARIANT:-full}
CUSTOM_INSTRUCTION=${CUSTOM_INSTRUCTION:-}
ROBOTWIN_EVAL_VIDEO_LOG=${ROBOTWIN_EVAL_VIDEO_LOG:-true}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/instruction_probe"}

case "${TASK_NAME}:${INSTRUCTION_VARIANT}" in
  place_burger_fries:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  place_burger_fries:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Place the hamburg on the tray."
    ;;
  place_burger_fries:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Place the french fries on the tray."
    ;;
  place_burger_fries:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  put_bottles_dustbin:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  put_bottles_dustbin:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Put one bottle into the dustbin."
    ;;
  put_bottles_dustbin:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Stack the bowls at the center of the table."
    ;;
  put_bottles_dustbin:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  stack_blocks_three:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  stack_blocks_three:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Move the red block to the target position."
    ;;
  stack_blocks_three:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Put the bottle into the dustbin."
    ;;
  stack_blocks_three:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  hanging_mug:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  hanging_mug:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Place the mug at the middle handover position."
    ;;
  hanging_mug:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Put one bottle into the dustbin."
    ;;
  hanging_mug:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  put_object_cabinet:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  put_object_cabinet:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Open the cabinet drawer."
    ;;
  put_object_cabinet:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Place the red block at the center of the table."
    ;;
  put_object_cabinet:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  handover_block:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  handover_block:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Move the red block to the center handover point."
    ;;
  handover_block:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Open the cabinet drawer."
    ;;
  handover_block:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  handover_mic:full)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  handover_mic:correct_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Move the microphone to the center handover point."
    ;;
  handover_mic:wrong_subtask)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="Stack the green block on the red block."
    ;;
  handover_mic:empty)
    LOW_LEVEL_INSTRUCTION_OVERRIDE=""
    ;;
  *:custom)
    LOW_LEVEL_INSTRUCTION_OVERRIDE="${CUSTOM_INSTRUCTION}"
    ;;
  *)
    echo "Unsupported TASK_NAME/INSTRUCTION_VARIANT: ${TASK_NAME}/${INSTRUCTION_VARIANT}" >&2
    echo "Use full, correct_subtask, wrong_subtask, empty, or custom." >&2
    exit 2
    ;;
esac

if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR="${WORK_ROOT}/profile_results/instruction_probe/${TASK_NAME}_${INSTRUCTION_VARIANT}_$(date +%Y%m%d_%H%M%S)"
fi
LOG_PATH=${LOG_PATH:-"${LOG_DIR}/eval.log"}
TRACE_PATH=${TRACE_PATH:-"${LOG_DIR}/low_level_instruction_trace.jsonl"}
LOW_LEVEL_TRACE_EVERY_STEP=${LOW_LEVEL_TRACE_EVERY_STEP:-False}

mkdir -p \
  "${LOG_DIR}" \
  "${RUNTIME_CACHE_ROOT}/scratch" \
  "${RUNTIME_CACHE_ROOT}/matplotlib" \
  "${RUNTIME_CACHE_ROOT}/xdg" \
  "${WORK_ROOT}/model_cache/huggingface"
ln -sfn "${HY_VLA_DIR}/robotwin_eval" "${ROBOTWIN_DIR}/policy/hy_vla"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export TMPDIR="${RUNTIME_CACHE_ROOT}/scratch"
export MPLCONFIGDIR="${RUNTIME_CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg"
export HF_HOME="${WORK_ROOT}/model_cache/huggingface"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export PYTHONPATH="${HY_VLA_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
EVAL_RESULT_ROOT=${EVAL_RESULT_ROOT:-"${LOG_DIR}/eval_results"}
export ROBOTWIN_EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT}"
export ROBOTWIN_EVAL_VIDEO_LOG

# Some containers expose NVIDIA's EGL library but omit
# /etc/glvnd/egl_vendor.d. SAPIEN otherwise raises while importing. Use the
# SAPIEN-packaged vendor manifest when the platform has not already provided
# one. This does not change CUDA device visibility.
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -z "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]]; then
  SAPIEN_PACKAGE_DIR="$("${PYTHON}" -c \
    'import importlib.util; print(next(iter(importlib.util.find_spec("sapien").submodule_search_locations)))')"
  SAPIEN_EGL_ICD="${SAPIEN_PACKAGE_DIR}/vulkan_library/10_nvidia.json"
  if [[ -f "${SAPIEN_EGL_ICD}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${SAPIEN_EGL_ICD}"
  fi
fi

echo "RoboTwin task       : ${TASK_NAME}"
echo "Mode               : instruction probe, high-level disabled"
echo "Rollouts           : ${TEST_NUM}"
echo "Instruction variant: ${INSTRUCTION_VARIANT}"
if [[ "${INSTRUCTION_VARIANT}" == "full" ]]; then
  echo "Low-level override : <none; use RoboTwin sampled full instruction>"
elif [[ "${INSTRUCTION_VARIANT}" == "empty" ]]; then
  echo "Low-level override : <empty string>"
else
  echo "Low-level override : ${LOW_LEVEL_INSTRUCTION_OVERRIDE}"
fi
echo "Trace              : ${TRACE_PATH}"
echo "Eval log           : ${LOG_PATH}"

OVERRIDE_ARGS=()
if [[ "${INSTRUCTION_VARIANT}" != "full" ]]; then
  OVERRIDE_ARGS+=(--low_level_instruction_override "${LOW_LEVEL_INSTRUCTION_OVERRIDE}")
fi

(
  cd "${ROBOTWIN_DIR}"
  PYTHONWARNINGS=ignore::UserWarning \
  "${PYTHON}" -u script/eval_policy.py \
    --config policy/hy_vla/deploy_policy.yml \
    --overrides \
      --task_name "${TASK_NAME}" \
      --task_config "${TASK_CONFIG}" \
      --ckpt_setting Hy-Embodied-0.5-VLA-RoboTwin \
      --instruction_type "${INSTRUCTION_TYPE}" \
      --seed "${SEED}" \
      --test_num "${TEST_NUM}" \
      --policy_name policy.hy_vla \
      --ckpt_path "${CKPT_PATH}" \
      --hierarchical_enabled False \
      --low_level_instruction_trace_path "${TRACE_PATH}" \
      --low_level_instruction_trace_every_step "${LOW_LEVEL_TRACE_EVERY_STEP}" \
      "${OVERRIDE_ARGS[@]}" \
  2>&1 | sed -u 's/\r/\n/g' | sed -u '/step: /d'
) | tee "${LOG_PATH}"
