#!/usr/bin/env bash
set -euo pipefail

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_ROOT=${WORK_ROOT:-"$(cd "${HY_VLA_DIR}/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-"${WORK_ROOT}/RoboTwin"}
CKPT_PATH=${CKPT_PATH:-"${WORK_ROOT}/checkpoints/Hy-Embodied-0.5-VLA-RoboTwin"}
DEFAULT_PYTHON="${WORK_ROOT}/envs/robotwin_hyvla_eval/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="${HY_VLA_DIR}/.venv/bin/python"
fi
PYTHON=${PYTHON:-"${DEFAULT_PYTHON}"}
TEST_NUM=${TEST_NUM:-3}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
SEED=${SEED:-10000}
TASK_NAME=${TASK_NAME:-place_burger_fries}
LOG_DIR=${LOG_DIR:-"${WORK_ROOT}/profile_results/robotwin_${TASK_NAME}"}
LOG_PATH=${LOG_PATH:-"${LOG_DIR}/hierarchical.log"}
HL_LOG_PATH=${HL_LOG_PATH:-"${LOG_DIR}/hierarchical_decisions.jsonl"}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/robotwin_eval"}
ROBOTWIN_EVAL_RESULT_ROOT=${ROBOTWIN_EVAL_RESULT_ROOT:-"${ROBOTWIN_DIR}/eval_result"}
FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-5.0}
MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-1}
HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
ORACLE_STABILITY_STEPS=${ORACLE_STABILITY_STEPS:-3}
STOP_AFTER_ORACLE_HANDOFF=${STOP_AFTER_ORACLE_HANDOFF:-False}
CALIBRATION_MAX_SIM_TIME_S=${CALIBRATION_MAX_SIM_TIME_S:-0.0}
STOP_AFTER_MEASUREMENT_HANDOFF=${STOP_AFTER_MEASUREMENT_HANDOFF:-False}
INSTRUCTION_MODE=${INSTRUCTION_MODE:-subtask}
HL_BACKEND=${HL_BACKEND:-scripted}
HL_VLM_MAX_IMAGES=${HL_VLM_MAX_IMAGES:-8}
HL_VLM_MAX_NEW_TOKENS=${HL_VLM_MAX_NEW_TOKENS:-160}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-"${WORK_ROOT}/checkpoints/Qwen2.5-VL-32B-Instruct"}
QWEN_DEVICE=${QWEN_DEVICE:-auto}
QWEN_ATTENTION_IMPLEMENTATION=${QWEN_ATTENTION_IMPLEMENTATION:-flash_attention_2}
QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-200704}
QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-602112}
QWEN36_SERVER_URL=${QWEN36_SERVER_URL:-http://127.0.0.1:8001}
QWEN36_MODEL_NAME=${QWEN36_MODEL_NAME:-qwen3.6-27b-mtp}
QWEN36_REQUEST_TIMEOUT_S=${QWEN36_REQUEST_TIMEOUT_S:-120}
QWEN36_JPEG_QUALITY=${QWEN36_JPEG_QUALITY:-90}
QWEN36_IMAGE_SELECTION=${QWEN36_IMAGE_SELECTION:-tail}
QWEN36_IMAGE_MAX_SIDE=${QWEN36_IMAGE_MAX_SIDE:-0}

mkdir -p \
  "${LOG_DIR}" \
  "${RUNTIME_CACHE_ROOT}/scratch" \
  "${RUNTIME_CACHE_ROOT}/matplotlib" \
  "${RUNTIME_CACHE_ROOT}/xdg" \
  "${RUNTIME_CACHE_ROOT}/huggingface"
ln -sfn "${HY_VLA_DIR}/robotwin_eval" "${ROBOTWIN_DIR}/policy/hy_vla"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export TMPDIR="${RUNTIME_CACHE_ROOT}/scratch"
export MPLCONFIGDIR="${RUNTIME_CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg"
export HF_HOME="${RUNTIME_CACHE_ROOT}/huggingface"
export ROBOTWIN_EVAL_RESULT_ROOT
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export PYTHONPATH="${HY_VLA_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}

# Some container images expose NVIDIA's EGL library but omit
# /etc/glvnd/egl_vendor.d. SAPIEN otherwise raises before it can use its
# packaged fallback ICD. This only selects the EGL vendor manifest; it does
# not alter the platform-provided CUDA device visibility.
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -z "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]]; then
  SAPIEN_PACKAGE_DIR="$("${PYTHON}" -c \
    'import importlib.util; print(next(iter(importlib.util.find_spec("sapien").submodule_search_locations)))')"
  SAPIEN_EGL_ICD="${SAPIEN_PACKAGE_DIR}/vulkan_library/10_nvidia.json"
  if [[ -f "${SAPIEN_EGL_ICD}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${SAPIEN_EGL_ICD}"
  fi
fi

echo "RoboTwin task       : ${TASK_NAME}"
echo "Mode               : hierarchical high-level memory + Hy-VLA"
echo "Rollouts           : ${TEST_NUM}"
echo "Checkpoint         : ${CKPT_PATH}"
echo "RoboTwin dir       : ${ROBOTWIN_DIR}"
echo "Python             : ${PYTHON}"
echo "Eval log           : ${LOG_PATH}"
echo "High-level log     : ${HL_LOG_PATH}"
if [[ "${HANDOFF_TRIGGER}" == "forced_boundary" ]]; then
  echo "Forced boundary    : ${FORCED_BOUNDARY_S}s"
else
  echo "Forced boundary    : disabled (oracle-triggered handoff)"
fi
echo "Max handoffs       : ${MAX_FORCED_BOUNDARIES}"
echo "Handoff trigger    : ${HANDOFF_TRIGGER}"
echo "Oracle stability   : ${ORACLE_STABILITY_STEPS} steps"
echo "Stop after oracle  : ${STOP_AFTER_ORACLE_HANDOFF}"
echo "Calibration limit  : ${CALIBRATION_MAX_SIM_TIME_S}s"
echo "Stop after measure : ${STOP_AFTER_MEASUREMENT_HANDOFF}"
echo "Instruction mode   : ${INSTRUCTION_MODE}"
echo "High-level backend : ${HL_BACKEND}"
if [[ "${HL_BACKEND}" == "qwen25_vl" ]]; then
  echo "Qwen checkpoint    : ${QWEN_MODEL_PATH}"
  echo "Qwen device        : ${QWEN_DEVICE}"
elif [[ "${HL_BACKEND}" == "qwen36_vllm_mtp" ]]; then
  echo "Qwen3.6 server     : ${QWEN36_SERVER_URL}"
  echo "Qwen3.6 model      : ${QWEN36_MODEL_NAME}"
  echo "Qwen3.6 images     : ${QWEN36_IMAGE_SELECTION}, max_side=${QWEN36_IMAGE_MAX_SIDE}"
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
      --hierarchical_enabled True \
      --hierarchical_instruction_mode "${INSTRUCTION_MODE}" \
      --hierarchical_handoff_trigger "${HANDOFF_TRIGGER}" \
      --hierarchical_oracle_stability_steps "${ORACLE_STABILITY_STEPS}" \
      --hierarchical_stop_after_oracle_handoff "${STOP_AFTER_ORACLE_HANDOFF}" \
      --hierarchical_calibration_max_sim_time_s "${CALIBRATION_MAX_SIM_TIME_S}" \
      --hierarchical_stop_after_measurement_handoff "${STOP_AFTER_MEASUREMENT_HANDOFF}" \
      --hierarchical_forced_boundary_s "${FORCED_BOUNDARY_S}" \
      --hierarchical_max_forced_boundaries "${MAX_FORCED_BOUNDARIES}" \
      --hierarchical_backend "${HL_BACKEND}" \
      --hierarchical_vlm_max_images "${HL_VLM_MAX_IMAGES}" \
      --hierarchical_vlm_max_new_tokens "${HL_VLM_MAX_NEW_TOKENS}" \
      --hierarchical_qwen_model_path "${QWEN_MODEL_PATH}" \
      --hierarchical_qwen_device "${QWEN_DEVICE}" \
      --hierarchical_qwen_attention_implementation "${QWEN_ATTENTION_IMPLEMENTATION}" \
      --hierarchical_qwen_min_pixels "${QWEN_MIN_PIXELS}" \
      --hierarchical_qwen_max_pixels "${QWEN_MAX_PIXELS}" \
      --hierarchical_qwen36_server_url "${QWEN36_SERVER_URL}" \
      --hierarchical_qwen36_model_name "${QWEN36_MODEL_NAME}" \
      --hierarchical_qwen36_request_timeout_s "${QWEN36_REQUEST_TIMEOUT_S}" \
      --hierarchical_qwen36_jpeg_quality "${QWEN36_JPEG_QUALITY}" \
      --hierarchical_qwen36_image_selection "${QWEN36_IMAGE_SELECTION}" \
      --hierarchical_qwen36_image_max_side "${QWEN36_IMAGE_MAX_SIDE}" \
      --hierarchical_history_fps 1.0 \
      --hierarchical_source_fps 25.0 \
      --hierarchical_recent_frames_length 8 \
      --hierarchical_memory_length 8 \
      --hierarchical_keyframe_merge_distance 5 \
      --hierarchical_log_path "${HL_LOG_PATH}" \
  2>&1 | sed -u 's/\r/\n/g' | sed -u '/step: /d'
) | tee "${LOG_PATH}"
