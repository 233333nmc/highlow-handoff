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
LOW_LEVEL_TRACE_PATH=${LOW_LEVEL_TRACE_PATH:-"${LOG_DIR}/low_level_instruction_trace.jsonl"}
LOW_LEVEL_TRACE_EVERY_STEP=${LOW_LEVEL_TRACE_EVERY_STEP:-False}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/robotwin_eval"}
ROBOTWIN_EVAL_RESULT_ROOT=${ROBOTWIN_EVAL_RESULT_ROOT:-"${ROBOTWIN_DIR}/eval_result"}
FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-5.0}
MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-1}
HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
ORACLE_STABILITY_STEPS=${ORACLE_STABILITY_STEPS:-3}
HANDOFF_PAUSE_STEPS=${HANDOFF_PAUSE_STEPS:-0}
STOP_AFTER_ORACLE_HANDOFF=${STOP_AFTER_ORACLE_HANDOFF:-False}
CALIBRATION_MAX_SIM_TIME_S=${CALIBRATION_MAX_SIM_TIME_S:-0.0}
STOP_AFTER_MEASUREMENT_HANDOFF=${STOP_AFTER_MEASUREMENT_HANDOFF:-False}
INSTRUCTION_MODE=${INSTRUCTION_MODE:-subtask}
HL_BACKEND=${HL_BACKEND:-scripted}
HL_PROMPT_MODE=${HL_PROMPT_MODE:-legacy}
HL_PLANNER_MODE=${HL_PLANNER_MODE:-inline_vlm}
HL_PHASE_PROMPTS_JSON=${HL_PHASE_PROMPTS_JSON:-}
HL_INCLUDE_PHASE_PROFILE_HINTS=${HL_INCLUDE_PHASE_PROFILE_HINTS:-True}
HL_ASYNC_ENABLED=${HL_ASYNC_ENABLED:-False}
HL_ASYNC_DROP_STALE_S=${HL_ASYNC_DROP_STALE_S:-0.0}
HL_FORCE_VISUAL_ON_DECISION=${HL_FORCE_VISUAL_ON_DECISION:-False}
HL_VISUAL_CONTEXT_MODE=${HL_VISUAL_CONTEXT_MODE:-recent}
HL_RECENT_FRAMES_LENGTH=${HL_RECENT_FRAMES_LENGTH:-8}
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
QWEN36_OUTPUT_SCHEMA=${QWEN36_OUTPUT_SCHEMA:-legacy}
QWEN36_MTP_SPECULATIVE_TOKENS=${QWEN36_MTP_SPECULATIVE_TOKENS:-1}
QWEN35_SERVER_URL=${QWEN35_SERVER_URL:-http://127.0.0.1:8001}
QWEN35_MODEL_NAME=${QWEN35_MODEL_NAME:-qwen3.5-9b-mtp}
QWEN35_REQUEST_TIMEOUT_S=${QWEN35_REQUEST_TIMEOUT_S:-120}
QWEN35_JPEG_QUALITY=${QWEN35_JPEG_QUALITY:-90}
QWEN35_IMAGE_SELECTION=${QWEN35_IMAGE_SELECTION:-latest}
QWEN35_IMAGE_MAX_SIDE=${QWEN35_IMAGE_MAX_SIDE:-0}
QWEN35_OUTPUT_SCHEMA=${QWEN35_OUTPUT_SCHEMA:-minimal_status}
QWEN35_MTP_SPECULATIVE_TOKENS=${QWEN35_MTP_SPECULATIVE_TOKENS:-2}
GEMMA4_SERVER_URL=${GEMMA4_SERVER_URL:-http://127.0.0.1:8002}
GEMMA4_MODEL_NAME=${GEMMA4_MODEL_NAME:-gemma-4-12B-it}
GEMMA4_REQUEST_TIMEOUT_S=${GEMMA4_REQUEST_TIMEOUT_S:-120}
GEMMA4_JPEG_QUALITY=${GEMMA4_JPEG_QUALITY:-90}
GEMMA4_IMAGE_SELECTION=${GEMMA4_IMAGE_SELECTION:-latest}
GEMMA4_IMAGE_MAX_SIDE=${GEMMA4_IMAGE_MAX_SIDE:-0}
GEMMA4_OUTPUT_SCHEMA=${GEMMA4_OUTPUT_SCHEMA:-minimal_status}

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
echo "Low-level trace    : ${LOW_LEVEL_TRACE_PATH}"
if [[ "${HANDOFF_TRIGGER}" == "forced_boundary" ]]; then
  echo "Forced boundary    : ${FORCED_BOUNDARY_S}s"
else
  echo "Forced boundary    : disabled (oracle-triggered handoff)"
fi
echo "Max handoffs       : ${MAX_FORCED_BOUNDARIES}"
echo "Handoff trigger    : ${HANDOFF_TRIGGER}"
echo "Oracle stability   : ${ORACLE_STABILITY_STEPS} steps"
echo "Handoff pause      : ${HANDOFF_PAUSE_STEPS} sim steps"
echo "Stop after oracle  : ${STOP_AFTER_ORACLE_HANDOFF}"
echo "Calibration limit  : ${CALIBRATION_MAX_SIM_TIME_S}s"
echo "Stop after measure : ${STOP_AFTER_MEASUREMENT_HANDOFF}"
echo "Instruction mode   : ${INSTRUCTION_MODE}"
echo "High-level backend : ${HL_BACKEND}"
echo "Prompt mode        : ${HL_PROMPT_MODE}"
echo "Planner mode       : ${HL_PLANNER_MODE}"
echo "Profile hints      : ${HL_INCLUDE_PHASE_PROFILE_HINTS}"
if [[ -n "${HL_PHASE_PROMPTS_JSON}" ]]; then
  echo "Phase prompts JSON : ${HL_PHASE_PROMPTS_JSON}"
fi
echo "Async high-level   : ${HL_ASYNC_ENABLED}, drop_stale=${HL_ASYNC_DROP_STALE_S}s"
echo "Force decision img : ${HL_FORCE_VISUAL_ON_DECISION}"
echo "Visual context     : ${HL_VISUAL_CONTEXT_MODE}"
echo "Recent frame window: ${HL_RECENT_FRAMES_LENGTH}"
if [[ "${HL_BACKEND}" == "qwen25_vl" ]]; then
  echo "Qwen checkpoint    : ${QWEN_MODEL_PATH}"
  echo "Qwen device        : ${QWEN_DEVICE}"
elif [[ "${HL_BACKEND}" == "qwen36_vllm_mtp" ]]; then
  echo "Qwen3.6 server     : ${QWEN36_SERVER_URL}"
  echo "Qwen3.6 model      : ${QWEN36_MODEL_NAME}"
  echo "Qwen3.6 images     : ${QWEN36_IMAGE_SELECTION}, max_side=${QWEN36_IMAGE_MAX_SIDE}"
  echo "Qwen3.6 schema     : ${QWEN36_OUTPUT_SCHEMA}"
  echo "Qwen3.6 MTP tokens : ${QWEN36_MTP_SPECULATIVE_TOKENS}"
elif [[ "${HL_BACKEND}" == "qwen35_vllm_mtp" ]]; then
  echo "Qwen3.5 server     : ${QWEN35_SERVER_URL}"
  echo "Qwen3.5 model      : ${QWEN35_MODEL_NAME}"
  echo "Qwen3.5 images     : ${QWEN35_IMAGE_SELECTION}, max_side=${QWEN35_IMAGE_MAX_SIDE}"
  echo "Qwen3.5 schema     : ${QWEN35_OUTPUT_SCHEMA}"
  echo "Qwen3.5 MTP tokens : ${QWEN35_MTP_SPECULATIVE_TOKENS}"
elif [[ "${HL_BACKEND}" == "gemma4_vllm" ]]; then
  echo "Gemma4 server      : ${GEMMA4_SERVER_URL}"
  echo "Gemma4 model       : ${GEMMA4_MODEL_NAME}"
  echo "Gemma4 images      : ${GEMMA4_IMAGE_SELECTION}, max_side=${GEMMA4_IMAGE_MAX_SIDE}"
  echo "Gemma4 schema      : ${GEMMA4_OUTPUT_SCHEMA}"
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
      --hierarchical_prompt_mode "${HL_PROMPT_MODE}" \
      --hierarchical_planner_mode "${HL_PLANNER_MODE}" \
      --hierarchical_phase_prompts_json "${HL_PHASE_PROMPTS_JSON}" \
      --hierarchical_include_phase_profile_hints "${HL_INCLUDE_PHASE_PROFILE_HINTS}" \
      --hierarchical_async_enabled "${HL_ASYNC_ENABLED}" \
      --hierarchical_async_drop_stale_s "${HL_ASYNC_DROP_STALE_S}" \
      --hierarchical_force_visual_on_decision "${HL_FORCE_VISUAL_ON_DECISION}" \
      --hierarchical_visual_context_mode "${HL_VISUAL_CONTEXT_MODE}" \
      --hierarchical_oracle_stability_steps "${ORACLE_STABILITY_STEPS}" \
      --hierarchical_handoff_pause_steps "${HANDOFF_PAUSE_STEPS}" \
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
      --hierarchical_qwen36_output_schema "${QWEN36_OUTPUT_SCHEMA}" \
      --hierarchical_qwen36_mtp_speculative_tokens "${QWEN36_MTP_SPECULATIVE_TOKENS}" \
      --hierarchical_qwen35_server_url "${QWEN35_SERVER_URL}" \
      --hierarchical_qwen35_model_name "${QWEN35_MODEL_NAME}" \
      --hierarchical_qwen35_request_timeout_s "${QWEN35_REQUEST_TIMEOUT_S}" \
      --hierarchical_qwen35_jpeg_quality "${QWEN35_JPEG_QUALITY}" \
      --hierarchical_qwen35_image_selection "${QWEN35_IMAGE_SELECTION}" \
      --hierarchical_qwen35_image_max_side "${QWEN35_IMAGE_MAX_SIDE}" \
      --hierarchical_qwen35_output_schema "${QWEN35_OUTPUT_SCHEMA}" \
      --hierarchical_qwen35_mtp_speculative_tokens "${QWEN35_MTP_SPECULATIVE_TOKENS}" \
      --hierarchical_gemma4_server_url "${GEMMA4_SERVER_URL}" \
      --hierarchical_gemma4_model_name "${GEMMA4_MODEL_NAME}" \
      --hierarchical_gemma4_request_timeout_s "${GEMMA4_REQUEST_TIMEOUT_S}" \
      --hierarchical_gemma4_jpeg_quality "${GEMMA4_JPEG_QUALITY}" \
      --hierarchical_gemma4_image_selection "${GEMMA4_IMAGE_SELECTION}" \
      --hierarchical_gemma4_image_max_side "${GEMMA4_IMAGE_MAX_SIDE}" \
      --hierarchical_gemma4_output_schema "${GEMMA4_OUTPUT_SCHEMA}" \
      --hierarchical_history_fps 1.0 \
      --hierarchical_source_fps 25.0 \
      --hierarchical_recent_frames_length "${HL_RECENT_FRAMES_LENGTH}" \
      --hierarchical_memory_length 8 \
      --hierarchical_log_path "${HL_LOG_PATH}" \
      --low_level_instruction_trace_path "${LOW_LEVEL_TRACE_PATH}" \
      --low_level_instruction_trace_every_step "${LOW_LEVEL_TRACE_EVERY_STEP}" \
  2>&1 | sed -u 's/\r/\n/g' | sed -u '/step: /d'
) | tee "${LOG_PATH}"
