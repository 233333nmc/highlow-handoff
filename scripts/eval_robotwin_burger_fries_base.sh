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
LOG_DIR=${LOG_DIR:-"${WORK_ROOT}/profile_results/robotwin_burger_fries"}
LOG_PATH=${LOG_PATH:-"${LOG_DIR}/base.log"}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/robotwin_base"}

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

echo "RoboTwin task       : place_burger_fries"
echo "Mode               : base Hy-VLA"
echo "Rollouts           : ${TEST_NUM}"
echo "Checkpoint         : ${CKPT_PATH}"
echo "RoboTwin dir       : ${ROBOTWIN_DIR}"
echo "Python             : ${PYTHON}"
echo "Log                : ${LOG_PATH}"

(
  cd "${ROBOTWIN_DIR}"
  PYTHONWARNINGS=ignore::UserWarning \
  "${PYTHON}" -u script/eval_policy.py \
    --config policy/hy_vla/deploy_policy.yml \
    --overrides \
      --task_name place_burger_fries \
      --task_config "${TASK_CONFIG}" \
      --ckpt_setting Hy-Embodied-0.5-VLA-RoboTwin \
      --instruction_type "${INSTRUCTION_TYPE}" \
      --seed "${SEED}" \
      --test_num "${TEST_NUM}" \
      --policy_name policy.hy_vla \
      --ckpt_path "${CKPT_PATH}" \
      --hierarchical_enabled False \
  2>&1 | sed -u 's/\r/\n/g' | sed -u '/step: /d'
) | tee "${LOG_PATH}"
