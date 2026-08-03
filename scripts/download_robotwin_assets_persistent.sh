#!/usr/bin/env bash
set -euo pipefail

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_ROOT=${WORK_ROOT:-"$(cd "${HY_VLA_DIR}/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-"${WORK_ROOT}/RoboTwin"}
DEFAULT_PYTHON="${WORK_ROOT}/envs/robotwin_hyvla_eval/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="${HY_VLA_DIR}/.venv/bin/python"
fi
PYTHON=${PYTHON:-"${DEFAULT_PYTHON}"}

mkdir -p \
  "${WORK_ROOT}/.cache/tmp" \
  "${WORK_ROOT}/.cache/matplotlib" \
  "${WORK_ROOT}/.cache/xdg" \
  "${WORK_ROOT}/model_cache/huggingface" \
  "${ROBOTWIN_DIR}/assets"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export TMPDIR="${WORK_ROOT}/.cache/tmp"
export MPLCONFIGDIR="${WORK_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${WORK_ROOT}/.cache/xdg"
export HF_HOME="${WORK_ROOT}/model_cache/huggingface"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}

echo "RoboTwin dir     : ${ROBOTWIN_DIR}"
echo "Python           : ${PYTHON}"
echo "Persistent TMPDIR: ${TMPDIR}"
echo "HF_HOME          : ${HF_HOME}"
echo "HF_HUB_DISABLE_XET: ${HF_HUB_DISABLE_XET}"

(
  cd "${ROBOTWIN_DIR}/assets"
  "${PYTHON}" _download.py
  unzip -n background_texture.zip
  unzip -n embodiments.zip
  unzip -n objects.zip
)

(
  cd "${ROBOTWIN_DIR}"
  "${PYTHON}" ./script/update_embodiment_config_path.py
)

echo "RoboTwin assets are ready."
