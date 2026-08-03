#!/usr/bin/env bash
set -euo pipefail

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_ROOT=${WORK_ROOT:-"$(cd "${HY_VLA_DIR}/.." && pwd)"}
DEFAULT_PYTHON="${WORK_ROOT}/envs/robotwin_hyvla_eval/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="${HY_VLA_DIR}/.venv/bin/python"
fi
PYTHON=${PYTHON:-"${DEFAULT_PYTHON}"}

# The 32B checkpoint is the quality-first default for a dedicated 80GB
# high-level GPU. Set QWEN_MODEL_ID/QWEN_MODEL_DIR for a 7B latency baseline.
QWEN_MODEL_ID=${QWEN_MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}
QWEN_MODEL_DIR=${QWEN_MODEL_DIR:-"${WORK_ROOT}/checkpoints/Qwen2.5-VL-32B-Instruct"}

mkdir -p \
  "${QWEN_MODEL_DIR}" \
  "${WORK_ROOT}/model_cache/huggingface" \
  "${WORK_ROOT}/.cache/tmp"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export HF_HOME="${WORK_ROOT}/model_cache/huggingface"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export TMPDIR="${WORK_ROOT}/.cache/tmp"

echo "Model ID            : ${QWEN_MODEL_ID}"
echo "Persistent location : ${QWEN_MODEL_DIR}"
echo "HF cache            : ${HF_HOME}"
echo "Python              : ${PYTHON}"

QWEN_MODEL_ID="${QWEN_MODEL_ID}" \
QWEN_MODEL_DIR="${QWEN_MODEL_DIR}" \
"${PYTHON}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = os.environ["QWEN_MODEL_ID"]
model_dir = Path(os.environ["QWEN_MODEL_DIR"])
snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Downloaded {model_id} into {model_dir}")
PY
