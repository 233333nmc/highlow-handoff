#!/usr/bin/env bash
set -euo pipefail

# Download Google's instruction-tuned Gemma 4 12B checkpoint into the user's
# project directory. The model is open weights, but Hugging Face may require
# accepting the model terms and providing HF_TOKEN.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

HF_REPO=${GEMMA4_HF_REPO:-google/gemma-4-12B-it}
MODEL_DIR=${GEMMA4_MODEL_PATH:-"${WORK_ROOT}/checkpoints/google/gemma-4-12B-it"}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${WORK_ROOT}/runtime_cache/huggingface_gemma4"}

mkdir -p \
  "${MODEL_DIR}" \
  "${RUNTIME_CACHE_ROOT}/scratch" \
  "${RUNTIME_CACHE_ROOT}/hub"

export HF_HOME="${RUNTIME_CACHE_ROOT}"
export HF_HUB_CACHE="${RUNTIME_CACHE_ROOT}/hub"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export TMPDIR="${RUNTIME_CACHE_ROOT}/scratch"

echo "Gemma4 HF repo  : ${HF_REPO}"
echo "Local model dir : ${MODEL_DIR}"
echo "HF cache root   : ${HF_HOME}"

if command -v hf >/dev/null 2>&1; then
  hf download "${HF_REPO}" \
    --local-dir "${MODEL_DIR}"
else
  huggingface-cli download "${HF_REPO}" \
    --local-dir "${MODEL_DIR}" \
    --local-dir-use-symlinks False
fi
