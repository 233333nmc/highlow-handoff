#!/usr/bin/env bash
set -euo pipefail

HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
WORK_ROOT=${WORK_ROOT:-"$(cd "${HY_VLA_DIR}/.." && pwd)"}
EVAL_VENV=${EVAL_VENV:-"${WORK_ROOT}/envs/robotwin_hyvla_eval"}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.10}
PYPI_INDEX=${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}
PYTORCH_INDEX=${PYTORCH_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/cu128/}
TORCH_WHEEL=${TORCH_WHEEL:-https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl}
TORCHVISION_WHEEL=${TORCHVISION_WHEEL:-https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl}
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}
THIRD_PARTY_DIR=${THIRD_PARTY_DIR:-"${WORK_ROOT}/third_party"}
CUROBO_DIR=${CUROBO_DIR:-"${THIRD_PARTY_DIR}/curobo"}

mkdir -p \
  "${WORK_ROOT}/envs" \
  "${THIRD_PARTY_DIR}" \
  "${WORK_ROOT}/.cache/tmp" \
  "${WORK_ROOT}/.cache/matplotlib" \
  "${WORK_ROOT}/.cache/uv" \
  "${WORK_ROOT}/.cache/pip" \
  "${WORK_ROOT}/model_cache/huggingface"

export TMPDIR="${WORK_ROOT}/.cache/tmp"
export MPLCONFIGDIR="${WORK_ROOT}/.cache/matplotlib"
export UV_CACHE_DIR="${WORK_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${WORK_ROOT}/.cache/pip"
export HF_HOME="${WORK_ROOT}/model_cache/huggingface"

echo "Hy-VLA dir       : ${HY_VLA_DIR}"
echo "Eval venv        : ${EVAL_VENV}"
echo "Python           : ${PYTHON_BIN}"
echo "PyPI index       : ${PYPI_INDEX}"
echo "PyTorch index    : ${PYTORCH_INDEX}"
echo "Third-party dir  : ${THIRD_PARTY_DIR}"
echo "Persistent TMPDIR: ${TMPDIR}"

# Build a separate Python 3.10 unified eval environment. RoboTwin and Hy-VLA
# run inside one Python process during evaluation because RoboTwin imports
# policy.hy_vla, which imports hy_vla and loads the checkpoint. This is not
# the Hy-VLA training env; it is a joint runtime env for RoboTwin simulation
# plus Hy-VLA inference.
uv venv "${EVAL_VENV}" --python "${PYTHON_BIN}" --allow-existing

echo "[1/5] Installing torch/torchvision cu128 wheels..."
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  --index "${PYTORCH_INDEX}" \
  --index-strategy unsafe-best-match \
  "setuptools<81" \
  "numpy>=1.24,<2.0" \
  "${TORCH_WHEEL}" \
  "${TORCHVISION_WHEEL}"

echo "[2/5] Installing flash-attn prebuilt wheel..."
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  "${FLASH_ATTN_WHEEL}"

echo "[3/5] Installing minimal Hy-VLA runtime dependencies..."
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  "numpy>=1.24,<2.0" \
  "transformers==4.57.0" \
  "accelerate==1.12.0" \
  "safetensors>=0.4" \
  "huggingface-hub==0.36.2" \
  "timm==1.0.21" \
  "scipy>=1.11" \
  "h5py>=3.10" \
  "Pillow>=10.0" \
  "opencv-python-headless>=4.8" \
  "imageio>=2.35" \
  "imageio-ffmpeg>=0.5" \
  "matplotlib>=3.7" \
  "tqdm>=4.66"

echo "[4/5] Installing Hy-VLA package in editable mode without training deps..."
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  --no-deps \
  --editable "${HY_VLA_DIR}"

# Do not install RoboTwin/script/requirements.txt directly:
# - it pins torch==2.4.1, while Hy-VLA needs torch>=2.7,<2.8;
# - it pins scipy==1.10.1, while Hy-VLA needs scipy>=1.11.
#
# The list below contains the RoboTwin simulator/runtime packages that are
# missing from a clean Hy-VLA env. If a package is already installed by Hy-VLA,
# leave Hy-VLA's version in place.
echo "[5/5] Installing RoboTwin simulator/runtime dependencies..."
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  --index "${PYTORCH_INDEX}" \
  --index-strategy unsafe-best-match \
  "sapien==3.0.0b1" \
  "mplib==0.2.1" \
  "gymnasium==0.29.1" \
  "trimesh==4.4.3" \
  "open3d==0.18.0" \
  "transforms3d==0.4.2" \
  "toppra" \
  "zarr" \
  "openai" \
  "pyglet<2" \
  "moviepy" \
  "termcolor" \
  "av" \
  "ffmpeg"

echo "[extra] Installing RoboTwin cuRobo planner dependency..."
if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
fi
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  --editable "${CUROBO_DIR}" \
  --no-build-isolation
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --default-index "${PYPI_INDEX}" \
  "warp-lang==1.12.0" \
  "setuptools==69.5.1"

"${EVAL_VENV}/bin/python" - <<'PY'
import importlib.util

mods = [
    "torch",
    "flash_attn",
    "hy_vla",
    "sapien",
    "gymnasium",
    "mplib",
    "toppra",
    "transforms3d",
    "trimesh",
    "open3d",
    "zarr",
    "moviepy",
    "termcolor",
    "av",
    "curobo",
    "warp",
]
missing = [name for name in mods if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing packages after install: {missing}")

import torch
import torchvision

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("import check ok")
PY
