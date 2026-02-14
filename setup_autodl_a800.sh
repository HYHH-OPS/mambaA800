#!/usr/bin/env bash
# Prepare AutoDL A800 environment for this repo.
#
# Usage:
#   bash setup_autodl_a800.sh
# Optional env:
#   ENV_NAME=mamba_a800 PYTHON_VER=3.10 REPO_ROOT=/root/autodl-tmp/mamba bash setup_autodl_a800.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-mamba_a800}"
PYTHON_VER="${PYTHON_VER:-3.10}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "========== AutoDL A800 env setup =========="
echo "[info] env=$ENV_NAME python=$PYTHON_VER repo=$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found. Please use AutoDL image with conda preinstalled." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[skip] conda env '$ENV_NAME' already exists."
else
  echo "[1/5] creating conda env '$ENV_NAME'..."
  conda create -n "$ENV_NAME" "python=$PYTHON_VER" -y
fi

echo "[2/5] installing PyTorch (CUDA 12.1 wheels)..."
conda run -n "$ENV_NAME" --no-capture-output pip install --upgrade pip
conda run -n "$ENV_NAME" --no-capture-output pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[3/5] installing project requirements..."
conda run -n "$ENV_NAME" --no-capture-output pip install -r "$REPO_ROOT/requirements.txt"

echo "[4/5] optional acceleration: mamba-ssm + causal-conv1d (best effort)..."
if ! conda run -n "$ENV_NAME" --no-capture-output pip install mamba-ssm causal-conv1d; then
  echo "[warn] optional package install failed; training can still run with slower fallback."
fi

echo "[5/5] verify CUDA and GPU..."
conda run -n "$ENV_NAME" --no-capture-output python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo ""
echo "========== done =========="
echo "Next:"
echo "  conda activate $ENV_NAME"
echo "  cd $REPO_ROOT"
echo "  bash run_auto_curriculum_a800.sh"
