#!/usr/bin/env bash
# WSL 首次环境：安装 Python 依赖 + mamba-ssm。
# 在项目根目录执行：  bash scripts/wsl_setup_env.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "========== 1. 系统包 =========="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev || true

echo "========== 2. 虚拟环境 =========="
if [ ! -d "$REPO_DIR/.venv_wsl" ]; then
  python3 -m venv "$REPO_DIR/.venv_wsl"
  echo "已创建 .venv_wsl"
fi
source "$REPO_DIR/.venv_wsl/bin/activate"

echo "========== 3. PyTorch (CUDA) =========="
pip install --upgrade pip
# 若为 RTX 5090/Blackwell (sm_120)，请改用 nightly cu128，见 docs/WSL_TRAIN.md 第八节
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "========== 4. 项目依赖 =========="
pip install -r requirements.txt

echo "========== 5. mamba-ssm + causal-conv1d（可选，需 nvcc）=========="
pip install mamba-ssm causal-conv1d || true
if python3 -c "import mamba_ssm" 2>/dev/null; then
  echo "mamba_ssm 已安装，训练将使用 CUDA 加速"
else
  echo "未安装 mamba_ssm（缺 nvcc），将使用 sequential 实现，训练可正常进行但较慢"
fi

echo "========== 完成 =========="
echo "下次进入 WSL 后："
echo "  cd /mnt/d/mamba"
echo "  bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only"
