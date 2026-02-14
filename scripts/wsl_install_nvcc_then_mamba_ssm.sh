#!/usr/bin/env bash
# 在 WSL 中：先安装 CUDA Toolkit（提供 nvcc），再安装 mamba-ssm，以启用训练/推理加速。
# 用法：cd /mnt/d/mamba 后执行  bash scripts/wsl_install_nvcc_then_mamba_ssm.sh
# 需要 sudo 密码。安装完成后请  source .venv_wsl/bin/activate 再训练。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "========== 1. 安装 CUDA Toolkit（nvcc）=========="
echo "添加 NVIDIA WSL-Ubuntu 仓库并安装 cuda-toolkit-12-6 ..."
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc 已存在: $(nvcc -V 2>/dev/null | head -1)"
  read -p "是否跳过 CUDA 安装并直接安装 mamba-ssm? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    SKIP_CUDA=""
  else
    SKIP_CUDA=1
  fi
fi

if [ -z "$SKIP_CUDA" ]; then
  # NVIDIA 官方 WSL-Ubuntu 仓库（CUDA 12）
  if [ ! -f /usr/share/keyrings/cuda-*-keyring.gpg ] 2>/dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
  fi
  sudo apt-get update -qq
  sudo apt-get install -y cuda-toolkit-12-6 || sudo apt-get install -y cuda-toolkit-12-4 || sudo apt-get install -y cuda-toolkit
  export PATH=/usr/local/cuda/bin${PATH:+:$PATH}
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "请在新开终端中执行: export PATH=/usr/local/cuda/bin:\$PATH"
    echo "或将以下行加入 ~/.bashrc: export PATH=/usr/local/cuda/bin:\$PATH"
  fi
fi

echo "========== 2. 激活虚拟环境并安装 mamba-ssm =========="
[ -f "$REPO_DIR/.venv_wsl/bin/activate" ] && source "$REPO_DIR/.venv_wsl/bin/activate" || true
export PATH=/usr/local/cuda/bin${PATH:+:$PATH}

if python3 -c "import mamba_ssm" 2>/dev/null; then
  echo "mamba_ssm 已安装，跳过"
else
  pip install mamba-ssm causal-conv1d
  echo "mamba_ssm / causal-conv1d 安装完成"
fi

echo "========== 完成 =========="
echo "请将 CUDA 加入 PATH（若尚未加入）："
echo "  export PATH=/usr/local/cuda/bin:\$PATH"
echo "  echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc"
echo "然后训练："
echo "  cd $REPO_DIR && source .venv_wsl/bin/activate"
echo "  bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only"
