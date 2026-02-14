#!/usr/bin/env bash
# 在 WSL 中为 RTX 5090 (sm_120) 从源码编译 causal-conv1d 和 mamba-ssm，以启用 Fast Path。
# 使用前请确保已安装 CUDA Toolkit 12.8 且 nvcc 在 PATH 中。详见 docs/WSL_TRAIN.md 第十节。
#
# 用法：cd /mnt/d/mamba && source .venv_wsl/bin/activate && bash scripts/wsl_build_mamba_ssm_sm120.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "========== 检查 nvcc =========="
if ! command -v nvcc &>/dev/null; then
  echo "错误: 未找到 nvcc。请先安装 CUDA Toolkit 12.8 并设置 PATH，例如："
  echo "  export PATH=/usr/local/cuda/bin:\$PATH"
  echo "  nvcc -V"
  exit 1
fi
nvcc -V
echo ""

echo "========== 检查 PyTorch CUDA =========="
python3 -c "
import torch
assert torch.cuda.is_available(), 'PyTorch 未检测到 CUDA'
arch = torch.cuda.get_arch_list()
assert 'sm_120' in arch, f'当前 PyTorch 不支持 sm_120，arch_list={arch}。请安装 PyTorch nightly cu128，见 docs/WSL_TRAIN.md 第八节。'
print('PyTorch arch_list:', arch)
"
echo ""

echo "========== 卸载旧包 =========="
pip uninstall mamba-ssm causal-conv1d -y 2>/dev/null || true
echo ""

echo "========== 从源码编译（TORCH_CUDA_ARCH_LIST=12.0） =========="
export TORCH_CUDA_ARCH_LIST="12.0"
export PATH="/usr/local/cuda/bin:$PATH"

echo ">>> 编译 causal-conv1d ..."
pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation -v
echo ">>> 编译 mamba-ssm ..."
pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation -v

echo ""
echo "========== 完成 =========="
echo "请重新运行训练。若不再出现 'Falling back to the sequential implementation'，则 Fast Path 已启用。"
echo "可选：删除 train_vlm.py 中 sm_120 的 reference 回退补丁（约第 22–38 行），使 bridge 也走 CUDA。"
