#!/usr/bin/env bash
# 当 nvcc 不支持 sm_120 时，用 PTX 编译 causal-conv1d 和 mamba-ssm，
# 使 RTX 5090 通过驱动 JIT 运行。若训练卡死请先执行: pkill -9 python
#
# 用法：cd /mnt/d/mamba && source .venv_wsl/bin/activate && bash scripts/wsl_build_mamba_ssm_ptx.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "========== 检查 nvcc =========="
if ! command -v nvcc &>/dev/null; then
  echo "错误: 未找到 nvcc。请先安装 CUDA Toolkit 并设置 PATH。"
  exit 1
fi
nvcc -V
echo ""

echo "========== 卸载旧包 =========="
pip uninstall mamba-ssm causal-conv1d -y 2>/dev/null || true
echo ""

echo "========== 清除 pip 缓存（防止用到旧 wheel） =========="
pip cache purge 2>/dev/null || true
echo ""

echo "========== 从源码编译（TORCH_CUDA_ARCH_LIST=8.0;8.6;9.0+PTX） =========="
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"
export PATH="/usr/local/cuda/bin:$PATH"

echo ">>> 强制从源码编译 causal-conv1d（应有大量编译输出，约 1–3 分钟；若几秒结束则未真正编译）..."
pip install causal-conv1d --no-binary causal-conv1d --force-reinstall --no-build-isolation --no-cache-dir -v
echo ">>> 强制从源码编译 mamba-ssm ..."
pip install mamba-ssm --no-binary mamba-ssm --force-reinstall --no-build-isolation --no-cache-dir -v

echo ""
echo "========== 完成 =========="
echo "编译正常约需 2–5 分钟；若几秒就结束，可能是用了缓存或未真正编译。"
echo "接着运行训练（开启 Fast Path）："
echo "  export MAMBA_FORCE_CUDA=1"
echo "  bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only"
echo "若仍报 no kernel image，不要设 MAMBA_FORCE_CUDA，用默认 reference 实现训练。"
