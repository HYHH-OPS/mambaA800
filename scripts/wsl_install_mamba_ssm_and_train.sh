#!/usr/bin/env bash
# WSL：安装 mamba-ssm 并运行 Stage 2 训练。
# 用法：cd /mnt/d/mamba 后执行  bash scripts/wsl_install_mamba_ssm_and_train.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
[ -f "$REPO_DIR/.venv_wsl/bin/activate" ] && source "$REPO_DIR/.venv_wsl/bin/activate"

INSTALL_ONLY=false
TRAIN_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --install-only) INSTALL_ONLY=true ;;
    --train-only)   TRAIN_ONLY=true ;;
  esac
done

if [ "$TRAIN_ONLY" = false ]; then
  echo "========== 检查/安装 mamba-ssm、causal-conv1d =========="
  if python3 -c "import mamba_ssm" 2>/dev/null; then
    echo "mamba_ssm 已安装，跳过"
  else
    pip install mamba-ssm causal-conv1d
    echo "mamba_ssm / causal-conv1d 安装完成"
  fi
  [ "$INSTALL_ONLY" = true ] && echo "仅安装模式，退出" && exit 0
fi

export KMP_DUPLICATE_LIB_OK=TRUE
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi:" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

echo "========== Stage 2 训练 =========="
# 若 WSL 无法访问 Hugging Face，可先下载 Mamba 到本地再设置 MAMBA_MODEL，见 docs/WSL_TRAIN.md 第九节
EXTRA_ARGS=()
[ -n "$MAMBA_MODEL" ] && EXTRA_ARGS+=(--mamba_model "$MAMBA_MODEL")
python3 train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96 --max_text_len 512 "${EXTRA_ARGS[@]}"

echo "========== 完成 =========="
echo "Checkpoint: outputs/vision_bridge_vlm_final.pt"
