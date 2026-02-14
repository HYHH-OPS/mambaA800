#!/usr/bin/env bash
# Quick preflight for AutoDL A800 deployment.
# Usage:
#   bash scripts/autodl_preflight.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[preflight] repo: $REPO_ROOT"

need_paths=(
  "train_vlm.py"
  "inference.py"
  "config/paths.yaml"
  "data/medical_vlm_dataset.py"
  "model/forward_medical_vlm.py"
  "llm/mamba_loader.py"
  "run_auto_curriculum_a800.sh"
  "setup_autodl_a800.sh"
  "scripts/infer_template_strict.py"
)

missing=0
for p in "${need_paths[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "[missing] $p"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "[fail] code files are incomplete. push full repo first."
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu] nvidia-smi:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "[warn] nvidia-smi not found."
fi

if command -v python >/dev/null 2>&1; then
  echo "[python] version:"
  python -V
  python - << 'PY'
import torch
print("[torch]", torch.__version__)
print("[cuda_available]", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[gpu0]", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
else
  echo "[warn] python not found in PATH."
fi

echo "[hint] optional data/model checks:"
echo "  ls outputs/excel_caption/caption_train_template_shortq.csv"
echo "  ls outputs/excel_caption/caption_val_template_shortq.csv"
echo "  ls models/mamba-2.8b-hf/config.json"
echo "[ok] preflight finished."
