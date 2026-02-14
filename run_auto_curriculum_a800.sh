#!/usr/bin/env bash
# AutoDL A800 one-click auto curriculum training (Stage A -> Stage B -> strict validation).
#
# Usage:
#   bash run_auto_curriculum_a800.sh
#   EPOCH_A=2 EPOCH_B=20 BATCH_SIZE=2 GRAD_ACC=2 bash run_auto_curriculum_a800.sh
#   REPO_ROOT=/root/autodl-tmp/mamba PYTHON_EXE=python bash run_auto_curriculum_a800.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
EPOCH_A="${EPOCH_A:-2}"
EPOCH_B="${EPOCH_B:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
TRAIN_CSV="${TRAIN_CSV:-$REPO_ROOT/outputs/excel_caption/caption_train_template_shortq.csv}"
MAMBA_MODEL="${MAMBA_MODEL:-$REPO_ROOT/models/mamba-2.8b-hf}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/auto_curriculum_a800}"
RES_OUT="${RES_OUT:-$REPO_ROOT/mamba-res}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# A800 (sm_80) can use native CUDA fast path, keep this disabled unless you explicitly need override.
export MAMBA_FORCE_CUDA="${MAMBA_FORCE_CUDA:-0}"

cd "$REPO_ROOT"

STAGE_A_OUT="$OUT_ROOT/stageA_36_256"
STAGE_B_OUT="$OUT_ROOT/stageB_64_512"
mkdir -p "$STAGE_A_OUT" "$STAGE_B_OUT" "$RES_OUT"

run_train_stage() {
  local name="$1"
  local output_dir="$2"
  local epochs="$3"
  local max_visual_tokens="$4"
  local max_text_len="$5"
  local vision_checkpoint="${6:-}"

  local final_ckpt="$output_dir/vision_bridge_vlm_final.pt"
  if [[ -f "$final_ckpt" ]]; then
    echo "[skip] $name already complete: $final_ckpt"
    return 0
  fi

  local resume_ckpt=""
  if [[ -n "$vision_checkpoint" && -f "$vision_checkpoint" ]]; then
    resume_ckpt="$vision_checkpoint"
  else
    local latest_step
    latest_step="$(ls -1t "$output_dir"/vision_bridge_vlm_step*.pt 2>/dev/null | head -n 1 || true)"
    if [[ -n "$latest_step" ]]; then
      resume_ckpt="$latest_step"
      echo "[resume] $name from latest step checkpoint: $resume_ckpt"
    fi
  fi

  local cmd=(
    "$PYTHON_EXE" train_vlm.py
    --csv "$TRAIN_CSV"
    --mamba_model "$MAMBA_MODEL"
    --epochs "$epochs"
    --batch_size "$BATCH_SIZE"
    --lr 1e-5
    --max_visual_tokens "$max_visual_tokens"
    --max_text_len "$max_text_len"
    --gradient_accumulation_steps "$GRAD_ACC"
    --gradient_checkpointing
    --align_vocab
    --num_workers 4
    --log_every_steps 1
    --save_every_steps 100
    --output_dir "$output_dir"
  )
  if [[ -n "$resume_ckpt" ]]; then
    cmd+=(--vision_checkpoint "$resume_ckpt")
  fi

  echo "[run] $name => max_visual_tokens=$max_visual_tokens max_text_len=$max_text_len epochs=$epochs batch_size=$BATCH_SIZE grad_acc=$GRAD_ACC"
  "${cmd[@]}"

  if [[ ! -f "$final_ckpt" ]]; then
    echo "[error] $name finished but checkpoint not found: $final_ckpt" >&2
    return 1
  fi
}

echo "[info] repo=$REPO_ROOT"
echo "[info] python=$PYTHON_EXE"
echo "[info] train_csv=$TRAIN_CSV"
echo "[info] mamba_model=$MAMBA_MODEL"

run_train_stage "Stage A (survival)" "$STAGE_A_OUT" "$EPOCH_A" 36 256
STAGE_A_CKPT="$STAGE_A_OUT/vision_bridge_vlm_final.pt"
run_train_stage "Stage B (quality)" "$STAGE_B_OUT" "$EPOCH_B" 64 512 "$STAGE_A_CKPT"

if [[ "$SKIP_VALIDATE" != "1" ]]; then
  FINAL_CKPT="$STAGE_B_OUT/vision_bridge_vlm_final.pt"
  echo "[run] strict validation => min_chars=180 max_new_tokens=640 constrained_decode"
  "$PYTHON_EXE" scripts/infer_template_strict.py \
    --checkpoint "$FINAL_CKPT" \
    --mamba_model "$MAMBA_MODEL" \
    --val_sample \
    --num_val 20 \
    --max_new_tokens 640 \
    --max_visual_tokens 64 \
    --min_chars 180 \
    --max_retries 2 \
    --num_beams 4 \
    --constrained_decode \
    --length_penalty 1.2 \
    --repetition_penalty 1.15 \
    --out_dir "$RES_OUT"

  latest_run="$(ls -1dt "$RES_OUT"/run_strict_* 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_run" ]]; then
    "$PYTHON_EXE" scripts/evaluate_generation_run.py --run_dir "$latest_run"
  fi
fi

echo "[done] Auto curriculum pipeline completed on A800."
