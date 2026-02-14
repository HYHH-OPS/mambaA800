$ErrorActionPreference = "Stop"
Set-Location "D:\mamba"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"

# If you have rebuilt mamba/causal-conv kernels for sm_120, set to 1 for much faster speed.
# Otherwise keep 0 to avoid CUDA kernel image errors.
if (-not $env:MAMBA_FORCE_CUDA) { $env:MAMBA_FORCE_CUDA = "0" }

python train_vlm.py `
  --vision_checkpoint d:/mamba/outputs/struct/vision_bridge_vlm_final.pt `
  --csv d:/mamba/outputs/excel_caption/caption_train_template_shortq.csv `
  --mamba_model d:/mamba/models/mamba-2.8b-hf `
  --epochs 30 `
  --batch_size 1 `
  --lr 1e-5 `
  --max_visual_tokens 64 `
  --max_text_len 768 `
  --gradient_accumulation_steps 4 `
  --log_every_steps 10 `
  --save_every_steps 200 `
  --num_workers 0 `
  --align_vocab `
  --length_audit_samples 257 `
  --output_dir d:/mamba/outputs/template_hq
