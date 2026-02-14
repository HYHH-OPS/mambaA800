$ErrorActionPreference = "Stop"
Set-Location "D:\mamba"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:MAMBA_FORCE_CUDA) { $env:MAMBA_FORCE_CUDA = "0" }

python scripts/infer_template_strict.py `
  --checkpoint d:/mamba/outputs/template_hq/vision_bridge_vlm_final.pt `
  --mamba_model d:/mamba/models/mamba-2.8b-hf `
  --val_sample `
  --num_val 20 `
  --max_new_tokens 640 `
  --max_visual_tokens 64 `
  --min_chars 180 `
  --max_retries 2 `
  --do_sample `
  --temperature 0.7 `
  --repetition_penalty 1.15 `
  --length_penalty 1.15 `
  --out_dir d:/mamba-res

# Replace with the generated run_strict_* directory.
# python scripts/evaluate_generation_run.py --run_dir d:/mamba-res/run_strict_YYYYMMDD_HHMMSS
