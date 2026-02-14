# Stage 2 VLM caption training. Run in mamba5090. Check CUDA: python scripts/verify_torch_cuda.py

$ErrorActionPreference = "Stop"
if (Test-Path "D:\mamba") { Set-Location "D:\mamba" } else { Set-Location $PSScriptRoot }
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "========== Stage 2: VLM training ==========" -ForegroundColor Cyan
Write-Host "BF16 + num_workers=4 + max_visual_tokens=96" -ForegroundColor Gray
Write-Host "batch_size=1 for stable 32GB VRAM; try --batch_size 2 if you have headroom" -ForegroundColor Gray
Write-Host "Watch caption_loss: aim below 2.0 for readable reports" -ForegroundColor Gray
Write-Host "From scratch (no old vision checkpoint) to avoid bad Stage1 weights" -ForegroundColor Gray
Write-Host "提速与 mamba_ssm 安装: 见 docs/SPEED_AND_MAMBA_SSM.md" -ForegroundColor DarkGray
Write-Host ""

python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96 --from_scratch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== Done ==========" -ForegroundColor Green
Write-Host "Checkpoint: outputs/vision_bridge_vlm_final.pt" -ForegroundColor Gray
Write-Host "Test: python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --llm_device auto --max_new_tokens 512" -ForegroundColor Gray
