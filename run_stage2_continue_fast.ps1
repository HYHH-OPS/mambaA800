# 续训 + 提速：max_visual_tokens=64，序列更短、每步更快（推理时须同样 --max_visual_tokens 64）
$ErrorActionPreference = "Stop"
if (Test-Path "D:\mamba") { Set-Location "D:\mamba" } else { Set-Location $PSScriptRoot }
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "========== Stage 2: 续训（快速，max_visual_tokens=64）==========" -ForegroundColor Cyan
Write-Host "推理时请加: --max_visual_tokens 64" -ForegroundColor Yellow
Write-Host ""

python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== Done ==========" -ForegroundColor Green
Write-Host "Test: python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 64 --llm_device auto" -ForegroundColor Gray
