# 修复乱码 + VLM 训练 + 生成（按顺序执行）
# 用法: cd D:\mamba ; conda activate mamba5090 ; .\run_fix_and_train_then_generate.ps1

$ErrorActionPreference = "Stop"
Set-Location (if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot })

Write-Host "========== 1. VLM 训练（2 epoch，caption 损失）==========" -ForegroundColor Cyan
python train_vlm.py --epochs 2 --batch_size 1 --lr 1e-5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== 2. 图像→文本生成检测 ==========" -ForegroundColor Cyan
python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 196 --max_new_tokens 512 --llm_device auto
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== 完成 ==========" -ForegroundColor Green
Write-Host "若无 VLM 权重则用: python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_best_val.pt" -ForegroundColor Gray
