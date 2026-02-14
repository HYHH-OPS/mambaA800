# 夜间 300 轮 + 超参筛选（睡觉前执行）
# 会依次跑 4 组参数 (lr × batch_size)，每组 300 轮，早上看 outputs/overnight_best_summary.txt
# 用法: cd D:\mamba ; conda activate mamba5090 ; .\run_overnight_300.ps1

$MambaRoot = if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot }
Set-Location $MambaRoot
Write-Host "========== 夜间 300 轮训练 + 超参筛选 ==========" -ForegroundColor Cyan
Write-Host "请确保已: conda activate mamba5090" -ForegroundColor Yellow
Write-Host ""

& python scripts/run_overnight_300.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "早上查看: D:\mamba\outputs\overnight_best_summary.txt" -ForegroundColor Green
