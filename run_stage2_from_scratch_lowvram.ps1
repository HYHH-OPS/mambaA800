# Stage 2 从零初始化 + 低显存：解决「看图不准」与显存溢出导致的生成截断
# 用法: .\run_stage2_from_scratch_lowvram.ps1
# 默认不用 8bit（避免 Windows 上 bitsandbytes 依赖）；需 8bit 时先设 $env:LLM_8BIT = "1" 并 pip install bitsandbytes
$ErrorActionPreference = "Stop"
if (Test-Path "D:\mamba") { Set-Location "D:\mamba" } else { Set-Location $PSScriptRoot }

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "========== Stage 2: 从零初始化 + 低显存（防截断）==========" -ForegroundColor Cyan
Write-Host "from_scratch + max_visual_tokens=64, max_text_len=256 + gradient_accumulation_steps=4（等效 batch 4，显存约 1/4）" -ForegroundColor Yellow
$llm8Arg = @()
if ($env:LLM_8BIT -eq "1") { $llm8Arg = @("--llm_8bit"); Write-Host "LLM_8BIT=1: 启用 8-bit 加载" -ForegroundColor Gray }
Write-Host ""

$mambaArg = @()
if ($env:MAMBA_MODEL) { $mambaArg = @("--mamba_model", $env:MAMBA_MODEL) }
# 使用字面量 30 避免 PowerShell 变量展开为空
$trainArgs = @(
    "--from_scratch", "--max_visual_tokens", "64", "--max_text_len", "256",
    "--batch_size", "1", "--gradient_accumulation_steps", "4",
    "--epochs", "30", "--lr", "2e-5"
)
if ($env:STAGE2_EPOCHS) { $trainArgs[9] = $env:STAGE2_EPOCHS }
python train_vlm.py @trainArgs @mambaArg @llm8Arg
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== Done ==========" -ForegroundColor Green
Write-Host "推理时请用: --max_visual_tokens 64（若训练时用了 LLM_8BIT 则再加 --llm_8bit）" -ForegroundColor Gray
Write-Host "Test: python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 64 --llm_device auto" -ForegroundColor Gray
