# Stage 2 稳健模式：关闭 CMI/RoI 后使用，显存约 12-16GB，无 8-bit 依赖
# 前置：config/paths.yaml 中 use_cmi: false, roi_side: null
# 若显存仍满/僵尸进程：先执行 taskkill /f /im python.exe 再运行本脚本
# 用法: .\run_stage2_stable.ps1  或直接复制下面等效命令到终端
$ErrorActionPreference = "Stop"
if (Test-Path "D:\mamba") { Set-Location "D:\mamba" } else { Set-Location $PSScriptRoot }

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host '========== Stage 2: stable (from_scratch + grad accum) ==========' -ForegroundColor Cyan
Write-Host 'batch_size=1, gradient_accumulation_steps=8, epochs=10. Ensure paths.yaml use_cmi: false' -ForegroundColor Yellow
Write-Host ''

$mambaArg = @()
if ($env:MAMBA_MODEL) { $mambaArg = @('--mamba_model', $env:MAMBA_MODEL) }

python train_vlm.py --from_scratch --batch_size 1 --gradient_accumulation_steps 8 --epochs 10 --lr 2e-5 @mambaArg
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host '========== Done ==========' -ForegroundColor Green
Write-Host 'Test: python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --llm_device auto' -ForegroundColor Gray
