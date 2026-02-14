# 从 vision_bridge_vlm_final.pt 继续 Stage 2，不 from_scratch；与 run_stage2_train.ps1 的 96 一致
# 提速：改用 --max_visual_tokens 64 可更快（推理时须同样加 --max_visual_tokens 64）；安装 mamba_ssm 可大幅加速，见 docs/SPEED_AND_MAMBA_SSM.md
$ErrorActionPreference = "Stop"
if (Test-Path "D:\mamba") { Set-Location "D:\mamba" } else { Set-Location $PSScriptRoot }

# 避免 OpenMP 冲突：多个库自带 libiomp5md.dll 时需设置此变量（仅当前进程）
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 若报错 Install transformers (with Mamba support)，请先执行: conda activate mamba5090
# Windows 下直接用 Conda 训练见: docs/WINDOWS_CONDA_TRAIN.md（无需 WSL）
# 若要在 WSL 中安装 mamba-ssm 并训练，见: docs/WSL_TRAIN.md 与 scripts/wsl_install_mamba_ssm_and_train.sh
Write-Host "========== Stage 2: 续训（Continue）==========" -ForegroundColor Cyan
Write-Host "加载 outputs/vision_bridge_vlm_final.pt，max_visual_tokens=96" -ForegroundColor Gray
Write-Host "Loss 记录到 outputs/stage2_train_log.csv，可画曲线判断是否继续训" -ForegroundColor Gray
Write-Host "提速与 mamba_ssm 安装: 见 docs/SPEED_AND_MAMBA_SSM.md" -ForegroundColor DarkGray
Write-Host ""

# 本地 Mamba：设置 $env:MAMBA_MODEL = "D:/mamba/models/mamba-2.8b-hf" 后本脚本会自动传入
$mambaArg = @()
if ($env:MAMBA_MODEL) { $mambaArg = @("--mamba_model", $env:MAMBA_MODEL) }
# 显存吃紧：设置 $env:LOW_VRAM = "1" 则使用 --max_visual_tokens 64 --max_text_len 256；设置 $env:LLM_8BIT = "1" 则使用 --llm_8bit
$vramArg = @()
if ($env:LOW_VRAM -eq "1") { $vramArg = @("--max_visual_tokens", "64", "--max_text_len", "256"); Write-Host "LOW_VRAM: 使用 max_visual_tokens=64, max_text_len=256" -ForegroundColor Yellow }
$llm8Arg = @()
if ($env:LLM_8BIT -eq "1") { $llm8Arg = @("--llm_8bit"); Write-Host "LLM_8BIT: 启用 8-bit 加载以省显存" -ForegroundColor Yellow }
$epochNum = "30"
if ($env:STAGE2_EPOCHS) { $epochNum = $env:STAGE2_EPOCHS }
$epochArg = @("--epochs", [string]$epochNum)
$visArg = @("--max_visual_tokens", "96")
if ($env:LOW_VRAM -eq "1") { $visArg = @() }
$fromScratchArg = @()
if ($env:FROM_SCRATCH -eq "1") { $fromScratchArg = @("--from_scratch"); Write-Host "FROM_SCRATCH: 重新初始化 Bridge，不加载 Stage 1 权重" -ForegroundColor Yellow }
python train_vlm.py @epochArg --batch_size 1 --lr 2e-5 @visArg @mambaArg @vramArg @llm8Arg @fromScratchArg
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========== Done ==========" -ForegroundColor Green
Write-Host "Checkpoint: outputs/vision_bridge_vlm_final.pt" -ForegroundColor Gray
$visTest = "96"; if ($env:LOW_VRAM -eq "1") { $visTest = "64" }
$testCmd = "python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens $visTest --llm_device auto"
if ($env:LLM_8BIT -eq "1") { $testCmd = $testCmd + " --llm_8bit" }
Write-Host "Test: $testCmd" -ForegroundColor Gray
