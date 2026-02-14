# RTX 5090：在 mamba5090 环境下启动医学 VLM 训练
# 用法: 先执行 .\setup_mamba_5090.ps1 完成环境安装，再执行本脚本
#   cd D:\mamba  ;  .\run_train_5090.ps1
# 可选参数: .\run_train_5090.ps1 -Epochs 5 -BatchSize 2

param(
    [int] $Epochs = 3,
    [int] $BatchSize = 4,
    [double] $Lr = 1e-4
)

$EnvName = "mamba5090"
$MambaRoot = if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot }

Set-Location $MambaRoot
Write-Host "========== 医学 VLM 训练 (RTX 5090) ==========" -ForegroundColor Cyan
Write-Host "环境: $EnvName, 项目: $MambaRoot" -ForegroundColor Yellow
Write-Host "epochs=$Epochs batch_size=$BatchSize lr=$Lr" -ForegroundColor Yellow
Write-Host ""

conda run -n $EnvName --no-capture-output python train.py --epochs $Epochs --batch_size $BatchSize --lr $Lr
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "训练结束，检查 outputs/ 目录" -ForegroundColor Green
