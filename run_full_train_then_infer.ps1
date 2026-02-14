# 完整流程：两阶段训练 + 验证集抽样生成，结果写入 D:\mamba-res
# 用法: cd D:\mamba ; .\run_full_train_then_infer.ps1

param(
    [int] $Stage1Epochs = 3,
    [int] $Stage1BatchSize = 4,
    [double] $Stage1Lr = 1e-4,
    [int] $Stage2Epochs = 2,
    [int] $Stage2BatchSize = 1,
    [double] $Stage2Lr = 1e-5,
    [int] $NumValSamples = 3
)

$EnvName = "mamba5090"
$MambaRoot = if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot }
$OutDir = Join-Path $MambaRoot "outputs"
$Stage1Ckpt = Join-Path $OutDir "vision_bridge_best_val.pt"
$VlmCkpt = Join-Path $OutDir "vision_bridge_vlm_final.pt"
$ResDir = "D:\mamba-res"

Set-Location $MambaRoot

# ---------- Stage 1 ----------
Write-Host "========== Stage 1: Vision + Bridge ==========" -ForegroundColor Cyan
conda run -n $EnvName --no-capture-output python train.py --epochs $Stage1Epochs --batch_size $Stage1BatchSize --lr $Stage1Lr
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path $Stage1Ckpt)) { Write-Host "未找到 $Stage1Ckpt" -ForegroundColor Red; exit 1 }

# ---------- Stage 2 ----------
Write-Host "========== Stage 2: VLM 微调 ==========" -ForegroundColor Cyan
conda run -n $EnvName --no-capture-output python train_vlm.py --epochs $Stage2Epochs --batch_size $Stage2BatchSize --lr $Stage2Lr --vision_checkpoint $Stage1Ckpt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---------- 推理：验证集抽样，写入 D:\mamba-res ----------
Write-Host "========== 推理：验证集 $NumValSamples 条 -> $ResDir ==========" -ForegroundColor Cyan
conda run -n $EnvName --no-capture-output python inference.py --checkpoint $VlmCkpt --val_sample --num_val $NumValSamples --out_dir $ResDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "========== 完成：训练权重在 $OutDir，生成报告在 $ResDir\run_* ==========" -ForegroundColor Green
