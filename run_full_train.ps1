# 完整两阶段训练：Stage1 Vision+Bridge -> Stage2 VLM 微调（图像→报告）
# 用法: cd D:\mamba ; .\run_full_train.ps1
# 可选: .\run_full_train.ps1 -Stage1Epochs 5 -Stage2Epochs 3

param(
    [int] $Stage1Epochs = 3,
    [int] $Stage1BatchSize = 4,
    [double] $Stage1Lr = 1e-4,
    [int] $Stage2Epochs = 2,
    [int] $Stage2BatchSize = 1,
    [double] $Stage2Lr = 1e-5
)

$EnvName = "mamba5090"
$MambaRoot = if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot }
$OutDir = Join-Path $MambaRoot "outputs"
$Stage1Ckpt = Join-Path $OutDir "vision_bridge_best_val.pt"

Set-Location $MambaRoot
Write-Host "========== 完整训练：Stage1 + Stage2 ==========" -ForegroundColor Cyan
Write-Host "项目: $MambaRoot" -ForegroundColor Yellow
Write-Host ""

# ---------- Stage 1: Vision + Bridge ----------
Write-Host "---------- Stage 1: Vision + Bridge ----------" -ForegroundColor Green
Write-Host "epochs=$Stage1Epochs batch_size=$Stage1BatchSize lr=$Stage1Lr" -ForegroundColor Gray
conda run -n $EnvName --no-capture-output python train.py --epochs $Stage1Epochs --batch_size $Stage1BatchSize --lr $Stage1Lr
if ($LASTEXITCODE -ne 0) {
    Write-Host "Stage 1 失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
if (-not (Test-Path $Stage1Ckpt)) {
    Write-Host "未找到 Stage 1 权重: $Stage1Ckpt，请检查 outputs/ 目录" -ForegroundColor Red
    exit 1
}
Write-Host "Stage 1 完成，权重: $Stage1Ckpt" -ForegroundColor Green
Write-Host ""

# ---------- Stage 2: VLM 微调（图像→报告）----------
Write-Host "---------- Stage 2: VLM 微调（图像→报告）----------" -ForegroundColor Green
Write-Host "epochs=$Stage2Epochs batch_size=$Stage2BatchSize lr=$Stage2Lr checkpoint=$Stage1Ckpt" -ForegroundColor Gray
conda run -n $EnvName --no-capture-output python train_vlm.py --epochs $Stage2Epochs --batch_size $Stage2BatchSize --lr $Stage2Lr --vision_checkpoint $Stage1Ckpt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Stage 2 失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "Stage 2 完成，最终 VLM 权重: $OutDir\vision_bridge_vlm_final.pt" -ForegroundColor Green
Write-Host ""
Write-Host "========== 全部训练完成 ==========" -ForegroundColor Cyan
$VlmCkpt = Join-Path $OutDir "vision_bridge_vlm_final.pt"
$ResDir = "D:\mamba-res"
Write-Host "生成报告落盘目录: $ResDir" -ForegroundColor Gray
Write-Host "单张图推理: python inference.py --checkpoint $VlmCkpt --image `"D:\nnunet_raw\Dataset503_TBLesion_327\imagesTr\xxx.nii.gz`" --out_dir $ResDir" -ForegroundColor Gray
Write-Host "验证集抽样: python inference.py --checkpoint $VlmCkpt --val_sample --num_val 5 --out_dir $ResDir" -ForegroundColor Gray
