# RTX 5090：创建医学 VLM (nnU-Net + Vim + Mamba) 的 Anaconda 环境
# 用法: 在 PowerShell 中执行 .\setup_mamba_5090.ps1
# 需要: 已安装 Anaconda/Miniconda，NVIDIA 驱动 566.03+（5090）

$EnvName = "mamba5090"
$MambaRoot = if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot }

Write-Host "========== RTX 5090 医学 VLM 环境安装 ==========" -ForegroundColor Cyan
Write-Host "环境名: $EnvName" -ForegroundColor Yellow
Write-Host "项目根: $MambaRoot" -ForegroundColor Yellow
Write-Host ""

# 1. 创建 conda 环境（Python 3.10）
if (-not (conda env list | Select-String -Pattern "^\s*$EnvName\s")) {
    Write-Host "[1/4] 创建 conda 环境 $EnvName (Python 3.10)..." -ForegroundColor Yellow
    conda create -n $EnvName python=3.10 -y
} else {
    Write-Host "[1/4] 环境 $EnvName 已存在，跳过创建" -ForegroundColor Gray
}

# 2. 安装 PyTorch nightly (CUDA 12.8) 支持 RTX 5090 (sm_120)
# 直接在本 shell 激活环境后 pip，避免 conda run 缓冲导致“卡住”无输出
Write-Host "[2/4] 安装 PyTorch nightly cu128 (RTX 5090)，约 2–5 分钟请稍候..." -ForegroundColor Yellow
& conda run -n $EnvName --no-capture-output pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --progress-bar on
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [TIP] 若长时间无输出，请 Ctrl+C 后手动执行:" -ForegroundColor Yellow
    Write-Host "  conda activate $EnvName" -ForegroundColor Gray
    Write-Host "  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128" -ForegroundColor Gray
    exit 1
}

# 3. 安装本项目依赖（不含 torch）
Write-Host "[3/4] 安装医学 VLM 依赖..." -ForegroundColor Yellow
$Req = Join-Path $MambaRoot "requirements_5090.txt"
if (Test-Path $Req) {
    conda run -n $EnvName pip install -r $Req
} else {
    conda run -n $EnvName pip install numpy PyYAML nibabel SimpleITK pandas matplotlib tqdm transformers accelerate sentencepiece tokenizers
}

# 4. 可选：Mamba SSM（5090 上推荐，加速 Vim 块）
Write-Host "[4/4] 可选安装 mamba-ssm（若失败可跳过）..." -ForegroundColor Yellow
conda run -n $EnvName pip install mamba-ssm causal-conv1d 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [INFO] mamba-ssm 安装失败将使用 Linear 占位，不影响训练" -ForegroundColor Gray
}

# 验证
Write-Host ""
Write-Host "验证 PyTorch 与 CUDA..." -ForegroundColor Yellow
conda run -n $EnvName python -c @"
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    arch = torch.cuda.get_arch_list()
    print('sm_120:', any('sm_120' in str(a) for a in arch))
    print('device:', torch.cuda.get_device_name(0))
"@

Write-Host ""
Write-Host "========== 安装完成 ==========" -ForegroundColor Green
Write-Host "激活环境: conda activate $EnvName" -ForegroundColor Green
Write-Host "训练命令: cd $MambaRoot ; python train.py" -ForegroundColor Green
Write-Host ""
