# 使用 Hugging Face 镜像运行检测，避免 WinError 10060
# 用法: .\run_check_with_mirror.ps1
#       .\run_check_with_mirror.ps1 -Image "D:\nnunet_raw\...\xxx.nii.gz"

param([string] $Image = "")

$env:HF_ENDPOINT = "https://hf-mirror.com"
Set-Location (if (Test-Path "D:\mamba") { "D:\mamba" } else { $PSScriptRoot })
if ($Image) {
    python scripts/check_image_to_text.py --image $Image --llm_device auto --max_new_tokens 512
} else {
    python scripts/check_image_to_text.py --llm_device auto --max_new_tokens 512
}
