# 仅列出 C 盘上 WSL 相关目录及大小，不删除任何文件
# 用法: 在 PowerShell 中执行 .\scripts\find_wsl_on_c.ps1
$ErrorActionPreference = "Continue"
Write-Host "========== C 盘 WSL 相关目录及占用 ==========" -ForegroundColor Cyan
$locations = @(
    (Join-Path $env:LOCALAPPDATA "Packages"),
    (Join-Path $env:USERPROFILE "AppData\Local\Temp"),
    (Join-Path $env:ProgramData "Microsoft")
)
$totalBytes = 0
$packagesPath = Join-Path $env:LOCALAPPDATA "Packages"
if (Test-Path $packagesPath) {
    Get-ChildItem $packagesPath -Directory -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match "Ubuntu|Canonical|WSL|wsl|Linux"
    } | ForEach-Object {
        $dir = $_.FullName
        $size = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $totalBytes += $size
        $sizeGB = [math]::Round($size / 1GB, 2)
        Write-Host ("  {0:N2} GB  {1}" -f $sizeGB, $dir)
    }
}
$wslData = Join-Path $env:ProgramData "Microsoft\WSL"
if (Test-Path $wslData) {
    $size = (Get-ChildItem $wslData -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $totalBytes += $size
    $sizeGB = [math]::Round($size / 1GB, 2)
    Write-Host ("  {0:N2} GB  {1}" -f $sizeGB, $wslData)
}
Write-Host ""
$totalGB = [math]::Round($totalBytes / 1GB, 2)
Write-Host ("上述 WSL 相关合计约: {0:N2} GB" -f $totalGB) -ForegroundColor Yellow
Write-Host '彻底释放: 见 docs/WSL_CLEANUP_C_DRIVE.md (如 wsl --unregister 发行版名)' -ForegroundColor Gray
