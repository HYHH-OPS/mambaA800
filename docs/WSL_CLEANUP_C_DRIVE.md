# 释放 C 盘：查找并删除 WSL 相关占用

WSL 的虚拟磁盘和元数据通常都在 C 盘，可能占用几十 GB。下面用 **PowerShell 在 C 盘查找** WSL 相关位置，再按需**安全删除**以腾出空间。

## 一、查找 C 盘上 WSL 占用的位置与大小

在 **PowerShell（管理员非必须）** 中执行下面脚本，会列出 C 盘上常见 WSL 相关目录及大小（只读，不删除任何东西）：

```powershell
# 在 C 盘查找 WSL 相关目录并显示大小
$locations = @(
    "$env:LOCALAPPDATA\Packages\*Ubuntu*",
    "$env:LOCALAPPDATA\Packages\*Canonical*",
    "$env:LOCALAPPDATA\Packages\*WSL*",
    "$env:LOCALAPPDATA\Packages\*wsl*",
    "$env:USERPROFILE\AppData\Local\Temp\*wsl*",
    "$env:ProgramData\Microsoft\WSL"
)
foreach ($pattern in $locations) {
    Get-Item $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeGB = [math]::Round($size / 1GB, 2)
        Write-Host ("{0:N2} GB  {1}" -f $sizeGB, $_.FullName)
    }
}
```

也可以直接运行项目里的脚本（仅列出大小，不删除）：

```powershell
cd D:\mamba
.\scripts\find_wsl_on_c.ps1
```

## 二、安全释放空间的几种方式

### 方式 A：彻底卸载 WSL 发行版（会删除该发行版内所有数据）

若不再使用 WSL，可直接卸载发行版，对应虚拟磁盘（如 `ext4.vhdx`）占用的空间会被释放。

1. **用命令查看已安装的发行版：**
   ```powershell
   wsl -l -v
   ```

2. **卸载指定发行版（会永久删除该发行版及其中所有文件）：**
   ```powershell
   wsl --unregister <发行版名称>
   ```
   例如卸载 Ubuntu：
   ```powershell
   wsl --unregister Ubuntu
   ```

3. **或从系统设置卸载：**  
   **设置 → 应用 → 已安装的应用** → 找到 “Ubuntu” 或 “Windows Subsystem for Linux” 相关应用 → **卸载**。

卸载后，`%LOCALAPPDATA%\Packages\CanonicalGroupLimited.*\LocalState` 下的 `ext4.vhdx` 等会被系统清理，C 盘空间会明显增加。

### 方式 B：只关闭 WSL，再压缩虚拟磁盘（保留发行版）

若还想保留 WSL，可先关闭再压缩 VHD，能回收一部分空间：

1. 关闭 WSL：
   ```powershell
   wsl --shutdown
   ```

2. 在 **磁盘管理** 中压缩对应 VHD：
   - 虚拟磁盘路径通常类似：  
     `C:\Users\<你的用户名>\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu*\LocalState\ext4.vhdx`
   - **磁盘管理 → 操作 → 附加 VHD** 挂载该 vhdx，再 **操作 → 分离 VHD**（不要在里面格式化或删分区）。
   - 或使用 PowerShell 压缩（需 Hyper-V 或相关模块）：  
     [微软文档：压缩 VHD](https://docs.microsoft.com/en-us/powershell/module/hyper-v/optimize-vhd)

### 方式 C：将 WSL 发行版导出到 D 盘再卸载（可选）

若希望保留一份备份再清 C 盘：

```powershell
wsl --shutdown
wsl --export Ubuntu D:\backup-ubuntu.tar
wsl --unregister Ubuntu
```

之后若需要可再用 `wsl --import` 从 `D:\backup-ubuntu.tar` 导入到 D 盘。

## 三、注意

- **项目代码在 D:\mamba**：卸载或删除的是 WSL 系统盘（C 盘上的 vhdx），不会动 D 盘上的 `D:\mamba`。
- **卸载前**：确认 WSL 里没有只存在于 WSL 内的重要数据；若有，先复制到 Windows 盘（如 `D:\`）再卸载。
- 执行 `wsl --unregister` 前请再次确认发行版名称，避免误删。

按上述步骤即可在 C 盘上**查找**并**按需删除** WSL 相关占用，释放空间。
