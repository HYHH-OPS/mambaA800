# 训练/推理提速 与 mamba_ssm 安装使用

## 一、不装 mamba_ssm 时的提速

当前脚本已做：

- **BF16 混合精度**（CUDA 下默认）
- **DataLoader num_workers=4**（减少 GPU 等数据）
- **--max_visual_tokens 96**（序列越短 Mamba 越省时）

可进一步提速（略牺牲一点视觉细粒度）：

| 方法 | 命令/设置 | 说明 |
|------|-----------|------|
| 减少视觉 token | `--max_visual_tokens 64` | 训练和推理都要用同一值，否则对齐会偏 |
| 显存有余时加大 batch | `--batch_size 2` | 能跑再试，OOM 则改回 1 |
| Windows 下 DataLoader 报错 | `--num_workers 0` | 仅当多进程报错时改用 |

**续训时「更快」示例（64 视觉 token，推理时也要带 `--max_visual_tokens 64`）：**

```powershell
python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 64
```

---

## 二、安装 mamba_ssm 后能快多少

未安装时，Mamba 使用 **sequential 实现**（纯 PyTorch），功能正确但慢。  
安装 **mamba-ssm**（及可选 causal-conv1d）后，会用 CUDA 内核，**训练和推理都可明显加速**（数量级级别），无需改代码，安装即生效。

---

## 三、如何安装 mamba_ssm

### 3.1 Linux / WSL2（推荐）

在 **Ubuntu / WSL2** 下成功率最高：

```bash
# 1. 确保已装好 CUDA 版 PyTorch（与当前 CUDA 版本一致）
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 示例 cu121

# 2. 安装 mamba-ssm（会编译 CUDA 扩展，需要 nvcc）
pip install mamba-ssm

# 3. 可选：进一步加速
pip install causal-conv1d
```

若系统没有 `nvcc`，先装 CUDA Toolkit，并把 `bin` 加入 PATH，再执行上面 `pip install mamba-ssm`。  
**WSL 一键做法**：在项目目录执行 `bash scripts/wsl_install_nvcc_then_mamba_ssm.sh`（详见 `docs/WSL_TRAIN.md`）。

### 3.2 Windows 本机（需 CUDA 编译环境）

1. **安装 CUDA Toolkit**  
   版本要和当前 PyTorch 的 CUDA 一致（如 11.8 或 12.x），安装时勾选「加入 PATH」。

2. **确认 nvcc 可用**  
   以管理员打开 PowerShell：
   ```powershell
   nvcc -V
   ```
   若找不到命令，把 CUDA 的 `bin` 目录加入系统 PATH。

3. **在同一环境中安装（避免构建用错 CPU 版 PyTorch）**  
   ```powershell
   pip install --no-build-isolation mamba-ssm
   ```
   若报错，可再安装 Visual Studio「使用 C++ 的桌面开发」后重试。

Windows 上编译失败很常见，若折腾不动，**直接用 WSL2 装**更省事。

---

## 四、如何使用 mamba_ssm（无需改代码）

**安装好后不用改项目里任何代码。**

- 训练：照常运行 `train_vlm.py` 或 `run_stage2_continue.ps1`，若检测到 mamba_ssm，会自动走 CUDA 快路径。
- 推理：照常运行 `scripts/check_image_to_text.py`，同样会自动用 mamba_ssm。

终端里若不再出现 “The fast path is not available... Falling back to the sequential implementation”，即表示已在使用 mamba_ssm 加速。

---

## 五、小结

| 目标 | 做法 |
|------|------|
| 先能跑、少折腾 | 不装 mamba_ssm，用当前脚本即可；可加 `--max_visual_tokens 64` 略提速。 |
| 明显提速训练/推理 | 在 **WSL2/Linux** 中安装 `mamba-ssm`（及可选 `causal-conv1d`），安装即生效。 |
| 在 Windows 本机装 | 装好 CUDA Toolkit + nvcc，用 `pip install --no-build-isolation mamba-ssm`；失败则改用 WSL2。 |

更多安装报错（如 `bare_metal_version`、`nvcc not found`）见：`docs/FAQ_MAMBA_SSM_INSTALL.md`。
