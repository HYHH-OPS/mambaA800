# mamba-ssm 安装失败（Windows / bare_metal_version / nvcc）

## 你刚遇到的错误（直接对应）

若终端里出现：

- **`nvcc was not found`** → 当前环境没有 NVIDIA CUDA 编译器（或未加入 PATH），mamba-ssm 无法编译 CUDA 扩展。
- **`bare_metal_version is not defined`** → 因未检测到 nvcc，setup.py 走了“无 CUDA”分支，该分支里变量未定义导致报错。
- **`torch.__version__ = 2.10.0+cpu`** → 这是 **pip 构建隔离**里看到的；隔离环境里装的是 CPU 版 PyTorch，看不到你本机已装的 CUDA 版。

**结论**：在 Windows 上直接用 `pip install mamba-ssm` 很容易失败，**不装也可以正常训练和推理**（会慢一些，用 sequential 实现）。若一定要在 Windows 上装，见下方「方案二」；否则推荐用 WSL2 或直接不装。

---

## 现象（完整列表）

执行 `pip install mamba-ssm` 时可能出现：

- `NameError: name 'bare_metal_version' is not defined`
- `mamba_ssm was requested, but nvcc was not found`
- `torch.__version__ = 2.10.0+cpu`（构建隔离环境下看到的是 CPU 版 PyTorch）

## 原因

1. **mamba-ssm 需要从源码编译 CUDA 扩展**，必须能找到 **nvcc**（NVIDIA CUDA 编译器）。
2. 官方主要支持 **Linux**；在 **Windows** 上需要正确安装 CUDA Toolkit 并把 nvcc 加入 PATH。
3. 当检测不到 CUDA/nvcc 时，其 `setup.py` 中 `bare_metal_version` 未定义，会触发 `NameError`。

## 可选方案

### 方案一：直接放弃安装 mamba-ssm（推荐）

本项目在**未安装** mamba-ssm 时会自动使用 **sequential 实现**（纯 PyTorch），功能完整，只是推理更慢。

- 无需安装 mamba-ssm 即可完成训练与生成。
- 若只是验证流程或数据量不大，可直接忽略该警告。

### 方案二：在 Windows 上尝试安装 mamba-ssm（需要 CUDA 编译环境）

**前提**：必须先有 `nvcc`（NVIDIA CUDA 编译器）。没有则不要装，直接用方案一（不装 mamba-ssm）。

1. **安装 CUDA Toolkit**  
   从 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 下载与当前 PyTorch CUDA 版本一致的版本（如 11.8 或 12.x），安装时勾选「加入 PATH」。

2. **确认 nvcc 可用**  
   新开 PowerShell（建议以管理员运行），执行：
   ```powershell
   nvcc -V
   ```
   若提示“找不到命令”，把 CUDA 安装目录下的 `bin`（如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`）加入系统环境变量 PATH。

3. **确认当前环境是 CUDA 版 PyTorch**  
   ```powershell
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
   应看到带 `+cu...` 的版本和 `True`。若是 `+cpu`，请先在本环境中安装 CUDA 版 PyTorch。

4. **用当前环境构建，不用隔离环境**  
   ```powershell
   pip install --no-build-isolation mamba-ssm
   ```
   `--no-build-isolation` 会让构建使用你当前的 PyTorch 和 nvcc，避免隔离里出现 CPU 版 torch 和没有 nvcc 的问题。

若仍报错，可再安装 **Visual Studio** 的「使用 C++ 的桌面开发」工作负载后重试。若多次失败，建议改用方案一（不装）或方案三（WSL2）。

### 方案三：在 WSL2（Linux）中安装

在 WSL2 的 Ubuntu 等环境中安装 CUDA Toolkit 和 nvcc，再用 pip 安装 mamba-ssm，成功率更高。参见 [state-spaces/mamba](https://github.com/state-spaces/mamba#installation) 的 Linux 安装说明。

### 方案四：RTX 5090 (sm_120) 启用 Fast Path

若显卡为 **RTX 5090**（Blackwell，sm_120），预编译的 mamba-ssm 通常**未**包含对应 CUDA 内核，会出现 "Falling back to the sequential implementation"、速度很慢、显存占用偏大。需在 WSL 中**用 CUDA 12.8 + 指定架构从源码编译**。详见 **docs/WSL_TRAIN.md 第十节**，或直接运行：

```bash
cd /mnt/d/mamba && source .venv_wsl/bin/activate && bash scripts/wsl_build_mamba_ssm_sm120.sh
```

前提：已安装 CUDA Toolkit 12.8、nvcc 在 PATH 中，且 PyTorch 为 nightly cu128（支持 sm_120）。

## 数值一致性（长序列生成）

在极长序列生成时，sequential 实现与 CUDA kernel 可能存在细微数值差异，可能影响生成稳定性。若长文出现异常，可优先尝试安装 mamba_ssm；参见 `docs/INFERENCE_AND_HALLUCINATION.md`。

## 预编译 wheel 下载失败（Remote end closed connection）

在 WSL 中执行 `pip install mamba-ssm` 时，若出现 **"Guessing wheel URL: ... error: Remote end closed connection without response"**，说明从 GitHub 拉取预编译包时连接被中断（网络或墙）。

**处理步骤：**

1. **先重试一次**（有时是瞬时网络问题）  
   ```bash
   pip install mamba-ssm causal-conv1d
   ```

2. **若仍失败，改为从源码编译**（需已安装 nvcc 并加入 PATH）  
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   nvcc -V   # 确认有输出
   pip install mamba-ssm --no-binary mamba-ssm
   # causal-conv1d 若已装过可跳过
   ```

3. **若未装 nvcc 或编译报错**  
   可先不装 mamba-ssm，直接训练（用 sequential，较慢）。等网络稳定或装好 nvcc 后再试上述步骤。

---

## 小结

| 目标           | 建议 |
|----------------|------|
| 先跑通训练/推理 | 不装 mamba-ssm，使用 sequential 即可。 |
| 需要加速推理   | 在 Linux/WSL2 中安装 mamba-ssm；或在 Windows 上装好 CUDA + nvcc 后使用 `pip install --no-build-isolation mamba-ssm`。 |
| 长文生成异常   | 可尝试安装 mamba_ssm 以与训练/推理数值一致；并配合贪心解码与 Stage 2 充分训练。 |
