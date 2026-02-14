# 在 WSL 中安装 mamba-ssm 并训练

在 WSL（Ubuntu）下可以正常安装 mamba-ssm 和 causal-conv1d，训练与推理会使用 CUDA 快路径，速度明显快于 Windows 下的 sequential 实现。

## 一、前提

- 已安装 **WSL2** 和 **Ubuntu**（或其它 Linux 发行版）。
- 已安装 **NVIDIA 驱动**（Windows 侧），WSL 内可用 `nvidia-smi`。
- 项目在 Windows 盘上，WSL 中通过 `/mnt/d/mamba` 访问（对应 `D:\mamba`）；或已将项目复制到 WSL 家目录如 `~/mamba`。

## 二、路径说明（重要）

- **config/paths.yaml** 里使用 Windows 路径（如 `d:/mamba/...`）时，在 WSL 下运行会自动转换为 `/mnt/d/mamba/...`，无需改配置。
- **CSV 里的 image_path** 若为 Windows 路径（如 `C:\Users\...`），在 WSL 下也会自动转为 `/mnt/c/Users/...`。
- 若数据只在 Windows 盘上，请确保在 WSL 里能访问：`ls /mnt/d/mamba/outputs/excel_caption/caption_train.csv` 能列出文件。

## 三、一键安装 + 训练

在 WSL 终端中：

```bash
# 进入项目目录（二选一）
cd /mnt/d/mamba
# 或  cd ~/mamba

# 安装 mamba-ssm 并开始训练
bash scripts/wsl_install_mamba_ssm_and_train.sh
```

- **只安装不训练**：`bash scripts/wsl_install_mamba_ssm_and_train.sh --install-only`
- **已装过 mamba-ssm，只训练**：`bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only`

## 四、手动安装 mamba-ssm（可选）

若希望先单独安装再自己调参训练：

```bash
# 建议在虚拟环境中
pip install torch  # 需为 CUDA 版
pip install mamba-ssm causal-conv1d
```

然后按需运行：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python3 train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96
```

## 五、mamba-ssm 安装失败（nvcc 未找到）时 — 提速必做

在 WSL 中若出现 `nvcc was not found` 或 `bare_metal_version is not defined`，说明当前环境没有 CUDA 编译器，**训练会使用 sequential 实现，速度很慢**。

### 方案 A：一键安装 nvcc + mamba-ssm（推荐，提速明显）

在 WSL 项目目录下执行（会提示输入 sudo 密码）：

```bash
cd /mnt/d/mamba
sed -i 's/\r$//' scripts/wsl_install_nvcc_then_mamba_ssm.sh   # 若从 Windows 复制过脚本，先去掉 \r
bash scripts/wsl_install_nvcc_then_mamba_ssm.sh
```

脚本会：安装 NVIDIA CUDA Toolkit（提供 nvcc）→ 再 `pip install mamba-ssm causal-conv1d`。完成后**新开终端或执行** `export PATH=/usr/local/cuda/bin:$PATH`，再运行训练即可使用 CUDA 快路径。

### 方案 B：不装 nvcc，直接训练（慢）

直接运行 `bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only`，会用 sequential 实现，速度较慢但结果一致。

### 方案 C：手动安装 nvcc 再装 mamba-ssm

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
export PATH=/usr/local/cuda/bin:$PATH
nvcc -V   # 确认后
source .venv_wsl/bin/activate
pip install mamba-ssm causal-conv1d
```

## 六、从 Windows 进入 WSL 并跑脚本

在 **PowerShell** 或 **CMD** 中：

```powershell
wsl
cd /mnt/d/mamba
bash scripts/wsl_install_mamba_ssm_and_train.sh
```

训练结束后，权重在 `outputs/vision_bridge_vlm_final.pt`，在 Windows 下也可直接用该权重做推理（与在 WSL 中训练一致）。

---

## 七、mamba-ssm 未安装时

若 WSL 中未成功安装 mamba-ssm（缺 nvcc），**不影响训练**：直接执行 `bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only` 即可，程序会自动使用 sequential 实现，仅速度较慢。

---

## 八、RTX 5090 / Blackwell (sm_120) 在 WSL 中

若出现提示：**“NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation”**，说明当前 WSL 里装的是 cu121 等旧版 PyTorch，**不支持 RTX 5090**，训练可能无法正确使用 GPU。

**解决：在 WSL 虚拟环境中安装支持 sm_120 的 PyTorch Nightly（CUDA 12.8）**：

```bash
cd /mnt/d/mamba
source .venv_wsl/bin/activate

# 卸载当前 PyTorch 后安装 Nightly cu128（支持 Blackwell/sm_120）
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**验证**：应看到 `sm_120` 且 GPU 可用：

```bash
python3 -c "import torch; print('arch_list:', torch.cuda.get_arch_list()); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

然后再执行 `bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only` 进行训练。更多 PyTorch/5090 说明见 `docs/STAGE2_AND_PYTORCH.md`。

---

## 九、Hugging Face 网络不可达（加载 Mamba 报错）

若出现 **`[Errno 101] Network is unreachable`** 或 **`thrown while requesting HEAD https://huggingface.co/state-spaces/mamba-2.8b-hf/...`**，说明 WSL 里无法访问 Hugging Face，无法在线拉取 Mamba 权重。

### 方案 A：使用本地已下载的 Mamba 模型（推荐）

1. **在能访问 Hugging Face 的环境里先下载一次**（例如在 Windows PowerShell 下，或换网络后的 WSL）：
   - 在项目目录创建目录：`mkdir -p models`（WSL 下为 `mkdir -p /mnt/d/mamba/models`）。
   - 在**能联网**的环境执行一次（仅下载，不训练）：
   ```bash
   # 在能访问 Hugging Face 的终端（如 Windows 或另一台机器）
   cd /mnt/d/mamba   # 或 D:\mamba
   source .venv_wsl/bin/activate   # 或 conda activate mamba5090
   python3 -c "
   from transformers import AutoTokenizer, MambaForCausalLM
   path = 'models/mamba-2.8b-hf'
   tok = AutoTokenizer.from_pretrained('state-spaces/mamba-2.8b-hf')
   model = MambaForCausalLM.from_pretrained('state-spaces/mamba-2.8b-hf')
   tok.save_pretrained(path)
   model.save_pretrained(path)
   print('已保存到', path)
   "
   ```
   若在 Windows 下用 conda，可把 `path` 设为 `D:/mamba/models/mamba-2.8b-hf`，保存后 WSL 用 `/mnt/d/mamba/models/mamba-2.8b-hf`。

2. **在 WSL 里用本地路径启动训练**（不再访问外网）：
   ```bash
   cd /mnt/d/mamba
   source .venv_wsl/bin/activate
   export KMP_DUPLICATE_LIB_OK=TRUE
   # 使用本地 Mamba 路径（按你实际保存位置改）
   python3 train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96 --max_text_len 512 --mamba_model /mnt/d/mamba/models/mamba-2.8b-hf
   ```
   或通过环境变量让脚本自动带 `--mamba_model`：
   ```bash
   export MAMBA_MODEL=/mnt/d/mamba/models/mamba-2.8b-hf
   bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only
   ```

### 方案 B：使用 HF 缓存 + 离线模式

若你**曾经**在 WSL 里成功拉取过该模型，缓存一般在 `~/.cache/huggingface/hub/`。可先断网或不再访问 HF，然后强制离线加载：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only
```

若从未在 WSL 下载过该模型，缓存里没有，此方式会报错，请用方案 A。

---

## 十、为 RTX 5090 (sm_120) 启用 Fast Path：从源码编译 mamba-ssm

若你看到 **"The fast path is not available ... Falling back to the sequential implementation"** 或 **"no kernel image is available for execution on the device"**，说明当前安装的 mamba-ssm / causal-conv1d **没有**为 RTX 5090 (sm_120) 编译可用的 CUDA 内核，训练会报错或退回到串行实现。要启用加速（Fast Path），需要在 WSL 里**从源码重新编译**这两个包。

### 若训练卡死、显存占满、GPU 占用 100% 但温度很低（假死）

CUDA 报错后进程可能卡住不退出，显存一直被占。先**强制结束进程、释放显存**，再重新编译/训练：

```bash
# 在 WSL 终端执行，结束所有 Python 进程
pkill -9 python
```

执行后任务管理器里显存应回到接近 0，再继续下面的编译或训练。

### 前提

- **CUDA Toolkit 12.8**（或支持 Blackwell/sm_120 的版本）。WSL 里若之前装的是 12.6，需升级到 12.8 才能让 nvcc 为 sm_120 生成代码。
- 已安装 PyTorch nightly cu128（支持 sm_120），见第八节。

### 步骤 1：安装 CUDA 12.8 并确认 nvcc（若尚未安装）

在 WSL 中：

```bash
# 若已有 /usr/local/cuda，先看版本： ls /usr/local/cuda/version.txt 或 nvcc -V
# 安装 CUDA 12.8（WSL-Ubuntu，按官方文档选择对应 repo）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
export PATH=/usr/local/cuda/bin:$PATH
nvcc -V   # 应显示 12.8.x
```

若发行版暂无 cuda-toolkit-12-8，可到 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 选 WSL-Ubuntu 与 12.8 下载 runfile 或对应包。

### 步骤 2：卸载旧包并从源码编译（指定 sm_120）

在项目目录、已激活 `.venv_wsl` 的前提下：

```bash
cd /mnt/d/mamba
source .venv_wsl/bin/activate

# 若脚本在 Windows 下编辑过，先去掉 CRLF，避免 $'\r': command not found
sed -i 's/\r$//' scripts/wsl_build_mamba_ssm_sm120.sh

# 卸载可能存在的预编译包
pip uninstall mamba-ssm causal-conv1d -y

# 为 RTX 5090 (sm_120) 指定架构，使用当前环境的 PyTorch 和 nvcc 编译
export TORCH_CUDA_ARCH_LIST="12.0"
export PATH=/usr/local/cuda/bin:$PATH

# 先装 causal-conv1d，再装 mamba-ssm（mamba-ssm 依赖 causal-conv1d）
pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation -v
pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation -v
```

若希望同时保留对旧显卡的兼容，可设 `TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"`（编译时间会更长）。

### 步骤 2 备选：nvcc 不支持 12.0 时用 PTX（通用指令集）

若当前只有 CUDA 12.6 等、nvcc 不支持 `-arch=sm_120`，可改用 **PTX**：为 9.0 生成 PTX，让驱动在 RTX 5090 上 JIT 编译运行。在 WSL、已激活 `.venv_wsl` 下：

```bash
cd /mnt/d/mamba
source .venv_wsl/bin/activate
pip uninstall causal-conv1d mamba-ssm -y

# 使用 8.0;8.6;9.0+PTX，使新显卡可通过 PTX 运行
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"
export PATH=/usr/local/cuda/bin:$PATH

pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation -v
pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation -v
```

或直接运行项目中的 PTX 编译脚本（会先卸载再按上述环境变量编译）。**务必加 `--no-cache-dir`**，否则 pip 可能用缓存的旧 wheel，不会从源码编 mamba-ssm：

```bash
sed -i 's/\r$//' scripts/wsl_build_mamba_ssm_ptx.sh
bash scripts/wsl_build_mamba_ssm_ptx.sh
```

若已出现过 `Using cached mamba_ssm-2.3.0-...whl`，说明之前没真正从源码编，需清缓存后重编。

**终极修复（一条龙，同一终端内执行）**：下面整段复制到 WSL 终端一次执行，保证 `TORCH_CUDA_ARCH_LIST` 与安装在同一进程内生效。**causal-conv1d 必须看到大量编译滚屏（约 1–3 分钟）**；若几秒就 “Successfully installed” 且无 `-gencode arch=compute_...` 之类输出，说明用了缓存、未真正编译。

```bash
cd /mnt/d/mamba && source .venv_wsl/bin/activate
pip uninstall causal-conv1d mamba-ssm -y
pip cache purge
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"
export PATH=/usr/local/cuda/bin:$PATH
pip install causal-conv1d --no-binary causal-conv1d --force-reinstall --no-build-isolation --no-cache-dir -v
pip install mamba-ssm --no-binary mamba-ssm --force-reinstall --no-build-isolation --no-cache-dir -v
```

**核弹级重装（逐条执行便于观察）**：若怀疑 causal-conv1d 仍被缓存，可逐行执行并观察 causal-conv1d 那一步是否有长时间编译输出、是否出现 `-gencode arch=compute_90` 等字样：

```bash
pip uninstall causal-conv1d mamba-ssm -y
pip cache purge
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"
export PATH=/usr/local/cuda/bin:$PATH
pip install causal-conv1d --no-binary causal-conv1d --force-reinstall --no-build-isolation --no-cache-dir -v
# 上一步应滚屏 1–3 分钟，出现 gencode 等编译信息
pip install mamba-ssm --no-binary mamba-ssm --force-reinstall --no-build-isolation --no-cache-dir -v
```

完成后运行训练并开启 Fast Path：  
`export MAMBA_FORCE_CUDA=1 && bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only`

成功后**开启 Fast Path**：设置环境变量 `MAMBA_FORCE_CUDA=1` 再运行训练，程序将不再强制 reference 实现，而是使用 PTX 编译出的 CUDA 内核。若仍报 `no kernel image`，则不要设置该变量（保持默认回退补丁）。

```bash
export MAMBA_FORCE_CUDA=1
bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only
```

### 步骤 3：验证

重新运行训练，**不应**再出现 "Falling back to the sequential implementation"；显存与速度会更合理：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export MAMBA_MODEL=/mnt/d/mamba/models/mamba-2.8b-hf
bash scripts/wsl_install_mamba_ssm_and_train.sh --train-only
```

### 步骤 4：启用 Fast Path 后去掉 reference 回退（可选）

当前 `train_vlm.py` 在检测到 sm_120 时会强制 bridge 使用 reference 实现，以避免「no kernel image」报错。**从源码为 sm_120 编译成功后**，若希望 bridge 也走 CUDA 快路径，可删除或注释掉 `train_vlm.py` 中约第 22–38 行的 sm_120 补丁块（以 `# RTX 5090 (sm_120)` 开头、`except Exception: pass` 结尾的那段）。删除后若不再报错，即表示 Fast Path 已全程生效。
