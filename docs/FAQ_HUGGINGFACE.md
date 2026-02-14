# Hugging Face 连接失败 (WinError 10060) 解决办法

报错类似：`[WinError 10060] ... while requesting HEAD https://huggingface.co/state-spaces/mamba-2.8b-hf/...`  
说明无法连上 Hugging Face 下载 Mamba 模型，可按下面任选一种方式解决。

---

## 办法一：使用国内镜像（推荐）

在运行检测/推理**之前**，先设置环境变量，让程序从镜像站拉取模型：

**PowerShell（当前终端有效）：**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts/check_image_to_text.py
```

**长期生效**：在系统环境变量里新增 `HF_ENDPOINT` = `https://hf-mirror.com`，或写入 conda 激活脚本。

---

## 办法二：使用已下载的本地模型

在**网络正常或开 VPN** 时，先把模型下载到本地，之后检测/推理都用本地路径，不再访问外网。

### 1. 下载到本地

在能访问 Hugging Face 的环境下执行一次（或在本机开 VPN 后执行）：

```powershell
cd D:\mamba
conda activate mamba5090
python -c "
from huggingface_hub import snapshot_download
snapshot_download('state-spaces/mamba-2.8b-hf', local_dir='D:/mamba/models/mamba-2.8b-hf')
"
```

若已配置镜像，也可在设置 `HF_ENDPOINT=https://hf-mirror.com` 后执行上面命令，从镜像下载。

### 2. 指定本地路径运行检测

```powershell
python scripts/check_image_to_text.py --mamba_model D:/mamba/models/mamba-2.8b-hf
```

或推理：

```powershell
python inference.py --val_sample --mamba_model D:/mamba/models/mamba-2.8b-hf
```

这样运行时不会再请求 Hugging Face，可避免 WinError 10060。

---

## 办法三：使用已有缓存且离线

若之前成功下载过该模型，缓存一般在 `C:\Users\你的用户名\.cache\huggingface\hub`。可离线使用：

```powershell
$env:HF_HUB_OFFLINE = "1"
python scripts/check_image_to_text.py
```

若仍尝试访问网络，可同时设置镜像并离线：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_OFFLINE = "1"
python scripts/check_image_to_text.py
```

---

## 小结

| 情况           | 建议 |
|----------------|------|
| 国内网络       | 先设 `HF_ENDPOINT=https://hf-mirror.com` 再运行 |
| 能下载一次     | 用 `snapshot_download` 下到本地，后用 `--mamba_model 本地路径` |
| 已有缓存       | 设 `HF_HUB_OFFLINE=1` 离线运行 |

---

## CUDA 显存不足 (OutOfMemoryError)

若加载完 Mamba 后报错 `CUDA out of memory`，说明 32G 显存在「慢速 Mamba 路径」下不够。可用下面任一方式：

1. **减少视觉 token（默认已开）**：脚本已默认把 784 个视觉 token 池化到 196，若仍 OOM，可再减：
   ```powershell
   python scripts/check_image_to_text.py --max_visual_tokens 49
   ```

2. **把 Mamba 放到 CPU**（推荐，防 OOM）：显存只给 Vision+Bridge，Mamba 用 CPU（会慢一些）。脚本已默认 `--llm_device cpu`：
   ```powershell
   python scripts/check_image_to_text.py --llm_device cpu
   ```

3. **缩短生成长度**：`--max_new_tokens 50`
