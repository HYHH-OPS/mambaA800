# Stage 2 训练与 PyTorch 环境（解决俄文/乱码）

## 问题简述

- **俄文/乱码**：模型未学好「见图写中文报告」，会瞎编预训练里的英文/俄文片段。
- **mamba-ssm 安装失败**：可放弃安装，只用 sequential 实现；不影响正确性，只影响速度。
- **PyTorch 为 CPU 版**：必须换成 CUDA 版才能在 RTX 5090 上正常训练与推理。

---

## 第一步：修复 PyTorch（必须）

在 **mamba5090** 环境下执行：

```powershell
# 卸载可能存在的 CPU 版本
pip uninstall torch torchvision -y

# 安装 CUDA 12.8 (Nightly)，支持 RTX 5090（约 2.8GB，下载较久）
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**验证**：安装完成后在 Python 里检查：

```python
import torch
print(torch.cuda.is_available())       # 必须是 True
print(torch.version.cuda)              # 应为 12.8 或类似
print(torch.cuda.get_device_name(0))   # 应为 NVIDIA GeForce RTX 5090
```

或直接运行项目内脚本：

```powershell
python scripts/verify_torch_cuda.py
```

若 `cuda.is_available()` 为 False 或版本不对，训练/推理都无法正确使用 GPU。

---

## 第二步：跳过 mamba-ssm，直接做 Stage 2 训练

不要继续折腾 `pip install mamba-ssm`。未安装时程序会自动使用 sequential 实现，**不会导致俄文或乱码**。

在 **D:\mamba** 下运行 Stage 2（图文对齐）训练：

```powershell
# 推荐：从头训，不加载旧 Stage1 权重（旧权重若由错误代理损失训出会导致“全黑”输出）
python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96 --from_scratch
```

- **--from_scratch**：不加载 `vision_bridge_best_val.pt` / `vision_bridge_final.pt`，随机初始化。若此前 Stage1 用错误损失训过，务必加此参数或先删除/重命名 outputs 下旧 pt 文件。

- **若 OOM**：改为 `--batch_size 1`，或加 `--max_visual_tokens 96` 再试 `--batch_size 2/4`。batch_size 8 在 32GB 下易 OOM。
- **提速（无 mamba_ssm 时）**：脚本已默认开启 **BF16 混合精度**、**DataLoader num_workers=4**、**--max_visual_tokens 96**；可进一步用 `--max_visual_tokens 64` 换速度。
- 输出权重默认在 `outputs/vision_bridge_vlm_final.pt`。

也可用一键脚本（参数已按上述写好）：

```powershell
.\run_stage2_train.ps1
```

---

## 提速方案（无 mamba_ssm 时）

| 方案 | 说明 | 已用/用法 |
|------|------|-----------|
| **BF16 混合精度** | RTX 5090 对 bfloat16 硬件加速，显存减半、速度提升约 30%–50% | 已默认开启（CUDA 时） |
| **减少视觉 Token** | 序列越短 Mamba 越省时；64–96 对报告通常够用 | `--max_visual_tokens 96`（默认脚本）或 `64` |
| **DataLoader 预取** | 多线程读图，减少 GPU 等待 | `--num_workers 4`（Windows 报错可 `--num_workers 0`） |

终极加速：在 WSL2 (Ubuntu) 中安装 `mamba-ssm`，训练可再快一个数量级。

---

## 第三步：监控 caption_loss

训练时请关注终端里的 **caption_loss**：

| 阶段     | 典型范围   | 说明 |
|----------|------------|------|
| 刚开始   | 6.0 ~ 8.0  | 正常，模型还未对齐 |
| 几轮后   | 3.0 ~ 4.0  | 开始学会图文对应 |
| **目标** | **< 2.0**  | 生成报告才会像人话 |

只有 loss 降到 **2.0 以下** 后，用该权重做生成，中文报告质量才会明显改善。若 loss 长期居高（例如一直 >4），说明训练未生效，需检查数据路径与 CSV 配置。

---

## 小结

1. **PyTorch**：必须为 CUDA 版并在验证脚本中通过，否则 5090 无法用于训练。
2. **mamba-ssm**：可不安装，用 sequential 即可跑通训练与推理。
3. **Stage 2**：用 `train_vlm.py --epochs 20 --batch_size 2 --lr 2e-5`（或 `run_stage2_train.ps1`），并盯住 **caption_loss 降到 2.0 以下**。
