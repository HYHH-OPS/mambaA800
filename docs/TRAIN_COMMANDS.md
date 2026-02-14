# 训练代码与运行命令

## 1. 环境与配置

- **环境**: `mamba5090`（Python 3.10 + PyTorch + CUDA）
- **数据与路径**: 由 `config/paths.yaml` 指定（训练/验证 CSV、nnU-Net 编码器权重等）

```yaml
# config/paths.yaml 关键项
caption_csv_train: "d:/unn-net/train_radfm_315.csv"
caption_csv_val: "d:/unn-net/val_radfm_315.csv"
caption_prompt_json: "d:/unn-net/radfm_caption_prompt.json"
nnunet_encoder_checkpoint: "d:/nnunet_results/.../checkpoint_best.pth"
```

---

## 2. 训练脚本一览

| 脚本 | 作用 | 输出 |
|------|------|------|
| `train.py` | 训练 Vision + Vim Bridge（代理 L2 损失，验证前向/梯度） | `outputs/vision_bridge_best_val.pt`, `vision_bridge_final.pt` |
| `train_vlm.py` | 用「图像→报告」caption 损失微调 Vision+Bridge，Mamba 冻结 | 更好的图像→文本对齐 |
| `scripts/run_overnight_300.py` | 多组超参各跑 300 轮，按 val_loss 选最优并保存 | 多组 checkpoint + 日志 |

---

## 3. 推荐训练命令（RTX 5090）

### 3.1 快速跑通（Vision+Bridge）

```powershell
cd D:\mamba
conda activate mamba5090
python train.py --epochs 3 --batch_size 4 --lr 1e-4
```

或使用封装脚本（可改轮数与 batch）：

```powershell
cd D:\mamba
.\run_train_5090.ps1 -Epochs 5 -BatchSize 2 -Lr 1e-4
```

### 3.2 图像→报告微调（VLM caption）

需先有 Vision+Bridge 权重（如 `outputs/vision_bridge_best_val.pt`），再运行：

```powershell
cd D:\mamba
conda activate mamba5090
python train_vlm.py --epochs 2 --batch_size 1 --lr 1e-5 --vision_checkpoint outputs/vision_bridge_best_val.pt
```

### 3.2b 继续 Stage 2（从已有 VLM 权重续训）

若已跑过 Stage 2 得到 `vision_bridge_vlm_final.pt`，但生成仍乱码（如「外(IM12)」），多为 **caption_loss 未训够**。可**不传** `--from_scratch`，直接续训（不传 `--vision_checkpoint` 时会自动用 `outputs/vision_bridge_vlm_final.pt`）：

```powershell
cd D:\mamba
conda activate mamba5090
# 与首次 Stage 2 保持一致：max_visual_tokens=96（若你之前用 run_stage2_train.ps1）
python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96
```

- **如何看 loss**：训练时每 20 step 打印一次 `caption_loss`；每 epoch 结束会打「平均 caption_loss」；训练结束后会打「最近 5 个 epoch 平均 caption_loss」。
- **loss 曲线**：`outputs/stage2_train_log.csv`（列：step, epoch, caption_loss），可用 Excel 或 Python 画图，判断是否还需继续训或调小 lr。
- **与推理一致**：训练会写入 `outputs/stage2_config.json`（含 `max_visual_tokens`、`mamba_model`）。推理时请用相同 `--max_visual_tokens` 和相同 `--mamba_model`（默认均为 `state-spaces/mamba-2.8b-hf`）。

### 3.3 夜间 300 轮 + 超参筛选

```powershell
cd D:\mamba
conda activate mamba5090
python scripts/run_overnight_300.py
```

可选：

- `--epochs 100`：每组只跑 100 轮（快速试）
- `--no_grid`：不搜参，只跑一组默认参数 300 轮

---

## 4. 训练代码入口

- **主训练**: `D:\mamba\train.py`（见下方关键片段）
- **VLM 微调**: `D:\mamba\train_vlm.py`
- **超参网格**: `D:\mamba\scripts\run_overnight_300.py`

`train.py` 核心流程：读取 `paths.yaml` → 构建 `MedicalVLMDataset` 与 `build_medical_vlm_from_config` → 代理损失 `proxy_loss(visual_tokens)` → 每 10 step 打 train_loss，每 epoch 打 val_loss 并保存 best_val。

---

## 5. 检验文本生成（图像 → 文本）

训练完成后，用下面命令检验：输入一张图，模型生成诊断文本。

### 5.1 从验证集取一张图

```powershell
cd D:\mamba
conda activate mamba5090
python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_best_val.pt --max_visual_tokens 64 --max_new_tokens 80
```

### 5.2 指定 NIfTI 图像路径

```powershell
python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_best_val.pt --image "D:\nnunet_raw\Dataset503_TBLesion_327\imagesTr\0000978995_20260123_0000.nii.gz" --max_new_tokens 80
```

### 5.3 常用参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `--checkpoint` | Vision+Bridge 权重 | 默认会找 `outputs/vision_bridge_best_val.pt` 等 |
| `--max_visual_tokens` | 视觉 token 数量上限 | 64 更快、省显存；196 更慢 |
| `--max_new_tokens` | 生成文本最大 token 数 | 60–120 即可检验 |
| `--llm_device auto` | LLM 用 GPU（快） | 显存够用推荐 |
| `--llm_device cpu` | LLM 用 CPU（慢但不 OOM） | 显存不足时用 |

### 5.4 快速检验（少生成几个 token，便于快速跑通）

```powershell
python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_best_val.pt --max_visual_tokens 64 --max_new_tokens 20 --llm_device cpu
```

看到终端打印「输入图像」「问题」「生成文本」且无乱码，即说明图像→文本流程正常；若出现 `lm_head 已绑定到 input_embeddings`，说明词表已正确绑定。

### 5.5 避免 GPU OOM（CUDA out of memory）

- **默认**：脚本已改为 `--llm_device cpu`，直接运行即可，不会 OOM，但生成较慢。
- **使用 GPU**（推荐 5090）：
  - 方式 A：`--llm_device auto --max_visual_tokens 32`（脚本在 GPU 下会自动把视觉 token 上限压到 32）。
  - 方式 B：`--llm_device auto --llm_8bit`（需先 `pip install bitsandbytes`，8-bit 量化省显存；若当前 transformers 的 Mamba 不支持 8bit 会报错，可改用方式 A）。
- 推理里已做：prompt 截断到 128 token、vision 算完后 `torch.cuda.empty_cache()`，进一步减轻显存。

---

## 6. 论文方案：CMI 与 RoI（可选）

若遇到「生成卡死、OOM、乱码/无限重复」，多为视觉特征与 Mamba LLM 的握手问题，可启用论文方案：

| 方案 | 配置 | 作用 |
|------|------|------|
| **CMI**（Cross-Mamba Interaction） | `use_cmi: true`、可选 `cmi_compress_to: 512` | 用文本生成 SSM 参数，视觉流过 SSM 再融合，解耦视觉序列与上下文，减轻 OOM |
| **RoI 中心裁剪**（DRT-M3D 思路） | `roi_side: 7` | 对 2D 特征图中心裁剪为 7×7，视觉 token 从 784 降到 49，省显存且更聚焦 |

在 `config/paths.yaml` 中取消注释或添加：

```yaml
use_cmi: true
cmi_compress_to: 512
roi_side: 7
```

启用后需**重新训练**（或从无 CMI/RoI 的 checkpoint 非严格加载，此时 CMI 为随机初始化）。详见 `ARCHITECTURE.md` 第 7 节。
