# nnU-Net + Vim Bridge + Mamba-2 医学视觉-语言架构

## 设计目标

- **视觉侧**：nnU-Net 编码器特征，**减少下采样倍率**，保留 28×28 甚至更高分辨率（Mamba 能消化长序列）。
- **桥接器**：Vim (Vision Mamba) Block，将 2D 特征图转为序列并做双向 Mamba 建模。
- **语言侧**：使用 **Mamba-2.8B** 或 **OpenElm** 等预训练 checkpoint 初始化，不从头训练；医学显存受限场景下纯 Mamba 故事更清晰。

## 创新点（论文图）

| 方案 | 视觉特征分辨率 | 原因 |
|------|----------------|------|
| **Llama / 传统 VLM** | 压成 14×14（甚至更小） | Attention 对序列长度 O(L²)，必须压缩 |
| **本方案 (Mamba)** | 保留 **28×28** 或更高 | Mamba 线性复杂度 O(L)，可保留高分辨率 |

论文用图见：`scripts/plot_llama_vs_mamba_resolution.py` 生成。

---

## 1. 视觉侧：nnU-Net 编码器（“变长”、轻下采样）

- 沿用现有 **nnU-Net 2D** 编码器结构（与 `Dataset503_TBLesion_327` / `nnUNetPlans_2d` 一致），但**不把特征压到 14×14**。
- **下采样倍率**：默认 2D 为 7 次 stride=2 → 512→4；改为**只取前若干 stage**，使瓶颈特征图为 **28×28** 或 **32×32**。
  - 例如：仅用 4 次 2× 下采样 → 512/16 = **32×32**；或 3 次 → **64×64**。
- 实现上：从 nnU-Net 编码器**中间层**取出特征（如 stage 4 输出 32×32），或使用自定义的 “light” plans（减少 encoder stages / strides），保证输出空间尺寸 ≥ 28×28。

**数据与权重**：

- 输入：`d:\nnunet_preprocessed\Dataset503_TBLesion_327`（2D 预处理）或 `d:\nnunet_raw` 的 NIfTI。
- 编码器权重：可加载 `d:\nnunet_results\Dataset503_TBLesion_327\nnUNetTrainer__nnUNetPlans__2d\fold_0\checkpoint_best.pth` 的 encoder 部分（需与 “轻下采样” 结构对齐或做 stage 截断）。

---

## 2. 桥接器：Vim (Vision Mamba) Block

- **输入**：nnU-Net 编码器输出的 2D 特征图 `[B, C, H, W]`（如 H=W=28 或 32）。
- **步骤**：
  1. 展平为序列：`[B, C, H*W]` → 重排为 `[B, H*W, C]`。
  2. 可选：线性投影到 Mamba 的 `d_model`。
  3. 通过 **Vision Mamba (Vim) Block**：双向或单向 Mamba，得到 `[B, L, D]`，L = H×W。
- **输出**：与文本 token 拼接后送入 Mamba LLM。

Vim 的 “2D→1D” 做法：按行展平为 1D 序列，用双向 Mamba 建模（见文献）。本仓库实现见 `bridge/vim_bridge.py`。

---

## 3. LLM 侧：预训练 Mamba，不从头训练

- **首选**：**Mamba-2.8B**（HuggingFace: `state-spaces/mamba-2.8b-hf`），用 `MambaForCausalLM` + `AutoTokenizer` 加载。
- **备选**：**OpenElm** 等其它 Mamba 系预训练模型。
- **策略**：仅训练 **视觉编码器 + 桥接器 + 投影层**，或对 LLM 做 LoRA/轻量微调；**不从头训练 Mamba**。
- 医学显存受限时优先 **纯 Mamba**，避免 Jamba 等混合架构带来的额外显存与实现复杂度。

---

## 4. 数据与配置

- **nnU-Net 相关路径**（在 `config/paths.yaml` 或环境变量中）：
  - `nnunet_raw`: `d:\nnunet_raw`
  - `nnunet_preprocessed`: `d:\nnunet_preprocessed`
  - `nnunet_results`: `d:\nnunet_results`
  - `nnunet_data`: `d:\nnunet_data`
- **报告/问答数据**：沿用 `d:\unn-net` 的 RadFM 风格 CSV（`image_path`, `question`, `answer`）及 `train_radfm_315.csv` / `val_radfm_315.csv`、`radfm_caption_prompt.json`，可与 nnU-Net 预处理或 raw NIfTI 路径对齐。
- **Dataset**：可复用 `unn-net/radfm_dataset_lung_caption.py` 的接口，改为从 nnU-Net 2D slice 或 3D 体数据中取 patch，并走 nnU-Net 编码器 + Vim + Mamba 管线。

---

## 5. 目录结构（本仓库 d:\mamba）

```
mamba/
├── ARCHITECTURE.md           # 本说明
├── config/
│   └── paths.yaml           # nnunet_* 与 checkpoint 路径
├── vision/
│   └── nnunet_encoder.py    # nnU-Net 编码器，轻下采样 → 28×28/32×32
├── bridge/
│   ├── vim_bridge.py        # Vim Block：2D 特征 → 序列 → Mamba
│   └── cmi_connector.py     # CMI：文本→SSM 参数，视觉流过 SSM（可选）
├── llm/
│   └── mamba_loader.py      # 加载 Mamba-2.8B / OpenElm
├── data/
│   └── medical_vlm_dataset.py  # 对接 nnunet + CSV caption
├── model/
│   └── forward_medical_vlm.py # Vision + Bridge 前向，输出视觉 token
├── scripts/
│   └── plot_llama_vs_mamba_resolution.py  # 论文图：Llama 压缩 vs Mamba 高分辨率
└── requirements.txt
```

---

## 6. 论文图说明

脚本 `scripts/plot_llama_vs_mamba_resolution.py` 生成一张对比图：

- **左**：传统方案（如 Llama-VLM）— 视觉特征被压缩到 14×14，信息损失。
- **右**：本方案 — Mamba 线性复杂度，保留 28×28 甚至更高分辨率特征。

可用于顶刊/顶会的方法对比图。

---

## 7. 论文方案：Connector 握手与 OOM 缓解

视觉特征（Vim 输出）与 Mamba LLM 之间的「握手」失败会导致生成卡死、显存 OOM 或乱码/无限重复。本仓库实现了两篇论文的解法，可通过配置开关启用。

### 7.1 CMI（Cross-Mamba Interaction）— 推荐

**来源**：CMI-MTL (2511.01357)。

**问题**：把视觉 token 简单 concat 到文本前，Mamba 的状态空间未正确用视觉信息初始化，LLM 容易「看不见」图像。

**做法**：
- 不把视觉特征当 token 直接拼接，而是当**参数**：用**文本 prompt** 生成 SSM 的参数 (Δ, B, C)。
- 让 Vim 输出的视觉特征流过该**文本条件 SSM**，再投影到 LLM 维度。
- 解耦视觉序列长度与 LLM 上下文，避免生成阶段显存爆炸或卡顿。

**实现**：`bridge/cmi_connector.py` 中的 `CMIConnector`；在 `MedicalVLM` 中通过 `use_cmi: true` 与可选 `cmi_compress_to` 启用。训练与推理时都会用「问题」嵌入作为 text 输入生成 SSM 参数。

**配置**（`config/paths.yaml`）：
```yaml
use_cmi: true
cmi_compress_to: 512   # 可选，将视觉序列压缩到固定长度
```

### 7.2 RoI / 中心裁剪（DRT-M3D Tandem 思路）

**来源**：Dual Res Tandem Mamba 3D (3772_Dual_Res_Tandem_Mamba_3D)。

**问题**：3D/大图经 Vim 后 token 过多（如 10k+），导致 OOM 或生成极慢。

**做法**：不把整图特征都喂给 LLM；对 2D 特征图做**中心裁剪**（RoI），只保留中心区域（如 7×7），使视觉 token 从 784 降到 49，既省显存又让特征更聚焦。

**实现**：在 `MedicalVLM` 的 `forward` 中，encoder 输出后若设置 `roi_side`，则对 `[B,C,H,W]` 做中心裁剪为 `[B,C,roi_side,roi_side]`，再送入 Vim Bridge。对应配置：

```yaml
roi_side: 7   # 7×7=49 个视觉 token
```

可与 CMI 同时使用：先 RoI 减 token，再 CMI 融合。
