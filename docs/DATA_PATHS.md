# 数据与文档路径说明（2D / 3D / 病例文档）

以下路径与 `config/paths.yaml` 对应，供训练脚本自动查找。

---

## 1. 预处理好的 2D 数据（nnU-Net）

| 用途     | 路径 |
|----------|------|
| 2D 预处理 | `d:\nnunet_preprocessed\Dataset503_TBLesion_327\nnUNetPlans_2d\` |
| 格式     | `*_0000.b2nd` / `*_seg.b2nd`、`*.pkl`（nnUNet 预处理后） |
| 说明     | 与 Dataset503 2D plans 一致，patch 512×512；需 nnUNet 库读取 b2nd。 |

---

## 2. 预处理好的 3D 数据（nnU-Net）

| 用途     | 路径 |
|----------|------|
| 3D 全分辨率 | `d:\nnunet_preprocessed\Dataset503_TBLesion_327\nnUNetPlans_3d_fullres\` |
| 3D 低分辨率 | `d:\nnunet_preprocessed\Dataset503_TBLesion_327\nnUNetPlans_3d_lowres\` |
| 格式     | 同上，`.b2nd` / `.pkl` |
| 说明     | 当前训练脚本使用 **raw NIfTI + 2D slice** 以简化依赖；若需直接用 3D 预处理，可接 nnUNet 的 Dataset 类。 |

---

## 3. Raw 影像（NIfTI）

| 用途     | 路径 |
|----------|------|
| 原始影像 | `d:\nnunet_raw\Dataset503_TBLesion_327\imagesTr\*.nii.gz` |
| 标签     | `d:\nnunet_raw\Dataset503_TBLesion_327\labelsTr\*.nii.gz` |
| 说明     | 训练时 **MedicalVLMDataset** 默认从 CSV 的 `image_path` 读 NIfTI，取中间 slice 做 2D 输入。 |

---

## 4. 病例 / 文档（报告与问答）

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练 CSV | `d:\unn-net\train_radfm_315.csv` | 列：`image_path`, `question`, `answer` |
| 验证 CSV | `d:\unn-net\val_radfm_315.csv` | 同上 |
| Prompt JSON | `d:\unn-net\radfm_caption_prompt.json` | `caption_prompt` 列表，用于随机 question |
| 指令数据 (327) | `d:\unn-net\radfm_instruction_data_327.json` | 多轮对话格式：`image` + `conversations` (human/gpt) |
| 指令数据 | `d:\unn-net\radfm_instruction_data.json` | 同上 |

训练脚本默认使用 **train_radfm_315.csv** 与 **val_radfm_315.csv**（及 radfm_caption_prompt.json），与 RadFM 风格一致。

---

## 5. nnU-Net 训练结果（编码器权重）

| 用途     | 路径 |
|----------|------|
| 2D 最佳权重 | `d:\nnunet_results\Dataset503_TBLesion_327\nnUNetTrainer__nnUNetPlans__2d\fold_0\checkpoint_best.pth` |
| 3D 全分辨率 | `d:\nnunet_results\Dataset503_TBLesion_327\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_best.pth` |

`config/paths.yaml` 中的 `nnunet_encoder_checkpoint` 指向 2D 权重，用于视觉编码器初始化（若 key 匹配）。
