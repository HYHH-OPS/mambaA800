# Stage 2 续训与推理：生成物路径一览

## 一、你运行后「最后生成的东西」的地址

### 1. Stage 2 训练产生的文件（在 `outputs/` 下）

| 路径 | 说明 |
|------|------|
| **`D:\mamba\outputs\vision_bridge_vlm_final.pt`** | Stage 2 最终权重（续训会覆盖），推理时用 `--checkpoint` 指向它。 |
| **`D:\mamba\outputs\stage2_config.json`** | 本次训练配置：`max_visual_tokens`、`mamba_model`、`lr`、`epochs` 等；推理时与之一致。 |
| **`D:\mamba\outputs\stage2_train_log.csv`** | 每 20 step 一条的 caption_loss 记录（列：step, epoch, caption_loss），用于画曲线、判断是否继续训。 |

### 2. 推理（生成报告）产生的目录与文件

| 路径 | 说明 |
|------|------|
| **`D:\mamba-res\run_YYYYMMDD_HHMMSS\`** | 每次运行 `check_image_to_text.py` 会新建一个以时间命名的子目录。 |
| **`D:\mamba-res\run_YYYYMMDD_HHMMSS\meta.json`** | 当次运行的参数与路径记录。 |
| **`D:\mamba-res\run_YYYYMMDD_HHMMSS\gen.txt`** | 当次生成的文本（若脚本写了该文件）。 |
| 若使用了 `--save_out 某路径` | 生成内容会额外保存到你指定的那个文件地址。 |

### 3. 脚本与文档（已实现好的东西）

| 路径 | 说明 |
|------|------|
| **`D:\mamba\run_stage2_continue.ps1`** | 一键从 `vision_bridge_vlm_final.pt` 续训（max_visual_tokens=96，20 epoch）。 |
| **`D:\mamba\train_vlm.py`** | Stage 2 训练入口；不传 `--vision_checkpoint` 时自动用 `outputs/vision_bridge_vlm_final.pt` 续训。 |
| **`D:\mamba\scripts\check_image_to_text.py`** | 图像→文本推理；会读 `stage2_config.json` 并提示是否与训练配置一致。 |
| **`D:\mamba\docs\TRAIN_COMMANDS.md`** | 训练命令说明，含「3.2b 继续 Stage 2」与 loss 查看方式。 |
| **`D:\mamba\docs\INFERENCE_AND_HALLUCINATION.md`** | 推理与幻觉说明，含训练配置核对清单、如何确认 max_visual_tokens。 |

---

## 二、常用命令与对应生成物

- **续训 Stage 2（并生成上述 3 个 outputs 文件）**
  ```powershell
  D:\mamba\run_stage2_continue.ps1
  ```
  或：
  ```powershell
  cd D:\mamba
  python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96
  ```
  → 生成/覆盖：`D:\mamba\outputs\vision_bridge_vlm_final.pt`、`stage2_config.json`、`stage2_train_log.csv`。

- **推理（生成报告并落盘）**
  ```powershell
  python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 96 --llm_device auto
  ```
  → 生成：`D:\mamba-res\run_YYYYMMDD_HHMMSS\` 及其中 `meta.json` 等。

---

## 三、总结：最后生成的东西地址

- **训练产物**：`D:\mamba\outputs\` 下的  
  - `vision_bridge_vlm_final.pt`  
  - `stage2_config.json`  
  - `stage2_train_log.csv`  

- **推理产物**：`D:\mamba-res\run_YYYYMMDD_HHMMSS\` 及你通过 `--save_out` 指定的文件。

- **脚本与文档**：见上表，均在 `D:\mamba\` 与 `D:\mamba\docs\`。
