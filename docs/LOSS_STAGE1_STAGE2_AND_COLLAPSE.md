# Loss 异常与 Stage 1/Stage 2 区别（防「练废」）

## 1. 核心诊断：两种 Loss 不要搞混

| 日志/文件 | 对应脚本 | Loss 含义 | 健康范围 |
|-----------|----------|-----------|----------|
| **overnight_lr0.0001_bs2_....log** | **train.py（Stage 1）** | **Proxy L2**：`visual_tokens.pow(2).mean()`，只训 Vision+Bridge，无 LLM | 数值可以很低（甚至 0.00005），但**过低说明「练废」** |
| **stage2_train_log.csv** | **train_vlm.py（Stage 2）** | **Caption CrossEntropy**：预测报告正文的下一个 token | **1.0～3.0 起步，目标 0.5～1.0**，绝不应长期 &lt; 0.1 |

- **Stage 1** 的 loss 不是 CrossEntropy。它把「视觉 token 的 L2 范数」压小，模型很容易学会输出接近 0 的向量，loss 就会掉到 0.0000x。此时**视觉特征已退化**，不适合作为 Stage 2 的初始化。
- **Stage 2** 的 loss 才是「正常 NLP 意义」上的 loss：预测下一个词，健康时应在 0.5～2.0 之间；若一开始就 &lt; 0.1，多半是 **label masking 错误**（例如把 padding 或图像部分也算进 loss）。

---

## 2. 为什么 overnight 日志会「崩塌」？

你看到的 **overnight_lr0.0001_bs2_20260204_0132.log** 是 **Stage 1（train.py）**，不是 Stage 2。崩塌原因可以归纳为：

1. **学习率过大（lr=0.0001 = 1e-4）**  
   Stage 1 默认已改为 1e-5；若当时用 1e-4，模型会很快把 proxy loss 压到接近 0（视觉输出趋近零）。

2. **数据量小（训练集 254 条）**  
   小数据 + 高学习率 → 很快过拟合到「输出接近零」这一 trivial 解。

3. **Stage 1 没有 LLM**  
   train.py 只训 Vision+Bridge，不涉及 LLM，所以不存在「LLM 未冻结」的问题；崩塌纯粹是 **proxy 目标太简单 + 高 LR** 导致。

**结论**：该次 Stage 1 产出的 **vision_bridge_best_val.pt / vision_bridge_final.pt**（以及同一次 run 的 step 检查点）**不要**再用来初始化 Stage 2，否则相当于从「已退化的视觉特征」继续训。

---

## 3. 解决建议（Action Plan）

### 第一步：停用「练废」的 Stage 1 权重

- 凡是由 **loss 已掉到 0.0000x** 的那次 Stage 1 跑出来的 checkpoint（如 `vision_bridge_best_val.pt`、`vision_bridge_final.pt`、`vision_bridge_step500.pt` 等），**不要再作为 Stage 2 的 --vision_checkpoint**。
- 可选：移到备份目录或删除，避免误用。  
  ```powershell
  # 可选：备份后删除，避免误用
  # Move-Item D:\mamba\outputs\vision_bridge_final.pt D:\mamba\outputs\archive_collapsed_stage1\
  ```

### 第二步：Stage 2 必须用的配置

- **Learning rate**：**2e-5（0.00002）**，不要用 1e-4。你的 `stage2_config.json` 里已是 2e-05，保持即可。
- **Batch size**：1 或 2（显存允许可 2）。
- **Epochs**：数据少时建议 **50～100**；先跑 20 看曲线，再决定是否加 epoch。
- **初始化**：用 **`--from_scratch`** 或**不传** `--vision_checkpoint`（脚本会优先用 `vision_bridge_vlm_final.pt` 续训）；**不要**用那次「loss 0.0000x」的 Stage 1 权重。

### 第三步：代码侧已加强的检查（防再次崩塌）

- **train.py（Stage 1）**  
  - 默认 lr 已改为 **1e-5**。  
  - 若某 epoch 平均 train_loss &lt; 0.001，会打印提示：不建议用本阶段权重初始化 Stage 2，Stage 2 请用 `--from_scratch`。

- **train_vlm.py（Stage 2）**  
  - **LLM 冻结**：启动时检查「所有 LLM 参数 `requires_grad=False`」，否则直接报错。  
  - **Label masking**：首步若 caption_loss &lt; 0.1，会警告「请检查 label masking（视觉+问题部分应为 -100）」。

### 第四步：健康的 Stage 2 Loss 曲线预期

Caption loss（CrossEntropy）大致应呈以下趋势：

- **Start**：约 **3.0～5.0**（或更高）
- **Epoch 5**：约 **2.0～2.5**
- **Epoch 20**：约 **1.0～1.5**
- **End**：约 **0.5～0.8**（**不应长期低于 0.1**；若长期 &lt; 0.1 且生成仍乱码，多半是 label 或数据有问题）

数据很少（如 254 条）时，可以多跑 epoch（50～100），并配合 `stage2_train_log.csv` 画曲线判断是否收敛。

---

## 4. 小结

| 项目 | 说明 |
|------|------|
| **overnight 日志里 0.00005** | Stage 1 的 **proxy L2 loss**，不是 CrossEntropy；过低 = 视觉特征退化，该次 Stage 1 权重不要给 Stage 2 用。 |
| **stage2_train_log 里 1.7** | Stage 2 的 **caption loss**，属于健康范围，需继续训到 0.5～1.0 再评估生成。 |
| **防止再次崩塌** | Stage 1 用 1e-5；Stage 2 用 2e-5、确认 LLM 冻结、不用「loss 极低」的 Stage 1 权重；用 `--from_scratch` 或已有 `vision_bridge_vlm_final.pt` 续训。 |
| **健康 caption loss** | 起步 3～5，结束 0.5～0.8，不应长期 &lt; 0.1。 |

当前 **run_stage2_continue.ps1** 与 **stage2_config.json** 已满足「lr=2e-5、max_visual_tokens=96、从 vlm_final 续训」；只要**不要**用那次 0.0000x loss 的 Stage 1 权重，按现有流程继续跑 Stage 2 即可。
