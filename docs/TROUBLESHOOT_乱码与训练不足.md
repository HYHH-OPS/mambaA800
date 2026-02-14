# 问题分析：生成「外(IM12)、叛(IM13)」乱码

## 你的问题出在哪里（两处）

### 1. 词表不一致导致「多出来的 3 个 token」被解码成乱字（已修）

- **现象**：终端里 `len(tokenizer)=50277`，`config.vocab_size=50280`，生成里出现「外、叛、叻」等单字。
- **原因**：模型输出的 logits 维度是 50280，argmax 可能得到 50277、50278、50279 这三个 ID；tokenizer 只有 50277 个 token，对这三个 ID 的解码行为未定义或会映射到错误字符，就变成乱码。
- **已做修改**：在 `inference.py` 里增加了 **`_VocabSizeMaskProcessor`**：在生成时把「ID ≥ 50277」的 logits 置为 -inf，这样永远不会选到 50277/50278/50279，解码只在 tokenizer 有效范围内，可减轻这类乱字。
- **你需要做的**：无需改命令，直接用当前代码重新跑推理即可（同上 checkpoint + `--max_visual_tokens 96`）。

---

### 2. Stage 2 训练严重不足（主要矛盾）

- **现象**：`outputs/stage2_train_log.csv` 里只有很少几行（例如 step 20、40），caption_loss 约 1.7 → 1.18，远未跑完 20 个 epoch。
- **原因**：模型几乎没学过「看图→对应报告」的映射。训练数据里有很多「（IM110）」这类层号，模型只学会了「括号+IM+数字」的格式，但该输出什么字还没学好（预测崩塌），所以会胡编单字。
- **结论**：**继续把 Stage 2 训下去** 是解决乱码的根本办法；只修词表可以少一些怪字，但若 caption_loss 没下去，整体仍会不通顺或乱写。

---

## 建议执行顺序

1. **先按原指令继续 Stage 2**（把整段 20 epoch 跑完或至少跑满几小时）  
   - `cd D:\mamba` → `conda activate mamba5090` → `.\run_stage2_continue.ps1`  
   - 或：`python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --max_visual_tokens 96`

2. **看 loss**  
   - 看终端里每 20 step 的 caption_loss、每 epoch 结束的平均 loss。  
   - 用 Excel 打开 `D:\mamba\outputs\stage2_train_log.csv` 画折线图，确认 loss 明显下降并趋于稳定（目标可先按「平均 caption_loss 到 2.0 以下」再测生成）。

3. **再跑推理**  
   - 训练一段时间（或跑完）后再用当前权重测：  
     `python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 96 --llm_device auto --max_new_tokens 512`  
   - 此时推理已带「词表 mask」修复，不会选到 50277/50278/50279；若仍乱码，多半是 Stage 2 还需继续训或稍调小 lr 再训。

4. **可选：确认训练配置**  
   - `type D:\mamba\outputs\stage2_config.json`  
   - 推理时用的 `--max_visual_tokens` 和 `--mamba_model` 与这里一致即可。

---

## 小结

| 问题 | 原因 | 处理 |
|------|------|------|
| 50277 vs 50280 导致怪字 | 模型可能输出 50277/50278/50279，tokenizer 解码异常 | 已加 logits mask，推理时自动生效 |
| 整段仍乱码、格式对内容错 | Stage 2 只跑了很少 step，caption_loss 未下去 | 继续跑 Stage 2（跑满 20 epoch 或看 loss 曲线再决定），再测生成 |

当前你的 **stage2_config.json** 里已是 `max_visual_tokens=96`、`mamba_model=state-spaces/mamba-2.8b-hf`，与推理命令一致，无需改配置；**把 Stage 2 认真跑完/跑久一点** 是下一步最关键的一步。

---

## 关于 Loss 异常与「练废」的 Stage 1

若你看到 **overnight 日志**里 loss 掉到 0.00005，那是 **Stage 1（train.py）** 的 **proxy L2 loss**，不是 Stage 2 的 caption loss。该次 Stage 1 权重已「练废」（视觉输出趋近零），**不要**用作 Stage 2 的初始化。Stage 2 请用 `--from_scratch` 或直接续训已有的 `vision_bridge_vlm_final.pt`。  
完整说明与健康 Loss 曲线预期见：**docs/LOSS_STAGE1_STAGE2_AND_COLLAPSE.md**。
