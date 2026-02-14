# 推理与幻觉说明（lm_head、贪心解码、视觉对齐）

## 1. lm_head.weight | MISSING 是设计行为

**现象**：加载 Mamba 时日志显示 `lm_head.weight | MISSING`，随后执行「Tying lm_head → backbone.embeddings」。

**原因**：

- **Stage 2 只训练并保存 Vision+Bridge**（`vision_bridge_vlm_final.pt` 里只有 `vision_bridge.state_dict()`），不保存 LLM 权重。
- 预训练模型 `state-spaces/mamba-2.8b-hf` 的 checkpoint 本身不包含 `lm_head`（由 HF 实现决定）。
- 训练时 LLM 被冻结，且已将 `lm_head.weight` 绑定到 `backbone.embeddings.weight`（weight tying），因此**没有单独训练或保存 lm_head**。

**结论**：推理时把 lm_head 绑定到 backbone.embeddings 是**预期做法**，不是「权重丢失」。若你将来改为微调 LLM 并单独保存 lm_head，再在推理里加载该权重即可。

---

## 2. 幻觉与推理策略（SUV、PET-CT 术语、乱码）

**现象**：输入为**胸部 CT**，却生成 **SUV 2.7** 等 PET-CT 专用术语，或出现「张）」等乱码。

**原因**：

- **视觉-语言未对齐**：Stage 2 训练不足或 caption_loss 未充分下降时，模型更多依赖预训练文本分布，而不是当前图像。
- **采样带来随机性**：`do_sample=True` 时，模型易采样到与图像无关的医疗词汇（如 SUV、血管等）。

**建议**：

1. **默认使用贪心解码**：脚本已默认 `do_sample=False`（贪心），减轻与图像无关的幻觉。
2. 需要更多样性时再显式加 `--do_sample`。
3. 若仍出现 SUV、PET 等与 CT 不符的内容，或贪心解码下出现「外（IM12）、叛（IM13）」等模仿层号但用字错误，均说明**图文对齐不足**或**词表映射未学好**，需继续做 Stage 2 训练直至 caption_loss 明显下降。

**关于「IM12、IM13」乱码**：训练/验证集报告文本中本身含有「（IM110）」「（IM136）」等层号写法，模型在模仿该模式，但下一个字的预测错误（如应输出「1」却输出「外」），属**未对齐**或**词表映射**问题；并非「图像占位符 token 被当成输出」。训练时已对图像与提示部分做 label mask（labels 前 L_vis+q_len 为 -100），模型只学习预测报告正文。

**重要：本仓库未添加特殊 Token**  
`train_vlm.py` 与 `llm/mamba_loader.py` 中**没有** `add_special_tokens` 或 `resize_token_embeddings`。因此推理时**不要**在未与训练一致的情况下添加 `<IM12>` 等特殊 token 或执行 `resize_token_embeddings`，否则会引入新的错位（新 token 对应随机 embedding）。若你在其他分支/脚本里为训练加过特殊 token，推理时必须用**同一套** tokenizer 与 resize，再加载对应权重。

---

## 3. 视觉特征诊断（排查「看图不对齐」）

若怀疑模型「没看图」或输入异常（全 0/NaN），可在推理时加 `--debug_vision`：

```powershell
python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --debug_vision
```

会打印进入 LLM 前的视觉 token 统计：`min/max/mean/has_nan`。若 min≈max≈0 或 has_nan=True，需检查图像预处理与窗宽窗位（CT 的 windowing）。

---

## 4. Checkpoint 检测与词表一致性

**查看 .pt 里实际保存了哪些 key**（判断是否含 lm_head/backbone）：

```powershell
python scripts/inspect_checkpoint.py --checkpoint outputs/vision_bridge_vlm_final.pt
```

若输出为「仅 Vision+Bridge」：推理时 LLM 从 HF 加载，lm_head 需 Tying，与当前设计一致。若将来改为微调并保存 lm_head，需在推理中加载该权重并避免 Tying。

**词表一致性**：推理与训练需使用同一 tokenizer（同一 `mamba_model`）。若训练时曾改过 vocab_size 或加入特殊 token，推理时必须一致，否则易出现乱码或「张）」式错位。运行 `check_image_to_text.py` 时会打印 `len(tokenizer)` 与 `config.vocab_size`，二者应相等。

**max_visual_tokens 一致**：训练默认 `--max_visual_tokens` 为 196（或你传入的值），推理时建议与训练一致（例如训练用了 96 则推理也传 `--max_visual_tokens 96`），避免视觉序列长度不一致导致行为异常。

---

## 5. 训练配置核对清单（推理乱码时必查）

若出现「外（IM12）、叛（IM13）」式乱码，说明模型在模仿报告格式但缺乏正确视觉引导（Prediction Collapse），需按顺序排查：

| 步骤 | 操作 | 说明 |
|------|------|------|
| **1** | **查 Stage 2 真实配置** | 见下方「如何确认 Stage 2 的 max_visual_tokens」。未手动改过且用 `run_full_train.ps1` 则为 **196**；用 `run_stage2_train.ps1` 则为 **96**。 |
| **2** | **推理参数对齐** | 推理时显式传与训练一致的 `--max_visual_tokens`（如 196）。建议 `--no_do_sample` 或 `do_sample=False` 使用贪心解码。 |
| **3** | **词表一致性** | 运行 `check_image_to_text.py` 看终端：`len(tokenizer)` 与 `config.vocab_size` 应相等。不等则说明 tokenizer 加载方式与训练不一致（如 fast/slow 或不同 `from_pretrained` 路径），需改为与训练完全相同的 `mamba_model` 路径。 |
| **4** | **仍乱码 → 训练不足** | 若以上都对齐仍乱码，则 Stage 2 **训练不足**（caption_loss 未降到足够低，如 2.0 以下）。解决：用当前 `vision_bridge_vlm_final.pt` 继续 Stage 2；可调小学习率或确认 **冻结 LLM、只训 Projector**。 |

**原理简述**：训练时 N=196 个视觉 token，推理时若 N=96，相当于给模型「半张图」或密度突变，LLM 收到的是错位/噪声，只能按文本惯性乱输出。词表不一致则 ID 与 embedding 错位，同样导致乱码。

### 如何确认 Stage 2 的 max_visual_tokens

1. **看启动方式（最直接）**
   - 若用 **`run_stage2_train.ps1`** 启动 → 脚本里写死了 `--max_visual_tokens 96`，即训练用的是 **96**。推理时必须传 `--max_visual_tokens 96`。
   - 若用 **`run_full_train.ps1`** 或直接 `python train_vlm.py ...` 且未传 `--max_visual_tokens` → 使用 `train_vlm.py` 默认 **196**。推理用默认即可或显式 `--max_visual_tokens 196`。

2. **看 outputs 下的配置文件（若存在）**
   - 训练脚本会在 `output_dir`（默认 `outputs/`）下写入 **`stage2_config.json`**，内有 `max_visual_tokens`。例如：
     ```powershell
     type outputs\stage2_config.json
     ```

3. **看训练时的终端/日志**
   - 训练开始时会打印一行：`max_visual_tokens=96（推理时须与此一致）`。若你保存了当时终端输出或重定向到 log，可搜索该字符串。

**你当前情况**：若 `vision_bridge_vlm_final.pt` 是用 `run_stage2_train.ps1` 训出来的，则训练为 **96**，推理请用：
`python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_vlm_final.pt --max_visual_tokens 96 --llm_device auto`

## 6. mamba_ssm 与长序列数值一致性

未安装 mamba_ssm 时，Mamba 使用 **sequential 实现**（纯 PyTorch），功能正常但更慢。在极长序列生成时，sequential 与 CUDA kernel 可能存在**细微数值差异**，进而影响长文质量。若长文生成异常，可参考 `docs/FAQ_MAMBA_SSM_INSTALL.md` 尝试安装 mamba_ssm（建议在 Linux/WSL2 下安装）。
