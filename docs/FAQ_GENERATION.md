# 生成结果是乱码（monster/abroad 等）说明

## 现象

- 先重复问题「请分析这幅胸部CT影像…」，后面变成无意义英文重复词（monster、abroad、unfavorable 等）。
- 加载时出现：`lm_head.weight | MISSING | those params were newly initialized`。

## 原因（为何不对）

1. **lm_head 未从 checkpoint 加载**  
   Mamba 2.8B 的 HF 权重里可能没有 `lm_head.weight`，或 key 与当前 `MambaForCausalLM` 不一致，导致输出头被**随机初始化**。用随机头做生成会得到乱码或重复词，所以当前输出是**错的**。

2. **当前只训了 Vision+Bridge**  
   训练阶段只优化了视觉编码器 + Vim 桥接，用的是代理损失（L2），**没有**用「图像–报告」的文本损失去对齐 Mamba。因此即便 lm_head 正常，模型也从未学过「看到 CT → 写报告」，生成质量也会很有限。

## 正确输出应长什么样

- 应是一段**连贯的中文诊断描述**，例如：「影像所见：双肺…；诊断意见：…」。
- 不应是：重复问题 + 一堆无关英文词。

## 已做修复（本项目）

1. **lm_head 与 backbone.embeddings 绑定**  
   在 `llm/mamba_loader.py` 中，加载 Mamba 后执行：  
   `model.lm_head.weight = model.backbone.embeddings.weight`  
   这样不会再出现「lm_head 随机初始化 → 乱码」。重新跑检测即可验证。

2. **VLM 图像→报告训练**  
   运行 `python train_vlm.py --epochs 2 --batch_size 1` 会用「问题+报告」做 caption 损失，只训练 Vision+Bridge，得到 `outputs/vision_bridge_vlm_final.pt`。再用该权重做生成效果更好。

3. **一键脚本**  
   `.\run_fix_and_train_then_generate.ps1` 会先跑 VLM 训练再跑生成检测。

## 可以怎么做（可选）

1. **确认/修复 lm_head 加载**  
   若 HF 上该模型实际把输出层放在别的 key 下（例如 tied embedding），需要在加载后做一次 key 映射或手动把对应权重赋给 `lm_head`，避免 MISSING。可先检查 checkpoint 里所有 key（例如 `state_dict` 的 keys），再对 `lm_head` 做对齐。

2. **换用完整加载的 Mamba 做测试**  
   例如先试更小且确认能完整加载的模型（如 `mamba-130m-hf`），看同一套「图像 → 视觉 token → Mamba」流程是否至少能生成**连贯句子**（哪怕内容一般）。若小模型正常、2.8B 仍乱码，多半仍是 2.8B 的 lm_head 未正确加载。

3. **做真正的「图像→报告」训练**  
   要得到可用报告，需要：
   - 用「图像 + 问题」作为输入、**报告文本**作为监督，
   - 对 Vision + Bridge +（可选）Mamba 或至少对「视觉→语言」的投影/适配层做 **caption/报告生成损失**（如 cross-entropy on next token），  
   而不是只训代理损失。这样模型才会学到「看到影像 → 生成诊断文本」。

总结：**当前这种「先重复问题 + 英文乱码」的输出是不对的**；要修需要解决 lm_head 加载问题，并做图像–报告联合训练才能得到正确、可用的报告生成。
