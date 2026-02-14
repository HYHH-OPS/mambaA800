"""
CMI-style Connector (Cross-Mamba Interaction): 用文本生成 SSM 参数，让视觉特征流过 SSM，再投影到 LLM 维度。

参考 CMI-MTL 思路：不把视觉特征当 token 简单拼接，而是用文本控制 SSM(Δ,B,C)，
让视觉序列流过该 SSM 后与文本融合，解耦视觉序列长度与 LLM 上下文，减轻 OOM/卡顿。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CMIConnector(nn.Module):
    """
    文本条件 SSM：用 text_embeds 生成 (delta, B, C)，视觉特征 x 流过 SSM 得到 y，再投影到 d_model。
    输入: visual_feats [B, Lv, Dv], text_embeds [B, Lt, Dt]
    输出: [B, Lv, d_model] 或 [B, L_out, d_model]（若 compress 则对 Lv 做池化）
    """

    def __init__(
        self,
        d_visual: int,
        d_text: int,
        d_model: int,
        d_state: int = 64,
        compress_to: int | None = None,
    ):
        super().__init__()
        self.d_state = d_state
        self.compress_to = compress_to
        self.x_proj = nn.Linear(d_visual, d_state)
        # 文本 context -> SSM 参数（标量 delta，向量 B、C）
        self.proj_delta = nn.Sequential(nn.Linear(d_text, d_state), nn.Sigmoid())  # [B, d_state] 逐元做 gate
        self.proj_B = nn.Linear(d_text, d_state)
        self.proj_C = nn.Linear(d_text, d_state)
        self.out_proj = nn.Linear(d_state, d_model)

    def forward(
        self,
        visual_feats: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        visual_feats: [B, Lv, Dv]
        text_embeds: [B, Lt, Dt]
        return: [B, L', d_model]，L' = Lv 或 compress_to
        """
        # 重要：对齐 device + dtype，避免 cpu/cuda 或 bf16/float32 混用导致 Linear 报错
        target_device = self.x_proj.weight.device
        target_dtype = self.x_proj.weight.dtype

        if visual_feats.device != target_device:
            visual_feats = visual_feats.to(device=target_device, non_blocking=True)
        if text_embeds.device != target_device:
            text_embeds = text_embeds.to(device=target_device, non_blocking=True)
        if visual_feats.dtype != target_dtype:
            visual_feats = visual_feats.to(dtype=target_dtype)
        if text_embeds.dtype != target_dtype:
            text_embeds = text_embeds.to(dtype=target_dtype)

        B, Lv, _ = visual_feats.shape
        text_context = text_embeds.mean(dim=1)  # [B, Dt]
        delta = self.proj_delta(text_context)   # [B, d_state]
        B_vec = self.proj_B(text_context)       # [B, d_state]
        C_vec = self.proj_C(text_context)       # [B, d_state]
        x = self.x_proj(visual_feats)           # [B, Lv, d_state]

        # 离散 SSM 扫描: h_t = (1 - delta) * h_{t-1} + delta * B_vec * x_t, y_t = C_vec * h_t
        # delta/B_vec/C_vec: [B, d_state], x: [B, Lv, d_state]
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(Lv):
            h = (1 - delta) * h + delta * B_vec * x[:, t, :]
            yt = C_vec * h
            ys.append(yt)
        y = torch.stack(ys, dim=1)  # [B, Lv, d_state]

        out = self.out_proj(y)  # [B, Lv, d_model]
        if self.compress_to is not None and Lv > self.compress_to:
            # 均匀取 compress_to 个位置，模拟 RoI 压缩
            indices = torch.linspace(0, Lv - 1, self.compress_to, device=out.device).long()
            out = out[:, indices, :]
        return out
