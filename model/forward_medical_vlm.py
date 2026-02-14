"""
医学 VLM 前向：Vision (nnU-Net 轻下采样) → [可选 RoI 中心裁剪] → Vim Bridge → [可选 CMI] → 视觉 token。

- RoI（DRT-M3D 思路）：对 2D 特征图做中心裁剪，减少视觉 token 数，减轻 OOM。
- CMI（Cross-Mamba Interaction）：用文本生成 SSM 参数，视觉特征流过 SSM 再投影，解耦视觉序列与 LLM 上下文。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# 项目内模块（需从 repo 根运行或 PYTHONPATH 包含 d:\mamba）
try:
    from vision import build_nnunet_encoder_light
    from bridge import VimBridge, CMIConnector
except ImportError:
    import sys
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from vision import build_nnunet_encoder_light
    from bridge import VimBridge, CMIConnector


def _center_crop_2d(feat: torch.Tensor, size: int) -> torch.Tensor:
    """对 [B, C, H, W] 做中心裁剪为 [B, C, size, size]。"""
    _, _, H, W = feat.shape
    if H < size or W < size:
        return feat
    start_h = (H - size) // 2
    start_w = (W - size) // 2
    return feat[:, :, start_h : start_h + size, start_w : start_w + size]


class MedicalVLM(nn.Module):
    """
    Vision: nnU-Net 编码器 → [可选 RoI 中心裁剪] → Vim Bridge → [可选 CMI] → 视觉 token [B, L, D]。
    LLM 部分不在此类中；调用方将 visual_tokens 与 text embeddings 拼接后送入 Mamba。
    """

    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        encoder_output_stage: int = 4,
        encoder_target_spatial: int = 28,
        use_pooling: bool = True,
        pool_size: int = 12,
        bridge_d_model: int = 2560,
        bridge_bidirectional: bool = True,
        roi_side: Optional[int] = None,
        use_cmi: bool = False,
        cmi_compress_to: Optional[int] = None,
        cmi_d_state: int = 64,
    ):
        super().__init__()
        self.encoder = build_nnunet_encoder_light(
            checkpoint_path=encoder_checkpoint,
            output_stage_index=encoder_output_stage,
            target_spatial_size=encoder_target_spatial,
        )
        self.use_pooling = use_pooling
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size)) if use_pooling else nn.Identity()
        self.bridge = VimBridge(
            in_channels=self.encoder.out_channels,
            d_model=bridge_d_model,
            bidirectional=bridge_bidirectional,
        )
        self.bridge_d_model = bridge_d_model
        self.roi_side = roi_side
        spatial = encoder_target_spatial
        if roi_side is not None:
            spatial = min(roi_side, spatial)
        if self.use_pooling:
            spatial = self.pool_size
        self.visual_seq_len = spatial * spatial

        self.cmi_connector: Optional[CMIConnector] = None
        if use_cmi:
            # d_text = LLM 嵌入维（与 bridge_d_model 一致），视觉来自 bridge 输出
            self.cmi_connector = CMIConnector(
                d_visual=bridge_d_model,
                d_text=bridge_d_model,
                d_model=bridge_d_model,
                d_state=cmi_d_state,
                compress_to=cmi_compress_to,
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [B, 1, H, W]，如 [B, 1, 512, 512]
        return: [B, L, D] 视觉 token 序列，L 由 encoder 空间尺寸与 roi_side 决定，D=bridge_d_model
        """
        feat = self.encoder(image)
        if self.roi_side is not None:
            feat = _center_crop_2d(feat, self.roi_side)
        feat = self.pool(feat)
        return self.bridge(feat)


def build_medical_vlm_from_config(config: dict) -> MedicalVLM:
    """从 config（如 config/paths.yaml）构建 MedicalVLM。"""
    return MedicalVLM(
        encoder_checkpoint=config.get("nnunet_encoder_checkpoint"),
        encoder_output_stage=config.get("encoder_output_stage", 4),
        encoder_target_spatial=config.get("encoder_target_spatial", 28),
        use_pooling=config.get("use_pooling", True),
        pool_size=config.get("pool_size", 12),
        bridge_d_model=config.get("bridge_d_model", 2560),
        bridge_bidirectional=config.get("bridge_bidirectional", True),
        roi_side=config.get("roi_side"),
        use_cmi=config.get("use_cmi", False),
        cmi_compress_to=config.get("cmi_compress_to"),
        cmi_d_state=config.get("cmi_d_state", 64),
    )
