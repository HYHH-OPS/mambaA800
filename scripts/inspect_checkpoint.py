"""
检测 Vision+Bridge 或 VLM  checkpoint 中实际保存了哪些权重。
用于判断：是否只含 vision/bridge（设计如此）、是否误含 lm_head/backbone、以及 key 前缀。

用法:
  python scripts/inspect_checkpoint.py
  python scripts/inspect_checkpoint.py --checkpoint outputs/vision_bridge_vlm_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="查看 .pt checkpoint 的 keys")
    ap.add_argument("--checkpoint", type=str, default=None, help=".pt 路径，默认 outputs/vision_bridge_vlm_final.pt")
    args = ap.parse_args()

    ckpt_path = args.checkpoint or str(REPO / "outputs" / "vision_bridge_vlm_final.pt")
    path = Path(ckpt_path)
    if not path.exists():
        print(f"文件不存在: {path}")
        return 1

    print(f"加载: {path}", flush=True)
    ckpt = __import__("torch").load(path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict):
        top_keys = list(ckpt.keys())
        print(f"顶层 keys ({len(top_keys)} 个): {top_keys}", flush=True)

        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
        top_keys = []
        print("Checkpoint 为 state_dict 本身（无顶层包装）", flush=True)

    if hasattr(state, "keys"):
        keys = list(state.keys())
        print(f"\nstate_dict 共 {len(keys)} 个 key", flush=True)
        print("前 30 个 key:", flush=True)
        for k in keys[:30]:
            print(f"  {k}")
        if len(keys) > 30:
            print(f"  ... 及其余 {len(keys) - 30} 个")

        has_lm_head = any("lm_head" in k for k in keys)
        has_backbone = any("backbone" in k for k in keys)
        has_vision = any("vision" in k.lower() or "encoder" in k.lower() or "bridge" in k.lower() for k in keys)
        print("\n判断:", flush=True)
        print(f"  含 lm_head: {has_lm_head}", flush=True)
        print(f"  含 backbone: {has_backbone}", flush=True)
        print(f"  含 vision/encoder/bridge: {has_vision}", flush=True)
        if has_lm_head:
            print("  → 推理时应加载此 checkpoint 中的 lm_head，不要走 Tying。", flush=True)
        elif has_vision and not has_backbone:
            print("  → 仅 Vision+Bridge（设计如此）；推理时 LLM 从 HF 加载，lm_head 需 Tying 到 backbone.embeddings。", flush=True)
        else:
            print("  → 请根据上述 key 判断用途。", flush=True)
    else:
        print("无法解析 state_dict", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
