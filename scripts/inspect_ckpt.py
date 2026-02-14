"""
在终端查看 .pt 里有什么（key、step、部分权重 shape）。
用法: python scripts/inspect_ckpt.py
      或: python scripts/inspect_ckpt.py D:\mamba\outputs\vision_bridge_final.pt
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "outputs" / "vision_bridge_final.pt")
    if not Path(path).exists():
        print(f"文件不存在: {path}")
        return 1
    print(f"加载: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    print("顶层 Keys:", list(ckpt.keys()))
    if "step" in ckpt:
        print("step:", ckpt["step"])
    if "val_loss" in ckpt:
        print("val_loss:", ckpt["val_loss"])
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print(f"model_state_dict 共 {len(state)} 个张量:")
        for i, (k, v) in enumerate(list(state.items())[:10]):
            print(f"  {k}: {tuple(v.shape)}")
        if len(state) > 10:
            print(f"  ... 还有 {len(state) - 10} 个")
    return 0

if __name__ == "__main__":
    sys.exit(main())
