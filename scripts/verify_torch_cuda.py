"""
验证当前环境 PyTorch 是否为 CUDA 版、能否识别 GPU（如 RTX 5090）。
用法: python scripts/verify_torch_cuda.py
"""
import sys
import torch

def main():
    ok = torch.cuda.is_available()
    cuda_ver = getattr(torch.version, "cuda", None)
    device_name = torch.cuda.get_device_name(0) if ok else "N/A"
    print("torch.cuda.is_available():", ok)
    print("torch.version.cuda:", cuda_ver)
    print("torch.cuda.get_device_name(0):", device_name)
    if not ok:
        print("\n当前为 CPU 版 PyTorch 或未检测到 GPU，请按 docs/STAGE2_AND_PYTORCH.md 重装 CUDA 版。")
        return 1
    if "5090" in device_name or "5090" in device_name.upper():
        print("\nRTX 5090 已识别，可进行 Stage 2 训练。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
