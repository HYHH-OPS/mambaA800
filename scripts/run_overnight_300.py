"""
夜间 300 轮训练 + 超参筛选：依次跑多组 (lr, batch_size)，每组 300 epochs，最后选出 val_loss 最好的参数并保存摘要。

用法（睡觉前执行）:
  cd D:\mamba
  conda activate mamba5090
  python scripts/run_overnight_300.py

可选:
  python scripts/run_overnight_300.py --epochs 100          # 每组只跑 100 轮（快速试）
  python scripts/run_overnight_300.py --no_grid              # 不搜参，只跑 300 轮单组默认参数
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# 要筛选的参数网格：每组跑 --epochs 轮，选 val_loss 最小的
LR_GRID = [1e-4, 5e-5]
BATCH_SIZE_GRID = [2, 4]
DEFAULT_EPOCHS = 300


def run_one(lr: float, batch_size: int, epochs: int) -> tuple[float | None, str]:
    """跑一次 train.py，返回 (最佳 val_loss, 该次 log 路径)。本组跑完后把 best_val 权重另存一份避免被下一组覆盖。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_name = f"overnight_lr{lr}_bs{batch_size}_{stamp}.log"
    log_path = OUT / log_name
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--output_dir", str(OUT),
    ]
    best_val = None
    val_loss_pattern = re.compile(r"val_loss\s+([\d.]+)")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# lr={lr} batch_size={batch_size} epochs={epochs}\n")
        f.write(f"# cmd: {' '.join(cmd)}\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
            m = val_loss_pattern.search(line)
            if m:
                v = float(m.group(1))
                if best_val is None or v < best_val:
                    best_val = v
        p.wait()
    # 本组跑完后把当前 best 权重另存，避免下一组覆盖
    src = OUT / "vision_bridge_best_val.pt"
    if src.exists():
        dst = OUT / f"vision_bridge_best_lr{lr}_bs{batch_size}.pt"
        import shutil
        shutil.copy2(src, dst)
    return best_val, str(log_path)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="夜间 300 轮训练 + 超参筛选")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"每组训练轮数，默认 {DEFAULT_EPOCHS}")
    ap.add_argument("--no_grid", action="store_true", help="不搜参，只跑一组默认 lr=1e-4, batch_size=4")
    args = ap.parse_args()

    if args.no_grid:
        runs = [(1e-4, 4)]
    else:
        runs = [(lr, bs) for lr in LR_GRID for bs in BATCH_SIZE_GRID]

    print(f"共 {len(runs)} 组参数，每组 {args.epochs} 轮。开始时间: {datetime.now().isoformat()}", flush=True)
    results = []
    for i, (lr, bs) in enumerate(runs):
        print(f"\n========== 第 {i+1}/{len(runs)} 组: lr={lr}, batch_size={bs} ==========", flush=True)
        best_val, log_path = run_one(lr, bs, args.epochs)
        results.append({"lr": lr, "batch_size": bs, "best_val_loss": best_val, "log": log_path})

    # 按 best_val_loss 排序，越小越好
    results.sort(key=lambda x: (x["best_val_loss"] or float("inf")))

    # 把“最佳参数”对应的 checkpoint 复制为 overnight 最佳
    best = results[0]
    best_ckpt = OUT / f"vision_bridge_best_lr{best['lr']}_bs{best['batch_size']}.pt"
    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, OUT / "vision_bridge_overnight_best.pt")

    summary_path = OUT / "overnight_best_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Overnight run finished: {datetime.now().isoformat()}\n")
        f.write(f"Epochs per run: {args.epochs}\n\n")
        f.write("All runs (best first):\n")
        for r in results:
            f.write(f"  lr={r['lr']}, batch_size={r['batch_size']} -> best val_loss={r['best_val_loss']}  log={r['log']}\n")
        f.write("\nBest params:\n")
        b = results[0]
        f.write(f"  lr={b['lr']}, batch_size={b['batch_size']}, best_val_loss={b['best_val_loss']}\n")
        f.write("\nCheckpoints: 每组最佳已存为 outputs/vision_bridge_best_lr{X}_bs{Y}.pt；综合最佳已复制为 outputs/vision_bridge_overnight_best.pt\n")
    print(f"\n========== 全部完成 ==========", flush=True)
    print(f"最佳参数: lr={results[0]['lr']}, batch_size={results[0]['batch_size']}, val_loss={results[0]['best_val_loss']}", flush=True)
    print(f"摘要已写: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
