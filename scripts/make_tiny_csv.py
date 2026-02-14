"""
Create a tiny CSV subset for overfit/debug.

Usage:
  python scripts/make_tiny_csv.py --src d:/mamba/outputs/excel_caption/caption_train_mask.csv --dst d:/mamba/outputs/excel_caption/caption_tiny.csv --n 8
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Create a tiny CSV subset")
    ap.add_argument("--src", required=True, help="Source CSV")
    ap.add_argument("--dst", required=True, help="Destination CSV")
    ap.add_argument("--n", type=int, default=8, help="Number of samples")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"Source not found: {src}")
        return 1

    # Read with utf-8 or fallback to gbk.
    rows = []
    for enc in ("utf-8", "gbk"):
        try:
            with src.open("r", encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            continue
    if not rows:
        with src.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sample[0].keys())
        w.writeheader()
        w.writerows(sample)
    print(f"Wrote {len(sample)} rows to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
