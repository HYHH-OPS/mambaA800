"""
Field extraction quality report.

Usage:
  python scripts/field_quality.py --csv d:/mamba/outputs/excel_caption/caption_train_struct.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import Counter


FIELDS = ["位置", "大小", "类型", "征象", "数量", "影像序号"]


def _read_rows(csv_path: Path):
    for enc in ("utf-8", "gbk"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with answer column (structured)")
    args = ap.parse_args()

    rows = _read_rows(Path(args.csv))
    if not rows:
        print("No rows.")
        return 1
    if "answer" not in rows[0]:
        print("CSV has no 'answer' column.")
        return 1

    counts = Counter()
    for r in rows:
        a = str(r.get("answer", "") or "")
        for f in FIELDS:
            if f in a:
                counts[f] += 1

    total = len(rows)
    print(f"Total rows: {total}")
    for f in FIELDS:
        c = counts.get(f, 0)
        pct = (100.0 * c / total) if total else 0.0
        print(f"{f}: {c} ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
