"""
Analyze caption homogeneity and length stats.

Usage:
  python scripts/analyze_captions.py --csv d:/mamba/outputs/excel_caption/caption_train_thorax.csv --sample 50
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _read_rows(csv_path: Path):
    for enc in ("utf-8", "gbk"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _char_ngrams(s: str, n: int) -> list[str]:
    s = s.replace("\n", "")
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with answer column")
    ap.add_argument("--sample", type=int, default=50, help="Number of samples to print")
    ap.add_argument("--ngram", type=int, default=3, help="Char n-gram for uniqueness ratio")
    ap.add_argument("--prefix_len", type=int, default=20, help="Prefix length for template similarity")
    args = ap.parse_args()

    rows = _read_rows(Path(args.csv))
    if not rows:
        print("No rows.")
        return 1
    if "answer" not in rows[0]:
        print("CSV has no 'answer' column.")
        return 1

    answers = [str(r.get("answer", "") or "") for r in rows]
    lengths = [len(a) for a in answers]
    avg_len = sum(lengths) / max(1, len(lengths))
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    # Prefix similarity
    prefixes = [a.strip().replace("\n", " ")[: args.prefix_len] for a in answers]
    prefix_counts = Counter(prefixes)
    most_common = prefix_counts.most_common(10)

    # N-gram uniqueness
    all_ngrams = []
    for a in answers:
        all_ngrams.extend(_char_ngrams(a, args.ngram))
    uniq_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    uniq_ratio = (uniq_ngrams / total_ngrams) if total_ngrams else 0.0

    print(f"Total rows: {len(rows)}")
    print(f"Answer length: avg={avg_len:.1f}, min={min_len}, max={max_len}")
    print(f"Char {args.ngram}-gram uniqueness ratio: {uniq_ratio:.4f} ({uniq_ngrams}/{total_ngrams})")
    print("Top prefix patterns:")
    for p, c in most_common:
        print(f"  {c:4d}  {p}")

    print("\nSample answers:")
    for i, a in enumerate(answers[: min(args.sample, len(answers))], 1):
        print(f"[{i}] {a[:400]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
