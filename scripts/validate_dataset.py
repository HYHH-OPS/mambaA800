"""
Quick dataset sanity checks:
- image_path / mask_path existence
- image<->mask case-id match
- basic text encoding/garbage heuristic
- optional mask non-empty check (requires SimpleITK)

Usage:
  python scripts/validate_dataset.py --csv d:/mamba/outputs/excel_caption/caption_train_mask.csv
  python scripts/validate_dataset.py --csv ... --samples 20 --check_mask
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple


SUSPECT_CHARS = set("�锟锛锝锘锕锢锣锤锧锫锬锭锯锱锲锳锴锵锶锷锸锹锺锻锼锽锾锿")


def _suspect_score(text: str) -> int:
    if not text:
        return 0
    return sum(1 for ch in text if ch in SUSPECT_CHARS)


def _case_id_from_image(image_path: str) -> str:
    try:
        return Path(image_path).parent.name
    except Exception:
        return ""


def _case_id_from_mask(mask_path: str) -> str:
    try:
        p = Path(mask_path)
        name = p.name
        # Handle .nii.gz explicitly
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        else:
            stem = p.stem
        if stem.endswith("_0000"):
            stem = stem[:-5]
        return stem
    except Exception:
        return ""


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    # Try utf-8 first, then gbk as fallback.
    for enc in ("utf-8", "gbk"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    # Last resort: replace errors
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _check_mask_nonempty(mask_path: Path) -> Tuple[bool, str]:
    try:
        import SimpleITK as sitk
        import numpy as np
    except Exception:
        return False, "SimpleITK not available"
    try:
        img = sitk.ReadImage(str(mask_path))
        arr = sitk.GetArrayFromImage(img)
        nz = int(np.count_nonzero(arr))
        return nz > 0, f"nonzero={nz}"
    except Exception as e:
        return False, f"read_error={e}"


def main():
    ap = argparse.ArgumentParser(description="Quick dataset sanity checks")
    ap.add_argument("--csv", type=str, required=True, help="CSV with image_path, question, answer, mask_path")
    ap.add_argument("--samples", type=int, default=20, help="How many rows to sample for checks")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--check_mask", action="store_true", help="Check if masks are non-empty (SimpleITK required)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    rows = _read_rows(csv_path)
    if not rows:
        print("CSV is empty or unreadable.")
        return 1

    random.seed(args.seed)
    sample_rows = random.sample(rows, min(args.samples, len(rows)))

    missing_images = 0
    missing_masks = 0
    id_mismatch = 0
    suspect_text = 0
    mask_empty = 0

    print(f"Total rows: {len(rows)}")
    print(f"Sampled rows: {len(sample_rows)}")

    for i, r in enumerate(sample_rows, 1):
        img_path = r.get("image_path", "")
        mask_path = r.get("mask_path", "")
        q = r.get("question", "")
        a = r.get("answer", "")

        img_exists = Path(img_path).exists()
        mask_exists = Path(mask_path).exists() if mask_path else False
        if not img_exists:
            missing_images += 1
        if not mask_exists:
            missing_masks += 1

        img_id = _case_id_from_image(img_path)
        mask_id = _case_id_from_mask(mask_path)
        if img_id and mask_id and img_id != mask_id:
            id_mismatch += 1

        s_score = _suspect_score(q) + _suspect_score(a)
        if s_score > 0:
            suspect_text += 1

        msg = f"[{i}] img_exists={img_exists} mask_exists={mask_exists} id_match={img_id==mask_id} img_id={img_id} mask_id={mask_id}"
        if s_score > 0:
            msg += f" suspect_chars={s_score}"
        if args.check_mask and mask_exists:
            ok, info = _check_mask_nonempty(Path(mask_path))
            if not ok:
                mask_empty += 1
            msg += f" mask_check={info}"
        print(msg)

    print("\nSummary:")
    print(f"  missing_images: {missing_images}")
    print(f"  missing_masks: {missing_masks}")
    print(f"  id_mismatch: {id_mismatch}")
    print(f"  suspect_text_rows: {suspect_text}")
    if args.check_mask:
        print(f"  mask_empty_or_error: {mask_empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
