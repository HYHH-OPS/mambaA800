from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


TEMPLATE_KEYS = ["所见：", "结论：", "建议：", "病理倾向："]
NON_THORAX_RE = re.compile(r"胃|肝|脾|胰|肾|子宫|卵巢|前列腺|膀胱|颅脑|甲状腺|乳腺")
BAD_CHAR_RE = re.compile(r"\ufffd|锟|�")


def evaluate_run(run_dir: Path) -> dict:
    gen_files = sorted(run_dir.glob("sample_*_gen.txt"))
    if not gen_files:
        raise FileNotFoundError(f"No sample_*_gen.txt found in {run_dir}")

    lengths: list[int] = []
    template_ok = 0
    non_thorax = 0
    empty = 0
    bad_char = 0
    details = []

    for fp in gen_files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        l = len(text)
        lengths.append(l)
        has_template = all(k in text for k in TEMPLATE_KEYS)
        has_non_thorax = bool(NON_THORAX_RE.search(text))
        is_empty = (l == 0)
        has_bad = bool(BAD_CHAR_RE.search(text))

        template_ok += int(has_template)
        non_thorax += int(has_non_thorax)
        empty += int(is_empty)
        bad_char += int(has_bad)

        details.append(
            {
                "file": fp.name,
                "length": l,
                "template_ok": has_template,
                "non_thorax": has_non_thorax,
                "empty": is_empty,
                "bad_char": has_bad,
                "head": text[:120].replace("\n", " "),
            }
        )

    n = len(gen_files)
    return {
        "run_dir": str(run_dir),
        "num_samples": n,
        "avg_length": mean(lengths),
        "median_length": median(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "template_complete_rate": template_ok / n,
        "non_thorax_rate": non_thorax / n,
        "empty_rate": empty / n,
        "bad_char_rate": bad_char / n,
        "threshold_check": {
            "avg_length_gt_160": mean(lengths) > 160,
            "template_complete_gt_0_80": (template_ok / n) > 0.80,
            "non_thorax_lt_0_10": (non_thorax / n) < 0.10,
            "bad_char_lt_0_05": (bad_char / n) < 0.05,
        },
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate generation run quality metrics")
    ap.add_argument("--run_dir", required=True, type=str, help="run_YYYYMMDD_HHMMSS or run_strict_YYYYMMDD_HHMMSS directory")
    ap.add_argument("--save_json", type=str, default=None, help="Optional output json path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        return 1

    result = evaluate_run(run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
