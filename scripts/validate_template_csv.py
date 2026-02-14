from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


TEMPLATE_HEADERS = ["所见：", "结论：", "建议：", "病理倾向："]
NON_THORAX_RE = re.compile(r"胃|肝|脾|胰|肾|子宫|卵巢|前列腺|膀胱|颅脑|甲状腺|乳腺")


def validate_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if "answer" not in df.columns:
        raise ValueError(f"{path} has no answer column")

    n = len(df)
    template_ok = 0
    non_thorax = 0
    short = 0
    bad_rows = []
    for i, row in df.iterrows():
        ans = str(row.get("answer", ""))
        has_template = all(h in ans for h in TEMPLATE_HEADERS)
        has_non_thorax = bool(NON_THORAX_RE.search(ans))
        is_short = len(ans) < 80
        template_ok += int(has_template)
        non_thorax += int(has_non_thorax)
        short += int(is_short)
        if (not has_template) or has_non_thorax or is_short:
            bad_rows.append(
                {
                    "row": int(i),
                    "template_ok": has_template,
                    "non_thorax": has_non_thorax,
                    "is_short": is_short,
                    "head": ans[:120].replace("\n", " "),
                }
            )

    return {
        "csv": str(path),
        "rows": n,
        "template_complete_rate": template_ok / n if n else 0.0,
        "non_thorax_rate": non_thorax / n if n else 0.0,
        "short_rate_lt80chars": short / n if n else 0.0,
        "bad_rows_preview": bad_rows[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate template structure and chest-only tendency in caption CSV")
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--save_json", type=str, default=None)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"CSV not found: {path}")
        return 1

    res = validate_csv(path)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
