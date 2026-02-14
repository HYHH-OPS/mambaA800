# -*- coding: utf-8 -*-
"""
Extract structured nodule fields from answer text and rewrite answer.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RE_SIZE = re.compile(r"(\\d+(?:\\.\\d+)?)\\s*(mm|cm)", re.IGNORECASE)
RE_TYPE = re.compile(r"(\\u5b9e\\u6027|\\u78e8\\u73bb\\u7483|\\u6df7\\u5408|\\u4e9a\\u5b9e\\u6027|\\u90e8\\u5206\\u5b9e\\u6027)")
RE_LOC = re.compile(r"(\\u5de6|\\u53f3|\\u53cc)?\\s*(\\u4e0a\\u53f6|\\u4e2d\\u53f6|\\u4e0b\\u53f6|\\u820c\\u53f6|\\u80ba\\u95e8|\\u80ba\\u5c16|\\u80ba\\u5e95|\\u524d\\u57fa\\u5e95\\u6bb5|\\u540e\\u57fa\\u5e95\\u6bb5|\\u80cc\\u6bb5|\\u524d\\u6bb5|\\u540e\\u6bb5|\\u5916\\u4fa7\\u6bb5|\\u5185\\u4fa7\\u6bb5|\\u4e0a\\u6bb5)")
RE_SIGNS = re.compile(r"(\\u6bdb\\u523a|\\u5206\\u53f6|\\u80f8\\u819c\\u7275\\u62c9|\\u7a7a\\u6ce1|\\u8840\\u7ba1\\u96c6\\u675f|\\u652f\\u6c14\\u7ba1\\u5145\\u6c14|\\u9499\\u5316|\\u7a7a\\u6d1e|\\u574f\\u6b7b|\\u536b\\u661f\\u7076)")
RE_COUNT = re.compile(r"(\\u5355\\u53d1|\\u591a\\u53d1|\\u591a\\u4e2a|\\u6570\\u4e2a)")
RE_IM = re.compile(r"\\bIM\\d+\\b", re.IGNORECASE)

FIELD_IM = "\\u5f71\\u50cf\\u5e8f\\u53f7"
FIELD_LOC = "\\u4f4d\\u7f6e"
FIELD_SIZE = "\\u5927\\u5c0f"
FIELD_TYPE = "\\u7c7b\\u578b"
FIELD_COUNT = "\\u6570\\u91cf"
FIELD_SIGNS = "\\u5f81\\u8c61"


def extract_fields(text: str) -> dict[str, str]:
    t = text or ""
    sizes = ["".join(m) for m in RE_SIZE.findall(t)]
    types = list(dict.fromkeys(RE_TYPE.findall(t)))
    locs = ["".join(filter(None, m)) for m in RE_LOC.findall(t)]
    locs = list(dict.fromkeys([l for l in locs if l.strip()]))
    signs = list(dict.fromkeys(RE_SIGNS.findall(t)))
    counts = RE_COUNT.findall(t)
    count = counts[0] if counts else ""
    ims = list(dict.fromkeys(RE_IM.findall(t)))

    return {
        FIELD_LOC: "\\u3001".join(locs),
        FIELD_SIZE: "\\u3001".join(sizes),
        FIELD_TYPE: "\\u3001".join(types),
        FIELD_SIGNS: "\\u3001".join(signs),
        FIELD_COUNT: count,
        FIELD_IM: "\\u3001".join(ims),
    }


def build_answer(fields: dict[str, str], fallback: str) -> str:
    lines = []
    if fields.get(FIELD_IM):
        lines.append(FIELD_IM + ": " + fields[FIELD_IM])
    if fields.get(FIELD_LOC):
        lines.append(FIELD_LOC + ": " + fields[FIELD_LOC])
    if fields.get(FIELD_SIZE):
        lines.append(FIELD_SIZE + ": " + fields[FIELD_SIZE])
    if fields.get(FIELD_TYPE):
        lines.append(FIELD_TYPE + ": " + fields[FIELD_TYPE])
    if fields.get(FIELD_COUNT):
        lines.append(FIELD_COUNT + ": " + fields[FIELD_COUNT])
    if fields.get(FIELD_SIGNS):
        lines.append(FIELD_SIGNS + ": " + fields[FIELD_SIGNS])
    if not lines:
        return fallback.strip()
    return "\\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV with answer")
    ap.add_argument("--output", required=True, help="Output CSV")
    ap.add_argument("--drop_empty", action="store_true", help="Drop rows with no extracted fields")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}")
        return 1

    df = pd.read_csv(inp)
    if "answer" not in df.columns:
        print("CSV has no 'answer' column")
        return 1

    new_answers = []
    keep_mask = []
    for a in df["answer"].fillna("").astype(str).tolist():
        fields = extract_fields(a)
        out = build_answer(fields, a)
        new_answers.append(out)
        keep_mask.append(bool(out.strip()))

    if args.drop_empty:
        df = df[keep_mask].reset_index(drop=True)
        new_answers = [a for a in new_answers if a.strip()]

    df["answer"] = new_answers
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
