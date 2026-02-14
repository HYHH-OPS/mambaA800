"""
Convert caption CSV answers to a fixed clinical template to reduce hallucination.

Template:
  所见：
  结论：
  建议：
  病理倾向：

Usage:
  python scripts/template_caption_csv.py --input d:/mamba/outputs/excel_caption/caption_train_struct.csv --output d:/mamba/outputs/excel_caption/caption_train_template.csv
  python scripts/template_caption_csv.py --input d:/mamba/outputs/excel_caption/caption_val_struct.csv --output d:/mamba/outputs/excel_caption/caption_val_template.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TEMPLATE_QUESTION = (
    "请基于胸部CT仅按以下固定模板输出，禁止输出模板外内容。\n"
    "所见：<影像学所见，含定位/大小/征象>\n"
    "结论：<1-3条结论>\n"
    "建议：<随访或检查建议>\n"
    "病理倾向：<炎性/肿瘤性/待定及理由>"
)


FINDING_KW = (
    "胸廓", "气管", "支气管", "双肺", "肺门", "胸膜", "纵隔", "纹理", "结节", "病灶",
    "磨玻璃", "实性", "空泡", "分叶", "毛刺", "强化", "积液", "淋巴结", "钙化", "条索"
)
REC_KW = ("建议", "随访", "复查", "结合临床", "专科", "进一步", "对比")
PATHO_KW = ("病理", "肿瘤", "肺ca", "癌", "恶性", "转移", "炎性", "待排", "性质待定")


def _strip_num_prefix(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\d一二三四五六七八九十]+[、\.\)：:]\s*", "", s)
    s = re.sub(r"^[⒈⒉⒊⒋⒌⒍⒎⒏⒐]\s*", "", s)
    return s.strip()


def _split_sentences(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"[\n。；;]+", text)
    out: list[str] = []
    for x in raw:
        x = _strip_num_prefix(x)
        if x:
            out.append(x)
    return out


def _join_or_default(items: list[str], default: str, limit: int) -> str:
    if not items:
        return default
    return "；".join(items[:limit]) + "。"


def to_template(answer: str) -> str:
    sents = _split_sentences(answer)
    findings: list[str] = []
    conclusions: list[str] = []
    recs: list[str] = []
    patho: list[str] = []

    for s in sents:
        has_rec = any(k in s for k in REC_KW)
        has_patho = any(k in s.lower() for k in PATHO_KW)
        has_finding = any(k in s for k in FINDING_KW) or bool(re.search(r"\d+\s*mm", s.lower()))

        if has_rec:
            recs.append(s)
        if has_patho:
            patho.append(s)
        if has_finding and not has_rec:
            findings.append(s)
        if (not has_rec) and (("考虑" in s) or ("可能" in s) or ("待定" in s) or ("结节" in s) or ("病灶" in s)):
            conclusions.append(s)

    # Backoff: if nothing classified into conclusions, reuse non-recommendation lines.
    if not conclusions:
        conclusions = [x for x in sents if not any(k in x for k in REC_KW)]
    if not findings:
        findings = conclusions[:]

    findings_txt = _join_or_default(findings, "胸部CT示双肺见异常影，具体定位与大小需结合原片。", limit=4)
    concl_txt = _join_or_default(conclusions, "双肺结节/病灶性质待定。", limit=3)
    rec_txt = _join_or_default(recs, "建议结合临床并短期随访复查胸部CT。", limit=2)
    patho_txt = _join_or_default(patho, "倾向炎性或待定，需结合临床与随访结果判定。", limit=2)

    return (
        f"所见：{findings_txt}\n"
        f"结论：{concl_txt}\n"
        f"建议：{rec_txt}\n"
        f"病理倾向：{patho_txt}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Template-ize caption CSV answers for stable clinical output")
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--keep_question", action="store_true", help="Keep original question instead of fixed template question")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}")
        return 1

    df = pd.read_csv(inp)
    if "answer" not in df.columns:
        print("CSV has no 'answer' column.")
        return 1

    df["answer"] = df["answer"].fillna("").astype(str).apply(to_template)
    if (not args.keep_question) and ("question" in df.columns):
        df["question"] = TEMPLATE_QUESTION

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

