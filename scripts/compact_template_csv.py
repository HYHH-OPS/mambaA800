from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


HEADERS = ("所见：", "结论：", "建议：", "病理倾向：")


def _clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _split_template_sections(text: str) -> dict[str, str]:
    txt = _clean_text(text)
    out = {h: "" for h in HEADERS}
    if not txt:
        return out

    cur = None
    for line in txt.split("\n"):
        line = line.strip()
        if not line:
            continue
        hit = None
        for h in HEADERS:
            if line.startswith(h):
                out[h] = line[len(h):].strip()
                cur = h
                hit = h
                break
        if hit is None:
            if cur is not None:
                out[cur] = (out[cur] + " " + line).strip()

    # Fallback: if no template markers were found, use the full text as findings/conclusion.
    if not any(out.values()):
        out["所见："] = txt
        out["结论："] = txt
        out["建议："] = "建议结合临床并复查胸部CT。"
        out["病理倾向："] = "炎性或肿瘤性待定。"
    return out


def _truncate_by_sentences(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    parts = re.split(r"(?<=[。；;])", text)
    buf = ""
    for p in parts:
        if not p:
            continue
        cand = (buf + p).strip()
        if len(cand) > max_chars:
            break
        buf = cand
    if buf:
        return buf
    return text[:max_chars].rstrip("，,。；; ") + "。"


def compact_answer(
    answer: str,
    max_findings: int,
    max_conclusion: int,
    max_advice: int,
    max_pathology: int,
) -> str:
    sec = _split_template_sections(answer)
    f = _truncate_by_sentences(sec["所见："], max_findings)
    c = _truncate_by_sentences(sec["结论："], max_conclusion)
    a = _truncate_by_sentences(sec["建议："], max_advice)
    p = _truncate_by_sentences(sec["病理倾向："], max_pathology)
    return f"所见：{f}\n结论：{c}\n建议：{a}\n病理倾向：{p}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact template caption CSV to reduce truncation in training")
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--max_findings", type=int, default=220)
    ap.add_argument("--max_conclusion", type=int, default=140)
    ap.add_argument("--max_advice", type=int, default=80)
    ap.add_argument("--max_pathology", type=int, default=120)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}")
        return 1

    df = pd.read_csv(inp)
    if "answer" not in df.columns:
        print("CSV missing column: answer")
        return 1

    before_len = df["answer"].fillna("").astype(str).str.len()
    df["answer"] = df["answer"].fillna("").astype(str).apply(
        lambda x: compact_answer(
            x,
            max_findings=args.max_findings,
            max_conclusion=args.max_conclusion,
            max_advice=args.max_advice,
            max_pathology=args.max_pathology,
        )
    )
    after_len = df["answer"].astype(str).str.len()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = {
        "input": str(inp),
        "output": str(out),
        "rows": int(len(df)),
        "before_chars_p50": int(before_len.quantile(0.5)),
        "before_chars_p90": int(before_len.quantile(0.9)),
        "after_chars_p50": int(after_len.quantile(0.5)),
        "after_chars_p90": int(after_len.quantile(0.9)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
