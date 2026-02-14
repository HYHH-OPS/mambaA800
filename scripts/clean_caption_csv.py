"""
清洗 caption CSV：剔除 answer 中含电话/广告等垃圾文本的样本，避免训练时学到幻觉模板。
同时会去掉 answer 中的性别、年龄表述，并对剩余文本做规范化（合并多余换行/空格）。
训练前建议先运行本脚本，并将 config/paths.yaml 中 caption_csv_train 指向 *_clean.csv。

用法:
  python scripts/clean_caption_csv.py
  python scripts/clean_caption_csv.py --input d:/mamba/outputs/excel_caption/caption_full.csv --output d:/mamba/outputs/excel_caption/caption_full_clean.csv
  python scripts/clean_caption_csv.py --drop_missing_paths   # 同时剔除 image_path 不存在的行，防止图文错位导致幻觉
  python scripts/clean_caption_csv.py --min_answer_chars 50 --require_keywords "结节|mm|肺叶"   # 只保留病理特征丰富的样本
  python scripts/clean_caption_csv.py --drop_generic_only   # 剔除仅含「两肺无明显」等空洞短样本，强化结构化输出
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _strip_sex_age_and_normalize(answer: str) -> str:
    """去掉 answer 中的性别、年龄表述（整行或句首），并对剩余文本规范化。"""
    if not answer or not isinstance(answer, str):
        return ""
    s = answer.strip()
    # 按行处理：去掉仅包含「性别：…」「年龄：…岁」或「性别：…，年龄：…岁」的行
    sex_age_line = re.compile(
        r"^(\s*性别\s*[：:]\s*[^，\n]+\s*([，,]\s*年龄\s*[：:]\s*[^\n]*岁)?\s*$)|"
        r"^(\s*年龄\s*[：:]\s*[^\n]*岁\s*$)"
    )
    lines = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        if sex_age_line.match(line):
            continue
        # 去掉行首的「性别：…，」或「年龄：…岁，」片段（保留该行其余内容）
        line = re.sub(r"^性别\s*[：:]\s*[^，\n]+[，,]\s*", "", line)
        line = re.sub(r"^年龄\s*[：:]\s*[^\n]*岁\s*[，,]\s*", "", line)
        if line.strip():
            lines.append(line.strip())
    # 合并连续空格
    s = "\n".join(lines)
    s = re.sub(r"[ \t]+", " ", s)
    return "\n".join(ln.strip() for ln in s.split("\n") if ln.strip())


def _filter_lines(answer: str, keep_re: re.Pattern | None, drop_re: re.Pattern | None) -> str:
    if not answer or not isinstance(answer, str):
        return ""
    lines = []
    for line in answer.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        if drop_re and drop_re.search(ln):
            continue
        if keep_re and not keep_re.search(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Clean caption CSV: remove rows with ad/phone in answer")
    ap.add_argument("--input", type=str, default=None, help="Input CSV path; default from config caption_csv_train")
    ap.add_argument("--output", type=str, default=None, help="Output CSV path; default input_clean.csv")
    ap.add_argument("--dry_run", action="store_true", help="Only print count of bad rows, do not write")
    ap.add_argument("--drop_missing_paths", action="store_true", help="Drop rows where image_path file does not exist (avoid image-text mismatch)")
    ap.add_argument("--min_answer_chars", type=int, default=0, help="只保留 answer 字数>=此值的行，0 表示不过滤")
    ap.add_argument("--require_keywords", type=str, default=None, help="只保留 answer 含关键词（正则）的行，如 结节|mm|肺叶|段|病灶")
    ap.add_argument("--drop_generic_only", action="store_true", help="剔除仅含「两肺无明显」等空洞描述且无尺寸/定位的短样本，强化结构化输出")
    ap.add_argument("--force_numbered_prefix", action="store_true", help="每条 answer 开头加 1、 强制分条格式，便于模型学习结构化输出")
    ap.add_argument("--keep_line_regex", type=str, default=None, help="仅保留包含该正则的行（逐行过滤）")
    ap.add_argument("--drop_line_regex", type=str, default=None, help="删除包含该正则的行（逐行过滤）")
    ap.add_argument("--thorax_only", action="store_true", help="只保留胸部/肺部相关行，删除腹部/盆腔/头颅等系统表述")
    args = ap.parse_args()

    if args.input and Path(args.input).exists():
        inp = Path(args.input)
    else:
        from data.medical_vlm_dataset import load_paths_config
        config = load_paths_config(REPO / "config" / "paths.yaml")
        train_csv = config.get("caption_csv_train")
        if not train_csv or not Path(train_csv).exists():
            print("No input CSV. Use --input or set caption_csv_train in config/paths.yaml")
            return 1
        inp = Path(train_csv)

    out = Path(args.output) if args.output else inp.parent / (inp.stem + "_clean" + inp.suffix)

    try:
        import pandas as pd
    except ImportError:
        print("Need pandas: pip install pandas")
        return 1

    df = pd.read_csv(inp)
    if "answer" not in df.columns:
        print("CSV has no 'answer' column")
        return 1

    # 广告/电话/垃圾模板：避免训练学到幻觉
    bad_re = re.compile(
        r"请拨打电话|联系客服|客服电话|微信联系|联系电话|电话\s*[:：]\s*\d+|"
        r"\+\s*\d{2}\s*[-]?\s*\d+|\d{11}|QQ\s*\d+|手机\s*[:：]?\s*\d+|"
        r"悲剧情况|敬请关注|如需更多|扫码关注|添加微信|微信号"
    )
    mask_bad = df["answer"].fillna("").astype(str).str.contains(bad_re)
    n_bad = mask_bad.sum()
    df_clean = df[~mask_bad].copy()

    # 去掉 answer 中的性别、年龄表述，并对文本规范化（首尾空白、连续换行/空格）
    df_clean["answer"] = df_clean["answer"].fillna("").astype(str).apply(_strip_sex_age_and_normalize)
    print("已去除 answer 中性别/年龄表述并对文本规范化")

    # 逐行过滤（可选）
    keep_re = re.compile(args.keep_line_regex) if args.keep_line_regex else None
    drop_re = re.compile(args.drop_line_regex) if args.drop_line_regex else None
    if args.thorax_only:
        # Keep thorax/lung related lines, drop abdomen/pelvis/head/ENT/etc.
        keep_re = re.compile(r"肺|胸|肺叶|肺段|气管|支气管|肺门|胸膜|纵隔|肺纹理|呼吸", re.IGNORECASE)
        drop_re = re.compile(r"肝|肾|脾|胰|腹|盆腔|子宫|卵巢|前列腺|膀胱|头颅|颅脑|鼻窦|咽喉|甲状腺|乳腺|PET/CT|PET-CT", re.IGNORECASE)
    if keep_re or drop_re:
        before = len(df_clean)
        df_clean["answer"] = df_clean["answer"].fillna("").astype(str).apply(lambda s: _filter_lines(s, keep_re, drop_re))
        df_clean = df_clean[df_clean["answer"].str.strip().astype(bool)].reset_index(drop=True)
        print(f"按行过滤后: {before} -> {len(df_clean)}")

    # 可选：剔除 image_path 不存在的行，防止图像与报告错位导致幻觉
    if args.drop_missing_paths and "image_path" in df_clean.columns:
        def exists(p):
            try:
                return Path(str(p).strip()).exists()
            except Exception:
                return False
        mask_exists = df_clean["image_path"].astype(str).apply(exists)
        n_missing = (~mask_exists).sum()
        df_clean = df_clean[mask_exists].reset_index(drop=True)
        print(f"Removed {n_missing} rows with missing image_path file")
    else:
        n_missing = 0

    # 可选：剔除「空洞」样本（仅有无明显变化等短句、且无尺寸/结节等）
    if getattr(args, "drop_generic_only", False):
        generic_re = re.compile(r"两肺无明显|未见明显异常|无明显变化|两肺对称.*未见", re.IGNORECASE)
        has_rich = re.compile(r"\d+\s*mm|结节|肺叶|肺段|病灶|病理")
        def is_generic_only(a):
            a = (a or "").strip()
            if len(a) >= 60:
                return False
            if has_rich.search(a):
                return False
            return bool(generic_re.search(a))
        mask_keep = ~df_clean["answer"].fillna("").astype(str).apply(is_generic_only)
        before = len(df_clean)
        df_clean = df_clean[mask_keep].reset_index(drop=True)
        print(f"drop_generic_only: 剔除空洞短样本后 {before} -> {len(df_clean)} 行")
    if len(df_clean) == 0:
        print("警告: 过滤后无样本，请放宽条件")
        return 1

    # 可选：只保留病理特征丰富的样本，减少「无异常」类捷径
    if args.min_answer_chars > 0 or (args.require_keywords and args.require_keywords.strip()):
        before_rich = len(df_clean)
        mask = df_clean["answer"].fillna("").astype(str).str.len() >= args.min_answer_chars
        if args.require_keywords and args.require_keywords.strip():
            try:
                mask = mask & df_clean["answer"].fillna("").astype(str).str.contains(args.require_keywords.strip(), regex=True)
            except re.error:
                mask = mask & df_clean["answer"].fillna("").astype(str).str.contains(re.escape(args.require_keywords.strip()))
        df_clean = df_clean[mask].reset_index(drop=True)
        print(f"保留病理丰富样本（min_chars>={args.min_answer_chars or '-'}, keywords={args.require_keywords or '-'}）: {before_rich} -> {len(df_clean)}")
    if len(df_clean) == 0:
        print("警告: 过滤后无样本，请放宽条件")
        return 1

    print(f"Input: {inp}, rows: {len(df)}")
    print(f"Removed {n_bad} rows with ad/phone-like text")
    if n_missing:
        print(f"Remaining after drop_missing_paths: {len(df_clean)} rows")
    else:
        print(f"Remaining: {len(df_clean)} rows")

    if getattr(args, "force_numbered_prefix", False):
        def add_prefix(a):
            a = (a or "").strip()
            if a.startswith("1、") or a.startswith("1."):
                return a
            return "1、" + a
        df_clean["answer"] = df_clean["answer"].fillna("").astype(str).apply(add_prefix)
        print("已为 answer 添加 1、 前缀（已有则保留）")

    # 检查训练文本是否过短（仅 3–5 个词会导致生成过短/内容稀薄）
    answers = df_clean["answer"].fillna("").astype(str)
    word_counts = answers.str.split().str.len()
    mean_words = float(word_counts.mean()) if len(word_counts) else 0
    n_very_short = int((word_counts <= 5).sum())
    pct_short = (100.0 * n_very_short / len(df_clean)) if len(df_clean) else 0
    print(f"Answer 长度: 平均 {mean_words:.1f} 词/条, {n_very_short} 条 ≤5 词 ({pct_short:.1f}%)")
    if pct_short > 20 or mean_words < 15:
        print("警告: 大量答案为 3–5 词级别，训练易导致生成过短、内容稀薄，建议用 --min_answer_chars/--require_keywords 保留更丰富样本", flush=True)

    if args.dry_run:
        return 0
    df_clean.to_csv(out, index=False)
    print(f"Saved: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
