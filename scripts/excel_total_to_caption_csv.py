r"""
把「2021-2024肺结节数据整理总表.xlsx」的“总表”转换为训练用 caption CSV：

输出 CSV 列：
- image_path: 由 image_root + 序号拼出来（例如 D:/CT_NIFTI/1.nii.gz）
- question: 固定为“请按模板生成病例报告...”
- answer: 结构化病例报告（报告所见、报告结论、病理诊断等），不含性别/年龄；文本已规范化（合并多余换行与空格）

用法（PowerShell）：
  # 方式一：用 pairs_full.csv 合并得到真实 CT 路径（推荐）
  python scripts/excel_total_to_caption_csv.py ^
    --excel "C:\Users\ASUS\Desktop\2021-2024肺结节数据整理总表.xlsx" ^
    --out_dir "D:\mamba\outputs\excel_caption" ^
    --pairs_csv "D:\unn-net\pairs_full.csv"

  # 方式二：用 image_root + 序号 拼路径（目录中需有 1.nii.gz, 2.nii.gz, ...）
  python scripts/excel_total_to_caption_csv.py ^
    --excel "..." --out_dir "..." --image_root "D:\CT_NIFTI_NUM"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _yn(v) -> str:
    if pd.isna(v):
        return "未知"
    try:
        iv = int(v)
        if iv == 0:
            return "无"
        if iv == 1:
            return "有"
    except Exception:
        pass
    return str(v)


def _pick_col(cols: list[str], startswith: str) -> str | None:
    for c in cols:
        if str(c).startswith(startswith):
            return c
    return None


def _normalize_report_text(s: str) -> str:
    """对报告文本规范化：去首尾空白、合并连续换行/空格、全角标点数字转半角（可选）。"""
    import re
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    # 连续换行或空格合并为单个换行/空格
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    # 每行去首尾空格
    lines = [line.strip() for line in s.split("\n") if line.strip()]
    return "\n".join(lines)


def row_to_report(r: pd.Series, cols: list[str]) -> str:
    # 仅使用报告相关列，不包含性别、年龄
    conclusion = r.get("报告结论", "")
    findings = r.get("报告所见", "")
    pathology = r.get("病理诊断", "")

    parts = []
    if isinstance(findings, str) and findings.strip():
        parts.append(str(findings).strip())
    if isinstance(conclusion, str) and conclusion.strip():
        parts.append(str(conclusion).strip())
    if isinstance(pathology, str) and pathology.strip():
        parts.append("病理诊断：" + str(pathology).strip())

    raw = "\n".join(parts).strip()
    return _normalize_report_text(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, required=True, help="总表 Excel 路径")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录")
    ap.add_argument("--image_root", type=str, default=None, help="CT NIfTI 根目录（包含 1.nii.gz/2.nii.gz/...）")
    ap.add_argument("--pairs_csv", type=str, default=None, help="pairs_full.csv 等，含 serial 与 ct_nii，按序号合并得到 image_path")
    ap.add_argument("--pairs_id_col", type=str, default="serial", help="pairs 中与总表「序号」对应的列名")
    ap.add_argument("--pairs_path_col", type=str, default="ct_nii", help="pairs 中 CT NIfTI 路径列名")
    ap.add_argument("--sheet_index", type=int, default=1, help="使用第几个 sheet（默认 1，通常是“总表”）")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--answer_prefix", type=str, default=None, help="在每条 answer 开头加固定引导词，如 报告结论： 便于模型学习格式")
    ap.add_argument("--min_answer_chars", type=int, default=0, help="过滤掉 answer 字数少于此的样本，减少「无异常」类捷径；0 表示不过滤")
    ap.add_argument("--require_keywords", type=str, default=None, help="只保留 answer 中含指定关键词的样本（正则），如 结节|mm|肺叶|段|病灶；留空则不过滤")
    ap.add_argument("--force_numbered_prefix", action="store_true", help="每条 answer 开头加 1、 强制分条格式，便于模型学习病灶定位/尺寸等结构化输出")
    args = ap.parse_args()

    if not args.image_root and not args.pairs_csv:
        print("请至少提供 --image_root 或 --pairs_csv，以得到 image_path。")
        return 1

    xls = pd.ExcelFile(args.excel)
    sheet = xls.sheet_names[args.sheet_index]
    df = pd.read_excel(args.excel, sheet_name=sheet)
    cols = list(df.columns)

    # 序号列
    if "序号" in df.columns:
        seq = df["序号"].astype(int)
    else:
        seq = df.iloc[:, 0].astype(int)

    from data.medical_vlm_dataset import CAPTION_DEFAULT_QUESTION_NO_NL
    question = CAPTION_DEFAULT_QUESTION_NO_NL

    out = pd.DataFrame()
    if args.pairs_csv and Path(args.pairs_csv).exists():
        pairs = pd.read_csv(args.pairs_csv, encoding="utf-8-sig")
        pairs_id = pairs[args.pairs_id_col].astype(int)
        id_to_path = dict(zip(pairs_id.tolist(), pairs[args.pairs_path_col].astype(str).tolist()))
        paths = []
        missing = []
        for i in seq.tolist():
            if i in id_to_path:
                paths.append(id_to_path[i])
            else:
                if args.image_root:
                    paths.append(str(Path(args.image_root) / f"{i}.nii.gz"))
                else:
                    paths.append("")
                    missing.append(i)
        out["image_path"] = paths
        if missing:
            print(f"警告: 以下序号在 pairs 中未找到，image_path 为空或用 image_root 补齐: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    elif args.image_root:
        out["image_path"] = [str(Path(args.image_root) / f"{i}.nii.gz") for i in seq.tolist()]
    else:
        out["image_id"] = seq.tolist()

    # 若存在 image_path 列，过滤掉没有有效路径的行，避免训练时用随机噪声图像干扰图像→文本对齐
    if "image_path" in out.columns:
        before = len(out)
        # 路径非空且不是纯空白
        mask_non_empty = out["image_path"].astype(str).str.strip() != ""
        out = out[mask_non_empty].reset_index(drop=True)
        print(f"过滤无有效 image_path 的行: {before} -> {len(out)}")
    out["question"] = question
    prefix = (args.answer_prefix or "").strip()
    if prefix and not prefix.endswith("\n"):
        prefix = prefix + "\n"
    numbered = "1、" if getattr(args, "force_numbered_prefix", False) else ""
    out["answer"] = [numbered + prefix + row_to_report(df.iloc[i], cols) for i in range(len(out))]

    # 优先保留病理特征丰富的样本，减少「无异常」等短报告导致的捷径学习
    if args.min_answer_chars > 0 or (args.require_keywords and args.require_keywords.strip()):
        before = len(out)
        mask = pd.Series([True] * len(out))
        if args.min_answer_chars > 0:
            mask = mask & (out["answer"].astype(str).str.len() >= args.min_answer_chars)
        if args.require_keywords and args.require_keywords.strip():
            import re
            try:
                mask = mask & out["answer"].astype(str).str.contains(args.require_keywords.strip(), regex=True)
            except re.error:
                mask = mask & out["answer"].astype(str).str.contains(re.escape(args.require_keywords.strip()))
        out = out[mask].reset_index(drop=True)
        print(f"保留病理特征丰富样本（min_answer_chars>={args.min_answer_chars or '-'}，keywords={args.require_keywords or '-'}）: {before} -> {len(out)} 行")
    if len(out) == 0:
        print("警告: 过滤后无样本，请放宽 --min_answer_chars 或 --require_keywords")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "caption_full.csv"
    out.to_csv(full_csv, index=False, encoding="utf-8-sig")

    # 简单随机切分 train/val
    out_shuf = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = max(1, int(len(out_shuf) * args.val_ratio))
    val = out_shuf.iloc[:n_val]
    train = out_shuf.iloc[n_val:]
    train.to_csv(out_dir / "caption_train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(out_dir / "caption_val.csv", index=False, encoding="utf-8-sig")

    print("sheet:", sheet)
    print("rows:", len(out))
    print("wrote:", str(full_csv))
    print("wrote:", str(out_dir / "caption_train.csv"))
    print("wrote:", str(out_dir / "caption_val.csv"))
    if not args.image_root:
        print("注意: 未提供 --image_root，输出为 image_id；训练时需在 Dataset 传 image_root 来拼 image_path。")


if __name__ == "__main__":
    main()

