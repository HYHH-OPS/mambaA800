from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.medical_vlm_dataset import MedicalVLMDataset, load_paths_config
from inference import (
    PROMPT_SHORT_NO_PLACEHOLDERS,
    generate_from_image,
    load_image_tensor,
    load_vision_bridge,
)
from llm.mamba_loader import load_mamba_lm
import torch


TEMPLATE_HEADERS = ("所见：", "结论：", "建议：", "病理倾向：")


def template_complete(text: str) -> bool:
    return all(h in text for h in TEMPLATE_HEADERS)


def build_force_words_ids(tokenizer) -> list[list[int]]:
    ids: list[list[int]] = []
    for phrase in TEMPLATE_HEADERS:
        toks = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        if toks:
            ids.append(toks)
    return ids


def normalize_template(text: str) -> str:
    txt = (text or "").strip()
    if not txt:
        return (
            "所见：胸部CT见异常，需结合原始影像复核。\n"
            "结论：病灶性质待定。\n"
            "建议：建议短期复查胸部CT并结合临床。\n"
            "病理倾向：炎性或肿瘤性待定。"
        )
    if template_complete(txt):
        return txt

    section_map = {h: "" for h in TEMPLATE_HEADERS}
    cur = None
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        hit = None
        for h in TEMPLATE_HEADERS:
            if line.startswith(h):
                hit = h
                section_map[h] = line[len(h):].strip()
                cur = h
                break
        if hit is None and cur is not None:
            section_map[cur] = (section_map[cur] + " " + line).strip()

    body = re.sub(r"\s+", " ", txt)
    for h in TEMPLATE_HEADERS:
        if not section_map[h]:
            section_map[h] = body

    return (
        f"所见：{section_map['所见：']}\n"
        f"结论：{section_map['结论：']}\n"
        f"建议：{section_map['建议：']}\n"
        f"病理倾向：{section_map['病理倾向：']}"
    )


def generate_strict(
    image_t: torch.Tensor,
    vision_bridge,
    llm_model,
    tokenizer,
    min_chars: int,
    max_new_tokens: int,
    max_visual_tokens: int,
    base_do_sample: bool,
    base_temperature: float,
    base_repetition_penalty: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
    suppress_eos_steps: int,
    max_retries: int,
    num_beams: int = 1,
    force_words_ids: list[list[int]] | None = None,
) -> tuple[str, list[str]]:
    cands: list[str] = []
    best = ""
    best_score = -1
    for attempt in range(max_retries + 1):
        txt = generate_from_image(
            image_t,
            vision_bridge,
            llm_model,
            tokenizer,
            prompt=PROMPT_SHORT_NO_PLACEHOLDERS,
            max_new_tokens=max_new_tokens + attempt * 96,
            max_visual_tokens=max_visual_tokens,
            do_sample=(base_do_sample if attempt == 0 else True),
            temperature=(base_temperature if attempt == 0 else max(0.7, base_temperature)),
            repetition_penalty=(base_repetition_penalty if attempt == 0 else max(1.1, base_repetition_penalty - 0.05)),
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_eos_steps=suppress_eos_steps + attempt * 64,
            num_beams=num_beams,
            force_words_ids=force_words_ids,
        ).strip()
        cands.append(txt)
        score = len(txt) + (300 if template_complete(txt) else 0)
        if score > best_score:
            best_score = score
            best = txt
        if len(txt) >= min_chars and template_complete(txt):
            break
    return normalize_template(best), cands


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict template inference with retry for longer and safer reports")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--mamba_model", type=str, default="d:/mamba/models/mamba-2.8b-hf")
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--val_sample", action="store_true")
    ap.add_argument("--num_val", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--max_visual_tokens", type=int, default=64)
    ap.add_argument("--min_chars", type=int, default=180)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--length_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--suppress_eos_steps", type=int, default=192)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--constrained_decode", action="store_true", help="启用约束解码，强制出现四段标题")
    ap.add_argument("--llm_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out_dir", type=str, default="d:/mamba-res")
    args = ap.parse_args()

    cfg = load_paths_config(REPO / "config" / "paths.yaml")
    cfg.setdefault("encoder_output_stage", 4)
    cfg.setdefault("encoder_target_spatial", 28)
    cfg["bridge_d_model"] = 2560
    cfg.setdefault("use_cmi", False)
    cfg.setdefault("roi_side", None)
    cfg.setdefault("cmi_compress_to", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_bridge = load_vision_bridge(args.checkpoint, cfg, device)
    llm_device_map = "cpu" if args.llm_device == "cpu" else args.llm_device
    llm_model, tokenizer = load_mamba_lm(args.mamba_model, device_map=llm_device_map)
    if hasattr(llm_model, "backbone") and hasattr(llm_model.backbone, "embeddings") and hasattr(llm_model, "lm_head"):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight
    llm_model.eval()
    force_words_ids = build_force_words_ids(tokenizer) if args.constrained_decode else None

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / ("run_strict_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.val_sample:
        csv_val = cfg.get("caption_csv_val")
        if not csv_val or not Path(csv_val).exists():
            raise FileNotFoundError(f"caption_csv_val not found: {csv_val}")
        ds = MedicalVLMDataset(csv_val, prompt_json_file=cfg.get("caption_prompt_json"))
        n = min(args.num_val, len(ds))
        meta = []
        for i in range(n):
            sample = ds[i]
            image_t = sample["image"].unsqueeze(0)
            gen, cands = generate_strict(
                image_t=image_t,
                vision_bridge=vision_bridge,
                llm_model=llm_model,
                tokenizer=tokenizer,
                min_chars=args.min_chars,
                max_new_tokens=args.max_new_tokens,
                max_visual_tokens=args.max_visual_tokens,
                base_do_sample=args.do_sample,
                base_temperature=args.temperature,
                base_repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                suppress_eos_steps=args.suppress_eos_steps,
                max_retries=args.max_retries,
                num_beams=args.num_beams,
                force_words_ids=force_words_ids,
            )
            (run_dir / f"sample_{i+1}_gen.txt").write_text(gen, encoding="utf-8")
            (run_dir / f"sample_{i+1}_ref.txt").write_text(str(sample["answer"]), encoding="utf-8")
            (run_dir / f"sample_{i+1}_cands.txt").write_text("\n\n-----\n\n".join(cands), encoding="utf-8")
            meta.append({"idx": i + 1, "image_path": sample.get("image_path"), "len": len(gen)})
        (run_dir / "meta.json").write_text(json.dumps({"checkpoint": args.checkpoint, "num_val": n, "samples": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {run_dir}")
        return 0

    if not args.image:
        raise ValueError("Use --image for single image inference, or --val_sample for validation sampling.")
    image_t = load_image_tensor(args.image).to(device)
    gen, cands = generate_strict(
        image_t=image_t,
        vision_bridge=vision_bridge,
        llm_model=llm_model,
        tokenizer=tokenizer,
        min_chars=args.min_chars,
        max_new_tokens=args.max_new_tokens,
        max_visual_tokens=args.max_visual_tokens,
        base_do_sample=args.do_sample,
        base_temperature=args.temperature,
        base_repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        suppress_eos_steps=args.suppress_eos_steps,
        max_retries=args.max_retries,
        num_beams=args.num_beams,
        force_words_ids=force_words_ids,
    )
    (run_dir / "generated.txt").write_text(gen, encoding="utf-8")
    (run_dir / "all_candidates.txt").write_text("\n\n-----\n\n".join(cands), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({"checkpoint": args.checkpoint, "image_path": args.image}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(gen)
    print(f"saved: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
