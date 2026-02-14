"""
检测：训练后的参数 + 图像 能否生成文本。
默认从验证集取 1 张图，加载 Vision+Bridge 与 Mamba，打印「输入图像」和「生成文本」。

用法:
  python scripts/check_image_to_text.py
  python scripts/check_image_to_text.py --image D:/nnunet_raw/.../xxx.nii.gz
  python scripts/check_image_to_text.py --checkpoint outputs/vision_bridge_overnight_best.pt --image ...
  # 若 WinError 10060（连不上 Hugging Face），可先设镜像再运行，或用本地模型：
  #   $env:HF_ENDPOINT = "https://hf-mirror.com"
  #   python scripts/check_image_to_text.py --mamba_model D:/mamba/models/mamba-2.8b-hf

防 OOM：默认 --llm_device auto；若显存不足可改为 --llm_device cpu。当 llm_device=auto 且 --max_visual_tokens>196 时会自动限制为 196。可选 --llm_8bit（需 pip install bitsandbytes）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.medical_vlm_dataset import load_paths_config, CAPTION_DEFAULT_QUESTION

# 与训练格式对齐：推理 prompt 与 train_vlm 中 full_text = question+"\n"+answer 的「问题+换行」完全一致，避免模型进入未见过上下文
DEFAULT_QUESTION = CAPTION_DEFAULT_QUESTION


def _safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_slice_artifacts(out_dir: Path, image_t):
    """
    保存推理用的 2D slice。
    - 尝试保存 PNG（若 pillow 可用）
    - 否则保存 NPY
    image_t: torch.Tensor, 形状可为 [1,1,H,W] 或 [1,H,W] 或 [H,W]
    """
    try:
        import torch
        import numpy as np
    except Exception:
        return

    if hasattr(image_t, "detach"):
        arr = image_t.detach().float().cpu()
        while arr.dim() > 2:
            arr = arr[0]
        arr = arr.numpy()
    else:
        arr = image_t
    # 归一化到 0..255 便于保存 png
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn > 1e-8:
        vis = (arr - mn) / (mx - mn)
    else:
        vis = arr * 0.0
    vis_u8 = (vis * 255.0).clip(0, 255).astype("uint8")

    out_dir.mkdir(parents=True, exist_ok=True)
    # png
    try:
        from PIL import Image

        Image.fromarray(vis_u8).save(out_dir / "slice.png")
        return
    except Exception:
        pass
    # fallback npy
    try:
        import numpy as np

        np.save(str(out_dir / "slice.npy"), arr)
    except Exception:
        return


def main():
    ap = argparse.ArgumentParser(description="检测：图像 + 训练权重 → 生成文本")
    ap.add_argument("--checkpoint", type=str, default=None, help="Vision+Bridge 权重，默认 outputs 下 best_val / overnight_best / final")
    ap.add_argument("--image", type=str, default=None, help="NIfTI 图像路径；不指定则从验证集取 1 张")
    ap.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf", help="Mamba 模型：HF 模型 id 或本地路径")
    ap.add_argument("--max_new_tokens", type=int, default=512, help="生成 token 上限，医学报告建议 512 避免被截断；默认贪心解码")
    ap.add_argument("--max_visual_tokens", type=int, default=196, help="视觉 token 上限；必须与 Stage 2 训练时一致（train_vlm 默认 196），否则易「看图不对齐」乱码")
    ap.add_argument("--llm_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="LLM 设备：auto 使用 GPU（推荐），cpu 防 OOM，cuda 强制 GPU")
    ap.add_argument("--llm_8bit", action="store_true", help="LLM 8-bit 加载（需 bitsandbytes），显著省显存")
    ap.add_argument("--repetition_penalty", type=float, default=1.2, help="重复惩罚；严重重复/循环时可提高到 1.5，默认 1.2")
    ap.add_argument("--temperature", type=float, default=0.7, help="采样温度，仅 do_sample 时有效；略高可减轻立即 EOS，默认 0.7")
    ap.add_argument("--top_p", type=float, default=0.95, help="采样 top_p，仅 do_sample 时有效，默认 0.95")
    ap.add_argument("--no_do_sample", action="store_true", help="强制贪心解码（医疗报告常用，减轻与图无关幻觉）")
    ap.add_argument("--do_sample", action="store_true", help="开启采样（严重重复时可试 do_sample + temperature 0.7 + top_p 0.9 打破循环）")
    ap.add_argument("--debug_vision", action="store_true", help="打印视觉特征统计（min/max/mean/NaN），用于排查「看图不对齐」")
    ap.add_argument("--out_dir", type=str, default="D:/mamba-res", help="落盘保存目录（默认 D:/mamba-res）")
    ap.add_argument("--save_out", type=str, default=None, help="额外保存生成文本到指定文件路径（例如 D:/mamba-res/gen.txt）")
    args = ap.parse_args()
    # 医疗报告默认贪心解码（--no_do_sample 为默认），减轻与图像无关的幻觉；严重重复时可试 --do_sample
    args.do_sample = getattr(args, "do_sample", False)
    if getattr(args, "no_do_sample", False):
        args.do_sample = False
    # GPU 时仅当用户显式传了过大值才截断，默认 196 与训练一致，不自动改为 96（否则易导致「外(IM12)」式乱码）
    if args.llm_device == "auto" and args.max_visual_tokens > 196:
        args.max_visual_tokens = 196
        print(f"已限制 max_visual_tokens=196（与训练默认一致）以防 OOM", flush=True)

    config = load_paths_config(REPO / "config" / "paths.yaml")
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config["bridge_d_model"] = 2560

    out_dir = REPO / "outputs"
    ckpt = args.checkpoint
    if not ckpt or not Path(ckpt).exists():
        for name in ["vision_bridge_vlm_final.pt", "vision_bridge_best_val.pt", "vision_bridge_overnight_best.pt", "vision_bridge_final.pt"]:
            p = out_dir / name
            if p.exists():
                ckpt = str(p)
                break
    if not ckpt or not Path(ckpt).exists():
        print("未找到 checkpoint，请先训练或指定 --checkpoint")
        return 1

    # 若为 Stage 2 产出且存在 stage2_config.json，提示与训练配置一致（同 tokenizer/max_visual_tokens）
    stage2_cfg_path = out_dir / "stage2_config.json"
    if Path(ckpt).name == "vision_bridge_vlm_final.pt" and stage2_cfg_path.exists():
        try:
            sc = json.loads(stage2_cfg_path.read_text(encoding="utf-8"))
            mv = sc.get("max_visual_tokens")
            mm = sc.get("mamba_model")
            if mv is not None and args.max_visual_tokens != mv:
                print(f"提示: stage2_config.json 显示训练时 max_visual_tokens={mv}，当前为 {args.max_visual_tokens}；建议加 --max_visual_tokens {mv}", flush=True)
            if mm and args.mamba_model != mm:
                print(f"提示: stage2_config.json 显示训练时 mamba_model={mm}；建议与训练一致以保持词表一致", flush=True)
        except Exception:
            pass

    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")

    print("加载 Vision+Bridge...", flush=True)
    from inference import load_vision_bridge, generate_from_image, load_image_tensor
    vision_bridge = load_vision_bridge(ckpt, config, device)

    print("加载 Mamba LLM（可能较久）...", flush=True)
    from llm.mamba_loader import load_mamba_lm
    llm_device_map = "cpu" if args.llm_device == "cpu" else ("auto" if device.type == "cuda" else None)
    llm_model, tokenizer = load_mamba_lm(
        args.mamba_model,
        device_map=llm_device_map,
        load_in_8bit=getattr(args, "llm_8bit", False),
    )
    # 与 train_vlm.py 一致：绑定 lm_head。Stage 2 只保存 Vision+Bridge，不保存 LLM；HF 的 mamba-2.8b 本身无 lm_head，
    # 绑定到 embeddings 是设计行为，并非「权重丢失」。若训练时微调了 lm_head 并单独保存，此处需改为加载该权重。
    if hasattr(llm_model, "backbone") and hasattr(llm_model.backbone, "embeddings") and hasattr(llm_model, "lm_head"):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight
        print("执行权重绑定 (Tying lm_head → backbone.embeddings)；详见 docs/INFERENCE_AND_HALLUCINATION.md", flush=True)
    llm_model.eval()
    # 词表一致性：本仓库 train_vlm.py 未添加特殊 token、未 resize_token_embeddings，推理应与训练同源 tokenizer
    vocab_size = len(tokenizer)
    model_vocab = getattr(llm_model.config, "vocab_size", None)
    print(f"词表: len(tokenizer)={vocab_size} config.vocab_size={model_vocab}（须一致，否则易乱码）", flush=True)
    if model_vocab is not None and vocab_size != model_vocab:
        print("警告: tokenizer 与模型 vocab_size 不一致，可能导致「外(IM12)」式乱码；请勿在推理端单独 resize_token_embeddings，除非训练时也添加了相同特殊 token。", flush=True)
    print(f"max_visual_tokens={args.max_visual_tokens}（须与 Stage 2 训练时一致；train_vlm 默认 196）", flush=True)
    llm_dev = next(llm_model.parameters()).device
    print(f"LLM 设备: {llm_dev}（OOM 时请用 --llm_device cpu）", flush=True)
    if llm_dev.type == "cuda":
        # 诊断：是否落入 sequential fallback（无 fast kernel）会非常慢
        try:
            import mamba_ssm  # noqa: F401
            print("检测: 已安装 mamba_ssm（若仍提示 fast path unavailable，可继续安装 causal-conv1d/kernels）。", flush=True)
        except Exception:
            print("提示: 未安装 mamba_ssm，Mamba 使用 sequential 实现（较慢但可用）。可选加速见 docs/FAQ_MAMBA_SSM_INSTALL.md", flush=True)

    # 准备落盘目录：每次运行一个子目录，便于你检查
    base_out = Path(args.out_dir)
    run_dir = base_out / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoint": ckpt,
        "mamba_model": args.mamba_model,
        "llm_device": str(llm_dev),
        "max_new_tokens": args.max_new_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "config_paths_yaml": str(REPO / "config" / "paths.yaml"),
    }

    if args.image:
        path = Path(args.image)
        # 拒绝占位符 "..." 或无效路径，避免 HDF5/NIfTI 报错
        if args.image.strip() in ("...", ".", "..") or not path.exists():
            print("请使用真实的 NIfTI 路径，例如:", flush=True)
            print('  --image "D:\\nnunet_raw\\Dataset503_TBLesion_327\\imagesTr\\0000978995_20260123_0000.nii.gz"', flush=True)
            print("或省略 --image，从验证集自动取一张图。", flush=True)
            return 1
        if not (args.image.endswith(".nii.gz") or args.image.endswith(".nii")):
            print(f"当前仅支持 .nii / .nii.gz 文件，收到: {args.image}", flush=True)
            return 1
        image_t = load_image_tensor(args.image)
        prompt = DEFAULT_QUESTION
        print(f"输入图像: {args.image}", flush=True)
        print(f"问题: {prompt[:80]}...", flush=True)
        raw_out = []
        text = generate_from_image(
            image_t, vision_bridge, llm_model, tokenizer, prompt=prompt,
            max_new_tokens=args.max_new_tokens, device=device, max_visual_tokens=args.max_visual_tokens,
            do_sample=args.do_sample, temperature=getattr(args, "temperature", 0.7), top_p=getattr(args, "top_p", 0.95),
            repetition_penalty=getattr(args, "repetition_penalty", 1.2), raw_out=raw_out,
            debug_vision=getattr(args, "debug_vision", False),
        )
        if not text.strip():
            print("生成文本: （空或已过滤）", flush=True)
            if raw_out and raw_out[0].strip():
                print(f"原始输出（未过滤）: {raw_out[0][:600]}", flush=True)
            else:
                print("原始输出为空（模型可能立即 EOS）。可试 --do_sample 或提高 --repetition_penalty 1.5", flush=True)
        else:
            print(f"生成文本:\n{text.encode('gbk', errors='replace').decode('gbk')}", flush=True)
        meta.update({"image_path": args.image, "prompt": prompt})
        _save_slice_artifacts(run_dir, image_t)
        _safe_write_text(run_dir / "generated.txt", text)
    else:
        from data.medical_vlm_dataset import MedicalVLMDataset
        csv_val = config.get("caption_csv_val")
        if not csv_val or not Path(csv_val).exists():
            print("验证集 CSV 不存在，请指定 --image")
            return 1
        val_ds = MedicalVLMDataset(csv_val, prompt_json_file=config.get("caption_prompt_json"))
        sample = val_ds[0]
        image_t = sample["image"].unsqueeze(0)
        question = sample.get("question") or DEFAULT_QUESTION
        print(f"输入图像: {sample['image_path']}", flush=True)
        print(f"问题: {question[:80]}...", flush=True)
        raw_out = []
        text = generate_from_image(
            image_t, vision_bridge, llm_model, tokenizer, prompt=question,
            max_new_tokens=args.max_new_tokens, device=device, max_visual_tokens=args.max_visual_tokens,
            do_sample=args.do_sample, temperature=getattr(args, "temperature", 0.7), top_p=getattr(args, "top_p", 0.95),
            repetition_penalty=getattr(args, "repetition_penalty", 1.2), raw_out=raw_out,
            debug_vision=getattr(args, "debug_vision", False),
        )
        if not text.strip():
            print("生成文本: （空或已过滤）", flush=True)
            if raw_out and raw_out[0].strip():
                print(f"原始输出（未过滤）: {raw_out[0][:600]}", flush=True)
            else:
                print("原始输出为空。可试 --do_sample 或 --repetition_penalty 1.5", flush=True)
        else:
            print(f"生成文本:\n{text.encode('gbk', errors='replace').decode('gbk')}", flush=True)
        print(f"\n参考回答: {sample['answer'][:200]}...", flush=True)
        meta.update(
            {
                "image_path": sample["image_path"],
                "prompt": question,
                "reference_answer_head": sample["answer"][:200],
            }
        )
        _save_slice_artifacts(run_dir, image_t)
        _safe_write_text(run_dir / "generated.txt", text)

    # 保存 meta
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # 额外保存到指定路径（兼容你想要固定文件名）
    if args.save_out:
        _safe_write_text(Path(args.save_out), text)
    print(f"\n已落盘保存到: {run_dir}", flush=True)

    print("\n检测完成：图像 → 文本 流程正常。", flush=True)
    print("若生成仍为英文或乱码，请确保已跑完 Stage 2 图文对齐训练（caption_loss 下降）。", flush=True)
    print("若生成为空：可试 --do_sample 或 --repetition_penalty 1.5；确认 checkpoint 为 Stage 2 训练后的权重。", flush=True)
    print("若生成内容单薄（缺病灶定位/尺寸）：可试 --do_sample --temperature 0.7 增加多样性，或增加训练 epoch（如 30–40）。", flush=True)
    print("若「看图不准」：先用 --no_do_sample 观察最稳定输出；勿对 tokenizer 做 resize_token_embeddings，否则易索引偏移乱码。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
