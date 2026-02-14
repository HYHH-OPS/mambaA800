"""
从 NIfTI 图像 + 分割 mask 生成：结节勾画叠加图（overlay.png）+ 结节信息表（nodules.csv）。
依赖：nibabel, scipy, matplotlib, pandas（conda/pip 安装）。
用法:
  python scripts/nodule_overlay_and_stats.py --image "D:/nnunet_raw/.../xxx_0000.nii.gz" --mask "D:/nnunet_results/.../xxx.nii.gz" --output_dir "D:/mamba-res/nodule_xxx"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import pandas as pd
except ImportError:
    pd = None


def load_nifti(path: str):
    if nib is None:
        raise ImportError("需要安装 nibabel: pip install nibabel")
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()
    return data, spacing


def find_best_slice(mask_3d: np.ndarray) -> int:
    if mask_3d.ndim != 3:
        return 0
    counts = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    if counts.max() == 0:
        return mask_3d.shape[0] // 2
    return int(np.argmax(counts))


def compute_nodule_stats(mask_3d: np.ndarray, spacing) -> "pd.DataFrame":
    if pd is None:
        raise ImportError("需要安装 pandas: pip install pandas")
    binary = mask_3d > 0
    labeled, num = ndi.label(binary)
    dz, dy, dx = spacing[:3] if len(spacing) >= 3 else (1.0, 1.0, 1.0)
    voxel_volume_ml = (dz * dy * dx) / 1000.0
    rows = []
    for nid in range(1, num + 1):
        coords = np.argwhere(labeled == nid)
        if coords.size == 0:
            continue
        z_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]
        voxel_count = coords.shape[0]
        volume_ml = voxel_count * voxel_volume_ml
        cz = float(z_idx.mean() * dz)
        cy = float(y_idx.mean() * dy)
        cx = float(x_idx.mean() * dx)
        rows.append({
            "nodule_id": nid,
            "voxel_count": int(voxel_count),
            "volume_ml": round(volume_ml, 6),
            "center_z_mm": round(cz, 2),
            "center_y_mm": round(cy, 2),
            "center_x_mm": round(cx, 2),
        })
    return pd.DataFrame(rows)


def save_overlay_png(image_3d: np.ndarray, mask_3d: np.ndarray, out_path: Path) -> None:
    if plt is None:
        raise ImportError("需要安装 matplotlib: pip install matplotlib")
    if image_3d.ndim == 4:
        image_3d = image_3d[..., 0]
    assert image_3d.ndim == 3 and mask_3d.ndim == 3
    z = find_best_slice(mask_3d)
    img = np.asarray(image_3d[z], dtype=np.float32)
    msk = (mask_3d[z] > 0)
    vmin, vmax = np.percentile(img, (1, 99))
    img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(~msk, msk), cmap="autumn", alpha=0.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="结节勾画图 + 结节信息表")
    ap.add_argument("--image", required=True, help="CT NIfTI 路径")
    ap.add_argument("--mask", required=True, help="分割 mask NIfTI 路径")
    ap.add_argument("--output_dir", default="D:/mamba-res/nodules", help="输出目录")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("加载图像和 mask ...")
    img3d, spacing_img = load_nifti(args.image)
    msk3d, spacing_msk = load_nifti(args.mask)
    if img3d.shape != msk3d.shape:
        raise ValueError(f"image 形状 {img3d.shape} 与 mask {msk3d.shape} 不一致")
    spacing = spacing_img[:3] if len(spacing_img) >= 3 else (1.0, 1.0, 1.0)
    print("计算结节统计 ...")
    df = compute_nodule_stats(msk3d, spacing)
    csv_path = out_dir / "nodules.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {csv_path}")
    print("生成勾画图 ...")
    png_path = out_dir / "overlay.png"
    save_overlay_png(img3d, msk3d, png_path)
    print(f"已保存: {png_path}")
    print("结节数量:", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
