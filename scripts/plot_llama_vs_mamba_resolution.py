"""
论文用图：Llama 需要压缩特征 vs Mamba 保留高分辨率特征

生成一张顶刊级别对比图，说明：
- 左：传统 VLM（如 Llama + ViT）必须将 Feature Map 压成 14×14，信息损失。
- 右：本方案（Mamba）线性复杂度，可保留 28×28 甚至更高分辨率。

用法: python scripts/plot_llama_vs_mamba_resolution.py
输出: scripts/fig_llama_vs_mamba_resolution.png
"""

import math
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def draw_schematic(ax, title: str, grid_size: int, color_theme: str, is_compressed: bool):
    """在 ax 上画一张示意图：grid_size x grid_size 的 feature map。"""
    n = grid_size
    # 画网格
    for i in range(n + 1):
        ax.axhline(i, color="gray", linewidth=0.5)
        ax.axvline(i, color="gray", linewidth=0.5)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")

    if color_theme == "red":
        face = "#ffcccc"
        edge = "#cc0000"
    else:
        face = "#ccffcc"
        edge = "#006600"

    # 填充格子表示“保留的特征”
    for i in range(n):
        for j in range(n):
            rect = mpatches.Rectangle((j, i), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.8)
            ax.add_patch(rect)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_flow_arrow(ax, x0, y0, x1, y1, label: str, color: str):
    """画带标签的箭头。"""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(mid_x, mid_y, label, fontsize=9, ha="center", color=color)


def main():
    if not HAS_MATPLOTLIB:
        print("Install matplotlib to generate the figure: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 左：Llama / 传统 VLM — 压缩到 14×14
    ax_left = axes[0]
    draw_schematic(
        ax_left,
        "Llama / Attention-based VLM\n(Feature Map → 14×14, heavy compression)",
        grid_size=14,
        color_theme="red",
        is_compressed=True,
    )
    ax_left.text(7, -1.2, "14×14 → 196 tokens", fontsize=11, ha="center")
    ax_left.text(7, -2.0, "O(L²) attention → must compress", fontsize=10, ha="center", style="italic")

    # 右：Mamba — 保留 28×28
    ax_right = axes[1]
    draw_schematic(
        ax_right,
        "Mamba-based VLM (Ours)\n(Feature Map → 28×28, high resolution)",
        grid_size=28,
        color_theme="green",
        is_compressed=False,
    )
    ax_right.text(14, -1.2, "28×28 → 784 tokens", fontsize=11, ha="center")
    ax_right.text(14, -2.0, "O(L) SSM → keep resolution", fontsize=10, ha="center", style="italic")

    fig.suptitle(
        "Vision–Language: Llama Needs Compression vs Mamba Retains High-Resolution Features",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "fig_llama_vs_mamba_resolution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
