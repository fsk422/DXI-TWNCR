# -*- coding: utf-8 -*-
"""经典 Demšar CD 图（单轴紧凑版）—— k=12，DXI-TWNCR 拆为 Incr1 / IncrK 双标注。

风格对齐 _gen_cd_classic.py（nemenyi_cd_classic.pdf）：
  - 单横轴 + 秩次刻度，左右两列方法名（左=最优若干、右=其余，倒序），
    用引导线连接到轴上的平均秩次位置。
  - 顶部红色 CD 参考条；轴下方红色非显著区间条。
  - 我们的算法（DXI-TWNCR$_1$ / DXI-TWNCR$_K$）统一深红加粗 + 浅红底，
    显示为 "DXI-TWNCR (Incr1)" / "DXI-TWNCR (IncrK)"。

数据源：Significance_incrk_{AUC,PR,F1}.csv（method, avg_rank, CD, ...）。
输出：
  - paper/figures/nemenyi_cd.pdf      （AUC-ROC，论文 fig:cd 引用）
  - paper/figures/nemenyi_cd_pr.pdf   （AUC-PR）
  - paper/figures/nemenyi_cd_f1.pdf   （F1）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(BASE), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 我们的算法（两变体）——统一高亮
OURS = {"DXI-TWNCR$_1$", "DXI-TWNCR$_K$"}


def display_label(name):
    if name == "DXI-TWNCR$_1$":
        return "Incr1"
    if name == "DXI-TWNCR$_K$":
        return "IncrK"
    return name


def rank_to_color(r, rmin, rmax):
    t = np.clip((r - rmin) / (rmax - rmin), 0.0, 1.0)
    return (0.0, 0.25, 0.85 - t * 0.45)


def maximal_intervals(sorted_ranks, cd):
    K = len(sorted_ranks)
    intervals_raw = []
    for i in range(K - 1):
        j = i + 1
        while j < K and sorted_ranks[j] - sorted_ranks[i] < cd + 1e-12:
            j += 1
        j -= 1
        if j > i:
            intervals_raw.append((i, j))
    intervals_raw = list(dict.fromkeys(intervals_raw))
    intervals = []
    for a in intervals_raw:
        contained = False
        for b in intervals_raw:
            if a == b:
                continue
            if b[0] <= a[0] and b[1] >= a[1]:
                contained = True
                break
        if not contained:
            intervals.append(a)
    return intervals


def draw_classic(names, ranks, cd, metric_label, out_path):
    K = len(names)
    N = 14
    sorted_idx = np.argsort(ranks)
    sorted_names = np.array([names[i] for i in sorted_idx])
    sorted_ranks = np.array([ranks[i] for i in sorted_idx])

    rmin = float(sorted_ranks[0])
    rmax = float(sorted_ranks[-1])

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'mathtext.fontset': 'stix',
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_axis_off()

    x_left, x_right = 0.5, K + 0.5
    y_axis = 2.0
    ax.plot([x_left, x_right], [y_axis, y_axis], 'k-', linewidth=2.0)

    for t in range(1, K + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.09], 'k-', linewidth=1.2)
        ax.text(t, y_axis + 0.22, str(t), ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')

    # --- CD bar ---
    cd_x_start = x_left
    cd_x_end = cd_x_start + cd
    cd_y = y_axis + 0.45
    ax.plot([cd_x_start, cd_x_end], [cd_y, cd_y], 'r-', linewidth=2.2)
    ax.plot([cd_x_start, cd_x_start], [cd_y - 0.05, cd_y + 0.05], 'r-', linewidth=2.0)
    ax.plot([cd_x_end, cd_x_end], [cd_y - 0.05, cd_y + 0.05], 'r-', linewidth=2.0)
    ax.text(cd_x_start + cd / 2, cd_y + 0.12, f'CD = {cd:.2f}',
            ha='center', va='bottom', color='#CC0000', fontsize=12, fontweight='bold')

    # --- 名称分列：左=最优 K//2，右=其余（倒序） ---
    split_idx = K // 2
    right_items = list(zip(sorted_names[split_idx:], sorted_ranks[split_idx:]))[::-1]
    left_items = list(zip(sorted_names[:split_idx], sorted_ranks[:split_idx]))

    y_text_start = y_axis - 0.32
    y_step = 0.30
    TARGET = (0.85, 0.05, 0.05)

    def _draw(items, x_label_anchor, ha):
        for i, (name, r) in enumerate(items):
            y_pos = y_text_start - i * y_step
            ours = name in OURS
            c = TARGET if ours else rank_to_color(r, rmin, rmax)
            fw, fs = ('bold', 13.0) if ours else ('normal', 11.5)
            bg = 'none'
            label = display_label(name)
            ax.plot([r, r], [y_axis, y_pos], color=c, linewidth=1.5, alpha=0.95, zorder=2)
            ax.plot([r, x_label_anchor], [y_pos, y_pos], color=c, linewidth=1.5, alpha=0.95, zorder=2)
            ax.text(x_label_anchor + (-0.05 if ha == 'right' else 0.05), y_pos, label,
                    ha=ha, va='center', color=c, fontsize=fs, fontweight=fw, zorder=10,
                    bbox=dict(boxstyle='square,pad=0.12', facecolor=bg, edgecolor='none', alpha=1.0))

    _draw(right_items, x_right + 0.55, 'left')
    _draw(left_items, x_left - 0.55, 'right')

    # --- 红色非显著区间条 ---
    intervals = maximal_intervals(sorted_ranks, cd)
    intervals_rank = sorted(
        [(float(min(sorted_ranks[i], sorted_ranks[j])),
          float(max(sorted_ranks[i], sorted_ranks[j])))
         for (i, j) in intervals],
        key=lambda t: (t[1] - t[0]), reverse=True)

    y_connect_start = y_axis - 0.16
    y_connect_step = 0.14
    levels_data = []
    for (lo, hi) in intervals_rank:
        lvl = None
        while lvl is None:
            placed = False
            for lv, entries in enumerate(levels_data):
                if all(hi < lo2 or hi2 < lo for (lo2, hi2) in entries):
                    lvl = lv
                    entries.append((lo, hi))
                    placed = True
                    break
            if lvl is None and not placed:
                levels_data.append([(lo, hi)])
                lvl = len(levels_data) - 1
        yl = y_connect_start - lvl * y_connect_step
        ax.plot([lo, hi], [yl, yl], color='#CC0000', linewidth=2.0, solid_capstyle='butt', zorder=5)
        ax.plot([lo, lo], [yl - 0.07, yl + 0.07], color='#CC0000', linewidth=2.4, solid_capstyle='butt', zorder=5)
        ax.plot([hi, hi], [yl - 0.07, yl + 0.07], color='#CC0000', linewidth=2.4, solid_capstyle='butt', zorder=5)

    pad_x = 3.4
    max_rows = max(len(right_items), len(left_items))
    ax.set_xlim(x_left - pad_x, x_right + pad_x)
    ax.set_ylim(y_text_start - max_rows * y_step - 0.4, y_axis + 1.20)

    plt.tight_layout(pad=0.25)
    plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"[{metric_label}] CD = {cd:.3f}, {len(intervals)} 个非显著区间 -> {out_path}")


def main():
    jobs = [
        ("AUC", "nemenyi_cd.pdf"),
        ("PR", "nemenyi_cd_pr.pdf"),
        ("F1", "nemenyi_cd_f1.pdf"),
    ]
    for metric, fname in jobs:
        csv = os.path.join(BASE, f"Significance_incrk_{metric}.csv")
        df = pd.read_csv(csv)
        names = df["method"].astype(str).tolist()
        ranks = df["avg_rank"].astype(float).tolist()
        cd = float(df["CD"].iloc[0])
        draw_classic(names, ranks, cd, metric, os.path.join(FIG_DIR, fname))


if __name__ == "__main__":
    main()