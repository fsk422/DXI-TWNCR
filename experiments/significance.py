# -*- coding: utf-8 -*-
"""§8.6 统计显著性检验（k=12：DXI-TWNCR₁ + DXI-TWNCR_K + 10 baseline）。

在 significance_test.py 基础上扩展：
  1) 新增 AUC-ROC 指标（此前仅 PR/F1）。
  2) 将 DXI-TWNCR 拆分为 DXI-TWNCR₁（Incr1）与 DXI-TWNCR_K（IncrK），k 由 11 → 12。
  3) Nemenyi CD 的 q_α 按标准 Demšar 表值取 k=12 → 3.268（studentized_range/sqrt2）。

数据源：
  - AUC/AP : IncrCore_checkpoint_OVERNIGHT.csv（AUC_Incr1/AUC_IncrK、AP_Incr1/AP_IncrK）
             + BaselinesUnified_checkpoint.csv（AUC_xxx/AP_xxx）
  - F1     : F1_complete.csv（F1_ITW-NCR=Incr1 + 10 baseline）
             + F1_incrk.csv（F1_ITW-NCR_K=IncrK）

输出：
  - Significance_incrk_{AUC,PR,F1}.csv（方法 × 平均指标值 + 平均秩）
  - paper/figures/nemenyi_cd_incrk_{auc,pr,f1}.pdf
  - 控制台打印 LaTeX 表体与脚注（供人工核对后写入论文）。
"""
import numpy as np, pandas as pd, os
from scipy.stats import chi2, studentized_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(BASE), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

INCR_CSV = os.path.join(BASE, "IncrCore_checkpoint_OVERNIGHT.csv")
BASELINE_CSV = os.path.join(BASE, "BaselinesUnified_checkpoint.csv")
F1_CSV = os.path.join(BASE, "F1_complete.csv")
F1_INCRK_CSV = os.path.join(BASE, "F1_incrk.csv")

BASELINES = ["IncLOF", "HST", "RRCF", "xStream", "LODA",
             "EILOF", "SiForest", "HBOS", "ECOD", "BWOAIF"]
DXI1 = "DXI-TWNCR$_1$"
DXIK = "DXI-TWNCR$_K$"
PROPOSED = {DXI1, DXIK}
METHODS = [DXI1, DXIK] + BASELINES


def base_col(prefix, m):
    return f"{prefix}_{m}" if m != "BWOAIF" else f"{prefix}_BWOAIF+DIFFI"


def build_auc_ap(prefix):
    incr = pd.read_csv(INCR_CSV).set_index("dataset")
    base = pd.read_csv(BASELINE_CSV).set_index("dataset")
    assert set(incr.index) == set(base.index), "数据集集合不一致"
    datasets = list(incr.index)
    data = {
        DXI1: incr[f"{prefix}_Incr1"].astype(float).values,
        DXIK: incr[f"{prefix}_IncrK"].astype(float).values,
    }
    for m in BASELINES:
        data[m] = base[base_col(prefix, m)].astype(float).values
    return pd.DataFrame(data, index=datasets)


def build_f1():
    f1 = pd.read_csv(F1_CSV).set_index("dataset")
    incrk = pd.read_csv(F1_INCRK_CSV).set_index("dataset")
    assert set(f1.index) == set(incrk.index), "F1 数据集集合不一致"
    datasets = list(f1.index)
    data = {
        DXI1: f1["F1_ITW-NCR"].astype(float).values,
        DXIK: incrk["F1_ITW-NCR_K"].astype(float).values,
    }
    for m in BASELINES:
        data[m] = f1[base_col("F1", m)].astype(float).values
    return pd.DataFrame(data, index=datasets)


def friedman_nemenyi(mat, alpha=0.05):
    N, k = mat.shape
    ranks = np.array([rankdata_average(-mat.iloc[i].values.astype(float))
                      for i in range(N)])
    avg_ranks = ranks.mean(axis=0)
    chi2_stat = (12.0 * N / (k * (k + 1))) * np.sum((avg_ranks - (k + 1) / 2.0) ** 2)
    p = float(chi2.sf(chi2_stat, k - 1))
    q = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * N)))
    return avg_ranks, chi2_stat, p, cd, ranks, q


def rankdata_average(a):
    # scipy.stats.rankdata(a, method='average')，避免依赖
    from scipy.stats import rankdata
    return rankdata(a, method="average")


def fmt_p(p):
    if p < 1e-9:
        return "$p<10^{-9}$"
    if p < 1e-7:
        return "$p<10^{-7}$"
    if p < 1e-5:
        return "$p<10^{-5}$"
    return f"$p={p:.2e}$"


def process(metric_key, mat, out_csv, out_pdf):
    N, k = mat.shape
    avg_ranks, chi2_stat, p, cd, ranks, q = friedman_nemenyi(mat)
    order = np.argsort(avg_ranks)
    cols = list(mat.columns)

    rows = []
    for j in order:
        m = cols[j]
        rows.append({"method": m,
                     "mean_metric": float(mat[m].mean()),
                     "avg_rank": float(avg_ranks[j])})
    df = pd.DataFrame(rows)
    last_rank = float(avg_ranks[order[-1]])
    df["sig_vs_worst"] = df["avg_rank"].apply(lambda r: (last_rank - r) > cd)
    df["chi2"] = chi2_stat
    df["p"] = p
    df["CD"] = cd
    df["q"] = q
    df.to_csv(out_csv, index=False)

    # LaTeX 表体
    lines = []
    for _, r in df.iterrows():
        name = r["method"]
        rank_txt = f"{r['avg_rank']:.2f}"
        dag = r"$^{\dagger}$" if (last_rank - r["avg_rank"]) > cd else ""
        if name in PROPOSED:
            lines.append(r"\textbf{" + name + r"} & \textbf{" + rank_txt + "}" + dag + r"\\")
        else:
            lines.append(name + " & " + rank_txt + dag + r"\\")
    body = "\n".join(lines)
    footer = (r"\multicolumn{2}{l}{\footnotesize $\chi^2=" + f"{chi2_stat:.2f}" +
              r"$, " + fmt_p(p) + r", CD$=" + f"{cd:.3f}" +
              r"$ ($\alpha=0.05$, $k=12$, $N=14$)}\\" + "\n" +
              r"\multicolumn{2}{l}{\footnotesize $^{\dagger}$ 表示与末位方法的秩次差超过 CD，差异显著。}")

    print(f"\n{'='*88}")
    print(f"  {metric_key} — Friedman/Nemenyi（k={k}, N={N}, q={q:.4f}）")
    print(f"  χ² = {chi2_stat:.4f}  p = {p:.3e}  CD = {cd:.4f}")
    print(f"{'='*88}")
    print(f"  {'方法':<16s} {'平均指标值':>12s} {'平均秩':>8s}  {'显著优于末位':>6s}")
    for _, r in df.iterrows():
        mark = "†" if r["sig_vs_worst"] else ""
        print(f"  {r['method']:<16s} {r['mean_metric']:>12.4f} {r['avg_rank']:>8.3f}  {mark:>6s}")

    # 末位方法名（供脚注"与末位方法（XXX）"）
    worst_name = cols[order[-1]]
    print(f"  >> 末位方法：{worst_name}（rank={last_rank:.3f}）")
    for v in (DXI1, DXIK):
        r = df[df["method"] == v]["avg_rank"].values[0]
        pos = int(np.where(order == cols.index(v))[0][0]) + 1
        print(f"  >> {v}: 平均秩 = {r:.3f}（顺位第 {pos}）")

    print(f"\n  --- LaTeX 表体 ---")
    print(body)
    print(footer)
    plot_cd(metric_key, cols, order, avg_ranks, cd, k, N, out_pdf,
            chi2_stat, p, worst_name)
    return df


def plot_cd(metric_key, cols, order, avg_ranks, cd, k, n, out_path,
            chi2_stat, p, worst_name):
    sorted_names = [cols[j] for j in order]
    sorted_ranks = [float(avg_ranks[j]) for j in order]
    K = len(sorted_names)
    ranks = np.array(sorted_ranks, float)

    cliques_raw = []
    for i in range(K):
        j = i
        while j + 1 < K and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:
            cliques_raw.append((i, j))
    cliques = []
    for (a, b) in cliques_raw:
        is_subset = False
        for (c, d) in cliques_raw:
            if (a, b) != (c, d) and c <= a and b <= d and (c < a or b < d):
                is_subset = True
                break
        if not is_subset:
            cliques.append((a, b))
    levels = [-1] * len(cliques)
    for i, (a, b) in enumerate(cliques):
        for lvl in range(len(cliques)):
            ok = True
            for j in range(i):
                if levels[j] == lvl:
                    c, d = cliques[j]
                    if not (ranks[b] < ranks[c] - 0.01 or ranks[a] > ranks[d] + 0.01):
                        ok = False
                        break
            if ok:
                levels[i] = lvl
                break
        if levels[i] == -1:
            levels[i] = max(levels) + 1
    sorted_idx = sorted(range(len(cliques)), key=lambda i: (levels[i], cliques[i][0]))
    cliques = [cliques[i] for i in sorted_idx]
    levels = [levels[i] for i in sorted_idx]
    if len(levels) > 0:
        max_level = max(levels)
        levels = [max_level - lv for lv in levels]

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(1.0, 12.5)
    ax.invert_xaxis()
    ax.set_ylim(-0.9, 2.0)
    ax.yaxis.set_visible(False)
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#333333")
    ax.set_xticks(range(2, 13))
    ax.tick_params(axis="x", labelsize=9.5, color="#333333")
    ax.set_xlabel("Average Rank  (smaller = better)", fontsize=11, labelpad=5, color="#333333")
    ax.grid(axis="x", alpha=0.20, linestyle="--", linewidth=0.5, color="#888888")

    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        if name in PROPOSED:
            ax.scatter(rank, 0.0, s=130, c="#E74C3C", marker="D",
                       edgecolors="#922B21", linewidths=1.6, zorder=10)
        else:
            ax.scatter(rank, 0.0, s=85, c="#2980B9", marker="o",
                       edgecolors="#1A5276", linewidths=1.0, zorder=10)

    label_y_base = -0.16
    label_stagger = {}
    prev_rank = None
    for i in range(K):
        if prev_rank is not None and ranks[i] - prev_rank < 0.55:
            label_stagger[i] = not label_stagger.get(i - 1, False)
        else:
            label_stagger[i] = False
        prev_rank = ranks[i]
    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        offset = -22 if not label_stagger[i] else 22
        va = "top" if offset < 0 else "bottom"
        y_anchor = label_y_base if offset < 0 else label_y_base + 0.05
        if name in PROPOSED:
            ax.annotate(name, xy=(rank, y_anchor), xytext=(0, offset),
                        textcoords="offset points", fontsize=9.5, fontweight="bold",
                        color="#C0392B", ha="center", va=va, zorder=10)
        else:
            ax.annotate(name, xy=(rank, y_anchor), xytext=(0, offset),
                        textcoords="offset points", fontsize=9,
                        color="#2C3E50", ha="center", va=va, zorder=10)

    cd_y = 1.15
    cd_start = ranks[0]
    ax.plot([cd_start, cd_start + cd], [cd_y, cd_y], "-", color="#E74C3C",
            linewidth=5.0, solid_capstyle="butt", zorder=8)
    ax.plot([cd_start, cd_start], [cd_y - 0.12, cd_y + 0.12], "k-", linewidth=1.8, zorder=8)
    ax.plot([cd_start + cd, cd_start + cd], [cd_y - 0.12, cd_y + 0.12], "k-", linewidth=1.8, zorder=8)
    ax.text(cd_start + cd / 2, cd_y + 0.16, f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#C0392B")
    ax.text(cd_start - 0.35, cd_y, "Critical\nDifference", ha="right", va="center",
            fontsize=8, color="#666666", style="italic")

    bar_colors = ["#5DADE2", "#F5B041", "#58D68D", "#AF7AC5"]
    bar_y_base = 0.28
    bar_spacing = 0.22
    for idx, ((a, b), lvl) in enumerate(zip(cliques, levels)):
        x0, x1 = ranks[a], ranks[b]
        y = bar_y_base + lvl * bar_spacing
        color = bar_colors[lvl % len(bar_colors)]
        ax.plot([x0, x1], [y, y], "-", color=color, linewidth=10,
                solid_capstyle="round", alpha=0.80, zorder=5)
        ax.plot([x0, x1], [y, y], "-", color="#34495E", linewidth=1.0,
                solid_capstyle="round", alpha=0.45, zorder=5)
        for j in range(a, b + 1):
            ax.plot([ranks[j], ranks[j]], [0, y], "-", color="#95A5A6",
                    linewidth=1.2, alpha=0.65, zorder=2)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E74C3C",
               markersize=10, markeredgecolor="#922B21", markeredgewidth=1.5,
               label="DXI-TWNCR$_1$ / DXI-TWNCR$_K$ (Proposed)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980B9",
               markersize=8, markeredgecolor="#1A5276", markeredgewidth=1.0,
               label="Baseline methods"),
        Line2D([0], [0], color="#5DADE2", linewidth=8, alpha=0.80,
               label="Non-significant groups (rank diff < CD)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.85, edgecolor="#BDC3C7", ncol=1, bbox_to_anchor=(0.99, 0.99))

    stats_text = (r"$\mathbf{Friedman\ test:}$"
                  f"\n$\chi^2({k-1}) = {chi2_stat:.2f}$"
                  f"\n$p$ = " + (f"$<10^{{-9}}$" if p < 1e-9 else f"${p:.2e}$") +
                  f"\n$k = {k}$ methods, $N = {n}$ datasets"
                  f"\nNemenyi CD = {cd:.3f} ($\\alpha$ = 0.05)")
    ax.text(0.01, 0.96, stats_text, transform=ax.transAxes, fontsize=7.8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FAFAFA",
                      edgecolor="#CCCCCC", linewidth=0.6, alpha=0.92),
            family="monospace")

    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print(f"  CD 图 → {out_path}")


if __name__ == "__main__":
    auc_mat = build_auc_ap("AUC")
    ap_mat = build_auc_ap("AP")
    f1_mat = build_f1()
    print(f"数据集（{len(auc_mat)}）: {list(auc_mat.index)}")

    process("AUC", auc_mat,
            os.path.join(BASE, "Significance_incrk_AUC.csv"),
            os.path.join(FIG_DIR, "nemenyi_cd_incrk_auc.pdf"))
    process("PR", ap_mat,
            os.path.join(BASE, "Significance_incrk_PR.csv"),
            os.path.join(FIG_DIR, "nemenyi_cd_incrk_pr.pdf"))
    process("F1", f1_mat,
            os.path.join(BASE, "Significance_incrk_F1.csv"),
            os.path.join(FIG_DIR, "nemenyi_cd_incrk_f1.pdf"))