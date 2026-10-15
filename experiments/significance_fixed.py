# -*- coding: utf-8 -*-
"""用修复版 IncLOF/HST 覆盖后重跑 Friedman+Nemenyi（k=12, N=14）。

仅后处理（不重跑模型、不改动原始 baseline 记录）：
  - 读 BaselinesFixed.csv 的 AUC/AP/F1，覆盖 baseline 矩阵里的 IncLOF/HST；
  - 其余 8 个 baseline 与 DXI-TWNCR1/K 保持原值；
  - 重算秩次、χ²、p、CD，重绘经典 Demšar CD 图（覆盖 paper/figures 下的三张图）。

数据源口径与 significance.py 完全一致，仅 IncLOF/HST 两列替换为修复版。
"""
import sys, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from significance import (
    friedman_nemenyi, DXI1, DXIK, BASELINES, base_col, fmt_p)
from gen_cd import draw_classic

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(BASE), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

INCR_CSV = os.path.join(BASE, "IncrCore_checkpoint_OVERNIGHT.csv")
BASELINE_CSV = os.path.join(BASE, "BaselinesUnified_checkpoint.csv")
FIXED_CSV = os.path.join(BASE, "BaselinesFixed.csv")
F1_CSV = os.path.join(BASE, "F1_complete.csv")
F1_INCRK_CSV = os.path.join(BASE, "F1_incrk.csv")

METHODS = [DXI1, DXIK] + BASELINES


def build_auc_ap(prefix):
    incr = pd.read_csv(INCR_CSV).set_index("dataset")
    base = pd.read_csv(BASELINE_CSV).set_index("dataset")
    fixed = pd.read_csv(FIXED_CSV).set_index("dataset")
    datasets = list(incr.index)
    data = {
        DXI1: incr.loc[datasets, f"{prefix}_Incr1"].astype(float).values,
        DXIK: incr.loc[datasets, f"{prefix}_IncrK"].astype(float).values,
    }
    for m in BASELINES:
        if m == "IncLOF":
            data[m] = fixed.loc[datasets, f"{prefix}_IncLOF_fix"].astype(float).values
        elif m == "HST":
            data[m] = fixed.loc[datasets, f"{prefix}_HST_fix"].astype(float).values
        else:
            data[m] = base.loc[datasets, base_col(prefix, m)].astype(float).values
    return pd.DataFrame(data, index=datasets)


def build_f1():
    f1 = pd.read_csv(F1_CSV).set_index("dataset")
    incrk = pd.read_csv(F1_INCRK_CSV).set_index("dataset")
    fixed = pd.read_csv(FIXED_CSV).set_index("dataset")
    datasets = list(f1.index)
    data = {
        DXI1: f1.loc[datasets, "F1_ITW-NCR"].astype(float).values,
        DXIK: incrk.loc[datasets, "F1_ITW-NCR_K"].astype(float).values,
    }
    for m in BASELINES:
        if m == "IncLOF":
            data[m] = fixed.loc[datasets, "F1_IncLOF_fix"].astype(float).values
        elif m == "HST":
            data[m] = fixed.loc[datasets, "F1_HST_fix"].astype(float).values
        else:
            data[m] = f1.loc[datasets, base_col("F1", m)].astype(float).values
    return pd.DataFrame(data, index=datasets)


def process(metric_key, mat, out_csv, out_pdf):
    N, k = mat.shape
    avg_ranks, chi2_stat, p, cd, ranks, q = friedman_nemenyi(mat)
    cols = list(mat.columns)
    order = np.argsort(avg_ranks)
    rows = []
    for j in order:
        m = cols[j]
        rows.append({"method": m, "mean_metric": float(mat[m].mean()),
                     "avg_rank": float(avg_ranks[j])})
    df = pd.DataFrame(rows)
    last_rank = float(avg_ranks[order[-1]])
    df["sig_vs_worst"] = df["avg_rank"].apply(lambda r: (last_rank - r) > cd)
    df["chi2"] = chi2_stat
    df["p"] = p
    df["CD"] = cd
    df["q"] = q
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*76}")
    print(f"  {metric_key} — Friedman/Nemenyi（k={k}, N={N}, q={q:.4f}, CD={cd:.4f}）")
    print(f"  χ² = {chi2_stat:.4f}   p = {p:.3e}")
    print(f"{'='*76}")
    print(f"  {'顺位':<4s}{'方法':<18s}{'均值':>10s}{'平均秩':>8s}  显著优于末位")
    for pos, (_, r) in enumerate(df.iterrows(), 1):
        mark = "†" if r["sig_vs_worst"] else ""
        print(f"  {pos:<4d}{r['method']:<18s}{r['mean_metric']:>10.4f}{r['avg_rank']:>8.3f}  {mark}")
    worst = cols[order[-1]]
    print(f"  >> 末位方法：{worst}（rank={last_rank:.3f}）")
    for v in (DXI1, DXIK):
        rv = df[df["method"] == v]["avg_rank"].values[0]
        posv = int(np.where(order == cols.index(v))[0][0]) + 1
        print(f"  >> {v}: 平均秩 = {rv:.3f}（第 {posv} 位）")

    # LaTeX 表体（供论文使用）
    lines = []
    for _, r in df.iterrows():
        dag = r"$^{\dagger}$" if (last_rank - r["avg_rank"]) > cd else ""
        name = r["method"]
        if name in (DXI1, DXIK):
            lines.append(r"\textbf{" + name + r"} & \textbf{" + f"{r['avg_rank']:.2f}" + "}" + dag + r"\\")
        else:
            lines.append(name + " & " + f"{r['avg_rank']:.2f}" + dag + r"\\")
    body = "\n".join(lines)
    footer = (r"\multicolumn{2}{l}{\footnotesize $\chi^2=" + f"{chi2_stat:.2f}" +
              r"$, " + fmt_p(p) + r", CD$=" + f"{cd:.3f}" +
              r"$ ($\alpha=0.05$, $k=12$, $N=14$)}\\" + "\n" +
              r"\multicolumn{2}{l}{\footnotesize $^{\dagger}$ 表示与末位方法的秩次差超过 CD，差异显著。}")
    print(f"\n  --- LaTeX 表体 ---\n{body}\n{footer}")

    names = df["method"].astype(str).tolist()
    ranks_ = df["avg_rank"].astype(float).tolist()
    draw_classic(names, ranks_, cd, metric_key, out_pdf)
    return df


if __name__ == "__main__":
    auc_mat = build_auc_ap("AUC")
    ap_mat = build_auc_ap("AP")
    f1_mat = build_f1()
    print(f"数据集（{len(auc_mat)}）：{list(auc_mat.index)}")

    # —— 两个修复算法三指标结果表 ——
    fixed = pd.read_csv(FIXED_CSV).set_index("dataset")
    print("\n" + "=" * 76)
    print("  修复版 IncLOF / HST 三指标结果（AUC / AP / F1）")
    print("=" * 76)
    print(f"  {'dataset':<14s}{'AUC_ILOF':>10s}{'AP_ILOF':>10s}{'F1_ILOF':>10s}"
          f"{'AUC_HST':>10s}{'AP_HST':>10s}{'F1_HST':>10s}")
    for ds in fixed.index:
        r = fixed.loc[ds]
        print(f"  {ds:<14s}{r['AUC_IncLOF_fix']:>10.4f}{r['AP_IncLOF_fix']:>10.4f}"
              f"{r['F1_IncLOF_fix']:>10.4f}{r['AUC_HST_fix']:>10.4f}"
              f"{r['AP_HST_fix']:>10.4f}{r['F1_HST_fix']:>10.4f}")
    print("  " + "-" * 76)
    print(f"  {'MEAN':<14s}{fixed['AUC_IncLOF_fix'].mean():>10.4f}"
          f"{fixed['AP_IncLOF_fix'].mean():>10.4f}{fixed['F1_IncLOF_fix'].mean():>10.4f}"
          f"{fixed['AUC_HST_fix'].mean():>10.4f}{fixed['AP_HST_fix'].mean():>10.4f}"
          f"{fixed['F1_HST_fix'].mean():>10.4f}")

    process("AUC", auc_mat,
            os.path.join(BASE, "Significance_incrk_fixed_AUC.csv"),
            os.path.join(FIG_DIR, "nemenyi_cd.pdf"))
    process("PR", ap_mat,
            os.path.join(BASE, "Significance_incrk_fixed_PR.csv"),
            os.path.join(FIG_DIR, "nemenyi_cd_pr.pdf"))
    process("F1", f1_mat,
            os.path.join(BASE, "Significance_incrk_fixed_F1.csv"),
            os.path.join(FIG_DIR, "nemenyi_cd_f1.pdf"))