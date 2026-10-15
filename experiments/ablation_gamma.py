"""§8.5 γ 消融（新增脚本，2026-09-13）。

来源：本脚本复制 `experiments/drift.py` 的漂移数据划分 + Incr1 增量管线，
      将遗忘强度 γ 参数化：n_max = int((1+γ)·U₀)，去掉硬编码 1.1 与 6000 cap。
差异（相对 drift.py）：
  - 仅跑 Incr1 一种变体（DXI-TWNCR）。
  - γ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}（6 档，0.05 步长）；γ=0.10 复用
    `DRIFT_checkpoint.csv` 的 Incr1 L0/L3 现成数据（n_max<6000，cap 不影响），故新跑 5 档。
  - 只跑 L0（无漂移）与 L3（强漂移）两档。
  - 遗忘复用 `incr_core.phase_c_reset`（已修复版，`xo[:dx]`），经 drift.run_incr1 调用。
硬约束：旧文件 `drift.py` / `incr_core.py` 零改动。

数据集：thyroid(T2, U₀=2263)、shuttle(T4, U₀=3200)。
场景：L0 + L3。指标：sw_post AUC（滑窗 300）。
规模：5 档新跑 × 2 数据集 × 2 场景 = 20 次。
输出：GammaAblation_checkpoint.csv（逐档落盘）+ GammaAblation.xlsx。
"""
import numpy as np, sys, os, time, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from drift import (
    load_ds, stratified_sample, build_full_no_nom, apply_drift,
    sliding_window_auc, run_incr1,
)

SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(OUT_DIR, "GammaAblation_checkpoint.csv")
BETA = 0.67
WINDOW = 300

GAMMA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
REUSE_GAMMA = {0.10}  # 复用 DRIFT_checkpoint.csv 的 Incr1 L0/L3

DRIFT_LEVELS = [("L0", 0.0, 0.0), ("L3", 1.0, 1.5)]

DATASETS = [
    ("thyroid", "Norm", "Numeric_thyroid_3772_6_93_Norm.mat", "T2"),
    ("shuttle",  "High", "Numeric_shuttle_49097_9_3511_Norm.mat", "T4"),
]
TIER_PARAMS = {"T2": {"alpha": 0.60, "P0": "N"}, "T4": {"alpha": 0.40, "P0": 8000}}


def load_ckpt():
    if os.path.exists(CKPT_FILE):
        try:
            return pd.read_csv(CKPT_FILE)
        except Exception:
            pass
    return None


def save_ckpt(rows):
    tmp = CKPT_FILE + f".tmp.{os.getpid()}"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, CKPT_FILE)


REUSE_META = {
    "thyroid": {"tier": "T2", "U0": 2263},
    "shuttle":  {"tier": "T4", "U0": 3200},
}


def load_reuse_gamma():
    """从 DRIFT_checkpoint.csv 提取 Incr1 L0/L3 现成数据，映射为 γ=0.10 复用行。

    DRIFT_checkpoint 无 n_resets 列，故复用行的 n_resets 记为 NaN（γ 消融指标不含 n_resets）。
    """
    dpath = os.path.join(OUT_DIR, "DRIFT_checkpoint.csv")
    if not os.path.exists(dpath):
        return []
    dft = pd.read_csv(dpath)
    dft = dft[dft["method"] == "Incr1"]
    rows = []
    for ds, meta in REUSE_META.items():
        for lvl in ["L0", "L3"]:
            sub = dft[(dft["dataset"] == ds) & (dft["level"] == lvl)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            n_max = int((1 + 0.10) * meta["U0"])
            rows.append({
                "dataset": ds, "tier": meta["tier"], "gamma": 0.10, "level": lvl,
                "U0": meta["U0"], "n_max": n_max,
                "AUC_pre": float(r["AUC_pre"]), "AUC_post": float(r["AUC_post"]),
                "sw_pre": float(r["sw_pre"]), "sw_post": float(r["sw_post"]),
                "n_resets": np.nan, "t_ms": float(r["t_ms"]),
            })
    return rows


def run_one_ds(name, sub, fname, tier, global_rng):
    print(f"\n{'='*60}\n  {name}  (Tier={tier})\n{'='*60}", flush=True)
    tp = TIER_PARAMS[tier]
    alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname)
    N, d = X_all.shape

    P0_val = N if tp["P0"] == "N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, global_rng)
    Np0 = len(X_p0)
    U_size = max(50, int(Np0 * alpha))
    x_size = int(U_size * (1 - BETA))
    p1_size = U_size - x_size
    p3_size = max(20, Np0 - U_size)
    mask_p0 = np.zeros(N, bool)
    mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]
    y4 = y_all[~mask_p0]
    U_init = X_p0[:U_size]
    Xidx_init = np.arange(p1_size, U_size)
    S_X = np.vstack([X_p0[U_size:U_size + p3_size], X4]) if len(X4) > 0 else X_p0[U_size:U_size + p3_size]
    S_y = np.concatenate([y_p0[U_size:U_size + p3_size], y4]) if len(y4) > 0 else y_p0[U_size:U_size + p3_size]
    mid = len(S_X) // 2
    S1_X = S_X[:mid]
    S1_y = S_y[:mid]
    S2_o = S_X[mid:]
    S2_y = S_y[mid:]

    dU, Lo, NE, NPB, I, de, m, Xt, nX = build_full_no_nom(U_init, Xidx_init)
    print(f"  N={N} d={d} U={U_size} P1={p1_size} X={x_size} T={len(S_X)} drift@{mid}", flush=True)

    rows = []
    for gamma in GAMMA_GRID:
        if gamma in REUSE_GAMMA:
            print(f"  γ={gamma:.2f} SKIP — 复用 DRIFT_checkpoint.csv", flush=True)
            continue
        n_max = int((1 + gamma) * U_size)
        for lvl, sigma, shift in DRIFT_LEVELS:
            S2 = S2_o.copy() if lvl == "L0" else apply_drift(S2_o, sigma, shift, d)
            sX = np.vstack([S1_X, S2])
            s_y = np.concatenate([S1_y, S2_y])
            t0 = time.perf_counter()
            sc, nr = run_incr1(dU.copy(), Xidx_init.copy(), nX, Lo.copy(), NE.copy(),
                               NPB.copy(), I.copy(), de, m, sX, n_max, True)
            el_ms = (time.perf_counter() - t0) * 1000
            ap = roc_auc_score(s_y[:mid], sc[:mid]) if len(np.unique(s_y[:mid])) >= 2 else np.nan
            apo = roc_auc_score(s_y[mid:], sc[mid:]) if len(np.unique(s_y[mid:])) >= 2 else np.nan
            sw = sliding_window_auc(sc, s_y, WINDOW)
            swp = float(np.nanmean(sw[:mid]))
            swpo = float(np.nanmean(sw[mid:]))
            print(f"    γ={gamma:.2f} {lvl}: pre={ap:.4f} post={apo:.4f} "
                  f"sw_pre={swp:.4f} sw_post={swpo:.4f} resets={nr} t={el_ms:.0f}ms", flush=True)
            rows.append({
                "dataset": name, "tier": tier, "gamma": gamma, "level": lvl,
                "U0": U_size, "n_max": n_max,
                "AUC_pre": ap, "AUC_post": apo, "sw_pre": swp, "sw_post": swpo,
                "n_resets": nr, "t_ms": el_ms,
            })
    return rows


if __name__ == "__main__":
    ckpt = load_ckpt()
    all_rows = [] if ckpt is None else ckpt.to_dict("records")

    done_set = set()
    if ckpt is not None:
        for _, row in ckpt.iterrows():
            done_set.add((row["dataset"], f"{row['gamma']:.2f}", row["level"]))

    # 复用 γ=0.10（来自 DRIFT_checkpoint.csv 的 Incr1 L0/L3）
    for r in load_reuse_gamma():
        key = (r["dataset"], f"{r['gamma']:.2f}", r["level"])
        if key not in done_set:
            all_rows.append(r)
            done_set.add(key)
    save_ckpt(all_rows)

    global_rng = np.random.RandomState(SEED)
    for name, sub, fname, tier in DATASETS:
        rows = []
        try:
            rows = run_one_ds(name, sub, fname, tier, global_rng)
        except Exception as e:
            print(f"  ERROR {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        for r in rows:
            key = (r["dataset"], f"{r['gamma']:.2f}", r["level"])
            if key in done_set:
                continue
            all_rows.append(r)
            done_set.add(key)
        save_ckpt(all_rows)
        print(f"  [{name}] DONE — {len(rows)} rows", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_excel(os.path.join(OUT_DIR, "GammaAblation.xlsx"), index=False)

    print(f"\n{'='*70}")
    print("  γ 消融 — 汇总（sw_post AUC）")
    print(f"{'='*70}")
    for ds in [d[0] for d in DATASETS]:
        for lvl in ["L0", "L3"]:
            sub = df[(df.dataset == ds) & (df.level == lvl)].sort_values("gamma")
            vals = " ".join(f"γ{g:.2f}={r:.4f}" for g, r in zip(sub["gamma"], sub["sw_post"]))
            print(f"  {ds:<9s} {lvl}: {vals}")

    print(f"\n  Done. {len(df)} runs. Checkpoint: {CKPT_FILE}")