"""§8.5 β 消融（新增脚本，2026-09-13）。

来源：本脚本复制 `experiments/incr_core.py` 的数据划分 + `run_incr_1` 增量管线，
      将 β（背景占比）由固定 0.67 参数化为 0.05 步长网格扫描。
差异（相对 incr_core.py）：
  - 仅跑 Incr1 一种变体（DXI-TWNCR）。
  - β = |P1|/|U| ∈ {0.20, 0.25, …, 0.80}（13 档），X_size = int(U·(1-β))，
    P1 = U - X_size（沿用主实验取整锚点）。
  - 固定 λ=1.0（delta=std，不除以 λ）、K=1、p1x_sep=True、n_max=min(1.1·U₀, 6000)。
  - reset/遗忘复用 `incr_core.phase_c_reset`（已修复版 `xo[:dx]` 降级最早对象）。
硬约束：不得使用 `ablation_lambda.py` 内置的旧 `phase_c_reset`（`xo[-dx:]` 降级方向 bug）。

数据集：creditA(T1) / mushroom_01(T2) / annthyroid(T3) / adult_01(T4)。共 13×4=52 次。
指标：AUC-ROC（主）、AUC-PR（附）、n_resets、avg_update_ms。
输出：BetaAblation_checkpoint.csv（逐档落盘）+ BetaAblation.xlsx。
"""
import numpy as np, sys, os, time, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from incr_core import (
    load_ds, stratified_sample, compute_auc, run_incr_1,
)

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(OUT_DIR, "BetaAblation_checkpoint.csv")
N_MAX_CAP = 6000
MAX_EVAL = 3000

BETA_GRID = list(np.round(np.arange(0.20, 0.805, 0.05), 2))  # 13 档

DATASETS = [
    ("creditA",      "Norm", "Mixed_creditA_425_15_42_Norm.mat",              "T1"),
    ("mushroom_01",  "Norm", "Categorical_mushroom_01_4293_22_85_Norm.mat",   "T2"),
    ("annthyroid",   "Norm", "Numeric_annthyroid_7200_6_534_Norm.mat",        "T3"),
    ("adult_01",     "High", "Mixed_adult_01_34357_14_343_Norm.mat",          "T4"),
]

TIER_PARAMS = {
    "T1": {"alpha": 0.60, "P0": "N"},
    "T2": {"alpha": 0.60, "P0": "N"},
    "T3": {"alpha": 0.60, "P0": 5000},
    "T4": {"alpha": 0.40, "P0": 8000},
}


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


def run_one_beta(ds_name, sub, fname, tier, beta, global_rng):
    tp = TIER_PARAMS[tier]
    alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname)
    N, d = X_all.shape
    na_all = y_all.sum()

    P0_val = N if tp["P0"] == "N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, global_rng)
    Np0 = len(X_p0)
    U_size = max(50, int(Np0 * alpha))
    x_size = int(U_size * (1.0 - beta))
    p1_size = U_size - x_size
    p3_size = max(20, Np0 - U_size)
    mask_p0 = np.zeros(N, bool)
    mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]
    y4 = y_all[~mask_p0]

    U = X_p0[:U_size]
    Xidx_init = np.arange(p1_size, U_size)

    eval_X = np.vstack([X_p0[U_size:U_size + p3_size], X4]) if len(X4) > 0 else X_p0[U_size:U_size + p3_size]
    eval_y = np.concatenate([y_p0[U_size:U_size + p3_size], y4]) if len(y4) > 0 else y_p0[U_size:U_size + p3_size]
    n_full = len(eval_X)

    cap = min(MAX_EVAL, n_full)
    if cap < n_full:
        sample_rng = np.random.RandomState(SEED)
        eval_X, eval_y, _ = stratified_sample(eval_X, eval_y, cap, sample_rng)
    n_eval = len(eval_X)

    n_max = min(int(1.1 * U_size), N_MAX_CAP)

    label = f"{beta:.2f}"
    print(f"  [β={label}] U={U_size} P1={p1_size} X={x_size} eval={n_eval}/{n_full} n_max={n_max}")

    t0 = time.perf_counter()
    auc, apr, avg_t, n_resets, _ = run_incr_1(U, Xidx_init, eval_X, eval_y, n_max, d, None)
    total_sec = time.perf_counter() - t0

    return {
        "dataset": ds_name, "tier": tier, "beta": beta,
        "N": N, "d": d, "AR": na_all / N,
        "P0": P0_val, "P1_size": p1_size, "X_size": x_size,
        "U_size": U_size, "n_eval": n_eval, "n_max": n_max,
        "auc_roc": auc, "auc_pr": apr,
        "avg_update_ms": avg_t, "n_resets": n_resets, "total_sec": total_sec,
    }


if __name__ == "__main__":
    ckpt_df = load_ckpt()
    all_rows = [] if ckpt_df is None else ckpt_df.to_dict("records")

    done_set = set()
    if ckpt_df is not None:
        for _, row in ckpt_df.iterrows():
            done_set.add((row["dataset"], f"{row['beta']:.2f}"))

    global_rng = np.random.RandomState(SEED)
    for ds_name, sub, fname, tier in DATASETS:
        for beta in BETA_GRID:
            label = f"{beta:.2f}"
            key = (ds_name, label)
            if key in done_set:
                print(f"  [{ds_name}|β={label}] SKIP — already done")
                continue
            try:
                print(f"\n  [{ds_name}|β={label}] RUNNING...")
                r = run_one_beta(ds_name, sub, fname, tier, beta, global_rng)
                all_rows.append(r)
                save_ckpt(all_rows)
                print(f"    AUC={r['auc_roc']:.4f}  AP={r['auc_pr']:.4f}  "
                      f"t={r['avg_update_ms']:.2f}ms/pt  resets={r['n_resets']}")
            except Exception as e:
                print(f"  ERROR on {ds_name}|β={label}: {e}")
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out_xlsx = os.path.join(OUT_DIR, "BetaAblation.xlsx")
    df.to_excel(out_xlsx, index=False)

    print(f"\n{'='*80}")
    print("  β 消融 — 汇总（AUC-ROC）")
    print(f"{'='*80}")
    for ds_name, _, _, _ in DATASETS:
        ddf = df[df["dataset"] == ds_name].sort_values("beta")
        line = f"  {ds_name:<12s} " + " ".join(f"{v:>7.4f}" for v in ddf["auc_roc"])
        print(line)

    print(f"\n  Done. {len(df)} runs. Output: {out_xlsx}")
    print(f"  Checkpoint: {CKPT_FILE}")