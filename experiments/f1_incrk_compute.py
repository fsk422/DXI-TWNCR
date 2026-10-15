# -*- coding: utf-8 -*-
"""F1 — ITW-NCR / DXI-TWNCR_K (IncrK batch incremental path).

Copied from _f1_complete.py (the script that produced F1_complete.csv).
Only change: swap run_incr_1 -> run_incr_K, build_full_no_nom -> build_full
(needed to obtain NOM_full for the batch path). Baselines are NOT recomputed
— their F1 already exists in F1_complete.csv.

Checkpoint: _f1_incrk_ckpt.csv  |  Output: F1_incrk.csv
"""
import numpy as np, scipy.io as sio, sys, os, time, warnings, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score
import importlib.util as iu

SEED, BETA, MAX_EVAL, N_MAX_CAP = 42, 0.67, 3000, 6000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(OUT_DIR, "_f1_incrk_ckpt.csv")

# ── Import from incr_core ──
spec = iu.spec_from_file_location("incr", os.path.join(OUT_DIR, "incr_core.py"))
incr = iu.module_from_spec(spec); spec.loader.exec_module(incr)
load_ds, stratified_sample = incr.load_ds, incr.stratified_sample
build_full, run_incr_K = incr.build_full, incr.run_incr_K

DATASETS = [
    ("creditA","Norm","Mixed_creditA_425_15_42_Norm.mat","T1"),
    ("german","Norm","Mixed_german_714_20_14_Norm.mat","T1"),
    ("vote","Norm","Categorical_vote_296_16_29_Norm.mat","T1"),
    ("heart","Norm","Mixed_heart_166_13_16_Norm.mat","T1"),
    ("thyroid","Norm","Numeric_thyroid_3772_6_93_Norm.mat","T2"),
    ("mushroom_01","Norm","Categorical_mushroom_01_4293_22_85_Norm.mat","T2"),
    ("cardio","Norm","Numeric_cardio_1831_21_176_Norm.mat","T2"),
    ("pageblocks","Norm","Numeric_pageblocks_5171_10_258_Norm.mat","T3"),
    ("pendigits","Norm","Numeric_pendigits_6870_16_156_Norm.mat","T3"),
    ("annthyroid","Norm","Numeric_annthyroid_7200_6_534_Norm.mat","T3"),
    ("mammography","Norm","Numeric_mammography_11183_6_260_Norm.mat","T4"),
    ("adult_01","High","Mixed_adult_01_34357_14_343_Norm.mat","T4"),
    ("shuttle","High","Numeric_shuttle_49097_9_3511_Norm.mat","T4"),
    ("forestcover","High","Numeric_forestcover_286048_10_2747_Norm.mat","T5"),
]
TIER_PARAMS = {"T1":{"alpha":0.60,"P0":"N"},"T2":{"alpha":0.60,"P0":"N"},
               "T3":{"alpha":0.60,"P0":5000},"T4":{"alpha":0.40,"P0":8000},
               "T5":{"alpha":0.30,"P0":15000}}

def make_split(sub, fname, tier, rng):
    """Identical to incr_core.run_one split logic (and _f1_complete.py)."""
    tp = TIER_PARAMS[tier]; alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname); N, d = X_all.shape
    P0_val = N if tp["P0"]=="N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, rng)
    Np0 = len(X_p0); U_size = max(50, int(Np0*alpha))
    x_size = int(U_size*(1-BETA)); p1_size = U_size-x_size; p3_size = max(20, Np0-U_size)
    mask_p0 = np.zeros(N, bool); mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]; y4 = y_all[~mask_p0]
    U = X_p0[:U_size]; Xidx_init = np.arange(p1_size, U_size)
    eval_X = np.vstack([X_p0[U_size:U_size+p3_size], X4]) if len(X4)>0 else X_p0[U_size:U_size+p3_size]
    eval_y = np.concatenate([y_p0[U_size:U_size+p3_size], y4]) if len(y4)>0 else y_p0[U_size:U_size+p3_size]
    n_full = len(eval_X); cap = min(MAX_EVAL, n_full)
    if cap < n_full:
        sr = np.random.RandomState(SEED); eval_X, eval_y, _ = stratified_sample(eval_X, eval_y, cap, sr)
    return N, d, y_all.sum()/N, U_size, x_size, len(eval_X), U, eval_X, eval_y, p1_size, Xidx_init

def compute_f1(scores, y, AR):
    thr = np.percentile(scores, 100*(1-AR)); pred = (scores > thr).astype(int)
    return f1_score(y, pred, zero_division=0)

# ── Checkpoint ──
if os.path.exists(CKPT):
    ckpt_rows = pd.read_csv(CKPT).to_dict("records")
    done_ds = {r["dataset"] for r in ckpt_rows}
else:
    ckpt_rows, done_ds = [], set()

global_rng = np.random.RandomState(SEED)

for ds_name, sub, fname, tier in DATASETS:
    if ds_name in done_ds:
        # replay RNG for consistency
        Xa, ya = load_ds(sub, fname); N, d = Xa.shape
        tp = TIER_PARAMS[tier]; P0v = N if tp["P0"]=="N" else min(tp["P0"], N)
        stratified_sample(Xa, ya, P0v, global_rng)
        print(f"  [{ds_name}] SKIP"); continue

    print(f"\n{'='*60}\n  {ds_name}  ({tier})\n{'='*60}")
    try:
        N, d, AR, U_size, x_size, n_eval, U, eval_X, eval_y, p1_size, Xidx_init = \
            make_split(sub, fname, tier, global_rng)
    except Exception as e:
        print(f"  SPLIT ERROR: {e}"); import traceback; traceback.print_exc(); continue

    K_val = max(1, int(x_size*0.02)); n_max = min(int(1.1*U_size), N_MAX_CAP)
    print(f"  N={N} d={d} AR={AR:.4f} U={U_size} X={x_size} K={K_val} n_max={n_max} Eval={n_eval} na={eval_y.sum()}")

    row = {"dataset": ds_name, "N": N, "d": d, "AR": AR, "tier": tier,
           "U": U_size, "X0": x_size, "K": K_val, "n_max": n_max, "n_eval": n_eval}

    # ── ITW-NCR IncrK (exact path as AUC experiment run_incr_K) ──
    t0 = time.perf_counter()
    data_U, Lower, NEB, NPB, NOM_full, I, delta, m_out, Xt, nX = build_full(U.copy(), Xidx_init.copy())
    t_build = (time.perf_counter()-t0)*1000
    print(f"  Base build: {t_build:.0f}ms  m={m_out}  nX={nX}")

    try:
        t0 = time.perf_counter()
        _, _, _, _, scores_incrK = run_incr_K(U.copy(), Xidx_init.copy(), eval_X, eval_y,
                                              n_max, m_out, delta.copy(), K_val, NOM_full)
        f1_val = compute_f1(scores_incrK, eval_y, AR)
        t_elapsed = time.perf_counter()-t0
        print(f"  {'ITW-NCR IncrK':<20s} F1={f1_val:.4f}  t={t_elapsed:.1f}s  ({t_elapsed*1000/n_eval:.1f}ms/pt)")
        row["F1_ITW-NCR_K"] = f1_val
    except Exception as e:
        print(f"  {'ITW-NCR IncrK':<20s} ERROR: {e}"); import traceback; traceback.print_exc()
        row["F1_ITW-NCR_K"] = np.nan

    ckpt_rows.append(row)
    pd.DataFrame(ckpt_rows).to_csv(CKPT, index=False)
    print(f"  -> checkpoint ({len(ckpt_rows)}/14 datasets)")

# ── Final output ──
df = pd.DataFrame(ckpt_rows)
out_csv = os.path.join(OUT_DIR, "F1_incrk.csv")
df.to_csv(out_csv, index=False)
print(f"\n{'='*60}")
print(df[["dataset","K","AR","F1_ITW-NCR_K"]].to_string(index=False))
print(f"Mean F1_ITW-NCR_K = {df['F1_ITW-NCR_K'].mean():.4f}")
print(f"DONE. {len(ckpt_rows)} datasets -> F1_incrk.csv")