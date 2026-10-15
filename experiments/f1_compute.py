# -*- coding: utf-8 -*-
"""
F1 Score Computation — V3: incremental path for ITW-NCR speed.
Uses partial_fit + predict (Incr1-style), same as incr_core experiment.
"""
import numpy as np, scipy.io as sio, sys, os, time, warnings, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score

SEED = 42; BETA = 0.67; MAX_EVAL = 3000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_FILE = os.path.join(OUT_DIR, "F1_v3_checkpoint.csv")

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

def load_ds(sub, fname):
    fp = os.path.join(BASE, "DATASETS", sub, fname)
    m = sio.loadmat(fp); d = m["dataset"]
    X = d[:,:-1].astype(np.float64); y = (d[:,-1].ravel()>=0.5).astype(np.int64)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    return X, y

def stratified_sample(X, y, n_out, rng):
    N = len(X); n_out = min(n_out, N)
    idx_n = rng.permutation(np.where(y==0)[0]); idx_a = rng.permutation(np.where(y==1)[0])
    na, nn = len(idx_a), len(idx_n); r = na/N if N>0 else 0
    n_a = min(na, max(1, int(n_out*r))); n_n = min(nn, n_out-n_a)
    sel = rng.permutation(np.concatenate([idx_a[:n_a], idx_n[:n_n]]))
    return X[sel], y[sel], sel

def make_split(sub, fname, tier, rng):
    tp = TIER_PARAMS[tier]; alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname); N, d = X_all.shape
    P0_val = N if tp["P0"]=="N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, rng)
    Np0 = len(X_p0); U_size = max(50, int(Np0*alpha))
    x_size = int(U_size*(1-BETA)); p1_size = U_size-x_size; p3_size = max(20, Np0-U_size)
    mask_p0 = np.zeros(N,bool); mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]; y4 = y_all[~mask_p0]
    U = X_p0[:U_size]
    eval_X = np.vstack([X_p0[U_size:U_size+p3_size], X4]) if len(X4)>0 else X_p0[U_size:U_size+p3_size]
    eval_y = np.concatenate([y_p0[U_size:U_size+p3_size], y4]) if len(y4)>0 else y_p0[U_size:U_size+p3_size]
    cap = min(MAX_EVAL, len(eval_X))
    if cap < len(eval_X):
        sr = np.random.RandomState(SEED+1); eval_X, eval_y, _ = stratified_sample(eval_X, eval_y, cap, sr)
    return N, d, y_all.sum()/N, U_size, x_size, len(eval_X), U, eval_X, eval_y, p1_size

def compute_f1(scores, y, AR):
    thr = np.percentile(scores, 100*(1-AR)); pred = (scores > thr).astype(int)
    return f1_score(y, pred, zero_division=0)

def load_ckpt():
    if os.path.exists(CKPT_FILE):
        try: return pd.read_csv(CKPT_FILE)
        except: pass
    return None

def save_ckpt(rows):
    pd.DataFrame(rows).to_csv(CKPT_FILE, index=False)

def build_baselines():
    from src.detectors.baselines import (
        IncrementalLOF, HalfSpaceTrees, RRCFWrapper, XStreamDetector,
        LODAIncremental, HBOSIncremental, ECODIncremental,
        BWOAIFDIFFI, EILOFDetector, SiForestDetector,
    )
    return {
        "IncLOF": lambda: IncrementalLOF(window_size=128, n_neighbors=10),
        "HST": lambda: HalfSpaceTrees(n_trees=25, max_depth=10),
        "RRCF": lambda: RRCFWrapper(n_trees=20),
        "xStream": lambda: XStreamDetector(n_components=50, n_hash_fns=10),
        "LODA": lambda: LODAIncremental(n_random_cuts=100, n_bins=10),
        "EILOF": lambda: EILOFDetector(window_size=256, n_neighbors=20),
        "SiForest": lambda: SiForestDetector(n_trees=50, sample_size=128, window_size=1024),
        "HBOS": lambda: HBOSIncremental(),
        "ECOD": lambda: ECODIncremental(),
        "BWOAIF+DIFFI": lambda: BWOAIFDIFFI(n_trees=100, sample_size=256, window_size=512),
    }

if __name__ == "__main__":
    ckpt = load_ckpt()
    baselines = build_baselines()
    global_rng = np.random.RandomState(SEED)
    all_rows = []

    for ds_name, sub, fname, tier in DATASETS:
        if ckpt is not None and ds_name in ckpt["dataset"].values:
            prev = ckpt[ckpt["dataset"]==ds_name]
            if "F1_ITW-NCR" in prev.columns and prev["F1_ITW-NCR"].notna().iloc[0]:
                print(f"  [{ds_name}] SKIP")
                all_rows.append(prev.iloc[0].to_dict())
                continue

        print(f"\n{'='*60}\n  {ds_name}  ({tier})\n{'='*60}")
        try:
            N, d, AR, U_size, x_size, n_eval, U, eval_X, eval_y, p1_size = \
                make_split(sub, fname, tier, global_rng)
        except Exception as e:
            print(f"  SPLIT ERROR: {e}"); continue
        print(f"  N={N} d={d} AR={AR:.4f} U={U_size} Eval={n_eval} na_eval={eval_y.sum()}")
        row = {"dataset": ds_name, "N": N, "d": d, "AR": AR, "tier": tier}

        # --- ITW-NCR: INC1 path (partial_fit then predict, 1000x faster) ---
        try:
            from src.detectors.ix_wncrod import IXWNCROD
            t0 = time.perf_counter()
            det = IXWNCROD(lammda=1.0, fast_update=True, batch_size=1, optimize_level=0)
            det.fit(U)
            scores = np.zeros(n_eval)
            for i in range(n_eval):
                det.partial_fit(eval_X[i:i+1])
                s = det.predict(eval_X[i:i+1])
                scores[i] = s[0]
            f1_val = compute_f1(scores, eval_y, AR)
            t_total = time.perf_counter() - t0
            avg_ms = t_total * 1000 / n_eval
            print(f"  {'ITW-NCR':<16s} F1={f1_val:.4f} t={t_total:.1f}s ({avg_ms:.1f}ms/pt)")
            row["F1_ITW-NCR"] = f1_val
        except Exception as e:
            print(f"  {'ITW-NCR':<16s} ERROR: {e}"); row["F1_ITW-NCR"] = np.nan

        # --- Baselines ---
        for bname, bfn in baselines.items():
            try:
                t0 = time.perf_counter()
                det = bfn(); det.fit(U)
                scores = np.zeros(n_eval)
                for i in range(n_eval):
                    scores[i] = det.predict(eval_X[i:i+1]).ravel()[0]
                f1_val = compute_f1(scores, eval_y, AR)
                t_total = time.perf_counter() - t0
                print(f"  {bname:<16s} F1={f1_val:.4f} t={t_total:.1f}s")
                row[f"F1_{bname}"] = f1_val
            except Exception as e:
                print(f"  {bname:<16s} ERROR: {e}"); row[f"F1_{bname}"] = np.nan

        all_rows.append(row); save_ckpt(all_rows)
        print(f"  -> checkpoint ({len(all_rows)} ds)")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "F1_final.csv"), index=False)
    df.to_excel(os.path.join(OUT_DIR, "F1_final.xlsx"), index=False)
    print(f"\nDONE. {len(all_rows)} datasets -> F1_final.xlsx")
