"""Unified Baseline Benchmark — same data split as incr_core.py.

Uses shared `global_rng` to ensure P1/X/P3/P4 split identical to six-variant experiment.
10 baselines on 11 datasets.  Checkpoint-enabled.
"""
import numpy as np, scipy.io as sio, sys, os, time, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

SEED = 42; BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(OUT_DIR, "BaselinesUnified_checkpoint.csv")
MAX_EVAL = 3000; BETA = 0.67

DATASETS = [
    ("creditA",      "Norm", "Mixed_creditA_425_15_42_Norm.mat",              "T1"),
    ("german",       "Norm", "Mixed_german_714_20_14_Norm.mat",               "T1"),
    ("thyroid",      "Norm", "Numeric_thyroid_3772_6_93_Norm.mat",            "T2"),
    ("mushroom_01",  "Norm", "Categorical_mushroom_01_4293_22_85_Norm.mat",   "T2"),
    ("pageblocks",   "Norm", "Numeric_pageblocks_5171_10_258_Norm.mat",       "T3"),
    ("pendigits",    "Norm", "Numeric_pendigits_6870_16_156_Norm.mat",        "T3"),
    ("annthyroid",   "Norm", "Numeric_annthyroid_7200_6_534_Norm.mat",        "T3"),
    ("mammography",  "Norm", "Numeric_mammography_11183_6_260_Norm.mat",      "T4"),
    ("adult_01",     "High", "Mixed_adult_01_34357_14_343_Norm.mat",          "T4"),
    ("shuttle",      "High", "Numeric_shuttle_49097_9_3511_Norm.mat",         "T4"),
    ("forestcover",  "High", "Numeric_forestcover_286048_10_2747_Norm.mat",   "T5"),
    # ——— Plan C additions ———
    ("vote",         "Norm", "Categorical_vote_296_16_29_Norm.mat",            "T1"),
    ("heart",        "Norm", "Mixed_heart_166_13_16_Norm.mat",                 "T1"),
    ("cardio",       "Norm", "Numeric_cardio_1831_21_176_Norm.mat",            "T2"),
]
TIER_PARAMS = {"T1":{"alpha":0.60,"P0":"N"},"T2":{"alpha":0.60,"P0":"N"},
               "T3":{"alpha":0.60,"P0":5000},"T4":{"alpha":0.40,"P0":8000},
               "T5":{"alpha":0.30,"P0":15000}}

def load_ds(sub, fname):
    fp = os.path.join(BASE, "DATASETS", sub, fname)
    m = sio.loadmat(fp); d = m["dataset"]
    X = d[:, :-1].astype(np.float64); y = (d[:, -1].ravel() >= 0.5).astype(np.int64)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    return X, y

def stratified_sample(X, y, n_out, rng):
    N = len(X); n_out = min(n_out, N)
    idx_n = rng.permutation(np.where(y == 0)[0])
    idx_a = rng.permutation(np.where(y == 1)[0])
    na, nn = len(idx_a), len(idx_n)
    r = na / N if N > 0 else 0
    n_a = min(na, max(1, int(n_out * r)))
    n_n = min(nn, n_out - n_a)
    sel = rng.permutation(np.concatenate([idx_a[:n_a], idx_n[:n_n]]))
    return X[sel], y[sel], sel

def compute_auc(s, l):
    if len(np.unique(l)) < 2: return float('nan'), float('nan')
    return roc_auc_score(l, s), average_precision_score(l, s)

def load_ckpt():
    if os.path.exists(CKPT_FILE):
        try: return pd.read_csv(CKPT_FILE)
        except: pass
    return None

def save_ckpt(rows):
    pd.DataFrame(rows).to_csv(CKPT_FILE, index=False)

def make_split(name, sub, fname, tier, rng):
    tp = TIER_PARAMS[tier]; alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname)
    N, d = X_all.shape; na_all = y_all.sum()

    P0_val = N if tp["P0"] == "N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, rng)
    Np0 = len(X_p0)
    U_size = max(50, int(Np0 * alpha))
    x_size = int(U_size * (1 - BETA))
    p1_size = U_size - x_size
    p3_size = max(20, Np0 - U_size)
    mask_p0 = np.zeros(N, bool); mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]; y4 = y_all[~mask_p0]

    U = X_p0[:U_size]; Xidx_init = np.arange(p1_size, U_size)
    eval_X = np.vstack([X_p0[U_size:U_size+p3_size], X4]) if len(X4)>0 else X_p0[U_size:U_size+p3_size]
    eval_y = np.concatenate([y_p0[U_size:U_size+p3_size], y4]) if len(y4)>0 else y_p0[U_size:U_size+p3_size]

    cap = min(MAX_EVAL, len(eval_X))
    if cap < len(eval_X):
        sample_rng = np.random.RandomState(SEED)
        eval_X, eval_y, _ = stratified_sample(eval_X, eval_y, cap, sample_rng)

    return N, d, na_all/N, U_size, x_size, len(eval_X), U, eval_X, eval_y, p1_size

def build_baselines():
    from src.detectors.baselines import (
        IncrementalLOF, HalfSpaceTrees, RRCFWrapper, XStreamDetector,
        LODAIncremental, HBOSIncremental, ECODIncremental,
        BWOAIFDIFFI, EILOFDetector, SiForestDetector,
    )
    return {
        "IncLOF":       lambda: IncrementalLOF(window_size=128, n_neighbors=10),
        "HST":          lambda: HalfSpaceTrees(n_trees=25, max_depth=10),
        "RRCF":         lambda: RRCFWrapper(n_trees=20),
        "xStream":      lambda: XStreamDetector(n_components=50, n_hash_fns=10),
        "LODA":         lambda: LODAIncremental(n_random_cuts=100, n_bins=10),
        "EILOF":        lambda: EILOFDetector(window_size=256, n_neighbors=20),
        "SiForest":     lambda: SiForestDetector(n_trees=50, sample_size=128, window_size=1024),
        "HBOS":         lambda: HBOSIncremental(),
        "ECOD":         lambda: ECODIncremental(),
        "BWOAIF+DIFFI": lambda: BWOAIFDIFFI(n_trees=100, sample_size=256, window_size=512),
    }

def run_one(name, sub, fname, tier, global_rng, baselines, ckpt):
    done_set = set()
    if ckpt is not None and name in ckpt["dataset"].values:
        prev = ckpt[ckpt["dataset"] == name].iloc[0]
        done_set = {c for c in baselines if f"AUC_{c}" in prev and not pd.isna(prev[f"AUC_{c}"])}
        if len(done_set) == len(baselines):
            print(f"  [{name}] SKIP — all {len(baselines)} baselines done")
            return [prev.to_dict()]

    print(f"\n{'='*65}\n  {name}  (Tier={tier})\n{'='*65}")

    N, d, AR, U_size, x_size, n_eval, U, eval_X, eval_y, p1_size = \
        make_split(name, sub, fname, tier, global_rng)
    print(f"  N={N}  d={d}  AR={AR:.4f}  U={U_size}  P1={p1_size}  X={x_size}  Eval={n_eval}")

    row = {"dataset": name, "N": N, "d": d, "AR": AR, "tier": tier,
           "U": U_size, "X0": x_size, "n_eval": n_eval}

    for bname, bfn in baselines.items():
        if bname in done_set:
            print(f"    {bname:<16s} SKIP (done)")
            continue

        try:
            t0 = time.perf_counter()
            det = bfn()
            det.fit(U)

            t0s = time.perf_counter()
            scores = np.zeros(len(eval_X))
            for ei in range(len(eval_X)):
                try:
                    s = det.predict(eval_X[ei:ei+1])[0]
                except Exception:
                    s = det.predict(eval_X[ei:ei+1]).ravel()[0]
                scores[ei] = s
            t_score = (time.perf_counter() - t0s) * 1000 / len(eval_X)
            auc, apr = compute_auc(scores, eval_y)
            print(f"    {bname:<16s} AUC={auc:.4f}  APR={apr:.4f}  t={t_score:.1f}ms/pt")
        except Exception as e:
            print(f"    {bname:<16s} ERROR: {e}")
            auc, apr, t_score = np.nan, np.nan, np.nan

        row[f"AUC_{bname}"] = auc; row[f"AP_{bname}"] = apr; row[f"t_{bname}_ms"] = t_score

    return [row]

if __name__ == "__main__":
    ckpt = load_ckpt()
    all_rows = []
    global_rng = np.random.RandomState(SEED)
    baselines = build_baselines()

    for name, sub, fname, tier in DATASETS:
        try:
            rows = run_one(name, sub, fname, tier, global_rng, baselines, ckpt)
            for r in rows:
                if r['dataset'] == name:
                    found = False
                    for i,ar in enumerate(all_rows):
                        if ar['dataset'] == name:
                            all_rows[i] = r; found = True; break
                    if not found: all_rows.append(r)
            save_ckpt(all_rows)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out_xlsx = os.path.join(OUT_DIR, "BaselinesUnified.xlsx")
    df.to_excel(out_xlsx, index=False)

    names = list(baselines.keys())
    print(f"\n{'='*130}")
    print("  BASELINES UNIFIED — SUMMARY")
    print(f"{'='*130}")
    hdr = f"  {'Dataset':<14s} " + " ".join(f"{n:>10s}" for n in names)
    print(hdr)
    print("  " + "-" * (14 + 10 * len(names)))
    for _, row in df.iterrows():
        aucs = [row.get(f"AUC_{n}", np.nan) for n in names]
        line = f"  {row['dataset']:<14s} " + " ".join(
            f"{a:>10.4f}" if not np.isnan(a) else f"{'---':>10s}" for a in aucs)
        print(line)

    print(f"\n  AVG AUC: " + " ".join(
        f"{df[f'AUC_{n}'].mean():>10.4f}" for n in names))

    print(f"\n  XLSX → {out_xlsx}")
