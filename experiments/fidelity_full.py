"""Fidelity — ITW-NCR vs ECOD vs HBOS vs LODA on all 14 datasets."""
import numpy as np, scipy.io as sio, sys, os, time, pandas as pd, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors.baselines import HBOSIncremental, ECODIncremental, LODAIncremental

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_JSON = os.path.join(OUT_DIR, "Fidelity14_full_checkpoint.json")

DATASETS = [
    ("creditA","Norm","Mixed_creditA_425_15_42_Norm.mat"),
    ("german","Norm","Mixed_german_714_20_14_Norm.mat"),
    ("vote","Norm","Categorical_vote_296_16_29_Norm.mat"),
    ("heart","Norm","Mixed_heart_166_13_16_Norm.mat"),
    ("thyroid","Norm","Numeric_thyroid_3772_6_93_Norm.mat"),
    ("mushroom_01","Norm","Categorical_mushroom_01_4293_22_85_Norm.mat"),
    ("cardio","Norm","Numeric_cardio_1831_21_176_Norm.mat"),
    ("pageblocks","Norm","Numeric_pageblocks_5171_10_258_Norm.mat"),
    ("pendigits","Norm","Numeric_pendigits_6870_16_156_Norm.mat"),
    ("annthyroid","Norm","Numeric_annthyroid_7200_6_534_Norm.mat"),
    ("mammography","Norm","Numeric_mammography_11183_6_260_Norm.mat"),
    ("adult_01","High","Mixed_adult_01_34357_14_343_Norm.mat"),
    ("shuttle","High","Numeric_shuttle_49097_9_3511_Norm.mat"),
    ("forestcover","High","Numeric_forestcover_286048_10_2747_Norm.mat"),
]
N_REF = 1500
N_ANOM = 30

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
    n_a = min(len(idx_a), max(1, int(n_out * len(idx_a) / N)))
    n_n = min(len(idx_n), n_out - n_a)
    sel = rng.permutation(np.concatenate([idx_a[:n_a], idx_n[:n_n]]))
    return X[sel], y[sel], sel

def fidelity_top3(detector, X_ab, scores_ab, phi_ab, col_means, rng):
    """Remove top-3 |phi| attributes → column means → measure drop."""
    na, dd = phi_ab.shape; fds = []
    for i in range(na):
        top3 = np.argsort(-np.abs(phi_ab[i]))[:3]
        xm = X_ab[i].copy(); xm[top3] = col_means[top3]
        sm = detector.predict(xm.reshape(1,-1))[0]
        if np.isfinite(sm) and abs(scores_ab[i]) > 1e-12:
            fds.append(max(0., (scores_ab[i]-sm)/abs(scores_ab[i])))
    return np.mean(fds) if fds else float("nan")

def fidelity_rand3(detector, X_ab, scores_ab, dd, col_means, rng):
    """Random 3 attributes → column means, average over 10 trials."""
    rand_all = []
    for _ in range(10):
        rds = []
        for i in range(len(X_ab)):
            r3 = rng.choice(dd, 3, replace=False)
            xm = X_ab[i].copy(); xm[r3] = col_means[r3]
            sm = detector.predict(xm.reshape(1,-1))[0]
            if np.isfinite(sm) and abs(scores_ab[i]) > 1e-12:
                rds.append(max(0., (scores_ab[i]-sm)/abs(scores_ab[i])))
        if rds: rand_all.append(np.mean(rds))
    return np.mean(rand_all) if rand_all else float("nan")

def main():
    rng = np.random.RandomState(SEED)
    done = {}
    if os.path.exists(CKPT_JSON):
        done = json.load(open(CKPT_JSON))

    for name, sub, fname in DATASETS:
        if name in done and "fid_ECOD" in done[name]:
            print(f"[{name}] SKIP")
            continue

        print(f"\n[{name}]", end=" ", flush=True)
        try:
            X_all, y_all = load_ds(sub, fname)
            N, d = X_all.shape
            if N > N_REF:
                X_ref, y_ref, _ = stratified_sample(X_all, y_all, N_REF, rng)
            else:
                X_ref, y_ref = X_all, y_all
            col_means = X_ref.mean(axis=0)

            anom_idx = np.where(y_ref == 1)[0]
            if len(anom_idx) == 0: print("no anom"); continue
            if len(anom_idx) > N_ANOM: anom_idx = rng.choice(anom_idx, N_ANOM, replace=False)
            X_ab = X_ref[anom_idx]; na = len(anom_idx)
            print(f"n={na} d={d}", end=" ", flush=True)

            result = {}

            # ── ITW-NCR (use the standalone MNOFComputer) ──
            # Reuse from previous run
            prev_json = os.path.join(OUT_DIR, "Fidelity14_final_checkpoint.json")
            if os.path.exists(prev_json):
                prev = json.load(open(prev_json))
                if name in prev and "fid_NCR" in prev[name]:
                    result["fid_NCR"] = prev[name]["fid_NCR"]
                    result["fid_Rand3"] = prev[name]["fid_Rand3"]
                    print(f"NCR={result['fid_NCR']:.3f}", end=" ", flush=True)

            # ── ECOD ──
            det_e = ECODIncremental(); det_e.fit(X_ref)
            s_e = det_e.predict(X_ab); p_e = det_e.get_attribution(X_ab)
            s_e=np.nan_to_num(s_e,nan=0); p_e=np.nan_to_num(p_e,nan=0)
            result["fid_ECOD"] = fidelity_top3(det_e, X_ab, s_e, p_e, col_means, rng)
            result["fid_ECOD_Rand3"] = fidelity_rand3(det_e, X_ab, s_e, d, col_means, rng)
            print(f"ECO={result['fid_ECOD']:.3f}", end=" ", flush=True)

            # ── HBOS ──
            det_h = HBOSIncremental(n_bins=10); det_h.fit(X_ref)
            s_h = det_h.predict(X_ab); p_h = det_h.get_attribution(X_ab)
            s_h=np.nan_to_num(s_h,nan=0); p_h=np.nan_to_num(p_h,nan=0)
            result["fid_HBOS"] = fidelity_top3(det_h, X_ab, s_h, p_h, col_means, rng)
            result["fid_HBOS_Rand3"] = fidelity_rand3(det_h, X_ab, s_h, d, col_means, rng)
            print(f"HBO={result['fid_HBOS']:.3f}", end=" ", flush=True)

            # ── LODA ──
            det_l = LODAIncremental(n_random_cuts=100, n_bins=10); det_l.fit(X_ref)
            s_l = det_l.predict(X_ab); p_l = det_l.get_attribution(X_ab)
            s_l=np.nan_to_num(s_l,nan=0); p_l=np.nan_to_num(p_l,nan=0)
            result["fid_LODA"] = fidelity_top3(det_l, X_ab, s_l, p_l, col_means, rng)
            result["fid_LODA_Rand3"] = fidelity_rand3(det_l, X_ab, s_l, d, col_means, rng)
            print(f"LOD={result['fid_LODA']:.3f}")

            if name not in done or "fid_NCR" not in done.get(name, {}):
                done[name] = {}
            done[name].update(result)
            json.dump(done, open(CKPT_JSON, 'w'))

        except Exception as e:
            print(f"ERR: {e}"); import traceback; traceback.print_exc()

    # ── Summary ──
    rows = [{"dataset": k, **v} for k, v in done.items() if "fid_NCR" in v]
    df = pd.DataFrame(rows)
    df.to_csv(CKPT_JSON.replace('.json','.csv'), index=False)

    ds_map = {"creditA":"Cre","german":"Ger","vote":"Vot","heart":"Hea",
              "thyroid":"Thy","mushroom_01":"Mus","cardio":"Crd",
              "pageblocks":"Pag","pendigits":"Pen","annthyroid":"Ann",
              "mammography":"Mam","adult_01":"Adu","shuttle":"Shu","forestcover":"For"}
    methods = [("fid_Rand3","Rand3"),("fid_NCR","ITW"),("fid_ECOD","ECOD"),("fid_HBOS","HBOS"),("fid_LODA","LODA")]

    print(f"\n\n{'='*60}\nFULL FIDELITY ({len(df)} datasets)")
    print('='*60)
    for mc, mn in methods:
        if mc in df.columns:
            v = df[mc].dropna()
            if len(v) > 0: print(f"  {mn}: mean={v.mean():.4f} ({len(v)}ds)")

    print("\n% LaTeX:")
    print(r"\begin{table}[htbp]")
    print(r"\centering\caption{全14数据集忠实度（Top-3消融，粗体最优）}\label{tab:fid_full}\footnotesize")
    print(r"\begin{tabular}{l"+"r"*len(methods)+r"}\toprule")
    print("DS & " + " & ".join([m[1] for m in methods]) + r"\\\midrule")
    all_vals = {m[0]: [] for m in methods}
    for _, row in df.iterrows():
        ds = ds_map.get(row["dataset"], row["dataset"][:3])
        vals = [row.get(mc, float("nan")) for mc, _ in methods]
        valid = [v for v in vals if not np.isnan(v)]
        best = max(valid) if valid else float("nan")
        parts = [ds]
        for v in vals:
            s = f'{v:.3f}' if not np.isnan(v) else '---'
            if not np.isnan(v) and v == best: s = f'\\textbf{{{s}}}'
            parts.append(s)
            if not np.isnan(v): all_vals[mc].append(v)
        print(" & ".join(parts) + r" \\")
    print(r"\midrule")
    parts = ["Mean"]
    for mc, _ in methods:
        parts.append(f'{np.mean(all_vals[mc]):.3f}' if all_vals[mc] else '---')
    print(" & ".join(parts) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
