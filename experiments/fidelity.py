"""Fidelity on all 14 datasets — optimized NOM caching."""
import numpy as np, scipy.io as sio, sys, os, time, pandas as pd, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_JSON = os.path.join(OUT_DIR, "Fidelity14_final_checkpoint.json")

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
N_REF = 1500  # max reference points
N_ANOM = 30   # max anomaly points

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

class MNOFComputer:
    """Compute MNOF + phi for a fixed reference set with cached NOM."""

    def __init__(self, X_ref):
        n, m = X_ref.shape
        self.m = m
        # P1/X split
        np1 = max(2, int(n * 0.2))
        nx = n - np1
        idx = np.random.RandomState(SEED + (n * 31 + m * 17) % 100000).permutation(n)
        self.X_p1 = X_ref[idx[:np1]]; self.X_x = X_ref[idx[np1:]]
        self.nx = nx

        # delta
        self.delta = np.zeros(m)
        for j in range(m):
            std_j = np.std(X_ref[:, j])
            self.delta[j] = 0.0 if std_j < 1e-12 else std_j / 1.0

        # NOM on X (cached)
        self.NOM = np.zeros((nx, nx), dtype=np.float64)
        for j in range(m):
            dj = np.abs(self.X_x[:, [j]] - self.X_x[:, j])
            self.NOM += (dj > self.delta[j]).astype(np.float64)

        # Three-way regions
        self.POS = np.ones((m, nx), dtype=bool)
        self.Lower = np.zeros((m, nx))
        for j in range(m):
            dp1j = np.abs(self.X_x[:, [j]] - self.X_p1[:, j])
            self.POS[j] = ~np.any(dp1j <= self.delta[j], axis=1)
            dxj = np.abs(self.X_x[:, [j]] - self.X_x[:, j])
            self.Lower[j] = 1.0 - np.max(
                np.concatenate([(dp1j <= self.delta[j]).astype(np.float64),
                                (dxj <= self.delta[j]).astype(np.float64)], axis=1), axis=1)
            self.Lower[j] = np.clip(self.Lower[j], 0.0, 1.0)

        IB = 1.0 - self.Lower
        self.NEB = np.min(IB, axis=0)
        self.NPB = IB - self.NEB

        # Thresholds
        self.D1 = m / 3.0; self.D2 = m / 2.0; self.D3 = 0.9 * m

        # Precomputed mask sums (fast path)
        self.neb_sum = np.sum(np.minimum(self.NEB, self.NOM <= self.D1))
        self.lower_sums = np.array([np.sum(np.minimum(self.Lower[j,:], self.NOM >= self.D3)) for j in range(m)])
        self.npb_sums = np.array([np.sum(np.minimum(self.NPB[j,:], self.NOM >= self.D2)) for j in range(m)])

    def predict(self, Xq):
        nq = Xq.shape[0]; m = self.m
        scores = np.zeros(nq); phi = np.zeros((nq, m))
        for qi in range(nq):
            for j in range(m):
                in_n = np.sum(np.abs(Xq[qi, j] - self.X_x[:, j]) <= self.delta[j])
                w = (1.0/m) * (1.0 - np.sqrt((in_n+1.0)/(self.nx+1.0)))
                ndf_j = (self.neb_sum + self.lower_sums[j] + self.npb_sums[j]) / self.nx
                phi[qi, j] = ndf_j * w
                scores[qi] += phi[qi, j]
        return scores, phi

def main():
    rng = np.random.RandomState(SEED)
    done = {}
    if os.path.exists(CKPT_JSON):
        done = json.load(open(CKPT_JSON))

    for name, sub, fname in DATASETS:
        if name in done:
            print(f"[{name}] SKIP")
            continue

        print(f"\n[{name}]", end=" ", flush=True)
        try:
            X_all, y_all = load_ds(sub, fname)
            N, d = X_all.shape
            # Cap
            if N > N_REF:
                X_ref, y_ref, _ = stratified_sample(X_all, y_all, N_REF, rng)
            else:
                X_ref, y_ref = X_all, y_all
            col_means = X_ref.mean(axis=0)

            # Select anomalies
            anom_idx = np.where(y_ref == 1)[0]
            if len(anom_idx) == 0: print("no anom"); continue
            if len(anom_idx) > N_ANOM: anom_idx = rng.choice(anom_idx, N_ANOM, replace=False)
            X_ab = X_ref[anom_idx]; na = len(anom_idx)
            print(f"n={na}", end=" ", flush=True)

            # Compute base scores
            comp = MNOFComputer(X_ref)
            s_base, p_base = comp.predict(X_ab)

            # Fidelity: top-3
            fds = []
            for i in range(na):
                top3 = np.argsort(-np.abs(p_base[i]))[:3]
                xm = X_ab[i].copy(); xm[top3] = col_means[top3]
                sm, _ = comp.predict(xm.reshape(1,-1))
                if np.isfinite(sm[0]) and abs(s_base[i]) > 1e-12:
                    fds.append(max(0., (s_base[i]-sm[0])/abs(s_base[i])))
            fid_ncr = np.mean(fds) if fds else float("nan")
            print(f"fid={fid_ncr:.3f}", end=" ", flush=True)

            # Rand3
            rand_all = []
            for _ in range(10):
                rds = []
                for i in range(na):
                    r3 = rng.choice(d, 3, replace=False)
                    xm = X_ab[i].copy(); xm[r3] = col_means[r3]
                    sm, _ = comp.predict(xm.reshape(1,-1))
                    if np.isfinite(sm[0]) and abs(s_base[i]) > 1e-12:
                        rds.append(max(0., (s_base[i]-sm[0])/abs(s_base[i])))
                if rds: rand_all.append(np.mean(rds))
            fid_rand3 = np.mean(rand_all) if rand_all else float("nan")
            print(f"rand={fid_rand3:.3f}")

            done[name] = {"fid_NCR": fid_ncr, "fid_Rand3": fid_rand3}
            json.dump(done, open(CKPT_JSON, 'w'))

        except Exception as e:
            print(f"ERR: {e}")

    # Save
    rows = [{"dataset": k, **v} for k, v in done.items() if not v.get("skipped") and "fid_NCR" in v]
    df = pd.DataFrame(rows)
    df.to_csv(CKPT_JSON.replace('.json','.csv'), index=False)

    ds_map = {"creditA":"Cre","german":"Ger","vote":"Vot","heart":"Hea",
              "thyroid":"Thy","mushroom_01":"Mus","cardio":"Crd",
              "pageblocks":"Pag","pendigits":"Pen","annthyroid":"Ann",
              "mammography":"Mam","adult_01":"Adu","shuttle":"Shu","forestcover":"For"}

    print(f"\n\nFIDELITY ({len(df)} datasets)")
    for col in ['fid_NCR','fid_Rand3']:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: mean={vals.mean():.4f}")

    print("\n% LaTeX:")
    print(r"\begin{table}[htbp]")
    print(r"\centering\caption{消融忠实度（Top-3, 14数据集）}\label{tab:fid14}\footnotesize")
    print(r"\begin{tabular}{lrr}\toprule")
    print(r"DS & Rand3 & ITW-NCR\\\midrule")
    for _, row in df.iterrows():
        ds = ds_map.get(row["dataset"], row["dataset"][:3])
        r3 = row.get("fid_Rand3", float("nan"))
        fn = row.get("fid_NCR", float("nan"))
        print(f"  {ds} & {r3:.3f} & \\textbf{{{fn:.3f}}} \\\\")
    print(r"\midrule")
    avg_r3 = df["fid_Rand3"].mean() if "fid_Rand3" in df.columns else 0
    avg_fn = df["fid_NCR"].mean() if "fid_NCR" in df.columns else 0
    print(f"  Mean & {avg_r3:.3f} & \\textbf{{{avg_fn:.3f}}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
