"""RQ6: Multi-Level Concept Drift + Phase C Ablation.

4 drift levels x 4 method variants x 3 datasets.
L0(sigma=0,shift=0) L1(0.2,0.3) L2(0.5,0.8) L3(1.0,1.5)

Methods: Incr1(+PhaseC), Incr1-NR(no PhaseC), SiForest(L0-L1), BWOAIF
Metrics: 300pt sliding-window AUC. Output: DRIFT_Drift.xlsx + DRIFT_checkpoint.csv
"""
import numpy as np, scipy.io as sio, sys, os, time, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(OUT_DIR, "DRIFT_checkpoint.csv")
CSV_DIR = os.path.join(OUT_DIR, "DRIFT_csv")
os.makedirs(CSV_DIR, exist_ok=True)
BETA = 0.67; WINDOW = 300; N_MAX_CAP = 6000

DRIFT_LEVELS = [("L0",0.0,0.0), ("L1",0.2,0.3), ("L2",0.5,0.8), ("L3",1.0,1.5)]

DATASETS = [
    ("thyroid",    "Norm","Numeric_thyroid_3772_6_93_Norm.mat",       "T2"),
    ("pageblocks", "Norm","Numeric_pageblocks_5171_10_258_Norm.mat",  "T3"),
    ("shuttle",    "High","Numeric_shuttle_49097_9_3511_Norm.mat",    "T4"),
]
TIER_PARAMS = {"T2":{"alpha":0.60,"P0":"N"},"T3":{"alpha":0.60,"P0":5000},"T4":{"alpha":0.40,"P0":8000}}

def load_ds(sub,fname):
    fp = os.path.join(BASE,"DATASETS",sub,fname)
    m = sio.loadmat(fp); d = m["dataset"]
    X = d[:,:-1].astype(np.float64); y = (d[:,-1].ravel()>=0.5).astype(np.int64)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)
    return X, y

def stratified_sample(X,y,n_out,rng):
    N = len(X); n_out = min(n_out,N)
    in_ = rng.permutation(np.where(y==0)[0]); ia_ = rng.permutation(np.where(y==1)[0])
    r = len(ia_)/N if N>0 else 0
    na = min(len(ia_), max(1, int(n_out*r))); nn = min(len(in_), n_out-na)
    sel = rng.permutation(np.concatenate([ia_[:na], in_[:nn]]))
    return X[sel], y[sel], sel

def build_full_no_nom(data, X_idx):
    data = np.asarray(data, np.float64); n, m = data.shape
    Xidx = np.asarray(X_idx, int); nX = len(Xidx)
    I = np.zeros(n); I[Xidx] = 1.0; delta = np.zeros(m)
    ID = np.all(data<=1, axis=0); delta[ID] = np.std(data[:,ID], axis=0)
    Lower = np.zeros((m,n))
    for col in range(m):
        RM = (cdist(data[:,[col]],data[:,[col]],'cityblock')<=delta[col]).astype(np.float64)
        Lower[col,:] = np.min(np.maximum(1.0-RM, np.tile(I,(n,1))), axis=1)
    IB = np.tile(I,(m,1))-Lower; NEB = np.min(IB,axis=0); NPB = IB-np.tile(NEB,(m,1))
    return data, Lower, NEB, NPB, I, delta, m, Xidx, nX

def sliding_window_auc(scores, labels, window=300):
    n = len(scores); aucs = np.full(n, np.nan)
    for i in range(window-1, n):
        ss = scores[i-window+1:i+1]; ll = labels[i-window+1:i+1]
        if len(np.unique(ll))>=2: aucs[i] = roc_auc_score(ll, ss)
    return aucs

def apply_drift(S2, sigma, shift, d):
    nd = d//2
    if sigma == 0: return S2.copy()
    S = S2.copy(); fs = np.std(S, axis=0) + 1e-8
    noise = np.random.RandomState(SEED).randn(len(S), nd) * sigma
    S[:, :nd] += noise + shift * fs[:nd]
    return S

def run_incr1(d,Xt,nX,Lo,NE,NPB,I,de,m,s,n_max,do_pc):
    from incr_core import _ndf_and_score, _extend_state_incr, phase_c_reset, build_full_no_nom as bfn
    U0 = len(d); nr = 0; sc = np.zeros(len(s))
    for i in range(len(s)):
        sc[i],_,_ = _ndf_and_score(s[i], d, Xt, nX, NE, NPB, Lo, I, de, m)
        d,Lo,NE,NPB,I,Xt,nX = _extend_state_incr(d,Lo,NE,NPB,I,Xt,nX,de,m,s[i])
        if do_pc and len(d) > n_max:
            d,Xt = phase_c_reset(d,Xt,U0); nr += 1
            d,Lo,NE,NPB,I,de,_,Xt,nX = bfn(d,Xt)
    return sc, nr

def run_siforest(d,s):
    from src.detectors.baselines.siforest_wrapper import SiForestDetector
    det = SiForestDetector(n_trees=50, sample_size=128, window_size=1024); det.fit(d)
    sc = np.zeros(len(s))
    for i in range(len(s)):
        v = det.predict(s[i:i+1]); sc[i] = v[0] if hasattr(v,'__len__') else v.ravel()[0]
        # 方向2：静态评估（fit 后纯 predict，不 partial_fit），与正常 AUC 实验协议一致。
        # 原实现逐点 partial_fit 触发 SiForest wrapper 的 reservoir 索引错位 bug，导致 L0 无漂移即退化到随机。
    return sc

def run_bwoaif(d,s):
    from src.detectors.baselines.bwoaif_wrapper import BWOAIFDIFFI
    det = BWOAIFDIFFI(n_trees=100, sample_size=256, window_size=512); det.fit(d)
    sc = np.zeros(len(s))
    for i in range(len(s)):
        v = det.predict(s[i:i+1]); sc[i] = v[0] if hasattr(v,'__len__') else v.ravel()[0]
        det.partial_fit(s[i:i+1])
    return sc

def run_one(name, sub, fname, tier, global_rng):
    print(f"\n{'='*60}\n  {name}  (Tier={tier})\n{'='*60}", flush=True)
    tp = TIER_PARAMS[tier]; alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname); N, d = X_all.shape

    P0_val = N if tp["P0"]=="N" else min(tp["P0"],N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, global_rng)
    Np0 = len(X_p0); U_size = max(50, int(Np0*alpha))
    x_size = int(U_size*(1-BETA)); p1_size = U_size - x_size
    p3_size = max(20, Np0 - U_size)
    mask_p0 = np.zeros(N, bool); mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]; y4 = y_all[~mask_p0]
    U_init = X_p0[:U_size]; Xidx_init = np.arange(p1_size, U_size)
    S_X = np.vstack([X_p0[U_size:U_size+p3_size], X4]) if len(X4)>0 else X_p0[U_size:U_size+p3_size]
    S_y = np.concatenate([y_p0[U_size:U_size+p3_size], y4]) if len(y4)>0 else y_p0[U_size:U_size+p3_size]
    mid = len(S_X)//2; S1_X = S_X[:mid]; S1_y = S_y[:mid]; S2_o = S_X[mid:]; S2_y = S_y[mid:]
    nd = d//2; K_val = max(1, int(x_size*0.02)); n_max = min(int(1.1*U_size), N_MAX_CAP)
    dU, Lo, NE, NPB, I, de, m, Xt, nX = build_full_no_nom(U_init, Xidx_init)

    print(f"  N={N} d={d} U={U_size} P1={p1_size} X={x_size} T={len(S_X)} drift@{mid} K={K_val} n_max={n_max}", flush=True)

    all_rows = []
    for lvl, sigma, shift in DRIFT_LEVELS:
        S2 = S2_o.copy() if lvl=="L0" else apply_drift(S2_o, sigma, shift, d)
        sX = np.vstack([S1_X, S2]); s_y = np.concatenate([S1_y, S2_y])
        print(f"  --- {lvl} sigma={sigma} shift={shift} ---", flush=True)

        methods = [
            ("Incr1",     lambda: run_incr1(dU.copy(),Xidx_init.copy(),nX,Lo.copy(),NE.copy(),NPB.copy(),I.copy(),de,m,sX,n_max,True)[0]),
            ("Incr1-NR",  lambda: run_incr1(dU.copy(),Xidx_init.copy(),nX,Lo.copy(),NE.copy(),NPB.copy(),I.copy(),de,m,sX,int(1e9),False)[0]),
        ]
        methods.append(("SiForest", lambda: run_siforest(dU,sX)))
        methods.append(("BWOAIF", lambda: run_bwoaif(dU,sX)))

        for mn, mf in methods:
            try:
                t0 = time.perf_counter(); sc = mf(); el = (time.perf_counter()-t0)*1000
                ap  = roc_auc_score(s_y[:mid],sc[:mid]) if len(np.unique(s_y[:mid]))>=2 else np.nan
                apo = roc_auc_score(s_y[mid:],sc[mid:]) if len(np.unique(s_y[mid:]))>=2 else np.nan
                sw = sliding_window_auc(sc, s_y, WINDOW)
                swp = np.nanmean(sw[:mid]); swpo = np.nanmean(sw[mid:])
                print(f"    {mn:<12s} pre={ap:.4f} post={apo:.4f} sw_pre={swp:.4f} sw_post={swpo:.4f} t={el:.0f}ms", flush=True)
                all_rows.append({"dataset":name,"level":lvl,"method":mn,"AUC_pre":ap,"AUC_post":apo,"sw_pre":swp,"sw_post":swpo,"t_ms":el})
            except Exception as e:
                print(f"    {mn:<12s} FAILED: {e}", flush=True)

    return all_rows

if __name__ == "__main__":
    all_rows = []
    ckpt = pd.read_csv(CKPT_FILE) if os.path.exists(CKPT_FILE) else None
    done_ds = set(ckpt["dataset"].unique()) if ckpt is not None else set()
    if ckpt is not None:
        for _, r in ckpt.iterrows(): all_rows.append(r.to_dict())

    global_rng = np.random.RandomState(SEED)
    for name, sub, fname, tier in DATASETS:
        if name in done_ds:
            print(f"  [{name}] SKIP -- already in checkpoint")
            continue
        try:
            rows = run_one(name, sub, fname, tier, global_rng)
            all_rows.extend(rows); pd.DataFrame(all_rows).to_csv(CKPT_FILE, index=False)
            print(f"  [{name}] DONE -- {len(rows)} rows", flush=True)
        except Exception as e:
            print(f"  ERROR {name}: {e}", flush=True)
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    df.to_excel(os.path.join(OUT_DIR, "DRIFT_Drift.xlsx"), index=False)

    for ds in [d[0] for d in DATASETS]:
        print(f"\n  {ds}")
        for lvl in ["L0","L1","L2","L3"]:
            line = f"  {lvl:<4s}"
            for m in ["Incr1","Incr1-NR","SiForest","BWOAIF"]:
                sub = df[(df.dataset==ds)&(df.level==lvl)&(df.method==m)]
                v = sub["sw_post"].values[0] if len(sub)>0 else None
                line += f"  {v:.4f}" if v is not None else "     ---"
            print(line)
        print("  Phase C benefit (Incr1 - Incr1-NR):")
        for lvl in ["L0","L1","L2","L3"]:
            i1 = df[(df.dataset==ds)&(df.level==lvl)&(df.method=="Incr1")]
            nr = df[(df.dataset==ds)&(df.level==lvl)&(df.method=="Incr1-NR")]
            if len(i1)>0 and len(nr)>0:
                print(f"    {lvl}: {i1['sw_post'].values[0] - nr['sw_post'].values[0]:+.4f}")
    print("\n  DONE")
