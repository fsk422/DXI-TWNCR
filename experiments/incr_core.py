"""IX-WNCROD Incremental Core — 4 Variants on 14 Datasets.

State: (data, Lower, NEB, NPB, I, Xidx, nX) — O(m·n), no O(n^2) matrix.
Incr-1: Theorem-driven per-point O(m·n) extension. No NOM copy. No rebuild.
Incr-K: Batch incremental, O(K·m·n) per batch, amortized O(m·n)/pt.
Baselines: Rebuild-1, Rebuild-K.

Invariants: single SEED=42, unified eval set, append-only insertion.
"""
import numpy as np, scipy.io as sio, sys, os, time, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")
# try: sys.stdout.reconfigure(encoding='utf-8')
# except: pass

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Allow overnight run to use a separate checkpoint file via env var
# e.g. $env:IXAD_RUN_TAG="OVERNIGHT" to write to IncrCore_checkpoint_OVERNIGHT.csv
_RUN_TAG = os.environ.get("IXAD_RUN_TAG", "")
_SFX = f"_{_RUN_TAG}" if _RUN_TAG else ""
CKPT_FILE = os.path.join(OUT_DIR, f"IncrCore_checkpoint{_SFX}.csv")
N_MAX_CAP = 6000
BETA = 0.67
MAX_EVAL = 3000

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
TIER_PARAMS = {
    "T1": {"alpha": 0.60, "P0": "N"},
    "T2": {"alpha": 0.60, "P0": "N"},
    "T3": {"alpha": 0.60, "P0": 5000},
    "T4": {"alpha": 0.40, "P0": 8000},
    "T5": {"alpha": 0.30, "P0": 15000},
}

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

def phase_c_reset(U, X_indices, U0_size):
    n_U = len(U); k = n_U - U0_size
    if k <= 0:
        return U, X_indices
    pm = np.ones(n_U, bool); pm[X_indices] = False
    p1o = np.where(pm)[0]; xo = np.asarray(X_indices)
    ep = min(k, len(p1o)); ex = max(0, k - len(p1o))
    dx = min(k, len(xo))
    ev = set(p1o[:ep]) | set(xo[:ex])
    dg = set(xo[:dx]) if dx > 0 else set()   # fix: 降级最先加入X的对象（原 xo[-dx:] 为最新）
    nxo = []; o2n = {}; ni = 0
    for i in range(n_U):
        if i not in ev:
            o2n[i] = ni; ni += 1
    for xi in X_indices:
        if xi not in ev and xi not in dg:
            nxo.append(o2n[xi])
    return U[list(o2n.keys())], np.array(nxo, dtype=int)

def build_full(data, X_indices, frozen_delta=None):
    data = np.asarray(data, np.float64); n, m = data.shape
    Xidx = np.asarray(X_indices, int); nX = len(Xidx)
    I = np.zeros(n); I[Xidx] = 1.0
    if frozen_delta is not None:
        delta = frozen_delta.copy()
    else:
        delta = np.zeros(m)
        ID = np.all(data <= 1, axis=0)
        delta[ID] = np.std(data[:, ID], axis=0)

    Lower = np.zeros((m, n))
    for col in range(m):
        RM = (cdist(data[:, [col]], data[:, [col]], 'cityblock') <= delta[col]).astype(np.float64)
        Lower[col, :] = np.min(np.maximum(1.0 - RM, np.tile(I, (n, 1))), axis=1)

    IB = np.tile(I, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    D = np.zeros((n, n))
    for col in range(m):
        D += (cdist(data[:, [col]], data[:, [col]], 'cityblock') <= delta[col]).astype(np.float64)
    NOM_full = m - D

    return data, Lower, NEB, NPB, NOM_full, I, delta, m, Xidx, nX

def build_full_no_nom(data, X_indices):
    """build_full without NOM_full — for variants that don't need it."""
    data = np.asarray(data, np.float64); n, m = data.shape
    Xidx = np.asarray(X_indices, int); nX = len(Xidx)
    I = np.zeros(n); I[Xidx] = 1.0
    delta = np.zeros(m); ID = np.all(data <= 1, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0)

    Lower = np.zeros((m, n))
    for col in range(m):
        RM = (cdist(data[:, [col]], data[:, [col]], 'cityblock') <= delta[col]).astype(np.float64)
        Lower[col, :] = np.min(np.maximum(1.0 - RM, np.tile(I, (n, 1))), axis=1)

    IB = np.tile(I, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    return data, Lower, NEB, NPB, I, delta, m, Xidx, nX


def score_one_flip(data, Lower, NEB, NPB, NOM_full, Xidx, nX, I, delta, m, x_q):
    x_q = np.asarray(x_q, np.float64).ravel()
    nU = len(data); nXn = nX + 1

    p1m = np.ones(nU, bool); p1m[Xidx] = False
    rm_q = np.zeros((m, nU))
    for col in range(m):
        rm_q[col, :] = (np.abs(data[:, col] - x_q[col]) <= delta[col]).astype(np.float64)

    nom_q = np.zeros(nX)
    for col in range(m):
        nom_q += (np.abs(data[Xidx, col] - x_q[col]) > delta[col]).astype(np.float64)

    XOMe = np.zeros((nXn, nXn))
    XOMe[:nX, :nX] = NOM_full[np.ix_(Xidx, Xidx)]
    XOMe[nX, :nX] = nom_q; XOMe[:nX, nX] = nom_q

    NEBext = np.zeros(nXn); NEBext[:nX] = NEB[Xidx]
    Loext = np.zeros((m, nXn)); Loext[:, :nX] = Lower[:, Xidx]
    NPBext = np.zeros((m, nXn)); NPBext[:, :nX] = NPB[:, Xidx]

    lo_new = np.zeros(m)
    for col in range(m):
        lo_new[col] = np.min(1.0 - rm_q[col, p1m]) if p1m.any() else 1.0
    Loext[:, nX] = lo_new

    ib_new = 1.0 - lo_new; neb_new = np.min(ib_new)
    NEBext[nX] = neb_new; NPBext[:, nX] = ib_new - neb_new

    neb_flip = 1.0 - np.clip(NEBext, -1e-10, 1.0)
    npb_flip = 1.0 - np.clip(NPBext, -1e-10, 1.0)
    neb_cnt = np.sum(np.minimum(neb_flip, XOMe[nX, :] <= m / 3.0))

    w_q = np.zeros(m); ndf = np.zeros(m)
    for col in range(m):
        lc = np.sum(np.minimum(Loext[col, :], XOMe[nX, :] >= 0.9 * m))
        pc = np.sum(np.minimum(npb_flip[col, :], XOMe[nX, :] >= m / 2.0))
        ndf[col] = (neb_cnt + lc + pc) / nXn
        nnb = np.sum(rm_q[col, Xidx]) + 1
        w_q[col] = 1.0 - np.sqrt(nnb / nXn)

    return np.mean(ndf * w_q)


def _nom_row(x_q, data_X, delta, m):
    nom = np.zeros(len(data_X))
    for col in range(m):
        nom += (np.abs(data_X[:, col] - x_q[col]) > delta[col]).astype(np.float64)
    return nom

def _nom_batch(batch, data_X, delta, m):
    K, nX = len(batch), len(data_X)
    nom = np.zeros((K, nX))
    for col in range(m):
        dist = cdist(batch[:, [col]], data_X[:, [col]], 'cityblock')
        nom += (dist > delta[col]).astype(np.float64)
    return nom

def _ndf_and_score(x_q, data, Xidx, nX, NEB, NPB, Lower, I, delta, m):
    """Compute MNOF for one point using temp-extend (matches rebuild exactly).
    
    Constructs temporary extension matrices for x_q (as if x_q were in X),
    scores, then discards. Returns (mnoF, nom_row, rm_per_col).
    """
    nU = len(data)
    p1_mask = (I == 0)

    nom_row = _nom_row(x_q, data[Xidx], delta, m)
    nXn = nX + 1

    rm_per_col = np.zeros(m)
    for col in range(m):
        rm_per_col[col] = np.sum(np.abs(data[Xidx, col] - x_q[col]) <= delta[col]) + 1

    lo_new = np.zeros(m)
    for col in range(m):
        rm_col = (np.abs(data[:, col] - x_q[col]) <= delta[col]).astype(np.float64)
        lo_new[col] = np.min(1.0 - rm_col[p1_mask]) if p1_mask.any() else 1.0

    ib_new = 1.0 - lo_new; neb_new = np.min(ib_new); npb_new = ib_new - neb_new

    neb_flip = 1.0 - np.clip(np.append(NEB[Xidx], neb_new), -1e-10, 1.0)
    npb_flip = 1.0 - np.clip(np.concatenate([NPB[:, Xidx], npb_new.reshape(m, 1)], axis=1), -1e-10, 1.0)
    lo_ext = np.concatenate([Lower[:, Xidx], lo_new.reshape(m, 1)], axis=1)

    nom_ext = np.append(nom_row, 0.0)
    d1, d2, d3 = m / 3.0, m / 2.0, 0.9 * m

    neb_cnt = np.sum(np.minimum(neb_flip, nom_ext <= d1))
    ndf_cols = np.zeros(m)
    for col in range(m):
        lc = np.sum(np.minimum(lo_ext[col, :], nom_ext >= d3))
        pc = np.sum(np.minimum(npb_flip[col, :], nom_ext >= d2))
        ndf_cols[col] = (neb_cnt + lc + pc) / nXn

    w = 1.0 - np.sqrt(rm_per_col / nXn)
    return np.mean(ndf_cols * w), nom_row, rm_per_col


def _extend_state_incr(data, Lower, NEB, NPB, I, Xidx, nX, delta, m, x_pt):
    """TRUE incremental state extension for ONE new point.
    Only updates Lower/NEB/NPB. No NOM matrix. O(m·nU).
    """
    nU_old = len(data)
    x_pt = np.asarray(x_pt, np.float64).ravel()
    data_new = np.vstack([data, x_pt.reshape(1, -1)])
    I_new = np.append(I, 1.0)
    Xidx_new = np.append(Xidx, nU_old)

    p1_old = (I == 0)

    Lower_new = np.concatenate([Lower, np.zeros((m, 1))], axis=1)
    for col in range(m):
        dist = cdist(data[:, [col]], x_pt.reshape(1, -1)[:, [col]], 'cityblock').ravel()
        rm = (dist <= delta[col]).astype(np.float64)
        lo_val = np.min(1.0 - rm[p1_old]) if p1_old.any() else 1.0
        Lower_new[col, nU_old] = lo_val

    IB_new = np.tile(I_new, (m, 1)) - Lower_new
    NEB_new = np.min(IB_new, axis=0)
    NPB_new = IB_new - np.tile(NEB_new, (m, 1))

    return data_new, Lower_new, NEB_new, NPB_new, I_new, Xidx_new, nX + 1


def _extend_state_batch(data, Lower, NEB, NPB, I, Xidx, nX, delta, m, batch, XOM_old):
    """Incremental batch state extension. O(K·m·nU + K·m·nX).
    Extends Lower/NEB/NPB + extends XOM with K new rows/cols (no O(nX^2) rebuild).
    Returns (new_state, XOM_new, neb_extended, lower_extended, npb_extended)
    """
    K = len(batch); nU_old = len(data); nU_new = nU_old + K; nX_new = nX + K
    data_new = np.vstack([data, batch])
    idx_new_X = np.arange(nU_old, nU_new)
    Xidx_new = np.concatenate([Xidx, idx_new_X])
    I_new = np.append(I, np.ones(K))

    p1_old = (I == 0)

    Lower_new = np.concatenate([Lower, np.zeros((m, K))], axis=1)
    for col in range(m):
        dist_new = cdist(batch[:, [col]], data[:, [col]], 'cityblock').astype(np.float64)
        rm_new = (dist_new <= delta[col]).astype(np.float64)
        for ki in range(K):
            lo_val = np.min(1.0 - rm_new[ki, p1_old]) if p1_old.any() else 1.0
            Lower_new[col, nU_old + ki] = lo_val

    IB_new = np.tile(I_new, (m, 1)) - Lower_new
    NEB_new = np.min(IB_new, axis=0)
    NPB_new = IB_new - np.tile(NEB_new, (m, 1))

    nom_old = _nom_batch(batch, data[Xidx], delta, m)

    if len(batch) >= 2:
        nom_self = np.zeros((K, K))
        for col in range(m):
            dist_self = cdist(batch[:, [col]], batch[:, [col]], 'cityblock')
            nom_self += (dist_self > delta[col]).astype(np.float64)
    else:
        nom_self = np.zeros((1, 1))

    XOM_new = np.zeros((nX_new, nX_new))
    XOM_new[:nX, :nX] = XOM_old
    XOM_new[nX:, :nX] = nom_old
    XOM_new[:nX, nX:] = nom_old.T
    XOM_new[nX:, nX:] = nom_self

    neb_ext = NEB_new[Xidx_new].copy()
    lower_ext = Lower_new[:, Xidx_new].copy()
    npb_ext = NPB_new[:, Xidx_new].copy()

    return (data_new, Lower_new, NEB_new, NPB_new, I_new, Xidx_new, nX_new), \
           XOM_new, neb_ext, lower_ext, npb_ext


def _score_batch_points(batch, data, Xidx_new, nX_new, neb_ext, npb_ext, lower_ext, XOM, delta, m):
    K = len(batch); nX = nX_new - K
    neb_flip = 1.0 - np.clip(neb_ext, -1e-10, 1.0)
    npb_flip = 1.0 - np.clip(npb_ext, -1e-10, 1.0)
    d1, d2, d3 = m / 3.0, m / 2.0, 0.9 * m

    scores = np.zeros(K)
    for ki in range(K):
        nom_row = XOM[nX + ki, :]
        neb_cnt = np.sum(np.minimum(neb_flip, nom_row <= d1))
        ndf = np.zeros(m)
        rm_per_col = np.zeros(m)
        for col in range(m):
            ndf[col] = (neb_cnt + np.sum(np.minimum(lower_ext[col, :], nom_row >= d3)) +
                        np.sum(np.minimum(npb_flip[col, :], nom_row >= d2))) / nX_new
            rm_per_col[col] = np.sum(np.abs(data[Xidx_new, col] - batch[ki, col]) <= delta[col])
        w = 1.0 - np.sqrt(rm_per_col / nX_new)
        scores[ki] = np.mean(ndf * w)
    return scores


def run_rebuild_1(U_init, Xidx_init, eval_X, eval_y, n_max, m):
    U_p1x = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(U_p1x); scores = np.zeros(len(eval_X))
    times = np.zeros(len(eval_X)); n_resets = 0

    _, _, _, _, _, delta_frozen, _, _, _ = build_full_no_nom(U_p1x, Xidx)

    for ei in range(len(eval_X)):
        t0 = time.perf_counter()
        xp = eval_X[ei]
        data, L, NEB, NPB, NOM_full, I, delta, m2, Xt, nX = build_full(U_p1x, Xidx, frozen_delta=delta_frozen)
        scores[ei] = score_one_flip(data, L, NEB, NPB, NOM_full, Xt, nX, I, delta, m2, xp)
        U_p1x = np.vstack([U_p1x, xp.reshape(1, -1)])
        Xidx = np.append(Xidx, len(U_p1x) - 1)
        times[ei] = (time.perf_counter() - t0) * 1000
        if len(U_p1x) > n_max:
            U_p1x, Xidx = phase_c_reset(U_p1x, Xidx, U0)
            n_resets += 1
            _, _, _, _, _, delta_frozen, _, _, _ = build_full_no_nom(U_p1x, Xidx)
        if ei % 400 == 0:
            print(f"    Rebuild1 progress: {ei}/{len(eval_X)}", flush=True)
    auc, apr = compute_auc(scores, eval_y)
    return auc, apr, np.mean(times), n_resets, scores

def run_rebuild_K(U_init, Xidx_init, eval_X, eval_y, n_max, m, K_val):
    U_p1x = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(U_p1x); n_eval = len(eval_X)
    scores = np.zeros(n_eval); t_total = 0; n_resets = 0; pos = 0

    _, _, _, _, _, delta_frozen, _, _, _ = build_full_no_nom(U_p1x, Xidx)

    while pos < n_eval:
        be = min(pos + K_val, n_eval); batch = eval_X[pos:be]
        t0 = time.perf_counter()
        data, L, NEB, NPB, NOM_full, I, delta, m2, Xt, nX = build_full(U_p1x, Xidx, frozen_delta=delta_frozen)
        for bi, bpt in enumerate(batch):
            scores[pos + bi] = score_one_flip(data, L, NEB, NPB, NOM_full, Xt, nX,
                                               I, delta, m2, bpt)
        U_p1x = np.vstack([U_p1x, batch])
        Xidx = np.append(Xidx, np.arange(len(U_p1x) - len(batch), len(U_p1x)))
        t_total += (time.perf_counter() - t0) * 1000
        pos = be
        print(f"    RebuildK batch: {pos}/{n_eval}", flush=True)
        if len(U_p1x) > n_max:
            U_p1x, Xidx = phase_c_reset(U_p1x, Xidx, U0)
            n_resets += 1
            _, _, _, _, _, delta_frozen, _, _, _ = build_full_no_nom(U_p1x, Xidx)
    auc, apr = compute_auc(scores, eval_y)
    return auc, apr, t_total / n_eval, n_resets, scores

def run_incr_1(U_init, Xidx_init, eval_X, eval_y, n_max, m, delta):
    data = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(data); scores = np.zeros(len(eval_X))
    times = np.zeros(len(eval_X)); n_resets = 0

    data, Lower, NEB, NPB, I, delta, m2, Xt, nX = build_full_no_nom(data, Xidx)

    for ei in range(len(eval_X)):
        x_pt = eval_X[ei]
        t0 = time.perf_counter()

        scores[ei], _, _ = _ndf_and_score(x_pt, data, Xt, nX, NEB, NPB, Lower, I, delta, m)

        data, Lower, NEB, NPB, I, Xt, nX = _extend_state_incr(
            data, Lower, NEB, NPB, I, Xt, nX, delta, m, x_pt)

        times[ei] = (time.perf_counter() - t0) * 1000

        if ei % 400 == 0:
            print(f"    Incr1 progress: {ei}/{len(eval_X)}", flush=True)

        if len(data) > n_max:
            data, Xt = phase_c_reset(data, Xt, U0)
            n_resets += 1
            data, Lower, NEB, NPB, I, delta, m2, Xt, nX = build_full_no_nom(data, Xt)

    auc, apr = compute_auc(scores, eval_y)
    return auc, apr, np.mean(times), n_resets, scores

def run_incr_K(U_init, Xidx_init, eval_X, eval_y, n_max, m, delta, K_val, NOM_full):
    data = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(data); n_eval = len(eval_X)
    scores = np.zeros(n_eval); t_total = 0; n_resets = 0; pos = 0

    data, Lower, NEB, NPB, I, delta_m, _, Xt, nX = build_full_no_nom(data, Xidx)
    XOM = NOM_full[np.ix_(Xidx, Xidx)].copy()

    while pos < n_eval:
        be = min(pos + K_val, n_eval); batch = eval_X[pos:be]
        t0 = time.perf_counter()

        (data, Lower, NEB, NPB, I, Xt, nX), XOM, neb_ext, lower_ext, npb_ext = \
            _extend_state_batch(data, Lower, NEB, NPB, I, Xt, nX, delta, m, batch, XOM)

        batch_scores = _score_batch_points(batch, data, Xt, nX, neb_ext, npb_ext,
                                           lower_ext, XOM, delta, m)

        for bi in range(len(batch)):
            scores[pos + bi] = batch_scores[bi]

        t_total += (time.perf_counter() - t0) * 1000
        pos = be
        print(f"    IncrK batch: {pos}/{n_eval}", flush=True)

        if len(data) > n_max:
            data, Xt = phase_c_reset(data, Xt, U0)
            n_resets += 1
            data, Lower, NEB, NPB, I, delta_m, _, Xt, nX = build_full_no_nom(data, Xt)
            _, _, _, _, NOM_rebuild, _, _, _, _, _ = build_full(data, Xt)
            XOM = NOM_rebuild[np.ix_(Xt, Xt)].copy()

    auc, apr = compute_auc(scores, eval_y)
    return auc, apr, t_total / n_eval, n_resets, scores


def run_one(name, sub, fname, tier, global_rng):
    print(f"\n{'='*60}\n  {name}  (Tier={tier})\n{'='*60}")
    tp = TIER_PARAMS[tier]; alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname)
    N, d = X_all.shape; na_all = y_all.sum()
    print(f"  N={N}  d={d}  AR={na_all/N:.4f}")

    P0_val = N if tp["P0"] == "N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, global_rng)
    Np0 = len(X_p0)
    U_size = max(50, int(Np0 * alpha))
    x_size = int(U_size * (1 - BETA))
    p1_size = U_size - x_size
    p3_size = max(20, Np0 - U_size)
    mask_p0 = np.zeros(N, bool); mask_p0[sel_p0] = True
    X4 = X_all[~mask_p0]; y4 = y_all[~mask_p0]

    U = X_p0[:U_size]
    Xidx_init = np.arange(p1_size, U_size)

    eval_X = np.vstack([X_p0[U_size:U_size+p3_size], X4]) if len(X4) > 0 else X_p0[U_size:U_size+p3_size]
    eval_y = np.concatenate([y_p0[U_size:U_size+p3_size], y4]) if len(y4) > 0 else y_p0[U_size:U_size+p3_size]
    n_full = len(eval_X)

    cap = min(MAX_EVAL, n_full)
    if cap < n_full:
        sample_rng = np.random.RandomState(SEED)
        eval_X, eval_y, _ = stratified_sample(eval_X, eval_y, cap, sample_rng)
    n_eval = len(eval_X)
    print(f"  U={U_size}  P1={p1_size}  X={x_size}  eval={n_eval}/{n_full}")

    K_val = max(1, int(x_size * 0.02))
    n_max = min(int(1.1 * U_size), N_MAX_CAP)
    print(f"  K={K_val}  n_max={n_max}")

    t0 = time.perf_counter()
    data_U, Lower, NEB, NPB, NOM_full, I, delta, m, Xt, nX = build_full(U, Xidx_init)
    t_build = (time.perf_counter() - t0) * 1000
    print(f"  Base build: {t_build:.0f}ms  m={m}  nX={nX}")

    r = {"dataset": name, "N": N, "d": d, "AR": na_all/N, "tier": tier,
         "U0": U_size, "X0": x_size, "K": K_val, "n_max": n_max,
         "n_eval": n_eval, "t_build_ms": t_build}

    def _record(label, auc, apr, t_ms, nr, base_scores=None, other_scores=None):
        r[f"AUC_{label}"] = auc; r[f"AP_{label}"] = apr
        r[f"t_{label}_ms"] = t_ms; r[f"reset_{label}"] = nr
        if base_scores is not None and other_scores is not None and len(base_scores) == len(other_scores):
            r[f"dmax_{label}"] = np.max(np.abs(base_scores - other_scores))
        else:
            r[f"dmax_{label}"] = np.nan

    a3, p3, t3, nr3, s3 = run_rebuild_1(U, Xidx_init, eval_X, eval_y, n_max, m)
    _record("Rebuild1", a3, p3, t3, nr3)
    print(f"  [Rebuild-1]    AUC={a3:.4f}  t={t3:.0f}ms/pt  resets={nr3}")

    a4, p4, t4, nr4, s4 = run_rebuild_K(U, Xidx_init, eval_X, eval_y, n_max, m, K_val)
    _record("RebuildK", a4, p4, t4, nr4)
    print(f"  [Rebuild-K]    AUC={a4:.4f}  t={t4:.0f}ms/pt  resets={nr4}")

    a5, p5, t5, nr5, s5 = run_incr_1(U, Xidx_init, eval_X, eval_y, n_max, m, delta)
    _record("Incr1", a5, p5, t5, nr5, s3, s5)
    eq5 = r.get("dmax_Incr1", np.nan)
    print(f"  [Incr-1]  **   AUC={a5:.4f}  t={t5:.2f}ms/pt  resets={nr5}  dmax={eq5:.2e}")

    a6, p6, t6, nr6, s6 = run_incr_K(U, Xidx_init, eval_X, eval_y, n_max, m, delta, K_val, NOM_full)
    _record("IncrK", a6, p6, t6, nr6, s3, s6)
    eq6 = r.get("dmax_IncrK", np.nan)
    print(f"  [Incr-K]  **   AUC={a6:.4f}  t={t6:.2f}ms/pt  resets={nr6}  dmax={eq6:.2e}")

    print(f"\n  speedup: Incr1/{'Rebuild1':>9s} = {t3/t5:.1f}x  |  IncrK/{'RebuildK':>9s} = {t4/t6:.1f}x  |  Incr1/IncrK = {t5/t6:.2f}x")
    print(f"  Δ(Rebuild1-Incr1) = {a3-a5:+.4f}  "
          f"Δ(Rebuild1-IncrK) = {a3-a6:+.4f}")
    return r


if __name__ == "__main__":
    ckpt = load_ckpt()
    all_rows = []
    global_rng = np.random.RandomState(SEED)

    for name, sub, fname, tier in DATASETS:
        if ckpt is not None and name in ckpt["dataset"].values:
            prev = ckpt[ckpt["dataset"] == name].iloc[0]
            if prev.get("AUC_Incr1") is not None and not pd.isna(prev["AUC_Incr1"]):
                print(f"  [{name}] SKIP — already done (AUC_Incr1={prev['AUC_Incr1']:.4f})")
                all_rows.append(prev.to_dict())
                continue
        try:
            r = run_one(name, sub, fname, tier, global_rng)
            all_rows.append(r)
            save_ckpt(all_rows)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out_xlsx = os.path.join(OUT_DIR, f"IncrCore{_SFX}.xlsx")
    df.to_excel(out_xlsx, index=False)

    variant_names = ["Rebuild1", "RebuildK", "Incr1", "IncrK"]
    print(f"\n{'='*120}")
    print("  INCREMENTAL CORE — FINAL SUMMARY")
    print(f"{'='*120}")
    header = f"  {'Dataset':<14s} " + " ".join(f"{v:>10s}" for v in variant_names)
    print(header)
    print("  " + "-" * (14 + 10 * len(variant_names)))
    for _, row in df.iterrows():
        aucs = [row.get(f"AUC_{v}", np.nan) for v in variant_names]
        line = f"  {row['dataset']:<14s} " + " ".join(
            f"{a:>10.4f}" if not np.isnan(a) else f"{'---':>10s}" for a in aucs)
        print(line)

    print(f"\n  AVG AUC: " + " ".join(
        f"{df[f'AUC_{v}'].mean():>10.4f}" for v in variant_names))

    print(f"\n  TIMING (ms/pt):")
    print(f"  {'Dataset':<14s} " + " ".join(f"{v:>10s}" for v in variant_names))
    print("  " + "-" * (14 + 10 * len(variant_names)))
    for _, row in df.iterrows():
        ts = [row.get(f"t_{v}_ms", np.nan) for v in variant_names]
        line = f"  {row['dataset']:<14s} " + " ".join(
            f"{t:>10.1f}" if not np.isnan(t) else f"{'---':>10s}" for t in ts)
        print(line)

    print(f"\n  EQUIVALENCE (dmax vs Rebuild-1):")
    for _, row in df.iterrows():
        di = row.get("dmax_Incr1", np.nan); dk = row.get("dmax_IncrK", np.nan)
        print(f"  {row['dataset']:<14s}  Incr1: {di:.2e}  IncrK: {dk:.2e}")

    print(f"\n  XLSX → {out_xlsx}")
