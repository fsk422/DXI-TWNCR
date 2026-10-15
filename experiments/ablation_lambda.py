"""RQ5: Ablation & Sensitivity — P1/X separation, λ, K.

Three phases on 4 datasets (creditA, mushroom_01, annthyroid, adult_01):
  RQ5a: P1/X separation ablation (w/ vs w/o)   — 2 levels × 4 ds =  8 runs
  RQ5b: λ sensitivity (0.5,1.0,1.5,2.0,3.0)    — 5 levels × 4 ds = 20 runs
  RQ5c: K sensitivity (1,5,10,20,50)             — 5 levels × 4 ds = 20 runs
  Total: 48 runs

Based on incr_core.py building blocks.
Output: RQ5_Ablation.xlsx + RQ5_checkpoint.csv
"""
import numpy as np, scipy.io as sio, sys, os, time, warnings, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(OUT_DIR, "RQ5_checkpoint.csv")
N_MAX_CAP = 6000
BETA = 0.67
MAX_EVAL = 3000

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

PHASES = {
    "RQ5a": {
        "label": "P1/X Separation",
        "factor": "p1x_sep",
        "levels": [
            {"name": "w/",   "p1x_sep": True,  "lambda_val": 1.0, "K_val": 1},
            {"name": "w/o",  "p1x_sep": False, "lambda_val": 1.0, "K_val": 1},
        ],
    },
    "RQ5b": {
        "label": "Lambda Sensitivity",
        "factor": "lambda",
        "levels": [
            {"name": "λ=0.5",  "p1x_sep": True, "lambda_val": 0.5, "K_val": 1},
            {"name": "λ=1.0",  "p1x_sep": True, "lambda_val": 1.0, "K_val": 1},
            {"name": "λ=1.5",  "p1x_sep": True, "lambda_val": 1.5, "K_val": 1},
            {"name": "λ=2.0",  "p1x_sep": True, "lambda_val": 2.0, "K_val": 1},
            {"name": "λ=3.0",  "p1x_sep": True, "lambda_val": 3.0, "K_val": 1},
        ],
    },
    "RQ5c": {
        "label": "K Sensitivity",
        "factor": "K",
        "levels": [
            {"name": "K=1",   "p1x_sep": True, "lambda_val": 1.0, "K_val": 1},
            {"name": "K=5",   "p1x_sep": True, "lambda_val": 1.0, "K_val": 5},
            {"name": "K=10",  "p1x_sep": True, "lambda_val": 1.0, "K_val": 10},
            {"name": "K=20",  "p1x_sep": True, "lambda_val": 1.0, "K_val": 20},
            {"name": "K=50",  "p1x_sep": True, "lambda_val": 1.0, "K_val": 50},
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════
# Core building blocks (from incr_core.py, parameterised)
# ═══════════════════════════════════════════════════════════════════

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
    tmp = CKPT_FILE + f".tmp.{os.getpid()}"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, CKPT_FILE)

def phase_c_reset(U, X_indices, U0_size):
    n_U = len(U); k = n_U - U0_size
    if k <= 0:
        return U, X_indices
    pm = np.ones(n_U, bool); pm[X_indices] = False
    p1o = np.where(pm)[0]; xo = np.asarray(X_indices)
    ep = min(k, len(p1o)); ex = max(0, k - len(p1o))
    dx = min(k, len(xo))
    ev = set(p1o[:ep]) | set(xo[:ex])
    dg = set(xo[-dx:]) if dx > 0 else set()
    nxo = []; o2n = {}; ni = 0
    for i in range(n_U):
        if i not in ev:
            o2n[i] = ni; ni += 1
    for xi in X_indices:
        if xi not in ev and xi not in dg:
            nxo.append(o2n[xi])
    return U[list(o2n.keys())], np.array(nxo, dtype=int)

def _compute_delta_param(data, m, lambda_val):
    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / lambda_val
    return delta

def build_full_param(data, X_indices, lambda_val, p1x_sep):
    data = np.asarray(data, np.float64); n, m = data.shape
    Xidx = np.asarray(X_indices, int); nX = len(Xidx)
    if p1x_sep:
        I = np.zeros(n); I[Xidx] = 1.0
    else:
        I = np.ones(n)
    delta = _compute_delta_param(data, m, lambda_val)

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

def build_full_no_nom_param(data, X_indices, lambda_val, p1x_sep):
    data = np.asarray(data, np.float64); n, m = data.shape
    Xidx = np.asarray(X_indices, int); nX = len(Xidx)
    if p1x_sep:
        I = np.zeros(n); I[Xidx] = 1.0
    else:
        I = np.ones(n)
    delta = _compute_delta_param(data, m, lambda_val)

    Lower = np.zeros((m, n))
    for col in range(m):
        RM = (cdist(data[:, [col]], data[:, [col]], 'cityblock') <= delta[col]).astype(np.float64)
        Lower[col, :] = np.min(np.maximum(1.0 - RM, np.tile(I, (n, 1))), axis=1)

    IB = np.tile(I, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    return data, Lower, NEB, NPB, I, delta, m, Xidx, nX


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
    nU = len(data)
    p1_mask = (I == 0)
    nXn = nX + 1

    rm_per_col = np.zeros(m)
    for col in range(m):
        rm_per_col[col] = np.sum(np.abs(data[Xidx, col] - x_q[col]) <= delta[col]) + 1

    lo_new = np.zeros(m)
    for col in range(m):
        rm_col = (np.abs(data[:, col] - x_q[col]) <= delta[col]).astype(np.float64)
        lo_new[col] = np.min(1.0 - rm_col[p1_mask]) if p1_mask.any() else 1.0

    nom_row = _nom_row(x_q, data[Xidx], delta, m)
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
        ndf = np.zeros(m); rm_per_col = np.zeros(m)
        for col in range(m):
            ndf[col] = (neb_cnt + np.sum(np.minimum(lower_ext[col, :], nom_row >= d3)) +
                        np.sum(np.minimum(npb_flip[col, :], nom_row >= d2))) / nX_new
            rm_per_col[col] = np.sum(np.abs(data[Xidx_new, col] - batch[ki, col]) <= delta[col])
        w = 1.0 - np.sqrt(rm_per_col / nX_new)
        scores[ki] = np.mean(ndf * w)
    return scores


def _region_ratios(Lower, NEB, Xidx):
    neb_mask = NEB[Xidx] > 0.5
    lower_mean = Lower[:, Xidx].mean(axis=0)
    pos_mask = (~neb_mask) & (lower_mean > 0.9)
    npb_mask = (~neb_mask) & (~pos_mask)
    return float(np.mean(neb_mask)), float(np.mean(npb_mask)), float(np.mean(pos_mask))


# ═══════════════════════════════════════════════════════════════════
# Run variants
# ═══════════════════════════════════════════════════════════════════

def run_incr_1(U_init, Xidx_init, eval_X, eval_y, n_max, m, delta):
    data = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(data); scores = np.zeros(len(eval_X))
    times = np.zeros(len(eval_X)); n_resets = 0

    data, Lower, NEB, NPB, I, delta, m2, Xt, nX = build_full_no_nom_param(data, Xidx, 1.0, True)
    neb_ratio, npb_ratio, pos_ratio = _region_ratios(Lower, NEB, Xt)

    for ei in range(len(eval_X)):
        x_pt = eval_X[ei]
        t0 = time.perf_counter()
        scores[ei], _, _ = _ndf_and_score(x_pt, data, Xt, nX, NEB, NPB, Lower, I, delta, m)
        data, Lower, NEB, NPB, I, Xt, nX = _extend_state_incr(
            data, Lower, NEB, NPB, I, Xt, nX, delta, m, x_pt)
        times[ei] = (time.perf_counter() - t0) * 1000
        if len(data) > n_max:
            data, Xt = phase_c_reset(data, Xt, U0)
            n_resets += 1
            data, Lower, NEB, NPB, I, delta, m2, Xt, nX = build_full_no_nom_param(data, Xt, 1.0, True)

    auc, apr = compute_auc(scores, eval_y)
    complet_gap = float(np.nan)
    return auc, apr, np.mean(times), n_resets, scores, neb_ratio, npb_ratio, pos_ratio, complet_gap


def run_incr_1_param(U_init, Xidx_init, eval_X, eval_y, n_max, m, lambda_val, p1x_sep):
    data = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(data); scores = np.zeros(len(eval_X))
    times = np.zeros(len(eval_X)); n_resets = 0
    all_phi = []

    data, Lower, NEB, NPB, I, delta, m2, Xt, nX = build_full_no_nom_param(data, Xidx, lambda_val, p1x_sep)
    neb_ratio, npb_ratio, pos_ratio = _region_ratios(Lower, NEB, Xt)

    for ei in range(len(eval_X)):
        x_pt = eval_X[ei]
        t0 = time.perf_counter()
        sc, nom_row, rm_col = _ndf_and_score(x_pt, data, Xt, nX, NEB, NPB, Lower, I, delta, m)

        nXn = nX + 1; ndf_j = np.zeros(m)
        p1_mask = (I == 0)
        lo_new = np.zeros(m)
        for col in range(m):
            rm_col_vec = (np.abs(data[:, col] - x_pt[col]) <= delta[col]).astype(np.float64)
            lo_new[col] = np.min(1.0 - rm_col_vec[p1_mask]) if p1_mask.any() else 1.0
        ib_new = 1.0 - lo_new; neb_new = np.min(ib_new); npb_new = ib_new - neb_new
        neb_flip = 1.0 - np.clip(np.append(NEB[Xt], neb_new), -1e-10, 1.0)
        npb_flip = 1.0 - np.clip(np.concatenate([NPB[:, Xt], npb_new.reshape(m, 1)], axis=1), -1e-10, 1.0)
        lo_ext = np.concatenate([Lower[:, Xt], lo_new.reshape(m, 1)], axis=1)
        nom_ext = np.append(nom_row, 0.0)
        neb_cnt = np.sum(np.minimum(neb_flip, nom_ext <= m / 3.0))
        for col in range(m):
            lc = np.sum(np.minimum(lo_ext[col, :], nom_ext >= 0.9 * m))
            pc = np.sum(np.minimum(npb_flip[col, :], nom_ext >= m / 2.0))
            ndf_j[col] = (neb_cnt + lc + pc) / nXn
        w = 1.0 - np.sqrt(rm_col / nXn)
        phi_j = ndf_j * w / m
        all_phi.append(phi_j)
        scores[ei] = sc

        data, Lower, NEB, NPB, I, Xt, nX = _extend_state_incr(
            data, Lower, NEB, NPB, I, Xt, nX, delta, m, x_pt)
        times[ei] = (time.perf_counter() - t0) * 1000
        if len(data) > n_max:
            data, Xt = phase_c_reset(data, Xt, U0)
            n_resets += 1
            data, Lower, NEB, NPB, I, delta, m2, Xt, nX = build_full_no_nom_param(data, Xt, lambda_val, p1x_sep)

    auc, apr = compute_auc(scores, eval_y)

    if len(all_phi) > 0:
        phi_arr = np.array(all_phi)
        complet_gap = float(np.mean(np.abs(scores - phi_arr.sum(axis=1))))
    else:
        complet_gap = float('nan')

    return auc, apr, np.mean(times), n_resets, scores, neb_ratio, npb_ratio, pos_ratio, complet_gap


def run_incr_K_param(U_init, Xidx_init, eval_X, eval_y, n_max, m, lambda_val, p1x_sep, K_val):
    data = U_init.copy(); Xidx = Xidx_init.copy()
    U0 = len(data); n_eval = len(eval_X)
    scores = np.zeros(n_eval); t_total = 0; n_resets = 0; pos = 0
    all_phi = []

    data, Lower, NEB, NPB, I, delta, _, Xt, nX = build_full_no_nom_param(data, Xidx, lambda_val, p1x_sep)
    _, _, _, _, NOM_full, _, _, _, _, _ = build_full_param(data, Xidx, lambda_val, p1x_sep)
    XOM = NOM_full[np.ix_(Xidx, Xidx)].copy()
    neb_ratio, npb_ratio, pos_ratio = _region_ratios(Lower, NEB, Xt)

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

        if len(data) > n_max:
            data, Xt = phase_c_reset(data, Xt, U0)
            n_resets += 1
            data, Lower, NEB, NPB, I, delta, _, Xt, nX = build_full_no_nom_param(data, Xt, lambda_val, p1x_sep)
            _, _, _, _, NOM_rebuild, _, _, _, _, _ = build_full_param(data, Xt, lambda_val, p1x_sep)
            XOM = NOM_rebuild[np.ix_(Xt, Xt)].copy()

    auc, apr = compute_auc(scores, eval_y)
    avg_t = t_total / n_eval if n_eval > 0 else float('nan')
    complet_gap = float('nan')
    return auc, apr, avg_t, n_resets, scores, neb_ratio, npb_ratio, pos_ratio, complet_gap


# ═══════════════════════════════════════════════════════════════════
# Per-phase runner
# ═══════════════════════════════════════════════════════════════════

def run_one_level(phase_id, ds_name, sub, fname, tier, level_cfg):
    tp = TIER_PARAMS[tier]; alpha = tp["alpha"]
    X_all, y_all = load_ds(sub, fname)
    N, d = X_all.shape; na_all = y_all.sum()

    local_rng = np.random.RandomState(SEED)
    P0_val = N if tp["P0"] == "N" else min(tp["P0"], N)
    X_p0, y_p0, sel_p0 = stratified_sample(X_all, y_all, P0_val, local_rng)
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

    lambda_val = level_cfg["lambda_val"]
    p1x_sep = level_cfg["p1x_sep"]
    K_val = level_cfg["K_val"]
    level_name = level_cfg["name"]
    n_max = min(int(1.1 * U_size), N_MAX_CAP)

    print(f"  [{level_name}] U={U_size} P1={p1_size} X={x_size} eval={n_eval}/{n_full} "
          f"n_max={n_max} K={K_val} λ={lambda_val} p1x_sep={p1x_sep}")

    t0 = time.perf_counter()
    if K_val == 1:
        auc, apr, avg_t, n_resets, _, neb_ratio, npb_ratio, pos_ratio, complet_gap = \
            run_incr_1_param(U, Xidx_init, eval_X, eval_y, n_max, d, lambda_val, p1x_sep)
    else:
        auc, apr, avg_t, n_resets, _, neb_ratio, npb_ratio, pos_ratio, complet_gap = \
            run_incr_K_param(U, Xidx_init, eval_X, eval_y, n_max, d, lambda_val, p1x_sep, K_val)
    total_sec = time.perf_counter() - t0

    result = {
        "phase": phase_id, "dataset": ds_name, "tier": tier,
        "level": level_name, "level_desc": level_cfg.get("desc", ""),
        "N": N, "d": d, "AR": na_all / N,
        "P0": P0_val, "P1_size": p1_size, "X_size": x_size,
        "P3_size": p3_size, "P4_size": len(X4), "U_size": U_size,
        "n_eval": n_eval, "n_max": n_max,
        "lambda_val": lambda_val, "p1x_sep": p1x_sep, "K_val": K_val,
        "auc_roc": auc, "auc_pr": apr, "f1": float('nan'),
        "completeness_gap": complet_gap,
        "neb_ratio": neb_ratio, "npb_ratio": npb_ratio, "pos_ratio": pos_ratio,
        "avg_update_ms": avg_t, "avg_detect_ms": avg_t,
        "n_resets": n_resets, "total_sec": total_sec,
    }
    print(f"    AUC={auc:.4f}  AP={apr:.4f}  t={avg_t:.2f}ms/pt  "
          f"resets={n_resets}  NEB={neb_ratio:.3f} NPB={npb_ratio:.3f} POS={pos_ratio:.3f}  "
          f"ε={complet_gap:.2e}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ckpt_df = load_ckpt()
    all_rows = [] if ckpt_df is None else ckpt_df.to_dict("records")

    done_set = set()
    if ckpt_df is not None:
        for _, row in ckpt_df.iterrows():
            done_set.add((row["phase"], row["dataset"], row["level"]))

    for phase_id, phase_info in PHASES.items():
        print(f"\n{'#'*70}\n#  PHASE {phase_id}: {phase_info['label']}\n{'#'*70}")
        for ds_name, sub, fname, tier in DATASETS:
            for level_cfg in phase_info["levels"]:
                lvl = level_cfg["name"]
                key = (phase_id, ds_name, lvl)
                if key in done_set:
                    print(f"  [{phase_id}|{ds_name}|{lvl}] SKIP — already done")
                    continue
                try:
                    print(f"\n  [{phase_id}|{ds_name}|{lvl}] RUNNING...")
                    r = run_one_level(phase_id, ds_name, sub, fname, tier, level_cfg)
                    all_rows.append(r)
                    save_ckpt(all_rows)
                except Exception as e:
                    print(f"  ERROR on {phase_id}|{ds_name}|{lvl}: {e}")
                    import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out_xlsx = os.path.join(OUT_DIR, "RQ5_Ablation.xlsx")
    df.to_excel(out_xlsx, index=False)

    print(f"\n{'='*80}")
    print("  RQ5 — FINAL SUMMARY")
    print(f"{'='*80}")

    for phase_id in ["RQ5a", "RQ5b", "RQ5c"]:
        pdf = df[df["phase"] == phase_id]
        if len(pdf) == 0:
            continue
        print(f"\n  {phase_id} — {PHASES[phase_id]['label']}:")
        print(f"  {'Dataset':<14s} " + " ".join(f"{l:>12s}" for l in pdf["level"].unique()))
        print(f"  {'-'*14} " + "-" * (12 * len(pdf["level"].unique())))
        for ds in DATASETS:
            ds_name = ds[0]
            dsr = pdf[pdf["dataset"] == ds_name]
            aucs = []
            for _, row in dsr.iterrows():
                aucs.append(f"{row['auc_roc']:>12.4f}" if not np.isnan(row['auc_roc']) else f"{'---':>12s}")
            print(f"  {ds_name:<14s} " + " ".join(aucs))

    print(f"\n  Done. {len(df)} runs total. Output: {out_xlsx}")
    print(f"  Checkpoints: {CKPT_FILE}")
