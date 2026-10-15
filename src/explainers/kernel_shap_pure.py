"""Pure Python Kernel SHAP implementation.

Implements the Shapley kernel weighting + weighted linear regression
algorithm from Lundberg & Lee (2017), without depending on shap library.
"""
import numpy as np
from scipy.special import comb
from sklearn.linear_model import LinearRegression


def shapley_kernel_weights(Z):
    """Compute Shapley kernel weights for coalition matrix Z.
    Z: (n_samples, d) binary matrix, 1 = feature present, 0 = absent.
    Returns: w (n_samples,) weight vector.
    """
    d = Z.shape[1]
    s = Z.sum(axis=1)  # coalition sizes
    w = np.zeros(len(Z))
    for i in range(len(Z)):
        sz = int(s[i])
        if sz == 0 or sz == d:
            w[i] = 1e8  # large but finite weight for full/empty coalitions
        else:
            denom = comb(d, sz, exact=False) * sz * (d - sz)
            w[i] = (d - 1) / max(denom, 1e-12)
    return w


def sample_coalitions(d, nsamples, rng):
    """Sample nsamples coalitions (binary masks) via uniform sampling."""
    Z = rng.randint(0, 2, size=(nsamples, d))
    Z[0] = np.ones(d)     # full coalition
    if nsamples > 1:
        Z[1] = np.zeros(d)  # empty coalition
    return Z


def kernel_shap(model_fn, X, bg, nsamples=200, seed=42):
    """Compute Kernel SHAP values for each row of X.

    Parameters
    ----------
    model_fn : callable
        Model prediction function f(x) -> (n,) array.
    X : ndarray (n_explain, d)
        Points to explain.
    bg : ndarray (n_bg, d)
        Background data.
    nsamples : int
        Number of coalitions to sample.
    seed : int
        Random seed.

    Returns
    -------
    shap_values : ndarray (n_explain, d)
    base_value : float
        E[f(x)] over background.
    """
    rng = np.random.RandomState(seed)
    n_explain, d = X.shape
    n_bg = len(bg)

    base_value = float(np.mean(model_fn(bg)))

    Z = sample_coalitions(d, nsamples, rng)
    w = shapley_kernel_weights(Z)

    X_bg = bg[0]  # use a single reference point for masking

    shap_values = np.zeros((n_explain, d))
    for i in range(n_explain):
        x = X[i]
        X_z = np.zeros((nsamples, d))
        for k in range(nsamples):
            mask = Z[k].astype(bool)
            xk = X_bg.copy()
            xk[mask] = x[mask]
            X_z[k] = xk

        f_z = model_fn(X_z)

        reg = LinearRegression(fit_intercept=True)
        reg.fit(Z, f_z, sample_weight=w)

        phi = reg.coef_
        shap_values[i] = phi

    return shap_values, base_value


if __name__ == "__main__":
    import sys, os, time, warnings
    warnings.filterwarnings("ignore")
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, BASE)
    from src.detectors.ix_wncrod import IXWNCROD
    import scipy.io as sio

    fp = os.path.join(BASE, "DATASETS", "Norm", "Numeric_thyroid_3772_6_93_Norm.mat")
    m = sio.loadmat(fp); d = m["dataset"]
    X = d[:, :-1].astype(np.float64)
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12)

    U = X[:300]
    bg = U[:100]
    det = IXWNCROD(lammda=1.0, n_max=500, fast_update=False, batch_size=1, optimize_level=1)
    print("fitting...", flush=True)
    det.fit(U)
    print("fit OK", flush=True)

    X_exp = U[:5]
    scores = det.predict(X_exp)
    print("scores:", scores.round(4), flush=True)

    def fn(x):
        return det.predict(x).ravel()

    print("computing SHAP nsamples=200 (slow)...", flush=True)
    t0 = time.perf_counter()
    vals, base = kernel_shap(fn, X_exp, bg, nsamples=200, seed=42)
    t1 = time.perf_counter()
    print("time: %.1fs" % (t1 - t0), flush=True)
    print("shap sum:", vals.sum(axis=1).round(5), flush=True)
    print("scores:", scores.round(5), flush=True)
    print("base:", base, flush=True)
    print("completeness raw:", np.mean(np.abs(scores - vals.sum(axis=1))), flush=True)
    print("DONE", flush=True)
