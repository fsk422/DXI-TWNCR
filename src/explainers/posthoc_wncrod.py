"""Post-hoc explanation wrappers for WNCROD-Batch.

Wraps SHAP KernelExplainer and LIME to explain WNCROD-Batch predictions.
Provides a uniform interface for comparison with intrinsic three-way attribution.
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def explain_shap_wncrod(detector, X, X_background, nsamples=200):
    """SHAP KernelExplainer on WNCROD-Batch.

    Parameters
    ----------
    detector : WNCRODBase
        A fitted WNCROD-Batch detector.
    X : np.ndarray (n, d)
        Points to explain.
    X_background : np.ndarray (m, d)
        Background/reference data for SHAP.
    nsamples : int
        Number of evaluations for Kernel SHAP.

    Returns
    -------
    contributions : np.ndarray (n, d)
        SHAP values per feature.
    time_ms : float
        Average explanation time per point in milliseconds.
    """
    import time
    import shap

    t0 = time.perf_counter()

    def predict_fn(x):
        return detector.predict(x).ravel()

    n_bg = min(len(X_background), 100)
    bg = X_background[:n_bg]

    explainer = shap.KernelExplainer(predict_fn, bg)

    n_explain = min(len(X), 30)
    shap_values = explainer.shap_values(X[:n_explain], nsamples=nsamples)

    elapsed = (time.perf_counter() - t0) * 1000
    return np.array(shap_values), elapsed / n_explain


def explain_lime_wncrod(detector, X, X_background, n_samples=2000):
    """LIME explainer on WNCROD-Batch.

    Parameters
    ----------
    detector : WNCRODBase
        Fitted WNCROD-Batch.
    X : np.ndarray (n, d)
        Points to explain.
    X_background : np.ndarray (m, d)
        Training data for discretizer.
    n_samples : int
        Number of LIME samples.

    Returns
    -------
    contributions : np.ndarray (n, d)
        LIME feature weights.
    time_ms : float
        Average explanation time per point in milliseconds.
    """
    import time
    from lime.lime_tabular import LimeTabularExplainer

    t0 = time.perf_counter()
    n_explain = min(len(X), 30)

    explainer = LimeTabularExplainer(
        X_background,
        mode='regression',
        feature_names=list(range(X.shape[1])),
        discretize_continuous=False,
    )

    def predict_fn(x):
        return detector.predict(x).ravel()

    contributions = np.zeros((n_explain, X.shape[1]))
    for i in range(n_explain):
        exp = explainer.explain_instance(
            X[i], predict_fn, num_features=X.shape[1], num_samples=n_samples
        )
        for feat, weight in exp.as_list():
            j = int(feat) if isinstance(feat, (int, float)) else int(str(feat).replace('f', '').split('<=')[0].split('>')[0].strip())
            if 0 <= j < X.shape[1]:
                contributions[i, j] = weight

    elapsed = (time.perf_counter() - t0) * 1000
    return contributions, elapsed / n_explain
