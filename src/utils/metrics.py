"""Evaluation metrics for detection and explanation quality."""
from typing import Optional
import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


def compute_all_metrics(
    y_true: npt.NDArray[np.int64],
    y_scores: npt.NDArray[np.floating],
    threshold: Optional[float] = None,
) -> dict[str, float]:
    """Compute comprehensive detection performance metrics.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Binary ground-truth labels (1 = anomaly).
    y_scores : np.ndarray of shape (n_samples,)
        Anomaly scores (higher = more anomalous).
    threshold : float, optional
        Threshold for binarization. If None, uses percentile-based.

    Returns
    -------
    metrics : dict
        Dictionary with precision, recall, f1, auc_roc, auc_pr.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if threshold is None:
        threshold = np.percentile(y_scores, 95)

    y_pred = (y_scores >= threshold).astype(np.int64)

    results = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        results["auc_roc"] = float(roc_auc_score(y_true, y_scores))
        results["auc_pr"] = float(average_precision_score(y_true, y_scores))
    else:
        results["auc_roc"] = 0.0
        results["auc_pr"] = 0.0

    return results


def compute_explanation_metrics(
    attribution_mask: npt.NDArray[np.bool_],
    contributions: npt.NDArray[np.floating],
    y_true: npt.NDArray[np.int64],
    top_k: int = 3,
) -> dict[str, float]:
    """Compute explanation quality metrics using ground-truth attribution.

    Parameters
    ----------
    attribution_mask : np.ndarray of shape (n_samples, n_features)
        Ground-truth: which features caused each anomaly.
    contributions : np.ndarray of shape (n_samples, n_features)
        Predicted per-feature contributions.
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth anomaly labels.
    top_k : int
        Number of top features to check for hit rate.

    Returns
    -------
    metrics : dict
        Explanation quality metrics.
    """
    mask = attribution_mask.astype(np.float64)
    phi = np.abs(contributions)

    anomaly_idx = np.where(y_true == 1)[0]
    if len(anomaly_idx) == 0:
        return {"hit_rate@k": 0.0, "faithfulness_corr": 0.0, "mean_rank": 0.0}

    mask_anom = mask[anomaly_idx]
    phi_anom = phi[anomaly_idx]

    n_anom = len(anomaly_idx)
    hits = 0
    total_rank = 0.0

    for i in range(n_anom):
        ranked = np.argsort(phi_anom[i])[::-1]
        top_pred = ranked[:top_k]
        true_feats = np.where(mask_anom[i] > 0)[0]
        if len(true_feats) > 0 and len(np.intersect1d(top_pred, true_feats)) > 0:
            hits += 1
        for feat in true_feats:
            rank = np.where(ranked == feat)[0]
            if len(rank) > 0:
                total_rank += rank[0] + 1

    n_true_feats = max(1, np.sum(mask_anom))
    hit_rate = hits / n_anom if n_anom > 0 else 0.0
    mean_rank = total_rank / n_true_feats if n_true_feats > 0 else 0.0

    faithfulness_corr = 0.0
    for j in range(attribution_mask.shape[1]):
        if np.sum(mask[:, j]) > 1:
            corr = np.corrcoef(mask[:, j], phi[:, j])[0, 1]
            if not np.isnan(corr):
                faithfulness_corr += corr
    faithfulness_corr /= max(1, attribution_mask.shape[1])

    return {
        f"hit_rate@{top_k}": float(hit_rate),
        "faithfulness_corr": float(faithfulness_corr),
        "mean_rank": float(mean_rank),
    }


def compute_convergence_metrics(
    param_trace: list[npt.NDArray[np.floating]],
    batch_param: npt.NDArray[np.floating],
) -> dict[str, float]:
    """Measure how quickly incremental parameters converge to batch.

    Parameters
    ----------
    param_trace : list of np.ndarray
        Parameter values at each incremental step.
    batch_param : np.ndarray
        Full-batch parameter values (ground truth).

    Returns
    -------
    metrics : dict
        final_gap, avg_gap, convergence_rate.
    """
    if not param_trace:
        return {"final_gap": np.inf, "avg_gap": np.inf, "convergence_rate": np.inf}

    gaps = [np.linalg.norm(p - batch_param) for p in param_trace]
    final_gap = gaps[-1]
    avg_gap = float(np.mean(gaps))

    convergence_rate = np.inf
    if len(gaps) >= 2:
        rates = []
        for i in range(1, len(gaps)):
            if gaps[i - 1] > 1e-12:
                rates.append(gaps[i] / gaps[i - 1])
        convergence_rate = float(np.mean(rates)) if rates else np.inf

    return {
        "final_gap": float(final_gap),
        "avg_gap": float(avg_gap),
        "convergence_rate": float(convergence_rate),
    }
