"""EILOF: efficient incremental local outlier factor."""
import numpy as np
from scipy.spatial.distance import cdist


class EILOFDetector:
    """EILOF: Efficient Incremental Local Outlier Factor (Hu et al. 2025).

    arXiv:2501.01061. Unlike standard ILOF, EILOF only computes LOF
    scores for new incoming data points without recomputing LOF scores
    for existing points. This approximation is both more efficient and
    often more accurate in practice.

    Parameters
    ----------
    window_size : int
        Sliding window capacity for reference set.
    n_neighbors : int
        Number of neighbours for LOF computation.
    """

    def __init__(self, window_size=256, n_neighbors=20):
        self.window_size = window_size
        self.n_neighbors = min(n_neighbors, window_size - 1)
        self._ref_set = None
        self._ref_indices = None
        self._next_idx = 0
        self._fitted = False

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = min(X.shape[0], self.window_size)
        self._ref_set = X[-n:].copy()
        self._ref_indices = list(range(n))
        self._next_idx = n
        self._fitted = True
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for x in X:
            if self._ref_set.shape[0] >= self.window_size:
                oldest = np.argmin(self._ref_indices)
                self._ref_set[oldest] = x
                self._ref_indices[oldest] = self._next_idx
            else:
                self._ref_set = np.vstack([self._ref_set, x.reshape(1, -1)])
                self._ref_indices.append(self._next_idx)
            self._next_idx += 1
        return self

    def _k_distance(self, x, ref_set, k):
        dists = cdist(x.reshape(1, -1), ref_set)[0]
        k_dists = np.sort(dists)
        return k_dists[k], k_dists[:k+1]

    def _lrd(self, x, ref_set, k):
        dists = cdist(x.reshape(1, -1), ref_set)[0]
        knn_idx = np.argsort(dists)[:k+1]
        knn = ref_set[knn_idx[1:]]
        reach_dists = []
        for p in knn:
            rd = max(self._k_distance(p, ref_set, k)[0],
                     cdist(x.reshape(1, -1), p.reshape(1, -1))[0][0])
            reach_dists.append(rd)
        return 1.0 / (np.mean(reach_dists) + 1e-12)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_ref = len(self._ref_set)
        k = min(self.n_neighbors, n_ref - 1)
        if k < 1:
            return np.zeros(X.shape[0])

        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            lrd_x = self._lrd(X[i], self._ref_set, k)
            dists = cdist(X[i:i+1], self._ref_set)[0]
            knn_idx = np.argsort(dists)[:k+1][1:]
            lrd_ratio_sum = 0.0
            for j in knn_idx:
                lrd_j = self._lrd(self._ref_set[j], self._ref_set, k)
                lrd_ratio_sum += lrd_j / (lrd_x + 1e-12)
            scores[i] = lrd_ratio_sum / k
        return scores
