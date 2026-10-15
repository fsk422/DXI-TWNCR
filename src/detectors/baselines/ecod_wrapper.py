"""ECOD: empirical-CDF based unsupervised outlier detection."""
import numpy as np


class ECODIncremental:
    """ECOD: Empirical-CDF-based Outlier Detection (Li, Zhao et al. 2023).

    Estimates per-feature ECDF incrementally. Anomaly score aggregates
    tail probabilities. Provides per-feature contributions.

    Parameters
    ----------
    contamination : float
        Expected anomaly fraction.
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self._sorted_data = None
        self._n = 0
        self._d = 0
        self._fitted = False

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._n, self._d = X.shape
        self._sorted_data = [np.sort(X[:, j]).copy() for j in range(self._d)]
        self._fitted = True
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self._sorted_data is None:
            return self.fit(X)
        for j in range(self._d):
            self._sorted_data[j] = np.sort(
                np.concatenate([self._sorted_data[j], X[:, j]])
            )
        self._n += X.shape[0]
        return self

    def _ecdf_score(self, x, sorted_data, direction='left'):
        n = len(sorted_data)
        if n == 0:
            return 0.0
        idx = np.searchsorted(sorted_data, x, side='right')
        p = idx / n if direction == 'left' else 1.0 - (idx / n)
        p = max(p, 1e-15)
        p = min(p, 1.0 - 1e-15)
        return -np.log(p)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            score = 0.0
            for j in range(self._d):
                left_score = self._ecdf_score(X[i, j], self._sorted_data[j], 'left')
                right_score = self._ecdf_score(X[i, j], self._sorted_data[j], 'right')
                score += max(left_score, right_score)
            scores[i] = score
        return scores

    def get_attribution(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        contributions = np.zeros((X.shape[0], self._d))
        for i in range(X.shape[0]):
            for j in range(self._d):
                left_s = self._ecdf_score(X[i, j], self._sorted_data[j], 'left')
                right_s = self._ecdf_score(X[i, j], self._sorted_data[j], 'right')
                contributions[i, j] = max(left_s, right_s)
        return contributions
