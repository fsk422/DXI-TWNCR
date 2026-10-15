"""xStream: hash-projection density-based stream anomaly detector."""
import numpy as np


class XStreamDetector:
    """Simplified xStream-like detector (Manzoor et al. 2018, KDD).

    Density-based ensemble with multi-scale random half-space chains.
    Maintains hash projection chains for O(1) update per point.

    Parameters
    ----------
    n_components : int
        Number of hash projection chains.
    n_hash_fns : int
        Number of hash functions per chain.
    """

    def __init__(self, n_components=50, n_hash_fns=10):
        self.n_components = n_components
        self.n_hash_fns = n_hash_fns
        self._projections = None
        self._counts = None
        self._fitted = False

    def _init_projections(self, d):
        self._projections = []
        for _ in range(self.n_components):
            chain = []
            for __ in range(self.n_hash_fns):
                proj = np.random.randn(d)
                proj /= np.linalg.norm(proj) + 1e-12
                offset = np.random.uniform(-2, 2)
                chain.append((proj, offset))
            self._projections.append(chain)
        self._counts = [{} for _ in range(self.n_components)]

    def _hash_point(self, x, chain):
        code = ""
        for proj, offset in chain:
            val = int((np.dot(proj, x) + offset) * 10)
            code += str(val % 16)
        return code

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self._projections is None:
            self._init_projections(X.shape[1])
        for x in X:
            for ci, chain in enumerate(self._projections):
                code = self._hash_point(x, chain)
                self._counts[ci][code] = self._counts[ci].get(code, 0) + 1
        self._fitted = True
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for x in X:
            for ci, chain in enumerate(self._projections):
                code = self._hash_point(x, chain)
                self._counts[ci][code] = self._counts[ci].get(code, 0) + 1
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        scores = np.zeros(X.shape[0])
        total = sum(sum(c.values()) for c in self._counts) + 1e-12
        for i in range(X.shape[0]):
            score = 0.0
            for ci, chain in enumerate(self._projections):
                code = self._hash_point(X[i], chain)
                count = self._counts[ci].get(code, 0)
                score += -np.log((count + 1) / total)
            scores[i] = score / self.n_components
        return scores
