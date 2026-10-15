"""RRCF: robust random cut forest (uses the optional ``rrcf`` package)."""
import numpy as np

try:
    import rrcf
    HAS_RRCF = True
except ImportError:
    HAS_RRCF = False


class RRCFWrapper:
    """Wrapper for Robust Random Cut Forest (Guha et al. 2016).

    Uses the Amazon rrcf package for streaming anomaly detection.
    Falls back to a simple simulation if rrcf is not installed.

    Parameters
    ----------
    n_trees : int
        Number of trees.
    shingle_size : int
        Shingle size for the RRCF.
    """

    def __init__(self, n_trees=40, shingle_size=4):
        self.n_trees = n_trees
        self.shingle_size = shingle_size
        self._forest = None
        self._window = []
        self._fitted = False
        self._counter = 0
        self._insert_order = []

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._window = list(X)
        self._counter = X.shape[0]
        self._insert_order = []
        if HAS_RRCF:
            self._forest = rrcf.RCTree()
            for pt in X:
                try:
                    self._forest.insert_point(pt, index=self._counter)
                    self._insert_order.append(self._counter)
                    self._counter += 1
                except Exception:
                    pass
        self._fitted = True
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if HAS_RRCF and self._forest is not None:
            for pt in X:
                try:
                    self._forest.insert_point(pt, index=self._counter)
                    self._insert_order.append(self._counter)
                    self._counter += 1
                    if len(self._insert_order) > 256:
                        oldest = self._insert_order.pop(0)
                        self._forest.forget_point(oldest)
                except Exception:
                    self._window.append(pt)
        else:
            for pt in X:
                self._window.append(pt)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if HAS_RRCF and self._forest is not None and len(self._insert_order) > 0:
            try:
                scores = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    scores[i] = self._forest.codisp(tuple(X[i]))
                return scores
            except Exception:
                pass

        scores = np.zeros(X.shape[0])
        if len(self._window) < 2:
            return scores
        window_arr = np.array(self._window)
        for i in range(X.shape[0]):
            dists = np.mean(np.abs(window_arr - X[i]), axis=1)
            scores[i] = float(np.sort(dists)[-min(5, len(dists))].mean())
        return scores
