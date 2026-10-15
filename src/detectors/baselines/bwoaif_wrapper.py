"""BWOAIF + DIFFI: incremental isolation forest with feature importance."""
import numpy as np
from scipy.spatial.distance import cdist


class BWOAIFDIFFI:
    """BWOAIF + DIFFI: Incremental iForest with Depth-based Feature Importance.

    BWOAIF (2023): Bayesian Window-based Online Anomaly Detection with
    Incremental Forest. Maintains an ensemble of isolation trees updated
    incrementally via sliding window subtree replacement.

    DIFFI (2023): Depth-based Feature Importance for Isolation Forest.
    Measures how often and how deeply each feature participates in splits.

    Parameters
    ----------
    n_trees : int
        Number of trees in the forest.
    sample_size : int
        Subsample size for each tree.
    max_depth : int or None
        Maximum tree depth.
    window_size : int
        Number of recent points retained for reference.
    """

    def __init__(self, n_trees=100, sample_size=256, max_depth=None, window_size=512):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.window_size = window_size
        self._trees = []
        self._window = None
        self._window_labels = None
        self._n_features = 0
        self._fitted = False
        self._update_count = 0

    def _build_tree(self, X, depth=0):
        n, d = X.shape
        max_d = self.max_depth if self.max_depth is not None else int(np.ceil(np.log2(max(n, 2))))
        if n <= 1 or depth >= max_d:
            return {
                "type": "leaf",
                "size": n,
                "depth": float(depth),
            }
        feature = np.random.randint(0, d)
        col = X[:, feature]
        mn, mx = col.min(), col.max()
        if mn >= mx:
            return {"type": "leaf", "size": n, "depth": float(depth)}
        split = np.random.uniform(mn, mx)
        left_mask = col < split
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"type": "leaf", "size": n, "depth": float(depth)}
        return {
            "type": "internal",
            "feature": feature,
            "threshold": split,
            "depth": float(depth),
            "left": self._build_tree(X[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], depth + 1),
        }

    def _path_length(self, x, tree):
        if tree["type"] == "leaf":
            size = max(tree["size"], 1)
            return tree["depth"] + (2.0 * (np.log(size - 1) + np.euler_gamma) if size > 2 else 0.0)
        if x[tree["feature"]] < tree["threshold"]:
            return self._path_length(x, tree["left"])
        else:
            return self._path_length(x, tree["right"])

    def _path_features(self, x, tree, feature_depths):
        if tree["type"] == "leaf":
            return
        feature = tree["feature"]
        feature_depths[feature] = feature_depths.get(feature, 0) + 1
        if x[tree["feature"]] < tree["threshold"]:
            self._path_features(x, tree["left"], feature_depths)
        else:
            self._path_features(x, tree["right"], feature_depths)

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._n_features = X.shape[1]
        n = min(len(X), self.window_size)
        self._window = X[-n:].copy()

        self._trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, min(self.sample_size, n), replace=False)
            self._trees.append(self._build_tree(self._window[idx]))
        self._fitted = True
        self._window_labels = None
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._window = np.vstack([self._window, X])
        overflow = self._window.shape[0] - self.window_size
        if overflow > 0:
            self._window = self._window[overflow:]

        self._update_count += X.shape[0]
        if self._update_count % max(1, self.window_size // 4) < X.shape[0]:
            n = len(self._window)
            for i in range(self.n_trees):
                idx = np.random.choice(n, min(self.sample_size, n), replace=False)
                self._trees[i] = self._build_tree(self._window[idx])
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if not self._trees:
            return np.zeros(X.shape[0])
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            lengths = []
            for tree in self._trees:
                lengths.append(self._path_length(X[i], tree))
            scores[i] = np.mean(lengths)
        n_ref = len(self._window)
        c_n = 2.0 * (np.log(max(n_ref - 1, 1)) + np.euler_gamma) - 2.0 * (n_ref - 1) / max(n_ref, 1)
        c_n = max(c_n, 1e-12)
        return 2.0 ** (-scores / c_n)

    def get_attribution(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples, n_features = X.shape
        contributions = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            feat_counts = {}
            for tree in self._trees:
                self._path_features(X[i], tree, feat_counts)
            total = sum(feat_counts.values()) or 1
            for k, v in feat_counts.items():
                contributions[i, k] = v / total

        scores = self.predict(X)
        contributions *= scores[:, np.newaxis]
        return contributions
