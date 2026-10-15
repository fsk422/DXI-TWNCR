"""SiForest: streaming isolation forest with reservoir sampling."""
import numpy as np


class SiForestDetector:
    """SiForest: Streaming Isolation Forest (Liu et al. PAKDD 2025).

    Combines isolation principles with reservoir sampling to align
    model with current data distributions. Uses subtree regrowing
    strategy for dynamic model updates.

    Parameters
    ----------
    n_trees : int
        Number of trees in the forest.
    sample_size : int
        Reservoir size for each tree.
    window_size : int
        Sliding window for recent data.
    max_depth : int or None
        Maximum tree depth.
    regrow_interval : int
        Number of updates between subtree regrowth cycles.
    """

    def __init__(self, n_trees=50, sample_size=128, window_size=1024,
                 max_depth=None, regrow_interval=100):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.window_size = window_size
        self.max_depth = max_depth
        self.regrow_interval = regrow_interval
        self._reservoirs = []
        self._trees = []
        self._window = None
        self._n_features = 0
        self._fitted = False
        self._update_count = 0

    def _build_tree(self, X, depth=0):
        n, d = X.shape
        max_d = self.max_depth if self.max_depth is not None else int(np.ceil(np.log2(max(n, 2))))
        if n <= 1 or depth >= max_d:
            return {"type": "leaf", "size": n, "depth": float(depth)}
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

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._n_features = X.shape[1]
        n = len(X)
        self._window = X.copy()

        self._reservoirs = []
        self._trees = []
        for _ in range(self.n_trees):
            r_size = min(self.sample_size, n)
            idx = np.arange(n)
            self._reservoirs.append(np.random.choice(idx, r_size, replace=False))
            self._trees.append(self._build_tree(X[self._reservoirs[-1]]))

        self._fitted = True
        return self

    def _reservoir_update(self, reservoir, N, new_idx):
        r = np.random.randint(0, N)
        if r < len(reservoir):
            reservoir = reservoir.copy()
            reservoir[r] = new_idx
        return reservoir

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._window = np.vstack([self._window, X])
        overflow = self._window.shape[0] - self.window_size
        if overflow > 0:
            self._window = self._window[overflow:]

        win_n = len(self._window)
        for x in X:
            self._update_count += 1
            new_idx = win_n - 1
            for i in range(self.n_trees):
                self._reservoirs[i] = self._reservoir_update(
                    self._reservoirs[i], self._update_count, new_idx - max(0, win_n - self.window_size)
                )

        if self._update_count % self.regrow_interval < X.shape[0]:
            win_arr = self._window[-self.window_size:]
            for i in range(self.n_trees):
                valid_res = self._reservoirs[i][self._reservoirs[i] < len(win_arr)]
                if len(valid_res) < 2:
                    valid_res = np.random.choice(len(win_arr), min(self.sample_size, len(win_arr)), replace=False)
                self._trees[i] = self._build_tree(win_arr[valid_res])

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
