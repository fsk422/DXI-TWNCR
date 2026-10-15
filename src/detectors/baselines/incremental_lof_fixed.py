# -*- coding: utf-8 -*-
"""IncrementalLOFFixed — 修复版，标准 LOF 评分 (Breunig et al., SIGMOD 2000)。

与原 incremental_lof.py 的差异（仅修复评分公式，不改接口/超参/窗口语义）：
  原实现:   score = d_{k+1}(x) / mean(d_1..d_k)   —— "局部稀疏度比例"
            丢掉了绝对密度尺度，且方向与局部密度反向：
              正常密集点  → 比值>1（较高）
              孤立离群点  → 比值≈1（较低）
            导致多数数据集上"正常点得分更高"，AUC 反了 / 偏低。
  本实现:   完整 LOF：
              reach_dist_k(p, o) = max(k_dist(o), d(p, o))
              LRD_k(p)  = k / Σ_{o∈N_k(p)} reach_dist_k(p, o)
              LOF_k(p)  = mean_{o∈N_k(p)} LRD_k(o) / LRD_k(p)
            LOF>1 表示 p 比其邻域更稀疏 → 越大越异常（方向正确）。

窗口语义与超参保持原样：参考窗 = U 的最后 window_size 个点，n_neighbors=k。
窗口内点的 k_dist、LRD 在 fit 时预计算一次。
"""
import numpy as np
from scipy.spatial.distance import cdist


class IncrementalLOFFixed:
    def __init__(self, window_size=128, n_neighbors=10, contamination=0.1):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._window = None
        self._kdist = None   # 每个窗口点的 k-距离（窗口内）
        self._lrd = None     # 每个窗口点的局部可达密度
        self._fitted = False

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] > self.window_size:
            X = X[-self.window_size:]
        self._window = X.copy()
        self._precompute()
        self._fitted = True
        return self

    def _precompute(self):
        n = self._window.shape[0]
        k = min(self.n_neighbors, n - 1)
        if k < 1:
            self._kdist = self._lrd = np.zeros(n)
            return
        D = cdist(self._window, self._window)
        D[np.diag_indices(n)] = np.inf
        order = np.argsort(D, axis=1)
        kdist = np.maximum(D[np.arange(n), order[:, k - 1]], 1e-12)
        self._kdist = kdist
        lrd = np.zeros(n)
        for o in range(n):
            nb = order[o, :k]
            reach = np.maximum(kdist[nb], D[o, nb])
            lrd[o] = k / (reach.sum() + 1e-12)
        self._lrd = lrd

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = self._window.shape[0]
        k = min(self.n_neighbors, n - 1)
        if k < 1:
            return np.zeros(X.shape[0])
        kdist = self._kdist
        lrd_ref = self._lrd
        scores = np.zeros(X.shape[0])
        for i, p in enumerate(X):
            dists = cdist(p[None, :], self._window)[0]
            nn = np.argsort(dists)[:k]
            reach = np.maximum(kdist[nn], dists[nn])
            lrd_p = k / (reach.sum() + 1e-12)
            scores[i] = lrd_ref[nn].mean() / (lrd_p + 1e-12)
        return scores