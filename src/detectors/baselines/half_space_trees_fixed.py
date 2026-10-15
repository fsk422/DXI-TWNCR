# -*- coding: utf-8 -*-
"""HalfSpaceTreesFixed — 修复版 (Tan et al., IJCAI 2011) mass-profile 评分。

与原 half_space_trees.py 的差异（仅修复评分机制，不改接口/超参）：
  原实现:   score = -mean(depth)
            且叶子只存在于 depth==max_depth，遍历循环 `while depth < max_depth`
            永远读不到叶子 → 每个点都恰好走到 depth=max_depth → 分数恒为 -10 → AUC=0.5。
  本实现:   score = -(Σ_trees Σ_{depth=0..max_depth} mass_t(n_d(x)))
            mass_t(n) = 参考窗内落入节点 n 的点数（质量剖面）。
            稀疏区域 → 路径上节点质量低 → -Σmass 更大 → 分数越高 → 越异常。

数学定义：
  对第 t 棵树，点 x 沿分裂路径依次到达节点 n_0(x), n_1(x), ..., n_D(x)（D=max_depth），
  质量剖面 m_t(x) = Σ_{d=0}^{D} mass_t(n_d(x))；
  异常分数 f_HST(x) = -(1/T) Σ_{t=1}^{T} m_t(x)，
  满足：f_HST 越大表示 x 越异常（与 AUC 的 higher=anomaly 约定一致）。
"""
import numpy as np


class HalfSpaceTreesFixed:
    def __init__(self, n_trees=25, max_depth=10, window_size=128):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.window_size = window_size
        self._splits = []   # per tree: (n_nodes, 3) [feat, threshold, 0]
        self._mass = []     # per tree: (n_nodes,) 参考质量
        self._fitted = False

    def _n_nodes(self):
        return 2 ** (self.max_depth + 1) - 1

    def _n_internal(self):
        return 2 ** self.max_depth - 1  # depth 0..max_depth-1

    def _build_tree(self, X):
        d = X.shape[1]
        n_nodes = self._n_nodes()
        n_internal = self._n_internal()
        splits = np.zeros((n_nodes, 3))
        for node_id in range(n_internal):
            feat = int(np.random.randint(0, d))
            q_min = float(X[:, feat].min())
            q_max = float(X[:, feat].max())
            if q_max - q_min < 1e-12:
                thr = q_min  # 常值特征：确定性路由（全部走右侧）
            else:
                thr = float(np.random.uniform(q_min, q_max))
            splits[node_id] = [feat, thr, 0]
        return splits

    def _fill_mass(self, X, splits):
        mass = np.zeros(splits.shape[0], dtype=np.float64)
        D = self.max_depth
        for p in X:
            node_id = 0
            for depth in range(D + 1):
                mass[node_id] += 1.0
                if depth == D:
                    break
                feat = int(splits[node_id, 0])
                thr = splits[node_id, 1]
                node_id = (2 * node_id + 1) if p[feat] < thr else (2 * node_id + 2)
        return mass

    def _score_one(self, p):
        D = self.max_depth
        total = 0.0
        for splits, mass in zip(self._splits, self._mass):
            node_id = 0
            for depth in range(D + 1):
                total += mass[node_id]
                if depth == D:
                    break
                feat = int(splits[node_id, 0])
                thr = splits[node_id, 1]
                node_id = (2 * node_id + 1) if p[feat] < thr else (2 * node_id + 2)
        return -total / max(1, self.n_trees)

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self._splits, self._mass = [], []
        for _ in range(self.n_trees):
            sp = self._build_tree(X)
            self._splits.append(sp)
            self._mass.append(self._fill_mass(X, sp))
        self._fitted = True
        return self

    def partial_fit(self, X):
        # 静态评估协议下无需流式更新；保持接口一致
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.array([self._score_one(x) for x in X])