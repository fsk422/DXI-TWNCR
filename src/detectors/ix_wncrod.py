"""IX-WNCROD: Bounded Exact Incremental with Periodic Reset.

Nine variants (V1–V9) controlled by fast_update + batch_size + optimize_level:

Update dimension (2×2):
  V2: fast=False, bs=1   — per-point full recompute O(n²d)
  V3: fast=True,  bs=1   — per-point incremental O(nd)
  V4: fast=True,  bs=K   — batch incremental O(nd) per batch
  V5: fast=False, bs=K   — batch full recompute O(n²d) per batch

Score optimization dimension (applied on top of V3/V4):
  V6 = V3 + opt2+opt3 (optimize_level=2, bs=1)
  V7 = V4 + opt2+opt3 (optimize_level=2, bs=K)
  V8 = V3 + opt3     (optimize_level=1, bs=1)
  V9 = V4 + opt3     (optimize_level=1, bs=K)

  V1 = WNCRODBase (separate class, batch-only, no attribution)

CORE INVARIANT: V2–V9 produce identical scores and attributions for any point.
"""
import time
import numpy as np
import numpy.typing as npt
from typing import Optional
from scipy.spatial.distance import cdist
from ..base.detector import BaseDetector


class IXWNCROD(BaseDetector):
    """Exact Incremental eXplainable WNCROD with bounded memory.

    Parameters
    ----------
    lammda : float
        Radius parameter (delta = sigma / lambda).
    n_max : int
        Maximum stored points. When exceeded, oldest n_max//5 are evicted.
    drift_threshold : float or None
        NOM z-score threshold for drift-triggered reset.
    fast_update : bool
        True → incremental update. False → full recompute.
    batch_size : int
        1 → per-point update. K>1 → batch update.
    optimize_level : int
        0 → base scoring (V3/V4).
        1 → opt3 only — fuse NEB/D1 (V8/V9).
        2 → opt2+opt3 — fuse + precomputed masks (V6/V7).
        3 → opt2 only — precomputed masks only (V10/V11).
    """

    def __init__(
        self,
        lammda: float = 1.0,
        n_max: int = 2000,
        drift_threshold: Optional[float] = 4.0,
        fast_update: bool = True,
        batch_size: int = 1,
        optimize_level: int = 0,
    ):
        super().__init__()
        self.lammda = lammda
        self.n_max = n_max
        self.drift_threshold = drift_threshold
        self.fast_update = fast_update
        self.batch_size = max(1, int(batch_size))
        self.optimize_level = optimize_level

        self._R: Optional[npt.NDArray[np.floating]] = None
        self._m: int = 0
        self._delta: Optional[npt.NDArray[np.floating]] = None
        self._Lower: Optional[npt.NDArray[np.floating]] = None
        self._NEB: Optional[npt.NDArray[np.floating]] = None
        self._NPB: Optional[npt.NDArray[np.floating]] = None
        self._NOM: Optional[npt.NDArray[np.floating]] = None
        self._weight: Optional[npt.NDArray[np.floating]] = None
        self._NDF: Optional[npt.NDArray[np.floating]] = None
        self._concept_mask: Optional[npt.NDArray[np.floating]] = None
        self._mask_D1: Optional[npt.NDArray[np.bool_]] = None
        self._mask_D2: Optional[npt.NDArray[np.bool_]] = None
        self._mask_D3: Optional[npt.NDArray[np.bool_]] = None
        self._base_nom_mean: float = 0.0
        self._base_nom_std: float = 1.0
        self._reset_count: int = 0
        self._n_added: int = 0
        self._batch_buffer: list = []

    # ── internal compute ──────────────────────────────────────────

    def _compute_delta(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        sigma = np.std(X, axis=0)
        sigma[sigma == 0] = 1e-6
        return sigma / self.lammda

    def _neighbourhood_matrix(
        self, X_query: npt.NDArray[np.floating], X_ref: npt.NDArray[np.floating], col: int
    ) -> npt.NDArray[np.floating]:
        dist = cdist(X_query[:, [col]], X_ref[:, [col]], metric="cityblock")
        return (dist <= self._delta[col]).astype(np.float64)

    def _compute_all(self, X: npt.NDArray[np.floating]):
        n = X.shape[0]
        m = self._m
        D1 = m / 3.0
        D2 = m / 2.0
        D3 = 0.9 * m

        Lower = np.zeros((m, n))
        RM_by_col = []
        cm = self._concept_mask if self._concept_mask is not None else np.ones(n)
        for col in range(m):
            RM = self._neighbourhood_matrix(X, X, col)
            RM_by_col.append(RM)
            Lower[col, :] = np.min(np.maximum(1.0 - RM, np.tile(cm, (n, 1))), axis=1)

        IB = np.ones((m, n)) - Lower
        NEB = np.min(IB, axis=0)
        NPB = IB - NEB

        D_tem = np.zeros((n, n))
        for col in range(m):
            D_tem += RM_by_col[col]
        NOM = m - D_tem

        weight = np.zeros((n, m))
        for col in range(m):
            RM = RM_by_col[col]
            for i in range(n):
                weight[i, col] = 1.0 - np.sqrt(np.sum(np.minimum(RM[i, :], 1.0)) / n)

        NDF = np.zeros((n, m))
        for i in range(n):
            nom_row = NOM[i, :]
            for col in range(m):
                neb_count = np.sum(np.minimum(NEB, nom_row <= D1))
                lower_count = np.sum(np.minimum(Lower[col, :], nom_row >= D3))
                npb_count = np.sum(np.minimum(NPB[col, :], nom_row >= D2))
                NDF[i, col] = (neb_count + lower_count + npb_count) / n

        return Lower, NEB, NPB, NOM, weight, NDF

    def _rebuild_masks(self):
        m = self._m
        D1 = m / 3.0
        D2 = m / 2.0
        D3 = 0.9 * m
        self._mask_D1 = (self._NOM <= D1)
        self._mask_D2 = (self._NOM >= D2)
        self._mask_D3 = (self._NOM >= D3)

    def _compute_scores_attr(self, X_query: npt.NDArray[np.floating]):
        if self.optimize_level == 3:
            return self._compute_scores_attr_opt2(X_query)
        if self.optimize_level >= 2:
            return self._compute_scores_attr_opt23(X_query)
        if self.optimize_level == 1:
            return self._compute_scores_attr_opt3(X_query)
        return self._compute_scores_attr_base(X_query)

    def _compute_scores_attr_base(self, X_query: npt.NDArray[np.floating]):
        ref_X = self._R
        n_ref = ref_X.shape[0]
        n_q = X_query.shape[0]
        m = self._m
        D1 = m / 3.0
        D2 = m / 2.0
        D3 = 0.9 * m

        scores = np.zeros(n_q)
        phi = np.zeros((n_q, m))

        for qi in range(n_q):
            x_q = X_query[qi:qi + 1]
            for col in range(m):
                dist = cdist(x_q[:, [col]], ref_X[:, [col]], metric="cityblock")
                RM = (dist <= self._delta[col]).astype(np.float64)[0]
                w_x = 1.0 - np.sqrt(np.sum(np.minimum(RM, 1.0)) / n_ref)

                neb_count = np.sum(np.minimum(self._NEB, self._NOM <= D1))
                lower_count = np.sum(np.minimum(self._Lower[col, :], self._NOM >= D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._NOM >= D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref

                phi[qi, col] = ndf_j * w_x / m
                scores[qi] += phi[qi, col]

        return scores, phi

    def _compute_scores_attr_opt2(self, X_query: npt.NDArray[np.floating]):
        ref_X = self._R
        n_ref = ref_X.shape[0]
        n_q = X_query.shape[0]
        m = self._m

        if self._mask_D1 is None:
            self._rebuild_masks()

        scores = np.zeros(n_q)
        phi = np.zeros((n_q, m))

        for qi in range(n_q):
            x_q = X_query[qi:qi + 1]
            for col in range(m):
                dist = cdist(x_q[:, [col]], ref_X[:, [col]], metric="cityblock")
                RM = (dist <= self._delta[col]).astype(np.float64)[0]
                w_x = 1.0 - np.sqrt(np.sum(np.minimum(RM, 1.0)) / n_ref)

                neb_count = np.sum(np.minimum(self._NEB, self._mask_D1))
                lower_count = np.sum(np.minimum(self._Lower[col, :], self._mask_D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._mask_D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref

                phi[qi, col] = ndf_j * w_x / m
                scores[qi] += phi[qi, col]

        return scores, phi

    def _compute_scores_attr_opt3(self, X_query: npt.NDArray[np.floating]):
        ref_X = self._R
        n_ref = ref_X.shape[0]
        n_q = X_query.shape[0]
        m = self._m
        D1 = m / 3.0
        D2 = m / 2.0
        D3 = 0.9 * m

        scores = np.zeros(n_q)
        phi = np.zeros((n_q, m))

        for qi in range(n_q):
            x_q = X_query[qi:qi + 1]
            neb_count = np.sum(np.minimum(self._NEB, self._NOM <= D1))
            for col in range(m):
                dist = cdist(x_q[:, [col]], ref_X[:, [col]], metric="cityblock")
                RM = (dist <= self._delta[col]).astype(np.float64)[0]
                w_x = 1.0 - np.sqrt(np.sum(np.minimum(RM, 1.0)) / n_ref)

                lower_count = np.sum(np.minimum(self._Lower[col, :], self._NOM >= D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._NOM >= D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref

                phi[qi, col] = ndf_j * w_x / m
                scores[qi] += phi[qi, col]

        return scores, phi

    def _compute_scores_attr_opt23(self, X_query: npt.NDArray[np.floating]):
        ref_X = self._R
        n_ref = ref_X.shape[0]
        n_q = X_query.shape[0]
        m = self._m

        if self._mask_D1 is None:
            self._rebuild_masks()

        scores = np.zeros(n_q)
        phi = np.zeros((n_q, m))

        for qi in range(n_q):
            x_q = X_query[qi:qi + 1]
            neb_count = np.sum(np.minimum(self._NEB, self._mask_D1))
            for col in range(m):
                dist = cdist(x_q[:, [col]], ref_X[:, [col]], metric="cityblock")
                RM = (dist <= self._delta[col]).astype(np.float64)[0]
                w_x = 1.0 - np.sqrt(np.sum(np.minimum(RM, 1.0)) / n_ref)

                lower_count = np.sum(np.minimum(self._Lower[col, :], self._mask_D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._mask_D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref

                phi[qi, col] = ndf_j * w_x / m
                scores[qi] += phi[qi, col]

        return scores, phi

    # ── V3: per-point incremental update ──────────────────────────

    def _incremental_update_one(self, x_new: npt.NDArray[np.floating]):
        n_old = self._R.shape[0] if self._R is not None else 0
        self._R = np.vstack([self._R, x_new]) if self._R is not None else x_new.copy()
        n_new = self._R.shape[0]
        m = self._m

        rm_new = np.zeros((m, n_old), dtype=np.float64)
        for col in range(m):
            dist = np.abs(self._R[:-1, col] - x_new[0, col])
            rm_new[col] = (dist <= self._delta[col]).astype(np.float64)

        for col in range(m):
            self._Lower[col] = np.minimum(self._Lower[col],
                                          np.maximum(1.0 - rm_new[col], 1.0))

        cm_old = self._concept_mask if self._concept_mask is not None else np.ones(n_old)
        new_lower = np.array([np.min(np.maximum(1.0 - rm_new[c], cm_old)) for c in range(m)])
        self._Lower = np.column_stack([self._Lower, new_lower.reshape(m, 1)])

        IB = 1.0 - self._Lower
        self._NEB = np.min(IB, axis=0)
        self._NPB = IB - self._NEB

        self._concept_mask = np.append(self._concept_mask, 1.0) if self._concept_mask is not None else None

        nom_new = m - rm_new.sum(axis=0)
        if self._NOM is not None and n_old > 0:
            self._NOM = np.vstack([
                np.column_stack([self._NOM, nom_new.reshape(-1, 1)]),
                np.concatenate([nom_new, [0.0]]).reshape(1, -1),
            ])
        else:
            self._NOM = np.zeros((n_new, n_new))

        if self.optimize_level >= 2 and self._NOM is not None:
            self._rebuild_masks()

    def _incremental_update_one_fused(self, x_new: npt.NDArray[np.floating]):
        n_old = self._R.shape[0] if self._R is not None else 0
        self._R = np.vstack([self._R, x_new]) if self._R is not None else x_new.copy()
        n_new = self._R.shape[0]
        m = self._m

        rm_new = np.zeros((m, n_old), dtype=np.float64)
        for col in range(m):
            dist = np.abs(self._R[:-1, col] - x_new[0, col])
            rm_new[col] = (dist <= self._delta[col]).astype(np.float64)

        for col in range(m):
            self._Lower[col] = np.minimum(self._Lower[col],
                                          np.maximum(1.0 - rm_new[col], 1.0))

        cm_old = self._concept_mask if self._concept_mask is not None else np.ones(n_old)
        new_lower = np.array([np.min(np.maximum(1.0 - rm_new[c], cm_old)) for c in range(m)])
        self._Lower = np.column_stack([self._Lower, new_lower.reshape(m, 1)])

        IB = 1.0 - self._Lower
        self._NEB = np.min(IB, axis=0)
        self._NPB = IB - self._NEB

        self._concept_mask = np.append(self._concept_mask, 1.0) if self._concept_mask is not None else None

        nom_new = m - rm_new.sum(axis=0)
        if self._NOM is not None and n_old > 0:
            self._NOM = np.vstack([
                np.column_stack([self._NOM, nom_new.reshape(-1, 1)]),
                np.concatenate([nom_new, [0.0]]).reshape(1, -1),
            ])
        else:
            self._NOM = np.zeros((n_new, n_new))

        if self.optimize_level >= 2 and self._NOM is not None:
            self._rebuild_masks()

        score, phi = self._score_from_rm_new(rm_new, n_new, n_old)
        return score, phi

    def _score_from_rm_new(self, rm_new, n_ref, n_old):
        m = self._m

        score = 0.0
        phi = np.zeros(m)

        if self.optimize_level >= 2:
            neb_count = np.sum(np.minimum(self._NEB, self._mask_D1))
            for col in range(m):
                rm_col_count = rm_new[col].sum() + 1.0
                w_x = 1.0 - np.sqrt(rm_col_count / n_ref)

                lower_count = np.sum(np.minimum(self._Lower[col, :], self._mask_D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._mask_D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref
                phi[col] = ndf_j * w_x / m
                score += phi[col]
        elif self.optimize_level == 1:
            D1, D2, D3 = m/3.0, m/2.0, 0.9*m
            neb_count = np.sum(np.minimum(self._NEB, self._NOM <= D1))
            for col in range(m):
                rm_col_count = rm_new[col].sum() + 1.0
                w_x = 1.0 - np.sqrt(rm_col_count / n_ref)

                lower_count = np.sum(np.minimum(self._Lower[col, :], self._NOM >= D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._NOM >= D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref
                phi[col] = ndf_j * w_x / m
                score += phi[col]
        else:
            D1, D2, D3 = m/3.0, m/2.0, 0.9*m
            for col in range(m):
                rm_col_count = rm_new[col].sum() + 1.0
                w_x = 1.0 - np.sqrt(rm_col_count / n_ref)

                neb_count = np.sum(np.minimum(self._NEB, self._NOM <= D1))
                lower_count = np.sum(np.minimum(self._Lower[col, :], self._NOM >= D3))
                npb_count = np.sum(np.minimum(self._NPB[col, :], self._NOM >= D2))
                ndf_j = (neb_count + lower_count + npb_count) / n_ref
                phi[col] = ndf_j * w_x / m
                score += phi[col]

        return score, phi

    # ── V4: batch incremental update ──────────────────────────────

    def _incremental_update_batch(self, X_batch: npt.NDArray[np.floating]):
        K = X_batch.shape[0]
        n_old = self._R.shape[0] if self._R is not None else 0
        self._R = np.vstack([self._R, X_batch]) if self._R is not None else X_batch.copy()
        n_new = self._R.shape[0]
        m = self._m

        if n_old == 0:
            self._Lower, self._NEB, self._NPB, self._NOM, self._weight, self._NDF = self._compute_all(self._R)
            if self.optimize_level >= 2:
                self._rebuild_masks()
            return

        rm_batch = np.zeros((m, K, n_old), dtype=np.float64)
        for col in range(m):
            dist = np.abs(self._R[:-K, col][np.newaxis, :] - X_batch[:, col][:, np.newaxis])
            rm_batch[col] = (dist <= self._delta[col]).astype(np.float64)

        for col in range(m):
            best_new = np.maximum(1.0 - rm_batch[col], 1.0)
            best_new_min = np.min(best_new, axis=0)
            self._Lower[col] = np.minimum(self._Lower[col], best_new_min)

        cm_old = self._concept_mask if self._concept_mask is not None else np.ones(n_old)
        lower_new = np.zeros((m, K))
        for col in range(m):
            for k in range(K):
                lower_new[col, k] = np.min(np.maximum(1.0 - rm_batch[col, k, :], cm_old))
        self._Lower = np.column_stack([self._Lower, lower_new])

        IB = 1.0 - self._Lower
        self._NEB = np.min(IB, axis=0)
        self._NPB = IB - self._NEB

        if self._concept_mask is not None:
            self._concept_mask = np.append(self._concept_mask, np.ones(K))
        else:
            self._concept_mask = np.ones(n_new)

        nom_new = m - rm_batch.sum(axis=0)
        if self._NOM is not None:
            self._NOM = np.vstack([
                np.column_stack([self._NOM, nom_new.T]),
                np.hstack([nom_new, np.zeros((K, K))]),
            ])
        else:
            self._NOM = np.zeros((n_new, n_new))

        if self.optimize_level >= 2:
            self._rebuild_masks()

    # ── V2/V5: full recompute ────────────────────────────────────

    def _full_recompute(self):
        self._Lower, self._NEB, self._NPB, self._NOM, self._weight, self._NDF = self._compute_all(self._R)
        if self._concept_mask is not None and len(self._concept_mask) != len(self._R):
            self._concept_mask = np.ones(len(self._R))
        if self.optimize_level >= 2:
            self._rebuild_masks()

    # ── drift / reset ─────────────────────────────────────────────

    def _check_drift(self, x_new: npt.NDArray[np.floating]) -> bool:
        if self.drift_threshold is None:
            return False
        if self._R is None or len(self._R) < 10:
            return False
        scores, _ = self._compute_scores_attr(x_new)
        nom_mean = float(np.mean(np.diag(self._NOM)))
        z = abs(nom_mean - self._base_nom_mean) / (self._base_nom_std + 1e-12)
        return z > self.drift_threshold

    def _reset(self):
        k = max(1, self.n_max // 5)
        self._R = self._R[k:]
        if self._concept_mask is not None:
            self._concept_mask = self._concept_mask[k:]
        self._delta = self._compute_delta(self._R)
        self._Lower, self._NEB, self._NPB, self._NOM, self._weight, self._NDF = self._compute_all(self._R)
        if self.optimize_level >= 2:
            self._rebuild_masks()
        self._reset_count += 1

    # ── public API ────────────────────────────────────────────────

    def fit(self, X: npt.NDArray[np.floating]) -> "IXWNCROD":
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] > self.n_max:
            X = X[-self.n_max:]
        self._m = X.shape[1]
        self.n_features_in_ = self._m
        self._R = X.copy()
        self._delta = self._compute_delta(X)
        self._concept_mask = np.ones(X.shape[0])
        self._Lower, self._NEB, self._NPB, self._NOM, self._weight, self._NDF = self._compute_all(X)
        if self.optimize_level >= 2:
            self._rebuild_masks()
        nom_vals = np.diag(self._NOM)
        self._base_nom_mean = float(np.mean(nom_vals))
        self._base_nom_std = float(np.std(nom_vals)) if float(np.std(nom_vals)) > 0 else 1e-6
        self._reset_count = 0
        self._n_added = 0
        self._batch_buffer = []
        self.fitted_ = True
        return self

    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        if not self.fitted_:
            raise RuntimeError("Detector not fitted.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        scores, _ = self._compute_scores_attr(X)
        return scores

    def predict_with_attribution(self, X: npt.NDArray[np.floating]):
        if not self.fitted_:
            raise RuntimeError("Detector not fitted.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return self._compute_scores_attr(X)

    def partial_fit(self, X: npt.NDArray[np.floating]) -> "IXWNCROD":
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        for i in range(X.shape[0]):
            x_i = X[i:i + 1]

            if self.batch_size <= 1:
                if self._check_drift(x_i):
                    self._reset()
                elif self._R is not None and self._R.shape[0] >= self.n_max:
                    self._reset()

                if self.fast_update:
                    self._incremental_update_one(x_i)
                else:
                    self._R = np.vstack([self._R, x_i]) if self._R is not None else x_i.copy()
                    self._full_recompute()
                self._n_added += 1
            else:
                self._batch_buffer.append(x_i.copy())
                if len(self._batch_buffer) >= self.batch_size:
                    self._flush_batch()

        return self

    def partial_fit_predict_explain(self, X: npt.NDArray[np.floating]):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]
        scores = np.zeros(n)
        phi = np.zeros((n, self._m))

        for i in range(n):
            x_i = X[i:i + 1]

            if self.batch_size <= 1:
                if self._check_drift(x_i):
                    self._reset()
                elif self._R is not None and self._R.shape[0] >= self.n_max:
                    self._reset()

                if self.fast_update:
                    s_i, p_i = self._incremental_update_one_fused(x_i)
                else:
                    self._R = np.vstack([self._R, x_i]) if self._R is not None else x_i.copy()
                    self._full_recompute()
                    s_i, p_i = self._compute_scores_attr(x_i)
                    s_i, p_i = s_i[0], p_i[0]
                self._n_added += 1
                scores[i] = s_i
                phi[i] = p_i
            else:
                self._batch_buffer.append(x_i.copy())
                if len(self._batch_buffer) >= self.batch_size or i == n - 1:
                    s_b, p_b = self._flush_batch_fused()
                    start = i - len(s_b) + 1
                    scores[start:start+len(s_b)] = s_b
                    phi[start:start+len(p_b)] = p_b

        return scores, phi

    def _flush_batch(self):
        if not self._batch_buffer:
            return
        X_batch = np.vstack(self._batch_buffer)
        K = len(self._batch_buffer)
        self._batch_buffer = []

        if self._R is not None and self._R.shape[0] + K >= self.n_max:
            self._reset()

        if self.fast_update:
            self._incremental_update_batch(X_batch)
        else:
            self._R = np.vstack([self._R, X_batch]) if self._R is not None else X_batch.copy()
            self._full_recompute()

        self._n_added += K

    def _flush_batch_fused(self):
        if not self._batch_buffer:
            return np.array([]), np.zeros((0, self._m))
        X_batch = np.vstack(self._batch_buffer)
        K = len(self._batch_buffer)
        self._batch_buffer = []

        if self._R is not None and self._R.shape[0] + K >= self.n_max:
            self._reset()

        if self.fast_update:
            self._incremental_update_batch(X_batch)
        else:
            self._R = np.vstack([self._R, X_batch]) if self._R is not None else X_batch.copy()
            self._full_recompute()

        self._n_added += K

        scores, phi = self._compute_scores_attr(X_batch)
        return scores, phi

    def flush(self):
        self._flush_batch()
        return self

    def get_attribution(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        _, phi = self._compute_scores_attr(X)
        return phi

    def get_attribution_subregions(self, X: npt.NDArray[np.floating]) -> dict:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        ref_X = self._R
        n_ref = ref_X.shape[0]
        n_q = X.shape[0]
        m = self._m
        D1 = m / 3.0
        D2 = m / 2.0
        D3 = 0.9 * m

        neb_phi = np.zeros((n_q, m))
        lower_phi = np.zeros((n_q, m))
        npb_phi = np.zeros((n_q, m))
        total_phi = np.zeros((n_q, m))

        for qi in range(n_q):
            x_q = X[qi:qi + 1]
            for col in range(m):
                dist = cdist(x_q[:, [col]], ref_X[:, [col]], metric="cityblock")
                RM = (dist <= self._delta[col]).astype(np.float64)[0]
                w_x = 1.0 - np.sqrt(np.sum(np.minimum(RM, 1.0)) / n_ref)

                neb_c = np.sum(np.minimum(self._NEB, self._NOM <= D1))
                low_c = np.sum(np.minimum(self._Lower[col, :], self._NOM >= D3))
                npb_c = np.sum(np.minimum(self._NPB[col, :], self._NOM >= D2))
                total = neb_c + low_c + npb_c
                if total > 0:
                    neb_phi[qi, col] = (neb_c / n_ref) * w_x / m
                    lower_phi[qi, col] = (low_c / n_ref) * w_x / m
                    npb_phi[qi, col] = (npb_c / n_ref) * w_x / m
                    total_phi[qi, col] = total * w_x / (n_ref * m)

        return {"NEB": neb_phi, "Lower": lower_phi, "NPB": npb_phi, "total": total_phi}

    def get_model_state(self) -> dict:
        state = super().get_model_state()
        state.update({
            "lammda": self.lammda, "n_max": self.n_max,
            "delta": self._delta.tolist() if self._delta is not None else None,
            "m_features": self._m, "n_reference": len(self._R) if self._R is not None else 0,
            "reset_count": self._reset_count,
            "batch_size": self.batch_size, "fast_update": self.fast_update,
            "optimize_level": self.optimize_level,
        })
        return state

    def set_model_state(self, state: dict) -> None:
        super().set_model_state(state)
        self.lammda = state.get("lammda", 1.0)
        self.n_max = state.get("n_max", 2000)
        self._m = state.get("m_features", 0)
        self._reset_count = state.get("reset_count", 0)
        self.batch_size = state.get("batch_size", 1)
        self.fast_update = state.get("fast_update", True)
        self.optimize_level = state.get("optimize_level", 0)
