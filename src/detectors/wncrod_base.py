"""Base (batch) WNCROD: three-way neighborhood characteristic-region detector.

Faithful batch reimplementation of the 3WNCROD outlier detector
(Zhang, Yuan & Miao, IEEE TKDE 2024). The proposed IXWNCROD builds on this
base by adding bounded-memory incremental updates and periodic reset.
"""
from typing import Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from ..base.detector import BaseDetector


class WNCRODBase(BaseDetector):
    """Original 3WNCROD outlier detector.

    Reproduced from:
    Zhang Xianyong, Yuan Zhong, Miao Duoqian.
    "Outlier Detection Using Three-Way Neighborhood Characteristic Regions
    and Corresponding Fusion Measurement." IEEE TKDE, 2024.

    This base version is the faithful batch-mode reimplementation with
    optional fuzzy neighborhood extension.

    Parameters
    ----------
    lammda : float, default=1.0
        Radius adjustment parameter. Smaller values produce larger radii.
    use_fuzzy : bool, default=False
        If True, use Gaussian fuzzy membership instead of hard threshold
        for neighborhood relations.
    """

    def __init__(self, lammda: float = 1.0, use_fuzzy: bool = False):
        super().__init__()
        self.lammda = lammda
        self.use_fuzzy = use_fuzzy
        self._X: Optional[npt.NDArray[np.floating]] = None
        self._delta: Optional[npt.NDArray[np.floating]] = None
        self._X_tem: Optional[npt.NDArray[np.int64]] = None
        self._n: int = 0
        self._m: int = 0
        self._weight: Optional[npt.NDArray[np.floating]] = None
        self._NOM: Optional[npt.NDArray[np.floating]] = None
        self._Lower: Optional[npt.NDArray[np.floating]] = None
        self._NPB: Optional[npt.NDArray[np.floating]] = None
        self._NEB: Optional[npt.NDArray[np.floating]] = None

    def _compute_delta(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        delta = np.zeros(self._m)
        id_numeric = np.all(X <= 1, axis=0)
        delta[id_numeric] = np.std(X[:, id_numeric], axis=0) / self.lammda
        return delta

    def _compute_neighborhood_matrix(
        self, X: npt.NDArray[np.floating], col: int
    ) -> npt.NDArray[np.floating]:
        if self.use_fuzzy:
            dist = cdist(X[:, [col]], X[:, [col]], metric="cityblock")
            delta_val = self._delta[col] if self._delta[col] > 0 else 0.01
            return np.exp(-(dist**2) / (2 * delta_val**2))
        else:
            return (cdist(X[:, [col]], X[:, [col]], metric="cityblock")
                    <= self._delta[col]).astype(np.float64)

    def fit(self, X: npt.NDArray[np.floating]) -> "WNCRODBase":
        self._X = np.asarray(X, dtype=np.float64)
        self._n, self._m = self._X.shape
        self.n_features_in_ = self._m
        self._X_tem = np.arange(self._n, dtype=np.int64)
        self._delta = self._compute_delta(self._X)
        D1 = self._m / 3
        D2 = self._m / 2
        D3 = 0.9 * self._m

        Lower = np.zeros((self._m, self._n))
        for col in range(self._m):
            RM = self._compute_neighborhood_matrix(self._X, col)
            if self.use_fuzzy:
                Lower[col, :] = np.min(np.maximum(1.0 - RM, 1.0), axis=1)
            else:
                X_mask = np.ones(self._n)
                Lower[col, :] = np.min(np.maximum(1.0 - RM, X_mask), axis=1)

        IB = np.ones((self._m, self._n)) - Lower
        NEB = np.min(IB, axis=0)
        NPB = IB - NEB

        norm = self._X.shape[0]
        self._weight = np.zeros((norm, self._m))
        for col in range(self._m):
            RM = self._compute_neighborhood_matrix(self._X, col)
            for i in range(norm):
                temp1 = RM[i, :]
                w_temp = 1.0 - np.sqrt(np.sum(np.minimum(temp1, 1.0)) / norm)
                self._weight[i, col] = w_temp

        D_tem = np.zeros((self._n, self._n))
        for col in range(self._m):
            RM = self._compute_neighborhood_matrix(self._X, col)
            if self.use_fuzzy:
                D_tem += RM
            else:
                D_tem += RM

        self._NOM = self._m - D_tem
        X_OM = self._NOM

        n_X = self._n
        NEB_num = np.zeros((n_X, self._m))
        Lower_num = np.zeros((n_X, self._m))
        NPB_num = np.zeros((n_X, self._m))

        for col in range(self._m):
            temp2 = Lower[col, :]
            temp3 = NPB[col, :]
            for i in range(n_X):
                temp1 = X_OM[i, :]
                NEB_num[i, col] = np.sum(np.minimum(NEB, temp1 <= D1))
                NPB_num[i, col] = np.sum(np.minimum(temp3, temp1 >= D2))
                Lower_num[i, col] = np.sum(np.minimum(temp2, temp1 >= D3))

        self._Lower = (NEB_num + Lower_num + NPB_num) / n_X
        self.fitted_ = True
        return self

    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        if not self.fitted_:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for idx in range(n_samples):
            x = X[idx:idx + 1]
            MNOF_sum = 0.0
            for col in range(self._m):
                dist = cdist(x[:, [col]], self._X[:, [col]], metric="cityblock")
                if self.use_fuzzy:
                    delta_val = self._delta[col] if self._delta[col] > 0 else 0.01
                    RM = np.exp(-(dist**2) / (2 * delta_val**2))[0]
                else:
                    RM = (dist <= self._delta[col]).astype(np.float64)[0]

                weight_x = 1.0 - np.sqrt(np.sum(np.minimum(RM, 1.0)) / self._n)
                MNOF_sum += self._Lower[0, col] * weight_x

            scores[idx] = MNOF_sum / self._m if self._m > 0 else 0.0

        return scores

    def partial_fit(self, X: npt.NDArray[np.floating]) -> "WNCRODBase":
        self._n = X.shape[0]
        self._m = X.shape[1]
        self.fit(X)
        return self

    def get_model_state(self) -> dict:
        state = super().get_model_state()
        state.update({
            "lammda": self.lammda,
            "use_fuzzy": self.use_fuzzy,
            "delta": self._delta.tolist() if self._delta is not None else None,
            "m_features": self._m,
            "n_samples": self._n,
        })
        return state

    def set_model_state(self, state: dict) -> None:
        super().set_model_state(state)
        self.lammda = state.get("lammda", 1.0)
        self.use_fuzzy = state.get("use_fuzzy", False)
        self._m = state.get("m_features", 0)
        self._n = state.get("n_samples", 0)
