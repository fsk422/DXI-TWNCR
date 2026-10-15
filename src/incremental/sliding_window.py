"""Sliding-window incremental update strategy."""
from typing import Optional
import time
import numpy as np
import numpy.typing as npt
from ..base.incremental import BaseIncrementalStrategy


class SlidingWindowIncremental(BaseIncrementalStrategy):
    """Sliding window incremental update strategy.

    Maintains a fixed-size window W of recent data points.
    On each update:
    - New points are appended
    - If window exceeds capacity, oldest points are expired

    Mathematical formulation:
        W_{t+1} = W_t \ {x_expired} \cup {x_new}
        |W_{t+1}| = W (constant)

    Complexity per update: O(W * m) for full recomputation strategy,
    or O(m) for sketch-based approximate update.
    """

    def __init__(self, window_size: int = 256):
        super().__init__(name="SlidingWindow")
        self.window_size = window_size
        self._window_data: Optional[npt.NDArray[np.floating]] = None
        self._window_indices: list[int] = []
        self._global_counter: int = 0

    @property
    def current_size(self) -> int:
        if self._window_data is None:
            return 0
        return self._window_data.shape[0]

    @property
    def window_data(self) -> Optional[npt.NDArray[np.floating]]:
        return self._window_data

    def update(
        self,
        detector_state: dict,
        X_new: npt.NDArray[np.floating],
        X_expired: Optional[npt.NDArray[np.floating]] = None,
    ) -> dict:
        t_start = time.perf_counter()

        if self._window_data is None:
            self._window_data = np.atleast_2d(X_new).copy()
            self._window_indices = list(range(X_new.shape[0]))
            self._global_counter = X_new.shape[0]
        else:
            X_new = np.atleast_2d(X_new)
            n_new = X_new.shape[0]

            self._window_data = np.vstack([self._window_data, X_new])
            self._window_indices.extend(
                range(self._global_counter, self._global_counter + n_new)
            )
            self._global_counter += n_new

            overflow = self._window_data.shape[0] - self.window_size
            if overflow > 0:
                X_expired = self._window_data[:overflow].copy()
                self._window_indices = self._window_indices[overflow:]
                self._window_data = self._window_data[overflow:]

        detector_state["window_data"] = self._window_data.copy()
        detector_state["window_indices"] = list(self._window_indices)
        detector_state["window_size_current"] = self._window_data.shape[0]

        self._n_updates += 1
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._update_times_ms.append(elapsed_ms)

        return detector_state

    def get_expired_data(self, n_expired: int) -> Optional[npt.NDArray[np.floating]]:
        """Get the oldest n_expired data points without removing them."""
        if self._window_data is None or n_expired <= 0:
            return None
        return self._window_data[:n_expired].copy()

    def reset(self) -> None:
        self._window_data = None
        self._window_indices = []
        self._global_counter = 0
        self._n_updates = 0
        self._update_times_ms = []
