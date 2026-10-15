"""Abstract base class for incremental update strategies.

Encodes the generic update operator ``theta_{t+1} = U(theta_t, x_{t+1})``
used by sliding-window and online update strategies in the IXAD framework.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import numpy.typing as npt


class BaseIncrementalStrategy(ABC):
    """Abstract base for incremental update strategies.

    Provides the generic update operator:
        θ_{t+1} = U(θ_t, x_{t+1}, M_t)

    where M_t is the summary data structure maintained by the strategy.

    Supported strategies:
    - SlidingWindow: fixed-size window, discard oldest on new arrival.
    - OnlineStream: O(1) or O(log n) per-point update of statistics.
    - AdaptiveThreshold: EWMA-based dynamic threshold.
    """

    def __init__(self, name: str = "BaseIncremental"):
        self.name = name
        self._n_updates: int = 0
        self._update_times_ms: list[float] = []

    @abstractmethod
    def update(
        self,
        detector_state: dict,
        X_new: npt.NDArray[np.floating],
        X_expired: Optional[npt.NDArray[np.floating]] = None,
    ) -> dict:
        """Apply incremental update to detector state.

        Parameters
        ----------
        detector_state : dict
            Current internal state of the detector.
        X_new : np.ndarray of shape (n_new, n_features)
            Newly arrived data.
        X_expired : np.ndarray of shape (n_expired, n_features), optional
            Data to remove (for sliding window strategies).

        Returns
        -------
        new_state : dict
            Updated detector state.
        """
        ...

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def avg_update_time_ms(self) -> float:
        if not self._update_times_ms:
            return 0.0
        return float(np.mean(self._update_times_ms))

    def get_statistics(self) -> dict:
        """Return strategy statistics."""
        return {
            "strategy": self.name,
            "n_updates": self._n_updates,
            "avg_update_time_ms": self.avg_update_time_ms,
        }
