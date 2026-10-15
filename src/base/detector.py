"""Abstract base interface for all anomaly detectors.

Defines the unified ``fit`` / ``predict`` / ``partial_fit`` contract used by
every detector (proposed and baseline) in the IXAD framework, plus optional
serialization hooks (``get_model_state`` / ``set_model_state``).
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import numpy.typing as npt


class BaseDetector(ABC):
    """Unified abstract interface for all anomaly detectors.

    All detectors (static and incremental) must implement this interface,
    ensuring consistent usage across the IXAD framework.

    Attributes
    ----------
    fitted_ : bool
        Whether the model has been fitted.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self):
        self.fitted_: bool = False
        self.n_features_in_: int = 0

    @abstractmethod
    def fit(self, X: npt.NDArray[np.floating]) -> "BaseDetector":
        """Fit the detector on initial data batch.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BaseDetector
        """
        ...

    @abstractmethod
    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Predict anomaly scores for given samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate more anomalous samples.
        """
        ...

    @abstractmethod
    def partial_fit(self, X: npt.NDArray[np.floating]) -> "BaseDetector":
        """Incrementally update the model with new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to incorporate.

        Returns
        -------
        self : BaseDetector
        """
        ...

    def fit_predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Fit and predict in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
        """
        self.fit(X)
        return self.predict(X)

    def get_model_state(self) -> dict:
        """Return a serializable snapshot of internal model state.

        Returns
        -------
        state : dict
            Model parameters and summary statistics.
        """
        return {"fitted": self.fitted_, "n_features": self.n_features_in_}

    def set_model_state(self, state: dict) -> None:
        """Restore model state from a previously saved snapshot.

        Parameters
        ----------
        state : dict
            Model state from get_model_state().
        """
        self.fitted_ = state.get("fitted", False)
        self.n_features_in_ = state.get("n_features", 0)
