"""Abstract interface and data container for attribution-based explanation.

Provides ``AttributionResult`` (structured per-attribute contributions with
completeness-gap tracking) and ``BaseExplainer`` (the intrinsic attribution
API: ``explain`` / ``incremental_explain`` / completeness & stability checks).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass
class AttributionResult:
    """Container for per-sample attribute-level attribution results.

    Attributes
    ----------
    point_indices : np.ndarray of shape (n_samples,)
        Indices of explained points.
    anomaly_scores : np.ndarray of shape (n_samples,)
        Original anomaly scores from the detector.
    contributions : np.ndarray of shape (n_samples, n_features)
        Attribute-level contributions. contribution[i, j] is the contribution
        of feature j to the anomaly score of sample i.
    sub_contributions : Optional dict
        Optional fine-grained decomposition, e.g.,
        {"NEB": array, "Lower": array, "NPB": array} for 3W-Region.
    explanation_method : str
        Name of the explanation method used.
    explanation_time_ms : np.ndarray of shape (n_samples,)
        Time taken per sample for explanation (milliseconds).
    completeness_gap : np.ndarray of shape (n_samples,)
        |f(x) - f(base) - sum(phi_j)|, measuring the completeness deviation.
        Zero for perfectly complete methods.
    """

    point_indices: npt.NDArray[np.int64]
    anomaly_scores: npt.NDArray[np.floating]
    contributions: npt.NDArray[np.floating]
    sub_contributions: Optional[dict] = None
    explanation_method: str = "unknown"
    explanation_time_ms: Optional[npt.NDArray[np.floating]] = None
    completeness_gap: Optional[npt.NDArray[np.floating]] = None

    @property
    def n_samples(self) -> int:
        return len(self.point_indices)

    @property
    def n_features(self) -> int:
        return self.contributions.shape[1]

    def top_features(self, sample_idx: int, k: int = 3) -> list[tuple[int, float]]:
        """Return top-k features by contribution magnitude for a sample."""
        idx = np.argsort(np.abs(self.contributions[sample_idx]))[::-1][:k]
        return [(int(i), float(self.contributions[sample_idx, i])) for i in idx]

    def to_dict(self) -> list[dict]:
        """Convert to list of dicts, one per sample."""
        results = []
        for i in range(self.n_samples):
            entry = {
                "point_index": int(self.point_indices[i]),
                "anomaly_score": float(self.anomaly_scores[i]),
                "contributions": {
                    f"feat{j}": float(self.contributions[i, j])
                    for j in range(self.n_features)
                },
                "explanation_method": self.explanation_method,
            }
            if self.explanation_time_ms is not None:
                entry["explanation_time_ms"] = float(self.explanation_time_ms[i])
            if self.completeness_gap is not None:
                entry["completeness_gap"] = float(self.completeness_gap[i])
            if self.sub_contributions:
                entry["sub_contributions"] = {
                    region: {
                        f"feat{j}": float(self.sub_contributions[region][i, j])
                        for j in range(self.n_features)
                    }
                    for region in self.sub_contributions
                }
            results.append(entry)
        return results


class BaseExplainer(ABC):
    """Unified abstract interface for attribute-level anomaly explanation.

    Core principle: explanations must be *intrinsic* — derived from the
    model's internal structure (tree paths, network gradients, distance
    components), not from post-hoc proxy models.

    All explainers must support completeness check:
        |- f(x) - f(x_base) - sum(phi_j) -|  < epsilon
    """

    def __init__(self, name: str = "BaseExplainer"):
        self.name = name

    @abstractmethod
    def explain(
        self,
        detector: "BaseDetector",
        X: npt.NDArray[np.floating],
        X_base: Optional[npt.NDArray[np.floating]] = None,
    ) -> AttributionResult:
        """Compute attribute-level contributions for given samples.

        Parameters
        ----------
        detector : BaseDetector
            The fitted anomaly detector to explain.
        X : np.ndarray of shape (n_samples, n_features)
            Samples to explain.
        X_base : np.ndarray of shape (n_base, n_features), optional
            Reference point(s) for baseline. Defaults to normal sample mean.

        Returns
        -------
        result : AttributionResult
            Structured attribution results.
        """
        ...

    @abstractmethod
    def incremental_explain(
        self,
        detector: "BaseDetector",
        X: npt.NDArray[np.floating],
        previous_result: Optional[AttributionResult] = None,
    ) -> AttributionResult:
        """Compute attributions efficiently after incremental model update.

        Parameters
        ----------
        detector : BaseDetector
            The updated detector.
        X : np.ndarray of shape (n_samples, n_features)
            Samples to explain.
        previous_result : AttributionResult, optional
            Previous attribution result for efficient update.

        Returns
        -------
        result : AttributionResult
        """
        ...

    def compute_completeness(
        self, detector: "BaseDetector", X: npt.NDArray[np.floating],
        contributions: npt.NDArray[np.floating],
        X_base: Optional[npt.NDArray[np.floating]] = None,
    ) -> npt.NDArray[np.floating]:
        """Compute completeness gap: |f(x) - f(x_base) - sum(phi_j)|.

        Parameters
        ----------
        detector : BaseDetector
        X : np.ndarray
        contributions : np.ndarray
            Per-feature contributions.
        X_base : np.ndarray, optional

        Returns
        -------
        gap : np.ndarray of shape (n_samples,)
        """
        f_x = detector.predict(X)
        if X_base is None:
            f_base = 0.0
        else:
            f_base = detector.predict(X_base)
            if f_base.ndim == 1:
                f_base = np.mean(f_base)
        sum_phi = np.sum(contributions, axis=1)
        return np.abs(f_x - f_base - sum_phi)

    def compute_stability(
        self,
        detector: "BaseDetector",
        X: npt.NDArray[np.floating],
        noise_scale: float = 0.01,
        n_trials: int = 10,
    ) -> npt.NDArray[np.floating]:
        """Measure attribution stability via Jensen-Shannon divergence
        under small input perturbations.

        Parameters
        ----------
        detector : BaseDetector
        X : np.ndarray of shape (n_samples, n_features)
        noise_scale : float
            Standard deviation of Gaussian noise relative to feature std.
        n_trials : int
            Number of perturbation trials.

        Returns
        -------
        js_divergences : np.ndarray of shape (n_samples,)
            Per-sample JS divergence of attribution under perturbations.
        """
        rng = np.random.RandomState(42)
        base_result = self.explain(detector, X)
        base_phi = np.abs(base_result.contributions)
        base_phi = base_phi / (np.sum(base_phi, axis=1, keepdims=True) + 1e-12)

        js_divs = np.zeros(X.shape[0])
        std = np.std(X, axis=0) * noise_scale

        for _ in range(n_trials):
            X_noisy = X + rng.randn(*X.shape) * std
            noisy_result = self.explain(detector, X_noisy)
            noisy_phi = np.abs(noisy_result.contributions)
            noisy_phi = noisy_phi / (np.sum(noisy_phi, axis=1, keepdims=True) + 1e-12)

            m = 0.5 * (base_phi + noisy_phi)
            kl_base = np.sum(base_phi * np.log((base_phi + 1e-12) / (m + 1e-12)), axis=1)
            kl_noisy = np.sum(noisy_phi * np.log((noisy_phi + 1e-12) / (m + 1e-12)), axis=1)
            js_divs += 0.5 * (kl_base + kl_noisy)

        return js_divs / n_trials
