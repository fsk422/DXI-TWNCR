"""Synthetic streaming-data generator with ground-truth attribution masks.

Produces multivariate streams with burst / periodic / mixed anomalies and
gradual concept drift, together with binary labels and per-feature masks that
identify which features genuinely caused each anomaly — enabling evaluation of
both detection performance and explanation quality.
"""
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt


class StreamDataGenerator:
    """Configurable streaming data generator for outlier detection.

    Generates multivariate time-series data with:
    - Burst anomalies: sudden spikes in specific features.
    - Gradual concept drift: slow shifting of normal distribution.
    - Periodic anomalies: cyclical anomalous patterns.
    - Ground-truth anomaly labels and *intrinsic attribution* masks
      (which features genuinely caused each anomaly).

    Mathematical model:
        x_t = baseline_t + drift_t + anomaly_t + noise_t

    where:
    - baseline_t ~ N(mu, Sigma) is the normal regime
    - drift_t = alpha * t/T is linear concept drift
    - anomaly_t is injected according to the specified pattern
    - noise_t ~ N(0, sigma_noise) is i.i.d. observation noise
    """

    def __init__(
        self,
        n_features: int = 6,
        n_base_samples: int = 500,
        n_stream_samples: int = 2000,
        anomaly_rate: float = 0.05,
        drift_scale: float = 0.0,
        burst_magnitude: float = 5.0,
        periodic_amplitude: float = 3.0,
        periodic_period: int = 200,
        noise_std: float = 0.1,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.n_base_samples = n_base_samples
        self.n_stream_samples = n_stream_samples
        self.anomaly_rate = anomaly_rate
        self.drift_scale = drift_scale
        self.burst_magnitude = burst_magnitude
        self.periodic_amplitude = periodic_amplitude
        self.periodic_period = periodic_period
        self.noise_std = noise_std
        self.rng = np.random.RandomState(random_state)

        self._normal_mean: Optional[npt.NDArray[np.floating]] = None
        self._normal_cov: Optional[npt.NDArray[np.floating]] = None
        self._attribution_mask: Optional[npt.NDArray[np.bool_]] = None
        self._generated: bool = False

    def _generate_correlation_matrix(self, d: int) -> npt.NDArray[np.floating]:
        A = self.rng.randn(d, d)
        cov = A @ A.T
        D = np.sqrt(np.diag(cov))
        corr = cov / np.outer(D, D)
        return corr

    def generate(self) -> Tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
    ]:
        """Generate the complete streaming dataset.

        Returns
        -------
        X : np.ndarray of shape (n_total, n_features)
            All data points (base + stream).
        y : np.ndarray of shape (n_total,)
            Ground-truth anomaly scores (0 = normal, >0 = degree of anomaly).
        labels : np.ndarray of shape (n_total,)
            Binary labels (1 = anomaly, 0 = normal).
        attr_mask : np.ndarray of shape (n_total, n_features)
            Which features genuinely caused each anomaly (bool).
        """
        d = self.n_features
        total_n = self.n_base_samples + self.n_stream_samples

        self._normal_mean = self.rng.randn(d) * 0.5
        corr = self._generate_correlation_matrix(d)
        scales = self.rng.uniform(0.5, 1.5, d)
        self._normal_cov = np.diag(scales) @ corr @ np.diag(scales)

        X_normal = self.rng.multivariate_normal(
            self._normal_mean, self._normal_cov, total_n
        )

        X = X_normal.copy()
        y = np.zeros(total_n)
        labels = np.zeros(total_n, dtype=np.int64)
        attr_mask = np.zeros((total_n, d), dtype=bool)

        for t in range(total_n):
            is_anomaly = self.rng.rand() < self.anomaly_rate
            is_burst = is_anomaly and self.rng.rand() < 0.4
            is_periodic = is_anomaly and not is_burst and self.rng.rand() < 0.5
            is_mixed = is_anomaly and not is_burst and not is_periodic

            drift = self.drift_scale * t / total_n
            X[t] += drift

            if is_burst:
                n_affected = max(1, self.rng.randint(1, d // 2 + 1))
                affected = self.rng.choice(d, n_affected, replace=False)
                sign = self.rng.choice([-1, 1])
                X[t, affected] += sign * self.burst_magnitude * self.rng.uniform(0.5, 1.5, n_affected)
                y[t] = float(np.mean(np.abs(X[t, affected] - X_normal[t, affected])))
                labels[t] = 1
                attr_mask[t, affected] = True

            elif is_periodic:
                phase = 2 * np.pi * t / self.periodic_period
                n_affected = max(1, self.rng.randint(1, d // 3 + 1))
                affected = self.rng.choice(d, n_affected, replace=False)
                X[t, affected] += self.periodic_amplitude * np.sin(phase)
                y[t] = float(np.mean(np.abs(X[t, affected] - X_normal[t, affected])))
                labels[t] = 1
                attr_mask[t, affected] = True

            elif is_mixed:
                n_affected = max(1, self.rng.randint(1, d))
                affected = self.rng.choice(d, n_affected, replace=False)
                X[t, affected] += self.rng.randn(n_affected) * self.burst_magnitude * 0.7
                y[t] = float(np.mean(np.abs(X[t, affected] - X_normal[t, affected])))
                labels[t] = 1
                attr_mask[t, affected] = True

            X[t] += self.rng.randn(d) * self.noise_std

        self._attribution_mask = attr_mask
        self._generated = True

        return X, y, labels, attr_mask

    def get_stream_batch(
        self, start_idx: int = 0
    ) -> Tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
    ]:
        """Get a batch from the generated stream."""
        if not self._generated:
            raise RuntimeError("Call generate() first.")
        raise NotImplementedError("Regenerate or reload full data.")

    @property
    def attribution_mask(self) -> Optional[npt.NDArray[np.bool_]]:
        """Ground-truth attribution mask: which features caused each anomaly."""
        return self._attribution_mask

    @property
    def normal_mean(self) -> Optional[npt.NDArray[np.floating]]:
        return self._normal_mean

    @property
    def normal_cov(self) -> Optional[npt.NDArray[np.floating]]:
        return self._normal_cov
