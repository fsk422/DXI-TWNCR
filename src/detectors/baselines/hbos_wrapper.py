"""HBOS: histogram-based outlier score."""
import numpy as np


class HBOSIncremental:
    """Histogram-Based Outlier Score (Goldstein & Dengel 2012).

    Builds per-feature histograms independently. Anomaly score is
    the negative log-density sum across features. Supports incremental
    histogram updates and provides per-feature contributions.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins per feature.
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self._histograms = None
        self._limits = None
        self._total_count = 0
        self._fitted = False

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        _, d = X.shape
        self._histograms = np.zeros((d, self.n_bins))
        self._limits = np.zeros((d, self.n_bins + 1))
        for j in range(d):
            self._histograms[j], self._limits[j] = np.histogram(
                X[:, j], bins=self.n_bins, density=False
            )
            self._histograms[j] = self._histograms[j].astype(np.float64) + 1e-12
            self._histograms[j] /= self._histograms[j].sum()
        self._total_count = X.shape[0]
        self._fitted = True
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for x in X:
            for j in range(len(self._histograms)):
                bin_idx = np.searchsorted(
                    self._limits[j, :self.n_bins - 1], x[j], side='left'
                )
                bin_idx = min(bin_idx, self.n_bins - 1)
                self._histograms[j, bin_idx] += 1.0
                self._histograms[j] /= self._histograms[j].sum()
            self._total_count += 1
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        scores = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            bin_idx = np.searchsorted(
                self._limits[j, :self.n_bins - 1], X[:, j], side='left'
            )
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            scores -= np.log(self._histograms[j, bin_idx])
        return scores

    def get_attribution(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples, n_features = X.shape
        contributions = np.zeros((n_samples, n_features))
        for j in range(n_features):
            bin_idx = np.searchsorted(
                self._limits[j, :self.n_bins - 1], X[:, j], side='left'
            )
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            contributions[:, j] = -np.log(self._histograms[j, bin_idx])
        return contributions
