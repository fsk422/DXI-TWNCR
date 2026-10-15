"""LODA: lightweight on-line detector of anomalies."""
import numpy as np


class LODAIncremental:
    """LODA: Lightweight On-line Detector of Anomalies (Pevny 2016).

    Ensemble of 1-D histogram detectors on random sparse projections.
    Supports incremental histogram updates and feature attribution.

    Parameters
    ----------
    n_random_cuts : int
        Number of random projection vectors.
    n_bins : int
        Number of histogram bins.
    """

    def __init__(self, n_random_cuts=100, n_bins=10):
        self.n_random_cuts = n_random_cuts
        self.n_bins = n_bins
        self._projections = None
        self._histograms = None
        self._limits = None
        self._total_count = 0
        self._fitted = False

    def fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n, d = X.shape
        n_nonzero = max(1, int(np.sqrt(d)))
        n_zero = d - n_nonzero

        self._projections = np.random.randn(self.n_random_cuts, d)
        for i in range(self.n_random_cuts):
            zeros = np.random.permutation(d)[:n_zero]
            self._projections[i, zeros] = 0.0

        self._histograms = np.zeros((self.n_random_cuts, self.n_bins))
        self._limits = np.zeros((self.n_random_cuts, self.n_bins + 1))

        for i in range(self.n_random_cuts):
            proj_data = X.dot(self._projections[i])
            self._histograms[i], self._limits[i] = np.histogram(
                proj_data, bins=self.n_bins, density=False
            )
            self._histograms[i] = self._histograms[i].astype(np.float64) + 1e-12
            self._histograms[i] /= self._histograms[i].sum()

        self._total_count = n
        self._fitted = True
        return self

    def partial_fit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        for x in X:
            for i in range(self.n_random_cuts):
                proj_val = float(x.dot(self._projections[i]))
                bin_idx = np.searchsorted(
                    self._limits[i, :self.n_bins - 1], proj_val, side='left'
                )
                bin_idx = min(bin_idx, self.n_bins - 1)
                self._histograms[i, bin_idx] += 1.0
                self._histograms[i] /= self._histograms[i].sum()
            self._total_count += 1
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        scores = np.zeros(X.shape[0])
        for i in range(self.n_random_cuts):
            proj_data = X.dot(self._projections[i])
            bin_idx = np.searchsorted(
                self._limits[i, :self.n_bins - 1], proj_data, side='left'
            )
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            scores -= np.log(self._histograms[i, bin_idx])
        return scores / self.n_random_cuts

    def get_attribution(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples, n_features = X.shape
        scores = np.zeros(n_samples)
        per_proj_scores = np.zeros((n_samples, self.n_random_cuts))

        for i in range(self.n_random_cuts):
            proj_data = X.dot(self._projections[i])
            bin_idx = np.searchsorted(
                self._limits[i, :self.n_bins - 1], proj_data, side='left'
            )
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            per_proj_scores[:, i] = -np.log(self._histograms[i, bin_idx])

        total_scores = per_proj_scores.sum(axis=1)
        total_scores[total_scores == 0] = 1e-12
        proj_weights = per_proj_scores / total_scores[:, np.newaxis]

        contributions = np.zeros((n_samples, n_features))
        for i in range(self.n_random_cuts):
            contributions += proj_weights[:, i:i+1] * np.abs(self._projections[i])

        row_sums = contributions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12
        contributions /= row_sums
        contributions *= total_scores[:, np.newaxis]

        return contributions
