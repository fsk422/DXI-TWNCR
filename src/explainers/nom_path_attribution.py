"""NOM-path attribution: per-feature disconnection from the normal reference."""
from typing import Optional
import numpy as np
import numpy.typing as npt
from ..base.explainer import BaseExplainer, AttributionResult
from ..base.detector import BaseDetector


class NOMPathAttribution(BaseExplainer):
    """Path-level attribution based on the Neighborhood Overlap Metric (NOM).

    Analyzes the NOM matrix to determine which features cause a sample
    to disconnect from the normal reference set.

    Mathematical formulation:
        phi_j^{NOM}(x) = |{x_k in X_ref : RM_j(x, x_k) = 0}| / |X_ref|

    where RM_j(x, x_k) is the neighborhood relation for feature j.
    A high phi_j means feature j causes x to be disconnected from
    many normal points, contributing strongly to its outlier status.

    Properties:
    - Interpretability: directly tells you which feature isolates the point.
    - Granularity: supports feature-level and pairwise-level analysis.
    """

    def __init__(self):
        super().__init__(name="NOMPathAttribution")

    def explain(
        self,
        detector: BaseDetector,
        X: npt.NDArray[np.floating],
        X_base: Optional[npt.NDArray[np.floating]] = None,
    ) -> AttributionResult:
        from ..detectors.ix_wncrod import IXWNCROD

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples, n_features = X.shape
        indices = np.arange(n_samples, dtype=np.int64)

        if not isinstance(detector, IXWNCROD):
            raise TypeError("NOMPathAttribution requires an IXWNCROD detector.")

        scores = detector.predict(X)
        times = np.zeros(n_samples)
        contributions = np.zeros((n_samples, n_features))

        ref_X = detector._slider.window_data
        delta = detector._delta
        use_fuzzy = detector.use_fuzzy

        for qi in range(n_samples):
            x_q = X[qi:qi + 1]
            for col in range(n_features):
                dist = np.abs(x_q[0, col] - ref_X[:, col])
                if use_fuzzy:
                    delta_val = float(delta[col]) if float(delta[col]) > 0 else 1e-6
                    RM = np.exp(-(dist ** 2) / (2 * delta_val ** 2))
                else:
                    RM = (dist <= delta[col]).astype(np.float64)

                disconnected = 1.0 - RM
                contributions[qi, col] = float(np.mean(disconnected))

        if scores is not None:
            contributions = contributions * scores[:, np.newaxis]

        completeness_gap = np.zeros(n_samples)

        return AttributionResult(
            point_indices=indices,
            anomaly_scores=scores,
            contributions=contributions,
            sub_contributions=None,
            explanation_method=self.name,
            explanation_time_ms=times,
            completeness_gap=completeness_gap,
        )

    def incremental_explain(
        self,
        detector: BaseDetector,
        X: npt.NDArray[np.floating],
        previous_result: Optional[AttributionResult] = None,
    ) -> AttributionResult:
        return self.explain(detector, X)
