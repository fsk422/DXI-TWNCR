"""Three-way region attribution: NEB / NPB / Lower decomposition of MNOF."""
from typing import Optional
import time
import numpy as np
import numpy.typing as npt
from ..base.explainer import BaseExplainer, AttributionResult
from ..base.detector import BaseDetector


class ThreeWayRegionAttribution(BaseExplainer):
    """Intrinsic attribution via three-way region decomposition.

    This explainer decomposes the MNOF anomaly score into per-feature
    contributions based on the model's internal NEB / NPB / Lower
    three-way region structure.

    Mathematical formulation:
        MNOF(x) = (1/m) * sum_j [ (NEB_j + NPB_j + Lower_j) / |X| * w_j(x) ]

        phi_j(x) = (NEB_j + NPB_j + Lower_j) / (|X| * m) * w_j(x)

    Properties:
    - Completeness: sum_j phi_j = MNOF(x) — exact equality.
    - Intrinsic consistency: phi_j derived directly from the three-way
      partition used by the model, not from a surrogate.
    - Interpretability: each phi_j quantifies how much feature j contributes
      to the anomaly decision through each of the three regions.

    This explainer is specific to IXWNCROD-class detectors.
    """

    def __init__(self):
        super().__init__(name="ThreeWayRegionAttribution")

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
            raise TypeError(
                "ThreeWayRegionAttribution requires an IXWNCROD detector. "
                f"Got {type(detector).__name__}."
            )

        scores = detector.predict(X)
        times = np.zeros(n_samples)

        sub_phi = detector.get_attribution_subregions(X)
        contributions = sub_phi["total"]

        sub_contributions = {
            "NEB": sub_phi["NEB"],
            "NPB": sub_phi["NPB"],
            "Lower": sub_phi["Lower"],
        }

        completeness_gap = np.zeros(n_samples)
        if X_base is None:
            completeness_gap = np.abs(scores - np.sum(contributions, axis=1))
        else:
            f_base = detector.predict(np.atleast_2d(X_base))
            f_base_mean = np.mean(f_base) if f_base.ndim == 1 else float(f_base)
            completeness_gap = np.abs(
                scores - f_base_mean - np.sum(contributions, axis=1)
            )

        return AttributionResult(
            point_indices=indices,
            anomaly_scores=scores,
            contributions=contributions,
            sub_contributions=sub_contributions,
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
