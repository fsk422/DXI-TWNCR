"""Base abstractions: detector, explainer, and incremental-strategy interfaces."""
from .detector import BaseDetector
from .explainer import BaseExplainer, AttributionResult
from .incremental import BaseIncrementalStrategy

__all__ = ["BaseDetector", "BaseExplainer", "AttributionResult", "BaseIncrementalStrategy"]
