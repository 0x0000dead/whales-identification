"""Inference subpackage for cetacean identification.

Public entry point: ``get_pipeline()`` returns a lazily-constructed
``InferencePipeline`` singleton used by the FastAPI lifespan.
"""

from .registry import get_pipeline
from .schemas import GateResult, PredictionResult, RejectionReason

__all__ = ["get_pipeline", "GateResult", "PredictionResult", "RejectionReason"]
