"""
Shared pipeline singletons loaded once at startup and injected via FastAPI dependency.
Avoids re-loading heavy ONNX models on every request.
"""
import os
from functools import lru_cache

from db import get_repository
from db.base import BaseFaceRepository
from face_pipeline import (
    FaceDetector,
    FaceRecognizer,
    FaceAuthenticator,
    LivenessDetector,
    KYCVerifier,
)


class PipelineState:
    """Holds all loaded pipeline objects. Instantiated once at startup."""

    def __init__(self, db_backend: str = "sqlite", mongo_uri: str = "mongodb://localhost:27017"):
        self.db_backend = db_backend
        self.mongo_uri = mongo_uri
        self.repo: BaseFaceRepository = get_repository(
            backend=db_backend,
            mongo_uri=mongo_uri,
        )
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.authenticator = FaceAuthenticator(
            recognizer=self.recognizer,
            repository=self.repo,
            threshold=float(os.getenv("AUTH_THRESHOLD", "0.363")),
        )
        self.liveness_detector = LivenessDetector(
            liveness_threshold=float(os.getenv("LIVENESS_THRESHOLD", "0.50")),
        )
        self.kyc_verifier = KYCVerifier(
            detector=self.detector,
            recognizer=self.recognizer,
            authenticator=self.authenticator,
            repository=self.repo,
            liveness_detector=self.liveness_detector,
            kyc_threshold=float(os.getenv("AUTH_THRESHOLD", "0.363")),
        )


# Module-level singleton (populated during app startup)
_pipeline: PipelineState | None = None


def init_pipeline(db_backend: str = "sqlite", mongo_uri: str = "mongodb://localhost:27017") -> PipelineState:
    global _pipeline
    _pipeline = PipelineState(db_backend=db_backend, mongo_uri=mongo_uri)
    return _pipeline


def get_pipeline() -> PipelineState:
    """FastAPI dependency: returns the loaded pipeline singleton."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() at startup.")
    return _pipeline
