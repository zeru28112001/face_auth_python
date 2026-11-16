from dataclasses import dataclass
from typing import Literal
import numpy as np

from db.base import BaseFaceRepository
from .detector import DetectedFace
from .recognizer import FaceRecognizer


@dataclass
class AuthResult:
    """Step 5: Authentication outcome."""
    status: Literal["GRANTED", "DENIED", "NO_FACE"]
    name: str
    similarity_score: float
    threshold: float
    detected_face: DetectedFace | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.status == "GRANTED"


class FaceAuthenticator:
    """Step 5: Authentication Engine.
    
    Orchestrates Feature Extraction, Matching against Database, and
    making the final Access Granted / Denied determination.
    """

    # SFace cosine similarity recommended threshold is typically ~0.363 - 0.40
    DEFAULT_COSINE_THRESHOLD = 0.363

    def __init__(
        self,
        recognizer: FaceRecognizer,
        repository: BaseFaceRepository,
        threshold: float = DEFAULT_COSINE_THRESHOLD,
    ):
        self.recognizer = recognizer
        self.repository = repository
        self.threshold = threshold
        # Cache of known embeddings to avoid DB reads on every single frame
        self._cached_faces: list[tuple[str, np.ndarray]] = []
        self.reload_faces()

    def reload_faces(self) -> int:
        """Reloads face records from repository into memory cache."""
        records = self.repository.get_all_faces()
        self._cached_faces = [(r.name, r.embedding) for r in records]
        return len(self._cached_faces)

    def authenticate_face(
        self,
        image: np.ndarray,
        detected_face: DetectedFace,
    ) -> AuthResult:
        """Step 3 -> Step 4 -> Step 5:
        1. Extract embedding vector (Step 3)
        2. Match against registered faces in database (Step 4)
        3. Determine Access Granted or Denied based on threshold (Step 5)
        """
        # Step 3: Feature Extraction
        query_embedding = self.recognizer.extract_feature(image, detected_face)

        # Step 4: Face Matching
        best_name, best_score = self.recognizer.find_best_match(
            query_embedding,
            self._cached_faces,
        )

        # Step 5: Authentication Decision
        if best_name is not None and best_score >= self.threshold:
            return AuthResult(
                status="GRANTED",
                name=best_name,
                similarity_score=best_score,
                threshold=self.threshold,
                detected_face=detected_face,
            )
        else:
            return AuthResult(
                status="DENIED",
                name=best_name or "Unknown",
                similarity_score=best_score,
                threshold=self.threshold,
                detected_face=detected_face,
            )

    def check_existing_face(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Checks if the given embedding matches any registered face in the database."""
        return self.recognizer.find_best_match(embedding, self._cached_faces)

    def register_new_face(
        self,
        image: np.ndarray,
        detected_face: DetectedFace,
        name: str,
        allow_duplicate: bool = False,
    ) -> tuple[bool, str]:
        """Extracts embedding from detected face and persists it to repository.
        
        Performs a pre-registration check to prevent duplicate face registration or identity conflicts.
        """
        clean_name = name.strip()
        if not clean_name:
            return False, "Name cannot be empty."

        # Step 3: Feature Extraction
        embedding = self.recognizer.extract_feature(image, detected_face)

        # Pre-registration check: Does this face already exist in the database?
        matched_name, similarity = self.check_existing_face(embedding)

        if matched_name is not None and similarity >= self.threshold:
            pct = int(round(similarity * 100))
            if not allow_duplicate:
                return (
                    False,
                    f"Face already registered as '{matched_name}' ({pct}% match).",
                )

        # Persist to database (SQLite / MongoDB)
        record_id = self.repository.add_face(clean_name, embedding)

        # Refresh in-memory cache
        self.reload_faces()
        return True, f"Successfully registered '{clean_name}' (ID: {record_id})."
