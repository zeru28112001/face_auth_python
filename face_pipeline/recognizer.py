from pathlib import Path
from typing import Sequence
import cv2
import numpy as np

from .detector import DetectedFace


class FaceRecognizer:
    """Step 3 (Feature Extraction) and Step 4 (Face Matching) using OpenCV SFace."""

    def __init__(self, model_path: str = "models/face_recognition_sface.onnx"):
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"SFace model not found at '{model_path}'. "
                "Ensure face_recognition_sface.onnx is downloaded into the models directory."
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(model_file),
            config="",
        )

    def extract_feature(self, image: np.ndarray, detected_face: DetectedFace) -> np.ndarray:
        """Step 3: Feature Extraction.
        
        Aligns the face using landmark geometry and extracts a 128-dimensional embedding vector.
        """
        # Align face based on landmarks
        aligned_face = self.recognizer.alignCrop(image, detected_face.raw_data)
        # Extract 128D L2-normalized feature vector
        embedding = self.recognizer.feature(aligned_face)
        return embedding

    def match(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Step 4: Face Matching.
        
        Computes cosine similarity between two 128D embeddings.
        Returns a score in range [-1.0, 1.0]. Higher is more similar.
        """
        score = self.recognizer.match(
            embedding1,
            embedding2,
            dis_type=cv2.FaceRecognizerSF_FR_COSINE,
        )
        return float(score)

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        known_embeddings: Sequence[tuple[str, np.ndarray]],
    ) -> tuple[str | None, float]:
        """Compares query embedding with a collection of known embeddings.
        
        Returns:
            (best_name, highest_score)
        """
        if not known_embeddings:
            return None, 0.0

        best_name = None
        best_score = -1.0

        for name, emb in known_embeddings:
            score = self.match(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, max(0.0, best_score)
