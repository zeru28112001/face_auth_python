from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box, 5 landmarks, and confidence score."""
    bbox: tuple[int, int, int, int]  # x, y, width, height (Step 1)
    landmarks: list[tuple[int, int]]  # 5 landmarks (Step 2: right_eye, left_eye, nose, right_mouth, left_mouth)
    score: float
    raw_data: np.ndarray  # Full 15-element array required by OpenCV alignCrop


class FaceDetector:
    """Step 1 (Face Detection) and Step 2 (Face Landmark Detection) using OpenCV YuNet."""

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet.onnx",
        conf_threshold: float = 0.85,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"YuNet model not found at '{model_path}'. "
                "Ensure face_detection_yunet.onnx is downloaded into the models directory."
            )

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._current_size: tuple[int, int] | None = None

        # Initialize detector
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_file),
            config="",
            input_size=(320, 320),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detects faces and their facial landmarks in an image."""
        h, w = image.shape[:2]
        if self._current_size != (w, h):
            self.detector.setInputSize((w, h))
            self._current_size = (w, h)

        retval, raw_faces = self.detector.detect(image)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        results: list[DetectedFace] = []
        for face in raw_faces:
            # 1. Face Detection: Bounding box [x, y, w, h]
            bbox = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))

            # Ignore tiny distant background noise or glitch detections
            if bbox[2] < 50 or bbox[3] < 50:
                continue

            # 2. Face Landmark Detection: 5 key points
            # Indices: 4,5: right eye | 6,7: left eye | 8,9: nose | 10,11: right mouth | 12,13: left mouth
            landmarks = [
                (int(face[4]), int(face[5])),
                (int(face[6]), int(face[7])),
                (int(face[8]), int(face[9])),
                (int(face[10]), int(face[11])),
                (int(face[12]), int(face[13])),
            ]
            score = float(face[14])
            results.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=landmarks,
                    score=score,
                    raw_data=face,
                )
            )

        return results

    @staticmethod
    def draw_face_annotations(
        image: np.ndarray,
        detected_face: DetectedFace,
        label: str = "Face",
        color: tuple[int, int, int] = (0, 255, 0),
        draw_landmarks: bool = True,
    ) -> None:
        """Visualizes the bounding box (Step 1) and landmark points (Step 2) on the image."""
        x, y, w, h = detected_face.bbox

        # 1. Draw Bounding Box (Step 1: Face Detection)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # 2. Draw Landmark Points (Step 2: Landmark Detection)
        if draw_landmarks:
            # Colors for landmarks: Right eye (blue), Left eye (cyan), Nose (yellow), Mouth (magenta, red)
            landmark_colors = [
                (255, 0, 0),    # Right eye
                (255, 255, 0),  # Left eye
                (0, 255, 255),  # Nose tip
                (255, 0, 255),  # Right mouth corner
                (0, 0, 255),    # Left mouth corner
            ]
            for (lx, ly), pt_color in zip(detected_face.landmarks, landmark_colors):
                cv2.circle(image, (lx, ly), 3, pt_color, -1)

        # Draw label background & text
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(y, label_size[1] + 10)
        cv2.rectangle(
            image,
            (x, top - label_size[1] - 8),
            (x + label_size[0] + 8, top),
            color,
            cv2.FILLED,
        )

        # Dynamic high-contrast text color (white on dark/red/orange, black on bright)
        brightness = color[2] * 0.299 + color[1] * 0.587 + color[0] * 0.114
        text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

        cv2.putText(
            image,
            label,
            (x + 4, top - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
