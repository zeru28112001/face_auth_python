from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
import cv2
import numpy as np

from db.base import BaseFaceRepository
from .authenticator import FaceAuthenticator
from .detector import DetectedFace, FaceDetector
from .liveness import LivenessDetector
from .recognizer import FaceRecognizer


@dataclass
class KYCDocument:
    """Represents a processed identity document (National ID / Passport / Driver License)."""
    raw_image: np.ndarray
    card_image: np.ndarray
    portrait_image: np.ndarray | None = None
    portrait_face: DetectedFace | None = None
    card_corners: list[tuple[int, int]] | None = None
    extracted_fields: dict = field(default_factory=dict)


@dataclass
class KYCResult:
    """Outcome of 1-to-1 Cross-Verification between ID document photo and live webcam face."""
    status: Literal[
        "APPROVED",
        "REJECTED",
        "NO_ID_FACE",
        "LIVENESS_FAILED",
        "IDENTITY_CONFLICT",
        "INVALID_DOCUMENT",
    ]
    is_verified: bool
    user_name: str
    id_number: str | None
    similarity_score: float
    liveness_score: float
    document: KYCDocument | None
    message: str
    metadata: dict = field(default_factory=dict)


class IDCardProcessor:
    """Handles ID document boundary detection, perspective deskewing, and portrait extraction."""

    # Standard ISO/IEC 7810 ID-1 aspect ratio is 85.60 mm x 53.98 mm ~= 1.586
    TARGET_WIDTH: int = 640
    TARGET_HEIGHT: int = 400

    @classmethod
    def order_corners(cls, pts: np.ndarray) -> np.ndarray:
        """Orders 4 corner coordinates: [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @classmethod
    def detect_and_crop_card(
        cls,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int]] | None]:
        """Detects the ID card quadrilateral boundary and applies perspective warp to deskew."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-scale edge detection
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        best_quad = None
        min_card_area = (w * h) * 0.10  # Must occupy at least 10% of frame

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_card_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Verify aspect ratio is roughly in range [1.1, 2.1]
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = cls.order_corners(pts)
                width_top = np.linalg.norm(ordered[1] - ordered[0])
                width_bot = np.linalg.norm(ordered[2] - ordered[3])
                height_left = np.linalg.norm(ordered[3] - ordered[0])
                height_right = np.linalg.norm(ordered[2] - ordered[1])

                avg_w = (width_top + width_bot) / 2.0
                avg_h = (height_left + height_right) / 2.0
                if avg_h > 0:
                    aspect = avg_w / avg_h
                    # ID cards can be oriented landscape (~1.58) or portrait (~0.63)
                    if 1.1 <= aspect <= 2.1 or 0.48 <= aspect <= 0.90:
                        best_quad = ordered
                        break

        if best_quad is not None:
            # Perspective Transform to canonical size
            dst = np.array(
                [
                    [0, 0],
                    [cls.TARGET_WIDTH - 1, 0],
                    [cls.TARGET_WIDTH - 1, cls.TARGET_HEIGHT - 1],
                    [0, cls.TARGET_HEIGHT - 1],
                ],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(best_quad, dst)
            warped = cv2.warpPerspective(image, M, (cls.TARGET_WIDTH, cls.TARGET_HEIGHT))
            corners = [(int(pt[0]), int(pt[1])) for pt in best_quad]
            return warped, corners

        # Fallback: Image is already cropped or card covers entire frame
        return image.copy(), None

    @classmethod
    def extract_card_portrait(
        cls,
        detector: FaceDetector,
        card_image: np.ndarray,
    ) -> tuple[np.ndarray | None, DetectedFace | None]:
        """Detects and crops the official portrait photo printed on the ID document."""
        faces = detector.detect(card_image)
        if not faces:
            return None, None

        # Pick the most prominent face on the card
        best_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        bx, by, bw, bh = best_face.bbox
        ch, cw = card_image.shape[:2]

        # Crop with 15% safety margin
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(cw, bx + bw + pad_x)
        y2 = min(ch, by + bh + pad_y)

        portrait_crop = card_image[y1:y2, x1:x2].copy()
        return portrait_crop, best_face

    @classmethod
    def process_document(
        cls,
        detector: FaceDetector,
        image: np.ndarray,
    ) -> KYCDocument:
        """Processes an ID document image: deskews the card and extracts the portrait photo."""
        card_image, corners = cls.detect_and_crop_card(image)
        portrait_crop, portrait_face = cls.extract_card_portrait(detector, card_image)

        return KYCDocument(
            raw_image=image,
            card_image=card_image,
            portrait_image=portrait_crop,
            portrait_face=portrait_face,
            card_corners=corners,
        )


class KYCVerifier:
    """eKYC Verification Engine: Orchestrates 1-to-1 Cross-Verification between ID Photo & Live Webcam."""

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        authenticator: FaceAuthenticator,
        repository: BaseFaceRepository,
        liveness_detector: LivenessDetector | None = None,
        kyc_threshold: float = 0.363,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.authenticator = authenticator
        self.repository = repository
        self.liveness_detector = liveness_detector
        self.kyc_threshold = kyc_threshold

    def verify_and_onboard(
        self,
        id_image: np.ndarray | str | Path,
        live_frame: np.ndarray,
        live_face: DetectedFace | None = None,
        user_name: str | None = None,
        id_number: str | None = None,
        document_type: str = "NATIONAL_ID",
        allow_duplicate: bool = False,
        auto_register: bool = True,
    ) -> KYCResult:
        """Executes full eKYC verification workflow:
        1. Process ID document & extract ID photo.
        2. Detect live face & execute Anti-Spoofing Liveness check.
        3. Extract 128D embeddings from both ID photo and live face.
        4. Measure 1-to-1 cosine similarity.
        5. Verify identity uniqueness & register user into database with KYC metadata.
        """
        # Load ID card image if path provided
        if isinstance(id_image, (str, Path)):
            id_path = Path(id_image)
            if not id_path.is_file():
                return KYCResult(
                    status="INVALID_DOCUMENT",
                    is_verified=False,
                    user_name=user_name or "Unknown",
                    id_number=id_number,
                    similarity_score=0.0,
                    liveness_score=0.0,
                    document=None,
                    message=f"ID image file not found: {id_image}",
                )
            card_bgr = cv2.imread(str(id_path))
            if card_bgr is None:
                return KYCResult(
                    status="INVALID_DOCUMENT",
                    is_verified=False,
                    user_name=user_name or "Unknown",
                    id_number=id_number,
                    similarity_score=0.0,
                    liveness_score=0.0,
                    document=None,
                    message="Failed to decode ID image file.",
                )
        else:
            card_bgr = id_image

        # 1. Process ID Document
        doc = IDCardProcessor.process_document(self.detector, card_bgr)
        if doc.portrait_face is None or doc.portrait_image is None:
            return KYCResult(
                status="NO_ID_FACE",
                is_verified=False,
                user_name=user_name or "Unknown",
                id_number=id_number,
                similarity_score=0.0,
                liveness_score=0.0,
                document=doc,
                message="Could not detect a clear portrait photo on the presented ID card.",
            )

        # 2. Process Live Face
        if live_face is None:
            live_faces = self.detector.detect(live_frame)
            if not live_faces:
                return KYCResult(
                    status="REJECTED",
                    is_verified=False,
                    user_name=user_name or "Unknown",
                    id_number=id_number,
                    similarity_score=0.0,
                    liveness_score=0.0,
                    document=doc,
                    message="No live face detected in front of the camera.",
                )
            live_face = max(live_faces, key=lambda f: f.bbox[2] * f.bbox[3])

        # 3. Liveness Check on Live Face
        liveness_score = 1.0
        if self.liveness_detector is not None:
            live_res = self.liveness_detector.check_liveness(live_frame, live_face)
            liveness_score = float(live_res.liveness_score)
            if not live_res.is_real:
                attack_type = live_res.attack_type
                return KYCResult(
                    status="LIVENESS_FAILED",
                    is_verified=False,
                    user_name=user_name or "Unknown",
                    id_number=id_number,
                    similarity_score=0.0,
                    liveness_score=liveness_score,
                    document=doc,
                    message=f"Live anti-spoofing check failed ({attack_type}). Potential presentation attack.",
                )

        # 4. Extract Embeddings & Compute 1-to-1 Cosine Similarity
        emb_id = self.recognizer.extract_feature(doc.card_image, doc.portrait_face)
        emb_live = self.recognizer.extract_feature(live_frame, live_face)
        similarity = float(self.recognizer.match(emb_id, emb_live))

        # 5. Threshold Evaluation
        if similarity < self.kyc_threshold:
            pct = int(round(similarity * 100))
            thresh_pct = int(round(self.kyc_threshold * 100))
            return KYCResult(
                status="REJECTED",
                is_verified=False,
                user_name=user_name or "Unknown",
                id_number=id_number,
                similarity_score=similarity,
                liveness_score=liveness_score,
                document=doc,
                message=f"Biometric mismatch: Live face does not match ID portrait ({pct}% vs required {thresh_pct}%).",
            )

        clean_name = (user_name or f"User_{id_number or 'KYC'}").strip()

        # 6. Uniqueness & Conflict Pre-check
        matched_name, db_similarity = self.authenticator.check_existing_face(emb_live)
        if matched_name is not None and db_similarity >= self.authenticator.threshold:
            if not allow_duplicate and matched_name != clean_name:
                pct = int(round(db_similarity * 100))
                return KYCResult(
                    status="IDENTITY_CONFLICT",
                    is_verified=False,
                    user_name=clean_name,
                    id_number=id_number,
                    similarity_score=similarity,
                    liveness_score=liveness_score,
                    document=doc,
                    message=f"Identity conflict: Biometrics already registered under '{matched_name}' ({pct}% match).",
                )

        # 7. Prepare Metadata
        kyc_metadata = {
            "kyc_verified": True,
            "id_number": id_number or "N/A",
            "document_type": document_type,
            "similarity_score": round(similarity, 3),
            "liveness_score": round(liveness_score, 3),
            "verified_at": datetime.now().isoformat(),
        }

        # 8. Auto-Registration into Database
        if auto_register:
            success, reg_msg = self.authenticator.register_new_face(
                image=live_frame,
                detected_face=live_face,
                name=clean_name,
                allow_duplicate=allow_duplicate or (matched_name == clean_name),
                metadata=kyc_metadata,
            )
            if not success:
                return KYCResult(
                    status="IDENTITY_CONFLICT",
                    is_verified=False,
                    user_name=clean_name,
                    id_number=id_number,
                    similarity_score=similarity,
                    liveness_score=liveness_score,
                    document=doc,
                    message=f"Failed to register KYC user: {reg_msg}",
                    metadata=kyc_metadata,
                )

        pct = int(round(similarity * 100))
        return KYCResult(
            status="APPROVED",
            is_verified=True,
            user_name=clean_name,
            id_number=id_number,
            similarity_score=similarity,
            liveness_score=liveness_score,
            document=doc,
            message=f"eKYC verification approved ({pct}% biometric match). User registered successfully.",
            metadata=kyc_metadata,
        )


def draw_kyc_modal(
    frame: np.ndarray,
    result: KYCResult,
    x_pos: int = 40,
    y_pos: int = 50,
) -> None:
    """Renders a comprehensive eKYC visual verification HUD modal on the frame."""
    fh, fw = frame.shape[:2]
    w_card = min(540, fw - 40)
    h_card = min(280, fh - 40)

    x1 = max(10, min(fw - w_card - 10, x_pos))
    y1 = max(10, min(fh - h_card - 10, y_pos))
    x2 = x1 + w_card
    y2 = y1 + h_card

    # Dark translucent card background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 24), cv2.FILLED)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    is_ok = result.is_verified
    accent_color = (0, 220, 0) if is_ok else (0, 0, 240)
    cv2.rectangle(frame, (x1, y1), (x2, y2), accent_color, 2)

    # Title Banner
    cv2.putText(
        frame,
        "eKYC IDENTITY VERIFICATION SYSTEM",
        (x1 + 16, y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Status Tag
    status_text = f"[{result.status}] {result.user_name}"
    cv2.putText(
        frame,
        status_text,
        (x1 + 16, y1 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        accent_color,
        2,
        cv2.LINE_AA,
    )

    # Side-by-side Portrait Inset
    thumb_size = 90
    ty = y1 + 68
    if result.document is not None and result.document.portrait_image is not None:
        p_thumb = cv2.resize(result.document.portrait_image, (thumb_size, thumb_size))
        tx1 = x1 + 16
        frame[ty:ty + thumb_size, tx1:tx1 + thumb_size] = p_thumb
        cv2.rectangle(frame, (tx1, ty), (tx1 + thumb_size, ty + thumb_size), (80, 80, 80), 1)
        cv2.putText(
            frame,
            "ID Photo",
            (tx1 + 12, ty + thumb_size + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Telemetry lines
    info_x = x1 + 130
    line_y = y1 + 84

    sim_pct = int(round(result.similarity_score * 100))
    live_pct = int(round(result.liveness_score * 100))

    cv2.putText(
        frame,
        f"Biometric Match  : {sim_pct}% (Threshold: 36%)",
        (info_x, line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    line_y += 24
    cv2.putText(
        frame,
        f"Anti-Spoof Live  : {live_pct}% (MiniFASNet)",
        (info_x, line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 220, 0) if live_pct >= 50 else (0, 160, 255),
        1,
        cv2.LINE_AA,
    )

    line_y += 24
    id_str = result.id_number or "N/A"
    cv2.putText(
        frame,
        f"ID Document No   : {id_str}",
        (info_x, line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Message bar at bottom
    cv2.putText(
        frame,
        result.message[:65],
        (x1 + 16, y2 - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
