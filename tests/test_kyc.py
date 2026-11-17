import unittest
from pathlib import Path
import cv2
import numpy as np

from db.sqlite_repo import SQLiteFaceRepository
from face_pipeline.authenticator import FaceAuthenticator
from face_pipeline.detector import FaceDetector, DetectedFace
from face_pipeline.liveness import LivenessDetector
from face_pipeline.recognizer import FaceRecognizer
from face_pipeline.kyc import (
    IDCardProcessor,
    KYCDocument,
    KYCResult,
    KYCVerifier,
    draw_kyc_modal,
)


class TestKYCModule(unittest.TestCase):
    """Unit tests for eKYC Onboarding & Cross-Verification Pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()
        cls.recognizer = FaceRecognizer()
        cls.test_db_path = "test_kyc.db"
        cls.repo = SQLiteFaceRepository(db_path=cls.test_db_path)
        cls.authenticator = FaceAuthenticator(
            recognizer=cls.recognizer,
            repository=cls.repo,
            threshold=0.363,
        )
        cls.verifier = KYCVerifier(
            detector=cls.detector,
            recognizer=cls.recognizer,
            authenticator=cls.authenticator,
            repository=cls.repo,
            kyc_threshold=0.363,
        )

        # Load real test face if available
        cls.real_img = cv2.imread("snapshots/snapshot_20261117_011633.png")

    @classmethod
    def tearDownClass(cls):
        p = Path(cls.test_db_path)
        if p.exists():
            p.unlink()

    def test_order_corners(self):
        """Validates quadrilateral corner ordering."""
        # Unordered points
        pts = np.array([[300, 200], [50, 50], [300, 50], [50, 200]], dtype=np.float32)
        ordered = IDCardProcessor.order_corners(pts)

        # top-left should be (50, 50)
        np.testing.assert_array_equal(ordered[0], [50, 50])
        # top-right should be (300, 50)
        np.testing.assert_array_equal(ordered[1], [300, 50])
        # bottom-right should be (300, 200)
        np.testing.assert_array_equal(ordered[2], [300, 200])
        # bottom-left should be (50, 200)
        np.testing.assert_array_equal(ordered[3], [50, 200])

    def test_detect_and_crop_card_fallback(self):
        """Validates graceful fallback when image is already cropped."""
        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cropped, corners = IDCardProcessor.detect_and_crop_card(dummy_img)
        self.assertEqual(cropped.shape, dummy_img.shape)

    def test_kyc_verification_flow(self):
        """Validates end-to-end KYC verification with live selfie vs mock ID card."""
        if self.real_img is None:
            self.skipTest("Snapshot image not found.")

        # Create a mock ID card containing the person's face photo
        card_canvas = np.full((400, 640, 3), 245, dtype=np.uint8)
        # Draw ID card elements
        cv2.rectangle(card_canvas, (10, 10), (630, 390), (200, 200, 200), 2)
        cv2.putText(card_canvas, "REPUBLIC IDENTITY CARD", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        cv2.putText(card_canvas, "ID: 12/AHA(N)998877", (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        cv2.putText(card_canvas, "Name: Alex Morgan", (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

        # Paste the face portrait onto the ID card
        faces = self.detector.detect(self.real_img)
        self.assertGreater(len(faces), 0, "No face detected in test image.")
        live_face = faces[0]
        bx, by, bw, bh = live_face.bbox
        face_crop = self.real_img[by:by + bh, bx:bx + bw]
        resized_portrait = cv2.resize(face_crop, (180, 220))
        card_canvas[100:320, 40:220] = resized_portrait

        # Perform eKYC verification
        result = self.verifier.verify_and_onboard(
            id_image=card_canvas,
            live_frame=self.real_img,
            live_face=live_face,
            user_name="Alex Morgan",
            id_number="12/AHA(N)998877",
            auto_register=True,
        )

        self.assertIsInstance(result, KYCResult)
        self.assertTrue(result.is_verified, f"KYC failed: {result.message}")
        self.assertEqual(result.status, "APPROVED")
        self.assertGreaterEqual(result.similarity_score, 0.363)
        self.assertEqual(result.user_name, "Alex Morgan")

        # Verify record in database
        records = self.repo.get_faces_by_name("Alex Morgan")
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0].metadata)
        self.assertTrue(records[0].metadata.get("kyc_verified"))
        self.assertEqual(records[0].metadata.get("id_number"), "12/AHA(N)998877")

    def test_kyc_no_face_on_id_card(self):
        """Validates rejection when presenting a blank or invalid document without portrait photo."""
        blank_card = np.full((400, 640, 3), 240, dtype=np.uint8)
        result = self.verifier.verify_and_onboard(
            id_image=blank_card,
            live_frame=self.real_img if self.real_img is not None else np.zeros((300, 300, 3), dtype=np.uint8),
            user_name="Invalid Test",
        )
        self.assertFalse(result.is_verified)
        self.assertEqual(result.status, "NO_ID_FACE")

    def test_draw_kyc_modal(self):
        """Validates KYC HUD rendering."""
        dummy_result = KYCResult(
            status="APPROVED",
            is_verified=True,
            user_name="Test User",
            id_number="ID-12345",
            similarity_score=0.88,
            liveness_score=0.99,
            document=None,
            message="Verified successfully",
        )
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_kyc_modal(canvas, dummy_result, x_pos=40, y_pos=50)
        self.assertEqual(canvas.shape, (720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
