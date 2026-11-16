import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from db import SQLiteFaceRepository
from face_pipeline import FaceDetector, FaceRecognizer, FaceAuthenticator, LivenessDetector, DetectedFace


class TestFacePipeline(unittest.TestCase):

    def test_sqlite_repository(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_faces.db"
            repo = SQLiteFaceRepository(db_path=str(db_file))

            self.assertEqual(repo.count(), 0)

            fake_embedding = np.random.randn(1, 128).astype(np.float32)
            fake_embedding /= np.linalg.norm(fake_embedding)

            # 1. Add face
            row_id = repo.add_face("Alice", fake_embedding)
            self.assertEqual(row_id, 1)
            self.assertEqual(repo.count(), 1)

            # 2. Retrieve faces
            records = repo.get_all_faces()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].name, "Alice")
            np.testing.assert_allclose(records[0].embedding, fake_embedding, atol=1e-5)

            # 3. Delete face by ID
            deleted = repo.delete_by_id(1)
            self.assertTrue(deleted)
            self.assertEqual(repo.count(), 0)

    def test_models_initialization(self):
        detector = FaceDetector()
        recognizer = FaceRecognizer()

        synthetic_frame = np.full((320, 320, 3), 128, dtype=np.uint8)
        faces = detector.detect(synthetic_frame)
        self.assertIsInstance(faces, list)

    def test_authentication_logic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_auth.db"
            repo = SQLiteFaceRepository(db_path=str(db_file))
            recognizer = FaceRecognizer()
            authenticator = FaceAuthenticator(recognizer=recognizer, repository=repo, threshold=0.363)

            emb1 = np.ones((1, 128), dtype=np.float32)
            emb1 /= np.linalg.norm(emb1)

            repo.add_face("Bob", emb1)
            authenticator.reload_faces()

            best_name, score = recognizer.find_best_match(emb1, authenticator._cached_faces)
            self.assertEqual(best_name, "Bob")
            self.assertGreater(score, 0.99)

    def test_duplicate_face_prevention(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_dup.db"
            repo = SQLiteFaceRepository(db_path=str(db_file))
            recognizer = FaceRecognizer()
            authenticator = FaceAuthenticator(recognizer=recognizer, repository=repo, threshold=0.363)

            # Register initial face
            emb1 = np.ones((1, 128), dtype=np.float32)
            emb1 /= np.linalg.norm(emb1)
            repo.add_face("Charlie", emb1)
            authenticator.reload_faces()

            # Mock check_existing_face directly
            matched_name, sim = authenticator.check_existing_face(emb1)
            self.assertEqual(matched_name, "Charlie")
            self.assertGreater(sim, 0.99)

    def test_liveness_detector(self):
        detector = LivenessDetector()
        synthetic_frame = np.full((320, 320, 3), 120, dtype=np.uint8)
        dummy_face = DetectedFace(
            bbox=(40, 40, 100, 100),
            landmarks=[(60, 60), (90, 60), (75, 80), (65, 100), (85, 100)],
            score=0.95,
            raw_data=np.zeros(15, dtype=np.float32),
        )

        result = detector.check_liveness(synthetic_frame, dummy_face)
        self.assertIn(result.classification, ["REAL", "IMAGE"])
        self.assertGreaterEqual(result.liveness_score, 0.0)
        self.assertLessEqual(result.liveness_score, 1.0)


if __name__ == "__main__":
    unittest.main()
