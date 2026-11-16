# Real-time Face Recognition & Authentication System

A modular Python face recognition and authentication application built with **OpenCV YuNet & SFace**, implementing a complete 5-stage facial processing pipeline with **SQLite** storage and out-of-the-box readiness for **MongoDB**.

---

## 5-Stage Face Authentication Architecture

1. **Face Detection**:
   - Uses OpenCV's deep-learning based `cv2.FaceDetectorYN` (YuNet) to detect faces in real-time.
   - Outputs bounding boxes `(x, y, w, h)` with confidence scores.
2. **Face Landmark Detection**:
   - Locates 5 facial keypoints: **Right Eye**, **Left Eye**, **Nose Tip**, **Right Mouth Corner**, and **Left Mouth Corner**.
   - Used for geometric normalization and pose alignment.
3. **Feature Extraction**:
   - Aligns the face using landmark geometry (`alignCrop`) and feeds it into `cv2.FaceRecognizerSF` (SFace).
   - Extracts a unique 128-dimensional L2-normalized numerical embedding vector.
4. **Face Matching**:
   - Computes Cosine Similarity between the incoming embedding and stored database embeddings.
5. **Liveness Detection (Anti-Spoofing)**:
   - Evaluates screen Moiré/grid frequencies, specular reflection/backlight glare, and multi-frame facial micro-motion.
   - Distinguishes live human faces from photos or smartphone screens.
6. **Authentication**:
   - Compares the similarity score against a threshold (default `0.363` for SFace cosine similarity).
   - **Real Live Person**: Green box -> `Granted: <Name> (<Score>)`
   - **2D Photo / Phone Screen**: Amber box -> `Image, Granted: <Name> (<Score>)`
   - **Unknown / Denied**: Red box -> `Denied: Unknown (<Score>)`

---

## Storage Architecture (SQLite + MongoDB Ready)

The storage layer uses an abstract Repository pattern (`BaseFaceRepository`):
- **SQLite (Default)**: Embedded, file-based (`faces.db`), zero external server required.
- **MongoDB**: Ready to switch with `--db mongodb`. Implemented in `db/mongo_repo.py` (`pymongo`).

---

## Project Structure

```
python_testing/
├── models/
│   ├── face_detection_yunet.onnx     # Deep learning face detector + 5 landmarks
│   ├── face_recognition_sface.onnx   # 128D feature extractor
│   └── minifasnet_v2.onnx            # Deep learning Anti-Spoofing / Liveness detector
├── db/
│   ├── base.py                       # Abstract BaseFaceRepository
│   ├── sqlite_repo.py                # SQLite repository (faces.db)
│   ├── mongo_repo.py                 # MongoDB repository (ready for MongoDB)
│   └── __init__.py                   # Storage factory
├── face_pipeline/
│   ├── detector.py                   # Stage 1 (Detection) & Stage 2 (Landmarks)
│   ├── recognizer.py                 # Stage 3 (Extraction) & Stage 4 (Matching)
│   ├── authenticator.py              # Stage 5 (Authentication decision)
│   └── __init__.py
├── main.py                           # Real-time live camera feed & authentication HUD
├── register_face.py                  # CLI tool to register, list, and delete faces
├── requirements.txt                  # Python dependencies
└── tests/
    └── test_pipeline.py              # Automated tests
```

---

## Quickstart

### 1. Installation & Setup
```bash
git clone https://github.com/zeru28112001/face_auth_python.git
cd face_auth_python

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download AI models (YuNet, SFace, MiniFASNet)
python download_models.py
```

### 2. Run Live Camera Recognition
```bash
python main.py
```
*(If your built-in Mac webcam is at index 1 instead of 0, pass `--index 1`)*

#### Live Controls:
- <kbd>r</kbd> : Register the currently detected face (prompts in terminal for the name).
- <kbd>l</kbd> : Print list of registered users in the terminal.
- <kbd>s</kbd> : Save a snapshot to the `snapshots/` directory.
- <kbd>q</kbd> or <kbd>ESC</kbd> : Quit the application.

---

### 2. Register Faces via CLI or Image

- **Register via Camera**:
  ```bash
  ./.venv/bin/python register_face.py --name "John Doe"
  ```
- **Register from an Image File**:
  ```bash
  ./.venv/bin/python register_face.py --name "Alice" --image path/to/photo.jpg
  ```
- **List Registered Faces**:
  ```bash
  ./.venv/bin/python register_face.py --list
  ```
- **Check for Duplicate Faces**:
  ```bash
  ./.venv/bin/python register_face.py --find-duplicates
  ```
- **Delete a Face by ID**:
  ```bash
  ./.venv/bin/python register_face.py --delete-id 2
  ```
- **Delete a Face by Name**:
  ```bash
  ./.venv/bin/python register_face.py --delete --name "Alice"
  ```

---

### 3. Switching from SQLite to MongoDB

When you are ready to switch to your MongoDB server:

1. Install `pymongo`:
   ```bash
   ./.venv/bin/pip install pymongo
   ```
2. Start the app with the `--db mongodb` flag:
   ```bash
   ./.venv/bin/python main.py --db mongodb
   ```
   *(Or configure URI in `db/mongo_repo.py` / environment variables)*
