import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2

from db import get_repository
from face_pipeline import (
    FaceDetector,
    FaceRecognizer,
    FaceAuthenticator,
    LivenessDetector,
    draw_alert_card,
)


def list_available_cameras(max_to_test: int = 5) -> list[int]:
    """Scans and lists available camera indices."""
    available = []
    print("Scanning available camera indices...")
    for idx in range(max_to_test):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            status = f"Working ({w}x{h})" if ret else "Opened (no frame yet)"
            print(f"  • Index {idx}: {status}")
            available.append(idx)
            cap.release()
    if not available:
        print("  No cameras found or camera access denied in macOS settings.")
    return available


def draw_hud(frame, status_text: str, total_registered: int, threshold: float) -> None:
    """Draws a clean top HUD overlay with system status."""
    h, w = frame.shape[:2]
    # Draw translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    info_text = f"Faces in DB: {total_registered} | Threshold: {threshold:.2f} | 'r': Register | 'l': List | 'q': Quit"
    cv2.putText(frame, info_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def run_face_system(
    camera_index: int = 0,
    threshold: float = 0.363,
    liveness_threshold: float = 0.50,
    db_backend: str = "sqlite",
    show_landmarks: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Runs real-time face detection, landmarking, extraction, matching, and authentication."""
    print("==================================================")
    print("      FACE DETECTION & AUTHENTICATION SYSTEM      ")
    print("==================================================")
    print(f"Database backend    : {db_backend}")
    print(f"Auth Threshold      : {threshold:.3f}")
    print(f"Liveness Threshold  : {liveness_threshold:.2f} (MiniFASNet Deep Learning)")
    print(f"Camera Index        : {camera_index}")
    print("--------------------------------------------------")

    # Initialize storage repository
    repo = get_repository(backend=db_backend)
    total_known = repo.count()
    print(f"Loaded {total_known} registered face(s) from database.")

    # Initialize Face Pipeline
    print("Initializing Face Detection, Recognition, and Anti-Spoofing models...")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    authenticator = FaceAuthenticator(recognizer=recognizer, repository=repo, threshold=threshold)
    liveness_detector = LivenessDetector(liveness_threshold=liveness_threshold)

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[Error] Unable to open camera index {camera_index}.")
        print("Troubleshooting:")
        print("  • If using built-in Mac webcam, try `--index 1` or `--index 2`.")
        print("  • Check macOS Camera permissions (System Settings -> Privacy & Security -> Camera).")
        return

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Face Recognition & Authentication (Press 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    snapshots_dir = Path("snapshots")
    active_alert = None  # Holds active on-screen modal alert {title, lines, type, expires}

    print("\nSystem running! Live controls:")
    print("  • 'r' : Register currently detected face")
    print("  • 'l' : Print list of registered users in terminal")
    print("  • 's' : Save snapshot image")
    print("  • 'q' / ESC : Quit application\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            display_frame = frame.copy()

            # Step 1: Face Detection & Step 2: Landmark Detection
            detected_faces = detector.detect(frame)

            for face in detected_faces:
                # 1. Liveness Check (MiniFASNet Deep Learning + Track)
                liveness = liveness_detector.check_liveness(frame, face)
                track = liveness_detector.get_track(liveness.track_id)

                # 2. Recognition & Authentication
                auth_result = authenticator.authenticate_face(frame, face)

                # 3. Stabilize identity across momentary blinks / head turns
                raw_match_name = auth_result.name if auth_result.is_authenticated else (auth_result.name if auth_result.name != "Unknown" else None)
                if track is not None:
                    is_auth, display_name, display_sim = track.update_identity(
                        raw_match_name,
                        auth_result.similarity_score,
                        threshold,
                    )
                    seen_count = track.seen_frames
                else:
                    is_auth = auth_result.is_authenticated
                    display_name = auth_result.name
                    display_sim = auth_result.similarity_score
                    seen_count = 5

                # 4. Filter transient 1-2 frame background ghost detections
                # If a face is unauthenticated and has appeared for less than 3 frames, don't flash a red box
                if not is_auth and seen_count < 3:
                    continue

                live_pct = int(round(liveness.liveness_score * 100))

                if is_auth:
                    if liveness.is_real:
                        # REAL PERSON (Green)
                        color = (0, 220, 0)
                        label = f"Granted: {display_name} ({display_sim:.2f}) [Live: {live_pct}%]"
                    else:
                        # 2D PHOTO / PHONE SCREEN (Orange Warning)
                        color = (0, 160, 255)
                        attack_tag = f"Image ({liveness.attack_type})" if liveness.attack_type != "NONE" else "Image"
                        label = f"{attack_tag}, Granted: {display_name} ({display_sim:.2f}) [Live: {live_pct}%]"
                else:
                    if liveness.is_real:
                        # UNKNOWN REAL PERSON (Red)
                        color = (0, 0, 240)
                        label = f"Denied: Unknown ({display_sim:.2f}) [Live: {live_pct}%]"
                    else:
                        # UNKNOWN PHOTO / SCREEN (Red-Orange)
                        color = (0, 100, 240)
                        attack_tag = f"Image ({liveness.attack_type})" if liveness.attack_type != "NONE" else "Image"
                        label = f"{attack_tag}, Denied: Unknown ({display_sim:.2f}) [Live: {live_pct}%]"

                # Draw bounding box and facial landmarks
                FaceDetector.draw_face_annotations(
                    display_frame,
                    face,
                    label=label,
                    color=color,
                    draw_landmarks=show_landmarks,
                )

            # Draw HUD
            draw_hud(display_frame, "", len(authenticator._cached_faces), threshold)

            # Draw on-screen alert card if active
            if active_alert:
                if time.time() < active_alert["expires"]:
                    draw_alert_card(
                        display_frame,
                        active_alert["title"],
                        active_alert["message"],
                        active_alert.get("hint", ""),
                        active_alert["type"],
                    )
                else:
                    active_alert = None

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                print("Exit requested.")
                break

            elif key == ord('l'):  # List faces
                records = repo.get_all_faces()
                print(f"\n--- Registered Users ({len(records)}) ---")
                for r in records:
                    print(f"  • {r.name} (Registered: {r.created_at})")
                print("------------------------------------------\n")
                names_preview = ", ".join(r.name for r in records[:3])
                if len(records) > 3:
                    names_preview += f" +{len(records)-3} more"
                active_alert = {
                    "title": "Registered Users",
                    "message": names_preview if records else "No faces registered yet",
                    "hint": f"Total: {len(records)} user(s) in database",
                    "type": "INFO",
                    "expires": time.time() + 3.5,
                }

            elif key == ord('s'):  # Screenshot
                snapshots_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = snapshots_dir / f"snapshot_{timestamp}.png"
                cv2.imwrite(str(filename), display_frame)
                print(f"Snapshot saved to {filename}")
                active_alert = {
                    "title": "Snapshot Saved",
                    "message": f"Saved {filename.name}",
                    "hint": "Check snapshots/ folder",
                    "type": "SUCCESS",
                    "expires": time.time() + 2.5,
                }

            elif key == ord('r'):  # Register detected face
                if not detected_faces:
                    print("\n[Notice] No face detected to register. Please face the camera.")
                    active_alert = {
                        "title": "No Face Detected",
                        "message": "Please look directly at the camera",
                        "hint": "Press [r] when ready",
                        "type": "ERROR",
                        "expires": time.time() + 3.0,
                    }
                    continue

                best_face = max(detected_faces, key=lambda f: f.bbox[2] * f.bbox[3])
                print("\n" + "=" * 45)
                print(">>> PAUSED FOR FACE REGISTRATION <<<")
                sys.stdout.write("Enter name for this face: ")
                sys.stdout.flush()
                name = sys.stdin.readline().strip()
                print("=" * 45)

                if name:
                    success, msg = authenticator.register_new_face(frame, best_face, name)
                    print(f"[{'SUCCESS' if success else 'BLOCKED'}] {msg}\n")
                    if success:
                        active_alert = {
                            "title": "Registered Successfully",
                            "message": f"Saved: {name}",
                            "hint": "Face recognized for authentication",
                            "type": "SUCCESS",
                            "expires": time.time() + 3.0,
                        }
                    else:
                        active_alert = {
                            "title": "Registration Blocked",
                            "message": msg,
                            "hint": "Cannot save conflicting identity",
                            "type": "ERROR",
                            "expires": time.time() + 5.0,
                        }
                else:
                    print("Registration cancelled (empty name).\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and system shut down cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Face Detection, Landmark Detection, and Authentication")
    parser.add_argument("--index", "-i", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--threshold", "-t", type=float, default=0.363, help="Cosine similarity threshold (default: 0.363)")
    parser.add_argument("--liveness-threshold", "-lt", type=float, default=0.50, help="Liveness threshold for MiniFASNet (default: 0.50)")
    parser.add_argument("--db", choices=["sqlite", "mongodb"], default="sqlite", help="Storage backend (default: sqlite)")
    parser.add_argument("--width", "-W", type=int, default=None, help="Desired frame width")
    parser.add_argument("--height", "-H", type=int, default=None, help="Desired frame height")
    parser.add_argument("--no-landmarks", action="store_true", help="Disable drawing facial landmark dots")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices")
    args = parser.parse_args()

    if args.list_cameras:
        list_available_cameras()
    else:
        run_face_system(
            camera_index=args.index,
            threshold=args.threshold,
            liveness_threshold=args.liveness_threshold,
            db_backend=args.db,
            show_landmarks=not args.no_landmarks,
            width=args.width,
            height=args.height,
        )
