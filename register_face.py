import argparse
import sys
import time
from pathlib import Path
import cv2

from db import get_repository
from face_pipeline import (
    FaceDetector,
    FaceRecognizer,
    FaceAuthenticator,
    draw_alert_card,
)


def register_from_image(
    authenticator: FaceAuthenticator,
    detector: FaceDetector,
    image_path: str,
    name: str,
    allow_duplicate: bool = False,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'.")
        return

    faces = detector.detect(img)
    if not faces:
        print(f"Error: No face detected in '{image_path}'.")
        return

    # Use the highest confidence or largest face
    target_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    success, msg = authenticator.register_new_face(img, target_face, name, allow_duplicate=allow_duplicate)
    print(f"[{'SUCCESS' if success else 'BLOCKED'}] {msg}")


def register_from_camera(
    authenticator: FaceAuthenticator,
    detector: FaceDetector,
    name: str,
    camera_index: int = 0,
    allow_duplicate: bool = False,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera index {camera_index}.")
        print("Tip: If you have an external webcam or Camo, try `--index 1` or check macOS camera permissions.")
        return

    print(f"\nRegistering face for '{name}'...")
    print("Instructions:")
    print("  • Look directly at the camera.")
    print("  • Press 'c' to capture face and save.")
    print("  • Press 'q' or 'ESC' to cancel.\n")

    window_name = f"Register Face: {name} (Press 'c' to capture, 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    active_alert = None  # Holds active on-screen modal alert {title, lines, type, expires}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            display_frame = frame.copy()
            faces = detector.detect(frame)

            if faces:
                best_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                FaceDetector.draw_face_annotations(
                    display_frame,
                    best_face,
                    label=f"Ready to capture: {name}",
                    color=(0, 255, 0),
                    draw_landmarks=True,
                )
            else:
                cv2.putText(
                    display_frame,
                    "No face detected - look at camera",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

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

            if key in (ord('q'), 27):
                print("Registration cancelled.")
                break
            elif key == ord('c'):
                if not faces:
                    print("Cannot capture: No face detected in frame!")
                    active_alert = {
                        "title": "No Face Detected",
                        "message": "Please look directly at camera",
                        "hint": "Press [c] when ready",
                        "type": "ERROR",
                        "expires": time.time() + 3.0,
                    }
                    continue

                best_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                success, msg = authenticator.register_new_face(
                    frame, best_face, name, allow_duplicate=allow_duplicate
                )
                print(f"[{'SUCCESS' if success else 'BLOCKED'}] {msg}")

                if success:
                    # Show visual green SUCCESS card on screen for 1.8 seconds
                    end_time = time.time() + 1.8
                    while time.time() < end_time:
                        ret, post_frame = cap.read()
                        if ret and post_frame is not None:
                            pf_copy = post_frame.copy()
                            draw_alert_card(
                                pf_copy,
                                "Registered Successfully",
                                f"Saved: {name}",
                                "Closing window...",
                                "SUCCESS",
                            )
                            cv2.imshow(window_name, pf_copy)
                        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                            break
                    break
                else:
                    # Show visual red BLOCKED card on screen
                    active_alert = {
                        "title": "Registration Blocked",
                        "message": msg,
                        "hint": "Press [c] to retry  |  [q] to cancel",
                        "type": "ERROR",
                        "expires": time.time() + 5.0,
                    }
    finally:
        cap.release()
        cv2.destroyAllWindows()


def list_registered_faces(repo) -> None:
    records = repo.get_all_faces()
    print(f"\n--- Registered Faces in Database ({len(records)} total) ---")
    if not records:
        print("No faces registered yet.")
        return
    for r in records:
        print(f"  • ID {r.id}: {r.name} (Registered: {r.created_at})")
    print("--------------------------------------------------\n")


def delete_face(repo, name: str) -> None:
    count = repo.delete_face(name)
    if count > 0:
        print(f"Successfully deleted {count} record(s) for '{name}'.")
    else:
        print(f"No records found with name '{name}'.")


def delete_face_by_id(repo, record_id: int) -> None:
    success = repo.delete_by_id(record_id)
    if success:
        print(f"Successfully deleted record with ID {record_id}.")
    else:
        print(f"No record found with ID {record_id}.")


def find_duplicates(repo, recognizer: FaceRecognizer, threshold: float = 0.363) -> None:
    records = repo.get_all_faces()
    print(f"\nScanning {len(records)} registered faces for duplicates (threshold >= {threshold:.2f})...")
    duplicates_found = 0

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            r1 = records[i]
            r2 = records[j]
            score = recognizer.match(r1.embedding, r2.embedding)
            if score >= threshold:
                duplicates_found += 1
                print(
                    f"  [DUPLICATE DETECTED] Similarity: {score:.3f}\n"
                    f"    1) ID: {r1.id} | Name: '{r1.name}' (Registered: {r1.created_at})\n"
                    f"    2) ID: {r2.id} | Name: '{r2.name}' (Registered: {r2.created_at})\n"
                )

    if duplicates_found == 0:
        print("No duplicate faces found in the database.")
    else:
        print(f"Total duplicate pairs detected: {duplicates_found}")
        print("To remove a duplicate, run:")
        print("  python register_face.py --delete-id <ID>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register, list, or delete faces in the database.")
    parser.add_argument("--name", "-n", type=str, help="Name of the person to register or delete")
    parser.add_argument("--image", "-img", type=str, help="Path to an image file (optional, otherwise uses camera)")
    parser.add_argument("--index", "-i", type=int, default=0, help="Camera index for live capture (default: 0)")
    parser.add_argument("--list", "-l", action="store_true", help="List all registered faces in the database")
    parser.add_argument("--delete", "-d", action="store_true", help="Delete face records by name")
    parser.add_argument("--delete-id", type=int, help="Delete a specific face record by ID")
    parser.add_argument("--find-duplicates", action="store_true", help="Scan and detect duplicate faces in database")
    parser.add_argument("--force", action="store_true", help="Allow registering face even if already exists in DB")
    parser.add_argument("--threshold", "-t", type=float, default=0.363, help="Similarity threshold (default: 0.363)")
    parser.add_argument("--db", choices=["sqlite", "mongodb"], default="sqlite", help="Storage backend (default: sqlite)")
    args = parser.parse_args()

    repo = get_repository(backend=args.db)

    if args.list:
        list_registered_faces(repo)
        return

    if args.delete_id is not None:
        delete_face_by_id(repo, args.delete_id)
        return

    if args.find_duplicates:
        recognizer = FaceRecognizer()
        find_duplicates(repo, recognizer, threshold=args.threshold)
        return

    if args.delete:
        if not args.name:
            print("Error: Please provide --name to delete.")
            return
        delete_face(repo, args.name)
        return

    if not args.name:
        parser.print_help()
        print("\nTip: Specify --name to register a new face, e.g. python register_face.py --name 'John'")
        return

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    authenticator = FaceAuthenticator(recognizer=recognizer, repository=repo, threshold=args.threshold)

    if args.image:
        register_from_image(authenticator, detector, args.image, args.name, allow_duplicate=args.force)
    else:
        register_from_camera(authenticator, detector, args.name, camera_index=args.index, allow_duplicate=args.force)


if __name__ == "__main__":
    main()
