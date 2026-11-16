import urllib.request
from pathlib import Path

MODELS = {
    "face_detection_yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "minifasnet_v2.onnx": "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
}

def download_models(target_dir: str = "models") -> None:
    models_path = Path(target_dir)
    models_path.mkdir(exist_ok=True)

    for filename, url in MODELS.items():
        file_path = models_path / filename
        if file_path.is_file() and file_path.stat().st_size > 1000:
            print(f"[OK] {filename} already exists.")
            continue

        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"[DONE] Saved {filename} ({file_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_models()
