"""Verified first-party model sources. Downloads are optional and local only."""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import Request, urlopen

MODELS = {
 "yunet.onnx": ("https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", 100_000),
 "sface.onnx": ("https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", 5_000_000),
 "expression.onnx": ("https://huggingface.co/opencv/opencv_zoo/resolve/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx", 1_000_000),
 "pose.task": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", 500_000),
 "hand.task": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 500_000),
}

def ensure_model(name: str, folder: Path | None = None) -> Path:
    if name not in MODELS:
        raise ValueError(f"Unsupported model {name}")
    directory = folder or Path(__file__).resolve().parent / "models"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    url, minimum = MODELS[name]
    if target.exists() and target.stat().st_size >= minimum:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    try:
        print(f"Downloading {name} from its official model publisher. First run requires internet...")
        request = Request(url, headers={"User-Agent": "Week2-CV-Internship-Demo/1.0"})
        with urlopen(request, timeout=90) as response, partial.open("wb") as output:
            while True:
                piece = response.read(1024 * 1024)
                if not piece:
                    break
                output.write(piece)
        if partial.stat().st_size < minimum:
            raise RuntimeError(f"Model {name} is incomplete ({partial.stat().st_size} bytes)")
        # Detect Git LFS pointers or HTML error pages before passing data to OpenCV.
        head = partial.open("rb").read(128).lower()
        if b"git-lfs" in head or b"<html" in head or b"<!doctype" in head:
            raise RuntimeError("Received a web page or a Git LFS pointer, not a model")
        partial.replace(target)
        return target
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to download {name}: {exc}. Check your network and retry") from exc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the models needed for this project")
    parser.add_argument("names", nargs="*", choices=sorted(MODELS))
    args = parser.parse_args()
    for name in (args.names or sorted(MODELS)):
        print(ensure_model(name))
