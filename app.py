"""Consent-based face detection and recognition with OpenCV YuNet + SFace."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
from face_core import DEFAULT_THRESHOLD, best_match, load_gallery, safe_name, save_gallery
from download_models import ensure_model

ROOT = Path(__file__).resolve().parent


def build_models():
    detector = cv2.FaceDetectorYN.create(str(ensure_model('yunet.onnx')), '', (640, 480), 0.75, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(ensure_model('sface.onnx')), '')
    return detector, recognizer


def detect_faces(detector, frame):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    return [] if faces is None else list(faces)


def feature_for(recognizer, frame, face):
    aligned = recognizer.alignCrop(frame, face)
    return recognizer.feature(aligned).reshape(-1).copy()


def annotate(frame, faces, features, gallery, threshold):
    canvas = frame.copy()
    for face, feature in zip(faces, features):
        match = best_match(feature, gallery, threshold)
        x, y, w, h = (int(v) for v in face[:4])
        color = (99, 217, 152) if match.accepted else (79, 191, 243)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        text = f"{match.name} | similarity {match.score:.2f}" if gallery else "Face detected | no enrolled faces"
        cv2.rectangle(canvas, (x, max(0, y-28)), (min(canvas.shape[1], x+max(230, len(text)*9)), y), (18, 29, 43), -1)
        cv2.putText(canvas, text, (x+4, max(16,y-9)), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
    return canvas


def enroll_image(path: Path, name: str, gallery_path: Path):
    safe_name(name)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError("The supplied file is not a readable image")
    detector, recognizer = build_models()
    faces = detect_faces(detector, frame)
    if len(faces) != 1:
        raise ValueError(f"For enrollment, provide exactly one clear face; detected {len(faces)}")
    gallery = load_gallery(gallery_path)
    gallery[name] = feature_for(recognizer, frame, faces[0])
    save_gallery(gallery_path, gallery)
    print(f"Enrolled {name!r}. Gallery now has {len(gallery)} identity entries. Raw photo was not saved.")


def main():
    parser = argparse.ArgumentParser(description="Private, consent-based face recognition demo")
    parser.add_argument('--enroll', metavar='NAME', help='Consent-based enrollment of one person')
    parser.add_argument('--image', type=Path, help='Use image to enroll or inspect instead of webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--gallery', type=Path, default=ROOT/'data'/'gallery.npz')
    args = parser.parse_args()
    if args.enroll:
        if args.image is None:
            parser.error('--enroll requires --image PATH')
        enroll_image(args.image, args.enroll.strip(), args.gallery)
        return
    detector, recognizer = build_models()
    gallery = load_gallery(args.gallery)
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error('Could not open supplied image')
        faces = detect_faces(detector, frame)
        features = [feature_for(recognizer, frame, f) for f in faces]
        print(f'Detected {len(faces)} faces. Enrolled identities: {len(gallery)}')
        cv2.imshow('VisionID • Press any key to close', annotate(frame, faces, features, gallery, args.threshold))
        cv2.waitKey(0); cv2.destroyAllWindows(); return
    video = cv2.VideoCapture(args.camera)
    if not video.isOpened():
        raise RuntimeError(f'Cannot access camera {args.camera}. Check camera privacy permissions.')
    last = time.perf_counter()
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                print('Camera stopped providing frames'); break
            frame = cv2.flip(frame, 1)
            faces = detect_faces(detector, frame)
            features = [feature_for(recognizer, frame, f) for f in faces]
            canvas = annotate(frame, faces, features, gallery, args.threshold)
            now = time.perf_counter(); fps = 1/max(now-last, 1e-6); last=now
            cv2.putText(canvas, f'VisionID  |  {len(faces)} face(s)  |  {fps:.0f} FPS  |  Q: Quit', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .63, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('VisionID | Local Face Recognition', canvas)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
    finally:
        video.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    try: main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc: raise SystemExit(f'Error: {exc}')
