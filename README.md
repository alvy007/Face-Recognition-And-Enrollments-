# VisionID | Face Detection & Recognition

**Week 2 • Assignment 1 • Medium**

An OpenCV-based face detector and local identity-matching demo. Detect multiple faces from webcam/image with **YuNet**, encode aligned faces with **SFace**, and match them against a gallery of consent-enrolled embeddings. Output includes bounding boxes, labels, and cosine similarity scores; unmatched faces are explicitly marked **Unknown**. The desktop version runs inference locally without uploading camera frames. The separate Streamlit demo runs inference on its hosting server for user-submitted images.

## Install (Windows / macOS / Linux)

Use **Python 3.11 or 3.12** recommended, 64-bit. From this folder:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py yunet.onnx sface.onnx
```

Models come from the [official OpenCV Zoo](https://github.com/opencv/opencv_zoo); about 38 MB combined. First download requires internet. Models excluded from Git and ZIP to respect size and licensing; model publisher's licensing applies. If automatic download fails, place correct ONNX files into `models/`.

### Consent-based enrollment

```bash
python app.py --enroll Alice --image path/to/alice_photo.jpg
python app.py --enroll Bob --image path/to/bob_photo.jpg
```

Each enrollment image must have **exactly one face**. This writes normalized feature vectors to `data/gallery.npz`; the image is not copied or saved. For two or more known individuals, repeat with their consent. To remove an enrolled identity, delete `data/gallery.npz` (removes all) or modify gallery locally using `face_core`.

### Inference

```bash
python app.py                     # laptop webcam (press Q or Esc to quit)
python app.py --camera 1          # another webcam
python app.py --image sample.jpg  # inspect an image
python app.py --threshold 0.4     # adjust similarity cutoff
```

If no identities are enrolled, app still detects faces and draws boxes. Recognition is not a guaranteed identity verification system: false acceptances and rejections occur due to lighting, angle, camera quality, and model biases. The initial **0.363 cosine** threshold is OpenCV's benchmark example, not a deployment security threshold. Do not use for security, attendance, surveillance, or decisions about people. Obtain informed permission from every subject.

## Testing

```bash
python -m pytest -q
```

Tests cover storage, normalization, name validation, and matching; a physical webcam/model-inference smoke test must be performed on a suitable laptop. The workbench in this chat lacks webcam access.

## Design

- `face_core.py`: pure embedding/recognition/storage logic, independently testable.
- `app.py`: OpenCV detection, aligned embedding, webcam/image controls.
- `download_models.py`: verified official model locations, local cache and sanity checks.
- `data/gallery.npz`: user-created biometric identifiers, ignored by Git.

**Privacy:** Keep consent records outside this demo, delete local embeddings on request, do not upload them to GitHub. Avoid testing on strangers' faces.

## Free browser demo: Streamlit Community Cloud

The desktop application continues to work with `python app.py` (continuous webcam).
For a public browser-based demo, `cloud/streamlit_app.py` uses webcam **snapshots** or uploaded
images. It runs YuNet/SFace server-side with no persistent face gallery. No identity
embeddings are committed to Git.

Run locally (recommended: separate cloud deployment virtual environment):

```bash
python -m pip install -r cloud/requirements.txt
python -m streamlit run cloud/streamlit_app.py
```

**Deploy free:**

1. Use the public GitHub repository: https://github.com/alvy007/Face-Recognition-And-Enrollments-
2. Go to https://share.streamlit.io and create a Community Cloud app.
3. Select the repository and `main` branch.
4. Set entrypoint `cloud/streamlit_app.py` and select Python 3.11 or 3.12 when available.
5. The `cloud/requirements.txt` contains Streamlit and headless OpenCV dependencies,
   while root `requirements.txt` remains for the local desktop app. No OpenCV variant conflict.
6. Open the public `*.streamlit.app` URL and verify camera snapshot and enrollment.
7. Submit the GitHub URL, tested live link, and project ZIP with the implementation remarks.

Photos submitted to a public Streamlit app are transmitted to the app server for
inference. Avoid sensitive photos; use only images you have permission to process.

## Project repository

https://github.com/alvy007/Face-Recognition-And-Enrollments-

The generated model files and any local enrolled facial embeddings are excluded from Git. Run the desktop commands in the project's root directory, not its parent folder.
