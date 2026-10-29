"""VisionID Web: ephemeral, consent-based face detection and matching demo.

Runs on Streamlit Community Cloud without cv2.imshow or a server-side camera.
The original desktop webcam app remains available via python app.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st

from app import annotate, build_models, detect_faces, feature_for
from face_core import DEFAULT_THRESHOLD, best_match, safe_name

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MAX_DIMENSION = 2500


def decode_picture(data: bytes) -> np.ndarray:
    """Decode an uploaded JPG/PNG or camera photo; apply memory limits."""
    if not data or len(data) > MAX_UPLOAD_SIZE:
        raise ValueError('Upload a nonempty JPG/PNG smaller than 10 MB.')
    buffer = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError('Could not read the supplied image.')
    height, width = frame.shape[:2]
    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return frame


@st.cache_resource(show_spinner='Loading the official OpenCV face models...')
def models_and_lock():
    # The detector mutates its input size. Share a lock among cloud sessions.
    detector, recognizer = build_models()
    return detector, recognizer, Lock()


def analyze(frame: np.ndarray):
    detector, recognizer, lock = models_and_lock()
    with lock:
        faces = detect_faces(detector, frame)
        features = [feature_for(recognizer, frame, face) for face in faces]
    return faces, features


def main():
    st.set_page_config(page_title='VisionID | Face AI Demo', page_icon='🔎', layout='wide')
    st.title('VisionID  ·  Face Detection & Recognition')
    st.caption('YuNet face detector + SFace embedding comparison, powered by OpenCV')
    st.info('This web version supports browser camera **snapshots** and image uploads. '
            'Continuous real-time video runs in the separate local desktop app (`python app.py`).')

    if 'identity_gallery' not in st.session_state:
        st.session_state.identity_gallery = {}
    gallery = st.session_state.identity_gallery

    with st.sidebar:
        st.subheader('Settings')
        threshold = st.slider('Minimum cosine similarity', -1.0, 1.0,
                              float(DEFAULT_THRESHOLD), 0.01,
                              help='Similarity is not a calibrated identity probability.')
        st.metric('Enrolled identities (this session)', len(gallery))
        if gallery:
            st.write('Names: ' + ', '.join(sorted(gallery)))
        if st.button('Forget all enrolled faces', use_container_width=True):
            gallery.clear()
            st.success('Face embeddings cleared from this session.')
        st.divider()
        st.caption('Photos are sent to the app server for inference. This app does not '
                   'write photos or face embeddings to disk. Session embeddings can be '
                   'cleared here; they do not persist across sessions. '
                   'Do not upload images without consent.')

    enroll_tab, recognize_tab, about_tab = st.tabs(['1 · Enroll (consent required)',
                                                    '2 · Detect and recognize',
                                                    '3 · About'])
    with enroll_tab:
        st.write('Enroll yourself or a person who has explicitly agreed to this demo.')
        with st.form('enrollment_form', clear_on_submit=False):
            name = st.text_input('Identity label', max_chars=40, placeholder='Your name')
            photo = st.file_uploader('One clear JPG/PNG image with exactly one face',
                                     type=['png', 'jpg', 'jpeg'], key='enroll_photo')
            consent = st.checkbox('I have the person’s permission to enroll this face.')
            submitted = st.form_submit_button('Enroll face', type='primary')
        if submitted:
            try:
                if not consent: raise ValueError('Permission is required before enrollment.')
                if photo is None: raise ValueError('Choose an image with one face.')
                label = safe_name(name)
                frame = decode_picture(photo.getvalue())
                faces, features = analyze(frame)
                if len(faces) != 1:
                    raise ValueError(f'Exactly one face is required; found {len(faces)}.')
                gallery[label] = features[0].copy()
                st.success(f'Enrolled {label}. Raw photo was not saved by the app.')
            except (ValueError, RuntimeError, cv2.error) as exc:
                st.error(str(exc))

    with recognize_tab:
        st.write('Test face detection with a picture or a snapshot from your webcam.')
        method = st.radio('Choose source', ['Webcam snapshot', 'Upload a photo'], horizontal=True)
        if method == 'Webcam snapshot':
            photo = st.camera_input('Capture a photo', key='web_photo')
        else:
            photo = st.file_uploader('Upload a JPG/PNG photo',
                                     type=['png','jpg','jpeg'], key='test_photo')
        if photo is not None:
            try:
                frame = decode_picture(photo.getvalue())
                faces, features = analyze(frame)
                annotated = annotate(frame, faces, features, gallery, threshold)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption=f'Detected {len(faces)} face(s)', use_container_width=True)
                st.metric('Detected faces', len(faces))
                if not gallery:
                    st.warning('No identities enrolled yet. Only detection labels are shown.')
                else:
                    for i, feature in enumerate(features, 1):
                        match = best_match(feature, gallery, threshold)
                        st.write(f'Face {i}: **{match.name}** · cosine similarity **{match.score:.3f}**')
            except (ValueError, RuntimeError, cv2.error) as exc:
                st.error(str(exc))

    with about_tab:
        st.markdown("""
        **How it works**

        1. YuNet detects face boxes and keypoints.
        2. SFace aligns each detected face and extracts an embedding.
        3. After consensual enrollment, cosine similarity matches the embedding
           to known labels. Unmatched faces are labeled **Unknown**.

        This is an educational demonstration, **not** reliable identity verification.
        Labels can be wrong due to lighting, pose, camera quality, bias, or other factors.
        Do not use this demo for surveillance, access control, or consequential decisions.
        """)

if __name__ == '__main__':
    main()
