"""VisionID browser demo: image detection and consent-based encrypted recognition.

Desktop OpenCV app handles continuous local webcam frames. This Streamlit
version accepts image uploads or one browser-camera snapshot at a time.
No uploaded image is intentionally persisted by this application.
"""
from __future__ import annotations

import hmac
import io
import sys
import time
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app import annotate, build_models, detect_faces, feature_for
from cloud.secure_gallery import (
    check_connection,
    clear_gallery,
    enroll_face,
    load_gallery,
    remove_face,
    validate_key,
)
from face_core import DEFAULT_THRESHOLD, best_match, safe_name

SECRET_NAMES = ("SUPABASE_DB_URL", "FACE_ENCRYPTION_KEY", "ADMIN_PASSWORD")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_IMAGE_SIDE = 2500
ADMIN_SESSION_SECONDS = 30 * 60
ADMIN_MAX_ATTEMPTS = 5


# Source code is supplied by the repository author, never by users; all
# dynamic identity labels are shown with native Streamlit text components.
STYLES = """
<style>
  .block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 4rem; }
  [data-testid="stSidebar"] { background: #0e1727; }
  .visionid-hero {
    background: linear-gradient(118deg, #132d3d 0%, #122033 54%, #192c47 100%);
    border: 1px solid #30495b; border-radius: 18px; padding: 28px 32px;
    margin: 0 0 18px 0;
  }
  .visionid-tag { font-size: 12px; letter-spacing: .16em; color: #79e5ce;
                  text-transform: uppercase; font-weight: 800; }
  .visionid-title { font-size: clamp(28px, 4vw, 43px); font-weight: 800;
                    letter-spacing: -.04em; color: #f6fbff; margin: 9px 0 8px; }
  .visionid-subtitle { max-width: 740px; color: #bed1df; font-size: 15px; line-height: 1.65; }
  .visionid-meta { color: #8daebc; font-size: 12px; margin-top: 15px; }
  .visionid-kicker { color: #7bdac7; letter-spacing: .08em;
                     text-transform: uppercase; font-size: 12px; font-weight: 800; }
  div[data-testid="stMetric"] { border: 1px solid rgba(120,148,173,.25);
    background: rgba(90,119,150,.08); padding: 14px; border-radius: 12px; }
  .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
    background: #138874; border-color: #138874; color: #fff;
  }
  @media (max-width: 640px) { .visionid-hero { padding: 20px; } }
</style>
"""


def decode_picture(data: bytes) -> np.ndarray:
    """Safely load a portrait/camera JPEG or PNG into a BGR OpenCV frame."""
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_UPLOAD_BYTES:
        raise ValueError("Choose a JPG/PNG image smaller than 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPEG and PNG images are supported.")
            width, height = original.size
            if width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise ValueError("Image exceeds the 12-megapixel safety limit.")
            image = ImageOps.exif_transpose(original)
            if max(image.size) > MAX_IMAGE_SIDE:
                image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not read this image. Try a standard JPG or PNG.") from exc


def storage_settings() -> tuple[str, dict[str, str]]:
    """Fail closed if only part of the persistent-storage secrets are given."""
    try:
        settings = {name: str(st.secrets.get(name, "")).strip() for name in SECRET_NAMES}
    except FileNotFoundError:
        settings = {name: "" for name in SECRET_NAMES}
    present = sum(bool(value) for value in settings.values())
    if present == 0:
        return "temporary", settings
    if present != len(SECRET_NAMES):
        return "incomplete", settings
    if len(settings["ADMIN_PASSWORD"]) < 16:
        return "weak_password", settings
    try:
        validate_key(settings["FACE_ENCRYPTION_KEY"])
    except (ValueError, TypeError, UnicodeError):
        return "invalid_key", settings
    if not settings["SUPABASE_DB_URL"].startswith(("postgresql://", "postgres://")):
        return "invalid_database", settings
    return "persistent", settings


@st.cache_resource(show_spinner="Loading YuNet and SFace models...")
def models_and_lock():
    """Cache downloaded OpenCV models, and serialize detector resizing."""
    detector, recognizer = build_models()
    return detector, recognizer, Lock()


def analyze(frame: np.ndarray, *, with_embeddings: bool) -> tuple[list, list[np.ndarray]]:
    detector, recognizer, lock = models_and_lock()
    with lock:
        faces = detect_faces(detector, frame)
        embeddings = [feature_for(recognizer, frame, face) for face in faces] if with_embeddings else []
    return faces, embeddings


def mark_detection_only(frame: np.ndarray, faces: list) -> np.ndarray:
    """Public detection should not compute embeddings or reveal stored labels."""
    marked = frame.copy()
    for face in faces:
        x, y, width, height = (int(v) for v in face[:4])
        cv2.rectangle(marked, (x, y), (x + width, y + height), (106, 216, 179), 2)
        cv2.putText(marked, "Face detected", (max(0, x), max(16, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 252, 250), 2, cv2.LINE_AA)
    return marked


def admin_controls(password_secret: str) -> bool:
    """Educational single-admin session login, not full account authentication."""
    now = time.time()
    if st.session_state.get("visionid_admin", False):
        if st.session_state.get("visionid_admin_until", 0) <= now:
            st.session_state.visionid_admin = False
            st.warning("Admin session expired. Sign in again.")
        else:
            remaining = max(1, int((st.session_state.visionid_admin_until - now) / 60))
            st.success(f"Admin access active · {remaining} min remaining")
            if st.button("Sign out", use_container_width=True):
                st.session_state.visionid_admin = False
                st.session_state.visionid_admin_until = 0
                st.rerun()
            return True

    attempts = int(st.session_state.get("visionid_admin_attempts", 0))
    if attempts >= ADMIN_MAX_ATTEMPTS:
        st.error("Too many sign-in attempts in this browser session.")
        st.caption("Close this session before trying again. This is a demo, not a production auth service.")
        return False
    with st.form("visionid_sign_in", clear_on_submit=True):
        password = st.text_input("Admin password", type="password", autocomplete="off")
        clicked = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if clicked:
        if hmac.compare_digest(password, password_secret):
            st.session_state.visionid_admin = True
            st.session_state.visionid_admin_until = now + ADMIN_SESSION_SECONDS
            st.session_state.visionid_admin_attempts = 0
            st.rerun()
        else:
            st.session_state.visionid_admin_attempts = attempts + 1
            st.error("Incorrect password")
    return False


def image_input(widget_key: str):
    source = st.radio("Choose image source", ["Upload image", "Camera snapshot"],
                      key=f"{widget_key}_source", horizontal=True)
    if source == "Camera snapshot":
        return st.camera_input("Capture a camera snapshot", key=f"{widget_key}_camera")
    return st.file_uploader("Choose a JPG or PNG image (maximum 10 MB)",
                            type=["jpg", "jpeg", "png"], key=f"{widget_key}_file")


def main():
    st.set_page_config(page_title="VisionID | Face AI Studio", page_icon="🔍", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown("""
      <div class="visionid-hero">
        <div class="visionid-tag">Computer Vision / Week 02</div>
        <div class="visionid-title">VisionID · Face AI Studio</div>
        <div class="visionid-subtitle">Face detection with OpenCV YuNet, consent-based
        face enrollment and SFace embedding matching. Explore the workflow through
        photo uploads or browser camera snapshots.</div>
        <div class="visionid-meta">PYTHON · OPENCV · STREAMLIT · ENCRYPTED POSTGRESQL</div>
      </div>
    """, unsafe_allow_html=True)
    mode, settings = storage_settings()
    error_help = {
        "incomplete": "Database setup is incomplete. Set all three Streamlit secrets, or remove all three for a temporary demo.",
        "weak_password": "Set an admin password of at least 16 characters in Streamlit Secrets.",
        "invalid_key": "FACE_ENCRYPTION_KEY is invalid. Generate a Fernet key and keep it in Streamlit Secrets.",
        "invalid_database": "SUPABASE_DB_URL must be a private PostgreSQL connection URI.",
    }
    if mode in error_help:
        st.error(error_help[mode])
        st.stop()

    if "identity_gallery" not in st.session_state:
        st.session_state.identity_gallery = {}
    temporary_gallery = st.session_state.identity_gallery
    admin = False

    with st.sidebar:
        st.markdown("<div class='visionid-kicker'>Workspace controls</div>", unsafe_allow_html=True)
        st.write("")
        if mode == "persistent":
            st.success("Encrypted database mode")
            st.caption("Database access is verified after admin sign-in.")
            admin = admin_controls(settings["ADMIN_PASSWORD"])
        else:
            st.warning("Temporary demo mode")
            st.caption("No database secrets set; each browser session has its own temporary gallery.")
        threshold = st.slider("Minimum cosine similarity", -1.0, 1.0,
                              value=float(DEFAULT_THRESHOLD), step=0.01,
                              help="This similarity is not a calibrated confidence percentage.")
        st.divider()
        st.caption("Camera snapshots and uploads are processed on the hosting server. "
                   "Use only consented images. No live continuous video is provided here.")
        st.caption("For educational demonstration only. Not suitable for surveillance, "
                   "access control or consequential identity decisions.")

    gallery: dict[str, np.ndarray] = {}
    connected = False
    if mode == "temporary":
        gallery = temporary_gallery
    elif admin:
        try:
            connected = check_connection(settings["SUPABASE_DB_URL"])
            if not connected:
                raise RuntimeError("Gallery schema unavailable")
            gallery = load_gallery(settings["SUPABASE_DB_URL"], settings["FACE_ENCRYPTION_KEY"])
        except Exception:
            st.error("Private gallery is currently unavailable. Verify the Supabase Session Pooler URI, "
                     "SQL schema and encryption key in Streamlit Secrets.")
            st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Face detector", "YuNet", help="OpenCV Zoo pretrained face detection")
    with col2:
        st.metric("Recognition model", "SFace", help="128-dimensional embedding comparison")
    with col3:
        st.metric("Storage", "Encrypted DB" if mode == "persistent" else "Session only")

    if mode == "persistent" and admin and connected:
        st.success("Database connection verified. Private gallery is accessible.")
    if mode == "persistent" and not admin:
        st.info("Public visitors can detect faces. Enrolling, viewing and matching saved identities "
                "require administrator sign-in.")
    tabs = st.tabs(["Detect & recognize", "Enroll a face", "Manage gallery", "About & privacy"])

    with tabs[0]:
        st.subheader("Face detection & recognition")
        st.caption("Try a well-lit, front-facing photo. Detection is public; identity matching is private.")
        source = image_input("visionid_detect")
        if source is not None:
            try:
                frame = decode_picture(source.getvalue())
                authorized_gallery = gallery if (mode == "temporary" or admin) else {}
                with st.spinner("Analyzing the image..."):
                    faces, embeddings = analyze(frame, with_embeddings=bool(authorized_gallery))
                if authorized_gallery:
                    result = annotate(frame, faces, embeddings, authorized_gallery, threshold)
                else:
                    result = mark_detection_only(frame, faces)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                         caption="Detected face regions and permitted labels", use_container_width=True)
                st.metric("Faces found", len(faces))
                if not faces:
                    st.warning("No faces detected. Try brighter lighting, front-facing pose or higher resolution.")
                elif authorized_gallery:
                    for position, feature in enumerate(embeddings, 1):
                        matched = best_match(feature, authorized_gallery, threshold)
                        st.write(f"Face {position}: **{matched.name}** · similarity {matched.score:.3f}")
                else:
                    st.caption("Only bounding boxes were generated. No identity matching was performed.")
            except (ValueError, cv2.error) as exc:
                st.error(str(exc))
            except Exception:
                st.error("Model or image analysis failed. Check the app logs and model download connectivity.")

    with tabs[1]:
        st.subheader("Consent-based enrollment")
        if mode == "persistent" and not admin:
            st.warning("Sign in as administrator in the sidebar before enrolling an identity.")
        else:
            st.caption("Enroll a single consenting person. Use exactly one clear face per image. "
                       + ("Encrypted embeddings expire after 30 days." if mode == "persistent"
                          else "This temporary enrollment disappears with the browser session."))
            with st.form("visionid_enroll_form"):
                label_input = st.text_input("Identity label", max_chars=40, placeholder="e.g., Alvy")
                portrait = st.file_uploader("One JPG/PNG portrait (maximum 10 MB)",
                                            type=["jpg", "jpeg", "png"], key="visionid_enroll_image")
                consent = st.checkbox("I confirm that this person explicitly consented to biometric enrollment.")
                submitted = st.form_submit_button("Enroll identity", type="primary")
            if submitted:
                try:
                    if not consent:
                        raise ValueError("Explicit consent confirmation is required.")
                    if portrait is None:
                        raise ValueError("Upload a clear portrait first.")
                    label = safe_name(label_input)
                    frame = decode_picture(portrait.getvalue())
                    with st.spinner("Detecting and encoding the face..."):
                        faces, embeddings = analyze(frame, with_embeddings=True)
                    if len(faces) != 1:
                        raise ValueError(f"Exactly one face is required. Found {len(faces)}.")
                    if mode == "persistent":
                        if label not in gallery and len(gallery) >= 100:
                            raise ValueError("Demo enrollment limit is 100 identities.")
                        enroll_face(settings["SUPABASE_DB_URL"], settings["FACE_ENCRYPTION_KEY"],
                                    label, embeddings[0])
                        st.success("Encrypted enrollment saved. It expires after 30 days.")
                    else:
                        temporary_gallery[label] = embeddings[0].copy()
                        st.success("Enrollment saved in this browser session.")
                    st.rerun()
                except (ValueError, cv2.error) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Enrollment failed. Please review the database connection and application logs.")

    with tabs[2]:
        st.subheader("Manage enrolled identities")
        if mode == "persistent" and not admin:
            st.info("Admin sign-in required to view or delete stored identities.")
        else:
            st.metric("Active enrolled identities", len(gallery))
            if gallery:
                st.dataframe([{"Label": label, "Template": "Encrypted" if mode == "persistent" else "Session only"}
                              for label in sorted(gallery)], hide_index=True, use_container_width=True)
                with st.form("visionid_delete_one"):
                    selected = st.selectbox("Choose one identity to delete", sorted(gallery))
                    confirm_one = st.checkbox("Confirm deletion of the selected enrollment")
                    delete_one = st.form_submit_button("Delete selected identity")
                if delete_one:
                    if not confirm_one:
                        st.error("Confirm deletion before continuing.")
                    else:
                        try:
                            if mode == "persistent":
                                remove_face(settings["SUPABASE_DB_URL"], selected)
                            else:
                                temporary_gallery.pop(selected, None)
                            st.success("Identity removed from active storage.")
                            st.rerun()
                        except Exception:
                            st.error("Deletion failed. Check the database connection.")
            else:
                st.info("No enrolled identities in this gallery.")
            with st.form("visionid_clear_all"):
                confirm_all = st.checkbox("I understand that this deletes every active enrollment")
                clear_all = st.form_submit_button("Delete all enrollments")
            if clear_all:
                if not confirm_all:
                    st.error("Check the confirmation box first.")
                else:
                    try:
                        if mode == "persistent":
                            clear_gallery(settings["SUPABASE_DB_URL"])
                        else:
                            temporary_gallery.clear()
                        st.success("All active enrollments deleted.")
                        st.rerun()
                    except Exception:
                        st.error("Deletion failed. Check the database connection.")

    with tabs[3]:
        st.subheader("How VisionID works")
        st.markdown("""
        **Step 1: Detect.** YuNet identifies face boxes in an uploaded or captured photo.  
        **Step 2: Enroll (with consent).** SFace produces a 128-dimensional embedding for one face.  
        **Step 3: Recognize.** The app compares embeddings using cosine similarity. Below-threshold matches appear as **Unknown**.  
        **Step 4: Manage.** The administrator can remove saved identities, with a default 30-day retention window.

        **Privacy commitments for this demo**
        - Public users cannot access the persistent identity gallery or match against it.
        - In persistent mode, the app stores encrypted embeddings, not copies of photos.
        - Images are processed on the Streamlit host and are not intentionally written to the database.
        - Expired rows are removed during the next authenticated gallery access, not by a background scheduler.
        - Database/cloud service backups may retain deleted data subject to provider policies.
        - Authentication uses one shared admin password and is **not** a production identity system.

        **Limitations:** The web demo uses single camera snapshots, not continuous live streaming.
        Similarity values are not calibrated confidence probabilities. Recognition can be inaccurate
        across pose, lighting, or different people. Do not use for attendance, surveillance,
        access control or consequential decisions.
        """)
        st.caption("Project technologies: Python · Streamlit · OpenCV YuNet/SFace · NumPy · Supabase PostgreSQL · Fernet")

    st.divider()
    st.caption("VisionID · Computer Vision Internship Project · Consent-based educational demonstration")


if __name__ == "__main__":
    main()
