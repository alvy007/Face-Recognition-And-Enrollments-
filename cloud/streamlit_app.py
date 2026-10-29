"""VisionID web demo: public face detection, private persistent recognition.

Storage is optional: with no secrets, the demo uses session-only enrollment.
With all three secrets, only the authenticated administrator can enroll,
recognize, or remove persistent encrypted embeddings in Supabase Postgres.
"""
from __future__ import annotations

import hmac
import sys
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
from app import annotate, build_models, detect_faces, feature_for
from cloud.secure_gallery import clear_gallery, enroll_face, load_gallery, remove_face
from face_core import DEFAULT_THRESHOLD, best_match, safe_name

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MAX_DIMENSION = 2500
SECRET_NAMES = ("SUPABASE_DB_URL", "FACE_ENCRYPTION_KEY", "ADMIN_PASSWORD")


def storage_settings() -> tuple[str, dict[str, str]]:
    try:
        supplied = {name: str(st.secrets.get(name, "")).strip() for name in SECRET_NAMES}
    except FileNotFoundError:
        supplied = {name: "" for name in SECRET_NAMES}
    present = [name for name, value in supplied.items() if value]
    if not present:
        return "temporary", supplied
    if len(present) != len(SECRET_NAMES):
        return "incomplete", supplied
    if len(supplied["ADMIN_PASSWORD"]) < 16:
        return "insecure_password", supplied
    return "persistent", supplied


def decode_picture(data: bytes) -> np.ndarray:
    """Decode a camera snapshot or upload with memory limits."""
    if not data or len(data) > MAX_UPLOAD_SIZE:
        raise ValueError("Upload a nonempty JPG/PNG smaller than 10 MB.")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not read the supplied image.")
    height, width = frame.shape[:2]
    if max(height, width) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(height, width)
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    return frame


@st.cache_resource(show_spinner="Loading OpenCV face models...")
def models_and_lock():
    detector, recognizer = build_models()
    return detector, recognizer, Lock()


def analyze(frame: np.ndarray):
    detector, recognizer, lock = models_and_lock()
    with lock:
        faces = detect_faces(detector, frame)
        features = [feature_for(recognizer, frame, face) for face in faces]
    return faces, features


def admin_controls(settings: dict[str, str]) -> bool:
    """Admin login is session-scoped. Never expose DB credentials in the UI."""
    if st.session_state.get("visionid_admin", False):
        st.success("Admin session active")
        if st.button("Log out", use_container_width=True):
            st.session_state.visionid_admin = False
            st.rerun()
    else:
        st.caption("Private identity gallery requires admin login.")
        with st.form("admin_login", clear_on_submit=True):
            password = st.text_input("Admin password", type="password")
            login = st.form_submit_button("Sign in")
        if login:
            if hmac.compare_digest(password, settings["ADMIN_PASSWORD"]):
                st.session_state.visionid_admin = True
                st.session_state.visionid_attempts = 0
                st.rerun()
            else:
                st.session_state.visionid_attempts = st.session_state.get("visionid_attempts", 0) + 1
                st.error("Invalid password")
                if st.session_state.visionid_attempts >= 5:
                    st.warning("Too many attempts in this session. Restart your session before retrying.")
    return bool(st.session_state.get("visionid_admin", False))


def main():
    st.set_page_config(page_title="VisionID | Privacy-first Face AI", page_icon="🔎", layout="wide")
    st.title("VisionID  ·  Face Detection & Recognition")
    st.caption("OpenCV YuNet detector · SFace embeddings · Consent-based recognition")
    st.info("Web demo uses **camera snapshots** and image uploads, not continuous live video. "
            "Your photos are transmitted to the Streamlit host for inference and not saved by this app.")

    mode, settings = storage_settings()
    if mode == "incomplete":
        st.error("Persistent storage is only partially configured. Add all three secrets or remove them all. "
                 "Enrollment and recognition are disabled until setup is complete.")
        st.stop()
    if mode == "insecure_password":
        st.error("ADMIN_PASSWORD must contain at least 16 characters. Fix it in Streamlit app Secrets.")
        st.stop()

    if "identity_gallery" not in st.session_state:
        st.session_state.identity_gallery = {}
    temporary_gallery = st.session_state.identity_gallery
    admin = False
    threshold = DEFAULT_THRESHOLD

    with st.sidebar:
        st.header("Controls")
        if mode == "persistent":
            st.success("Encrypted Supabase PostgreSQL storage configured")
            admin = admin_controls(settings)
        else:
            st.warning("Temporary demo mode: enrollment is lost on session reset or app restart. "
                       "No database is connected.")
        threshold = st.slider("Cosine similarity threshold", -1.0, 1.0, float(DEFAULT_THRESHOLD), 0.01,
                              help="Similarity is not a calibrated probability of identity.")
        st.divider()
        st.caption("Only enroll yourself or a person who has explicitly agreed. "
                   "Do not use this educational demo for surveillance, attendance, "
                   "access control, or decisions about people.")

    gallery: dict[str, np.ndarray] = {}
    if mode == "temporary":
        gallery = temporary_gallery
    elif admin:
        try:
            gallery = load_gallery(settings["SUPABASE_DB_URL"], settings["FACE_ENCRYPTION_KEY"])
        except Exception:
            st.error("Could not access the encrypted gallery. Check database URL, schema and encryption key "
                     "in Streamlit Secrets. Enrollment and recognition are unavailable.")
            st.stop()

    with st.sidebar:
        if admin or mode == "temporary":
            st.metric("Enrolled identities", len(gallery))
            if gallery:
                st.caption("Stored labels: " + ", ".join(sorted(gallery)))

    enroll_tab, recognize_tab, manage_tab = st.tabs([
        "1 · Enroll", "2 · Detect & recognize", "3 · Manage & privacy"
    ])

    with enroll_tab:
        if mode == "persistent" and not admin:
            st.warning("Admin sign-in is required to enroll and persist identities.")
        else:
            st.write("Enroll yourself or a person who has explicitly consented. "
                     + ("Enrollment expires after 30 days." if mode == "persistent" else
                        "This session-only enrollment disappears when the session ends."))
            with st.form("enrollment_form"):
                name = st.text_input("Face label", placeholder="Example: Alvy", max_chars=40)
                photo = st.file_uploader("Upload a clear JPG/PNG with exactly one face",
                                         type=["jpg", "jpeg", "png"], key="enroll_photo")
                consent = st.checkbox("I confirm this person has explicitly consented to enrollment.")
                save = st.form_submit_button("Enroll face", type="primary")
            if save:
                try:
                    if not consent:
                        raise ValueError("Explicit consent is required.")
                    if photo is None:
                        raise ValueError("Please upload one image.")
                    label = safe_name(name)
                    frame = decode_picture(photo.getvalue())
                    faces, features = analyze(frame)
                    if len(faces) != 1:
                        raise ValueError(f"Exactly one face required; found {len(faces)}.")
                    if mode == "persistent":
                        if label not in gallery and len(gallery) >= 100:
                            raise ValueError("Maximum 100 identities per demo gallery.")
                        enroll_face(settings["SUPABASE_DB_URL"], settings["FACE_ENCRYPTION_KEY"],
                                    label, features[0])
                        st.success(f"Enrolled {label} in encrypted database. Expires in 30 days.")
                    else:
                        temporary_gallery[label] = features[0].copy()
                        st.success(f"Temporarily enrolled {label} in this session.")
                    st.rerun()
                except (ValueError, RuntimeError, cv2.error) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Enrollment failed. Check your Supabase connection and table schema.")

    with recognize_tab:
        guest = mode == "persistent" and not admin
        if guest:
            st.info("Public detection mode. Admin login required for matching against saved identities.")
        method = st.radio("Input source", ["Webcam snapshot", "Upload photo"], horizontal=True)
        if method == "Webcam snapshot":
            photo = st.camera_input("Capture a face image")
        else:
            photo = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"], key="recognize_photo")
        if photo is not None:
            try:
                frame = decode_picture(photo.getvalue())
                faces, features = analyze(frame)
                safe_gallery = {} if guest else gallery
                marked = annotate(frame, faces, features, safe_gallery, threshold)
                st.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
                         caption=f"Detected {len(faces)} face(s)", use_container_width=True)
                st.metric("Faces detected", len(faces))
                if safe_gallery:
                    for i, feature in enumerate(features, 1):
                        match = best_match(feature, safe_gallery, threshold)
                        st.write(f"Face {i}: **{match.name}**, cosine similarity **{match.score:.3f}**")
                else:
                    st.caption("No identity matching performed. Only face detection results are shown.")
            except (ValueError, RuntimeError, cv2.error) as exc:
                st.error(str(exc))
            except Exception:
                st.error("Image processing failed. Please retry with another image.")

    with manage_tab:
        st.subheader("Consent and retention")
        st.write("The app stores **encrypted 128-dimensional face embeddings, not raw photos**. "
                 "Admin-only enrollment and matching protect saved identities from public visitors. "
                 "Cloud hosting and database providers process this data. "
                 "Deleted data may remain in provider backups for their retention period.")
        if mode == "persistent":
            st.write("Records expire after 30 days and are purged on the next authenticated gallery access.")
            if admin and gallery:
                with st.form("delete_one"):
                    selected = st.selectbox("Remove one enrolled identity", sorted(gallery))
                    delete = st.form_submit_button("Delete selected face")
                if delete:
                    try:
                        remove_face(settings["SUPABASE_DB_URL"], selected)
                        st.success(f"Removed {selected} from active database storage.")
                        st.rerun()
                    except Exception:
                        st.error("Deletion failed. Check your database connection.")
                with st.form("delete_all"):
                    confirmed = st.checkbox("I understand this removes all enrolled identities")
                    wipe = st.form_submit_button("Delete all enrolled faces")
                if wipe:
                    if not confirmed:
                        st.error("Confirm the deletion first.")
                    else:
                        try:
                            clear_gallery(settings["SUPABASE_DB_URL"])
                            st.success("All face embeddings removed from the active database table.")
                            st.rerun()
                        except Exception:
                            st.error("Deletion failed. Check your database connection.")
            elif not admin:
                st.info("Sign in as admin to inspect or delete enrolled identities.")
        else:
            if st.button("Clear temporary enrollment data", use_container_width=True):
                temporary_gallery.clear()
                st.success("Temporary embeddings cleared for this session.")
                st.rerun()
        st.warning("Educational portfolio demo. A real identity verification system requires stronger authentication, "
                   "formal consent processes, privacy assessment, and an independent security review.")


if __name__ == "__main__":
    main()
