# Assignment submission | Face Detection & Recognition

- GitHub Repository: https://github.com/alvy007/Face-Recognition-And-Enrollments-
- Live Link: pending Streamlit Community Cloud deployment; do not submit an unverified URL.
- Attachment: use this project's ZIP.

## Implementation remarks

Built a privacy-conscious real-time Face Detection & Recognition application using OpenCV's YuNet face detector and SFace embedding model. It detects multiple faces from webcam or images, extracts aligned facial feature vectors, compares them with consent-enrolled identities using cosine similarity, and displays bounding boxes, labels and scores. Implemented an Unknown classification, configurable match threshold, local biometric-template storage, image and webcam modes, input validation, offline inference and automated unit tests. Camera footage is not uploaded or recorded automatically. This is an educational demonstration, not an identity verification or surveillance product.

## Cloud demo remarks

The public Streamlit demo supports consent-based enrollment and facial matching from uploaded
images and browser camera snapshots. It is distinct from the continuous real-time desktop webcam
implementation. Use only the **verified deployed** Streamlit link when filling in the portal.
