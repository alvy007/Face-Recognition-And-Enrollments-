# Pynexia Week 2 · Assignment 1: Face Detection & Recognition

## Submission fields

**GitHub Repository**  
https://github.com/alvy007/Face-Recognition-And-Enrollments-

**Live Link**  
To be added only after a successful Streamlit Community Cloud deployment and browser acceptance test.

**File Attachment**  
Upload the final ZIP of this repository's source code, excluding models, photos, database values and credentials.

## Implementation Remarks (copy and paste)

Developed VisionID, a computer-vision application for detecting and comparing human faces using OpenCV's YuNet face detector and SFace feature embeddings. Implemented face bounding-box visualization, aligned embedding extraction, cosine-similarity matching, Unknown-face handling, and configurable similarity thresholds.

Created a Streamlit web interface supporting uploaded photos and browser-camera snapshots, along with a separate Python desktop application for continuous local webcam processing. Added explicit consent confirmation and admin-restricted enrollment and identity matching. Integrated optional Supabase PostgreSQL persistence with Fernet-encrypted biometric embeddings, 30-day expiration, individual deletion, and full-gallery deletion; raw enrollment photos are not stored by the app.

Added input validation, upload-size checks, privacy controls, documented deployment steps, and automated unit tests. The hosted demo is snapshot-based rather than continuous video. This is an educational demonstration, not a production identity-verification or surveillance system.

**Technologies:** Python, OpenCV, YuNet, SFace, NumPy, Streamlit, PostgreSQL (Supabase), Fernet encryption, Pytest and GitHub Actions.

## Do not mark complete until

- [x] Source code and setup instructions ready
- [x] Unit tests and CI workflow ready
- [ ] Streamlit deployment succeeds and starts the AI models
- [ ] Supabase Session Pooler secrets are configured privately
- [ ] Admin login and real PostgreSQL access verified
- [ ] Consented enrollment, recognition and deletion tested in deployed app
- [ ] Internship portal GitHub URL, verified live URL, remarks and ZIP submitted
