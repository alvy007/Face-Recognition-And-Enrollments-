# VisionID: deployment and submission checklist

## Repository
https://github.com/alvy007/Face-Recognition-And-Enrollments-

## Local desktop verification (Windows CMD)
```cmd
cd /d E:\OfficeWork\Internship\Pynexia\week2-four-computer-vision-projects\01-face-id
.venv\Scripts\activate.bat
python -m pytest -q
python app.py
```
The OpenCV camera preview is a separate window; move focus to it and press Q to quit.
If the camera is busy, close other camera programs. You can also try `python app.py --camera 1`.

For enrollment, find the actual picture in File Explorer and use **Copy as path**.
```cmd
python app.py --enroll Alvy --image "E:\OfficeWork\Internship\Pynexia\face-test\alvy.jpg"
```
The example path is illustrative: use a path to an existing photo with consent.
Never commit photos or biometric templates.

## Free web demo via Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app** and **Deploy a public app from GitHub**.
3. Repository: `alvy007/Face-Recognition-And-Enrollments-`
4. Branch: `main`
5. Main file path: `cloud/streamlit_app.py`
6. Python version: `3.12` (or `3.11` if needed).
7. Deploy and check **Manage app → Logs** for errors.
8. Test the **Enroll** tab with a consented portrait, and **Detect and recognize** with upload and camera snapshot.
9. Confirm the generated `https://...streamlit.app/` link works in a private browser tab. Streamlit performs server-side inference and needs internet.

## Internship submission fields
- Repository: https://github.com/alvy007/Face-Recognition-And-Enrollments-
- Live link: use the **actual verified** Streamlit URL, not an invented URL.
- Attachment: upload the final project ZIP without models or face photos.
- Implementation remarks: use `SUBMISSION.md`.

## Important limitation
The desktop app supports a continuous webcam feed; the free browser version uses snapshots, **not continuous video**. Facial matches are estimates, not verified identities. Only test with consented images.
