# VisionID · Deployment runbook

## 1. Code and validation

- Repository: https://github.com/alvy007/Face-Recognition-And-Enrollments-
- Branch: `main`
- Streamlit entrypoint: `cloud/streamlit_app.py`
- Python runtime: `3.12`
- App dependency file: `cloud/requirements.txt`
- GitHub Actions runs the unit suite for Python 3.11 and 3.12.

## 2. Prepare Supabase database once

Open https://supabase.com/dashboard, select the existing project, then open **SQL Editor**. Run `db/schema.sql` from the repository (idempotent; `CREATE TABLE IF NOT EXISTS`). Verify with:

```sql
SELECT to_regclass('public.visionid_faces') AS table_name;
```

Expected result: `visionid_faces`. No photo or embedding should be manually inserted for this test.

## 3. Get a private connection and generate secrets

Use **Connect → Session Pooler**, *session mode*, not Direct Connection. It should use a `pooler.supabase.com` host and port `5432`. Obtain the exact URI from Supabase's own dashboard, and substitute the database password privately. Avoid pasting the URI into a chat or public screenshot. If the database password includes reserved URL characters, percent-encode them as required.

Generate the encryption key and a random admin password **on your own computer**:

```cmd
python -m pip install cryptography
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Do not change `FACE_ENCRYPTION_KEY` after enrollment without re-enrolling or a planned key migration; the existing ciphertext will no longer decrypt.

## 4. Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Create a public app using `alvy007/Face-Recognition-And-Enrollments-`, branch `main`.
3. Main file: `cloud/streamlit_app.py`. Python: `3.12`.
4. In **Advanced settings → Secrets** add the following **with your real private values**:

```toml
SUPABASE_DB_URL = "YOUR_PRIVATE_SUPABASE_SESSION_POOLER_URI"
FACE_ENCRYPTION_KEY = "YOUR_GENERATED_FERNET_KEY"
ADMIN_PASSWORD = "YOUR_STRONG_RANDOM_PASSWORD"
```

5. Save and deploy (or reboot an existing deployment to reload the secrets).
6. Only share the final `*.streamlit.app` URL once app health is checked. Never share the Secrets page itself.

**Important:** The UI's 'Encrypted database mode' badge means the three secrets have valid formats. A **successful admin sign-in and private gallery load** is the real database connectivity verification.

## 5. Manual acceptance tests after deployment

- [ ] First page loads and no `ModuleNotFoundError`, `cv2.error` or model download failure appears in app logs.
- [ ] In an incognito window without admin login: upload a consented portrait; get bounding box(es) but no identity matching or saved labels.
- [ ] Administrator logs in using the private admin password; 'Database connection verified' appears.
- [ ] Enroll a clear, consented single-face portrait; success message appears.
- [ ] Reload the page, sign in again, and confirm saved identity is still listed.
- [ ] Test a permitted photo using upload and `st.camera_input` snapshot. A matching photo should label the enrolled identity; a different face may be `Unknown` depending on threshold.
- [ ] Delete the test identity and confirm it is no longer in the database gallery after a rerun.
- [ ] Sign out and confirm private gallery is no longer visible.
- [ ] Test from a mobile browser if available. Confirm the public app URL loads in a fresh browser session.

For 30-day expiration: expiry runs on the **next authenticated gallery access**, not automatically at the exact timestamp.

## 6. Internship submission

- Repository: https://github.com/alvy007/Face-Recognition-And-Enrollments-
- Live demo: enter the **actual verified** Streamlit URL, never an assumed domain.
- Source attachment: use the final project ZIP without ONNX files, photos, embeddings or Secrets.
- Implementation remarks: copy from `SUBMISSION.md`.

The desktop app runs continuous webcam video locally, but **the hosted web version uses camera snapshots**, not real-time video. State this distinction in the submission, rather than promising a continuous cloud video feed.

## Security limitations

This project is an internship demo with a single shared admin password. It does not implement hardened authentication, centrally enforced login rate limiting, role separation, detailed audit events, a consent revocation workflow across external systems, or production backup deletion. Do not enroll people without their explicit permission or use the app to verify identity for consequential purposes.
