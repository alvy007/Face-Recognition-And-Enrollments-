# VisionID · Face AI Studio

**Pynexia Internship · Week 2 · Face Detection & Recognition**

An educational, consent-first face detection and recognition project built with Python, OpenCV YuNet and SFace, and a responsive Streamlit web interface. It supports webcam snapshots, image uploads, private enrollment, and optional encrypted persistence in Supabase PostgreSQL. The local desktop application separately supports continuous webcam processing.

> **Not an identity verification product.** Recognition may be wrong because of lighting, pose, similarity between people or model biases. Do not use for surveillance, access control, attendance, or decisions about people.

## What's included

| Component | Function |
|---|---|
| `cloud/streamlit_app.py` | Web interface with face detection, admin sign-in, enrollment, gallery management and privacy notes |
| `cloud/secure_gallery.py` | Fernet-encrypted face embeddings, PostgreSQL connection, CRUD, and expiry handling |
| `app.py` | Desktop application with continuous local webcam detection and recognition |
| `face_core.py` | Cosine similarity, validated face labels, embedding normalization and local gallery |
| `download_models.py` | First-run download of official OpenCV YuNet and SFace ONNX models |
| `db/schema.sql` | Supabase PostgreSQL schema, RLS and restricted public role privileges |
| `.streamlit/config.toml` | Theme and maximum upload size |
| `tests/` | Pure logic, encryption, image safety, configuration and DB readiness tests |
| `DEPLOYMENT_GUIDE.md` | Complete deployment and secure-secret setup |
| `SUBMISSION.md` | Internship submission instructions and implementation remarks |

## Free web deployment

Host on **[Streamlit Community Cloud](https://share.streamlit.io/)** using:

- GitHub repo: `alvy007/Face-Recognition-And-Enrollments-`
- Branch: `main`
- Entry point: `cloud/streamlit_app.py`
- Python: `3.12`
- Dependencies: `cloud/requirements.txt` (automatically discovered next to entry point)

The first cold start downloads the official OpenCV ONNX models (~39 MB total). Internet access is required. **Camera snapshots** are supported in the web app, not a continuous webcam video stream. See [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md).

## Storage modes

**Without Secrets:** demo mode, session-only face enrollment, no database persistence. Each visitor has a separate temporary gallery.

**With all three Secrets:** encrypted PostgreSQL enrollment, admin-only identity matching, explicit consent, per-identity deletion and a 30-day expiry. The UI confirms database access only after an admin login and a real SQL check. Secrets are kept in Streamlit Cloud settings, never in Git.

Required Streamlit Secrets (values are placeholders, **not** real credentials):

```toml
SUPABASE_DB_URL = "postgresql://ROLE.PROJECT-REF:PASSWORD@REGIONAL-POOLER:5432/postgres"
FACE_ENCRYPTION_KEY = "A_FERNET_KEY_GENERATED_LOCALLY"
ADMIN_PASSWORD = "A_LONG_RANDOM_PASSWORD"
```

Use the **Session Pooler** URI shown under Supabase **Connect** (not the direct IPv6 `db...supabase.co` URI when on IPv4-only hosting). Execute [`db/schema.sql`](db/schema.sql) once in the Supabase SQL Editor. At least 16 characters are required for `ADMIN_PASSWORD`. Generate `FACE_ENCRYPTION_KEY` with Python's `cryptography.fernet.Fernet.generate_key()`.

**Never share passwords, full PostgreSQL URLs, encryption keys or enrolled face images in chat, screenshots, issues, or the repository.** A publishable Supabase API key is not used in this direct PostgreSQL integration.

## Run the web app locally (optional)

With Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r cloud/requirements.txt
python -m streamlit run cloud/streamlit_app.py
```

The Streamlit app can run in temporary mode with no secrets. For persistent storage, put the three secrets in Streamlit Cloud app Settings; never commit `.streamlit/secrets.toml`.

## Run the desktop webcam app (optional)

```powershell
python -m pip install -r requirements.txt
python download_models.py yunet.onnx sface.onnx
python app.py --camera 0
```

Press **Q** or **Esc** inside the OpenCV window to stop. To enroll your own or a consenting person's face locally:

```powershell
python app.py --enroll Alvy --image "C:\path\to\your\actual\portrait.jpg"
```

Local features are stored in `data/gallery.npz`, which is excluded from Git. Avoid uploading raw face photos or feature vectors to public repositories.

## Quality assurance

```bash
python -m pytest -q tests
python -m compileall -q app.py face_core.py download_models.py cloud
```

Automated tests run in GitHub Actions on Python 3.11 and 3.12. They do not prove that a remote host can reach Supabase or download the models. Those integrations must be tested after deployment. The app never claims an identity match is certain; cosine similarity is **not** a probability.

## Privacy and limits

- **Consent:** enrollment requires a checked consent confirmation and one-face photo.
- **Least exposure:** persistent face gallery is available only after admin password sign-in; public mode detects bounding boxes without creating recognition embeddings.
- **Encrypted storage:** only 128-dimensional SFace embeddings are stored in Fernet ciphertext, not photos. Cloud hosts still process uploaded images transiently.
- **Retention:** records expire after 30 days and are cleaned up on the next authenticated gallery access, not on an independent scheduled timer.
- **Deletion:** delete individual embeddings or all active enrollments. Provider backups may retain historical copies per their policies.
- **Educational security:** the one-password administration panel, including a 30-minute session limit and per-session attempt counter, is **not** production-grade authentication. A real service needs identity management, centralized rate limiting, consent records, auditing, privacy review and more.

The project does not rely on Supabase REST API or the public publishable key. Reusing a private database password across apps or storing it in GitHub is discouraged.

## License

MIT for project code. OpenCV Zoo models retain their upstream licenses and terms.
