# VisionID Final Development QA Report

## Verified offline

- [x] Python compile check across all source and tests
- [x] 40 of 40 offline unit tests pass in the available development environment
- [ ] Streamlit initial-render smoke test awaiting GitHub Actions execution
- [x] OpenCV bounding-box rendering functions behave as expected with synthetic test frames
- [x] Face embedding normalization, similarity, Unknown threshold behavior and local gallery persistence tested
- [x] Fernet encryption roundtrip, corruption, invalid-key, incorrect-key and feature-size validation tested
- [x] PostgreSQL schema readiness checks tested with simulated cursors (no live DB)
- [x] Browser image decode accepts JPEG/PNG, rejects invalid/oversized input and reduces large images
- [x] Partial/invalid cloud Secrets rejected; temporary mode and full configuration parsed
- [x] No model binaries, face images, saved gallery, passwords, or live secrets included in the submission archive

## Pending live acceptance tests

- [ ] Streamlit server startup and model download on the selected hosting environment
- [ ] Supabase Session Pooler real PostgreSQL connection and schema availability
- [ ] Admin login and persistence across server refresh
- [ ] Camera snapshot and uploaded photo detection with actual faces
- [ ] Recognition against consented identity, plus unknown-face test
- [ ] Enrolled identity removal and guest-access restrictions in a private browser

The available build environment does not include a running Streamlit server or access to the user's Supabase credentials, webcam or deployed cloud host. **Unit tests passing do not certify these external integrations.**

## Code and privacy notes

- Web demo is *snapshot-based*, separate from the continuous desktop webcam implementation.
- Admin authentication has a per-session attempt limit and fixed 30-minute expiry; it is not hardened against distributed brute-force attempts.
- Persistent gallery is encrypted with a user-supplied Fernet key, whose loss prevents decryption.
- Database expiry is cleaned on authenticated access, not on a background task.
- Do not use this project for surveillance, identity-based access decisions or attendance.
