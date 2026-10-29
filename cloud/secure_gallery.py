"""Admin-only, consent-based, encrypted Supabase PostgreSQL gallery.

No face images are retained. This educational app deliberately does not
expose stored identities to unauthenticated visitors.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from face_core import normalize, safe_name

MAX_IDENTITIES = 100
EMBEDDING_LENGTH = 128  # OpenCV SFace embedding dimension


def validate_key(key: str) -> bytes:
    if not isinstance(key, str):
        raise ValueError("FACE_ENCRYPTION_KEY must be a Fernet key string")
    encoded = key.encode("ascii")
    Fernet(encoded)  # validates format
    return encoded


def seal_embedding(feature: np.ndarray, key: str) -> bytes:
    normalized = normalize(feature)
    if normalized.size != EMBEDDING_LENGTH:
        raise ValueError("SFace embedding must contain 128 values")
    return Fernet(validate_key(key)).encrypt(normalized.tobytes())


def unseal_embedding(payload: bytes, key: str) -> np.ndarray:
    try:
        plain = Fernet(validate_key(key)).decrypt(bytes(payload))
    except InvalidToken as error:
        raise ValueError("Stored embedding cannot be decrypted with the configured key") from error
    if len(plain) != EMBEDDING_LENGTH * 4:
        raise ValueError("Invalid stored embedding length")
    return normalize(np.frombuffer(plain, dtype=np.float32).copy())


def _connect(dsn: str):
    """Import psycopg lazily so pure encryption tests don't need a server."""
    import psycopg
    if not isinstance(dsn, str) or not dsn.startswith(("postgresql://", "postgres://")):
        raise ValueError("Use the PostgreSQL connection string from Supabase Session Pooler")
    return psycopg.connect(dsn, connect_timeout=8, sslmode="require", prepare_threshold=None)


def load_gallery(dsn: str, key: str) -> dict[str, np.ndarray]:
    """Purge expired rows on admin access, then return a private gallery."""
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM public.visionid_faces WHERE expires_at <= NOW()")
            cursor.execute(
                "SELECT label, embedding_ciphertext FROM public.visionid_faces "
                "WHERE expires_at > NOW() ORDER BY label LIMIT %s", (MAX_IDENTITIES + 1,)
            )
            rows = cursor.fetchall()
    if len(rows) > MAX_IDENTITIES:
        raise ValueError("Gallery has more enrolled identities than this demo supports")
    return {safe_name(label): unseal_embedding(payload, key) for label, payload in rows}


def enroll_face(dsn: str, key: str, label: str, feature: np.ndarray) -> None:
    """The caller must have already verified explicit consent and admin auth."""
    label = safe_name(label)
    encrypted = seal_embedding(feature, key)
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO public.visionid_faces
                  (label, embedding_ciphertext, consent_confirmed, consented_at, expires_at)
                  VALUES (%s, %s, TRUE, NOW(), NOW() + INTERVAL '30 days')
                  ON CONFLICT (label) DO UPDATE SET
                    embedding_ciphertext = EXCLUDED.embedding_ciphertext,
                    consent_confirmed = TRUE,
                    consented_at = NOW(),
                    updated_at = NOW(),
                    expires_at = NOW() + INTERVAL '30 days'""",
                (label, encrypted)
            )


def remove_face(dsn: str, label: str) -> bool:
    label = safe_name(label)
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM public.visionid_faces WHERE label = %s", (label,))
            return cursor.rowcount > 0


def clear_gallery(dsn: str) -> int:
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM public.visionid_faces")
            return cursor.rowcount
