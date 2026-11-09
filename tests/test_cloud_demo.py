"""Offline tests of the Streamlit web pipeline, no camera/credentials required."""
from __future__ import annotations

import importlib.util
import io
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from cryptography.fernet import Fernet
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def web_app(monkeypatch):
    # Do not require a full Streamlit server to test pure image/config functions.
    fake = types.SimpleNamespace(cache_resource=lambda **kwargs: lambda func: func,
                                 secrets={})
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    spec = importlib.util.spec_from_file_location("visionid_offline_web_tests", ROOT / "cloud" / "streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake


def make_image(fmt="PNG", width=160, height=100) -> bytes:
    image = Image.new("RGB", (width, height), (24, 180, 125))
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_decode_valid_image_formats(web_app, format):
    app, _ = web_app
    frame = app.decode_picture(make_image(format))
    assert frame.shape == (100, 160, 3)
    assert frame.dtype == np.uint8


def test_picture_limits_and_invalid_inputs(web_app):
    app, _ = web_app
    for bad in (b"", b"not an image", b"a" * (app.MAX_UPLOAD_BYTES + 1)):
        with pytest.raises(ValueError):
            app.decode_picture(bad)


def test_large_photo_is_scaled_before_opencv(web_app):
    app, _ = web_app
    frame = app.decode_picture(make_image(width=3000, height=2000))
    assert max(frame.shape[:2]) <= app.MAX_IMAGE_SIDE


def test_public_detections_never_modify_original(web_app):
    app, _ = web_app
    frame = np.zeros((130, 150, 3), dtype=np.uint8)
    boxes = [np.array([20, 30, 55, 65], dtype=np.float32)]
    rendered = app.mark_detection_only(frame, boxes)
    assert rendered.shape == frame.shape
    assert rendered.sum() > 0
    assert frame.sum() == 0


def test_empty_secrets_enable_only_session_mode(web_app):
    app, fake = web_app
    fake.secrets = {}
    mode, values = app.storage_settings()
    assert mode == "temporary"
    assert not any(values.values())


def test_partial_secrets_fail_closed(web_app):
    app, fake = web_app
    fake.secrets = {"ADMIN_PASSWORD": "super-strong-unique-passphrase"}
    assert app.storage_settings()[0] == "incomplete"


def test_invalid_fernet_key_is_rejected(web_app):
    app, fake = web_app
    fake.secrets = {"SUPABASE_DB_URL": "postgresql://user:pw@some-server/db",
                    "FACE_ENCRYPTION_KEY": "not-a-fernet-key",
                    "ADMIN_PASSWORD": "super-strong-unique-passphrase"}
    assert app.storage_settings()[0] == "invalid_key"


def test_invalid_admin_password_and_uri_rejected(web_app):
    app, fake = web_app
    fake.secrets = {"SUPABASE_DB_URL": "postgresql://user:pw@some-server/db",
                    "FACE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                    "ADMIN_PASSWORD": "weak"}
    assert app.storage_settings()[0] == "weak_password"
    fake.secrets["ADMIN_PASSWORD"] = "super-strong-unique-passphrase"
    fake.secrets["SUPABASE_DB_URL"] = "http://invalid.example.com"
    assert app.storage_settings()[0] == "invalid_database"


def test_valid_persistent_configuration(web_app):
    app, fake = web_app
    fake.secrets = {"SUPABASE_DB_URL": "postgresql://user:pw@pooler.example.com/postgres",
                    "FACE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
                    "ADMIN_PASSWORD": "super-strong-unique-passphrase"}
    assert app.storage_settings()[0] == "persistent"
