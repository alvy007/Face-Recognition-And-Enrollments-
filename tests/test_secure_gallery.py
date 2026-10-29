"""Offline encryption and validation tests; no DB credentials required."""
import sys
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud.secure_gallery import (
    EMBEDDING_LENGTH, seal_embedding, unseal_embedding, validate_key
)


def test_encrypt_decrypt_unit_embedding():
    key = Fernet.generate_key().decode()
    feature = np.arange(1, EMBEDDING_LENGTH + 1, dtype=np.float32)
    ciphertext = seal_embedding(feature, key)
    assert isinstance(ciphertext, bytes)
    assert not np.array_equal(np.frombuffer(ciphertext[:512], dtype=np.uint8), feature.astype(np.uint8))
    result = unseal_embedding(ciphertext, key)
    np.testing.assert_allclose(result, feature / np.linalg.norm(feature), rtol=1e-6, atol=1e-6)


def test_encryption_randomizes_ciphertext():
    key = Fernet.generate_key().decode()
    x = np.ones(EMBEDDING_LENGTH, dtype=np.float32)
    assert seal_embedding(x, key) != seal_embedding(x, key)


def test_wrong_key_rejected():
    key = Fernet.generate_key().decode()
    other = Fernet.generate_key().decode()
    with pytest.raises(ValueError, match="decrypt"):
        unseal_embedding(seal_embedding(np.ones(128, dtype=np.float32), key), other)


def test_invalid_key_rejected():
    with pytest.raises((ValueError, Exception)):
        validate_key("not-a-valid-fernet-key")


def test_non_sface_dimension_rejected():
    with pytest.raises(ValueError, match="128"):
        seal_embedding(np.ones(8, dtype=np.float32), Fernet.generate_key().decode())


def test_corrupt_ciphertext_rejected():
    key = Fernet.generate_key().decode()
    with pytest.raises(ValueError, match="decrypt"):
        unseal_embedding(b"definitely-not-a-valid-token", key)


def test_empty_feature_rejected():
    with pytest.raises(ValueError):
        seal_embedding(np.array([], dtype=np.float32), Fernet.generate_key().decode())


def test_nan_feature_rejected():
    with pytest.raises(ValueError):
        seal_embedding(np.full(128, np.nan, dtype=np.float32), Fernet.generate_key().decode())
