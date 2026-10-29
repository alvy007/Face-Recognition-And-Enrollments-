import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from face_core import safe_name, normalize, similarity, best_match, load_gallery, save_gallery

def test_safe_name(): assert safe_name('Alice Smith') == 'Alice Smith'
@pytest.mark.parametrize('value', ['', '../escape', 'a/b', 'a\\b', '!'*50])
def test_bad_names(value):
    with pytest.raises(ValueError): safe_name(value)
def test_normalize_unit_length(): assert np.linalg.norm(normalize([3., 4.])) == pytest.approx(1)
def test_normalize_zero_rejected():
    with pytest.raises(ValueError): normalize([0, 0])
def test_cosine_same(): assert similarity([5,0], [1,0]) == pytest.approx(1)
def test_cosine_opposite(): assert similarity([5,0], [-1,0]) == pytest.approx(-1)
def test_cosine_orthogonal(): assert similarity([1,0], [0,1]) == pytest.approx(0)
def test_dimension_mismatch():
    with pytest.raises(ValueError): similarity([1,0], [1,0,0])
def test_accept_match(): assert best_match([1,0], {'Alice':np.array([1,0]),'Bob':np.array([0,1])}).name == 'Alice'
def test_reject_match(): assert best_match([0,1], {'Alice':np.array([1,0])}).name == 'Unknown'
def test_empty_gallery(): assert best_match([1,0], {}).accepted is False
def test_invalid_threshold():
    with pytest.raises(ValueError): best_match([1,0], {}, threshold=9)
def test_roundtrip(tmp_path):
    dest = tmp_path/'nested'/'gallery.npz'
    save_gallery(dest, {'Alice':np.array([3,4])})
    assert similarity(load_gallery(dest)['Alice'], [3,4]) == pytest.approx(1)
    assert list(load_gallery(dest)) == ['Alice']
def test_missing_gallery(tmp_path): assert load_gallery(tmp_path/'missing.npz') == {}
