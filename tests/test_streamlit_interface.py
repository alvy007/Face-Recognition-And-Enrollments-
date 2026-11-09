"""Basic Streamlit render smoke test without camera or database access.

Run in CI with cloud/requirements.txt installed. This intentionally does not
submit biometric data or invoke downloads of any ONNX face model.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_public_streamlit_page_loads_without_secrets():
    entrypoint = Path(__file__).resolve().parents[1] / "cloud" / "streamlit_app.py"
    app = AppTest.from_file(str(entrypoint), default_timeout=30).run()
    assert not app.exception, "Streamlit raised during initial public page rendering"
    assert len(app.tabs) == 4
