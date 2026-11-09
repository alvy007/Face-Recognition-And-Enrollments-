"""DB readiness detection with fake PostgreSQL cursor (no external DB needed)."""
from __future__ import annotations

from contextlib import contextmanager
import pytest
from cloud import secure_gallery


class FakeCursor:
    def __init__(self, found=True):
        self.found = found
        self.queries = []
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, statement): self.queries.append(statement)
    def fetchone(self):
        if len(self.queries) == 1:
            return ("visionid_faces" if self.found else None,)
        return (0,)


class FakeConnection:
    def __init__(self, found=True): self.cursor_obj = FakeCursor(found)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def cursor(self): return self.cursor_obj


def test_connection_ready_checks_schema_and_read_permission(monkeypatch):
    conn = FakeConnection()
    monkeypatch.setattr(secure_gallery, "_connect", lambda dsn: conn)
    assert secure_gallery.check_connection("postgresql://example") is True
    assert any("to_regclass" in query for query in conn.cursor_obj.queries)
    assert any("COUNT(*)" in query for query in conn.cursor_obj.queries)


def test_connection_rejects_missing_table(monkeypatch):
    monkeypatch.setattr(secure_gallery, "_connect", lambda dsn: FakeConnection(found=False))
    with pytest.raises(RuntimeError, match="has not been created"):
        secure_gallery.check_connection("postgresql://example")
