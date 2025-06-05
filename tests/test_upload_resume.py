from fastapi.testclient import TestClient
import main
import pytest

# disable startup DB seed
if main.seed_owner in main.app.router.on_startup:
    main.app.router.on_startup.remove(main.seed_owner)

@pytest.fixture
def client(monkeypatch):
    stored = []

    class FakeColl:
        async def insert_one(self, doc):
            stored.append(doc)

    monkeypatch.setattr(main, "add_resume_to_pinecone", lambda *a, **k: None)
    monkeypatch.setattr(main, "resumes_collection", FakeColl())
    async def _login():
        return {"username": "user"}

    main.app.dependency_overrides[main.require_login] = _login

    with TestClient(main.app) as c:
        yield c, stored
    main.app.dependency_overrides.pop(main.require_login, None)


def test_upload_resume_file(client):
    c, stored = client
    data = {"text": "Sample text"}
    resp = c.post("/upload_resume", data=data)
    assert resp.status_code == 200
    assert len(stored) == 1
    assert stored[0]["text"] == "Sample text"


def test_upload_resume_file_upload(client, monkeypatch):
    c, stored = client
    monkeypatch.setattr(main, "extract_text", lambda b, fn: "PDF text")
    files = {"file": ("resume.pdf", b"%PDF-1.4 ...")}
    data = {"name": "Alice"}
    resp = c.post("/upload_resume", data=data, files=files)
    assert resp.status_code == 200
    assert len(stored) == 1
    assert stored[0]["text"] == "PDF text"


def test_upload_resume_invalid_extension(client, monkeypatch):
    c, stored = client

    def fail(*a, **k):
        raise AssertionError("extract_text should not be called")

    monkeypatch.setattr(main, "extract_text", fail)
    files = {"file": ("resume.txt", b"ignored")}
    resp = c.post("/upload_resume", files=files)
    assert resp.status_code == 200
    assert len(stored) == 0


def test_upload_resume_too_large(client, monkeypatch):
    c, stored = client
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 10)

    def fail(*a, **k):
        raise AssertionError("extract_text should not be called")

    monkeypatch.setattr(main, "extract_text", fail)
    files = {"file": ("resume.pdf", b"x" * 20)}
    resp = c.post("/upload_resume", files=files)
    assert resp.status_code == 200
    assert len(stored) == 0
