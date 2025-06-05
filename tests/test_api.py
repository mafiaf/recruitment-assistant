import types
import pytest
from fastapi.responses import HTMLResponse
import main

class DummyRequest:
    def __init__(self, path="/"):
        self.url = types.SimpleNamespace(path=path)
        self.cookies = {}

@pytest.fixture
def authed(monkeypatch):
    monkeypatch.setattr(main, "require_login", lambda: {"username": "u", "role": "user"})
    monkeypatch.setattr(main, "get_current_user", lambda *a, **k: {"username": "u", "role": "user"})
    fake_openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))]
                )
            )
        )
    )
    monkeypatch.setattr(main, "openai", fake_openai)
    monkeypatch.setattr(main, "add_resume_to_pinecone", lambda *a, **k: None)
    monkeypatch.setattr(main, "resumes_collection", types.SimpleNamespace(insert_one=lambda doc: None))
    monkeypatch.setattr(main, "render", lambda *a, **kw: HTMLResponse("OK"))
    return None

@pytest.mark.asyncio
async def test_upload_resume_json(authed):
    resp = await main.upload_resume(DummyRequest("/upload_resume"), resume=main.ResumeUpload(name="John", text="bio"))
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_json(authed):
    resp = await main.chat(DummyRequest("/chat"), chat_data=main.ChatRequest(text="hello", candidate_ids=[]))
    assert resp.status_code == 200
    assert resp.body == b'{"reply":"hi"}' or resp.body == b'{"reply": "hi"}'


def test_delete_project(authed):
    resp = main.delete_project_route(DummyRequest("/delete_project"), ts="2023-01-01T00:00:00")
    assert resp.status_code == 303
