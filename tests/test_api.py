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
    async def _require_login():
        return {"username": "u", "role": "user"}

    async def _get_current_user(*a, **k):
        return {"username": "u", "role": "user"}

    monkeypatch.setattr(main, "require_login", _require_login)
    monkeypatch.setattr(main, "get_current_user", _get_current_user)
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
    async def _insert(doc):
        return None

    monkeypatch.setattr(main, "resumes_collection", types.SimpleNamespace(insert_one=_insert))
    async def _render(*a, **kw):
        return HTMLResponse("OK")

    monkeypatch.setattr(main, "render", _render)
    return None

@pytest.mark.asyncio
async def test_upload_resume_json(authed):
    resp = await main.upload_resume(DummyRequest("/upload_resume"), resume=main.ResumeUpload(name="John", text="bio"))
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_json(authed):
    resp = await main.chat(DummyRequest("/chat"), chat_data=main.ChatRequest(text="hello", candidate_ids=[]))
    assert resp.status_code == 200
    assert b'"reply":' in resp.body
    assert b'"time":' in resp.body


@pytest.mark.asyncio
async def test_delete_project(authed):
    resp = await main.delete_project_route(DummyRequest("/delete_project"), ts="2023-01-01T00:00:00")
    assert resp.status_code == 303
