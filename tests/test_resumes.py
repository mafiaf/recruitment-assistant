import types
import pytest
from fastapi.responses import HTMLResponse
import main

class DummyRequest:
    def __init__(self, path="/resumes"):
        self.url = types.SimpleNamespace(path=path)
        self.cookies = {}

@pytest.mark.asyncio
async def test_list_resumes_empty(monkeypatch):
    async def fake_page(page, per_page, filters):
        return []

    async def fake_count(filters):
        return 0

    async def fake_require_login():
        return {"username": "user"}

    async def fake_render(*a, **k):
        return HTMLResponse("OK")

    monkeypatch.setattr(main, "resumes_page", fake_page)
    monkeypatch.setattr(main, "resumes_count", fake_count)
    monkeypatch.setattr(main, "require_login", fake_require_login)
    monkeypatch.setattr(main, "render", fake_render)

    resp = await main.list_resumes(DummyRequest(), page=1)
    assert resp.status_code == 200
