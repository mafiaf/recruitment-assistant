import types
import pytest
from fastapi.responses import HTMLResponse
import main

class DummyRequest:
    def __init__(self, path="/match"):
        self.url = types.SimpleNamespace(path=path)
        self.cookies = {}

@pytest.mark.asyncio
async def test_match_requires_auth():
    with pytest.raises(main.HTTPException) as exc:
        await main.match_form(DummyRequest())
    assert exc.value.status_code == 302

@pytest.mark.asyncio
async def test_match_returns_form(monkeypatch):
    async def _require_login():
        return {"username": "u"}
    async def _get_current_user(*a, **k):
        return {"username": "u"}
    monkeypatch.setattr(main, "require_login", _require_login)
    monkeypatch.setattr(main, "get_current_user", _get_current_user)
    async def fake_render(*a, **kw):
        return HTMLResponse("OK")
    monkeypatch.setattr(main, "render", fake_render)
    resp = await main.match_form(DummyRequest())
    assert resp.status_code == 200
